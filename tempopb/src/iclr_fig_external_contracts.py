"""Trusted input capture and semantic contracts for ``fig_external``.

This module owns authentication and parsing only. It captures each trusted
Task 2 file through one no-follow descriptor, hashes the captured bytes, parses
those exact bytes, and validates the frozen table identities. Plotting belongs
to :mod:`iclr_fig_external`; destination mutation belongs to
:mod:`iclr_fig_external_io`.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import analyze_external_support_set_results as analysis_bundle
from analyze_external_support_set_results import CITY_INTERVAL_COLUMNS, EFFECT_COLUMNS


FIGURE_OUTPUTS = (
    "figures/fig_external.pdf",
    "figures/fig_external.png",
)
TASK2_OUTPUTS = tuple(
    name for name in analysis_bundle.OUTPUT_FILENAMES if name != "manifest.json"
)
TASK3_OUTPUTS = TASK2_OUTPUTS + FIGURE_OUTPUTS
TRUSTED_TASK2_OUTPUT_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "effects.csv": "1d384ae167d88fae6a3b90db8ab358caaae7ef583ebde05012885aae62a424cf",
        "city_intervals.csv": "4dbbaa952ec65e2d2a49f23c65731fea8d70768a24552ac9acfe3dc42a3114ff",
        "diagnostics.json": "c2c34f16d3bf05a59f7c3a15a9fc049d3c0a60f40803c9f56d8e9d9fe0a58dcb",
    }
)
EXPECTED_MANIFEST_KEYS = {
    "classification",
    "entry_point",
    "input_sha256",
    "output_sha256",
    "package_versions",
    "python_version",
    "schema_version",
}
CITY_ORDER = ("Poland/Katowice", "Poland/Krakow")
SEED_ORDER = (1, 2, 42)
EXPECTED_SERIES = frozenset(
    {
        "Poland/Katowice/Bogucice",
        "Poland/Katowice/Dąb",
        "Poland/Katowice/Dąbrówka Mała",
        "Poland/Katowice/Giszowiec",
        "Poland/Katowice/Kostuchna",
        "Poland/Katowice/Koszutka",
        "Poland/Katowice/Murcki",
        "Poland/Katowice/Osiedle Tysiąclecia",
        "Poland/Katowice/Osiedle Witosa",
        "Poland/Katowice/Podlesie",
        "Poland/Katowice/Zarzecze",
        "Poland/Katowice/Zawodzie",
        "Poland/Katowice/Załęże",
        "Poland/Katowice/Śródmieście",
        "Poland/Krakow/Bieńczyce",
        "Poland/Krakow/Bieżanów-Prokocim",
        "Poland/Krakow/Bronowice",
        "Poland/Krakow/Czyżyny",
        "Poland/Krakow/Dębniki",
        "Poland/Krakow/Grzegórzki",
        "Poland/Krakow/Krowodrza",
        "Poland/Krakow/Mistrzejowice",
        "Poland/Krakow/Nowa Huta",
        "Poland/Krakow/Podgórze",
        "Poland/Krakow/Podgórze Duchackie",
        "Poland/Krakow/Prądnik Biały",
        "Poland/Krakow/Prądnik Czerwony",
        "Poland/Krakow/Stare Miasto",
        "Poland/Krakow/Swoszowice",
        "Poland/Krakow/Wzgórza Krzesławickie",
        "Poland/Krakow/Zwierzyniec",
        "Poland/Krakow/Łagiewniki-Borek Fałęcki",
    }
)


@dataclass(frozen=True)
class CapturedFile:
    """One regular file captured from one stable no-follow descriptor."""

    path: Path
    content: bytes
    sha256: str


@dataclass(frozen=True)
class CapturedPlotInputs:
    """Authenticated inputs and the manifest state that registered them."""

    analysis_dir: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    state: str
    trusted_sha256: Mapping[str, str]
    effects: pd.DataFrame
    intervals: pd.DataFrame
    diagnostics: Mapping[str, Any]


def _fingerprint(identity: os.stat_result) -> tuple[int, ...]:
    return (
        identity.st_dev,
        identity.st_ino,
        identity.st_mode,
        identity.st_size,
        identity.st_mtime_ns,
        identity.st_ctime_ns,
    )


def _require_existing_directory(path: Path) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise ValueError("analysis_dir and every parent must be a directory") from error
        if stat.S_ISLNK(mode):
            raise ValueError(f"analysis_dir must not contain a symlinked parent: {current}")
        if not stat.S_ISDIR(mode):
            raise ValueError(f"analysis_dir contains a non-directory parent: {current}")
    return absolute.resolve(strict=True)


def capture_regular_file(path: Path) -> CapturedFile:
    """Read a stable regular file exactly once without following its final link."""
    try:
        path_before = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"required registered file is missing: {path.name}") from error
    if stat.S_ISLNK(path_before.st_mode) or not stat.S_ISREG(path_before.st_mode):
        raise ValueError(f"required registered file must be regular: {path.name}")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.EMLINK}:
            raise ValueError(f"required registered file must not be a symlink: {path.name}") from error
        raise
    try:
        descriptor_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_before.st_mode)
            or _fingerprint(descriptor_before) != _fingerprint(path_before)
        ):
            raise ValueError(f"registered file changed before capture: {path.name}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        descriptor_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    try:
        path_after = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"registered file replaced or changed during capture: {path.name}") from error
    if (
        _fingerprint(descriptor_before) != _fingerprint(descriptor_after)
        or _fingerprint(descriptor_after) != _fingerprint(path_after)
    ):
        raise ValueError(f"registered file replaced or changed during capture: {path.name}")
    content = b"".join(chunks)
    if len(content) != descriptor_after.st_size:
        raise ValueError(f"registered file changed size during capture: {path.name}")
    return CapturedFile(path, content, hashlib.sha256(content).hexdigest())


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_manifest(captured: CapturedFile) -> Mapping[str, Any]:
    manifest = json.loads(captured.content.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("analysis manifest must be a JSON object")
    return manifest


def _validate_manifest(manifest: Mapping[str, Any]) -> str:
    if set(manifest) != EXPECTED_MANIFEST_KEYS:
        raise ValueError("analysis manifest does not have the complete Task 2 shape")
    if manifest["classification"] != "falsified":
        raise ValueError("analysis manifest must preserve the falsified classification")
    if manifest["entry_point"] != "python src/analyze_external_support_set_results.py":
        raise ValueError("analysis manifest entry point is not the frozen Task 2 entry point")
    if manifest["schema_version"] != 1:
        raise ValueError("analysis manifest schema version must be 1")
    if manifest["input_sha256"] != {
        relative: analysis_bundle.EXPECTED_SHA256[relative]
        for relative in sorted(analysis_bundle.EXPECTED_SHA256)
    }:
        raise ValueError("analysis manifest input SHA-256 anchors do not match Task 2")
    package_versions = manifest["package_versions"]
    if (
        not isinstance(package_versions, dict)
        or set(package_versions) != {"pandas"}
        or not isinstance(package_versions["pandas"], str)
        or not package_versions["pandas"]
    ):
        raise ValueError("analysis manifest package_versions is incomplete")
    python_version = manifest["python_version"]
    if (
        not isinstance(python_version, str)
        or len(python_version.split(".")) != 3
        or not all(part.isdigit() for part in python_version.split("."))
    ):
        raise ValueError("analysis manifest python_version is invalid")
    output_hashes = manifest["output_sha256"]
    if not isinstance(output_hashes, dict):
        raise ValueError("analysis manifest output SHA-256 map is missing")
    if set(output_hashes) == set(TASK2_OUTPUTS):
        state = "task2"
    elif set(output_hashes) == set(TASK3_OUTPUTS):
        state = "task3"
    else:
        raise ValueError("analysis manifest output SHA-256 map is a partial state")
    if not all(_is_sha256(digest) for digest in output_hashes.values()):
        raise ValueError("analysis manifest contains an invalid output SHA-256")
    return state


def _require_finite_columns(
    table: pd.DataFrame, columns: Sequence[str], table_name: str
) -> None:
    for column in columns:
        numeric = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{table_name} column {column} must contain finite values")


def _decimal_direction(value: Any) -> str:
    decimal = Decimal(str(value))
    if not decimal.is_finite():
        raise ValueError("delta_csd must be finite")
    return "win" if decimal < 0 else "loss" if decimal > 0 else "tie"


def _validate_effects(effects: pd.DataFrame) -> None:
    if list(effects.columns) != EFFECT_COLUMNS or effects.shape != (192, 10):
        raise ValueError("effects.csv must preserve the final 192-row Task 2 schema")
    if effects.isna().any().any():
        raise ValueError("effects.csv must not contain missing values")
    _require_finite_columns(
        effects, ("seed", "delta_csd", "delta_exclusion", "welfare_ratio"), "effects.csv"
    )
    if not pd.api.types.is_integer_dtype(effects["seed"]):
        raise ValueError("effects.csv seed identities must be exact integers")
    if effects["actuated"].dtype != bool:
        raise ValueError("effects.csv actuated values must be Boolean")
    expected = {
        (arm, seed, series)
        for arm in ("endowment", "priority")
        for seed in SEED_ORDER
        for series in EXPECTED_SERIES
    }
    observed = set(effects[["arm", "seed", "series"]].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError("effects.csv arm/seed/series identities do not match Task 2")
    policies = effects.apply(lambda row: f"{row['arm']}/seed-{int(row['seed'])}", axis=1)
    if not effects["policy"].equals(policies):
        raise ValueError("effects.csv policy must equal arm/seed-{seed}")
    cities = effects["series"].map(lambda value: str(value).rsplit("/", 1)[0])
    if not effects["city"].equals(cities):
        raise ValueError("effects.csv city must be the parent prefix of series")
    if not effects["direction"].equals(effects["delta_csd"].map(_decimal_direction)):
        raise ValueError("effects.csv direction does not match the Decimal sign of delta_csd")


def _validate_intervals(intervals: pd.DataFrame) -> None:
    if list(intervals.columns) != CITY_INTERVAL_COLUMNS or intervals.shape != (6, 16):
        raise ValueError("city_intervals.csv must preserve the final six-row Task 2 schema")
    if not pd.api.types.is_integer_dtype(intervals["seed"]):
        raise ValueError("city_intervals.csv seed identities must be exact integers")
    identities = set(intervals[["seed", "city"]].itertuples(index=False, name=None))
    expected = {(seed, city) for seed in SEED_ORDER for city in CITY_ORDER}
    if identities != expected or intervals.duplicated(["seed", "city"]).any():
        raise ValueError("city_intervals.csv identities do not match every city/seed pair")
    non_holm = [column for column in intervals.columns if column != "holm_p_value"]
    if intervals[non_holm].isna().any().any():
        raise ValueError("city_intervals.csv contains unexpected missing values")
    _require_finite_columns(
        intervals,
        (
            "seed", "baseline_csd", "learned_csd", "difference", "ci_low", "ci_high",
            "p_value", "exclusion_difference", "welfare_ratio", "wins", "ties",
            "losses", "n_series",
        ),
        "city_intervals.csv",
    )
    primary_holm = intervals.loc[intervals["seed"] == 42, "holm_p_value"]
    other_holm = intervals.loc[intervals["seed"] != 42, "holm_p_value"]
    if (
        primary_holm.isna().any()
        or not np.isfinite(primary_holm.to_numpy(dtype=float)).all()
        or other_holm.notna().any()
    ):
        raise ValueError("city_intervals.csv Holm identities do not match the primary seed")
    if set(intervals["p_value_kind"]) != {"exact_sign_flip"}:
        raise ValueError("city_intervals.csv must retain exact sign-flip p-values")
    if not (
        (intervals["ci_low"] <= intervals["difference"])
        & (intervals["difference"] <= intervals["ci_high"])
    ).all():
        raise ValueError("city interval estimate must lie inside [ci_low, ci_high]")
    expected_counts = intervals["city"].map({"Poland/Katowice": 14, "Poland/Krakow": 18})
    if not intervals["n_series"].equals(expected_counts):
        raise ValueError("city_intervals.csv n_series does not match city identity")
    if not (intervals[["wins", "ties", "losses"]].sum(axis=1) == intervals["n_series"]).all():
        raise ValueError("city interval win/tie/loss counts must sum to n_series")


def _validate_diagnostics(diagnostics: Mapping[str, Any]) -> None:
    if set(diagnostics) != {
        "across_seeds", "actuation", "city_macro_differences", "decision",
        "descriptive_only", "minimum_demographic_coverage", "primary",
        "top3_improvement_concentration",
    }:
        raise ValueError("diagnostics.json does not have the complete Task 2 shape")
    if diagnostics.get("decision") != "falsified" or diagnostics.get("descriptive_only") is not True:
        raise ValueError("diagnostics must preserve the frozen descriptive falsified decision")
    if diagnostics.get("primary") != {"seed": 42, "wins": 8, "ties": 20, "losses": 4}:
        raise ValueError("diagnostics primary seed/count invariants changed")
    if diagnostics.get("across_seeds") != {
        "always_win": 4, "always_tie": 13, "switches_sign": 5,
        "mixed_zero": 10, "total_series": 32,
    }:
        raise ValueError("diagnostics across-seed count invariants changed")
    if diagnostics.get("actuation") != {"endowment": 12, "priority": 3}:
        raise ValueError("diagnostics actuation counts changed")
    city_macro = diagnostics.get("city_macro_differences")
    if not isinstance(city_macro, dict) or set(city_macro) != {"1", "2", "42"}:
        raise ValueError("diagnostics city-macro seed identities changed")
    finite = (
        *city_macro.values(), diagnostics.get("minimum_demographic_coverage"),
        diagnostics.get("top3_improvement_concentration"),
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in finite
    ):
        raise ValueError("diagnostics numeric summaries must be finite")


def capture_plot_inputs(
    analysis_dir: Path,
    *,
    trusted_hashes: Mapping[str, str] = TRUSTED_TASK2_OUTPUT_SHA256,
) -> CapturedPlotInputs:
    """Capture, authenticate, parse, and validate all dashboard inputs."""
    resolved = _require_existing_directory(Path(analysis_dir))
    manifest_file = capture_regular_file(resolved / "manifest.json")
    manifest = _parse_manifest(manifest_file)
    state = _validate_manifest(manifest)
    output_hashes = manifest["output_sha256"]

    captured_trusted: dict[str, CapturedFile] = {}
    for relative, pinned in trusted_hashes.items():
        captured = capture_regular_file(resolved / relative)
        if captured.sha256 != pinned:
            raise ValueError(f"trusted Task 2 SHA-256 mismatch for {relative}")
        if output_hashes.get(relative) != captured.sha256:
            raise ValueError(f"analysis manifest SHA-256 mismatch for {relative}")
        captured_trusted[relative] = captured
    for relative in output_hashes:
        if relative in captured_trusted:
            continue
        captured = capture_regular_file(resolved / relative)
        if captured.sha256 != output_hashes[relative]:
            raise ValueError(f"analysis manifest SHA-256 mismatch for {relative}")

    effects = pd.read_csv(io.BytesIO(captured_trusted["effects.csv"].content))
    intervals = pd.read_csv(io.BytesIO(captured_trusted["city_intervals.csv"].content))
    diagnostics = json.loads(captured_trusted["diagnostics.json"].content.decode("utf-8"))
    if not isinstance(diagnostics, dict):
        raise ValueError("diagnostics.json must be a JSON object")
    _validate_effects(effects)
    _validate_intervals(intervals)
    _validate_diagnostics(diagnostics)
    return CapturedPlotInputs(
        resolved,
        manifest,
        manifest_file.sha256,
        state,
        MappingProxyType(
            {relative: captured.sha256 for relative, captured in captured_trusted.items()}
        ),
        effects,
        intervals,
        diagnostics,
    )


def recheck_captured_inputs(captured: CapturedPlotInputs) -> None:
    """Immediately recheck paths and hashes before transaction commit."""
    resolved = _require_existing_directory(captured.analysis_dir)
    if resolved != captured.analysis_dir:
        raise ValueError("analysis_dir changed after capture")
    manifest = capture_regular_file(captured.analysis_dir / "manifest.json")
    if manifest.sha256 != captured.manifest_sha256:
        raise ValueError("analysis manifest changed after trusted input capture")
    for relative, expected in captured.trusted_sha256.items():
        current = capture_regular_file(captured.analysis_dir / relative)
        if current.sha256 != expected:
            raise ValueError(f"trusted Task 2 SHA-256 mismatch for {relative}")
    for relative, expected in captured.manifest["output_sha256"].items():
        if relative in captured.trusted_sha256:
            continue
        current = capture_regular_file(captured.analysis_dir / relative)
        if current.sha256 != expected:
            raise ValueError(f"registered output changed after capture: {relative}")
