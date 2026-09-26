"""Authenticated three-path parity checks for nonuniform Equal Shares.

This module compares the repository's independent reference, trace-producing,
and vectorized production kernels.  It never substitutes one kernel for
another and retains only canonical winner tuples plus bounded diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import resource
import struct
import sys
import time
from types import MappingProxyType
from typing import Iterable, Literal, Mapping

import numpy as np

from cohorts import group_outcome
from iclr_corpus import SeriesRef, Split
from iclr_env import RolloutState
from iclr_multicity_evaluate import (
    required_fit_inventory,
    verify_lock_fit_inventory,
)
from iclr_multicity_protocol import (
    FIT_SOURCE_FILES,
    FIRST_TEST_YEAR,
    PABULIB_COMMIT,
    _sha256,
    load_canonical_index_from_manifest,
    load_frozen_split,
    verify_multicity_protocol_lock,
)
from iclr_multicity_train import build_multicity_arm
from iclr_priority_mes import fast_mes_with_endowments
from iclr_policy import linear_policy
from iclr_residual_actuation import trace_equal_shares
from parse_pb import PBInstance, Project, Vote
from rules import mes_with_endowments


ROOT = Path(__file__).resolve().parent.parent
EXPECTED_MANIFEST_SHA256 = (
    "fa1fe4a407a6e0e7c8f5a1321dde38c65df93a32165c380866deb7128bcadf62"
)
EXPECTED_HELDOUT_RECEIPT_SHA256 = (
    "da4d9bedf568f496ae54c5433e99988edd62b20b5217ef27a9cb972d28da34ee"
)
SPLIT_NAMES = (
    "temporal_2022",
    "city_out_Poland_Gdynia",
    "city_out_Poland_Warszawa",
    "city_out_Poland_Łódź",
)
WARSAW_SENTINEL_FILENAMES = (
    "Poland_Warszawa_2020_Bielany.pb",
    "Poland_Warszawa_2020_Targowek.pb",
    "Poland_Warszawa_2026_Bemowo.pb",
    "Poland_Warszawa_2026_Bielany.pb",
    "Poland_Warszawa_2026_Targowek.pb",
)
REPRESENTATIVE_FILENAMES = (
    "Poland_Gdynia_2023_Kamienna_Gora__large.pb",
    "Poland_Lodz_2024_Rokicie.pb",
    "Poland_Warszawa_2024_Mokotow.pb",
)
STAGE_EXPECTED_COUNTS = {
    "warsaw-sentinels": (5, 3, 2),
    "representative-temporal-scored": (3, 3, 0),
    "temporal-scored": (223, 223, 0),
    "native": (397, 223, 174),
}
LONG_STAGES = ("temporal-scored", "native", "full")
FIT_CONTEXT_EXPECTED_COUNTS = {
    "city_out_Poland_Gdynia/endowment/seed-42": (124, 61, 63),
    "city_out_Poland_Warszawa/endowment/seed-42": (129, 54, 75),
    "city_out_Poland_Łódź/endowment/seed-42": (144, 108, 36),
    "temporal_2022/endowment/seed-1": (397, 223, 174),
    "temporal_2022/endowment/seed-2": (397, 223, 174),
    "temporal_2022/endowment/seed-42": (397, 223, 174),
}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_hard_gate_modes(completion_modes: tuple[bool, ...]) -> None:
    if (
        completion_modes != (False, True)
        or any(type(mode) is not bool for mode in completion_modes)
    ):
        raise ValueError(
            "hard-gate completion modes must be exactly (False, True)"
        )


def selected_long_stages(value: str) -> tuple[str, ...]:
    """Resolve the opt-in test stage; reject typos instead of silently skipping."""

    if value == "":
        return ()
    if value == "all":
        return LONG_STAGES
    if value not in LONG_STAGES:
        raise ValueError(f"unknown ICLR_TRACE_PARITY_STAGE: {value!r}")
    return (value,)


@dataclass(frozen=True)
class ThreePathParity:
    """Canonical winner outputs from the three independent kernels.

    ``trace`` is the winner tuple from a canonicalized outcome-equivalent MES
    witness.  It does not claim literal branch-iteration identity with the
    reference kernel's ``list(active)`` traversal.
    """

    instance_sha256: str
    completion: bool
    reference: tuple[str, ...]
    trace: tuple[str, ...]
    production: tuple[str, ...]

    @property
    def exact(self) -> bool:
        return self.reference == self.trace == self.production


@dataclass(frozen=True)
class ParityCase:
    """One immutable accepted-input request for the parity wrapper."""

    instance: PBInstance
    endowments: tuple[float, ...]
    completion: bool
    context: str
    phase: Literal["warmup", "scored", "probe"] = "probe"
    instance_sha256: str | None = None


@dataclass(frozen=True)
class ParityMismatch:
    """Bounded diagnostic retained when the three winner tuples differ."""

    context: str
    instance_path: str
    instance_sha256: str
    completion: bool
    reference: tuple[str, ...]
    witness: tuple[str, ...]
    production: tuple[str, ...]


@dataclass(frozen=True)
class ParityAuditReport:
    """Counter-only parity result; complete trace objects are never retained."""

    stage: str
    cases: int
    scored_cases: int
    warmup_cases: int
    probe_cases: int
    completion_counts: tuple[tuple[bool, int], ...]
    context_counts: tuple[tuple[str, int], ...]
    mismatches: tuple[ParityMismatch, ...]
    elapsed_seconds: float
    peak_rss_bytes: int
    gate_eligible: bool = False
    lock_sha256: str = ""
    heldout_receipt_sha256: str = ""
    manifest_sha256: str = ""
    policy_source_sha256: str = ""
    source_sha256: tuple[tuple[str, str], ...] = ()
    split_sha256: tuple[tuple[str, str], ...] = ()
    fit_sha256: tuple[tuple[str, str], ...] = ()
    fit_inventory_count: int = 0
    all_fit_sha256: tuple[tuple[str, str], ...] = ()
    context_sha256: str = ""

    @property
    def exact(self) -> bool:
        return not self.mismatches

    @property
    def gate_passed(self) -> bool:
        return self.gate_eligible and self.exact


class ParityMismatchError(RuntimeError):
    """Raised immediately at the first three-kernel winner mismatch."""

    def __init__(self, report: ParityAuditReport):
        self.report = report
        mismatch = report.mismatches[0]
        super().__init__(
            f"three-path parity mismatch in {mismatch.context}: "
            f"reference={mismatch.reference}, witness={mismatch.witness}, "
            f"production={mismatch.production}"
        )


@dataclass(frozen=True)
class FileSnapshot:
    """One immutable file read paired with the digest of those exact bytes."""

    path: Path
    content: bytes
    sha256: str


def _read_file_snapshot(path: Path) -> FileSnapshot:
    path = Path(path).resolve()
    content = path.read_bytes()
    return FileSnapshot(
        path=path,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _decode_json_snapshot(snapshot: FileSnapshot) -> Mapping[str, object]:
    try:
        payload = json.loads(snapshot.content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON snapshot: {snapshot.path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON snapshot must contain an object: {snapshot.path}")
    return payload


def _verify_heldout_receipt(
    receipt: Mapping[str, object],
    *,
    lock_sha256: str,
    fit_sha256: Mapping[str, str],
    structural_gates_sha256: str,
) -> None:
    """Authenticate the existing read-only held-out-opening receipt exactly."""

    expected_fields = {
        "schema_version",
        "lock_sha256",
        "fit_sha256",
        "structural_gates_sha256",
        "opened_at_utc",
    }
    if set(receipt) != expected_fields:
        raise RuntimeError(
            "held-out receipt fields differ from the exact schema"
        )
    if type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1:
        raise RuntimeError("held-out receipt schema must be exactly 1")
    if receipt["lock_sha256"] != lock_sha256:
        raise RuntimeError("held-out receipt lock SHA-256 differs")
    receipt_fits = receipt["fit_sha256"]
    if (
        not isinstance(receipt_fits, dict)
        or len(receipt_fits) != 24
        or len(fit_sha256) != 24
    ):
        raise RuntimeError("held-out receipt must bind exactly 24 fit SHA-256 values")
    if receipt_fits != dict(fit_sha256):
        raise RuntimeError("held-out receipt fit SHA-256 inventory differs")
    if receipt["structural_gates_sha256"] != structural_gates_sha256:
        raise RuntimeError("held-out receipt structural-gates SHA-256 differs")
    opened_at = receipt["opened_at_utc"]
    if not isinstance(opened_at, str) or not opened_at:
        raise RuntimeError("held-out receipt opened_at_utc must be a nonempty string")


def _verify_fit_snapshot_payload(
    spec: Mapping[str, object],
    snapshot: FileSnapshot,
    payload: Mapping[str, object],
    lock_payload: Mapping[str, object],
    lock_sha256: str,
) -> None:
    """Apply the frozen fit semantics to an already authenticated snapshot."""

    expected_fixed = {
        "generations": 30,
        "popsize": None,
        "sigma0": 0.4,
        "bound": 10.0,
        "welfare_penalty": 2.0,
        "welfare_floor": 1.0,
    }
    feature_counts = {"priority": 5, "endowment": 6, "outcome": 5}
    if payload.get("training_only") is not True:
        raise RuntimeError(f"fit is not training-only: {snapshot.path}")
    if payload.get("execution_mode") != "serial":
        raise RuntimeError(f"fit has invalid execution_mode: {snapshot.path}")
    config = payload.get("config", {})
    if not isinstance(config, dict):
        raise RuntimeError(f"fit has invalid config: {snapshot.path}")
    for key in ("split", "arm", "seed"):
        if config.get(key) != spec[key]:
            raise RuntimeError(
                f"fit {snapshot.path} has {key}={config.get(key)!r}, "
                f"expected {spec[key]!r}"
            )
    for key, expected in expected_fixed.items():
        if config.get(key) != expected:
            raise RuntimeError(
                f"fit {snapshot.path} has {key}={config.get(key)!r}, "
                f"expected {expected!r}"
            )
    result = payload.get("result", {})
    if not isinstance(result, dict):
        raise RuntimeError(f"fit has invalid result: {snapshot.path}")
    weights = np.asarray(result.get("best_weights", []), dtype=float)
    expected_count = feature_counts[str(spec["arm"])]
    if weights.shape != (expected_count,) or not np.all(np.isfinite(weights)):
        raise RuntimeError(f"fit {snapshot.path} has invalid best_weights")
    arm = build_multicity_arm(str(spec["arm"]))
    expected_arm = {
        "name": arm.name,
        "feature_names": list(arm.feature_names),
        "init_name": arm.init_name,
        "initial": arm.initial.tolist(),
    }
    if payload.get("arm") != expected_arm:
        raise RuntimeError(f"fit {snapshot.path} has invalid arm metadata")
    tracked = lock_payload.get("tracked_files", {})
    if not isinstance(tracked, dict):
        raise RuntimeError("protocol lock has invalid tracked_files")
    expected_sources = {
        name: tracked.get(f"source/{name}", {}).get("sha256")
        for name in FIT_SOURCE_FILES
    }
    if any(value is None for value in expected_sources.values()):
        raise RuntimeError("protocol lock is missing fit source hashes")
    expected_provenance = {
        "protocol_lock_sha256": lock_sha256,
        "pabulib_commit": PABULIB_COMMIT,
        "corpus_manifest_sha256": tracked.get(
            "artifact/corpus_manifest", {}
        ).get("sha256"),
        "split_sha256": tracked.get(f"split/{spec['split']}", {}).get("sha256"),
        "source_sha256": expected_sources,
    }
    if payload.get("provenance") != expected_provenance:
        raise RuntimeError(f"fit {snapshot.path} provenance differs from protocol lock")


def _optional_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_pb_snapshot(snapshot: FileSnapshot) -> PBInstance:
    """Parse the exact bytes already authenticated against the manifest."""

    text = snapshot.content.decode("utf-8", errors="replace")
    instance = PBInstance(path=str(snapshot.path))
    section: str | None = None
    header: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if not line.strip():
            continue
        upper = line.strip().upper()
        if upper in ("META", "PROJECTS", "VOTES"):
            section = upper
            header = []
            continue
        fields = line.split(";")
        if section == "META":
            if len(fields) >= 2 and fields[0] != "key":
                instance.meta[fields[0].strip()] = fields[1].strip()
        elif section == "PROJECTS":
            if not header:
                header = [field.strip() for field in fields]
                continue
            row = dict(zip(header, fields))
            project_id = row.get("project_id", "").strip()
            if not project_id:
                continue
            try:
                cost = float(row.get("cost", "0").replace(",", "."))
            except ValueError:
                cost = 0.0
            instance.projects[project_id] = Project(
                pid=project_id,
                cost=cost,
                selected=_optional_int(row.get("selected", "")),
                name=row.get("name", ""),
                category=row.get("category", ""),
                target=row.get("target", ""),
                neighborhood=row.get("neighborhood", ""),
            )
        elif section == "VOTES":
            if not header:
                header = [field.strip() for field in fields]
                continue
            row = dict(zip(header, fields))
            voter_id = row.get("voter_id", "").strip()
            projects = tuple(
                project for project in row.get("vote", "").split(",") if project
            )
            instance.votes.append(
                Vote(
                    vid=voter_id,
                    projects=projects,
                    age=_optional_int(row.get("age", "")),
                    sex=row.get("sex", "").strip().upper(),
                    neighborhood=row.get("neighborhood", "").strip(),
                )
            )
    return instance


@dataclass(frozen=True)
class AuthenticatedElection:
    """One manifest-addressed native election with its logical split year."""

    series: str
    year: int
    path: Path
    sha256: str
    phase: Literal["warmup", "scored"]
    instance: PBInstance


@dataclass(frozen=True)
class EndowmentFit:
    """One of the six prespecified deployed endowment-policy fits."""

    split: str
    seed: int
    path: Path
    sha256: str
    weights: tuple[float, ...]

    @property
    def context(self) -> str:
        return f"{self.split}/endowment/seed-{self.seed}"


@dataclass(frozen=True)
class AuthenticatedParityInputs:
    """Hash-bound audit inputs loaded once per process."""

    repo_root: Path
    result_root: Path
    data_dir: Path
    lock_sha256: str
    heldout_receipt_sha256: str
    manifest_sha256: str
    policy_source_sha256: str
    split_sha256: tuple[tuple[str, str], ...]
    fit_inventory_count: int
    all_fit_sha256: tuple[tuple[str, str], ...]
    endowment_fits: tuple[EndowmentFit, ...]
    elections: tuple[AuthenticatedElection, ...]
    index: Mapping[str, SeriesRef]
    splits: Mapping[str, Split]
    instances: Mapping[str, Mapping[int, PBInstance]]
    manifest_instance_sha256: Mapping[str, Mapping[int, str]]
    instance_semantic_sha256: Mapping[str, Mapping[int, str]]
    source_sha256: tuple[tuple[str, str], ...]
    authentication_fingerprint: str

    @property
    def series_count(self) -> int:
        return len(self.index)

    @property
    def election_count(self) -> int:
        return len(self.elections)


def _peak_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1024


def _report(
    *,
    stage: str,
    started: float,
    cases: int,
    scored_cases: int,
    warmup_cases: int,
    completion_counts: dict[bool, int],
    context_counts: dict[str, int],
    mismatches: tuple[ParityMismatch, ...] = (),
    lock_sha256: str = "",
    heldout_receipt_sha256: str = "",
    manifest_sha256: str = "",
    policy_source_sha256: str = "",
    source_sha256: tuple[tuple[str, str], ...] = (),
    split_sha256: tuple[tuple[str, str], ...] = (),
    fit_sha256: tuple[tuple[str, str], ...] = (),
    fit_inventory_count: int = 0,
    all_fit_sha256: tuple[tuple[str, str], ...] = (),
    context_sha256: str = "",
    gate_eligible: bool = False,
) -> ParityAuditReport:
    return ParityAuditReport(
        stage=stage,
        cases=cases,
        scored_cases=scored_cases,
        warmup_cases=warmup_cases,
        probe_cases=cases - scored_cases - warmup_cases,
        completion_counts=tuple(sorted(completion_counts.items())),
        context_counts=tuple(sorted(context_counts.items())),
        mismatches=mismatches,
        elapsed_seconds=time.perf_counter() - started,
        peak_rss_bytes=_peak_rss_bytes(),
        gate_eligible=gate_eligible,
        lock_sha256=lock_sha256,
        heldout_receipt_sha256=heldout_receipt_sha256,
        manifest_sha256=manifest_sha256,
        policy_source_sha256=policy_source_sha256,
        source_sha256=source_sha256,
        split_sha256=split_sha256,
        fit_sha256=fit_sha256,
        fit_inventory_count=fit_inventory_count,
        all_fit_sha256=all_fit_sha256,
        context_sha256=context_sha256,
    )


def _float_bits(values: Iterable[float]) -> tuple[str, ...]:
    return tuple(struct.pack("!d", float(value)).hex() for value in values)


def _authenticated_inputs_fingerprint(inputs: AuthenticatedParityInputs) -> str:
    """Hash every replay-relevant semantic and provenance field.

    ``split.train`` is deliberately excluded because replay never executes it.
    The complete raw split files, including training rows, remain provenance-
    bound by ``split_sha256`` and the verified protocol lock.
    """

    index_rows = []
    instance_rows = []
    for series, ref in inputs.index.items():
        index_rows.append(
            {
                "series": series,
                "key": ref.key,
                "years": list(ref.years),
                "paths": [str(Path(path).resolve()) for path in ref.paths],
            }
        )
        raw_digests = inputs.manifest_instance_sha256.get(series, {})
        semantic_digests = inputs.instance_semantic_sha256.get(series, {})
        for year in ref.years:
            instance_rows.append(
                {
                    "series": series,
                    "year": year,
                    "raw_sha256": raw_digests.get(year),
                    "semantic_sha256": semantic_digests.get(year),
                }
            )
    payload = {
        "roots": {
            "repo": str(Path(inputs.repo_root).resolve()),
            "result": str(Path(inputs.result_root).resolve()),
            "data": str(Path(inputs.data_dir).resolve()),
        },
        "provenance": {
            "lock_sha256": inputs.lock_sha256,
            "heldout_receipt_sha256": inputs.heldout_receipt_sha256,
            "manifest_sha256": inputs.manifest_sha256,
            "policy_source_sha256": inputs.policy_source_sha256,
            "source_sha256": list(inputs.source_sha256),
            "split_sha256": list(inputs.split_sha256),
            "fit_inventory_count": inputs.fit_inventory_count,
            "all_fit_sha256": list(inputs.all_fit_sha256),
        },
        "selected_fits": [
            {
                "context": fit.context,
                "path": str(Path(fit.path).resolve()),
                "sha256": fit.sha256,
                "weight_bits": list(_float_bits(fit.weights)),
            }
            for fit in inputs.endowment_fits
        ],
        "index": index_rows,
        "split_test": [
            {
                "name": name,
                "rows": [
                    [series, list(years)]
                    for series, years in inputs.splits[name].test
                ],
            }
            for name in SPLIT_NAMES
            if name in inputs.splits
        ],
        "instances": instance_rows,
        "elections": [
            {
                "series": row.series,
                "year": row.year,
                "path": str(Path(row.path).resolve()),
                "sha256": row.sha256,
                "phase": row.phase,
            }
            for row in inputs.elections
        ],
    }
    content = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def load_authenticated_parity_inputs(
    repo_root: Path = ROOT,
    *,
    result_root: Path | None = None,
    data_dir: Path | None = None,
) -> AuthenticatedParityInputs:
    """Authenticate all protocol inputs, then parse the native corpus once."""

    repo_root = Path(repo_root).resolve()
    result_root = (
        Path(result_root).resolve()
        if result_root is not None
        else repo_root / "results" / "iclr_multicity"
    )
    data_dir = (
        Path(data_dir).resolve()
        if data_dir is not None
        else repo_root / "data" / "pb_multicity"
    )
    if any(part.casefold() == "smoke" for part in result_root.parts):
        raise ValueError("smoke artifacts are forbidden in the parity audit")

    inventory = required_fit_inventory(result_root)
    if len(inventory) != 24:
        raise RuntimeError(
            f"expected the canonical 24-fit inventory, found {len(inventory)}"
        )
    fit_root = (result_root / "fits").resolve()
    for _, path in inventory:
        resolved = Path(path).resolve()
        if any(part.casefold() == "smoke" for part in resolved.parts):
            raise RuntimeError(f"smoke fit is forbidden: {resolved}")
        try:
            resolved.relative_to(fit_root)
        except ValueError as error:
            raise RuntimeError(
                f"fit path lies outside the authenticated fit root: {resolved}"
            ) from error

    lock_path = result_root / "protocol_lock.json"
    lock_payload = verify_multicity_protocol_lock(
        lock_path, repo_root, result_root, data_dir
    )
    verify_lock_fit_inventory(lock_payload)
    lock_sha256 = _sha256(lock_path)

    fit_snapshots: list[
        tuple[Mapping[str, object], FileSnapshot]
    ] = []
    fit_hashes: dict[str, str] = {}
    for spec, path in sorted(inventory, key=lambda item: str(item[1])):
        snapshot = _read_file_snapshot(path)
        try:
            label = str(snapshot.path.relative_to(repo_root))
        except ValueError as error:
            raise RuntimeError(
                f"fit path lies outside the repository root: {snapshot.path}"
            ) from error
        if label in fit_hashes:
            raise RuntimeError(f"duplicate fit receipt label: {label}")
        fit_snapshots.append((spec, snapshot))
        fit_hashes[label] = snapshot.sha256

    structural_snapshot = _read_file_snapshot(result_root / "structural_gates.json")
    receipt_snapshot = _read_file_snapshot(result_root / "heldout_opened.json")
    if receipt_snapshot.sha256 != EXPECTED_HELDOUT_RECEIPT_SHA256:
        raise RuntimeError(
            "held-out receipt SHA-256 differs from the frozen parity receipt: "
            f"{receipt_snapshot.sha256}"
        )
    receipt = _decode_json_snapshot(receipt_snapshot)
    _verify_heldout_receipt(
        receipt,
        lock_sha256=lock_sha256,
        fit_sha256=fit_hashes,
        structural_gates_sha256=structural_snapshot.sha256,
    )

    fit_payloads: list[
        tuple[Mapping[str, object], FileSnapshot, Mapping[str, object]]
    ] = []
    for spec, snapshot in fit_snapshots:
        payload = _decode_json_snapshot(snapshot)
        _verify_fit_snapshot_payload(
            spec, snapshot, payload, lock_payload, lock_sha256
        )
        fit_payloads.append((spec, snapshot, payload))

    manifest_path = result_root / "corpus_manifest.json"
    manifest_snapshot = _read_file_snapshot(manifest_path)
    manifest_sha256 = manifest_snapshot.sha256
    if manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "multicity manifest hash differs from the prespecified parity corpus: "
            f"{manifest_sha256}"
        )
    tracked = lock_payload["tracked_files"]
    if tracked["artifact/corpus_manifest"]["sha256"] != manifest_sha256:
        raise RuntimeError("protocol lock does not bind the authenticated manifest")

    index = load_canonical_index_from_manifest(manifest_path, data_dir)
    splits = {
        name: load_frozen_split(result_root / "splits" / f"{name}.json", index)
        for name in SPLIT_NAMES
    }
    split_hashes = tuple(
        (name, _sha256(result_root / "splits" / f"{name}.json"))
        for name in SPLIT_NAMES
    )
    for name, digest in split_hashes:
        if tracked[f"split/{name}"]["sha256"] != digest:
            raise RuntimeError(f"protocol lock does not bind split {name}")

    manifest = _decode_json_snapshot(manifest_snapshot)
    rows = {
        (str(row["series"]), int(row["year"])): row
        for row in manifest["files"]
    }
    temporal_scored = {
        (series, year)
        for series, years in splits["temporal_2022"].test
        for year in years
    }
    parsed: dict[str, Mapping[int, PBInstance]] = {}
    manifest_instance_sha256: dict[str, Mapping[int, str]] = {}
    instance_semantic_sha256: dict[str, Mapping[int, str]] = {}
    elections: list[AuthenticatedElection] = []
    for series, ref in sorted(index.items()):
        by_year: dict[int, PBInstance] = {}
        digest_by_year: dict[int, str] = {}
        semantic_by_year: dict[int, str] = {}
        for year, path in zip(ref.years, ref.paths):
            row = rows.get((series, year))
            if row is None or row["name"] != path.name:
                raise RuntimeError(
                    f"manifest does not bind indexed election {series} {year}"
                )
            instance_snapshot = _read_file_snapshot(path)
            if instance_snapshot.sha256 != row["sha256"]:
                raise RuntimeError(
                    f"parsed election bytes differ from manifest for {series} {year}"
                )
            instance = _parse_pb_snapshot(instance_snapshot)
            by_year[year] = instance
            digest_by_year[year] = str(row["sha256"])
            semantic_by_year[year] = _canonical_instance_sha256(instance)
            elections.append(
                AuthenticatedElection(
                    series=series,
                    year=year,
                    path=path,
                    sha256=str(row["sha256"]),
                    phase="scored"
                    if (series, year) in temporal_scored
                    else "warmup",
                    instance=instance,
                )
            )
        parsed[series] = MappingProxyType(by_year)
        manifest_instance_sha256[series] = MappingProxyType(digest_by_year)
        instance_semantic_sha256[series] = MappingProxyType(semantic_by_year)

    selected_fits: list[EndowmentFit] = []
    for spec, snapshot, payload in fit_payloads:
        split = str(spec["split"])
        seed = int(spec["seed"])
        if spec["arm"] != "endowment":
            continue
        if split == "temporal_2022":
            selected = seed in (1, 2, 42)
        else:
            selected = split in SPLIT_NAMES[1:] and seed == 42
        if not selected:
            continue
        result = payload["result"]
        assert isinstance(result, dict)
        weights = tuple(float(value) for value in result["best_weights"])
        selected_fits.append(
            EndowmentFit(
                split=split,
                seed=seed,
                path=snapshot.path,
                sha256=snapshot.sha256,
                weights=weights,
            )
        )
    if len(selected_fits) != 6:
        raise RuntimeError(
            f"expected six deployed endowment fits, found {len(selected_fits)}"
        )

    policy_source_sha256 = str(
        tracked["source/iclr_priority_mes.py"]["sha256"]
    )
    source_sha256 = tuple(
        (name, str(tracked[f"source/{name}"]["sha256"]))
        for name in FIT_SOURCE_FILES
    )
    inputs = AuthenticatedParityInputs(
        repo_root=repo_root,
        result_root=result_root,
        data_dir=data_dir,
        lock_sha256=lock_sha256,
        heldout_receipt_sha256=receipt_snapshot.sha256,
        manifest_sha256=manifest_sha256,
        policy_source_sha256=policy_source_sha256,
        split_sha256=split_hashes,
        fit_inventory_count=len(inventory),
        all_fit_sha256=tuple(sorted(fit_hashes.items())),
        endowment_fits=tuple(selected_fits),
        elections=tuple(elections),
        index=MappingProxyType(dict(index)),
        splits=MappingProxyType(splits),
        instances=MappingProxyType(parsed),
        manifest_instance_sha256=MappingProxyType(manifest_instance_sha256),
        instance_semantic_sha256=MappingProxyType(instance_semantic_sha256),
        source_sha256=source_sha256,
        authentication_fingerprint="",
    )
    return replace(
        inputs,
        authentication_fingerprint=_authenticated_inputs_fingerprint(inputs),
    )


def _validated_election_bindings(
    inputs: AuthenticatedParityInputs,
) -> tuple[
    Mapping[tuple[str, int], AuthenticatedElection],
    frozenset[tuple[str, int]],
]:
    """Validate every materialized row against independent authenticated maps."""

    index_series = set(inputs.index)
    if set(inputs.instances) != index_series:
        raise RuntimeError("authenticated instance series differ from the index")
    if set(inputs.manifest_instance_sha256) != index_series:
        raise RuntimeError("manifest digest series differ from the index")

    expected: dict[
        tuple[str, int], tuple[Path, str, PBInstance]
    ] = {}
    for series, ref in inputs.index.items():
        if ref.key != series or len(ref.years) != len(ref.paths):
            raise RuntimeError(f"invalid index binding for {series}")
        if len(set(ref.years)) != len(ref.years):
            raise RuntimeError(f"duplicate index years for {series}")
        instances = inputs.instances[series]
        digests = inputs.manifest_instance_sha256[series]
        if set(instances) != set(ref.years):
            raise RuntimeError(f"authenticated instance years differ for {series}")
        if set(digests) != set(ref.years):
            raise RuntimeError(f"manifest digest years differ for {series}")
        for year, path in zip(ref.years, ref.paths):
            key = (series, year)
            if key in expected:
                raise RuntimeError("index election identities contain duplicates")
            digest = digests[year]
            if not _is_sha256(digest):
                raise RuntimeError(f"invalid manifest digest for {series} {year}")
            expected[key] = (Path(path), digest, instances[year])

    temporal = inputs.splits.get("temporal_2022")
    if temporal is None or temporal.name != "temporal_2022":
        raise RuntimeError("authenticated temporal split is missing")
    temporal_rows = tuple(
        (series, year)
        for series, years in temporal.test
        for year in years
    )
    temporal_keys = frozenset(temporal_rows)
    if len(temporal_keys) != len(temporal_rows):
        raise RuntimeError("temporal-scored election identities contain duplicates")
    if not temporal_keys.issubset(expected):
        raise RuntimeError("temporal-scored election identities differ from the index")

    rows: dict[tuple[str, int], AuthenticatedElection] = {}
    for row in inputs.elections:
        key = (row.series, row.year)
        if key in rows:
            raise RuntimeError("authenticated election identities contain duplicates")
        if key not in expected:
            raise RuntimeError("authenticated election identities differ from the index")
        expected_path, expected_digest, expected_instance = expected[key]
        if Path(row.path) != expected_path:
            raise RuntimeError(f"authenticated election path differs for {key}")
        if row.sha256 != expected_digest:
            raise RuntimeError(f"authenticated election digest differs for {key}")
        expected_phase = "scored" if key in temporal_keys else "warmup"
        if row.phase != expected_phase:
            raise RuntimeError(f"authenticated election phase differs for {key}")
        if row.instance is not expected_instance:
            raise RuntimeError(f"authenticated election instance differs for {key}")
        if row.instance.path != str(expected_path):
            raise RuntimeError(f"parsed instance path differs for {key}")
        rows[key] = row
    if set(rows) != set(expected):
        raise RuntimeError("authenticated election identities differ from the index")
    return MappingProxyType(rows), temporal_keys


def select_stage_elections(
    inputs: AuthenticatedParityInputs,
    stage: str,
) -> tuple[AuthenticatedElection, ...]:
    """Select one result-independent audit stage from authenticated rows."""

    if stage not in STAGE_EXPECTED_COUNTS:
        raise ValueError(f"unknown parity audit stage: {stage!r}")
    rows, temporal_keys = _validated_election_bindings(inputs)
    by_filename: dict[str, AuthenticatedElection] = {}
    for row in rows.values():
        if row.path.name in by_filename:
            raise RuntimeError(
                f"authenticated election filename is not unique: {row.path.name}"
            )
        by_filename[row.path.name] = row
    if stage == "warsaw-sentinels":
        names = WARSAW_SENTINEL_FILENAMES
    elif stage == "representative-temporal-scored":
        names = REPRESENTATIVE_FILENAMES
    elif stage == "temporal-scored":
        selected = tuple(
            row for key, row in rows.items() if key in temporal_keys
        )
    elif stage == "native":
        selected = tuple(rows.values())
    if stage in ("warsaw-sentinels", "representative-temporal-scored"):
        missing = [name for name in names if name not in by_filename]
        if missing:
            raise RuntimeError(
                f"authenticated corpus is missing staged files: {', '.join(missing)}"
            )
        selected = tuple(by_filename[name] for name in names)
    selected_keys = {(row.series, row.year) for row in selected}
    if len(selected_keys) != len(selected):
        raise RuntimeError(f"{stage} election identities contain duplicates")
    if stage == "temporal-scored" and selected_keys != temporal_keys:
        raise RuntimeError("temporal-scored election identities differ")
    if stage == "native" and selected_keys != set(rows):
        raise RuntimeError("native election identities differ")
    observed = (
        len(selected),
        sum(row.phase == "scored" for row in selected),
        sum(row.phase == "warmup" for row in selected),
    )
    expected = STAGE_EXPECTED_COUNTS[stage]
    if observed != expected:
        raise RuntimeError(
            f"{stage} source counts differ: observed={observed}, expected={expected}"
        )
    return selected


@dataclass(frozen=True)
class _ValidatedParityContext:
    """Factory-validated, immutable replay inputs used by private gate cores."""

    repo_root: Path
    lock_sha256: str
    heldout_receipt_sha256: str
    manifest_sha256: str
    policy_source_sha256: str
    source_sha256: tuple[tuple[str, str], ...]
    split_sha256: tuple[tuple[str, str], ...]
    fit_inventory_count: int
    all_fit_sha256: tuple[tuple[str, str], ...]
    fingerprint: str
    endowment_fits: tuple[EndowmentFit, ...]
    stage_elections: Mapping[str, tuple[AuthenticatedElection, ...]]
    election_rows: Mapping[tuple[str, int], AuthenticatedElection]
    index: Mapping[str, SeriesRef]
    instances: Mapping[str, Mapping[int, PBInstance]]
    deployments: Mapping[str, tuple[tuple[str, tuple[int, ...]], ...]]


def _validate_authenticated_inputs(
    inputs: AuthenticatedParityInputs,
) -> _ValidatedParityContext:
    """Recompute the comprehensive factory fingerprint before any policy call."""

    if inputs.manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("authenticated manifest provenance differs")
    if inputs.heldout_receipt_sha256 != EXPECTED_HELDOUT_RECEIPT_SHA256:
        raise RuntimeError("authenticated receipt provenance differs")
    if not _is_sha256(inputs.lock_sha256):
        raise RuntimeError("authenticated lock provenance is invalid")

    split_hashes = dict(inputs.split_sha256)
    if (
        len(inputs.split_sha256) != len(SPLIT_NAMES)
        or set(split_hashes) != set(SPLIT_NAMES)
        or any(not _is_sha256(value) for value in split_hashes.values())
    ):
        raise RuntimeError("authenticated split provenance differs")
    source_hashes = dict(inputs.source_sha256)
    if (
        len(inputs.source_sha256) != len(FIT_SOURCE_FILES)
        or set(source_hashes) != set(FIT_SOURCE_FILES)
        or any(not _is_sha256(value) for value in source_hashes.values())
    ):
        raise RuntimeError("authenticated source provenance differs")
    if inputs.policy_source_sha256 != source_hashes["iclr_priority_mes.py"]:
        raise RuntimeError("authenticated policy source provenance differs")

    if inputs.fit_inventory_count != 24 or len(inputs.all_fit_sha256) != 24:
        raise RuntimeError("authenticated fit inventory must contain exactly 24 fits")
    all_fit_hashes = dict(inputs.all_fit_sha256)
    if len(all_fit_hashes) != 24 or any(
        not _is_sha256(value) for value in all_fit_hashes.values()
    ):
        raise RuntimeError("authenticated fit inventory differs")
    expected_fit_paths: dict[str, tuple[Mapping[str, object], Path]] = {}
    for spec, path in required_fit_inventory(inputs.result_root):
        resolved = Path(path).resolve()
        try:
            label = str(resolved.relative_to(inputs.repo_root))
        except ValueError as error:
            raise RuntimeError("authenticated fit path lies outside repository") from error
        expected_fit_paths[label] = (spec, resolved)
    if set(all_fit_hashes) != set(expected_fit_paths):
        raise RuntimeError("authenticated fit inventory paths differ")

    selected_specs: dict[str, tuple[Path, str]] = {}
    for label, (spec, path) in expected_fit_paths.items():
        split = str(spec["split"])
        seed = int(spec["seed"])
        if spec["arm"] != "endowment":
            continue
        if split == "temporal_2022":
            selected = seed in (1, 2, 42)
        else:
            selected = split in SPLIT_NAMES[1:] and seed == 42
        if selected:
            selected_specs[f"{split}/endowment/seed-{seed}"] = (
                path,
                all_fit_hashes[label],
            )
    fits = {fit.context: fit for fit in inputs.endowment_fits}
    if len(fits) != 6 or set(fits) != set(selected_specs):
        raise RuntimeError("authenticated six selected fit contexts differ")
    for context, fit in fits.items():
        expected_path, expected_digest = selected_specs[context]
        if Path(fit.path).resolve() != expected_path:
            raise RuntimeError(f"authenticated selected-fit path differs: {context}")
        if fit.sha256 != expected_digest:
            raise RuntimeError(f"authenticated selected-fit digest differs: {context}")
        if len(fit.weights) != 6 or any(
            not math.isfinite(value) for value in fit.weights
        ):
            raise RuntimeError(f"authenticated selected-fit weights differ: {context}")

    _verify_frozen_fit_source_counts(inputs)
    election_rows, _ = _validated_election_bindings(inputs)
    if set(inputs.instance_semantic_sha256) != set(inputs.index):
        raise RuntimeError("authenticated semantic instance series differ")
    for series, ref in inputs.index.items():
        semantic_by_year = inputs.instance_semantic_sha256[series]
        if set(semantic_by_year) != set(ref.years):
            raise RuntimeError(f"authenticated semantic instance years differ: {series}")
        for year in ref.years:
            semantic = _canonical_instance_sha256(inputs.instances[series][year])
            if semantic != semantic_by_year[year]:
                raise RuntimeError(
                    f"authenticated semantic instance fingerprint differs: {series} {year}"
                )

    observed_fingerprint = _authenticated_inputs_fingerprint(inputs)
    if (
        not _is_sha256(inputs.authentication_fingerprint)
        or observed_fingerprint != inputs.authentication_fingerprint
    ):
        raise RuntimeError("authenticated input fingerprint differs")

    stage_elections = MappingProxyType(
        {
            stage: select_stage_elections(inputs, stage)
            for stage in STAGE_EXPECTED_COUNTS
        }
    )
    index = MappingProxyType(dict(inputs.index))
    instances = MappingProxyType(
        {
            series: MappingProxyType(dict(by_year))
            for series, by_year in inputs.instances.items()
        }
    )
    deployments = MappingProxyType(
        {name: tuple(inputs.splits[name].test) for name in SPLIT_NAMES}
    )
    return _ValidatedParityContext(
        repo_root=Path(inputs.repo_root),
        lock_sha256=inputs.lock_sha256,
        heldout_receipt_sha256=inputs.heldout_receipt_sha256,
        manifest_sha256=inputs.manifest_sha256,
        policy_source_sha256=inputs.policy_source_sha256,
        source_sha256=tuple(inputs.source_sha256),
        split_sha256=tuple(inputs.split_sha256),
        fit_inventory_count=inputs.fit_inventory_count,
        all_fit_sha256=tuple(inputs.all_fit_sha256),
        fingerprint=inputs.authentication_fingerprint,
        endowment_fits=tuple(inputs.endowment_fits),
        stage_elections=stage_elections,
        election_rows=election_rows,
        index=index,
        instances=instances,
        deployments=deployments,
    )


def _audit_uniform_context(
    context: _ValidatedParityContext,
    stage: str,
    *,
    completion_modes: tuple[bool, ...],
    gate_eligible: bool,
) -> ParityAuditReport:
    """Run one uniform stage using only a prevalidated local context."""

    _require_hard_gate_modes(completion_modes)
    try:
        elections = context.stage_elections[stage]
    except KeyError as error:
        raise ValueError(f"unknown parity audit stage: {stage!r}") from error

    def cases() -> Iterable[ParityCase]:
        for completion in completion_modes:
            for election in elections:
                n_voters = len(election.instance.votes)
                endowments = (
                    tuple(
                        election.instance.budget / n_voters
                        for _ in range(n_voters)
                    )
                    if n_voters
                    else ()
                )
                yield ParityCase(
                    instance=election.instance,
                    endowments=endowments,
                    completion=completion,
                    context=f"uniform/{stage}",
                    phase=election.phase,
                    instance_sha256=election.sha256,
                )

    temporal_hash = tuple(
        row for row in context.split_sha256 if row[0] == "temporal_2022"
    )
    try:
        report = audit_partial_diagnostics(cases(), stage=stage)
    except ParityMismatchError as error:
        bound = replace(
            error.report,
            lock_sha256=context.lock_sha256,
            heldout_receipt_sha256=context.heldout_receipt_sha256,
            manifest_sha256=context.manifest_sha256,
            policy_source_sha256=context.policy_source_sha256,
            source_sha256=context.source_sha256,
            split_sha256=temporal_hash,
            fit_inventory_count=context.fit_inventory_count,
            all_fit_sha256=context.all_fit_sha256,
            context_sha256=context.fingerprint,
            gate_eligible=gate_eligible,
        )
        raise ParityMismatchError(bound) from None
    expected_rows = STAGE_EXPECTED_COUNTS[stage]
    expected = tuple(value * 2 for value in expected_rows)
    observed = (report.cases, report.scored_cases, report.warmup_cases)
    if observed != expected:
        raise RuntimeError(
            f"{stage} hard-gate counts differ: observed={observed}, expected={expected}"
        )
    return replace(
        report,
        lock_sha256=context.lock_sha256,
        heldout_receipt_sha256=context.heldout_receipt_sha256,
        manifest_sha256=context.manifest_sha256,
        policy_source_sha256=context.policy_source_sha256,
        source_sha256=context.source_sha256,
        split_sha256=temporal_hash,
        fit_inventory_count=context.fit_inventory_count,
        all_fit_sha256=context.all_fit_sha256,
        context_sha256=context.fingerprint,
        gate_eligible=gate_eligible,
    )


def audit_uniform_stage(
    inputs: AuthenticatedParityInputs,
    stage: str,
    *,
    completion_modes: tuple[bool, ...] = (False, True),
) -> ParityAuditReport:
    """Run caller-supplied uniform diagnostics; never return a hard gate."""

    _require_hard_gate_modes(completion_modes)
    context = _validate_authenticated_inputs(inputs)
    return _audit_uniform_context(
        context,
        stage,
        completion_modes=completion_modes,
        gate_eligible=False,
    )


def run_authenticated_uniform_stage(
    stage: str,
    *,
    repo_root: Path = ROOT,
    result_root: Path | None = None,
    data_dir: Path | None = None,
    completion_modes: tuple[bool, ...] = (False, True),
) -> ParityAuditReport:
    """Load, authenticate, seal, and execute one uniform hard gate."""

    _require_hard_gate_modes(completion_modes)
    context = _validate_authenticated_inputs(
        load_authenticated_parity_inputs(
            repo_root,
            result_root=result_root,
            data_dir=data_dir,
        )
    )
    return _audit_uniform_context(
        context,
        stage,
        completion_modes=completion_modes,
        gate_eligible=True,
    )


def _authenticated_election_sha256(
    inputs: AuthenticatedParityInputs, series: str, year: int
) -> str:
    matches = [
        election.sha256
        for election in inputs.elections
        if election.series == series and election.year == year
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"authenticated election digest is not unique for {series} {year}"
        )
    return matches[0]


def _verify_frozen_fit_source_counts(inputs: AuthenticatedParityInputs) -> None:
    contexts = [fit.context for fit in inputs.endowment_fits]
    if len(contexts) != 6 or set(contexts) != set(FIT_CONTEXT_EXPECTED_COUNTS):
        raise RuntimeError(
            "full gate requires exactly the six prespecified fit contexts"
        )
    if len(set(contexts)) != len(contexts):
        raise RuntimeError("full gate fit contexts must be unique")

    split_names = {fit.split for fit in inputs.endowment_fits}
    for split_name in sorted(split_names):
        split = inputs.splits.get(split_name)
        if split is None or split.name != split_name:
            raise RuntimeError(f"missing authenticated split {split_name}")
        series_rows = tuple(series for series, _ in split.test)
        if len(series_rows) != len(set(series_rows)):
            raise RuntimeError(f"{split_name} split.test identities contain duplicates")
        observed: dict[str, tuple[int, ...]] = {}
        for series, years in split.test:
            year_tuple = tuple(years)
            if len(year_tuple) != len(set(year_tuple)):
                raise RuntimeError(
                    f"{split_name} split.test identities contain duplicate years"
                )
            observed[series] = year_tuple
        if split_name == "temporal_2022":
            eligible = inputs.index.items()
        elif split_name.startswith("city_out_"):
            heldout_city = split_name.removeprefix("city_out_").replace("_", "/", 1)
            eligible = (
                (series, ref)
                for series, ref in inputs.index.items()
                if ref.city == heldout_city
            )
        else:
            raise RuntimeError(f"unknown deployment split {split_name}")
        expected = {
            series: tuple(year for year in ref.years if year >= FIRST_TEST_YEAR)
            for series, ref in eligible
        }
        if observed != expected:
            raise RuntimeError(
                f"{split_name} split.test identities differ from canonical semantics"
            )

    observed: dict[str, tuple[int, int, int]] = {}
    for fit in inputs.endowment_fits:
        split = inputs.splits.get(fit.split)
        if split is None or split.name != fit.split:
            raise RuntimeError(f"missing authenticated split for {fit.context}")
        total = scoring = warming = 0
        for series, scored_years in split.test:
            ref = inputs.index.get(series)
            if ref is None:
                raise RuntimeError(f"unknown deployment series {series}")
            scored_set = set(scored_years)
            if len(scored_set) != len(scored_years) or not scored_set.issubset(ref.years):
                raise RuntimeError(f"invalid scored years for {fit.context}: {series}")
            total += len(ref.years)
            scoring += len(scored_set)
            warming += len(ref.years) - len(scored_set)
        observed[fit.context] = (total, scoring, warming)
    if observed != FIT_CONTEXT_EXPECTED_COUNTS:
        raise RuntimeError(
            "full-gate fit source counts differ: "
            f"observed={observed}, expected={FIT_CONTEXT_EXPECTED_COUNTS}"
        )


def _audit_frozen_context(
    context: _ValidatedParityContext,
    *,
    completion_modes: tuple[bool, ...],
    gate_eligible: bool,
) -> ParityAuditReport:
    """Replay the six fits using only a prevalidated local context."""

    _require_hard_gate_modes(completion_modes)
    started = time.perf_counter()
    total = scored = warmup = 0
    completion_counts: dict[bool, int] = {}
    context_counts: dict[str, int] = {}
    fit_hashes = tuple((fit.context, fit.sha256) for fit in context.endowment_fits)

    def current_report(
        mismatches: tuple[ParityMismatch, ...] = (),
    ) -> ParityAuditReport:
        return _report(
            stage="full-split-correct-fits",
            started=started,
            cases=total,
            scored_cases=scored,
            warmup_cases=warmup,
            completion_counts=completion_counts,
            context_counts=context_counts,
            mismatches=mismatches,
            lock_sha256=context.lock_sha256,
            heldout_receipt_sha256=context.heldout_receipt_sha256,
            manifest_sha256=context.manifest_sha256,
            policy_source_sha256=context.policy_source_sha256,
            source_sha256=context.source_sha256,
            split_sha256=context.split_sha256,
            fit_sha256=fit_hashes,
            fit_inventory_count=context.fit_inventory_count,
            all_fit_sha256=context.all_fit_sha256,
            context_sha256=context.fingerprint,
            gate_eligible=gate_eligible,
        )

    for completion in completion_modes:
        for fit in context.endowment_fits:
            policy = linear_policy(fit.weights, scheme="age_sex")
            for series, scored_years in context.deployments[fit.split]:
                state = RolloutState()
                scored_set = set(scored_years)
                ref = context.index[series]
                for year in ref.years:
                    instance = context.instances[series][year]
                    endowments = tuple(policy(instance, state))
                    observed = compare_three_paths(
                        instance,
                        endowments,
                        completion=completion,
                        instance_sha256=context.election_rows[(series, year)].sha256,
                    )
                    phase = "scored" if year in scored_set else "warmup"
                    total += 1
                    scored += phase == "scored"
                    warmup += phase == "warmup"
                    completion_counts[completion] = (
                        completion_counts.get(completion, 0) + 1
                    )
                    context_counts[fit.context] = (
                        context_counts.get(fit.context, 0) + 1
                    )
                    if not observed.exact:
                        mismatch = ParityMismatch(
                            context=fit.context,
                            instance_path=instance.path,
                            instance_sha256=observed.instance_sha256,
                            completion=completion,
                            reference=observed.reference,
                            witness=observed.trace,
                            production=observed.production,
                        )
                        raise ParityMismatchError(current_report((mismatch,)))
                    # The next election sees only this policy's agreed outcome.
                    state.update(
                        group_outcome(instance, set(observed.reference), "age_sex")
                    )

    modes = len(completion_modes)
    expected = (1_588 * modes, 892 * modes, 696 * modes)
    observed_counts = (total, scored, warmup)
    if observed_counts != expected:
        raise RuntimeError(
            "split-correct fit audit count mismatch: "
            f"observed={observed_counts}, expected={expected}"
        )
    if any(completion_counts[mode] != 1_588 for mode in completion_modes):
        raise RuntimeError(
            f"completion-mode fit counts are not 1,588 each: {completion_counts}"
        )
    return current_report()


def audit_frozen_endowment_fits(
    inputs: AuthenticatedParityInputs,
    *,
    completion_modes: tuple[bool, ...] = (False, True),
) -> ParityAuditReport:
    """Run caller-supplied fit diagnostics; never return a hard gate."""

    _require_hard_gate_modes(completion_modes)
    context = _validate_authenticated_inputs(inputs)
    return _audit_frozen_context(
        context,
        completion_modes=completion_modes,
        gate_eligible=False,
    )


def run_authenticated_frozen_endowment_fits(
    *,
    repo_root: Path = ROOT,
    result_root: Path | None = None,
    data_dir: Path | None = None,
    completion_modes: tuple[bool, ...] = (False, True),
) -> ParityAuditReport:
    """Load, authenticate, seal, and execute the full split-correct hard gate."""

    _require_hard_gate_modes(completion_modes)
    context = _validate_authenticated_inputs(
        load_authenticated_parity_inputs(
            repo_root,
            result_root=result_root,
            data_dir=data_dir,
        )
    )
    return _audit_frozen_context(
        context,
        completion_modes=completion_modes,
        gate_eligible=True,
    )


def _canonical_instance_sha256(inst: PBInstance) -> str:
    """Hash synthetic/in-memory semantics without consulting ``inst.path``."""

    payload = {
        "meta": inst.meta,
        "projects": [
            {
                "pid": project.pid,
                "cost": project.cost,
                "selected": project.selected,
                "name": project.name,
                "category": project.category,
                "target": project.target,
                "neighborhood": project.neighborhood,
            }
            for _, project in sorted(inst.projects.items())
        ],
        "votes": [
            {
                "vid": vote.vid,
                "projects": list(vote.projects),
                "age": vote.age,
                "sex": vote.sex,
                "neighborhood": vote.neighborhood,
            }
            for vote in inst.votes
        ],
    }
    content = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def compare_three_paths(
    inst: PBInstance,
    endowments: Iterable[float],
    *,
    completion: bool,
    instance_sha256: str | None = None,
) -> ThreePathParity:
    """Run the same materialized endowments through all three real kernels."""

    n_voters = len(inst.votes)
    raw_budget = inst.meta.get("budget", "0")
    try:
        # Match PBInstance's production normalization without inheriting its
        # malformed-text fallback to zero.
        budget = float(raw_budget.replace(",", "."))
    except (AttributeError, TypeError, ValueError):
        if n_voters == 0:
            raise ValueError(
                "an empty instance must have exactly zero budget"
            ) from None
        raise ValueError(
            "a nonempty instance budget must be positive and finite"
        ) from None
    if n_voters == 0 and (not math.isfinite(budget) or budget != 0.0):
        raise ValueError("an empty instance must have exactly zero budget")
    if n_voters and (not math.isfinite(budget) or budget <= 0.0):
        raise ValueError("a nonempty instance budget must be positive and finite")
    materialized = tuple(float(value) for value in endowments)
    if len(materialized) != n_voters:
        raise ValueError("one endowment is required for every voter")
    if any(not math.isfinite(value) for value in materialized):
        raise ValueError("endowments must be finite")
    if any(value < 0.0 for value in materialized):
        raise ValueError("endowments must be nonnegative")
    tolerance = 1e-6 * max(1.0, abs(budget))
    if not math.isclose(
        math.fsum(materialized), budget, rel_tol=0.0, abs_tol=tolerance
    ):
        raise ValueError("endowments must sum to the municipal budget")
    if instance_sha256 is None:
        instance_sha256 = _canonical_instance_sha256(inst)
    elif not _is_sha256(instance_sha256):
        raise ValueError("instance SHA-256 must be 64 lowercase hexadecimal digits")
    reference = tuple(
        sorted(
            mes_with_endowments(
                inst, list(materialized), completion=completion
            )
        )
    )
    trace = tuple(
        sorted(
            trace_equal_shares(
                inst, materialized, completion=completion
            ).winners
        )
    )
    production = tuple(
        sorted(
            fast_mes_with_endowments(
                inst, materialized, completion=completion
            )
        )
    )
    return ThreePathParity(
        instance_sha256=instance_sha256,
        completion=completion,
        reference=reference,
        trace=trace,
        production=production,
    )


def audit_partial_diagnostics(
    cases: Iterable[ParityCase],
    *,
    stage: str,
) -> ParityAuditReport:
    """Run bounded diagnostics; this API is explicitly not a hard gate."""

    started = time.perf_counter()
    total = scored = warmup = 0
    completion_counts: dict[bool, int] = {}
    context_counts: dict[str, int] = {}
    for case in cases:
        observed = compare_three_paths(
            case.instance,
            case.endowments,
            completion=case.completion,
            instance_sha256=case.instance_sha256,
        )
        total += 1
        scored += case.phase == "scored"
        warmup += case.phase == "warmup"
        completion_counts[case.completion] = (
            completion_counts.get(case.completion, 0) + 1
        )
        context_counts[case.context] = context_counts.get(case.context, 0) + 1
        if not observed.exact:
            mismatch = ParityMismatch(
                context=case.context,
                instance_path=case.instance.path,
                instance_sha256=observed.instance_sha256,
                completion=case.completion,
                reference=observed.reference,
                witness=observed.trace,
                production=observed.production,
            )
            report = _report(
                stage=stage,
                started=started,
                cases=total,
                scored_cases=scored,
                warmup_cases=warmup,
                completion_counts=completion_counts,
                context_counts=context_counts,
                mismatches=(mismatch,),
            )
            raise ParityMismatchError(report)
    return _report(
        stage=stage,
        started=started,
        cases=total,
        scored_cases=scored,
        warmup_cases=warmup,
        completion_counts=completion_counts,
        context_counts=context_counts,
    )
