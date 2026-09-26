"""Trusted capture, schema, and semantic identity contracts for fig_external."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

import iclr_fig_external as external_figure
from iclr_fig_external import load_plot_tables


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis-output" / "external-support-set"
TRUSTED_TASK2_HASHES = {
    "effects.csv": "1d384ae167d88fae6a3b90db8ab358caaae7ef583ebde05012885aae62a424cf",
    "city_intervals.csv": "4dbbaa952ec65e2d2a49f23c65731fea8d70768a24552ac9acfe3dc42a3114ff",
    "diagnostics.json": "c2c34f16d3bf05a59f7c3a15a9fc049d3c0a60f40803c9f56d8e9d9fe0a58dcb",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_analysis_bundle(tmp_path: Path) -> Path:
    analysis_dir = tmp_path / "analysis"
    shutil.copytree(ANALYSIS_DIR, analysis_dir)
    return analysis_dir


def _reanchor_copied_input(
    monkeypatch: pytest.MonkeyPatch, analysis_dir: Path, relative: str
) -> None:
    digest = _sha256(analysis_dir / relative)
    manifest_path = analysis_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_sha256"][relative] = digest
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    trusted = dict(TRUSTED_TASK2_HASHES)
    trusted[relative] = digest
    monkeypatch.setattr(external_figure, "TRUSTED_TASK2_OUTPUT_SHA256", trusted)


def test_plot_tables_preserve_the_final_task_2_contract() -> None:
    effects, intervals, diagnostics = load_plot_tables(ANALYSIS_DIR)

    assert effects.shape == (192, 10)
    assert intervals.shape == (6, 16)
    assert diagnostics["decision"] == "falsified"
    assert diagnostics["primary"] == {
        "losses": 4,
        "seed": 42,
        "ties": 20,
        "wins": 8,
    }
    assert dict(external_figure.TRUSTED_TASK2_OUTPUT_SHA256) == TRUSTED_TASK2_HASHES


@pytest.mark.parametrize("manifest_change", ["missing-input-map", "wrong-input-anchor"])
def test_loader_rejects_incomplete_or_reanchored_manifest(
    tmp_path: Path, manifest_change: str
) -> None:
    analysis_dir = _copy_analysis_bundle(tmp_path)
    manifest_path = analysis_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest_change == "missing-input-map":
        manifest.pop("input_sha256")
    else:
        manifest["input_sha256"]["protocol_lock.json"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        load_plot_tables(analysis_dir)


def test_loader_rejects_mutable_manifest_reauthentication(tmp_path: Path) -> None:
    analysis_dir = _copy_analysis_bundle(tmp_path)
    effects_path = analysis_dir / "effects.csv"
    effects_path.write_bytes(effects_path.read_bytes() + b"\n")
    manifest_path = analysis_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_sha256"]["effects.csv"] = _sha256(effects_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError, match="trusted Task 2 SHA-256 mismatch for effects.csv"
    ):
        load_plot_tables(analysis_dir)


def test_loader_rejects_replacement_during_single_read_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis_dir = _copy_analysis_bundle(tmp_path)
    effects_path = analysis_dir / "effects.csv"
    effects_identity = effects_path.stat()
    replacement = tmp_path / "replacement.csv"
    replacement.write_bytes(b"replacement must never be parsed\n")
    real_read = os.read
    replaced = False

    def replace_after_read(file_descriptor: int, size: int) -> bytes:
        nonlocal replaced
        captured = real_read(file_descriptor, size)
        identity = os.fstat(file_descriptor)
        if (
            not replaced
            and captured
            and (identity.st_dev, identity.st_ino)
            == (effects_identity.st_dev, effects_identity.st_ino)
        ):
            os.replace(replacement, effects_path)
            replaced = True
        return captured

    monkeypatch.setattr(os, "read", replace_after_read)

    with pytest.raises(ValueError, match="replaced or changed during capture"):
        load_plot_tables(analysis_dir)
    assert replaced


def test_loader_parses_the_exact_captured_csv_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_read_csv = pd.read_csv
    captured_hashes: list[str] = []

    def require_captured_bytes(
        source: object, *args: object, **kwargs: object
    ) -> pd.DataFrame:
        assert isinstance(source, io.BytesIO)
        captured_hashes.append(hashlib.sha256(source.getvalue()).hexdigest())
        return real_read_csv(source, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", require_captured_bytes)
    effects, intervals, _ = load_plot_tables(ANALYSIS_DIR)

    assert effects.shape == (192, 10)
    assert intervals.shape == (6, 16)
    assert captured_hashes == [
        TRUSTED_TASK2_HASHES["effects.csv"],
        TRUSTED_TASK2_HASHES["city_intervals.csv"],
    ]


@pytest.mark.parametrize(
    "semantic_change", ("policy", "series", "city", "direction", "finite")
)
def test_loader_rejects_semantically_retampered_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, semantic_change: str
) -> None:
    analysis_dir = _copy_analysis_bundle(tmp_path)
    path = analysis_dir / "effects.csv"
    effects = pd.read_csv(path)
    if semantic_change == "policy":
        effects.loc[0, "policy"] = "endowment/seed-2"
    elif semantic_change == "series":
        original = effects.loc[0, "series"]
        effects.loc[effects["series"] == original, "series"] = "Poland/Katowice/Unknown"
    elif semantic_change == "city":
        effects.loc[0, "city"] = "Poland/Krakow"
    elif semantic_change == "direction":
        row = effects.index[effects["delta_csd"] < 0][0]
        effects.loc[row, "direction"] = "loss"
    else:
        effects.loc[0, "welfare_ratio"] = np.inf
    effects.to_csv(path, index=False, lineterminator="\n")
    _reanchor_copied_input(monkeypatch, analysis_dir, "effects.csv")

    with pytest.raises(ValueError):
        load_plot_tables(analysis_dir)


@pytest.mark.parametrize("semantic_change", ("outside-interval", "finite"))
def test_loader_rejects_semantically_retampered_intervals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, semantic_change: str
) -> None:
    analysis_dir = _copy_analysis_bundle(tmp_path)
    path = analysis_dir / "city_intervals.csv"
    intervals = pd.read_csv(path)
    if semantic_change == "outside-interval":
        intervals.loc[0, "difference"] = intervals.loc[0, "ci_high"] + 0.01
    else:
        intervals.loc[0, "baseline_csd"] = np.inf
    intervals.to_csv(path, index=False, lineterminator="\n")
    _reanchor_copied_input(monkeypatch, analysis_dir, "city_intervals.csv")

    with pytest.raises(ValueError):
        load_plot_tables(analysis_dir)


def test_loader_rejects_semantically_retampered_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    analysis_dir = _copy_analysis_bundle(tmp_path)
    path = analysis_dir / "diagnostics.json"
    diagnostics = json.loads(path.read_text(encoding="utf-8"))
    diagnostics["primary"]["wins"] = 7
    path.write_text(json.dumps(diagnostics), encoding="utf-8")
    _reanchor_copied_input(monkeypatch, analysis_dir, "diagnostics.json")

    with pytest.raises(ValueError, match="diagnostics"):
        load_plot_tables(analysis_dir)
