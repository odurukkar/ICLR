"""Synthetic Task 6 runtime, provenance, aggregation, and manifest tests."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import textwrap

import pytest

import iclr_residual_train as runtime
import test_iclr_residual_train as task5_fixtures


APPROVED_RUNTIME = {
    "os_name": "macOS",
    "os_version": "26.0",
    "os_build": "25A1",
    "architecture": "arm64",
    "model_name": "Mac mini",
    "chip_name": "Apple M5",
    "physical_memory_bytes": 30064771072,
    "physical_memory_display": "28 GiB",
    "python_implementation": "CPython",
    "python_version": "3.12.11",
    "numpy_version": "2.3.2",
    "execution_device": "cpu",
    "workers": 8,
    "peak_memory_method": "resource.getrusage(RUSAGE_SELF,RUSAGE_CHILDREN)",
    "peak_memory_units": "bytes",
    "peak_memory_interpretation": (
        "maximum observed per-process RSS, not concurrent aggregate memory"
    ),
}

FORBIDDEN_IDENTIFIERS = {
    "serial_number": "SERIAL-SENTINEL",
    "platform_UUID": "UUID-SENTINEL",
    "provisioning_UDID": "UDID-SENTINEL",
    "machine_model": "MODEL-NUMBER-SENTINEL",
    "order_number": "ORDER-SENTINEL",
    "activation_lock_status": "LOCK-SENTINEL",
    "boot_rom_version": "BOOTROM-SENTINEL",
    "os_loader_version": "LOADER-SENTINEL",
}


def _synthetic_inputs():
    folds, fit_records, series_records = task5_fixtures._development_gate_inputs_for_test()
    normalized_folds = []
    normalized_fit = []
    normalized_series = []
    for fold, fit_record, series_record in zip(folds, fit_records, series_records):
        split_sha = fold.provenance["split_sha256"]
        if len(split_sha) == 64:
            normalized_folds.append(fold)
            normalized_fit.append(fit_record)
            normalized_series.append(series_record)
            continue
        changed = object.__new__(type(fold))
        for name in (
            "name",
            "kind",
            "train",
            "test",
            "source_manifest_sha256",
            "provenance",
        ):
            value = getattr(fold, name)
            if name == "provenance":
                value = {
                    **dict(value),
                    "split_sha256": hashlib.sha256(split_sha.encode()).hexdigest(),
                }
            object.__setattr__(changed, name, value)
        fold_input = runtime._fold_input_payload(changed)
        fold_input_sha = runtime._canonical_sha256(fold_input)
        changed_fit = copy.deepcopy(fit_record)
        changed_fit.update(
            {
                "provenance": dict(changed.provenance),
                "fold_input": fold_input,
                "fold_input_sha256": fold_input_sha,
            }
        )
        _rehash(changed_fit)
        changed_series = copy.deepcopy(series_record)
        changed_series.update(
            {
                "provenance": dict(changed.provenance),
                "fold_input": fold_input,
                "fold_input_sha256": fold_input_sha,
                "fit_payload_sha256": changed_fit["payload_sha256"],
            }
        )
        _rehash(changed_series)
        normalized_folds.append(changed)
        normalized_fit.append(changed_fit)
        normalized_series.append(changed_series)
    folds = tuple(normalized_folds)
    fit_records = tuple(normalized_fit)
    series_records = tuple(normalized_series)
    provenance = {
        "old_pabulib_commit": "2f4321fec84069f50abf35fb5e90852013d17070",
        "city_corpus_manifest_sha256": folds[0].source_manifest_sha256,
        "city_split_sha256": {
            fold.name: fold.provenance["split_sha256"]
            for fold in folds
            if fold.kind == "city"
        },
        "district_split_sha256": {
            fold.name: fold.provenance["split_sha256"]
            for fold in folds
            if fold.kind == "district"
        },
        "fold_input_sha256": {
            fold.name: runtime._fold_input_sha256(fold) for fold in folds
        },
    }
    return folds, provenance, fit_records, series_records


def _install_synthetic_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    result_root = tmp_path / "results-development"
    analysis_root = tmp_path / "analysis-development"
    folds, provenance, fit_records, series_records = _synthetic_inputs()
    monkeypatch.setattr(runtime, "DEVELOPMENT_ROOT", result_root)
    monkeypatch.setattr(
        runtime, "DEVELOPMENT_ANALYSIS_ROOT", analysis_root, raising=False
    )
    monkeypatch.setattr(
        runtime,
        "_measure_production_runtime",
        lambda workers: {**APPROVED_RUNTIME, "workers": workers},
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "_collect_development_preparation_inputs",
        lambda: (folds, copy.deepcopy(provenance)),
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "_runtime_utc_now",
        lambda: "2026-08-19T00:00:00Z",
        raising=False,
    )
    clock = iter(float(value) for value in range(1000))
    monkeypatch.setattr(
        runtime, "_runtime_monotonic", lambda: next(clock), raising=False
    )
    monkeypatch.setattr(
        runtime,
        "_runtime_peak_memory_sample",
        lambda previous=None: {
            "parent_max_rss_bytes": 1048576,
            "completed_children_max_rss_bytes": 524288,
            "reported_peak_max_rss_bytes": 1048576,
        },
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "_load_development_folds_unchecked",
        lambda: pytest.fail("synthetic runtime reached the real fold loader"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "_load_city_development_folds",
        lambda: pytest.fail("synthetic runtime reached the real city loader"),
    )
    monkeypatch.setattr(
        runtime,
        "_load_district_development_folds",
        lambda: pytest.fail("synthetic runtime reached the real district loader"),
    )
    monkeypatch.setattr(
        runtime,
        "load_series",
        lambda *args, **kwargs: pytest.fail("synthetic runtime parsed PB data"),
    )
    return result_root, analysis_root, folds, fit_records, series_records


def _install_fake_fitter(
    monkeypatch: pytest.MonkeyPatch,
    folds,
    fit_records,
    series_records,
    *,
    fail_after: int | None = None,
):
    by_name = {
        fold.name: (copy.deepcopy(fit), copy.deepcopy(series))
        for fold, fit, series in zip(folds, fit_records, series_records)
    }
    calls: list[str] = []

    def fake_fit(fold, workers):
        assert workers == 8
        calls.append(fold.name)
        fit_record, series_record = by_name[fold.name]
        root = runtime.DEVELOPMENT_ROOT / "folds" / fold.name
        root.mkdir(parents=True, exist_ok=True)
        (root / "fit.json").write_bytes(runtime._canonical_json_bytes(fit_record))
        (root / "per_series.json").write_bytes(
            runtime._canonical_json_bytes(series_record)
        )
        if fail_after is not None and len(calls) == fail_after:
            raise RuntimeError("synthetic interruption")
        return copy.deepcopy(series_record["result"])

    monkeypatch.setattr(runtime, "fit_and_evaluate_development_fold", fake_fit)
    return calls


def _prepare(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    installed = _install_synthetic_runtime(monkeypatch, tmp_path)
    runtime._prepare_development(8)
    return installed


def _seal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    result_root, analysis_root, folds, fit_records, series_records = _prepare(
        monkeypatch, tmp_path
    )
    calls = _install_fake_fitter(
        monkeypatch, folds, fit_records, series_records
    )
    decision = runtime._run_guarded_development(8)
    return result_root, analysis_root, folds, fit_records, series_records, calls, decision


def _tree_snapshot(*roots: Path) -> dict[str, tuple[str, bytes | None]]:
    snapshot = {}
    for root in roots:
        if not root.exists():
            continue
        for path in sorted((root, *root.rglob("*")), key=lambda value: str(value)):
            relative = f"{root.name}/{path.relative_to(root).as_posix()}"
            if path.is_symlink():
                snapshot[relative] = ("symlink", str(path.readlink()).encode())
            elif path.is_dir():
                snapshot[relative] = ("directory", None)
            else:
                snapshot[relative] = ("file", path.read_bytes())
    return snapshot


def _rehash(payload: dict[str, object]) -> None:
    payload.pop("payload_sha256", None)
    payload["payload_sha256"] = runtime._canonical_sha256(payload)


def _replace_frozen_fold_for_runtime_test(fold, **changes):
    changed = object.__new__(type(fold))
    for name in (
        "name",
        "kind",
        "train",
        "test",
        "source_manifest_sha256",
        "provenance",
    ):
        object.__setattr__(changed, name, changes.get(name, getattr(fold, name)))
    return changed


def _fix1_metadata_descriptors(folds) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "name": fold.name,
            "kind": fold.kind,
            "source_manifest_sha256": fold.source_manifest_sha256,
            "split_sha256": fold.provenance["split_sha256"],
            "input_blob_sha256": dict(
                fold.provenance.get("input_blob_sha256", {})
            ),
        }
        for fold in folds
    )


def _install_fix1_metadata_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    installed = _install_synthetic_runtime(monkeypatch, tmp_path)
    _, _, folds, _, _ = installed
    _, provenance, _, _ = _synthetic_inputs()
    descriptors = _fix1_metadata_descriptors(folds)
    inputs = {
        **copy.deepcopy(provenance),
        "old_input_sha256": {},
    }
    monkeypatch.setattr(
        runtime,
        "_collect_development_preparation_inputs",
        lambda: (copy.deepcopy(descriptors), copy.deepcopy(inputs)),
    )
    monkeypatch.setattr(
        runtime, "_load_development_folds_unchecked", lambda: folds, raising=False
    )
    monkeypatch.setattr(
        runtime,
        "_load_city_development_folds",
        lambda: pytest.fail("metadata binding reached the real city loader"),
    )
    monkeypatch.setattr(
        runtime,
        "_load_district_development_folds",
        lambda: pytest.fail("metadata binding reached the real district loader"),
    )
    monkeypatch.setattr(
        runtime,
        "load_series",
        lambda *args, **kwargs: pytest.fail("metadata binding parsed PB data"),
    )
    return installed, descriptors


def _write_rehashed_run_log(path: Path, mutation) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    _rehash(payload)
    path.write_bytes(runtime._canonical_json_bytes(payload))
    return payload


def _fix1_synthetic_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, Path]:
    checkout = _make_synthetic_checkout(tmp_path / "checkout")
    city_results = checkout / "results" / "iclr_multicity"
    city_data = checkout / "data" / "pb_multicity"
    district_splits = checkout / "results" / "iclr_splits"
    district_data = checkout / "data" / "pb"
    for path in (
        city_results / "splits",
        city_data,
        district_splits,
        district_data,
    ):
        path.mkdir(parents=True, exist_ok=True)
    source_rows = (
        ("approved-score-2022.pb", "Poland/Test/Approved", 2022, True),
        ("approved-score-2023.pb", "Poland/Test/Approved", 2023, True),
        ("approved-train-2022.pb", "Poland/Train/Approved", 2022, True),
        ("city-only-2022.pb", "Poland/CityOnly/Approved", 2022, False),
    )
    manifest_rows = []
    for filename, series, year, in_district in source_rows:
        encoded = f"approved-old-input:{filename}\n".encode()
        (city_data / filename).write_bytes(encoded)
        if in_district:
            (district_data / filename).write_bytes(encoded)
        manifest_rows.append(
            {
                "name": filename,
                "source_relative": filename,
                "series": series,
                "year": year,
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    city_input = city_data / source_rows[0][0]
    district_input = district_data / source_rows[0][0]
    manifest = {
        "schema_version": 1,
        "source_commit": runtime._APPROVED_OLD_PABULIB_COMMIT,
        "source_dir": "synthetic-authority/source",
        "destination": "data/pb_multicity",
        "n_series": 3,
        "n_elections": 4,
        "n_files": 4,
        "files": manifest_rows,
    }
    manifest_path = city_results / "corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    for name in runtime._CITY_DEVELOPMENT_FOLDS:
        payload = {
            "schema_version": 1,
            "name": name,
            "note": f"synthetic city split {name}",
            "train_through": 2022,
            "first_test_year": 2023,
            "fit": {
                "Poland/CityOnly/Approved": [2022],
                "Poland/Train/Approved": [2022],
            },
            "warmup": {"Poland/Test/Approved": [2022]},
            "score": {"Poland/Test/Approved": [2023]},
            "counts": {
                "fit_series": 2,
                "fit_elections": 2,
                "warmup_series": 1,
                "warmup_elections": 1,
                "score_series": 1,
                "score_elections": 1,
            },
        }
        (city_results / "splits" / f"{name}.json").write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )
    district_rows = {
        "Poland/Test/Approved": [2022, 2023],
        "Poland/Train/Approved": [2022],
    }
    district_keys = sorted(district_rows)
    for fold_index, name in enumerate(runtime._DISTRICT_DEVELOPMENT_FOLDS):
        test_keys = {
            key
            for index, key in enumerate(district_keys)
            if index % 5 == fold_index
        }
        payload = {
            "name": name,
            "note": f"synthetic district split {name}",
            "train": [
                [key, district_rows[key]]
                for key in district_keys
                if key not in test_keys
            ],
            "test": [
                [key, district_rows[key]]
                for key in district_keys
                if key in test_keys
            ],
        }
        (district_splits / f"{name}.json").write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )
    monkeypatch.setattr(runtime, "ROOT", checkout)
    monkeypatch.setattr(runtime, "MULTICITY_RESULT_ROOT", city_results)
    monkeypatch.setattr(runtime, "MULTICITY_DATA_DIR", city_data)
    monkeypatch.setattr(runtime, "DISTRICT_SPLIT_ROOT", district_splits)
    monkeypatch.setattr(runtime, "DISTRICT_DATA_DIR", district_data)
    monkeypatch.setattr(
        runtime,
        "load_series",
        lambda *args, **kwargs: pytest.fail("preparation parsed PB data"),
    )
    monkeypatch.setattr(
        runtime,
        "_load_development_folds_unchecked",
        lambda: pytest.fail("preparation called the fold loader"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "_load_city_development_folds",
        lambda: pytest.fail("preparation called the city loader"),
    )
    monkeypatch.setattr(
        runtime,
        "_load_district_development_folds",
        lambda: pytest.fail("preparation called the district loader"),
    )
    return {
        "checkout": checkout,
        "city_data": city_data,
        "district_data": district_data,
        "city_input": city_input,
        "district_input": district_input,
        "city_manifest": manifest_path,
        "city_splits": city_results / "splits",
        "district_splits": district_splits,
    }


def _fix3_write_district_partition(
    paths: dict[str, Path],
    series_years: dict[str, list[int]],
    *,
    offset: int = 0,
) -> None:
    keys = sorted(series_years)
    for fold_index, name in enumerate(runtime._DISTRICT_DEVELOPMENT_FOLDS):
        test_keys = {
            key
            for index, key in enumerate(keys)
            if (index + offset) % 5 == fold_index
        }
        payload = {
            "name": name,
            "note": f"synthetic district split {name}",
            "train": [
                [key, list(series_years[key])]
                for key in keys
                if key not in test_keys
            ],
            "test": [
                [key, list(series_years[key])]
                for key in keys
                if key in test_keys
            ],
        }
        (paths["district_splits"] / f"{name}.json").write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )


def _fix3_add_series_to_city_authority(
    paths: dict[str, Path],
    *,
    filename: str,
    series: str,
    year: int,
    include_in_district: bool,
) -> None:
    encoded = f"approved-old-input:{filename}\n".encode()
    (paths["city_data"] / filename).write_bytes(encoded)
    if include_in_district:
        (paths["district_data"] / filename).write_bytes(encoded)

    manifest = json.loads(paths["city_manifest"].read_text(encoding="utf-8"))
    manifest["files"].append(
        {
            "name": filename,
            "source_relative": filename,
            "series": series,
            "year": year,
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    )
    manifest["files"].sort(key=lambda row: row["name"])
    manifest["n_series"] = len({row["series"] for row in manifest["files"]})
    manifest["n_elections"] = len(manifest["files"])
    manifest["n_files"] = len(manifest["files"])
    paths["city_manifest"].write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    for name in runtime._CITY_DEVELOPMENT_FOLDS:
        split_path = paths["city_splits"] / f"{name}.json"
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        payload["fit"][series] = [year]
        payload["counts"]["fit_series"] = len(payload["fit"])
        payload["counts"]["fit_elections"] = sum(
            len(years) for years in payload["fit"].values()
        )
        split_path.write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )


def _fix3_five_series_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[dict[str, Path], dict[str, list[int]]]:
    paths = _fix1_synthetic_authority(monkeypatch, tmp_path)
    series_years = {
        "Poland/Test/Approved": [2022, 2023],
        "Poland/Train/Approved": [2022],
    }
    for index in range(3):
        series = f"Poland/Train/Extra{index}"
        _fix3_add_series_to_city_authority(
            paths,
            filename=f"extra-{index}-2022.pb",
            series=series,
            year=2022,
            include_in_district=True,
        )
        series_years[series] = [2022]
    _fix3_write_district_partition(paths, series_years)
    return paths, series_years


def _fix3_three_city_two_district_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[dict[str, Path], dict[str, list[int]]]:
    paths = _fix1_synthetic_authority(monkeypatch, tmp_path)
    district = {
        "Poland/Test/Approved": [2022, 2023],
        "Poland/Train/Approved": [2022],
    }
    _fix3_write_district_partition(paths, district)
    return paths, district


def _fix3_guard_old_input_bytes(
    monkeypatch: pytest.MonkeyPatch, paths: dict[str, Path]
) -> None:
    real_sha = runtime._file_sha256

    def guarded_sha(path):
        candidate = Path(path)
        if (
            candidate == paths["city_data"]
            or paths["city_data"] in candidate.parents
            or candidate == paths["district_data"]
            or paths["district_data"] in candidate.parents
        ):
            pytest.fail("invalid metadata reached an approved old-input byte")
        return real_sha(path)

    monkeypatch.setattr(runtime, "_file_sha256", guarded_sha)


def _collect_synthetic_preparation_inputs(
    expected_city_series: int, expected_district_series: int
):
    return runtime._collect_development_preparation_inputs_for_counts(
        expected_city_series=expected_city_series,
        expected_district_series=expected_district_series,
    )


def _write_synthetic_pair(
    result_root: Path,
    fold,
    fit_record: dict[str, object],
    series_record: dict[str, object],
) -> None:
    root = result_root / "folds" / fold.name
    root.mkdir(parents=True, exist_ok=True)
    (root / "fit.json").write_bytes(runtime._canonical_json_bytes(fit_record))
    (root / "per_series.json").write_bytes(
        runtime._canonical_json_bytes(series_record)
    )


def _advance_synthetic_log_to_running(result_root: Path, folds) -> None:
    config = json.loads((result_root / "config.json").read_text(encoding="utf-8"))
    run_log = json.loads((result_root / "run_log.json").read_text(encoding="utf-8"))
    running = runtime._advance_run_log(
        run_log,
        "running",
        fold_views=[runtime._loaded_fold_view(fold) for fold in folds],
    )
    runtime._authenticate_run_log(running, config, finalized=False)
    (result_root / "run_log.json").write_bytes(
        runtime._canonical_json_bytes(running)
    )


def _refresh_manifest_run_log_hash(result_root: Path) -> None:
    run_log_path = result_root / "run_log.json"
    logical = "results/iclr_residual_upgrade/development/run_log.json"
    replacement = hashlib.sha256(run_log_path.read_bytes()).hexdigest()
    manifest_path = result_root / "manifest.sha256"
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    replaced = False
    for index, line in enumerate(lines):
        digest, retained = line.split("  ", 1)
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        if retained == logical:
            lines[index] = f"{replacement}  {logical}"
            replaced = True
    assert replaced
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _install_verify_only_fail_fast_sentinels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime,
        "_collect_development_preparation_inputs",
        lambda: pytest.fail("sealed mutation recollected old inputs"),
    )
    monkeypatch.setattr(
        runtime,
        "_load_development_folds_unchecked",
        lambda: pytest.fail("sealed mutation called a fold loader"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "_load_city_development_folds",
        lambda: pytest.fail("sealed mutation called the city loader"),
    )
    monkeypatch.setattr(
        runtime,
        "_load_district_development_folds",
        lambda: pytest.fail("sealed mutation called the district loader"),
    )
    monkeypatch.setattr(
        runtime,
        "load_series",
        lambda *args, **kwargs: pytest.fail("sealed mutation parsed PB data"),
    )
    monkeypatch.setattr(
        runtime,
        "fit_and_evaluate_development_fold",
        lambda *args, **kwargs: pytest.fail("sealed mutation called a fitter"),
    )
    monkeypatch.setattr(
        runtime,
        "_atomic_write",
        lambda *args, **kwargs: pytest.fail("sealed mutation called a writer"),
    )


def _rewrite_json(path: Path, mutation, *, sort_keys: bool = True) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(
        json.dumps(payload, sort_keys=sort_keys, separators=(",", ":")),
        encoding="utf-8",
    )


def test_task6_cli_has_exact_production_commands_and_removes_legacy_alias() -> None:
    prepare = runtime.parse_args(["prepare-development", "--workers", "8"])
    execute = runtime.parse_args(["development", "--workers", "8"])
    aggregate = runtime.parse_args(["aggregate-development", "--verify-only"])

    assert (prepare.command, prepare.workers) == ("prepare-development", 8)
    assert (execute.command, execute.workers) == ("development", 8)
    assert (aggregate.command, aggregate.verify_only) == (
        "aggregate-development",
        True,
    )
    for argv in (
        ["run-development", "--workers", "8"],
        ["prepare-development"],
        ["development"],
        ["aggregate-development"],
        ["aggregate-development", "--verify-only", "--workers", "8"],
    ):
        with pytest.raises(SystemExit):
            runtime.parse_args(argv)


@pytest.mark.parametrize("workers", (1, 7, 9, True, 8.0, "8"))
def test_production_python_entry_rejects_non_exact_worker_values_before_io(
    workers, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root = tmp_path / "result"
    analysis_root = tmp_path / "analysis"
    monkeypatch.setattr(runtime, "DEVELOPMENT_ROOT", result_root)
    monkeypatch.setattr(
        runtime, "DEVELOPMENT_ANALYSIS_ROOT", analysis_root, raising=False
    )
    monkeypatch.setattr(
        runtime,
        "_measure_production_runtime",
        lambda value: pytest.fail("invalid workers reached hardware measurement"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "_collect_development_preparation_inputs",
        lambda: pytest.fail("invalid workers reached development data"),
        raising=False,
    )

    with pytest.raises((TypeError, ValueError), match="workers|eight|8"):
        runtime._prepare_development(workers)

    assert not result_root.exists()
    assert not analysis_root.exists()


def test_current_or_unapproved_hardware_rejects_before_roots_and_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root = tmp_path / "result"
    analysis_root = tmp_path / "analysis"
    monkeypatch.setattr(runtime, "DEVELOPMENT_ROOT", result_root)
    monkeypatch.setattr(
        runtime, "DEVELOPMENT_ANALYSIS_ROOT", analysis_root, raising=False
    )
    monkeypatch.setattr(
        runtime,
        "_measure_production_runtime",
        lambda workers: {
            **APPROVED_RUNTIME,
            "workers": workers,
            "model_name": "MacBook Pro",
        },
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "_collect_development_preparation_inputs",
        lambda: pytest.fail("unapproved host reached development loader"),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="hardware|runtime|Mac mini"):
        runtime._prepare_development(8)

    assert not result_root.exists()
    assert not analysis_root.exists()


def test_profiler_parser_discards_every_device_identifier() -> None:
    hardware = {
        "machine_name": "Mac mini",
        "chip_type": "Apple M5",
        "physical_memory": "28 GB",
        **FORBIDDEN_IDENTIFIERS,
    }
    payload = {"SPHardwareDataType": [hardware]}

    parsed = runtime._parse_system_profiler_payload(payload)
    encoded = runtime._canonical_json_bytes(parsed).decode("utf-8")

    assert parsed == {
        "model_name": "Mac mini",
        "chip_name": "Apple M5",
        "physical_memory_display": "28 GB",
    }
    for key, value in FORBIDDEN_IDENTIFIERS.items():
        assert key not in parsed
        assert key not in encoded
        assert value not in encoded


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"SPHardwareDataType": []},
        {"SPHardwareDataType": "not-a-list"},
        {"SPHardwareDataType": [{}]},
        {"SPHardwareDataType": [{"machine_name": "Mac mini", "chip_type": 5}]},
    ),
)
def test_profiler_parser_rejects_malformed_or_incomplete_data(payload) -> None:
    with pytest.raises((TypeError, ValueError, RuntimeError), match="profiler|hardware"):
        runtime._parse_system_profiler_payload(payload)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("model_name", "MacBook Pro"),
        ("chip_name", "Apple M5 Pro"),
        ("physical_memory_bytes", 30064771071),
        ("os_name", "Linux"),
        ("architecture", "x86_64"),
        ("execution_device", "gpu"),
    ),
)
def test_hard_gate_rejects_each_wrong_measured_dimension(
    field: str, replacement: object
) -> None:
    profile = {**APPROVED_RUNTIME, field: replacement}

    with pytest.raises(RuntimeError, match="hardware|runtime|approved"):
        runtime._validate_production_runtime(profile, 8)


def test_failed_measurement_propagates_without_creating_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root = tmp_path / "result"
    monkeypatch.setattr(runtime, "DEVELOPMENT_ROOT", result_root)
    monkeypatch.setattr(
        runtime, "DEVELOPMENT_ANALYSIS_ROOT", tmp_path / "analysis", raising=False
    )
    monkeypatch.setattr(
        runtime,
        "_measure_production_runtime",
        lambda workers: (_ for _ in ()).throw(RuntimeError("system profiler failed")),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="profiler"):
        runtime._prepare_development(8)

    assert not result_root.exists()


def test_prepare_writes_only_authenticated_config_and_run_log_header(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, analysis_root, folds, _, _ = _prepare(monkeypatch, tmp_path)

    assert sorted(path.name for path in result_root.iterdir()) == [
        "config.json",
        "run_log.json",
    ]
    assert not analysis_root.exists()
    config = json.loads((result_root / "config.json").read_text(encoding="utf-8"))
    run_log = json.loads((result_root / "run_log.json").read_text(encoding="utf-8"))
    assert config["runtime"] == APPROVED_RUNTIME
    assert config["fold_inventory"] == [
        {"name": fold.name, "kind": fold.kind} for fold in folds
    ]
    assert run_log["state"] == "prepared"
    assert run_log["config_sha256"] == config["payload_sha256"]
    assert run_log["completed_folds"] == []
    assert config["core_sources"]["src/iclr_residual_protocol.py"] == {
        "state": "not_created",
        "sha256": None,
    }
    assert config["core_sources"]["src/iclr_residual_evaluate.py"] == {
        "state": "not_created",
        "sha256": None,
    }
    assert list(config["residual_tests"]) == sorted(config["residual_tests"])
    assert "tests/test_iclr_residual_runtime.py" in config["residual_tests"]
    for path in (result_root / "config.json", result_root / "run_log.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        unsigned = dict(payload)
        embedded = unsigned.pop("payload_sha256")
        assert embedded == runtime._canonical_sha256(unsigned)
        assert path.read_bytes() == runtime._canonical_json_bytes(payload)


def test_privacy_identifiers_never_reach_any_sealed_output_or_console(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parsed = runtime._parse_system_profiler_payload(
        {
            "SPHardwareDataType": [
                {
                    "machine_name": "Mac mini",
                    "chip_type": "Apple M5",
                    "physical_memory": "28 GB",
                    **FORBIDDEN_IDENTIFIERS,
                }
            ]
        }
    )
    installed = _install_synthetic_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runtime,
        "_measure_production_runtime",
        lambda workers: {
            **APPROVED_RUNTIME,
            **parsed,
            "physical_memory_display": "28 GiB",
            "workers": workers,
        },
        raising=False,
    )
    result_root, analysis_root, folds, fit_records, series_records = installed
    runtime._prepare_development(8)
    _install_fake_fitter(monkeypatch, folds, fit_records, series_records)
    runtime._run_guarded_development(8)
    captured = capsys.readouterr()
    material = captured.out + captured.err
    material += "".join(
        path.read_text(encoding="utf-8")
        for root in (result_root, analysis_root)
        for path in root.rglob("*")
        if path.is_file()
    )

    for key, value in FORBIDDEN_IDENTIFIERS.items():
        assert key not in material
        assert value not in material


def test_prepare_is_idempotent_only_for_exact_authenticated_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, _, _, _, _ = _prepare(monkeypatch, tmp_path)
    before = _tree_snapshot(result_root)

    runtime._prepare_development(8)

    assert _tree_snapshot(result_root) == before


@pytest.mark.parametrize("mode", ("partial", "divergent", "extra", "symlink"))
def test_prepare_rejects_partial_divergent_or_noncanonical_existing_state(
    mode: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, _, _, _, _ = _prepare(monkeypatch, tmp_path)
    if mode == "partial":
        (result_root / "run_log.json").unlink()
    elif mode == "divergent":
        (result_root / "config.json").write_bytes(b"{}\n")
    elif mode == "extra":
        (result_root / "unexpected.json").write_text("{}\n", encoding="utf-8")
    else:
        (result_root / "run_log.json").unlink()
        (result_root / "run_log.json").symlink_to(result_root / "config.json")
    before = _tree_snapshot(result_root)

    with pytest.raises(RuntimeError, match="preparation|divergent|unexpected|symlink"):
        runtime._prepare_development(8)

    assert _tree_snapshot(result_root) == before


def test_execution_requires_preparation_before_development_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root = tmp_path / "result"
    monkeypatch.setattr(runtime, "DEVELOPMENT_ROOT", result_root)
    monkeypatch.setattr(
        runtime, "DEVELOPMENT_ANALYSIS_ROOT", tmp_path / "analysis", raising=False
    )
    monkeypatch.setattr(
        runtime,
        "_measure_production_runtime",
        lambda workers: {**APPROVED_RUNTIME, "workers": workers},
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "_collect_development_preparation_inputs",
        lambda: pytest.fail("missing preparation reached development loader"),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="preparation|config"):
        runtime._run_guarded_development(8)

    assert not result_root.exists()


@pytest.mark.parametrize(
    ("field", "replacement"),
    (("python_version", "3.12.12"), ("numpy_version", "9.9.9"), ("os_build", "25A2")),
)
def test_execution_rejects_prepared_runtime_identity_drift(
    field: str, replacement: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, analysis_root, _, _, _ = _prepare(monkeypatch, tmp_path)
    before = _tree_snapshot(result_root, analysis_root)
    monkeypatch.setattr(
        runtime,
        "_measure_production_runtime",
        lambda workers: {**APPROVED_RUNTIME, field: replacement, "workers": workers},
    )

    with pytest.raises(RuntimeError, match="runtime|preparation|identity"):
        runtime._run_guarded_development(8)

    assert _tree_snapshot(result_root, analysis_root) == before


@pytest.mark.parametrize(
    "mutation",
    (
        "source",
        "residual_test",
        "corpus",
        "split",
        "optimizer",
        "inference",
    ),
)
def test_execution_rejects_stale_preparation_inputs_and_configuration(
    mutation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, analysis_root, _, _, _ = _prepare(monkeypatch, tmp_path)
    config_path = result_root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if mutation == "source":
        config["core_sources"]["src/iclr_residual_train.py"]["sha256"] = "0" * 64
    elif mutation == "residual_test":
        first = next(iter(config["residual_tests"]))
        config["residual_tests"][first] = "0" * 64
    elif mutation == "corpus":
        config["inputs"]["city_corpus_manifest_sha256"] = "0" * 64
    elif mutation == "split":
        first = next(iter(config["inputs"]["city_split_sha256"]))
        config["inputs"]["city_split_sha256"][first] = "0" * 64
    elif mutation == "optimizer":
        config["optimizer"]["residual"]["tau"] = 0.03
    else:
        config["inference"]["bootstrap_draws"] = 19_999
    _rehash(config)
    config_path.write_bytes(runtime._canonical_json_bytes(config))
    before = _tree_snapshot(result_root, analysis_root)

    with pytest.raises(RuntimeError, match="preparation|config|source|input"):
        runtime._run_guarded_development(8)

    assert _tree_snapshot(result_root, analysis_root) == before


@pytest.mark.parametrize("mutation", ("order", "name", "family", "count", "provenance"))
def test_execution_rejects_prepared_fold_inventory_or_provenance_drift(
    mutation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, analysis_root, folds, _, _ = _prepare(monkeypatch, tmp_path)
    _, provenance, _, _ = _synthetic_inputs()
    changed = list(folds)
    if mutation == "order":
        changed[0], changed[1] = changed[1], changed[0]
    elif mutation == "name":
        changed[0] = _replace_frozen_fold_for_runtime_test(
            changed[0], name="wrong_name"
        )
    elif mutation == "family":
        changed[0] = _replace_frozen_fold_for_runtime_test(
            changed[0], kind="district"
        )
    elif mutation == "count":
        changed.pop()
    else:
        changed[0] = _replace_frozen_fold_for_runtime_test(
            changed[0], provenance={**dict(changed[0].provenance), "split_sha256": "0" * 64}
        )
        provenance["fold_input_sha256"][changed[0].name] = runtime._fold_input_sha256(
            changed[0]
        )
    monkeypatch.setattr(
        runtime,
        "_collect_development_preparation_inputs",
        lambda: (tuple(changed), copy.deepcopy(provenance)),
    )
    before = _tree_snapshot(result_root, analysis_root)

    with pytest.raises(RuntimeError, match="preparation|fold|input|config"):
        runtime._run_guarded_development(8)

    assert _tree_snapshot(result_root, analysis_root) == before


def test_execution_resumes_complete_pair_and_rejects_one_file_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result_root, _, folds, fit_records, series_records = _prepare(monkeypatch, tmp_path)
    _advance_synthetic_log_to_running(result_root, folds)
    first_root = result_root / "folds" / folds[0].name
    first_root.mkdir(parents=True)
    (first_root / "fit.json").write_bytes(runtime._canonical_json_bytes(fit_records[0]))
    (first_root / "per_series.json").write_bytes(
        runtime._canonical_json_bytes(series_records[0])
    )
    calls = _install_fake_fitter(monkeypatch, folds, fit_records, series_records)

    runtime._run_guarded_development(8)

    assert folds[0].name not in calls
    assert set(calls) == {fold.name for fold in folds[1:]}
    assert "status=resumed" in capsys.readouterr().out

    partial_root = tmp_path / "partial"
    installed = _prepare(monkeypatch, partial_root)
    partial_result, partial_analysis, partial_folds, partial_fit, partial_series = installed
    target = partial_result / "folds" / partial_folds[0].name
    target.mkdir(parents=True)
    (target / "fit.json").write_bytes(runtime._canonical_json_bytes(partial_fit[0]))
    _install_fake_fitter(monkeypatch, partial_folds, partial_fit, partial_series)
    before = _tree_snapshot(partial_result, partial_analysis)
    with pytest.raises(RuntimeError, match="partial|pair"):
        runtime._run_guarded_development(8)
    assert _tree_snapshot(partial_result, partial_analysis) == before


def test_interrupted_execution_resumes_without_touching_completed_pairs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, _, folds, fit_records, series_records = _prepare(monkeypatch, tmp_path)
    _install_fake_fitter(
        monkeypatch, folds, fit_records, series_records, fail_after=3
    )
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        runtime._run_guarded_development(8)
    completed = {
        fold.name: _tree_snapshot(result_root / "folds" / fold.name)
        for fold in folds[:3]
    }
    calls = _install_fake_fitter(monkeypatch, folds, fit_records, series_records)

    runtime._run_guarded_development(8)

    assert calls == [fold.name for fold in folds[3:]]
    assert all(
        _tree_snapshot(result_root / "folds" / name) == snapshot
        for name, snapshot in completed.items()
    )


def test_sealed_execution_has_monotonic_log_exact_manifest_and_no_later_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, analysis_root, folds, _, _, calls, decision = _seal(
        monkeypatch, tmp_path
    )
    run_log = json.loads((result_root / "run_log.json").read_text(encoding="utf-8"))
    lines = (result_root / "manifest.sha256").read_text(encoding="utf-8").splitlines()
    before = _tree_snapshot(result_root, analysis_root)

    repeated = runtime._run_guarded_development(8)

    assert calls == [fold.name for fold in folds]
    assert decision == repeated
    assert run_log["state"] == "finalized"
    assert run_log["completed_folds"] == [fold.name for fold in folds]
    assert [event["elapsed_seconds"] for event in run_log["events"]] == sorted(
        event["elapsed_seconds"] for event in run_log["events"]
    )
    assert len(lines) == 23
    assert lines == sorted(lines, key=lambda line: line.split("  ", 1)[1])
    assert _tree_snapshot(result_root, analysis_root) == before


def test_aggregation_is_deterministic_canonical_and_reconstructs_six_conditions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, analysis_root, _, _, _, _, decision = _seal(monkeypatch, tmp_path)
    summary = json.loads((result_root / "summary.json").read_text(encoding="utf-8"))
    frozen_decision = json.loads(
        (result_root / "evidence_decision.json").read_text(encoding="utf-8")
    )
    per_series = (result_root / "per_series.csv").read_bytes()
    actuation = (analysis_root / "actuation.csv").read_bytes()
    report = (analysis_root / "report.md").read_bytes()

    assert frozen_decision == decision
    assert set(frozen_decision["conditions"]) == {
        "primary_improves_anchor",
        "all_held_cities_negative",
        "district_fold_majority",
        "dual_reference_safety",
        "all_seeds_negative",
        "actuation_coverage",
    }
    assert summary["decision"] == frozen_decision
    assert summary["counts"]["folds"] == 8
    assert summary["counts"]["manifest_entries"] == 23
    assert per_series.endswith(b"\n") and b"\r" not in per_series
    assert actuation.endswith(b"\n") and b"\r" not in actuation
    assert report.endswith(b"\n")
    assert result_root.joinpath("summary.json").read_bytes() == runtime._canonical_json_bytes(
        summary
    )


def test_csv_encoder_quotes_identifiers_and_rejects_signed_zero_or_nonfinite() -> None:
    encoded = runtime._encode_runtime_csv(
        ("series", "value"), (("Poland/A, district", 1.25),)
    )

    assert encoded == b'series,value\n"Poland/A, district",1.25\n'
    for value in (-0.0, math.inf, -math.inf, math.nan, True):
        with pytest.raises((TypeError, ValueError), match="finite|zero|numeric"):
            runtime._format_runtime_number(value)


def test_aggregate_builder_ignores_mapping_insertion_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, _, folds, fit_records, series_records = _install_synthetic_runtime(
        monkeypatch, tmp_path
    )
    shuffled_fit = tuple(dict(reversed(tuple(row.items()))) for row in fit_records)
    shuffled_series = tuple(
        dict(reversed(tuple(row.items()))) for row in series_records
    )
    run_log = {
        "state": "finalized",
        "elapsed_seconds": 8.0,
        "peak_memory_bytes": 1048576,
    }

    first = runtime._build_development_aggregate_bytes(
        folds, fit_records, series_records, run_log
    )
    second = runtime._build_development_aggregate_bytes(
        folds, shuffled_fit, shuffled_series, dict(reversed(tuple(run_log.items())))
    )

    assert first == second


def test_manifest_rejects_all_inventory_path_and_byte_attacks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, analysis_root, *_ = _seal(monkeypatch, tmp_path / "base")
    attacks = (
        "missing",
        "extra",
        "tampered",
        "noncanonical",
        "symlink",
        "escape",
        "absolute",
        "duplicate",
        "backslash",
        "directory",
    )
    for attack in attacks:
        attack_root = tmp_path / attack
        copied_result = attack_root / "result"
        copied_analysis = attack_root / "analysis"
        shutil.copytree(result_root, copied_result)
        shutil.copytree(analysis_root, copied_analysis)
        monkeypatch.setattr(runtime, "DEVELOPMENT_ROOT", copied_result)
        monkeypatch.setattr(runtime, "DEVELOPMENT_ANALYSIS_ROOT", copied_analysis)
        manifest = copied_result / "manifest.sha256"
        lines = manifest.read_text(encoding="utf-8").splitlines()
        if attack == "missing":
            (copied_result / "per_series.csv").unlink()
        elif attack == "extra":
            (copied_result / "extra.txt").write_text("extra", encoding="utf-8")
        elif attack == "tampered":
            (copied_result / "summary.json").write_bytes(b"{}\n")
        elif attack == "noncanonical":
            manifest.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")
        elif attack == "symlink":
            target = copied_result / "per_series.csv"
            target.unlink()
            target.symlink_to(copied_result / "summary.json")
        elif attack == "escape":
            lines[0] = lines[0].split("  ")[0] + "  ../escape.json"
            manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif attack == "absolute":
            lines[0] = lines[0].split("  ")[0] + "  /tmp/escape.json"
            manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        elif attack == "duplicate":
            manifest.write_text("\n".join([*lines, lines[0]]) + "\n", encoding="utf-8")
        elif attack == "backslash":
            lines[0] = lines[0].replace("/", "\\")
            manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            (copied_result / "unexpected-directory").mkdir()

        with pytest.raises(RuntimeError, match="manifest|artifact|inventory|symlink"):
            runtime._authenticate_development_manifest()


def test_verify_only_is_read_only_and_prints_stable_summary_and_decision_hashes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result_root, analysis_root, *_ = _seal(monkeypatch, tmp_path)
    capsys.readouterr()
    before = _tree_snapshot(result_root, analysis_root)

    assert runtime.main(["aggregate-development", "--verify-only"]) == 0
    first = capsys.readouterr().out
    assert runtime.main(["aggregate-development", "--verify-only"]) == 0
    second = capsys.readouterr().out

    assert first == second
    assert first.splitlines()[0].startswith("SUMMARY_SHA256 ")
    assert first.splitlines()[1].startswith("DECISION_SHA256 ")
    assert _tree_snapshot(result_root, analysis_root) == before


def test_synthetic_runtime_never_calls_fresh_or_bydgoszcz_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    installed = _install_synthetic_runtime(monkeypatch, tmp_path)
    result_root, analysis_root, folds, fit_records, series_records = installed
    monkeypatch.setattr(
        runtime,
        "load_development_folds",
        lambda: pytest.fail("injected preparation must not call a real loader"),
    )
    runtime._prepare_development(8)
    _install_fake_fitter(monkeypatch, folds, fit_records, series_records)

    runtime._run_guarded_development(8)

    material = b"".join(
        path.read_bytes()
        for root in (result_root, analysis_root)
        for path in root.rglob("*")
        if path.is_file()
    ).lower()
    assert b"bydgoszcz" not in material
    assert b"fresh" not in material


@pytest.mark.parametrize("entry", ("load", "fit", "run"))
def test_every_public_production_entry_rejects_unapproved_hardware_before_io(
    entry: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root = tmp_path / "result"
    monkeypatch.setattr(runtime, "DEVELOPMENT_ROOT", result_root)
    monkeypatch.setattr(
        runtime, "DEVELOPMENT_ANALYSIS_ROOT", tmp_path / "analysis", raising=False
    )
    monkeypatch.setattr(
        runtime,
        "_measure_production_runtime",
        lambda workers: {
            **APPROVED_RUNTIME,
            "workers": workers,
            "model_name": "unapproved-controller",
        },
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "_load_development_folds_unchecked",
        lambda: pytest.fail("public entry reached loader"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "_fit_and_evaluate_development_fold",
        lambda *args, **kwargs: pytest.fail("public entry reached fitter"),
    )
    monkeypatch.setattr(
        runtime,
        "_load_city_development_folds",
        lambda: pytest.fail("public entry reached city loader"),
    )
    monkeypatch.setattr(
        runtime,
        "_load_district_development_folds",
        lambda: pytest.fail("public entry reached district loader"),
    )
    monkeypatch.setattr(
        runtime,
        "_atomic_write",
        lambda *args, **kwargs: pytest.fail("public entry reached a production write"),
        raising=False,
    )
    fold = task5_fixtures._canonical_development_gate_folds()[0]

    with pytest.raises(RuntimeError, match="hardware|runtime|approved"):
        if entry == "load":
            runtime.load_development_folds()
        elif entry == "fit":
            runtime.fit_and_evaluate_development_fold(fold, 8)
        else:
            runtime.run_development(8)

    assert not result_root.exists()


@pytest.mark.parametrize("workers", (True, 8.0, 1, 7, 9))
def test_public_fit_and_run_reject_worker_aliases_before_measurement(
    workers, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fold = task5_fixtures._canonical_development_gate_folds()[0]
    result_root = tmp_path / "result"
    analysis_root = tmp_path / "analysis"
    monkeypatch.setattr(runtime, "DEVELOPMENT_ROOT", result_root)
    monkeypatch.setattr(
        runtime, "DEVELOPMENT_ANALYSIS_ROOT", analysis_root, raising=False
    )
    monkeypatch.setattr(
        runtime,
        "_measure_production_runtime",
        lambda value: pytest.fail("invalid public workers reached measurement"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "load_development_folds",
        lambda: pytest.fail("invalid public workers reached the fold loader"),
    )
    monkeypatch.setattr(
        runtime,
        "_fit_and_evaluate_development_fold",
        lambda *args, **kwargs: pytest.fail("invalid public workers reached fitter"),
    )
    monkeypatch.setattr(
        runtime,
        "_load_city_development_folds",
        lambda: pytest.fail("invalid public workers reached city loader"),
    )
    monkeypatch.setattr(
        runtime,
        "_load_district_development_folds",
        lambda: pytest.fail("invalid public workers reached district loader"),
    )
    monkeypatch.setattr(
        runtime,
        "_atomic_write",
        lambda *args, **kwargs: pytest.fail("invalid public workers reached write"),
        raising=False,
    )
    for call in (
        lambda: runtime.fit_and_evaluate_development_fold(fold, workers),
        lambda: runtime.run_development(workers),
    ):
        with pytest.raises((TypeError, ValueError), match="workers|eight|8"):
            call()
    assert not result_root.exists()
    assert not analysis_root.exists()


def test_public_entries_require_preparation_before_loader_fit_or_directory_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root = tmp_path / "missing-preparation"
    monkeypatch.setattr(runtime, "DEVELOPMENT_ROOT", result_root)
    monkeypatch.setattr(
        runtime, "DEVELOPMENT_ANALYSIS_ROOT", tmp_path / "analysis", raising=False
    )
    monkeypatch.setattr(
        runtime,
        "_measure_production_runtime",
        lambda workers: {**APPROVED_RUNTIME, "workers": workers},
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "_load_development_folds_unchecked",
        lambda: pytest.fail("missing preparation reached loader"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "_load_city_development_folds",
        lambda: pytest.fail("missing preparation reached city loader"),
    )
    monkeypatch.setattr(
        runtime,
        "_load_district_development_folds",
        lambda: pytest.fail("missing preparation reached district loader"),
    )
    monkeypatch.setattr(
        runtime,
        "_fit_and_evaluate_development_fold",
        lambda *args, **kwargs: pytest.fail("missing preparation reached fitter"),
    )
    monkeypatch.setattr(
        runtime,
        "_atomic_write",
        lambda *args, **kwargs: pytest.fail("missing preparation reached write"),
        raising=False,
    )
    fold = task5_fixtures._canonical_development_gate_folds()[0]
    for call in (
        runtime.load_development_folds,
        lambda: runtime.fit_and_evaluate_development_fold(fold, 8),
        lambda: runtime.run_development(8),
    ):
        with pytest.raises(RuntimeError, match="preparation|config"):
            call()
    assert not result_root.exists()


def test_legacy_red_route_is_fail_fast_and_cannot_read_or_create_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root = tmp_path / "result"
    analysis_root = tmp_path / "analysis"
    monkeypatch.setattr(runtime, "DEVELOPMENT_ROOT", result_root)
    monkeypatch.setattr(
        runtime, "DEVELOPMENT_ANALYSIS_ROOT", analysis_root, raising=False
    )
    monkeypatch.setattr(
        runtime,
        "_measure_production_runtime",
        lambda workers: {
            **APPROVED_RUNTIME,
            "workers": workers,
            "model_name": "unapproved-controller",
        },
        raising=False,
    )
    reached = []

    def forbidden(label):
        def fail(*args, **kwargs):
            del args, kwargs
            reached.append(label)
            raise RuntimeError(f"legacy-route sentinel: {label}")

        return fail

    monkeypatch.setattr(runtime, "load_development_folds", forbidden("public loader"))
    monkeypatch.setattr(
        runtime, "fit_and_evaluate_development_fold", forbidden("public fitter")
    )
    monkeypatch.setattr(
        runtime, "_load_city_development_folds", forbidden("city loader")
    )
    monkeypatch.setattr(
        runtime, "_load_district_development_folds", forbidden("district loader")
    )
    monkeypatch.setattr(
        runtime, "_fit_and_evaluate_development_fold", forbidden("private fitter")
    )
    monkeypatch.setattr(
        runtime, "_atomic_write", forbidden("production write"), raising=False
    )

    with pytest.raises(RuntimeError, match="hardware|runtime|approved|legacy-route"):
        runtime.run_development(8)

    assert reached in ([], ["public loader"])
    assert not result_root.exists()
    assert not analysis_root.exists()


def test_public_fit_rejects_supplied_fold_different_from_prepared_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, _, folds, _, _ = _prepare(monkeypatch, tmp_path)
    supplied = _replace_frozen_fold_for_runtime_test(
        folds[0], provenance={**dict(folds[0].provenance), "extra": "different"}
    )
    monkeypatch.setattr(
        runtime,
        "_fit_and_evaluate_development_fold",
        lambda *args, **kwargs: pytest.fail("mismatched fold reached fitter"),
    )

    with pytest.raises(RuntimeError, match="prepared|fold|identity"):
        runtime.fit_and_evaluate_development_fold(supplied, 8)


def _make_synthetic_checkout(path: Path) -> Path:
    source_names = (
        "cohorts.py",
        "iclr_history_free_audit.py",
        "iclr_multicity_protocol.py",
        "iclr_residual_actuation.py",
        "iclr_residual_policy.py",
        "iclr_residual_train.py",
        "parse_pb.py",
        "unrelated_runtime.py",
    )
    for name in source_names:
        target = path / "src" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# synthetic {name}\n", encoding="utf-8")
    for name in (
        "test_iclr_residual_actuation.py",
        "test_iclr_residual_policy.py",
        "test_iclr_residual_runtime.py",
        "test_iclr_residual_train.py",
    ):
        target = path / "tests" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# synthetic {name}\n", encoding="utf-8")
    (path / "pyproject.toml").write_text("[project]\nname='synthetic'\n", encoding="utf-8")
    (path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    return path


def test_runtime_source_inventory_is_complete_sorted_and_honest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkout = _make_synthetic_checkout(tmp_path / "checkout")
    monkeypatch.setattr(runtime, "ROOT", checkout)

    inventory = runtime._collect_runtime_source_inventory()

    assert list(inventory) == sorted(inventory)
    assert set(inventory) == {
        *(f"src/{path.name}" for path in (checkout / "src").glob("*.py")),
        "pyproject.toml",
        "uv.lock",
    }
    assert all(row["state"] == "present" for row in inventory.values())
    assert all(
        len(row["sha256"]) == 64 for row in inventory.values()
    )


@pytest.mark.parametrize(
    "relative",
    (
        "src/iclr_history_free_audit.py",
        "src/cohorts.py",
        "src/parse_pb.py",
        "src/iclr_multicity_protocol.py",
        "pyproject.toml",
        "uv.lock",
    ),
)
def test_runtime_dependency_mutation_stops_before_loader_fit_or_write(
    relative: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkout = _make_synthetic_checkout(tmp_path / "checkout")
    monkeypatch.setattr(runtime, "ROOT", checkout)
    result_root, analysis_root, folds, _, _ = _prepare(
        monkeypatch, tmp_path / "runtime"
    )
    (checkout / relative).write_text("# mutated after preparation\n", encoding="utf-8")
    monkeypatch.setattr(
        runtime,
        "_load_development_folds_unchecked",
        lambda: pytest.fail("dependency drift reached loader"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "fit_and_evaluate_development_fold",
        lambda *args, **kwargs: pytest.fail("dependency drift reached fit"),
    )
    before = _tree_snapshot(result_root, analysis_root)

    with pytest.raises(RuntimeError, match="source|runtime|preparation|inventory"):
        runtime._run_guarded_development(8)

    assert len(folds) == 8
    assert _tree_snapshot(result_root, analysis_root) == before


def test_old_commit_comes_from_bound_multicity_protocol_constant(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    import iclr_multicity_protocol

    monkeypatch.setattr(iclr_multicity_protocol, "PABULIB_COMMIT", "0" * 40)
    with pytest.raises(RuntimeError, match="commit|Pabulib|approved"):
        runtime._approved_old_pabulib_commit()


def test_run_log_uses_exact_state_machine_order_and_authenticated_pair_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, _, folds, *_ = _seal(monkeypatch, tmp_path)
    log = json.loads((result_root / "run_log.json").read_text(encoding="utf-8"))
    states = [event["state"] for event in log["events"]]
    fold_events = [event for event in log["events"] if "fold" in event]

    assert states == [
        "prepared",
        "running",
        *("running" for _ in folds),
        "folds_complete",
        "aggregated",
        "finalized",
    ]
    assert [event["fold"] for event in fold_events] == [fold.name for fold in folds]
    assert all(
        set(event["artifact_pair_sha256"]) == {"fit.json", "per_series.json"}
        and all(len(value) == 64 for value in event["artifact_pair_sha256"].values())
        for event in fold_events
    )


def test_runtime_summary_projection_contains_only_approved_operational_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, _, *_ = _seal(monkeypatch, tmp_path)
    log = json.loads((result_root / "run_log.json").read_text(encoding="utf-8"))

    projection = runtime._runtime_summary_projection(log)

    assert set(projection) == {
        "start_utc",
        "end_utc",
        "elapsed_seconds",
        "parent_max_rss_bytes",
        "completed_children_max_rss_bytes",
        "reported_peak_max_rss_bytes",
        "peak_memory_interpretation",
        "workers",
        "config_sha256",
    }
    report = (runtime.DEVELOPMENT_ANALYSIS_ROOT / "report.md").read_text(
        encoding="utf-8"
    )
    assert str(projection["reported_peak_max_rss_bytes"]) in report
    assert projection["peak_memory_interpretation"] in report


@pytest.mark.parametrize(
    "value",
    (
        "/Users/alice/private.pb",
        "../escape.pb",
        "data\\pool.pb",
        "data/pool\x00.pb",
        "unapproved/tree/pool.pb",
    ),
)
def test_persisted_path_canonicalizer_rejects_private_or_escaping_paths(
    value: str,
) -> None:
    with pytest.raises((TypeError, ValueError, RuntimeError), match="path|POSIX|root"):
        runtime._canonical_persisted_path(
            value, approved_prefixes=("data/pb/", "results/iclr_residual_upgrade/development/")
        )


def test_persisted_path_canonicalizer_accepts_only_relative_posix_or_synthetic_uri() -> None:
    assert runtime._canonical_persisted_path(
        "data/pb/pool.pb", approved_prefixes=("data/pb/",)
    ) == "data/pb/pool.pb"
    assert runtime._canonical_persisted_path(
        "synthetic://Poland/Test/2022", approved_prefixes=("data/pb/",)
    ) == "synthetic://Poland/Test/2022"


def test_sentinel_checkout_user_and_host_never_enter_persisted_or_printed_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinel = "ALICE-PRIVATE-HOST"
    checkout = _make_synthetic_checkout(tmp_path / sentinel / "checkout")
    monkeypatch.setattr(runtime, "ROOT", checkout)
    result_root, analysis_root, folds, fit_records, series_records = _prepare(
        monkeypatch, tmp_path / "outputs"
    )
    _install_fake_fitter(monkeypatch, folds, fit_records, series_records)
    runtime._run_guarded_development(8)
    captured = capsys.readouterr()
    retained = captured.out + captured.err
    retained += "".join(
        path.read_text(encoding="utf-8")
        for root in (result_root, analysis_root)
        for path in root.rglob("*")
        if path.is_file()
    )
    assert sentinel not in retained
    assert str(checkout) not in retained


def test_prepare_access_matrix_never_calls_pb_parser_or_fold_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_synthetic_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runtime,
        "_load_development_folds_unchecked",
        lambda: pytest.fail("prepare called fold loader"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime, "load_series", lambda *args, **kwargs: pytest.fail("prepare parsed PB")
    )

    runtime._prepare_development(8)


def test_verify_only_access_matrix_uses_only_sealed_task6_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _seal(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runtime,
        "_collect_development_preparation_inputs",
        lambda: pytest.fail("verify-only recollected old inputs"),
    )
    monkeypatch.setattr(
        runtime,
        "_load_development_folds_unchecked",
        lambda: pytest.fail("verify-only called fold loader"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime, "load_series", lambda *args, **kwargs: pytest.fail("verify-only parsed PB")
    )

    runtime._verify_development_outputs()


def test_hardware_probe_uses_absolute_apple_tools_and_minimal_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = tmp_path / "system_profiler"
    fake.write_text("#!/bin/sh\necho APPROVED-FAKE-SENTINEL\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{runtime.os.environ.get('PATH', '')}", prepend=False)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((tuple(command), kwargs))
        if command[0] == "/usr/sbin/system_profiler":
            return task5_fixtures.SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "SPHardwareDataType": [
                            {
                                "machine_name": "MacBook Pro",
                                "chip_type": "Apple M4",
                                "physical_memory": "16 GB",
                                "serial_number": "PRIVATE-SENTINEL",
                            }
                        ]
                    }
                ),
                stderr="PRIVATE-STDERR-SENTINEL",
            )
        if command[:2] == ["/usr/bin/sw_vers", "-productVersion"]:
            return task5_fixtures.SimpleNamespace(returncode=0, stdout="15.0\n", stderr="")
        if command[:2] == ["/usr/bin/sw_vers", "-buildVersion"]:
            return task5_fixtures.SimpleNamespace(returncode=0, stdout="24A1\n", stderr="")
        if command[0] == "/usr/sbin/sysctl":
            return task5_fixtures.SimpleNamespace(
                returncode=0, stdout=str(16 * 1024**3) + "\n", stderr=""
            )
        raise AssertionError(command)

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    monkeypatch.setattr(runtime, "_trusted_apple_tool", lambda path: Path(path))
    with pytest.raises(RuntimeError, match="hardware|runtime|approved") as error:
        runtime._validate_production_runtime(runtime._measure_production_runtime(8), 8)

    assert all(command[0].startswith("/usr/") for command, _ in calls)
    assert all(kwargs["shell"] is False for _, kwargs in calls)
    assert all(kwargs["env"] == {"LC_ALL": "C", "LANG": "C"} for _, kwargs in calls)
    retained = str(error.value) + capsys.readouterr().out + capsys.readouterr().err
    assert "APPROVED-FAKE-SENTINEL" not in retained
    assert "PRIVATE-SENTINEL" not in retained
    assert "PRIVATE-STDERR-SENTINEL" not in retained


def test_historical_verification_allows_only_task7_and_task8_additions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkout = _make_synthetic_checkout(tmp_path / "checkout")
    monkeypatch.setattr(runtime, "ROOT", checkout)
    result_root, analysis_root, folds, fit_records, series_records = _prepare(
        monkeypatch, tmp_path / "runtime"
    )
    _install_fake_fitter(monkeypatch, folds, fit_records, series_records)
    runtime._run_guarded_development(8)
    before_outputs = _tree_snapshot(result_root, analysis_root)
    for relative in (
        "src/iclr_residual_protocol.py",
        "tests/test_iclr_residual_protocol.py",
        "src/iclr_residual_evaluate.py",
        "tests/test_iclr_residual_evaluate.py",
    ):
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# approved later addition\n", encoding="utf-8")

    runtime._verify_development_outputs()

    assert _tree_snapshot(result_root, analysis_root) == before_outputs
    (checkout / "src" / "unlisted_after_task6.py").write_text(
        "# unlisted\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="source|inventory|addition"):
        runtime._verify_development_outputs()
    (checkout / "src" / "unlisted_after_task6.py").unlink()
    (checkout / "src" / "cohorts.py").write_text("# frozen mutation\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source|inventory|frozen"):
        runtime._verify_development_outputs()


def test_exact_aggregate_headers_order_primary_scan_and_float_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, analysis_root, folds, *_ = _seal(monkeypatch, tmp_path)
    per_lines = (result_root / "per_series.csv").read_text(encoding="utf-8").splitlines()
    actuation_lines = (analysis_root / "actuation.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert per_lines[0] == (
        "fold,kind,seed,series,city,cluster,residual_csd,residual_welfare,"
        "residual_exclusion,anchor_csd,anchor_welfare,anchor_exclusion,mes_csd,"
        "mes_welfare,mes_exclusion,delta_csd_vs_anchor,delta_welfare_vs_anchor,"
        "delta_exclusion_vs_anchor,residual_signature,anchor_signature,actuated"
    )
    assert actuation_lines[0] == (
        "fold,kind,seed,series,instance,is_primary_seed,actuated,first_grid_t,"
        "refined_low,refined_high,csd_direction,anchor_csd,changed_csd"
    )
    assert len(actuation_lines) - 1 == sum(len(fold.test) for fold in folds)
    assert all(",true," in line for line in actuation_lines[1:])
    assert runtime._format_runtime_number(0.0) == "0.0"


def test_peak_memory_projection_is_exact_integer_monotonic_maximum(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    readings = {
        runtime.resource.RUSAGE_SELF: task5_fixtures.SimpleNamespace(ru_maxrss=100),
        runtime.resource.RUSAGE_CHILDREN: task5_fixtures.SimpleNamespace(ru_maxrss=250),
    }
    monkeypatch.setattr(runtime.resource, "getrusage", lambda kind: readings[kind])

    first = runtime._runtime_peak_memory_sample()
    second = runtime._runtime_peak_memory_sample(
        {
            "parent_max_rss_bytes": 300,
            "completed_children_max_rss_bytes": 250,
            "reported_peak_max_rss_bytes": 300,
        }
    )

    assert first == {
        "parent_max_rss_bytes": 100,
        "completed_children_max_rss_bytes": 250,
        "reported_peak_max_rss_bytes": 250,
    }
    assert second == {
        "parent_max_rss_bytes": 300,
        "completed_children_max_rss_bytes": 250,
        "reported_peak_max_rss_bytes": 300,
    }
    for invalid in (-1, 1.5, True):
        readings[runtime.resource.RUSAGE_SELF] = task5_fixtures.SimpleNamespace(
            ru_maxrss=invalid
        )
        with pytest.raises((TypeError, ValueError), match="memory|RSS|integer"):
            runtime._runtime_peak_memory_sample()


def _protected_snapshot() -> dict[str, str]:
    paths = (
        runtime.ROOT.parent / "iclr_paper" / "tex" / "main.tex",
        runtime.ROOT.parent / "iclr_paper" / "tex" / "appendix.tex",
        runtime.ROOT.parent / "iclr_paper" / "tex" / "iclr3.pdf",
        runtime.ROOT.parent / "iclr_paper" / "tex" / "iclr3-draft.pdf",
        runtime.ROOT.parent / "iclr_paper" / "tex" / "iclr2.pdf",
    )
    snapshot = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file()
    }
    results = runtime.ROOT / "results"
    for path in results.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(results).as_posix()
        if relative.startswith("iclr_residual_upgrade/development/"):
            continue
        snapshot[f"results/{relative}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def test_all_real_host_negative_command_probes_preserve_protected_trees(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    before = _protected_snapshot()
    monkeypatch.setattr(runtime, "DEVELOPMENT_ROOT", tmp_path / "result")
    monkeypatch.setattr(
        runtime, "DEVELOPMENT_ANALYSIS_ROOT", tmp_path / "analysis", raising=False
    )
    monkeypatch.setattr(
        runtime,
        "_measure_production_runtime",
        lambda workers: {
            **APPROVED_RUNTIME,
            "workers": workers,
            "model_name": "unapproved-controller",
        },
        raising=False,
    )
    for argv in (
        ["prepare-development", "--workers", "8"],
        ["development", "--workers", "8"],
        ["aggregate-development", "--verify-only"],
    ):
        assert runtime.main(argv) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == "DEVELOPMENT_COMMAND_REJECTED\n"
        assert _protected_snapshot() == before


def test_task6_fix1_metadata_preparation_binds_loaded_semantic_fold_views(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    installed, descriptors = _install_fix1_metadata_runtime(monkeypatch, tmp_path)
    result_root, _, folds, fit_records, series_records = installed
    runtime._prepare_development(8)
    _install_fake_fitter(monkeypatch, folds, fit_records, series_records)

    decision = runtime._run_guarded_development(8)

    config = json.loads((result_root / "config.json").read_text(encoding="utf-8"))
    run_log = json.loads((result_root / "run_log.json").read_text(encoding="utf-8"))
    assert decision["classification"] in {"pass", "fail"}
    assert config["fold_bindings"] == list(descriptors)
    assert "fold_views" not in config
    assert [set(row) for row in run_log["fold_views"]] == [
        {
            "name",
            "kind",
            "fold_input_sha256",
            "training_series",
            "scored_series",
            "scored_instances",
        }
        for _ in folds
    ]
    assert (result_root / "manifest.sha256").is_file()


@pytest.mark.parametrize("state", ("folds_complete", "aggregated", "finalized"))
def test_task6_fix1_restart_recovers_each_durable_state_prefix_without_refit(
    state: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, analysis_root, folds, fit_records, series_records = _prepare(
        monkeypatch, tmp_path / "interrupted"
    )
    first_calls = _install_fake_fitter(
        monkeypatch, folds, fit_records, series_records
    )
    original = runtime._atomic_canonical_json
    armed = True

    def crash_after_state(path, payload):
        nonlocal armed
        original(path, payload)
        if armed and path.name == "run_log.json" and payload.get("state") == state:
            armed = False
            raise RuntimeError(f"synthetic crash after {state}")

    monkeypatch.setattr(runtime, "_atomic_canonical_json", crash_after_state)
    with pytest.raises(RuntimeError, match=f"after {state}"):
        runtime._run_guarded_development(8)
    monkeypatch.setattr(runtime, "_atomic_canonical_json", original)
    restart_calls = _install_fake_fitter(
        monkeypatch, folds, fit_records, series_records
    )

    runtime._run_guarded_development(8)
    recovered = _tree_snapshot(result_root, analysis_root)

    baseline_result, baseline_analysis, *_ = _seal(
        monkeypatch, tmp_path / "baseline"
    )
    assert first_calls == [fold.name for fold in folds]
    assert restart_calls == []
    assert recovered == _tree_snapshot(baseline_result, baseline_analysis)


@pytest.mark.parametrize(
    "artifact",
    (
        "per_series.csv",
        "summary.json",
        "evidence_decision.json",
        "actuation.csv",
        "report.md",
        "manifest.sha256",
    ),
)
def test_task6_fix1_restart_recovers_every_aggregate_write_prefix(
    artifact: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, analysis_root, folds, fit_records, series_records = _prepare(
        monkeypatch, tmp_path / "interrupted"
    )
    first_calls = _install_fake_fitter(
        monkeypatch, folds, fit_records, series_records
    )
    original = runtime._write_immutable_bytes
    armed = True

    def crash_after_artifact(path, encoded):
        nonlocal armed
        original(path, encoded)
        if armed and path.name == artifact:
            armed = False
            raise RuntimeError(f"synthetic crash after {artifact}")

    monkeypatch.setattr(runtime, "_write_immutable_bytes", crash_after_artifact)
    with pytest.raises(RuntimeError, match=f"after {re.escape(artifact)}"):
        runtime._run_guarded_development(8)
    monkeypatch.setattr(runtime, "_write_immutable_bytes", original)
    restart_calls = _install_fake_fitter(
        monkeypatch, folds, fit_records, series_records
    )

    runtime._run_guarded_development(8)
    recovered = _tree_snapshot(result_root, analysis_root)

    baseline_result, baseline_analysis, *_ = _seal(
        monkeypatch, tmp_path / "baseline"
    )
    assert first_calls == [fold.name for fold in folds]
    assert restart_calls == []
    assert recovered == _tree_snapshot(baseline_result, baseline_analysis)


@pytest.mark.parametrize(
    "mutation",
    ("unexpected_top_level", "impossible_event", "unknown_fold", "reordered_folds"),
)
def test_task6_fix1_run_log_rejects_forged_prefix_before_first_write(
    mutation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, analysis_root, folds, _, _ = _prepare(monkeypatch, tmp_path)
    log_path = result_root / "run_log.json"

    def mutate(payload):
        if mutation == "unexpected_top_level":
            payload["unexpected"] = "forged"
        elif mutation == "impossible_event":
            payload["events"].append(
                {
                    "event": "impossible",
                    "state": "finalized",
                    "elapsed_seconds": payload["elapsed_seconds"],
                }
            )
        elif mutation == "unknown_fold":
            payload["completed_folds"] = ["unknown_fold"]
        else:
            payload["completed_folds"] = [folds[1].name, folds[0].name]

    _write_rehashed_run_log(log_path, mutate)
    before = _tree_snapshot(result_root, analysis_root)
    monkeypatch.setattr(
        runtime,
        "_atomic_write",
        lambda *args, **kwargs: pytest.fail("forged run log reached a write"),
    )
    monkeypatch.setattr(
        runtime,
        "fit_and_evaluate_development_fold",
        lambda *args, **kwargs: pytest.fail("forged run log reached a fit"),
    )

    with pytest.raises(RuntimeError, match="run-log|event|completed|fold"):
        runtime._run_guarded_development(8)

    assert _tree_snapshot(result_root, analysis_root) == before


def test_task6_fix1_run_log_rejects_forged_pair_hash_before_next_fold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, analysis_root, folds, fit_records, series_records = _prepare(
        monkeypatch, tmp_path
    )
    _install_fake_fitter(monkeypatch, folds, fit_records, series_records)
    original = runtime._atomic_canonical_json
    armed = True

    def stop_after_first_completion(path, payload):
        nonlocal armed
        original(path, payload)
        if armed and payload.get("completed_folds") == [folds[0].name]:
            armed = False
            raise RuntimeError("synthetic stop after first fold")

    monkeypatch.setattr(runtime, "_atomic_canonical_json", stop_after_first_completion)
    with pytest.raises(RuntimeError, match="after first fold"):
        runtime._run_guarded_development(8)
    monkeypatch.setattr(runtime, "_atomic_canonical_json", original)
    log_path = result_root / "run_log.json"

    def forge(payload):
        event = next(row for row in payload["events"] if row.get("fold") == folds[0].name)
        event["artifact_pair_sha256"] = {
            "fit.json": "f" * 64,
            "per_series.json": "f" * 64,
        }

    _write_rehashed_run_log(log_path, forge)
    before = _tree_snapshot(result_root, analysis_root)
    monkeypatch.setattr(
        runtime,
        "fit_and_evaluate_development_fold",
        lambda *args, **kwargs: pytest.fail("forged pair hash reached next fit"),
    )
    monkeypatch.setattr(
        runtime,
        "_atomic_write",
        lambda *args, **kwargs: pytest.fail("forged pair hash reached a write"),
    )

    with pytest.raises(RuntimeError, match="run-log|artifact|pair|hash"):
        runtime._run_guarded_development(8)

    assert _tree_snapshot(result_root, analysis_root) == before


@pytest.mark.parametrize(
    ("argv", "entry"),
    (
        (("prepare-development", "--workers", "8"), "_prepare_development"),
        (("development", "--workers", "8"), "run_development"),
        (("aggregate-development", "--verify-only"), "_verify_development_outputs"),
    ),
)
def test_task6_fix1_cli_failure_is_one_fixed_private_free_stderr_line(
    argv: tuple[str, ...], entry: str, tmp_path: Path
) -> None:
    sentinel = "PRIVATE-ACCOUNT-CHECKOUT-SENTINEL"
    checkout = tmp_path / sentinel / "checkout"
    shutil.copytree(runtime.ROOT / "src", checkout / "src")
    result_root = checkout / "results" / "iclr_residual_upgrade" / "development"
    analysis_root = checkout / "analysis-output" / "iclr_residual_upgrade" / "development"
    driver = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        import iclr_residual_train as runtime
        root = Path(sys.argv[1])
        runtime.DEVELOPMENT_ROOT = root / 'results' / 'iclr_residual_upgrade' / 'development'
        runtime.DEVELOPMENT_ANALYSIS_ROOT = root / 'analysis-output' / 'iclr_residual_upgrade' / 'development'
        def reject(*args, **kwargs):
            raise RuntimeError(sys.argv[2])
        setattr(runtime, sys.argv[3], reject)
        raise SystemExit(runtime.main(sys.argv[4:]))
        """
    )
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(checkout / "src"),
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            driver,
            str(checkout),
            f"private failure at {checkout}",
            entry,
            *argv,
        ],
        cwd=checkout,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 1
    assert completed.stdout == b""
    assert completed.stderr == b"DEVELOPMENT_COMMAND_REJECTED\n"
    assert sentinel.encode() not in completed.stderr
    assert b"/Users/" not in completed.stderr
    assert not result_root.exists()
    assert not analysis_root.exists()


@pytest.mark.parametrize(
    "attack",
    ("extra", "valid_symlink", "broken_symlink", "directory", "fifo", "socket"),
)
def test_task6_fix1_closed_input_inventory_rejects_every_unlisted_entry_before_read(
    attack: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _fix1_synthetic_authority(monkeypatch, tmp_path)
    extra = paths["city_data"] / "unlisted-private-sentinel.bin"
    server = None
    if attack == "extra":
        extra.write_bytes(b"private")
    elif attack == "valid_symlink":
        extra.symlink_to(paths["city_input"])
    elif attack == "broken_symlink":
        extra.symlink_to(paths["city_data"] / "missing-private-target")
    elif attack == "directory":
        extra.mkdir()
    elif attack == "fifo":
        os.mkfifo(extra)
    else:
        server = socket.socket(socket.AF_UNIX)
        previous = Path.cwd()
        try:
            os.chdir(paths["city_data"])
            server.bind(extra.name)
        finally:
            os.chdir(previous)
    real_sha = runtime._file_sha256
    read_paths = []

    def guarded_sha(path):
        read_paths.append(Path(path))
        if Path(path) == extra:
            pytest.fail("unlisted input was read before closed-set validation")
        return real_sha(path)

    monkeypatch.setattr(runtime, "_file_sha256", guarded_sha)
    try:
        with pytest.raises(RuntimeError, match="input|inventory|extra|symlink|special"):
            _collect_synthetic_preparation_inputs(3, 2)
    finally:
        if server is not None:
            server.close()
    assert extra not in read_paths


@pytest.mark.parametrize("attack", ("missing", "expected_symlink"))
def test_task6_fix1_closed_input_inventory_rejects_missing_or_symlinked_approved_file(
    attack: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _fix1_synthetic_authority(monkeypatch, tmp_path)
    approved = paths["city_input"]
    approved.unlink()
    if attack == "expected_symlink":
        approved.symlink_to(paths["district_input"])

    with pytest.raises(RuntimeError, match="input|inventory|missing|symlink"):
        _collect_synthetic_preparation_inputs(3, 2)


@pytest.mark.parametrize(
    ("inventory", "entry_kind"),
    (
        ("source", "broken_symlink"),
        ("source", "directory"),
        ("residual_test", "valid_symlink"),
        ("residual_test", "broken_symlink"),
    ),
)
def test_task6_fix1_source_and_residual_inventories_reject_hidden_matching_entries(
    inventory: str,
    entry_kind: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkout = _make_synthetic_checkout(tmp_path / "checkout")
    monkeypatch.setattr(runtime, "ROOT", checkout)
    if inventory == "source":
        entry = checkout / "src" / "hidden.py"
        target = checkout / "src" / "cohorts.py"
        call = runtime._collect_runtime_source_inventory
    else:
        entry = checkout / "tests" / "test_iclr_residual_hidden.py"
        target = checkout / "tests" / "test_iclr_residual_train.py"
        call = runtime._collect_residual_test_inventory
    if entry_kind == "directory":
        entry.mkdir()
    elif entry_kind == "valid_symlink":
        entry.symlink_to(target)
    else:
        entry.symlink_to(entry.parent / "missing.py")

    with pytest.raises(RuntimeError, match="source|test|inventory|symlink|regular"):
        call()


def test_task6_fix1_report_renders_every_factual_evidence_family_once() -> None:
    decision = {
        "classification": "fail",
        "conditions": {
            "actuation_coverage": False,
            "all_held_cities_negative": True,
            "all_seeds_negative": True,
            "district_fold_majority": True,
            "dual_reference_safety": True,
            "primary_improves_anchor": True,
        },
        "thresholds": {
            "bootstrap_draws": 20000,
            "bootstrap_seed": 20260818,
            "bootstrap_upper_max": -0.0314159,
            "district_negative_min": 4,
            "welfare_floor": 0.991234,
            "actuation_rate_min": 0.101234,
            "actuated_city_min": 2,
        },
        "observed": {
            "primary_inference": {
                "mean": -0.125,
                "upper": -0.0625,
                "draws": 20000,
                "seed": 20260818,
            },
            "negative_district_folds": 5,
            "seed_equal_city_means": {"1": -0.25, "2": -0.375, "42": -0.5},
            "actuated_series": 7,
            "held_series": 17,
            "actuation_rate": 0.411765,
            "actuated_cities": ["CITY-COVERAGE-MARKER"],
            "city_safety": {
                "FOLD-SAFETY-MARKER": {
                    "city": "CITY-SAFETY-MARKER",
                    "residual_welfare": 9.125,
                    "anchor_welfare": 9.0,
                    "mes_welfare": 8.875,
                    "anchor_welfare_threshold": 8.91,
                    "mes_welfare_threshold": 8.78625,
                    "anchor_welfare_ratio": 1.0138888888888888,
                    "mes_welfare_ratio": 1.028169014084507,
                    "residual_exclusion": 0.125,
                    "anchor_exclusion": 0.25,
                    "mes_exclusion": 0.375,
                    "anchor_exclusion_threshold": 0.25,
                    "mes_exclusion_threshold": 0.375,
                    "comparisons": {
                        "welfare_vs_anchor": True,
                        "welfare_vs_mes": True,
                        "exclusion_vs_anchor": True,
                        "exclusion_vs_mes": True,
                    },
                }
            },
        },
        "report_facts": {
            "folds": [
                {
                    "fold": "FOLD-SELECTION-MARKER",
                    "anchor_family": "ANCHOR-FAMILY-MARKER",
                    "primary_seed": 42,
                }
            ]
        },
    }
    runtime_summary = {
        "workers": 8,
        "elapsed_seconds": 12.75,
        "reported_peak_max_rss_bytes": 3145728,
        "peak_memory_interpretation": "PEAK-MEMORY-MARKER",
    }
    report = runtime._report_bytes(
        decision, runtime_summary, {"EVIDENCE-HASH-MARKER": "e" * 64}
    ).decode("utf-8")
    markers = (
        "-0.0314159",
        "0.991234",
        "FOLD-SELECTION-MARKER",
        "ANCHOR-FAMILY-MARKER",
        "CITY-COVERAGE-MARKER",
        "FOLD-SAFETY-MARKER",
        "CITY-SAFETY-MARKER",
        "PEAK-MEMORY-MARKER",
        "EVIDENCE-HASH-MARKER",
    )
    assert all(report.count(marker) == 1 for marker in markers)
    assert all(name in report for name in decision["thresholds"])
    assert all(str(seed) in report for seed in (1, 2, 42))
    assert all(name in report for name in decision["conditions"])


def test_task6_fix1_verify_only_preserves_content_and_access_times(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result_root, analysis_root, *_ = _seal(monkeypatch, tmp_path)
    capsys.readouterr()
    summary_sha = hashlib.sha256((result_root / "summary.json").read_bytes()).hexdigest()
    decision_sha = hashlib.sha256(
        (result_root / "evidence_decision.json").read_bytes()
    ).hexdigest()
    sealed = tuple(
        path
        for root in (result_root, analysis_root)
        for path in (root, *root.rglob("*"))
    )
    old_atime = 946684800_000_000_000
    for path in reversed(sealed):
        stat_result = path.stat()
        os.utime(path, ns=(old_atime, stat_result.st_mtime_ns), follow_symlinks=False)
    before = {
        path: (path.stat().st_atime_ns, path.stat().st_mtime_ns)
        for path in sealed
    }
    monkeypatch.setattr(
        runtime,
        "_collect_development_preparation_inputs",
        lambda: pytest.fail("verify-only recollected old inputs"),
    )
    monkeypatch.setattr(
        runtime,
        "_load_development_folds_unchecked",
        lambda: pytest.fail("verify-only called a fold loader"),
        raising=False,
    )
    monkeypatch.setattr(
        runtime,
        "load_series",
        lambda *args, **kwargs: pytest.fail("verify-only parsed PB data"),
    )
    monkeypatch.setattr(
        runtime,
        "fit_and_evaluate_development_fold",
        lambda *args, **kwargs: pytest.fail("verify-only called a fitter"),
    )
    monkeypatch.setattr(
        runtime,
        "_atomic_write",
        lambda *args, **kwargs: pytest.fail("verify-only called a writer"),
    )

    assert runtime.main(["aggregate-development", "--verify-only"]) == 0
    captured = capsys.readouterr()
    after = {
        path: (path.stat().st_atime_ns, path.stat().st_mtime_ns)
        for path in sealed
    }
    assert captured.out == (
        f"SUMMARY_SHA256 {summary_sha}\nDECISION_SHA256 {decision_sha}\n"
    )
    assert captured.err == ""
    assert after == before


def test_task6_fix1_seed_harness_is_packaged_deterministic_and_read_only(
    tmp_path: Path,
) -> None:
    script = textwrap.dedent(
        """
        import hashlib
        import json
        from pathlib import Path
        import tempfile
        import pytest
        import test_iclr_residual_runtime as fixtures
        import iclr_residual_train as runtime
        with tempfile.TemporaryDirectory(prefix='task6-seed-receipt-') as raw:
            patch = pytest.MonkeyPatch()
            try:
                result, analysis, *_ = fixtures._seal(patch, Path(raw))
                before = fixtures._tree_snapshot(result, analysis)
                summary, decision = runtime._verify_development_outputs()
                manifest = hashlib.sha256((result / 'manifest.sha256').read_bytes()).hexdigest()
                after = fixtures._tree_snapshot(result, analysis)
                print(json.dumps({
                    'summary': summary,
                    'decision': decision,
                    'manifest': manifest,
                    'unchanged': before == after,
                }, sort_keys=True))
            finally:
                patch.undo()
        """
    )
    receipts = []
    for seed in (1, 2, 3, 42):
        environment = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": str(seed),
            "PYTHONPATH": os.pathsep.join(
                (str(runtime.ROOT / "src"), str(runtime.ROOT / "tests"))
            ),
        }
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        receipt = json.loads(completed.stdout.splitlines()[-1])
        assert receipt["unchanged"] is True
        receipts.append(receipt)
    assert receipts == [receipts[0]] * 4


def test_task6_fix2_data_ancestor_symlink_rejects_before_outside_input_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _fix1_synthetic_authority(monkeypatch, tmp_path)
    outside_data = tmp_path / "outside-approved-inputs"
    shutil.move(str(paths["checkout"] / "data"), outside_data)
    (paths["checkout"] / "data").symlink_to(outside_data, target_is_directory=True)
    real_sha = runtime._file_sha256

    def reject_outside_read(path):
        resolved = Path(path).resolve(strict=False)
        if resolved == outside_data or outside_data in resolved.parents:
            pytest.fail("ancestor symlink reached an outside input byte")
        return real_sha(path)

    monkeypatch.setattr(runtime, "_file_sha256", reject_outside_read)

    with pytest.raises(RuntimeError, match="ancestor|symlink|contain|input|root"):
        _collect_synthetic_preparation_inputs(3, 2)


def test_task6_fix2_source_linked_package_rejects_before_hidden_python_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkout = _make_synthetic_checkout(tmp_path / "checkout")
    outside = tmp_path / "outside-package"
    outside.mkdir()
    (outside / "hidden.py").write_text("PRIVATE-SOURCE-SENTINEL\n", encoding="utf-8")
    (checkout / "src" / "linked_package").symlink_to(
        outside, target_is_directory=True
    )
    monkeypatch.setattr(runtime, "ROOT", checkout)
    real_sha = runtime._file_sha256

    def reject_outside_read(path):
        resolved = Path(path).resolve(strict=False)
        if resolved == outside or outside in resolved.parents:
            pytest.fail("source inventory read through a linked package")
        return real_sha(path)

    monkeypatch.setattr(runtime, "_file_sha256", reject_outside_read)

    with pytest.raises(RuntimeError, match="source|ancestor|symlink|directory"):
        runtime._collect_runtime_source_inventory()


@pytest.mark.parametrize("inventory", ("source", "tests"))
def test_task6_fix2_inventory_rejects_symlinked_checkout_ancestor_before_read(
    inventory: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    outside = _make_synthetic_checkout(tmp_path / "outside-checkout")
    checkout = tmp_path / "linked-checkout"
    checkout.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(runtime, "ROOT", checkout)
    real_sha = runtime._file_sha256

    def reject_outside_read(path):
        resolved = Path(path).resolve(strict=False)
        if resolved == outside or outside in resolved.parents:
            pytest.fail("inventory read through a symlinked checkout ancestor")
        return real_sha(path)

    monkeypatch.setattr(runtime, "_file_sha256", reject_outside_read)
    call = (
        runtime._collect_runtime_source_inventory
        if inventory == "source"
        else runtime._collect_residual_test_inventory
    )

    with pytest.raises(RuntimeError, match="ancestor|symlink|contain|root"):
        call()


def test_task6_fix2_authority_directory_symlink_rejects_before_outside_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _fix1_synthetic_authority(monkeypatch, tmp_path)
    outside = tmp_path / "outside-city-splits"
    shutil.move(str(paths["city_splits"]), outside)
    paths["city_splits"].symlink_to(outside, target_is_directory=True)
    real_read_text = Path.read_text

    def reject_outside_read(path, *args, **kwargs):
        resolved = Path(path).resolve(strict=False)
        if resolved == outside or outside in resolved.parents:
            pytest.fail("authority read through a linked directory")
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", reject_outside_read)

    with pytest.raises(RuntimeError, match="authority|ancestor|symlink|split"):
        _collect_synthetic_preparation_inputs(3, 2)


@pytest.mark.parametrize(
    "artifact",
    ("fold_fit", "aggregate", "report", "run_log", "manifest"),
)
def test_task6_fix2_broken_output_leaf_is_rejected_without_follow_or_replace(
    artifact: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkout = _make_synthetic_checkout(tmp_path / "checkout")
    result_root = checkout / "results" / "iclr_residual_upgrade" / "development"
    analysis_root = (
        checkout / "analysis-output" / "iclr_residual_upgrade" / "development"
    )
    monkeypatch.setattr(runtime, "ROOT", checkout)
    monkeypatch.setattr(runtime, "DEVELOPMENT_ROOT", result_root)
    monkeypatch.setattr(runtime, "DEVELOPMENT_ANALYSIS_ROOT", analysis_root)
    if artifact == "fold_fit":
        path = result_root / "folds" / "synthetic_fold" / "fit.json"
        write = lambda: runtime._immutable_canonical_json(path, {"safe": True})
    elif artifact == "report":
        path = analysis_root / "report.md"
        write = lambda: runtime._write_immutable_bytes(path, b"safe\n")
    elif artifact == "run_log":
        path = result_root / "run_log.json"
        write = lambda: runtime._atomic_canonical_json(path, {"safe": True})
    elif artifact == "manifest":
        path = result_root / "manifest.sha256"
        write = lambda: runtime._write_immutable_bytes(path, b"0" * 64 + b"\n")
    else:
        path = result_root / "summary.json"
        write = lambda: runtime._write_immutable_bytes(path, b"{}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    outside_target = tmp_path / f"outside-{artifact}.sentinel"
    path.symlink_to(outside_target)

    with pytest.raises(RuntimeError, match="ancestor|symlink|contain|output|regular"):
        write()

    assert path.is_symlink()
    assert not outside_target.exists()


def test_task6_fix2_output_ancestor_symlink_rejects_before_outside_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    checkout = _make_synthetic_checkout(tmp_path / "checkout")
    outside = tmp_path / "outside-results"
    outside.mkdir()
    (checkout / "results").symlink_to(outside, target_is_directory=True)
    result_root = checkout / "results" / "iclr_residual_upgrade" / "development"
    monkeypatch.setattr(runtime, "ROOT", checkout)
    monkeypatch.setattr(runtime, "DEVELOPMENT_ROOT", result_root)
    target = result_root / "summary.json"

    with pytest.raises(RuntimeError, match="ancestor|symlink|contain|output|root"):
        runtime._write_immutable_bytes(target, b"{}\n")

    assert not (outside / "iclr_residual_upgrade" / "development" / "summary.json").exists()


@pytest.mark.parametrize(
    "mutation",
    (
        "unexpected_outer_key",
        "missing_outer_key",
        "training_not_list",
        "training_nonstring",
        "training_duplicate",
        "training_absolute",
        "training_escape",
        "training_empty_component",
        "training_backslash",
        "scored_nonstring",
        "scored_duplicate",
        "scored_reordered",
        "instances_not_list",
        "instance_not_mapping",
        "instance_unexpected_private_key",
        "instance_missing_key",
        "instance_wrong_series_type",
        "instance_wrong_path_type",
        "instance_duplicate",
        "instance_reordered",
        "instance_misbound_series",
        "instance_absolute",
        "instance_escape",
        "instance_empty_component",
        "instance_backslash",
        "sha_uppercase",
        "sha_nonhex",
        "sha_wrong_length",
    ),
)
def test_task6_fix2_sealed_fold_view_requires_exact_nested_schema_and_paths(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result_root, _, *_ = _seal(monkeypatch, tmp_path)
    capsys.readouterr()
    run_log_path = result_root / "run_log.json"
    run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
    view = run_log["fold_views"][0]
    if mutation == "unexpected_outer_key":
        view["unexpected"] = "PRIVATE-RUNLOG-SCHEMA-SENTINEL"
    elif mutation == "missing_outer_key":
        view.pop("training_series")
    elif mutation == "training_not_list":
        view["training_series"] = {"unexpected": "mapping"}
    elif mutation == "training_nonstring":
        view["training_series"][0] = 7
    elif mutation == "training_duplicate":
        view["training_series"].append(view["training_series"][0])
    elif mutation == "training_absolute":
        view["training_series"][0] = "/Users/PRIVATE-RUNLOG-SCHEMA-SENTINEL"
    elif mutation == "training_escape":
        view["training_series"][0] = "Poland/../PRIVATE-RUNLOG-SCHEMA-SENTINEL"
    elif mutation == "training_empty_component":
        view["training_series"][0] = "Poland//PRIVATE-RUNLOG-SCHEMA-SENTINEL"
    elif mutation == "training_backslash":
        view["training_series"][0] = "Poland\\PRIVATE-RUNLOG-SCHEMA-SENTINEL"
    elif mutation == "scored_nonstring":
        view["scored_series"][0] = False
    elif mutation == "scored_duplicate":
        view["scored_series"][1] = view["scored_series"][0]
    elif mutation == "scored_reordered":
        view["scored_series"] = list(reversed(view["scored_series"]))
    elif mutation == "instances_not_list":
        view["scored_instances"] = {"unexpected": "mapping"}
    elif mutation == "instance_not_mapping":
        view["scored_instances"][0] = "PRIVATE-RUNLOG-SCHEMA-SENTINEL"
    elif mutation == "instance_unexpected_private_key":
        view["scored_instances"][0]["unexpected_private_path"] = (
            "/Users/PRIVATE-RUNLOG-SCHEMA-SENTINEL"
        )
    elif mutation == "instance_missing_key":
        view["scored_instances"][0].pop("series")
    elif mutation == "instance_wrong_series_type":
        view["scored_instances"][0]["series"] = 11
    elif mutation == "instance_wrong_path_type":
        view["scored_instances"][0]["instance"] = ["not", "a", "path"]
    elif mutation == "instance_duplicate":
        view["scored_instances"][1] = copy.deepcopy(view["scored_instances"][0])
    elif mutation == "instance_reordered":
        view["scored_instances"] = list(reversed(view["scored_instances"]))
    elif mutation == "instance_misbound_series":
        view["scored_instances"][0]["series"] = view["scored_series"][1]
    elif mutation == "instance_absolute":
        view["scored_instances"][0]["instance"] = (
            "/Users/PRIVATE-RUNLOG-SCHEMA-SENTINEL"
        )
    elif mutation == "instance_escape":
        view["scored_instances"][0]["instance"] = (
            "data/pb/../PRIVATE-RUNLOG-SCHEMA-SENTINEL.pb"
        )
    elif mutation == "instance_empty_component":
        view["scored_instances"][0]["instance"] = (
            "data/pb//PRIVATE-RUNLOG-SCHEMA-SENTINEL.pb"
        )
    elif mutation == "instance_backslash":
        view["scored_instances"][0]["instance"] = (
            "data\\pb\\PRIVATE-RUNLOG-SCHEMA-SENTINEL.pb"
        )
    elif mutation == "sha_uppercase":
        view["fold_input_sha256"] = view["fold_input_sha256"].upper()
    elif mutation == "sha_nonhex":
        view["fold_input_sha256"] = "g" * 64
    else:
        view["fold_input_sha256"] = "0" * 63
    _rehash(run_log)
    run_log_path.write_bytes(runtime._canonical_json_bytes(run_log))
    _refresh_manifest_run_log_hash(result_root)
    _install_verify_only_fail_fast_sentinels(monkeypatch)

    with pytest.raises(RuntimeError, match="fold|view|schema|path|identity|inventory"):
        runtime._verify_development_outputs()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_task6_fix2_prepared_prefix_rejects_second_fold_pair_before_transition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, _, folds, fit_records, series_records = _prepare(
        monkeypatch, tmp_path
    )
    _write_synthetic_pair(
        result_root, folds[1], fit_records[1], series_records[1]
    )
    monkeypatch.setattr(
        runtime,
        "fit_and_evaluate_development_fold",
        lambda *args, **kwargs: pytest.fail("future pair reached the first fit"),
    )
    monkeypatch.setattr(
        runtime,
        "_atomic_write",
        lambda *args, **kwargs: pytest.fail("future pair reached the first write"),
    )

    with pytest.raises(RuntimeError, match="prefix|pair|inventory|prepared"):
        runtime._run_guarded_development(8)


def test_task6_fix2_running_prefix_accepts_only_immediate_complete_orphan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, _, folds, fit_records, series_records = _prepare(
        monkeypatch, tmp_path
    )
    _advance_synthetic_log_to_running(result_root, folds)
    _write_synthetic_pair(
        result_root, folds[0], fit_records[0], series_records[0]
    )
    calls = _install_fake_fitter(
        monkeypatch, folds, fit_records, series_records
    )

    runtime._run_guarded_development(8)

    run_log = json.loads((result_root / "run_log.json").read_text(encoding="utf-8"))
    assert calls == [fold.name for fold in folds[1:]]
    assert run_log["state"] == "finalized"
    assert run_log["completed_folds"] == [fold.name for fold in folds]
    assert (result_root / "manifest.sha256").is_file()


def test_task6_fix2_running_prefix_rejects_two_orphan_pairs_before_first_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, _, folds, fit_records, series_records = _prepare(
        monkeypatch, tmp_path
    )
    _advance_synthetic_log_to_running(result_root, folds)
    for index in (0, 1):
        _write_synthetic_pair(
            result_root, folds[index], fit_records[index], series_records[index]
        )
    monkeypatch.setattr(
        runtime,
        "fit_and_evaluate_development_fold",
        lambda *args, **kwargs: pytest.fail("two orphans reached a fit"),
    )
    monkeypatch.setattr(
        runtime,
        "_atomic_write",
        lambda *args, **kwargs: pytest.fail("two orphans reached a write"),
    )

    with pytest.raises(RuntimeError, match="orphan|pair|prefix|inventory"):
        runtime._run_guarded_development(8)


def test_task6_fix2_running_prefix_rejects_partial_orphan_before_first_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, _, folds, fit_records, _ = _prepare(monkeypatch, tmp_path)
    _advance_synthetic_log_to_running(result_root, folds)
    root = result_root / "folds" / folds[0].name
    root.mkdir(parents=True)
    (root / "fit.json").write_bytes(runtime._canonical_json_bytes(fit_records[0]))
    monkeypatch.setattr(
        runtime,
        "fit_and_evaluate_development_fold",
        lambda *args, **kwargs: pytest.fail("partial orphan reached a fit"),
    )
    monkeypatch.setattr(
        runtime,
        "_atomic_write",
        lambda *args, **kwargs: pytest.fail("partial orphan reached a write"),
    )

    with pytest.raises(RuntimeError, match="partial|pair|prefix|inventory"):
        runtime._run_guarded_development(8)


def test_task6_fix2_running_prefix_rejects_extra_artifact_before_first_action(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, _, folds, _, _ = _prepare(monkeypatch, tmp_path)
    _advance_synthetic_log_to_running(result_root, folds)
    (result_root / "unexpected-prefix-artifact.txt").write_text(
        "unexpected\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        runtime,
        "fit_and_evaluate_development_fold",
        lambda *args, **kwargs: pytest.fail("extra artifact reached a fit"),
    )
    monkeypatch.setattr(
        runtime,
        "_atomic_write",
        lambda *args, **kwargs: pytest.fail("extra artifact reached a write"),
    )

    with pytest.raises(RuntimeError, match="extra|prefix|inventory|unexpected"):
        runtime._run_guarded_development(8)


def _crash_synthetic_run_at_folds_complete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    installed = _prepare(monkeypatch, tmp_path)
    result_root, _, folds, fit_records, series_records = installed
    _install_fake_fitter(monkeypatch, folds, fit_records, series_records)
    original = runtime._atomic_canonical_json

    def crash_after_folds_complete(path, payload):
        original(path, payload)
        if path.name == "run_log.json" and payload.get("state") == "folds_complete":
            raise RuntimeError("synthetic crash after folds_complete")

    monkeypatch.setattr(runtime, "_atomic_canonical_json", crash_after_folds_complete)
    with pytest.raises(RuntimeError, match="after folds_complete"):
        runtime._run_guarded_development(8)
    monkeypatch.setattr(runtime, "_atomic_canonical_json", original)
    return installed


def test_task6_fix2_folds_complete_rejects_late_divergence_before_earlier_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, _, *_ = _crash_synthetic_run_at_folds_complete(
        monkeypatch, tmp_path
    )
    assert not (result_root / "per_series.csv").exists()
    (result_root / "summary.json").write_bytes(b"divergent-summary\n")
    monkeypatch.setattr(
        runtime,
        "fit_and_evaluate_development_fold",
        lambda *args, **kwargs: pytest.fail("divergent aggregate reached a fit"),
    )
    monkeypatch.setattr(
        runtime,
        "_write_immutable_bytes",
        lambda *args, **kwargs: pytest.fail("divergent aggregate reached a write"),
    )
    monkeypatch.setattr(
        runtime,
        "_atomic_write",
        lambda *args, **kwargs: pytest.fail("divergent aggregate reached an atomic write"),
    )

    with pytest.raises(RuntimeError, match="aggregate|summary|prefix|divergent"):
        runtime._run_guarded_development(8)


def test_task6_fix2_folds_complete_rejects_out_of_order_valid_aggregate_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, _, folds, fit_records, series_records = _prepare(
        monkeypatch, tmp_path
    )
    _install_fake_fitter(monkeypatch, folds, fit_records, series_records)
    original = runtime._write_immutable_bytes

    def crash_after_summary(path, encoded):
        original(path, encoded)
        if path.name == "summary.json":
            raise RuntimeError("synthetic crash after summary")

    monkeypatch.setattr(runtime, "_write_immutable_bytes", crash_after_summary)
    with pytest.raises(RuntimeError, match="after summary"):
        runtime._run_guarded_development(8)
    (result_root / "per_series.csv").unlink()
    monkeypatch.setattr(
        runtime,
        "_write_immutable_bytes",
        lambda *args, **kwargs: pytest.fail("out-of-order prefix reached a write"),
    )
    monkeypatch.setattr(
        runtime,
        "_atomic_write",
        lambda *args, **kwargs: pytest.fail("out-of-order prefix reached an atomic write"),
    )

    with pytest.raises(RuntimeError, match="aggregate|prefix|order|inventory"):
        runtime._run_guarded_development(8)


@pytest.mark.parametrize(
    "mutation",
    (
        "manifest_unexpected_key",
        "manifest_row_unexpected_key",
        "manifest_missing_schema",
        "manifest_wrong_series_count_type",
        "manifest_wrong_election_count",
        "manifest_reordered_rows",
        "manifest_duplicate_row",
        "city_wrong_name",
        "city_unexpected_key",
        "city_missing_counts",
        "city_wrong_year_type",
        "city_missing_warmup_coverage",
        "district_wrong_name",
        "district_unexpected_key",
        "district_missing_note",
        "district_reordered_years",
        "district_duplicate_series",
        "district_missing_test_coverage",
    ),
)
def test_task6_fix2_preparation_rejects_nonexact_manifest_and_split_authority(
    mutation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _fix1_synthetic_authority(monkeypatch, tmp_path)
    city_split = paths["city_splits"] / f"{runtime._CITY_DEVELOPMENT_FOLDS[0]}.json"
    district_split = (
        paths["district_splits"] / f"{runtime._DISTRICT_DEVELOPMENT_FOLDS[0]}.json"
    )
    if mutation == "manifest_unexpected_key":
        _rewrite_json(
            paths["city_manifest"], lambda payload: payload.update(unexpected=True)
        )
    elif mutation == "manifest_row_unexpected_key":
        _rewrite_json(
            paths["city_manifest"],
            lambda payload: payload["files"][0].update(unexpected=True),
        )
    elif mutation == "manifest_missing_schema":
        _rewrite_json(paths["city_manifest"], lambda payload: payload.pop("schema_version"))
    elif mutation == "manifest_wrong_series_count_type":
        _rewrite_json(paths["city_manifest"], lambda payload: payload.update(n_series="2"))
    elif mutation == "manifest_wrong_election_count":
        _rewrite_json(paths["city_manifest"], lambda payload: payload.update(n_elections=5))
    elif mutation == "manifest_reordered_rows":
        _rewrite_json(paths["city_manifest"], lambda payload: payload["files"].reverse())
    elif mutation == "manifest_duplicate_row":
        def duplicate_manifest_row(payload):
            payload["files"].append(copy.deepcopy(payload["files"][0]))
            payload["n_files"] += 1
            payload["n_elections"] += 1

        _rewrite_json(paths["city_manifest"], duplicate_manifest_row)
    elif mutation == "city_wrong_name":
        _rewrite_json(city_split, lambda payload: payload.update(name="wrong-city-fold"))
    elif mutation == "city_unexpected_key":
        _rewrite_json(city_split, lambda payload: payload.update(unexpected=True))
    elif mutation == "city_missing_counts":
        _rewrite_json(city_split, lambda payload: payload.pop("counts"))
    elif mutation == "city_wrong_year_type":
        _rewrite_json(
            city_split,
            lambda payload: payload["fit"].update(
                {"Poland/Train/Approved": "2022"}
            ),
        )
    elif mutation == "city_missing_warmup_coverage":
        _rewrite_json(city_split, lambda payload: payload.update(warmup={}))
    elif mutation == "district_wrong_name":
        _rewrite_json(
            district_split, lambda payload: payload.update(name="wrong-district-fold")
        )
    elif mutation == "district_unexpected_key":
        _rewrite_json(district_split, lambda payload: payload.update(unexpected=True))
    elif mutation == "district_missing_note":
        _rewrite_json(district_split, lambda payload: payload.pop("note"))
    elif mutation == "district_reordered_years":
        _rewrite_json(
            district_split,
            lambda payload: payload["test"][0][1].reverse(),
        )
    elif mutation == "district_duplicate_series":
        _rewrite_json(
            district_split,
            lambda payload: payload["train"].append(
                copy.deepcopy(payload["train"][0])
            ),
        )
    else:
        _rewrite_json(district_split, lambda payload: payload.update(test=[]))

    with pytest.raises(RuntimeError, match="manifest|split|schema|authority|inventory"):
        _collect_synthetic_preparation_inputs(3, 2)


def test_task6_fix2_exact_metadata_only_authority_remains_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fix1_synthetic_authority(monkeypatch, tmp_path)

    descriptors, provenance = _collect_synthetic_preparation_inputs(3, 2)

    assert len(descriptors) == 8
    assert [row["name"] for row in descriptors] == [
        *runtime._CITY_DEVELOPMENT_FOLDS,
        *runtime._DISTRICT_DEVELOPMENT_FOLDS,
    ]
    assert len(provenance["old_input_sha256"]) == 7


@pytest.mark.parametrize("mutation", ("private_extra", "duplicate_training"))
def test_task6_fix2_run_log_authenticator_itself_rejects_nested_view_mutation(
    mutation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, _, folds, _, _ = _prepare(monkeypatch, tmp_path)
    config = json.loads((result_root / "config.json").read_text(encoding="utf-8"))
    prepared = json.loads((result_root / "run_log.json").read_text(encoding="utf-8"))
    running = runtime._advance_run_log(
        prepared,
        "running",
        fold_views=[runtime._loaded_fold_view(fold) for fold in folds],
    )
    if mutation == "private_extra":
        running["fold_views"][0]["scored_instances"][0]["private"] = (
            "/Users/PRIVATE-RUNLOG-AUTHENTICATOR-SENTINEL"
        )
    else:
        running["fold_views"][0]["training_series"].append(
            running["fold_views"][0]["training_series"][0]
        )
    _rehash(running)

    with pytest.raises(RuntimeError, match="fold|view|schema|inventory|path"):
        runtime._authenticate_run_log(running, config, finalized=False)


@pytest.mark.parametrize("mutation", ("private_extra", "misbound_series"))
def test_task6_fix2_sealed_evidence_authenticator_itself_rejects_nested_view_mutation(
    mutation: str,
) -> None:
    folds, _, fit_records, series_records = _synthetic_inputs()
    views = [runtime._loaded_fold_view(fold) for fold in folds]
    if mutation == "private_extra":
        views[0]["scored_instances"][0]["private"] = (
            "/Users/PRIVATE-SEALED-EVIDENCE-SENTINEL"
        )
    else:
        views[0]["scored_instances"][0]["series"] = views[0]["scored_series"][1]

    with pytest.raises(RuntimeError, match="fold|view|schema|inventory|path|series"):
        runtime._sealed_development_evidence(views, fit_records, series_records)


def test_task6_fix2_report_renders_every_required_leaf_exactly_once() -> None:
    decision = {
        "classification": "fail",
        "conditions": {
            "primary_improves_anchor": True,
            "all_held_cities_negative": True,
            "district_fold_majority": True,
            "dual_reference_safety": False,
            "all_seeds_negative": True,
            "actuation_coverage": True,
        },
        "failed_conditions": ["dual_reference_safety"],
        "thresholds": {
            "bootstrap_draws": 20000,
            "bootstrap_seed": 20260818,
            "bootstrap_upper_max": -0.010101,
            "district_negative_min": 4,
            "welfare_floor": 0.990099,
            "actuation_rate_min": 0.101019,
            "actuated_city_min": 2,
        },
        "observed": {
            "primary_inference": {
                "mean": -0.111111,
                "lower": -0.222222,
                "upper": -0.333333,
                "draws": 20001,
                "seed": 20260819,
                "series_count": 101,
                "cluster_count": 102,
                "city_count": 103,
            },
            "held_city_means": {
                "HELD-CITY-A-MARKER": -0.223344,
                "HELD-CITY-B-MARKER": -0.223355,
            },
            "district_fold_means": {
                "DISTRICT-FOLD-A-MARKER": -0.334455,
                "DISTRICT-FOLD-B-MARKER": -0.334466,
            },
            "negative_district_folds": 5,
            "city_safety": {
                "SAFETY-FAILURE-FOLD-MARKER": {
                    "city": "SAFETY-CITY-MARKER",
                    "residual_welfare": 9.101001,
                    "anchor_welfare": 9.102002,
                    "mes_welfare": 9.103003,
                    "anchor_welfare_threshold": 9.104004,
                    "mes_welfare_threshold": 9.105005,
                    "anchor_welfare_ratio": 1.106006,
                    "mes_welfare_ratio": 1.107007,
                    "residual_exclusion": 0.108008,
                    "anchor_exclusion": 0.109009,
                    "mes_exclusion": 0.11001,
                    "anchor_exclusion_threshold": 0.111011,
                    "mes_exclusion_threshold": 0.112012,
                    "comparisons": {
                        "welfare_vs_anchor": False,
                        "welfare_vs_mes": True,
                        "exclusion_vs_anchor": True,
                        "exclusion_vs_mes": True,
                    },
                    "failed_comparisons": ["welfare_vs_anchor"],
                }
            },
            "safety_failures": ["SAFETY-FAILURE-FOLD-MARKER"],
            "seed_equal_city_means": {
                "1": -0.441111,
                "2": -0.442222,
                "42": -0.443333,
            },
            "actuated_series": 107,
            "held_series": 109,
            "actuation_rate": 0.456789,
            "actuated_cities": [
                "ACTUATED-CITY-A-MARKER",
                "ACTUATED-CITY-B-MARKER",
            ],
        },
        "report_facts": {
            "folds": [
                {
                    "fold": "FOLD-SELECTION-A-MARKER",
                    "anchor_family": "ANCHOR-FAMILY-A-MARKER",
                    "primary_seed": 1,
                },
                {
                    "fold": "FOLD-SELECTION-B-MARKER",
                    "anchor_family": "ANCHOR-FAMILY-B-MARKER",
                    "primary_seed": 42,
                },
            ]
        },
    }
    runtime_summary = {
        "workers": 8,
        "start_utc": "START-UTC-MARKER",
        "end_utc": "END-UTC-MARKER",
        "elapsed_seconds": 12.121212,
        "parent_max_rss_bytes": 310001,
        "completed_children_max_rss_bytes": 310002,
        "reported_peak_max_rss_bytes": 310003,
        "peak_memory_interpretation": "PEAK-INTERPRETATION-MARKER",
        "config_sha256": "c" * 64,
    }
    artifact_hashes = {
        "EVIDENCE-PATH-A-MARKER": "d" * 64,
        "EVIDENCE-PATH-B-MARKER": "e" * 64,
    }

    report = runtime._report_bytes(
        decision, runtime_summary, artifact_hashes
    ).decode("utf-8")

    unique_markers = (
        "-0.010101",
        "0.990099",
        "0.101019",
        "-0.111111",
        "-0.222222",
        "-0.333333",
        "HELD-CITY-A-MARKER",
        "HELD-CITY-B-MARKER",
        "-0.223344",
        "-0.223355",
        "DISTRICT-FOLD-A-MARKER",
        "DISTRICT-FOLD-B-MARKER",
        "-0.334455",
        "-0.334466",
        "SAFETY-FAILURE-FOLD-MARKER",
        "SAFETY-CITY-MARKER",
        "1.106006",
        "1.107007",
        "-0.441111",
        "-0.442222",
        "-0.443333",
        "0.456789",
        "ACTUATED-CITY-A-MARKER",
        "ACTUATED-CITY-B-MARKER",
        "FOLD-SELECTION-A-MARKER",
        "FOLD-SELECTION-B-MARKER",
        "ANCHOR-FAMILY-A-MARKER",
        "ANCHOR-FAMILY-B-MARKER",
        "START-UTC-MARKER",
        "END-UTC-MARKER",
        "12.121212",
        "310001",
        "310002",
        "310003",
        "PEAK-INTERPRETATION-MARKER",
        "EVIDENCE-PATH-A-MARKER",
        "EVIDENCE-PATH-B-MARKER",
    )
    assert all(report.count(marker) == 1 for marker in unique_markers)
    assert report.count("safety failure: `true`") == 1
    assert report.count(
        "`welfare_vs_anchor`: `false`; failed comparison: `true`"
    ) == 1


def _clone_library_double(attack: str):
    if attack == "missing_symbol":
        return object()
    if attack == "invalid_symbol":
        class InvalidCloneSymbol:
            def __setattr__(self, name, value):
                raise LookupError("PRIVATE-CLONE-SYMBOL-SENTINEL")

        class InvalidLibrary:
            clonefile = InvalidCloneSymbol()

        return InvalidLibrary()
    raise AssertionError(f"unexpected clone attack {attack}")


@pytest.mark.parametrize(
    "attack", ("missing_symbol", "cdll_failure", "invalid_symbol")
)
def test_task6_fix2_clone_setup_normalizes_before_sealed_content_read(
    attack: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root = tmp_path / "sealed-result"
    analysis_root = tmp_path / "sealed-analysis"
    result_root.mkdir()
    analysis_root.mkdir()
    monkeypatch.setattr(runtime, "DEVELOPMENT_ROOT", result_root)
    monkeypatch.setattr(runtime, "DEVELOPMENT_ANALYSIS_ROOT", analysis_root)
    monkeypatch.setattr(runtime.sys, "platform", "darwin")
    if attack == "cdll_failure":
        def fail_cdll(*args, **kwargs):
            raise LookupError("PRIVATE-CDLL-SENTINEL")

        monkeypatch.setattr(runtime.ctypes, "CDLL", fail_cdll)
    else:
        monkeypatch.setattr(
            runtime.ctypes,
            "CDLL",
            lambda *args, **kwargs: _clone_library_double(attack),
        )
    real_read_bytes = Path.read_bytes

    def reject_sealed_read(path, *args, **kwargs):
        candidate = Path(path)
        if candidate == result_root or result_root in candidate.parents:
            pytest.fail("clone setup failure read sealed result content")
        if candidate == analysis_root or analysis_root in candidate.parents:
            pytest.fail("clone setup failure read sealed analysis content")
        return real_read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", reject_sealed_read)
    before = (tuple(result_root.iterdir()), tuple(analysis_root.iterdir()))

    with pytest.raises(RuntimeError, match="no-atime|clone|read-only"):
        with runtime._noatime_sealed_output_snapshot():
            pytest.fail("invalid clone setup entered the verification body")

    assert (tuple(result_root.iterdir()), tuple(analysis_root.iterdir())) == before


@pytest.mark.parametrize(
    "attack", ("missing_symbol", "cdll_failure", "invalid_symbol")
)
def test_task6_fix2_clone_setup_cli_subprocess_has_fixed_private_free_failure(
    attack: str, tmp_path: Path
) -> None:
    sentinel = "PRIVATE-CLONE-SETUP-SENTINEL"
    checkout = tmp_path / sentinel / "checkout"
    shutil.copytree(runtime.ROOT / "src", checkout / "src")
    result_root = checkout / "synthetic-result"
    analysis_root = checkout / "synthetic-analysis"
    result_root.mkdir()
    analysis_root.mkdir()
    driver = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        import iclr_residual_train as runtime
        root = Path(sys.argv[1])
        attack = sys.argv[2]
        runtime.DEVELOPMENT_ROOT = root / 'synthetic-result'
        runtime.DEVELOPMENT_ANALYSIS_ROOT = root / 'synthetic-analysis'
        runtime.sys.platform = 'darwin'
        if attack == 'missing_symbol':
            runtime.ctypes.CDLL = lambda *args, **kwargs: object()
        elif attack == 'cdll_failure':
            def fail_cdll(*args, **kwargs):
                raise LookupError('PRIVATE-CDLL-SUBPROCESS-SENTINEL')
            runtime.ctypes.CDLL = fail_cdll
        else:
            class InvalidCloneSymbol:
                def __setattr__(self, name, value):
                    raise LookupError('PRIVATE-SYMBOL-SUBPROCESS-SENTINEL')
            class InvalidLibrary:
                clonefile = InvalidCloneSymbol()
            runtime.ctypes.CDLL = lambda *args, **kwargs: InvalidLibrary()
        raise SystemExit(runtime.main(['aggregate-development', '--verify-only']))
        """
    )
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(checkout / "src"),
    }

    completed = subprocess.run(
        [sys.executable, "-c", driver, str(checkout), attack],
        cwd=checkout,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 1
    assert completed.stdout == b""
    assert completed.stderr == b"DEVELOPMENT_COMMAND_REJECTED\n"
    assert sentinel.encode() not in completed.stderr
    assert b"/Users/" not in completed.stderr
    assert tuple(result_root.iterdir()) == ()
    assert tuple(analysis_root.iterdir()) == ()


def test_task6_fix2_cli_catches_every_ordinary_exception_only(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class UnexpectedDevelopmentFailure(Exception):
        pass

    monkeypatch.setattr(
        runtime,
        "_prepare_development",
        lambda workers: (_ for _ in ()).throw(
            UnexpectedDevelopmentFailure("PRIVATE-ORDINARY-EXCEPTION-SENTINEL")
        ),
    )

    assert runtime.main(["prepare-development", "--workers", "8"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "DEVELOPMENT_COMMAND_REJECTED\n"


@pytest.mark.parametrize("exception", (KeyboardInterrupt(), SystemExit(9)))
def test_task6_fix2_cli_does_not_catch_base_exception_subclasses(
    exception: BaseException, monkeypatch: pytest.MonkeyPatch
) -> None:
    def stop(workers):
        raise exception

    monkeypatch.setattr(runtime, "_prepare_development", stop)

    with pytest.raises(type(exception)):
        runtime.main(["prepare-development", "--workers", "8"])


def _fix3_install_private_pair_fitter(
    monkeypatch: pytest.MonkeyPatch,
    folds,
    fit_records,
    series_records,
) -> list[str]:
    records = {
        fold.name: (copy.deepcopy(fit), copy.deepcopy(series))
        for fold, fit, series in zip(folds, fit_records, series_records)
    }
    calls: list[str] = []

    def fake_private(fold, workers, **kwargs):
        assert workers == 8
        assert kwargs["root"] == runtime.DEVELOPMENT_ROOT
        calls.append(fold.name)
        fit_record, series_record = records[fold.name]
        _write_synthetic_pair(
            runtime.DEVELOPMENT_ROOT,
            fold,
            fit_record,
            series_record,
        )
        return copy.deepcopy(series_record["result"])

    monkeypatch.setattr(runtime, "_fit_and_evaluate_development_fold", fake_private)
    return calls


def _fix3_record_completed_prefix(
    result_root: Path,
    folds,
    fit_records,
    series_records,
    count: int,
) -> None:
    _advance_synthetic_log_to_running(result_root, folds)
    run_log_path = result_root / "run_log.json"
    run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
    for index in range(count):
        fold = folds[index]
        _write_synthetic_pair(
            result_root, fold, fit_records[index], series_records[index]
        )
        root = result_root / "folds" / fold.name
        run_log = runtime._advance_run_log(
            run_log,
            "running",
            fold=fold,
            pair_sha256={
                "fit.json": hashlib.sha256((root / "fit.json").read_bytes()).hexdigest(),
                "per_series.json": hashlib.sha256(
                    (root / "per_series.json").read_bytes()
                ).hexdigest(),
            },
        )
        run_log_path.write_bytes(runtime._canonical_json_bytes(run_log))


def test_task6_fix3_three_city_two_district_metadata_subset_is_accepted_without_loader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths, district = _fix3_three_city_two_district_authority(
        monkeypatch, tmp_path
    )

    descriptors, provenance = _collect_synthetic_preparation_inputs(3, 2)

    district_descriptors = [
        row for row in descriptors if row["kind"] == "district"
    ]
    assert len(descriptors) == 8
    assert len(district_descriptors) == 5
    assert {
        Path(path).name
        for row in district_descriptors
        for path in row["input_blob_sha256"]
    } == {
        "approved-score-2022.pb",
        "approved-score-2023.pb",
        "approved-train-2022.pb",
    }
    assert "city-only-2022.pb" not in {
        Path(path).name for row in district_descriptors for path in row["input_blob_sha256"]
    }
    assert len(district) == 2
    assert len(provenance["old_input_sha256"]) == 7
    assert paths["city_data"].is_dir() and paths["district_data"].is_dir()


@pytest.mark.parametrize("mutation", ("missing", "extra", "misbound"))
def test_task6_fix3_district_subset_mismatch_rejects_before_old_input_bytes(
    mutation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths, district = _fix3_three_city_two_district_authority(
        monkeypatch, tmp_path
    )
    if mutation == "missing":
        target = paths["district_splits"] / f"{runtime._DISTRICT_DEVELOPMENT_FOLDS[2]}.json"
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["train"].pop()
        target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    elif mutation == "extra":
        changed = {**district, "Poland/Outside/Unbound": [2022]}
        _fix3_write_district_partition(paths, changed)
    else:
        changed = copy.deepcopy(district)
        changed["Poland/Test/Approved"] = [2022, 2024]
        _fix3_write_district_partition(paths, changed)
    _fix3_guard_old_input_bytes(monkeypatch, paths)

    with pytest.raises(RuntimeError, match="district|subset|coverage|authority|partition"):
        _collect_synthetic_preparation_inputs(3, 2)


@pytest.mark.parametrize(
    "mutation",
    (
        "manifest_schema_bool",
        "city_schema_bool",
        "city_count_bool",
        "manifest_year_bool",
        "district_year_bool",
        "city_train_year_bool",
        "city_train_through_bool",
    ),
)
def test_task6_fix3_metadata_integer_fields_reject_boolean_aliases_before_inputs(
    mutation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths, _ = _fix3_five_series_authority(monkeypatch, tmp_path)
    city_split = paths["city_splits"] / f"{runtime._CITY_DEVELOPMENT_FOLDS[0]}.json"
    district_split = paths["district_splits"] / f"{runtime._DISTRICT_DEVELOPMENT_FOLDS[0]}.json"
    if mutation == "manifest_schema_bool":
        _rewrite_json(paths["city_manifest"], lambda row: row.update(schema_version=True))
    elif mutation == "city_schema_bool":
        _rewrite_json(city_split, lambda row: row.update(schema_version=True))
    elif mutation == "city_count_bool":
        _rewrite_json(city_split, lambda row: row["counts"].update(warmup_series=True))
    elif mutation == "manifest_year_bool":
        _rewrite_json(paths["city_manifest"], lambda row: row["files"][0].update(year=True))
    elif mutation == "district_year_bool":
        _rewrite_json(district_split, lambda row: row["test"][0][1].__setitem__(0, True))
    elif mutation == "city_train_year_bool":
        _rewrite_json(
            city_split,
            lambda row: row["fit"][next(iter(row["fit"]))].__setitem__(0, True),
        )
    else:
        _rewrite_json(city_split, lambda row: row.update(train_through=True))
    _fix3_guard_old_input_bytes(monkeypatch, paths)

    with pytest.raises(RuntimeError, match="manifest|split|schema|year|count|authority"):
        _collect_synthetic_preparation_inputs(6, 5)


def test_task6_fix3_exact_five_fold_district_partition_is_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fix3_five_series_authority(monkeypatch, tmp_path)

    descriptors, _ = _collect_synthetic_preparation_inputs(6, 5)

    assert [row["name"] for row in descriptors[-5:]] == list(
        runtime._DISTRICT_DEVELOPMENT_FOLDS
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "identical",
        "shifted_stride",
        "duplicate_test_membership",
        "missing_test_membership",
        "wrong_train_complement",
        "fold_name",
        "fold_row_order",
    ),
)
def test_task6_fix3_cross_fold_partition_mutations_reject_before_inputs(
    mutation: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths, series_years = _fix3_five_series_authority(monkeypatch, tmp_path)
    names = runtime._DISTRICT_DEVELOPMENT_FOLDS
    if mutation == "identical":
        source = json.loads(
            (paths["district_splits"] / f"{names[0]}.json").read_text(encoding="utf-8")
        )
        for name in names:
            payload = copy.deepcopy(source)
            payload["name"] = name
            (paths["district_splits"] / f"{name}.json").write_text(
                json.dumps(payload, sort_keys=True), encoding="utf-8"
            )
    elif mutation == "shifted_stride":
        _fix3_write_district_partition(paths, series_years, offset=1)
    elif mutation in {"duplicate_test_membership", "missing_test_membership"}:
        first = json.loads(
            (paths["district_splits"] / f"{names[0]}.json").read_text(encoding="utf-8")
        )
        second_path = paths["district_splits"] / f"{names[1]}.json"
        second = copy.deepcopy(first)
        second["name"] = names[1]
        second_path.write_text(json.dumps(second, sort_keys=True), encoding="utf-8")
    elif mutation == "wrong_train_complement":
        target = paths["district_splits"] / f"{names[0]}.json"
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["train"].pop()
        target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    elif mutation == "fold_name":
        target = paths["district_splits"] / f"{names[2]}.json"
        _rewrite_json(target, lambda row: row.update(name=names[1]))
    else:
        target = paths["district_splits"] / f"{names[0]}.json"
        _rewrite_json(target, lambda row: row["train"].reverse())
    _fix3_guard_old_input_bytes(monkeypatch, paths)

    with pytest.raises(RuntimeError, match="district|partition|membership|order|coverage|split"):
        _collect_synthetic_preparation_inputs(6, 5)


def test_task6_fix3_public_fold_rejects_second_fold_before_private_fit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, _, folds, _, _ = _prepare(monkeypatch, tmp_path)
    _advance_synthetic_log_to_running(result_root, folds)
    calls = []
    monkeypatch.setattr(
        runtime,
        "_fit_and_evaluate_development_fold",
        lambda *args, **kwargs: calls.append(args[0].name) or {},
    )

    with pytest.raises(RuntimeError, match="next|prefix|fold|running"):
        runtime.fit_and_evaluate_development_fold(folds[1], 8)

    assert calls == []


def test_task6_fix3_public_fold_allows_first_and_completed_prefix_next_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = _prepare(monkeypatch, tmp_path / "first")
    first_root, _, first_folds, first_fit, first_series = first
    _advance_synthetic_log_to_running(first_root, first_folds)
    first_calls = _fix3_install_private_pair_fitter(
        monkeypatch, first_folds, first_fit, first_series
    )

    runtime.fit_and_evaluate_development_fold(first_folds[0], 8)

    assert first_calls == [first_folds[0].name]
    assert (first_root / "folds" / first_folds[0].name / "fit.json").is_file()

    second = _prepare(monkeypatch, tmp_path / "next")
    second_root, _, second_folds, second_fit, second_series = second
    _fix3_record_completed_prefix(
        second_root, second_folds, second_fit, second_series, 1
    )
    second_calls = _fix3_install_private_pair_fitter(
        monkeypatch, second_folds, second_fit, second_series
    )

    runtime.fit_and_evaluate_development_fold(second_folds[1], 8)

    assert second_calls == [second_folds[1].name]


@pytest.mark.parametrize("state", ("prepared", "folds_complete"))
def test_task6_fix3_public_fold_rejects_nonrunning_state_before_private_fit(
    state: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, _, folds, fit_records, series_records = _prepare(
        monkeypatch, tmp_path
    )
    if state == "folds_complete":
        _fix3_record_completed_prefix(
            result_root, folds, fit_records, series_records, len(folds)
        )
        run_log_path = result_root / "run_log.json"
        run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
        run_log = runtime._advance_run_log(run_log, "folds_complete")
        run_log_path.write_bytes(runtime._canonical_json_bytes(run_log))
    calls = []
    monkeypatch.setattr(
        runtime,
        "_fit_and_evaluate_development_fold",
        lambda *args, **kwargs: calls.append(args[0].name) or {},
    )

    with pytest.raises(RuntimeError, match="running|state|prefix|fold"):
        runtime.fit_and_evaluate_development_fold(folds[0], 8)

    assert calls == []


@pytest.mark.parametrize("prefix", ("immediate_orphan", "partial", "future"))
def test_task6_fix3_public_fold_rejects_existing_output_prefix_before_private_fit(
    prefix: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, _, folds, fit_records, series_records = _prepare(
        monkeypatch, tmp_path
    )
    _advance_synthetic_log_to_running(result_root, folds)
    if prefix == "immediate_orphan":
        _write_synthetic_pair(
            result_root, folds[0], fit_records[0], series_records[0]
        )
    elif prefix == "partial":
        root = result_root / "folds" / folds[0].name
        root.mkdir(parents=True)
        (root / "fit.json").write_bytes(
            runtime._canonical_json_bytes(fit_records[0])
        )
    else:
        _write_synthetic_pair(
            result_root, folds[1], fit_records[1], series_records[1]
        )
    calls = []
    monkeypatch.setattr(
        runtime,
        "_fit_and_evaluate_development_fold",
        lambda *args, **kwargs: calls.append(args[0].name) or {},
    )

    with pytest.raises(RuntimeError, match="orphan|partial|future|prefix|pair|inventory"):
        runtime.fit_and_evaluate_development_fold(folds[0], 8)

    assert calls == []


def test_task6_fix3_final_run_log_event_is_acyclic_and_report_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, _, *_ = _seal(monkeypatch, tmp_path)
    run_log = json.loads((result_root / "run_log.json").read_text(encoding="utf-8"))

    assert set(run_log["events"][-1]) == {"event", "state", "elapsed_seconds"}
    assert run_log["events"][-1]["event"] == "finalized"
    assert "report_sha256" not in run_log["events"][-1]


def test_task6_fix3_report_evidence_is_manifest_set_minus_only_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, analysis_root, folds, *_ = _seal(monkeypatch, tmp_path)
    logical = runtime._logical_development_paths(folds)
    report_logical = "analysis-output/iclr_residual_upgrade/development/report.md"
    run_logical = "results/iclr_residual_upgrade/development/run_log.json"

    evidence = runtime._report_evidence_hashes(folds)
    report = (analysis_root / "report.md").read_text(encoding="utf-8")

    assert set(evidence) == set(logical) - {report_logical}
    assert len(evidence) == 22
    assert evidence[run_logical] == hashlib.sha256(
        (result_root / "run_log.json").read_bytes()
    ).hexdigest()
    assert report.count(f"`{run_logical}`") == 1
    assert report.count(evidence[run_logical]) == 1
    assert report_logical not in evidence


def _fix3_crash_after_final_run_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    installed = _prepare(monkeypatch, tmp_path)
    result_root, _, folds, fit_records, series_records = installed
    _install_fake_fitter(monkeypatch, folds, fit_records, series_records)
    original = runtime._atomic_canonical_json

    def crash(path, payload):
        original(path, payload)
        if path.name == "run_log.json" and payload.get("state") == "finalized":
            raise RuntimeError("synthetic crash after final run log")

    monkeypatch.setattr(runtime, "_atomic_canonical_json", crash)
    with pytest.raises(RuntimeError, match="after final run log"):
        runtime._run_guarded_development(8)
    monkeypatch.setattr(runtime, "_atomic_canonical_json", original)
    return installed


def test_task6_fix3_restart_after_final_run_log_before_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, analysis_root, folds, fit_records, series_records = (
        _fix3_crash_after_final_run_log(monkeypatch, tmp_path)
    )
    run_log = json.loads((result_root / "run_log.json").read_text(encoding="utf-8"))
    assert run_log["state"] == "finalized"
    assert not (analysis_root / "report.md").exists()
    assert not (result_root / "manifest.sha256").exists()
    _install_fake_fitter(monkeypatch, folds, fit_records, series_records)

    runtime._run_guarded_development(8)

    assert (analysis_root / "report.md").is_file()
    assert (result_root / "manifest.sha256").is_file()


def test_task6_fix3_restart_after_report_before_manifest_is_stable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    installed = _prepare(monkeypatch, tmp_path)
    result_root, analysis_root, folds, fit_records, series_records = installed
    _install_fake_fitter(monkeypatch, folds, fit_records, series_records)
    original = runtime._write_immutable_bytes

    def crash(path, encoded, **kwargs):
        original(path, encoded, **kwargs)
        if path.name == "report.md":
            raise RuntimeError("synthetic crash after report")

    monkeypatch.setattr(runtime, "_write_immutable_bytes", crash)
    with pytest.raises(RuntimeError, match="after report"):
        runtime._run_guarded_development(8)
    monkeypatch.setattr(runtime, "_write_immutable_bytes", original)
    run_log = json.loads((result_root / "run_log.json").read_text(encoding="utf-8"))
    report_before = (analysis_root / "report.md").read_bytes()
    assert run_log["state"] == "finalized"
    assert not (result_root / "manifest.sha256").exists()
    _install_fake_fitter(monkeypatch, folds, fit_records, series_records)

    runtime._run_guarded_development(8)

    assert (analysis_root / "report.md").read_bytes() == report_before
    assert (result_root / "manifest.sha256").is_file()


def test_task6_fix3_finalized_missing_report_resumes_but_divergent_report_rejects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, analysis_root, folds, fit_records, series_records = (
        _fix3_crash_after_final_run_log(monkeypatch, tmp_path / "missing")
    )
    _install_fake_fitter(monkeypatch, folds, fit_records, series_records)
    runtime._run_guarded_development(8)
    assert (analysis_root / "report.md").is_file()

    divergent = _fix3_crash_after_final_run_log(
        monkeypatch, tmp_path / "divergent"
    )
    divergent_result, divergent_analysis, divergent_folds, divergent_fit, divergent_series = divergent
    divergent_analysis.mkdir(parents=True, exist_ok=True)
    (divergent_analysis / "report.md").write_bytes(b"divergent-report\n")
    before = _tree_snapshot(divergent_result, divergent_analysis)
    _install_fake_fitter(
        monkeypatch, divergent_folds, divergent_fit, divergent_series
    )

    with pytest.raises(RuntimeError, match="report|divergent|immutable"):
        runtime._run_guarded_development(8)

    assert _tree_snapshot(divergent_result, divergent_analysis) == before


def test_task6_fix3_finalized_run_log_rejects_legacy_report_digest_field(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, _, *_ = _seal(monkeypatch, tmp_path)
    config = json.loads((result_root / "config.json").read_text(encoding="utf-8"))
    run_log = json.loads((result_root / "run_log.json").read_text(encoding="utf-8"))
    report_path = runtime.DEVELOPMENT_ANALYSIS_ROOT / "report.md"
    run_log["events"][-1]["report_sha256"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    _rehash(run_log)

    with pytest.raises(RuntimeError, match="run-log|final|event|schema"):
        runtime._authenticate_run_log(run_log, config, finalized=True)


def test_task6_fix3_finalized_run_log_without_report_digest_authenticates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, _, *_ = _seal(monkeypatch, tmp_path)
    config = json.loads((result_root / "config.json").read_text(encoding="utf-8"))
    run_log = json.loads((result_root / "run_log.json").read_text(encoding="utf-8"))
    run_log["events"][-1].pop("report_sha256", None)
    _rehash(run_log)

    authenticated = runtime._authenticate_run_log(
        run_log, config, finalized=True
    )

    assert authenticated["state"] == "finalized"


def test_task6_fix3_sealed_restart_performs_no_writer_call_after_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, analysis_root, *_ = _seal(monkeypatch, tmp_path)
    before = _tree_snapshot(result_root, analysis_root)
    monkeypatch.setattr(
        runtime,
        "_write_immutable_bytes",
        lambda *args, **kwargs: pytest.fail("sealed restart reached immutable writer"),
    )
    monkeypatch.setattr(
        runtime,
        "_atomic_canonical_json",
        lambda *args, **kwargs: pytest.fail("sealed restart reached run-log writer"),
    )

    runtime._run_guarded_development(8)

    assert _tree_snapshot(result_root, analysis_root) == before


def test_task6_fix3_aggregated_state_rejects_impossible_early_report_before_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_root, analysis_root, folds, fit_records, series_records = _prepare(
        monkeypatch, tmp_path
    )
    _install_fake_fitter(monkeypatch, folds, fit_records, series_records)
    original = runtime._atomic_canonical_json

    def crash(path, payload):
        original(path, payload)
        if path.name == "run_log.json" and payload.get("state") == "aggregated":
            raise RuntimeError("synthetic crash after aggregated log")

    monkeypatch.setattr(runtime, "_atomic_canonical_json", crash)
    with pytest.raises(RuntimeError, match="after aggregated log"):
        runtime._run_guarded_development(8)
    monkeypatch.setattr(runtime, "_atomic_canonical_json", original)
    analysis_root.mkdir(parents=True, exist_ok=True)
    (analysis_root / "report.md").write_bytes(b"impossible-early-report\n")
    monkeypatch.setattr(
        runtime,
        "_write_immutable_bytes",
        lambda *args, **kwargs: pytest.fail(
            "impossible early report reached an output writer"
        ),
    )

    with pytest.raises(RuntimeError, match="report|prefix|chronology|state"):
        runtime._run_guarded_development(8)

    assert not (result_root / "manifest.sha256").exists()


class _Fix4List(list):
    pass


class _Fix4Dict(dict):
    pass


class _Fix4Text(str):
    pass


def _fix4_cardinality_authority(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    city_count: int,
    district_count: int,
) -> tuple[dict[str, Path], dict[str, list[int]]]:
    assert 3 <= city_count and 2 <= district_count <= city_count
    paths = _fix1_synthetic_authority(monkeypatch, tmp_path)
    district = {
        "Poland/Test/Approved": [2022, 2023],
        "Poland/Train/Approved": [2022],
    }
    additional_district = district_count - len(district)
    if additional_district:
        shutil.copyfile(
            paths["city_data"] / "city-only-2022.pb",
            paths["district_data"] / "city-only-2022.pb",
        )
        district["Poland/CityOnly/Approved"] = [2022]
        additional_district -= 1
    for index in range(city_count - 3):
        series = f"Poland/Cardinality/Series{index:03d}"
        include_in_district = index < additional_district
        _fix3_add_series_to_city_authority(
            paths,
            filename=f"cardinality-{index:03d}-2022.pb",
            series=series,
            year=2022,
            include_in_district=include_in_district,
        )
        if include_in_district:
            district[series] = [2022]
    assert len(district) == district_count
    _fix3_write_district_partition(paths, district)
    return paths, district


def _fix4_install_metadata_route_sentinels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "load_development_folds",
        "_load_development_folds_unchecked",
        "_load_city_development_folds",
        "_load_district_development_folds",
        "fit_and_evaluate_development_fold",
        "_fit_and_evaluate_development_fold",
        "run_development",
        "_run_guarded_development",
    ):
        monkeypatch.setattr(
            runtime,
            name,
            lambda *args, _name=name, **kwargs: pytest.fail(
                f"metadata-only cardinality test reached {_name}"
            ),
            raising=False,
        )
    monkeypatch.setattr(
        runtime,
        "load_series",
        lambda *args, **kwargs: pytest.fail(
            "metadata-only cardinality test parsed PB data"
        ),
    )


@pytest.mark.parametrize(
    ("city_count", "district_count"),
    ((74, 74), (74, 19)),
)
def test_task6_fix4_production_cardinality_rejects_before_old_input_bytes(
    city_count: int,
    district_count: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, _ = _fix4_cardinality_authority(
        monkeypatch,
        tmp_path,
        city_count=city_count,
        district_count=district_count,
    )
    _fix4_install_metadata_route_sentinels(monkeypatch)
    _fix3_guard_old_input_bytes(monkeypatch, paths)

    with pytest.raises(RuntimeError, match="production|series|subset|cardinality"):
        runtime._collect_development_preparation_inputs()


def test_task6_fix4_production_shape_75_city_19_district_is_parser_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _, district = _fix4_cardinality_authority(
        monkeypatch, tmp_path, city_count=75, district_count=19
    )
    _fix4_install_metadata_route_sentinels(monkeypatch)

    descriptors, _ = runtime._collect_development_preparation_inputs()

    assert len(descriptors) == 8
    assert len(district) == 19
    assert [row["kind"] for row in descriptors] == ["city"] * 3 + ["district"] * 5


def test_task6_fix4_three_city_two_district_uses_private_count_seam(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fix1_synthetic_authority(monkeypatch, tmp_path)
    _fix4_install_metadata_route_sentinels(monkeypatch)

    descriptors, _ = _collect_synthetic_preparation_inputs(3, 2)

    assert len(descriptors) == 8
    assert [row["kind"] for row in descriptors] == ["city"] * 3 + ["district"] * 5


def test_task6_fix4_production_collector_exposes_no_count_override() -> None:
    with pytest.raises(TypeError):
        runtime._collect_development_preparation_inputs(
            expected_city_series=3,
            expected_district_series=2,
        )


def _fix4_mutate_prepared_run_log(
    mutation: str,
    candidate: dict[str, object],
    config: dict[str, object],
):
    if mutation == "schema_bool":
        candidate["schema_version"] = True
    elif mutation == "workers_float":
        candidate["workers"] = 8.0
    elif mutation == "peak_label_forged":
        candidate["peak_memory_interpretation"] = "forged-runtime-label"
    elif mutation == "memory_bool":
        candidate["parent_max_rss_bytes"] = True
    elif mutation == "memory_float":
        candidate["parent_max_rss_bytes"] = float(candidate["parent_max_rss_bytes"])
    elif mutation == "elapsed_bool":
        candidate["elapsed_seconds"] = True
    elif mutation == "elapsed_int":
        candidate["elapsed_seconds"] = 0
        candidate["events"][0]["elapsed_seconds"] = 0
    elif mutation == "wrong_state":
        candidate["state"] = "wrong-state"
    elif mutation == "state_text_subclass":
        candidate["state"] = _Fix4Text("prepared")
    elif mutation == "start_text_subclass":
        candidate["start_utc"] = _Fix4Text(candidate["start_utc"])
    elif mutation == "noncanonical_config_sha":
        config["payload_sha256"] = "A" * 64
        candidate["config_sha256"] = config["payload_sha256"]
    elif mutation == "events_list_subclass":
        candidate["events"] = _Fix4List(candidate["events"])
    elif mutation == "completed_list_subclass":
        candidate["completed_folds"] = _Fix4List(candidate["completed_folds"])
    elif mutation == "fold_views_list_subclass":
        candidate["fold_views"] = _Fix4List(candidate["fold_views"])
    elif mutation == "prepared_event_dict_subclass":
        candidate["events"][0] = _Fix4Dict(candidate["events"][0])
    elif mutation == "prepared_event_label_subclass":
        candidate["events"][0]["event"] = _Fix4Text("prepared")
    elif mutation == "top_mapping_subclass":
        _rehash(candidate)
        return _Fix4Dict(candidate), config
    elif mutation == "config_runtime_workers_float":
        config["runtime"]["workers"] = 8.0
        _rehash(config)
        candidate["config_sha256"] = config["payload_sha256"]
    elif mutation == "config_and_log_peak_label_forged":
        config["runtime"]["peak_memory_interpretation"] = "forged-runtime-label"
        _rehash(config)
        candidate["config_sha256"] = config["payload_sha256"]
        candidate["peak_memory_interpretation"] = "forged-runtime-label"
    else:  # pragma: no cover - parameter inventory is closed below
        raise AssertionError(mutation)
    _rehash(candidate)
    return candidate, config


@pytest.mark.parametrize(
    "mutation",
    (
        "schema_bool",
        "workers_float",
        "peak_label_forged",
        "memory_bool",
        "memory_float",
        "elapsed_bool",
        "elapsed_int",
        "wrong_state",
        "state_text_subclass",
        "start_text_subclass",
        "noncanonical_config_sha",
        "events_list_subclass",
        "completed_list_subclass",
        "fold_views_list_subclass",
        "prepared_event_dict_subclass",
        "prepared_event_label_subclass",
        "top_mapping_subclass",
        "config_runtime_workers_float",
        "config_and_log_peak_label_forged",
    ),
)
def test_task6_fix4_run_log_top_level_scalars_are_exact_before_write(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_root, analysis_root, *_ = _prepare(monkeypatch, tmp_path)
    config = json.loads((result_root / "config.json").read_text(encoding="utf-8"))
    candidate = json.loads(
        (result_root / "run_log.json").read_text(encoding="utf-8")
    )
    candidate, config = _fix4_mutate_prepared_run_log(
        mutation, candidate, config
    )
    before = _tree_snapshot(result_root, analysis_root)
    monkeypatch.setattr(
        runtime,
        "_atomic_canonical_json",
        lambda *args, **kwargs: pytest.fail("invalid run log reached a JSON writer"),
    )
    monkeypatch.setattr(
        runtime,
        "_write_immutable_bytes",
        lambda *args, **kwargs: pytest.fail("invalid run log reached an artifact writer"),
    )

    with pytest.raises(RuntimeError, match="run-log|runtime|schema|SHA|configuration"):
        runtime._authenticate_run_log(candidate, config, finalized=False)

    assert _tree_snapshot(result_root, analysis_root) == before


def _fix4_mutate_complete_run_log(
    mutation: str, candidate: dict[str, object]
) -> None:
    if mutation == "execution_elapsed_int":
        candidate["events"][1]["elapsed_seconds"] = int(
            candidate["events"][1]["elapsed_seconds"]
        )
    elif mutation == "execution_event_dict_subclass":
        candidate["events"][1] = _Fix4Dict(candidate["events"][1])
    elif mutation == "fold_pair_mapping_subclass":
        candidate["events"][2]["artifact_pair_sha256"] = _Fix4Dict(
            candidate["events"][2]["artifact_pair_sha256"]
        )
    elif mutation == "fold_name_text_subclass":
        candidate["events"][2]["fold"] = _Fix4Text(
            candidate["events"][2]["fold"]
        )
    elif mutation == "aggregate_mapping_subclass":
        candidate["events"][-2]["aggregate_sha256"] = _Fix4Dict(
            candidate["events"][-2]["aggregate_sha256"]
        )
    elif mutation == "final_state_text_subclass":
        candidate["events"][-1]["state"] = _Fix4Text("finalized")
    elif mutation == "view_series_list_subclass":
        candidate["fold_views"][0]["training_series"] = _Fix4List(
            candidate["fold_views"][0]["training_series"]
        )
    elif mutation == "view_instance_mapping_subclass":
        candidate["fold_views"][0]["scored_instances"][0] = _Fix4Dict(
            candidate["fold_views"][0]["scored_instances"][0]
        )
    else:  # pragma: no cover - parameter inventory is closed below
        raise AssertionError(mutation)
    _rehash(candidate)


@pytest.mark.parametrize(
    "mutation",
    (
        "execution_elapsed_int",
        "execution_event_dict_subclass",
        "fold_pair_mapping_subclass",
        "fold_name_text_subclass",
        "aggregate_mapping_subclass",
        "final_state_text_subclass",
        "view_series_list_subclass",
        "view_instance_mapping_subclass",
    ),
)
def test_task6_fix4_each_run_log_event_uses_exact_json_scalar_types(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_root, _, *_ = _seal(monkeypatch, tmp_path)
    config = json.loads((result_root / "config.json").read_text(encoding="utf-8"))
    candidate = json.loads(
        (result_root / "run_log.json").read_text(encoding="utf-8")
    )
    _fix4_mutate_complete_run_log(mutation, candidate)

    with pytest.raises(RuntimeError, match="run-log|event|fold|view|schema"):
        runtime._authenticate_run_log(candidate, config, finalized=True)


def _fix4_crash_after_aggregated_run_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    installed = _prepare(monkeypatch, tmp_path)
    result_root, _, folds, fit_records, series_records = installed
    _install_fake_fitter(monkeypatch, folds, fit_records, series_records)
    original = runtime._atomic_canonical_json

    def crash(path, payload):
        original(path, payload)
        if path.name == "run_log.json" and payload.get("state") == "aggregated":
            raise RuntimeError("synthetic crash after aggregated run log")

    monkeypatch.setattr(runtime, "_atomic_canonical_json", crash)
    with pytest.raises(RuntimeError, match="after aggregated run log"):
        runtime._run_guarded_development(8)
    monkeypatch.setattr(runtime, "_atomic_canonical_json", original)
    return installed


def test_task6_fix4_finalization_samples_once_before_run_log_report_and_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    installed = _fix4_crash_after_aggregated_run_log(monkeypatch, tmp_path)
    result_root, analysis_root, folds, fit_records, series_records = installed
    _install_fake_fitter(monkeypatch, folds, fit_records, series_records)
    log_path = result_root / "run_log.json"
    aggregated = json.loads(log_path.read_text(encoding="utf-8"))
    previous_memory = {
        name: aggregated[name]
        for name in (
            "parent_max_rss_bytes",
            "completed_children_max_rss_bytes",
            "reported_peak_max_rss_bytes",
        )
    }
    final_elapsed = float(aggregated["elapsed_seconds"]) + 7.25
    final_clock = float(aggregated["monotonic_start"]) + final_elapsed
    final_memory = {
        "parent_max_rss_bytes": previous_memory["parent_max_rss_bytes"] + 1024,
        "completed_children_max_rss_bytes": (
            previous_memory["completed_children_max_rss_bytes"] + 4096
        ),
    }
    final_memory["reported_peak_max_rss_bytes"] = max(final_memory.values())
    final_utc = "2026-08-19T00:00:59Z"
    chronology: list[str] = []

    def clock():
        chronology.append("clock")
        return final_clock

    def sample(previous=None):
        chronology.append("rss")
        assert previous == previous_memory
        return dict(final_memory)

    def utc_now():
        chronology.append("utc")
        return final_utc

    original_json = runtime._atomic_canonical_json
    original_bytes = runtime._write_immutable_bytes

    def record_json(path, payload):
        if path.name == "run_log.json" and payload.get("state") == "finalized":
            chronology.append("run_log")
        return original_json(path, payload)

    def record_bytes(path, encoded, **kwargs):
        if path.name == "report.md":
            chronology.append("report")
        elif path.name == "manifest.sha256":
            chronology.append("manifest")
        return original_bytes(path, encoded, **kwargs)

    monkeypatch.setattr(runtime, "_runtime_monotonic", clock)
    monkeypatch.setattr(runtime, "_runtime_peak_memory_sample", sample)
    monkeypatch.setattr(runtime, "_runtime_utc_now", utc_now)
    monkeypatch.setattr(runtime, "_atomic_canonical_json", record_json)
    monkeypatch.setattr(runtime, "_write_immutable_bytes", record_bytes)

    runtime._run_guarded_development(8)

    finalized = json.loads(log_path.read_text(encoding="utf-8"))
    report = (analysis_root / "report.md").read_text(encoding="utf-8")
    assert chronology == ["clock", "rss", "utc", "run_log", "report", "manifest"]
    assert finalized["elapsed_seconds"] == final_elapsed
    assert finalized["events"][-1]["elapsed_seconds"] == final_elapsed
    assert finalized["end_utc"] == final_utc
    for name, value in final_memory.items():
        assert finalized[name] == value
    projection = runtime._runtime_summary_projection(finalized)
    for name, value in projection.items():
        assert report.count(f"- `{name}`: `{value}`") == 1

    receipts_before = {
        "run_log": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        "report": hashlib.sha256(
            (analysis_root / "report.md").read_bytes()
        ).hexdigest(),
        "manifest": hashlib.sha256(
            (result_root / "manifest.sha256").read_bytes()
        ).hexdigest(),
    }
    before = _tree_snapshot(result_root, analysis_root)
    monkeypatch.setattr(
        runtime,
        "_runtime_monotonic",
        lambda: pytest.fail("finalized restart resampled the clock"),
    )
    monkeypatch.setattr(
        runtime,
        "_runtime_peak_memory_sample",
        lambda previous=None: pytest.fail("finalized restart resampled RSS"),
    )
    monkeypatch.setattr(
        runtime,
        "_runtime_utc_now",
        lambda: pytest.fail("finalized restart resampled UTC"),
    )

    runtime._run_guarded_development(8)

    receipts_after = {
        "run_log": hashlib.sha256(log_path.read_bytes()).hexdigest(),
        "report": hashlib.sha256(
            (analysis_root / "report.md").read_bytes()
        ).hexdigest(),
        "manifest": hashlib.sha256(
            (result_root / "manifest.sha256").read_bytes()
        ).hexdigest(),
    }
    assert receipts_after == receipts_before
    assert _tree_snapshot(result_root, analysis_root) == before
