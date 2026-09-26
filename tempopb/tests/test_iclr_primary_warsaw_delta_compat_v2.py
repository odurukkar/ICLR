"""Behavioral contracts for the append-only legacy compatibility amendment."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
THEORY = "results/iclr_theory_check.csv"
INVENTORY = "results/frozen_instances.txt"
TAG = "#approval-set-first-occurrence-v2"


def _compat():
    assert importlib.util.find_spec("iclr_primary_warsaw_delta_compat_v2") is not None, (
        "append-only compatibility amendment has not been implemented"
    )
    return importlib.import_module("iclr_primary_warsaw_delta_compat_v2")


@pytest.fixture
def comparison_inputs():
    snapshots = {name: (ROOT / name).read_bytes() for name in (THEORY, INVENTORY)}
    consumers = json.loads(
        (ROOT / "results/iclr_primary_warsaw_v2/evaluation/paper_consumers.json").read_bytes()
    )
    return snapshots, consumers


def _rewrite_theory(snapshots, edit):
    rows = list(csv.DictReader(io.StringIO(snapshots[THEORY].decode())))
    edit(rows)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    snapshots[THEORY] = stream.getvalue().encode()


def test_exact_adapters_disclose_every_coordinate_without_mutating_inputs(comparison_inputs):
    snapshots, consumers = comparison_inputs
    original = copy.deepcopy(comparison_inputs)
    old, new, disclosure = _compat().derive_comparison_views(snapshots, consumers)

    assert comparison_inputs == original
    assert set(name for name in old if old[name] != snapshots[name]) == {THEORY}
    assert old[INVENTORY] == snapshots[INVENTORY]
    cells = disclosure["not_applicable_cells"]
    assert len(cells) == 324
    assert len({(r["series"], r["year"], r["policy"], r["field"]) for r in cells}) == 324
    for policy in ("greedy-cost", "learned-outcome-f1.0", "learned-outcome-f0.0"):
        for field in ("kappa", "implied_floor"):
            assert sum(r["policy"] == policy and r["field"] == field for r in cells) == 54
    parsed = list(csv.DictReader(io.StringIO(old[THEORY].decode())))
    for row in cells:
        assert row["original_value"] == "nan" and row["derived_value"] == "NA"
        assert row["v2_value"] is None
        assert row["reason"]
        assert len(row["original_sha256"]) == len(row["derived_sha256"]) == 64
        match = [r for r in parsed if r["series"] == row["series"] and int(r["year"]) == row["year"] and r["policy"] == row["policy"]]
        assert len(match) == 1 and match[0][row["field"]] == "NA"
    mappings = disclosure["file_identity_mappings"]
    assert len(mappings) == len({r["derived_file"] for r in mappings}) == 132
    for row in mappings:
        assert row["original_file"] == row["derived_file"] + TAG
        assert row["reason"] and len(row["original_sha256"]) == len(row["derived_sha256"]) == 64
    new_files = {r["file"] for r in new["consumers"]["primary.corpus"]["values"]["instance_rows"]}
    assert new_files == {r["derived_file"] for r in mappings}
    assert disclosure["finite_numeric_values_changed"] == 0
    assert disclosure["original_theory_sha256"] == hashlib.sha256(snapshots[THEORY]).hexdigest()
    assert disclosure["derived_theory_sha256"] == hashlib.sha256(old[THEORY]).hexdigest()


@pytest.mark.parametrize("value", ["nan", "NaN", "inf", "-inf", "1e309"])
def test_out_of_scope_nonfinite_values_are_rejected(comparison_inputs, value):
    snapshots, consumers = comparison_inputs
    _rewrite_theory(snapshots, lambda rows: rows[0].__setitem__("min_approval_share", value))
    with pytest.raises(RuntimeError, match="finite|scope|mask"):
        _compat().derive_comparison_views(snapshots, consumers)


@pytest.mark.parametrize("side,value", [("old", "0.0"), ("old", "NA"), ("old", "NaN"), ("new", 0.0), ("new", float("nan"))])
def test_expected_not_applicable_mask_must_match_exactly(comparison_inputs, side, value):
    snapshots, consumers = comparison_inputs
    if side == "old":
        def edit(rows):
            next(r for r in rows if r["policy"] == "greedy-cost")["kappa"] = value
        _rewrite_theory(snapshots, edit)
    else:
        rows = consumers["consumers"]["mechanism.theory_floor"]["values"]["theory_rows"]
        next(r for r in rows if r["policy"] == "greedy-cost")["kappa"] = value
    with pytest.raises(RuntimeError, match="finite|scope|mask"):
        _compat().derive_comparison_views(snapshots, consumers)


@pytest.mark.parametrize("mutation", ["duplicate", "unexpected_tag", "double_tag", "plain", "series", "year", "missing", "old_duplicate"])
def test_inventory_adapter_rejects_collisions_and_nonbijections(comparison_inputs, mutation):
    snapshots, consumers = comparison_inputs
    rows = consumers["consumers"]["primary.corpus"]["values"]["instance_rows"]
    if mutation == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif mutation == "unexpected_tag":
        rows[0]["file"] = rows[0]["file"].removesuffix(TAG) + "#other"
    elif mutation == "double_tag":
        rows[0]["file"] += TAG
    elif mutation == "plain":
        rows[0]["file"] = rows[0]["file"].removesuffix(TAG)
    elif mutation == "series":
        rows[0]["series"] = "wrong-series"
    elif mutation == "year":
        rows[0]["year"] += 1
    elif mutation == "missing":
        rows.pop()
    else:
        snapshots[INVENTORY] += snapshots[INVENTORY].splitlines(keepends=True)[1]
    with pytest.raises(RuntimeError, match="inventory|suffix|bijection|coordinate"):
        _compat().derive_comparison_views(snapshots, consumers)


def test_nonfinite_in_another_snapshot_is_not_silently_normalized(comparison_inputs):
    snapshots, consumers = comparison_inputs
    snapshots["results/other.csv"] = b"x\nnan\n"
    with pytest.raises(RuntimeError, match="finite|scope"):
        _compat().derive_comparison_views(snapshots, consumers)


@pytest.fixture(scope="module")
def full_build():
    compat = _compat()
    paths = [ROOT / compat.base.V2_RESULT_RELATIVE / name for name in (compat.JSON_OUTPUT, compat.CSV_OUTPUT)]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) if path.exists() else None for path in paths}
    built = compat.build_delta_compat_audit(ROOT)
    assert before == {path: (path.read_bytes(), path.stat().st_mtime_ns) if path.exists() else None for path in paths}
    return built


def test_real_authenticated_full_build_is_read_only_and_preserves_all_comparisons(full_build):
    compat = _compat()
    payload, rows = full_build
    assert payload["status"] == "pass"
    assert payload["release_audit"]["verified_member_count"] == 55
    assert payload["fit_audit"]["total"] == 49
    assert payload["fit_audit"]["weights_unavailable"] == 7
    assert len(rows) == payload["evaluation_audit"]["metric_coordinates"] == 245
    assert payload["evaluation_audit"]["winner_comparable"] == 0
    assert payload["consumer_audit"]["consumer_compared"] == 25
    assert len(payload["consumer_audit"]["numeric_rows"]) == 7435
    assert payload["manuscript_audit"]["protected_macro_total"] == 589
    assert len(payload["manuscript_audit"]["rows"]) == 589
    assert payload["causal_attribution"]["duplicate_only_attribution_allowed"] is False
    amendment = payload["compatibility_amendment"]
    assert amendment["not_applicable_cell_count"] == 324
    assert amendment["file_identity_mapping_count"] == 132
    assert amendment["finite_numeric_values_changed"] == 0
    assert payload["delta_csv"]["relative_path"] == "v1_v2_delta_compat_rows.csv"
    bindings = amendment["sealed_file_sha256"]
    for name in ("src/iclr_primary_warsaw_delta_compat_v2.py", "tests/test_iclr_primary_warsaw_delta_compat_v2.py", "results/iclr_primary_warsaw_v2/protocol_lock.json", "results/iclr_primary_warsaw_v2/evaluation/summary.json"):
        assert bindings[f"repo/{name}"] == hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    assert compat.base._json_bytes(payload)


@pytest.fixture
def writable_fixture(tmp_path, full_build, monkeypatch):
    compat = _compat()
    payload, rows = copy.deepcopy(full_build)
    result = tmp_path / compat.base.V2_RESULT_RELATIVE
    result.mkdir(parents=True)
    sentinel = tmp_path / "sealed-input"
    sentinel.write_bytes(b"sealed")
    payload["compatibility_amendment"]["sealed_file_sha256"] = {
        "repo/sealed-input": hashlib.sha256(b"sealed").hexdigest()
    }
    monkeypatch.setattr(compat, "build_delta_compat_audit", lambda root: copy.deepcopy((payload, rows)))
    return compat, tmp_path, result, sentinel, payload, rows


def test_immutable_write_and_independent_check_only(writable_fixture):
    compat, root, result, sentinel, payload, rows = writable_fixture
    summary = compat.write_delta_compat_audit(root)
    paths = (result / compat.JSON_OUTPUT, result / compat.CSV_OUTPUT)
    first = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    assert summary["status"] == "pass" and summary["consumer_numeric_coordinates"] == 7435
    assert compat.write_delta_compat_audit(root) == summary
    assert compat.check_delta_compat_audit(root)["status"] == "pass"
    assert first == {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    sentinel.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="changed|digest"):
        compat.check_delta_compat_audit(root)


@pytest.mark.parametrize("target", ["json", "csv"])
def test_both_output_conflicts_are_rejected_before_either_commit(writable_fixture, target):
    compat, root, result, *_ = writable_fixture
    conflict = result / (compat.JSON_OUTPUT if target == "json" else compat.CSV_OUTPUT)
    absent = result / (compat.CSV_OUTPUT if target == "json" else compat.JSON_OUTPUT)
    conflict.write_bytes(b"unrelated existing output")
    with pytest.raises(RuntimeError):
        compat.write_delta_compat_audit(root)
    assert conflict.read_bytes() == b"unrelated existing output"
    assert not absent.exists()


def test_changed_second_build_and_last_boundary_input_drift_prevent_commit(writable_fixture, monkeypatch):
    compat, root, result, sentinel, payload, rows = writable_fixture
    calls = []
    def changed_build(root):
        calls.append(root)
        changed = copy.deepcopy(payload)
        if len(calls) == 2:
            changed["interpretation"] = "different"
        return changed, copy.deepcopy(rows)
    monkeypatch.setattr(compat, "build_delta_compat_audit", changed_build)
    with pytest.raises(RuntimeError, match="recomputation"):
        compat.write_delta_compat_audit(root)
    assert not (result / compat.CSV_OUTPUT).exists()
    calls.clear()
    def drifting_build(root):
        calls.append(root)
        if len(calls) == 2:
            sentinel.write_bytes(b"changed after snapshots")
        return copy.deepcopy((payload, rows))
    monkeypatch.setattr(compat, "build_delta_compat_audit", drifting_build)
    with pytest.raises(RuntimeError, match="changed|digest"):
        compat.write_delta_compat_audit(root)
    assert not (result / compat.CSV_OUTPUT).exists()


def test_writer_guard_and_invalid_summary_never_commit(writable_fixture):
    compat, root, result, sentinel, payload, rows = writable_fixture
    with compat.base.evaluate_v2.evaluation_writer_lock_v2(result):
        with pytest.raises(RuntimeError, match="single writer"):
            compat.write_delta_compat_audit(root)
    payload["fit_audit"] = None
    with pytest.raises(RuntimeError, match="summary|coverage"):
        compat.write_delta_compat_audit(root)
    assert not (result / compat.CSV_OUTPUT).exists()


def test_check_only_rejects_missing_or_tampered_outputs_without_creating_files(writable_fixture):
    compat, root, result, *_ = writable_fixture
    with pytest.raises(RuntimeError, match="missing"):
        compat.check_delta_compat_audit(root)
    assert list(result.iterdir()) == []
    compat.write_delta_compat_audit(root)
    (result / compat.JSON_OUTPUT).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="recomputation"):
        compat.check_delta_compat_audit(root)
