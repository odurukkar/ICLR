"""Append-only, comparison-only amendment for two protected legacy encodings.

The frozen delta auditor remains unchanged. This amendment authenticates its
original inputs and preserves all finite numeric comparisons and legacy gaps.
Only documented structural N/A tokens and the exact semantics filename tag are
adapted in memory. It does not estimate a duplicate-removal causal effect.
"""

from __future__ import annotations

import copy
import csv
import io
import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import iclr_primary_warsaw_delta_audit_v2 as base


ROOT = Path(__file__).resolve().parent.parent
THEORY_MEMBER = "results/iclr_theory_check.csv"
INVENTORY_MEMBER = "results/frozen_instances.txt"
SEMANTICS_SUFFIX = "#approval-set-first-occurrence-v2"
NA_POLICIES = frozenset({"greedy-cost", "learned-outcome-f1.0", "learned-outcome-f0.0"})
NA_FIELDS = frozenset({"kappa", "implied_floor"})
JSON_OUTPUT = Path("v1_v2_delta_compat_audit.json")
CSV_OUTPUT = Path("v1_v2_delta_compat_rows.csv")
NA_REASON = "Support-floor parameter and implied guarantee are structurally not applicable to this policy; protected string nan matches typed v2 null."
FILE_REASON = "Exact semantics-profile suffix is representation provenance; the original file, series, and year coordinate is identical."
SOURCE_RELATIVE = Path("src/iclr_primary_warsaw_delta_compat_v2.py")
TEST_RELATIVE = Path("tests/test_iclr_primary_warsaw_delta_compat_v2.py")
_IMPORTED_SOURCE_SHA256 = base._sha256(base._read_regular(Path(__file__), label="amendment source"))


def _value_sha(value: object) -> str:
    return base._sha256(base._json_bytes(value))


def _reject_nonfinite(value: object, *, label: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_nonfinite(child, label=f"{label}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite(child, label=f"{label}/{index}")
    elif type(value) in (str, int, float):
        try:
            numeric = float(value)
        except ValueError:
            return
        if not math.isfinite(numeric):
            raise RuntimeError(f"out-of-scope non-finite value: {label}")


def _theory_view(snapshot: bytes, rows: object) -> tuple[bytes, list[dict[str, object]]]:
    spec = base.CONSUMER_CSV_COMPARISON_SPECS["mechanism.theory_floor"][0]
    old_rows = base._load_csv_bytes(snapshot, label="protected theory compatibility")
    if not old_rows or any(tuple(row) != spec["legacy_fields"] for row in old_rows):
        raise RuntimeError("theory compatibility table schema differs")
    if type(rows) is not list or any(type(row) is not dict for row in rows):
        raise RuntimeError("theory compatibility typed rows differ")
    identity = spec["identity_fields"]
    old_index = {base._coordinate_key(row, identity, label="legacy theory"): row for row in old_rows}
    new_index = {base._coordinate_key(row, identity, label="v2 theory"): row for row in rows}
    if len(old_index) != len(old_rows) or len(new_index) != len(rows) or set(old_index) != set(new_index):
        raise RuntimeError("theory compatibility coordinate bijection differs")
    disclosure: list[dict[str, object]] = []
    for key, old in old_index.items():
        new = new_index[key]
        for field in spec["numeric_fields"]:
            if field not in new:
                raise RuntimeError("theory compatibility numeric field coverage differs")
            if old["policy"] in NA_POLICIES and field in NA_FIELDS:
                if old[field] != "nan" or new[field] is not None:
                    raise RuntimeError("theory compatibility exact N/A mask differs")
                disclosure.append({
                    "consumer_id": "mechanism.theory_floor",
                    "source_member": THEORY_MEMBER,
                    "table": "theory_rows",
                    "series": old["series"], "year": int(old["year"]),
                    "policy": old["policy"], "field": field,
                    "original_value": "nan", "derived_value": "NA", "v2_value": None,
                    "original_sha256": _value_sha("nan"), "derived_sha256": _value_sha("NA"),
                    "reason": NA_REASON,
                })
                old[field] = "NA"
            else:
                # No additional missingness is covered by this amendment.
                if base._legacy_numeric(old[field], label=f"legacy theory/{field}") is None or base._legacy_numeric(new[field], label=f"v2 theory/{field}") is None:
                    raise RuntimeError("theory compatibility additional missingness mask differs")
        _reject_nonfinite(old, label="derived theory")
    counts = Counter((row["policy"], row["field"]) for row in disclosure)
    if counts != Counter({(policy, field): 54 for policy in NA_POLICIES for field in NA_FIELDS}):
        raise RuntimeError("theory compatibility exact 324-cell mask differs")
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=spec["legacy_fields"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(old_rows)
    disclosure.sort(key=lambda row: (row["series"], row["year"], row["policy"], row["field"]))
    return output.getvalue().encode("utf-8"), disclosure


def _inventory_view(snapshot: bytes, rows: object) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    try:
        lines = snapshot.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise RuntimeError("protected inventory is not UTF-8") from exc
    old_coordinates: set[tuple[str, str, int]] = set()
    old_files: set[str] = set()
    for line in lines:
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 3 or not fields[0].endswith(".pb") or "#" in fields[0] or not fields[1]:
            raise RuntimeError("protected inventory coordinate schema differs")
        try:
            year = int(fields[2])
        except ValueError as exc:
            raise RuntimeError("protected inventory year differs") from exc
        if str(year) != fields[2] or fields[0] in old_files:
            raise RuntimeError("protected inventory duplicate or noncanonical coordinate")
        old_files.add(fields[0])
        old_coordinates.add((fields[0], fields[1], year))
    if len(old_coordinates) != 132 or type(rows) is not list or len(rows) != 132:
        raise RuntimeError("inventory compatibility requires exactly 132 coordinates")
    result = copy.deepcopy(rows)
    seen: set[str] = set()
    coordinates: set[tuple[str, str, int]] = set()
    mappings: list[dict[str, object]] = []
    for row in result:
        if type(row) is not dict or type(row.get("file")) is not str or type(row.get("series")) is not str or type(row.get("year")) is not int:
            raise RuntimeError("v2 inventory coordinate schema differs")
        original = row["file"]
        if not original.endswith(SEMANTICS_SUFFIX):
            raise RuntimeError("v2 inventory exact semantics suffix is missing")
        derived = original[:-len(SEMANTICS_SUFFIX)]
        if not derived.endswith(".pb") or "#" in derived or derived in seen:
            raise RuntimeError("v2 inventory suffix collision or duplicate coordinate")
        coordinate = (derived, row["series"], row["year"])
        if coordinate not in old_coordinates:
            raise RuntimeError("inventory file/series/year coordinate bijection differs")
        seen.add(derived)
        coordinates.add(coordinate)
        mappings.append({
            "consumer_id": "primary.corpus", "source_member": INVENTORY_MEMBER,
            "table": "instance_rows", "series": row["series"], "year": row["year"],
            "original_file": original, "derived_file": derived,
            "original_sha256": _value_sha(original), "derived_sha256": _value_sha(derived),
            "reason": FILE_REASON,
        })
        row["file"] = derived
    if coordinates != old_coordinates:
        raise RuntimeError("inventory full coordinate bijection differs")
    mappings.sort(key=lambda row: row["derived_file"])
    return result, mappings


def derive_comparison_views(
    snapshots: Mapping[str, bytes], consumers: Mapping[str, object]
) -> tuple[dict[str, bytes], dict[str, object], dict[str, object]]:
    """Derive views of already authenticated inputs without modifying originals.

    Authentication is deliberately owned by build_delta_compat_audit; this pure
    helper cannot produce or install an audit by itself.
    """
    base._validate_json_tree(consumers, label="original v2 consumers")
    _reject_nonfinite(consumers, label="original v2 consumers")
    old = dict(snapshots)
    new = copy.deepcopy(consumers)
    old[THEORY_MEMBER], cells = _theory_view(
        snapshots[THEORY_MEMBER], consumers["consumers"]["mechanism.theory_floor"]["values"]["theory_rows"]
    )
    instance_rows, mappings = _inventory_view(
        snapshots[INVENTORY_MEMBER], consumers["consumers"]["primary.corpus"]["values"]["instance_rows"]
    )
    new["consumers"]["primary.corpus"]["values"]["instance_rows"] = instance_rows
    for name, content in old.items():
        if name.endswith(".csv"):
            _reject_nonfinite(base._load_csv_bytes(content, label=name), label=name)
        elif name.endswith(".json"):
            _reject_nonfinite(base._load_json_bytes(content, label=name), label=name)
    disclosure = {
        "scope": "comparison-only representations; original artifacts are unchanged",
        "not_applicable_cells": cells, "not_applicable_cell_count": len(cells),
        "file_identity_mappings": mappings, "file_identity_mapping_count": len(mappings),
        "finite_numeric_values_changed": 0,
        "original_theory_sha256": base._sha256(snapshots[THEORY_MEMBER]),
        "derived_theory_sha256": base._sha256(old[THEORY_MEMBER]),
        "original_inventory_sha256": base._sha256(snapshots[INVENTORY_MEMBER]),
        "original_consumers_canonical_sha256": _value_sha(consumers),
        "derived_consumers_canonical_sha256": _value_sha(new),
        "value_digest_encoding": "SHA-256 of UTF-8 sorted, indented strict JSON with final newline",
    }
    return old, new, disclosure


def _binding_path(root: Path, name: str) -> Path:
    if not name.startswith(("repo/", "release/")):
        raise RuntimeError("amendment sealed file scope differs")
    scope, relative = name.split("/", 1)
    base._safe_member_name(relative, label="amendment sealed path")
    return (root if scope == "repo" else root.parent) / relative


def _assert_sealed_files_unchanged(root: Path, bindings: object) -> None:
    if type(bindings) is not dict or not bindings:
        raise RuntimeError("amendment sealed file digest map differs")
    for name, expected in bindings.items():
        if type(name) is not str or not base._is_sha256(expected):
            raise RuntimeError("amendment sealed file digest differs")
        content = base._read_regular(_binding_path(root, name), label=f"sealed amendment input {name}")
        if base._sha256(content) != expected:
            raise RuntimeError(f"sealed amendment input changed: {name}")


def _sealed_file_bindings(
    root: Path, release: Mapping[str, object], matrix: Mapping[str, object], evaluation: Mapping[str, object]
) -> dict[str, str]:
    result_root = root / base.V2_RESULT_RELATIVE
    lock_path = result_root / "protocol_lock.json"
    lock = base._load_json_bytes(base._read_regular(lock_path, label="sealed base lock"), label="sealed base lock")
    bindings: dict[str, str] = {}

    def add_repo(path: Path, digest: str) -> None:
        name = "repo/" + path.relative_to(root).as_posix()
        if name in bindings and bindings[name] != digest:
            raise RuntimeError(f"conflicting sealed input digests: {name}")
        bindings[name] = digest

    add_repo(lock_path, matrix["protocol_lock_sha256"])
    add_repo(base.protocol_v2.canonical_grid_path_v2(result_root), matrix["grid_sha256"])
    for row in base.protocol_v2.locked_fit_inventory_v2():
        add_repo(base.protocol_v2.canonical_fit_path_v2(result_root, row), matrix["fit_sha256"][row["fit_id"]])
    for record in lock["tracked_files"].values():
        relative = base._safe_member_name(record["path"], label="locked tracked path")
        add_repo(root / relative, record["sha256"])
    summary = base._load_json_bytes(base._read_regular(result_root / "evaluation/summary.json", label="sealed evaluation summary"), label="sealed evaluation summary")
    for relative, digest in summary["output_sha256"].items():
        add_repo(result_root / base._safe_member_name(relative, label="sealed evaluation path"), digest)
    add_repo(result_root / "evaluation/summary.json", evaluation["evaluation_summary_sha256"])
    add_repo(result_root / base.evaluate_v2.REPLAY_RECEIPT_RELATIVE_PATH, evaluation["evaluation_replay_receipt_sha256"])
    for name, record in release["verified_members"].items():
        add_repo(root / name, record["sha256"])
    for relative, digest in (
        (base.PROTECTED_PDF_RELATIVE, base.PROTECTED_PDF_SHA256),
        (base.PROTECTED_REPRO_RELATIVE, base.PROTECTED_REPRO_SHA256),
        (base.PROTECTED_SOURCE_RELATIVE, base.PROTECTED_SOURCE_SHA256),
        (Path("iclr_paper/tex/numbers.tex"), base.PROTECTED_NUMBERS_SHA256),
    ):
        bindings["release/" + relative.as_posix()] = digest
    for relative in (SOURCE_RELATIVE, TEST_RELATIVE):
        digest = base._sha256(base._read_regular(root / relative, label="amendment source/test"))
        if relative == SOURCE_RELATIVE and digest != _IMPORTED_SOURCE_SHA256:
            raise RuntimeError("amendment source changed since import")
        add_repo(root / relative, digest)
    bindings = dict(sorted(bindings.items()))
    _assert_sealed_files_unchanged(root, bindings)
    return bindings


def build_delta_compat_audit(repo_root: Path = ROOT) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Read-only full build using frozen authenticators and comparison helpers."""
    root = Path(repo_root).resolve()
    inventory = base.protocol_v2.locked_fit_inventory_v2()
    mapping = base.validate_v1_fit_mapping(base.v1_fit_mapping(), inventory)
    expected_fit_ids = [str(row["fit_id"]) for row in mapping]
    required_members = base._required_v1_lineage_members(mapping)
    release, snapshots = base.verify_protected_v1_release(root, required_members)
    if len(required_members) != 55 or release.get("verified_member_count") != 55 or set(snapshots) != set(required_members):
        raise RuntimeError("compatibility audit requires all 55 protected snapshots")
    matrix, payloads = base.authenticate_complete_v2_matrix(root)
    if set(payloads) != set(expected_fit_ids):
        raise RuntimeError("authenticated fit payload coverage differs")
    evaluation_auth, normalized_rows, consumers = base.authenticate_v2_evaluation_commit(root, matrix)
    bindings = _sealed_file_bindings(root, release, matrix, evaluation_auth)
    # Only after authenticating both unmodified lineages may views be derived.
    comparison_snapshots, comparison_consumers, amendment = derive_comparison_views(snapshots, consumers)
    v1_records = [base.extract_v1_fit_record(row, snapshots[str(row["source_member"])]) for row in mapping]
    fit_rows = [
        base.compare_fit_record(record, payloads[record["fit_id"]], v2_sha256=matrix["fit_sha256"][record["fit_id"]])
        for record in v1_records
    ]
    fit_audit = base._summarize_fit_rows(fit_rows, expected_fit_ids=expected_fit_ids)
    fit_audit["explicit_unserialized_fit_gaps"] = base.explicit_legacy_fit_gaps([row for row in fit_rows if row["weights_comparable"] is False])
    evaluation_summary, metric_rows, winner_rows = base.compare_protected_evaluation(v1_records, normalized_rows, expected_fit_ids=expected_fit_ids)
    evaluation_audit = {**evaluation_summary, "winner_changed_rows": None, "metric_rows": metric_rows, "winner_rows": winner_rows}
    consumer_summary, numeric_rows, consumer_rows = base.compare_protected_consumers(comparison_snapshots, comparison_consumers)
    consumer_audit = {**consumer_summary, "consumers": consumer_rows, "numeric_rows": numeric_rows}
    numbers = base._read_regular(root.parent / "iclr_paper/tex/numbers.tex", label="protected manuscript numbers")
    if base._sha256(numbers) != base.PROTECTED_NUMBERS_SHA256:
        raise RuntimeError("protected manuscript changed during compatibility build")
    macros = consumers.get("manuscript_macros")
    provenance = consumers.get("provenance")
    if type(macros) is not dict or not macros or type(provenance) is not dict:
        raise RuntimeError("primary-v2 macro mapping differs")
    macro_summary, macro_rows = base.classify_protected_manuscript_values(base.parse_numbers_tex(numbers), macros, provenance.get("macro_coordinates"))
    csv_bytes = base.serialize_delta_csv(metric_rows).encode("utf-8")
    payload = base.assemble_delta_payload(
        release_audit={**release, "required_lineage_member_count": len(required_members), "fit_mapping_count": len(mapping), "paper_consumer_count": len(base.evaluate_v2.PAPER_CONSUMER_CONTRACT_V2)},
        authenticated_inputs={"v2_protocol_lock_sha256": matrix["protocol_lock_sha256"], "v2_training_grid_sha256": matrix["grid_sha256"], "v2_fit_count": matrix["fit_count"], "v2_fit_sha256": matrix["fit_sha256"], "v2_evaluation": evaluation_auth},
        semantic_correction={"semantics_profile": "approval-set-first-occurrence-v2", "approval_set_normalization_changed": True, "deterministic_project_ordering_changed": True, "counterfactual_isolation": "not_identified"},
        fit_audit=fit_audit, evaluation_audit=evaluation_audit, consumer_audit=consumer_audit,
        manuscript_audit={**macro_summary, "rows": macro_rows}, csv_sha256=base._sha256(csv_bytes),
    )
    payload["delta_csv"]["relative_path"] = CSV_OUTPUT.as_posix()
    payload["compatibility_amendment"] = {
        "schema_version": 1,
        "classification": "append-only legacy representation compatibility amendment",
        "base_delta_module": "src/iclr_primary_warsaw_delta_audit_v2.py",
        "source_path": SOURCE_RELATIVE.as_posix(), "test_path": TEST_RELATIVE.as_posix(),
        "json_output": JSON_OUTPUT.as_posix(), "csv_output": CSV_OUTPUT.as_posix(),
        "sealed_file_sha256": bindings,
        **amendment,
    }
    _prepared_output(payload, metric_rows)
    _assert_sealed_files_unchanged(root, bindings)
    return payload, metric_rows


def _prepared_output(payload: Mapping[str, object], rows: list[dict[str, object]]) -> tuple[bytes, bytes, dict[str, object]]:
    sections = [payload.get(name) for name in ("fit_audit", "evaluation_audit", "consumer_audit", "manuscript_audit", "compatibility_amendment")]
    if any(type(section) is not dict for section in sections):
        raise RuntimeError("compatibility audit summary schema differs")
    fit, evaluation, consumer, manuscript, amendment = sections
    if (fit.get("total"), evaluation.get("metric_coordinates"), consumer.get("consumer_compared"), consumer.get("numeric_coordinates"), manuscript.get("protected_macro_total"), amendment.get("not_applicable_cell_count"), amendment.get("file_identity_mapping_count"), len(rows)) != (49, 245, 25, 7435, 589, 324, 132, 245):
        raise RuntimeError("compatibility audit complete summary coverage differs")
    csv_bytes = base.serialize_delta_csv(rows).encode("utf-8")
    if payload.get("delta_csv") != {"relative_path": CSV_OUTPUT.as_posix(), "sha256": base._sha256(csv_bytes)}:
        raise RuntimeError("compatibility payload does not bind its CSV")
    json_bytes = base._json_bytes(payload)
    summary = {
        "status": payload["status"], "fit_count": fit["total"],
        "evaluation_metric_coordinates": evaluation["metric_coordinates"],
        "winner_changed_rows": evaluation["winner_changed_rows"],
        "consumer_numeric_coordinates": consumer["numeric_coordinates"],
        "manuscript_value_count": manuscript["protected_macro_total"],
        "not_applicable_cell_count": amendment["not_applicable_cell_count"],
        "file_identity_mapping_count": amendment["file_identity_mapping_count"],
        "json_sha256": base._sha256(json_bytes), "csv_sha256": base._sha256(csv_bytes),
    }
    return json_bytes, csv_bytes, summary


def write_delta_compat_audit(repo_root: Path = ROOT) -> dict[str, object]:
    """Rebuild twice under one writer guard; preflight both; commit CSV then JSON."""
    root = Path(repo_root).resolve()
    result = root / base.V2_RESULT_RELATIVE
    with base.evaluate_v2.evaluation_writer_lock_v2(result):
        payload, rows = build_delta_compat_audit(root)
        json_bytes, csv_bytes, summary = _prepared_output(payload, rows)
        replay_payload, replay_rows = build_delta_compat_audit(root)
        replay_json, replay_csv, _ = _prepared_output(replay_payload, replay_rows)
        if (replay_json, replay_csv) != (json_bytes, csv_bytes):
            raise RuntimeError("independent compatibility audit recomputation changed output bytes")
        csv_path, json_path = result / CSV_OUTPUT, result / JSON_OUTPUT
        base.protocol_v2.preflight_immutable_bytes_artifact_v2(csv_path, csv_bytes, label="compatibility delta CSV")
        base.protocol_v2.preflight_immutable_bytes_artifact_v2(json_path, json_bytes, label="compatibility delta JSON")
        _assert_sealed_files_unchanged(root, payload["compatibility_amendment"]["sealed_file_sha256"])
        base.protocol_v2.write_immutable_bytes_artifact_v2(csv_path, csv_bytes, label="compatibility delta CSV")
        base.protocol_v2.write_immutable_bytes_artifact_v2(json_path, json_bytes, label="compatibility delta sealing JSON")
        # Summary validation and serialization finished before the first write.
        return summary


def check_delta_compat_audit(repo_root: Path = ROOT) -> dict[str, object]:
    """Independently rebuild and check both committed outputs without writing."""
    root = Path(repo_root).resolve()
    payload, rows = build_delta_compat_audit(root)
    expected_json, expected_csv, summary = _prepared_output(payload, rows)
    result = root / base.V2_RESULT_RELATIVE
    actual_csv = base._read_regular(result / CSV_OUTPUT, label="committed compatibility CSV")
    actual_json = base._read_regular(result / JSON_OUTPUT, label="committed compatibility JSON")
    if (actual_json, actual_csv) != (expected_json, expected_csv):
        raise RuntimeError("committed compatibility audit differs from independent recomputation")
    _assert_sealed_files_unchanged(root, payload["compatibility_amendment"]["sealed_file_sha256"])
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="immutably install the compatibility audit")
    group.add_argument("--check-only", action="store_true", help="check the committed compatibility audit")
    args = parser.parse_args(argv)
    result = write_delta_compat_audit(ROOT) if args.write else check_delta_compat_audit(ROOT)
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
