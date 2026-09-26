"""Authenticate and compare the multicity protocol-v1 and protocol-v2 lineages.

The v2 study corrects approval-ballot set semantics and also pins the later
deterministic project-order fix.  This audit therefore measures every fit and
every evaluation row without treating the observed movement as a
duplicate-only counterfactual.  It never changes either scientific lineage;
its two outputs are append-only post-run audit artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

import iclr_multicity_evaluate_v2 as evaluate_v2
import iclr_multicity_protocol_v2 as protocol_v2
from iclr_approval_semantics_v2 import (
    SEMANTICS_PROFILE,
    validate_manifest_receipt_identities_v2,
    validate_multicity_semantics_receipt,
)


ROOT = Path(__file__).resolve().parent.parent
V1_RESULT_RELATIVE = Path("results/iclr_multicity")
V2_RESULT_RELATIVE = Path("results/iclr_multicity_v2")
V1_RELEASE_RELATIVE = Path("output/artifact/ICLR2027_TempoPB_reproducibility.zip")
V1_RELEASE_SHA256 = "4adfa80db1467d8b7b4fd314699b3fbf69cc9057a640ef8816611e86500b3d20"

JSON_OUTPUT = Path("v1_v2_delta_audit.json")
CSV_OUTPUT = Path("v1_v2_delta_rows.csv")
FIXED_POLICIES = frozenset({"mes", "res-1.0", "greedy-count", "greedy-cost"})
METRIC_FIELDS = (
    "worst_csd",
    "mean_csd",
    "welfare",
    "cost_welfare",
    "exclusion",
)
UNCHANGED_METRIC_TOLERANCE = 1e-12
EPISODE_FIELDS = frozenset(
    {
        "series",
        "scored_years",
        "worst_csd",
        "worst_cohort",
        "mean_csd",
        "welfare",
        "cost_welfare",
        "exclusion",
        "years",
        "scored_outcomes",
    }
)
YEAR_FIELDS = frozenset(
    {
        "year",
        "scored",
        "n_winners",
        "spent",
        "welfare",
        "cost_welfare",
        "exclusion",
    }
)

V1_RELEASE_MEMBERS = (
    "results/iclr_multicity/protocol_lock.json",
    "results/iclr_multicity/heldout_opened.json",
    "results/iclr_multicity/evidence_decision.json",
    "results/iclr_multicity/evaluation/evidence_payload.json",
    "results/iclr_multicity/evaluation/per_series.csv",
    "results/iclr_multicity/evaluation/per_series.json",
    "results/iclr_multicity/evaluation/summary.json",
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise RuntimeError(f"non-finite JSON token: {value}")


def _reject_duplicate_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise RuntimeError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def _validate_json_tree(value: object, *, label: str) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise RuntimeError(f"{label} must contain only finite numbers")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_json_tree(child, label=f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if type(key) is not str:
                raise RuntimeError(f"{label} contains a non-string JSON key")
            _validate_json_tree(child, label=f"{label}.{key}")
        return
    raise RuntimeError(f"{label} contains a non-JSON value: {type(value).__name__}")


def _load_json_bytes(content: bytes, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(
            content.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    _validate_json_tree(payload, label=label)
    return payload


def _read_json(path: Path, *, label: str) -> tuple[dict[str, object], bytes]:
    content = protocol_v2.read_regular_bytes_artifact_v2(path, label=label)
    assert content is not None
    return _load_json_bytes(content, label=label), content


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise RuntimeError(f"{label} must be finite")
    return converted


def _weights(payload: Mapping[str, object], *, label: str) -> list[float]:
    raw = payload.get("best_weights")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(f"{label} best_weights must be a nonempty list")
    if any(type(value) is not float for value in raw):
        raise RuntimeError(f"{label} best_weights must contain exact JSON floats")
    return [
        _finite_number(value, label=f"{label} best_weights[{index}]")
        for index, value in enumerate(raw)
    ]


def compare_fit_payloads(
    coordinate: str,
    v1_payload: Mapping[str, object],
    v2_payload: Mapping[str, object],
    *,
    v1_sha256: str,
    v2_sha256: str,
) -> dict[str, object]:
    """Compare one authenticated fit while requiring the same experiment."""

    contract_fields = ("config", "arm", "execution_mode", "training_only")
    differing_contract = [
        field
        for field in contract_fields
        if not protocol_v2.exact_json_equal_v2(
            v1_payload.get(field),
            v2_payload.get(field),
        )
    ]
    if differing_contract:
        raise RuntimeError(
            f"fit contract differs for {coordinate}: {differing_contract}"
        )
    if (
        type(v1_payload.get("schema_version")) is not int
        or v1_payload.get("schema_version") != 1
        or type(v2_payload.get("schema_version")) is not int
        or v2_payload.get("schema_version") != 2
    ):
        raise RuntimeError(f"fit schema lineage differs for {coordinate}")

    result_v1 = v1_payload.get("result")
    result_v2 = v2_payload.get("result")
    if not isinstance(result_v1, Mapping) or not isinstance(result_v2, Mapping):
        raise RuntimeError(f"fit result is malformed for {coordinate}")
    weights_v1 = _weights(result_v1, label=f"v1 {coordinate}")
    weights_v2 = _weights(result_v2, label=f"v2 {coordinate}")
    if len(weights_v1) != len(weights_v2):
        raise RuntimeError(f"fit weight dimension differs for {coordinate}")

    for field in ("best_loss", "initial_loss"):
        if type(result_v1.get(field)) is not float or type(result_v2.get(field)) is not float:
            raise RuntimeError(
                f"fit {field} must be an exact JSON float for {coordinate}"
            )

    loss_v1 = _finite_number(result_v1.get("best_loss"), label="v1 best_loss")
    loss_v2 = _finite_number(result_v2.get("best_loss"), label="v2 best_loss")
    initial_v1 = _finite_number(
        result_v1.get("initial_loss"),
        label="v1 initial_loss",
    )
    initial_v2 = _finite_number(
        result_v2.get("initial_loss"),
        label="v2 initial_loss",
    )
    if (
        type(result_v1.get("n_evals")) is not int
        or type(result_v2.get("n_evals")) is not int
        or result_v1.get("n_evals") != result_v2.get("n_evals")
    ):
        raise RuntimeError(f"fit evaluation count differs for {coordinate}")
    if (
        type(result_v1.get("selection_source")) is not str
        or type(result_v2.get("selection_source")) is not str
        or result_v1.get("selection_source") != result_v2.get("selection_source")
    ):
        raise RuntimeError(f"fit selection source differs for {coordinate}")

    return {
        "coordinate": coordinate,
        "negative_control": coordinate.startswith("city_out_Poland_Warszawa/"),
        "v1_sha256": v1_sha256,
        "v2_sha256": v2_sha256,
        "weights_exact": weights_v1 == weights_v2,
        "weight_linf_delta": max(
            abs(value_v2 - value_v1)
            for value_v1, value_v2 in zip(weights_v1, weights_v2)
        ),
        "v1_best_loss": loss_v1,
        "v2_best_loss": loss_v2,
        "best_loss_delta_v2_minus_v1": loss_v2 - loss_v1,
        "initial_loss_delta_v2_minus_v1": initial_v2 - initial_v1,
        "n_evals": result_v2["n_evals"],
        "selection_source": result_v2["selection_source"],
    }


def _flatten_per_series(
    payload: Mapping[str, object],
    *,
    label: str,
) -> dict[tuple[str, str, str], Mapping[str, object]]:
    flattened: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for split, raw_policies in payload.items():
        if not isinstance(split, str) or not isinstance(raw_policies, Mapping):
            raise RuntimeError(f"{label} has malformed split coverage")
        for policy, raw_rows in raw_policies.items():
            if not isinstance(policy, str) or not isinstance(raw_rows, list):
                raise RuntimeError(f"{label} has malformed policy coverage")
            for row in raw_rows:
                if not isinstance(row, Mapping) or not isinstance(row.get("series"), str):
                    raise RuntimeError(f"{label} has a malformed per-series row")
                key = (split, policy, row["series"])
                if key in flattened:
                    raise RuntimeError(f"{label} contains duplicate coordinate {key}")
                flattened[key] = row
    return flattened


def _validate_episode(row: Mapping[str, object], *, label: str) -> None:
    if set(row) != EPISODE_FIELDS:
        raise RuntimeError(f"{label} row fields differ")
    if type(row.get("series")) is not str:
        raise RuntimeError(f"{label} series must be a string")
    scored_years = row.get("scored_years")
    if not isinstance(scored_years, list) or any(
        type(year) is not int for year in scored_years
    ):
        raise RuntimeError(f"{label} scored years must be exact JSON integers")
    if type(row.get("worst_cohort")) is not str:
        raise RuntimeError(f"{label} worst cohort must be a string")
    for field in METRIC_FIELDS:
        if type(row.get(field)) is not float:
            raise RuntimeError(f"{label} {field} must be an exact JSON float")
        _finite_number(row[field], label=f"{label} {field}")

    years = row.get("years")
    if not isinstance(years, list):
        raise RuntimeError(f"{label} years must be a list")
    for index, raw_year in enumerate(years):
        year_label = f"{label} years[{index}]"
        if not isinstance(raw_year, Mapping) or set(raw_year) != YEAR_FIELDS:
            raise RuntimeError(f"{year_label} fields differ")
        if type(raw_year.get("year")) is not int:
            raise RuntimeError(f"{year_label} year must be an exact JSON integer")
        if type(raw_year.get("scored")) is not bool:
            raise RuntimeError(f"{year_label} scored must be a boolean")
        if type(raw_year.get("n_winners")) is not int:
            raise RuntimeError(f"{year_label} n_winners must be an integer")
        for field in ("spent", "welfare", "cost_welfare", "exclusion"):
            if type(raw_year.get(field)) is not float:
                raise RuntimeError(f"{year_label} {field} must be an exact JSON float")
            _finite_number(raw_year[field], label=f"{year_label} {field}")

    outcomes = row.get("scored_outcomes")
    if not isinstance(outcomes, list):
        raise RuntimeError(f"{label} scored outcomes must be a list")
    for index, raw_outcome in enumerate(outcomes):
        outcome_label = f"{label} scored_outcomes[{index}]"
        if (
            not isinstance(raw_outcome, Mapping)
            or set(raw_outcome) != {"year", "winners"}
            or type(raw_outcome.get("year")) is not int
        ):
            raise RuntimeError(f"{outcome_label} fields or year type differ")
        winners = raw_outcome.get("winners")
        if (
            not isinstance(winners, list)
            or any(type(winner) is not str for winner in winners)
            or winners != sorted(set(winners))
        ):
            raise RuntimeError(f"{outcome_label} winners are noncanonical")
    if [outcome["year"] for outcome in outcomes] != scored_years:
        raise RuntimeError(f"{label} scored outcome years differ")


def _outcome_sha256(row: Mapping[str, object], *, label: str) -> str:
    outcomes = row.get("scored_outcomes")
    if not isinstance(outcomes, list):
        raise RuntimeError(f"{label} scored outcomes are malformed")
    encoded = json.dumps(
        outcomes,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(encoded)


def _is_wola_series(series: str) -> bool:
    return series == "Poland/Warszawa/Wola" or series.startswith(
        "Poland/Warszawa/Wola |"
    )


def compare_per_series_payloads(
    v1_payload: Mapping[str, object],
    v2_payload: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Compare every evaluation coordinate and enforce locality invariants."""

    flat_v1 = _flatten_per_series(v1_payload, label="v1 per-series evidence")
    flat_v2 = _flatten_per_series(v2_payload, label="v2 per-series evidence")
    if set(flat_v1) != set(flat_v2):
        missing = sorted(set(flat_v1) - set(flat_v2))
        extra = sorted(set(flat_v2) - set(flat_v1))
        raise RuntimeError(
            f"v1/v2 evaluation coordinates differ: missing={missing}, extra={extra}"
        )

    rows: list[dict[str, object]] = []
    max_metric_delta = {field: 0.0 for field in METRIC_FIELDS}
    fixed_outside = 0
    fixed_outside_winner_drift = 0
    same_winner_outside = 0
    winner_changed_coordinates = []
    row_exact_count = 0
    rows_by_split: dict[str, int] = {}

    for key in sorted(flat_v1):
        split, policy, series = key
        row_v1 = flat_v1[key]
        row_v2 = flat_v2[key]
        _validate_episode(row_v1, label=f"v1 {key}")
        _validate_episode(row_v2, label=f"v2 {key}")
        if not protocol_v2.exact_json_equal_v2(
            row_v1.get("scored_years"),
            row_v2.get("scored_years"),
        ):
            raise RuntimeError(f"scored years differ for {key}")
        winner_exact = protocol_v2.exact_json_equal_v2(
            row_v1.get("scored_outcomes"),
            row_v2.get("scored_outcomes"),
        )
        row_exact = protocol_v2.exact_json_equal_v2(row_v1, row_v2)
        if row_exact:
            row_exact_count += 1
        if not winner_exact:
            winner_changed_coordinates.append(
                {"split": split, "policy": policy, "series": series}
            )

        fixed_policy = policy in FIXED_POLICIES
        wola_series = _is_wola_series(series)
        if fixed_policy and not wola_series:
            fixed_outside += 1
            if not winner_exact:
                fixed_outside_winner_drift += 1
                raise RuntimeError(
                    f"fixed-policy winner drift outside Wola for {key}"
                )

        metric_deltas = {}
        for field in METRIC_FIELDS:
            value_v1 = _finite_number(
                row_v1.get(field),
                label=f"v1 {key} {field}",
            )
            value_v2 = _finite_number(
                row_v2.get(field),
                label=f"v2 {key} {field}",
            )
            if type(row_v1.get(field)) is not type(row_v2.get(field)):
                raise RuntimeError(f"metric JSON type differs for {key}: {field}")
            delta = value_v2 - value_v1
            metric_deltas[field] = delta
            max_metric_delta[field] = max(max_metric_delta[field], abs(delta))

        if winner_exact and not wola_series:
            same_winner_outside += 1
            if metric_deltas["welfare"] != 0 or metric_deltas["cost_welfare"] != 0:
                raise RuntimeError(f"same-winner welfare drift outside Wola for {key}")
            for field in ("worst_csd", "mean_csd", "exclusion"):
                if abs(metric_deltas[field]) > UNCHANGED_METRIC_TOLERANCE:
                    raise RuntimeError(
                        f"same-winner metric drift outside Wola for {key}: "
                        f"{field}={metric_deltas[field]}"
                    )

        rows_by_split[split] = rows_by_split.get(split, 0) + 1
        rows.append(
            {
                "split": split,
                "policy": policy,
                "series": series,
                "fixed_policy": fixed_policy,
                "wola_series": wola_series,
                "winner_exact": winner_exact,
                "row_exact": row_exact,
                "worst_cohort_exact": row_v1.get("worst_cohort")
                == row_v2.get("worst_cohort"),
                **{
                    f"{field}_delta_v2_minus_v1": metric_deltas[field]
                    for field in METRIC_FIELDS
                },
                "v1_outcome_sha256": _outcome_sha256(row_v1, label=f"v1 {key}"),
                "v2_outcome_sha256": _outcome_sha256(row_v2, label=f"v2 {key}"),
            }
        )

    summary = {
        "total_rows": len(rows),
        "rows_by_split": dict(sorted(rows_by_split.items())),
        "row_exact": row_exact_count,
        "row_changed": len(rows) - row_exact_count,
        "winner_exact_rows": len(rows) - len(winner_changed_coordinates),
        "winner_changed_rows": len(winner_changed_coordinates),
        "winner_changed_coordinates": winner_changed_coordinates,
        "fixed_policy_rows": sum(row["fixed_policy"] for row in rows),
        "fixed_policy_rows_outside_wola": fixed_outside,
        "fixed_policy_winner_drift_outside_wola": fixed_outside_winner_drift,
        "same_winner_rows_outside_wola": same_winner_outside,
        "same_winner_metric_tolerance": UNCHANGED_METRIC_TOLERANCE,
        "max_absolute_metric_delta": max_metric_delta,
    }
    return summary, rows


def _numeric_leaf_deltas(
    v1_value: object,
    v2_value: object,
    *,
    prefix: str = "",
) -> list[tuple[str, float]]:
    if type(v1_value) is not type(v2_value):
        raise RuntimeError(f"observed schema differs at {prefix or '<root>'}")
    if isinstance(v1_value, Mapping):
        if not isinstance(v2_value, Mapping) or set(v1_value) != set(v2_value):
            raise RuntimeError(f"observed schema differs at {prefix or '<root>'}")
        rows = []
        for key in sorted(v1_value):
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_numeric_leaf_deltas(v1_value[key], v2_value[key], prefix=child))
        return rows
    if isinstance(v1_value, list):
        if not isinstance(v2_value, list) or len(v1_value) != len(v2_value):
            raise RuntimeError(f"observed schema differs at {prefix or '<root>'}")
        rows = []
        for index, (child_v1, child_v2) in enumerate(zip(v1_value, v2_value)):
            rows.extend(
                _numeric_leaf_deltas(
                    child_v1,
                    child_v2,
                    prefix=f"{prefix}[{index}]",
                )
            )
        return rows
    if (
        not isinstance(v1_value, bool)
        and not isinstance(v2_value, bool)
        and isinstance(v1_value, (int, float))
        and isinstance(v2_value, (int, float))
    ):
        return [(prefix, float(v2_value) - float(v1_value))]
    if not protocol_v2.exact_json_equal_v2(v1_value, v2_value):
        raise RuntimeError(f"observed nonnumeric value differs at {prefix or '<root>'}")
    return []


def compare_decision_payloads(
    v1_payload: Mapping[str, object],
    v2_payload: Mapping[str, object],
) -> dict[str, object]:
    """Compare decision classifications and every named pass/fail condition."""

    expected_v1_fields = {
        "classification",
        "conditions",
        "gold_pass",
        "primary_seed",
    }
    if set(v1_payload) != expected_v1_fields:
        raise RuntimeError("v1 decision fields differ")
    if set(v2_payload) != expected_v1_fields | {"provenance"}:
        raise RuntimeError("v2 decision fields differ")
    for label, payload in (("v1", v1_payload), ("v2", v2_payload)):
        if payload.get("classification") not in {"gold", "silver", "falsified"}:
            raise RuntimeError(f"{label} decision classification is invalid")
        if type(payload.get("gold_pass")) is not bool:
            raise RuntimeError(f"{label} gold_pass must be a boolean")
        if type(payload.get("primary_seed")) is not str:
            raise RuntimeError(f"{label} primary_seed must be a string")
    if (
        v1_payload.get("primary_seed") != "42"
        or v2_payload.get("primary_seed") != "42"
    ):
        raise RuntimeError("v1/v2 primary_seed differs from frozen seed 42")

    conditions_v1 = v1_payload.get("conditions")
    conditions_v2 = v2_payload.get("conditions")
    if not isinstance(conditions_v1, Mapping) or not isinstance(
        conditions_v2, Mapping
    ):
        raise RuntimeError("v1/v2 decision conditions are malformed")
    if set(conditions_v1) != set(conditions_v2):
        raise RuntimeError("v1/v2 decision condition coverage differs")
    changed_pass_status = []
    for name in sorted(conditions_v1):
        row_v1 = conditions_v1[name]
        row_v2 = conditions_v2[name]
        if not isinstance(row_v1, Mapping) or not isinstance(row_v2, Mapping):
            raise RuntimeError(f"decision condition is malformed: {name}")
        if set(row_v1) != {"passed", "observed", "threshold"} or set(row_v2) != {
            "passed",
            "observed",
            "threshold",
        }:
            raise RuntimeError(f"decision condition fields differ: {name}")
        if type(row_v1.get("passed")) is not bool or type(row_v2.get("passed")) is not bool:
            raise RuntimeError(f"decision condition pass flag is malformed: {name}")
        if not protocol_v2.exact_json_equal_v2(
            row_v1.get("threshold"),
            row_v2.get("threshold"),
        ):
            raise RuntimeError(f"decision condition threshold differs: {name}")
        try:
            _numeric_leaf_deltas(
                row_v1.get("observed"),
                row_v2.get("observed"),
                prefix=f"conditions.{name}.observed",
            )
        except RuntimeError as exc:
            raise RuntimeError(f"observed schema differs for {name}: {exc}") from exc
        if row_v1["passed"] != row_v2["passed"]:
            changed_pass_status.append(name)

    v1_macro = conditions_v1["temporal_macro_at_most_minus_0_010"]["observed"]
    v2_macro = conditions_v2["temporal_macro_at_most_minus_0_010"]["observed"]
    numeric_deltas = []
    for name in sorted(conditions_v1):
        numeric_deltas.extend(
            _numeric_leaf_deltas(
                conditions_v1[name]["observed"],
                conditions_v2[name]["observed"],
                prefix=f"conditions.{name}.observed",
            )
        )
    return {
        "v1_classification": v1_payload.get("classification"),
        "v2_classification": v2_payload.get("classification"),
        "classification_changed": v1_payload.get("classification")
        != v2_payload.get("classification"),
        "v1_gold_pass": v1_payload.get("gold_pass"),
        "v2_gold_pass": v2_payload.get("gold_pass"),
        "condition_count": len(conditions_v1),
        "condition_pass_status_changed": changed_pass_status,
        "v1_primary_temporal_macro": _finite_number(
            v1_macro,
            label="v1 primary temporal macro",
        ),
        "v2_primary_temporal_macro": _finite_number(
            v2_macro,
            label="v2 primary temporal macro",
        ),
        "primary_temporal_macro_delta_v2_minus_v1": float(v2_macro)
        - float(v1_macro),
        "max_absolute_numeric_observation_delta": max(
            (abs(delta) for _, delta in numeric_deltas),
            default=0.0,
        ),
    }


def serialize_delta_csv(rows: Sequence[Mapping[str, object]]) -> str:
    """Serialize all per-series comparisons in deterministic coordinate order."""

    fields = (
        "split",
        "policy",
        "series",
        "fixed_policy",
        "wola_series",
        "winner_exact",
        "row_exact",
        "worst_cohort_exact",
        "worst_csd_delta_v2_minus_v1",
        "mean_csd_delta_v2_minus_v1",
        "welfare_delta_v2_minus_v1",
        "cost_welfare_delta_v2_minus_v1",
        "exclusion_delta_v2_minus_v1",
        "v1_outcome_sha256",
        "v2_outcome_sha256",
    )
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=lambda item: (item["split"], item["policy"], item["series"])):
        if set(row) != set(fields):
            raise RuntimeError("delta CSV row schema differs")
        writer.writerow(row)
    return handle.getvalue()


def _verify_v1_release(
    repo_root: Path,
    expected_local: Mapping[str, bytes],
) -> dict[str, object]:
    archive_path = repo_root.parent / V1_RELEASE_RELATIVE
    archive_bytes = protocol_v2.read_regular_bytes_artifact_v2(
        archive_path,
        label="protected v1 reproducibility archive",
    )
    assert archive_bytes is not None
    archive_sha256 = _sha256(archive_bytes)
    if archive_sha256 != V1_RELEASE_SHA256:
        raise RuntimeError(
            "protected v1 reproducibility archive digest differs: "
            f"{archive_sha256} != {V1_RELEASE_SHA256}"
        )
    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("protected v1 archive contains duplicate member names")
        for name in V1_RELEASE_MEMBERS:
            if name not in expected_local:
                raise RuntimeError(f"missing local comparison snapshot for {name}")
            try:
                released = archive.read(name)
            except KeyError as exc:
                raise RuntimeError(f"protected v1 archive is missing {name}") from exc
            if released != expected_local[name]:
                raise RuntimeError(f"local v1 artifact differs from protected release: {name}")
    return {
        "archive_sha256": archive_sha256,
        "verified_members": list(V1_RELEASE_MEMBERS),
    }


def _v1_fit_path(repo_root: Path, label: str) -> tuple[Path, str]:
    pure = PurePosixPath(label)
    prefix = PurePosixPath("results/iclr_multicity/fits")
    try:
        relative = pure.relative_to(prefix)
    except ValueError as exc:
        raise RuntimeError(f"v1 fit label lies outside its result root: {label}") from exc
    if not relative.parts or ".." in relative.parts:
        raise RuntimeError(f"v1 fit label is malformed: {label}")
    return repo_root.joinpath(*pure.parts), relative.as_posix()


def _recompute_v2_outputs(
    *,
    root: Path,
    v2_root: Path,
    data_dir: Path,
    v2_lock: protocol_v2.ProtocolLockSnapshotV2,
    semantics: Mapping[str, object],
    semantics_bytes: bytes,
    v2_opening: Mapping[str, object],
    v2_opening_bytes: bytes,
    v2_snapshots: Sequence[evaluate_v2.VerifiedFitSnapshotV2],
) -> dict[str, str]:
    """Recompute all five official outputs from authenticated in-memory inputs."""

    structural_path = v2_root / "structural_gates.json"
    structural_bytes = protocol_v2.read_regular_bytes_artifact_v2(
        structural_path,
        label="v2 structural gate",
    )
    assert structural_bytes is not None
    structural_sha256 = _sha256(structural_bytes)
    if v2_opening.get("structural_gates_sha256") != structural_sha256:
        raise RuntimeError("v2 opening receipt does not bind the structural gate")
    structural = protocol_v2.validate_structural_gates_v2(
        structural_path,
        semantics_receipt_sha256=_sha256(semantics_bytes),
        corpus_semantic_sha256=str(semantics["corpus_semantic_sha256"]),
        repo_root=root,
    )
    input_snapshot = protocol_v2.load_protocol_inputs_snapshot_v2(
        v2_lock.payload,
        v2_root,
        data_dir,
        protocol_v2.SPLIT_NAMES,
    )
    validate_manifest_receipt_identities_v2(input_snapshot.index, semantics)
    provenance = {
        "protocol_lock_sha256": v2_lock.sha256,
        "semantics_profile": SEMANTICS_PROFILE,
        "semantics_receipt_sha256": _sha256(semantics_bytes),
        "corpus_semantic_sha256": semantics["corpus_semantic_sha256"],
        "heldout_receipt_sha256": _sha256(v2_opening_bytes),
    }
    decision, evidence, summaries, per_series, evaluated = (
        evaluate_v2._compute_verified_evaluation_v2(
            input_snapshot.index,
            input_snapshot.splits,
            v2_snapshots,
            structural.get("status") == "pass",
            v2_lock.payload["evidence_gates"],
            semantics,
            provenance=provenance,
        )
    )
    prepared = evaluate_v2.prepare_evaluation_outputs_v2(
        v2_root,
        decision=decision,
        evidence=evidence,
        summaries=summaries,
        per_series_payload=per_series,
        evaluated=evaluated,
    )
    output_sha256 = {}
    for path, expected_bytes in prepared.artifacts:
        observed = protocol_v2.read_regular_bytes_artifact_v2(
            path,
            label="official protocol-v2 evaluation output",
        )
        if observed != expected_bytes:
            raise RuntimeError(
                "v2 output differs from authenticated recomputation: "
                f"{path.relative_to(v2_root)}"
            )
        output_sha256[path.relative_to(v2_root).as_posix()] = _sha256(
            expected_bytes
        )

    protocol_v2.assert_protocol_inputs_unchanged_v2(input_snapshot)
    evaluate_v2.assert_fit_snapshots_unchanged_v2(v2_snapshots)
    evaluate_v2.assert_protocol_lock_snapshot_unchanged_v2(
        v2_lock,
        repo_root=root,
        result_root=v2_root,
        data_dir=data_dir,
    )
    if (
        protocol_v2.read_regular_bytes_artifact_v2(
            structural_path,
            label="v2 structural gate",
        )
        != structural_bytes
        or protocol_v2.read_regular_bytes_artifact_v2(
            v2_root / "approval_semantics_receipt.json",
            label="v2 semantics receipt",
        )
        != semantics_bytes
        or protocol_v2.read_regular_bytes_artifact_v2(
            v2_root / "heldout_opened.json",
            label="v2 held-out opening receipt",
        )
        != v2_opening_bytes
    ):
        raise RuntimeError("v2 lineage changed during output recomputation")
    return dict(sorted(output_sha256.items()))


def build_delta_audit(
    repo_root: Path = ROOT,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Authenticate both lineages and construct the deterministic delta audit."""

    root = Path(repo_root).resolve()
    v1_root = root / V1_RESULT_RELATIVE
    v2_root = protocol_v2.canonical_result_root_v2(root / V2_RESULT_RELATIVE)
    data_dir = root / "data" / "pb_multicity"

    v2_lock = protocol_v2.verify_multicity_protocol_lock_snapshot_v2(
        v2_root / "protocol_lock.json",
        root,
        v2_root,
        data_dir,
    )
    evaluate_v2.verify_lock_fit_inventory_v2(v2_lock.payload)

    semantics, semantics_bytes = _read_json(
        v2_root / "approval_semantics_receipt.json",
        label="v2 approval semantics receipt",
    )
    validate_multicity_semantics_receipt(semantics)
    corpus_semantic_sha256 = semantics.get("corpus_semantic_sha256")
    if not isinstance(corpus_semantic_sha256, str):
        raise RuntimeError("v2 corpus semantic digest is malformed")

    v2_inventory = evaluate_v2.required_fit_inventory_v2(v2_root)
    v2_snapshots = evaluate_v2.verify_fit_inventory_v2(
        v2_inventory,
        v2_lock.payload,
        v2_lock.sha256,
        corpus_semantic_sha256=corpus_semantic_sha256,
        result_root=v2_root,
        repo_root=root,
    )
    v2_fit_by_coordinate = {}
    for snapshot in v2_snapshots:
        relative = snapshot.path.relative_to(v2_root / "fits").as_posix()
        v2_fit_by_coordinate[relative] = snapshot

    v2_opening, v2_opening_bytes = _read_json(
        v2_root / "heldout_opened.json",
        label="v2 held-out opening receipt",
    )
    expected_v2_fit_hashes = {
        snapshot.label: snapshot.sha256 for snapshot in v2_snapshots
    }
    if (
        v2_opening.get("lock_sha256") != v2_lock.sha256
        or v2_opening.get("semantics_profile") != SEMANTICS_PROFILE
        or v2_opening.get("semantics_receipt_sha256") != _sha256(semantics_bytes)
        or v2_opening.get("corpus_semantic_sha256") != corpus_semantic_sha256
        or v2_opening.get("fit_sha256") != expected_v2_fit_hashes
    ):
        raise RuntimeError("v2 opening receipt does not bind the verified lineage")

    recomputed_v2_output_sha256 = _recompute_v2_outputs(
        root=root,
        v2_root=v2_root,
        data_dir=data_dir,
        v2_lock=v2_lock,
        semantics=semantics,
        semantics_bytes=semantics_bytes,
        v2_opening=v2_opening,
        v2_opening_bytes=v2_opening_bytes,
        v2_snapshots=v2_snapshots,
    )

    v1_lock, v1_lock_bytes = _read_json(
        v1_root / "protocol_lock.json",
        label="v1 protocol lock",
    )
    v1_opening, v1_opening_bytes = _read_json(
        v1_root / "heldout_opened.json",
        label="v1 held-out opening receipt",
    )
    if _sha256(v1_lock_bytes) != v2_lock.payload.get("parent_protocol_lock_sha256"):
        raise RuntimeError("v1 protocol lock differs from the v2 parent binding")
    if _sha256(v1_opening_bytes) != v2_lock.payload.get("parent_heldout_receipt_sha256"):
        raise RuntimeError("v1 opening receipt differs from the v2 parent binding")
    if v1_opening.get("lock_sha256") != _sha256(v1_lock_bytes):
        raise RuntimeError("v1 opening receipt does not bind the v1 lock")

    raw_v1_fit_hashes = v1_opening.get("fit_sha256")
    if not isinstance(raw_v1_fit_hashes, Mapping) or len(raw_v1_fit_hashes) != 24:
        raise RuntimeError("v1 opening receipt does not contain exactly 24 fits")
    v1_fit_by_coordinate = {}
    for label, expected_sha256 in raw_v1_fit_hashes.items():
        if not isinstance(label, str) or not isinstance(expected_sha256, str):
            raise RuntimeError("v1 opening fit inventory is malformed")
        path, coordinate = _v1_fit_path(root, label)
        payload, content = _read_json(path, label=f"v1 fit {coordinate}")
        observed_sha256 = _sha256(content)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(f"v1 fit differs from opening receipt: {coordinate}")
        if coordinate in v1_fit_by_coordinate:
            raise RuntimeError(f"duplicate v1 fit coordinate: {coordinate}")
        v1_fit_by_coordinate[coordinate] = (payload, observed_sha256)
    if set(v1_fit_by_coordinate) != set(v2_fit_by_coordinate):
        raise RuntimeError("v1/v2 fit coordinate inventories differ")

    output_names = (
        "evidence_decision.json",
        "evaluation/evidence_payload.json",
        "evaluation/per_series.json",
        "evaluation/summary.json",
    )
    v1_outputs = {}
    v2_outputs = {}
    v1_output_bytes = {}
    v2_output_bytes = {}
    for relative in output_names:
        payload_v1, content_v1 = _read_json(
            v1_root / relative,
            label=f"v1 {relative}",
        )
        payload_v2, content_v2 = _read_json(
            v2_root / relative,
            label=f"v2 {relative}",
        )
        v1_outputs[relative] = payload_v1
        v2_outputs[relative] = payload_v2
        v1_output_bytes[relative] = content_v1
        v2_output_bytes[relative] = content_v2

    for result_root, output_bytes, label in (
        (v1_root, v1_output_bytes, "v1"),
        (v2_root, v2_output_bytes, "v2"),
    ):
        csv_relative = "evaluation/per_series.csv"
        content = protocol_v2.read_regular_bytes_artifact_v2(
            result_root / csv_relative,
            label=f"{label} {csv_relative}",
        )
        assert content is not None
        output_bytes[csv_relative] = content

    observed_v2_output_sha256 = {
        relative: _sha256(content)
        for relative, content in sorted(v2_output_bytes.items())
    }
    if observed_v2_output_sha256 != recomputed_v2_output_sha256:
        raise RuntimeError("v2 output hashes differ from authenticated recomputation")

    release_snapshots = {
        "results/iclr_multicity/protocol_lock.json": v1_lock_bytes,
        "results/iclr_multicity/heldout_opened.json": v1_opening_bytes,
        **{
            f"results/iclr_multicity/{relative}": content
            for relative, content in v1_output_bytes.items()
        },
    }
    release_audit = _verify_v1_release(root, release_snapshots)

    expected_v2_provenance = {
        "protocol_lock_sha256": v2_lock.sha256,
        "semantics_profile": SEMANTICS_PROFILE,
        "semantics_receipt_sha256": _sha256(semantics_bytes),
        "corpus_semantic_sha256": corpus_semantic_sha256,
        "heldout_receipt_sha256": _sha256(v2_opening_bytes),
    }
    for relative in ("evidence_decision.json", "evaluation/evidence_payload.json"):
        if not protocol_v2.exact_json_equal_v2(
            v2_outputs[relative].get("provenance"),
            expected_v2_provenance,
        ):
            raise RuntimeError(f"v2 output provenance differs: {relative}")

    fit_rows = []
    for coordinate in sorted(v1_fit_by_coordinate):
        payload_v1, sha_v1 = v1_fit_by_coordinate[coordinate]
        snapshot_v2 = v2_fit_by_coordinate[coordinate]
        fit_rows.append(
            compare_fit_payloads(
                coordinate,
                payload_v1,
                snapshot_v2.payload,
                v1_sha256=sha_v1,
                v2_sha256=snapshot_v2.sha256,
            )
        )
    negative_controls = [row for row in fit_rows if row["negative_control"]]
    fit_audit = {
        "total": len(fit_rows),
        "weights_exact": sum(row["weights_exact"] for row in fit_rows),
        "changed_weights": sum(not row["weights_exact"] for row in fit_rows),
        "negative_control_total": len(negative_controls),
        "negative_control_changed_weights": sum(
            not row["weights_exact"] for row in negative_controls
        ),
        "max_weight_linf_delta": max(row["weight_linf_delta"] for row in fit_rows),
        "max_absolute_best_loss_delta": max(
            abs(row["best_loss_delta_v2_minus_v1"]) for row in fit_rows
        ),
        "fits": fit_rows,
    }
    if fit_audit["total"] != 24 or fit_audit["negative_control_total"] != 5:
        raise RuntimeError("v1/v2 fit audit does not cover the exact 24-fit matrix")

    evaluation_audit, delta_rows = compare_per_series_payloads(
        v1_outputs["evaluation/per_series.json"],
        v2_outputs["evaluation/per_series.json"],
    )
    decision_audit = compare_decision_payloads(
        v1_outputs["evidence_decision.json"],
        v2_outputs["evidence_decision.json"],
    )
    expected_rows_by_split = {
        "temporal_2022": 975,
        "city_out_Poland_Warszawa": 162,
        "city_out_Poland_Gdynia": 189,
        "city_out_Poland_Łódź": 324,
    }
    if (
        evaluation_audit["total_rows"] != 1650
        or evaluation_audit["rows_by_split"] != dict(
            sorted(expected_rows_by_split.items())
        )
        or evaluation_audit["fixed_policy_rows"] != 600
        or evaluation_audit["fixed_policy_rows_outside_wola"] != 592
        or evaluation_audit["fixed_policy_winner_drift_outside_wola"] != 0
    ):
        raise RuntimeError("v1/v2 evaluation audit coverage differs from 1,650 rows")
    if decision_audit["condition_count"] != 9:
        raise RuntimeError("v1/v2 decision audit does not cover all nine conditions")

    tracked_v1 = v1_lock.get("tracked_files")
    tracked_v2 = v2_lock.payload.get("tracked_files")
    if not isinstance(tracked_v1, Mapping) or not isinstance(tracked_v2, Mapping):
        raise RuntimeError("protocol lock source inventories are malformed")
    cohorts_v1 = tracked_v1.get("source/cohorts.py")
    cohorts_v2 = tracked_v2.get("source/cohorts.py")
    if not isinstance(cohorts_v1, Mapping) or not isinstance(cohorts_v2, Mapping):
        raise RuntimeError("protocol locks do not bind cohorts.py")
    cohorts_v1_sha = cohorts_v1.get("sha256")
    cohorts_v2_sha = cohorts_v2.get("sha256")
    if not isinstance(cohorts_v1_sha, str) or not isinstance(cohorts_v2_sha, str):
        raise RuntimeError("cohorts.py lock digests are malformed")

    csv_text = serialize_delta_csv(delta_rows)
    payload = {
        "schema_version": 1,
        "status": "pass",
        "interpretation": "complete_corrected_lineage_rerun",
        "causal_attribution": {
            "duplicate_only_attribution_allowed": False,
            "reason": (
                "The v2 lineage jointly changes approval-set normalization and "
                "the locked deterministic project-order source; two of five "
                "negative-control fit weights also change."
            ),
            "v1_cohorts_sha256": cohorts_v1_sha,
            "v2_cohorts_sha256": cohorts_v2_sha,
            "cohorts_source_changed": cohorts_v1_sha != cohorts_v2_sha,
        },
        "semantic_correction": {
            "semantics_profile": semantics.get("semantics_profile"),
            "corpus_semantic_sha256": corpus_semantic_sha256,
            "counts": semantics.get("counts"),
            "anomalies": semantics.get("anomalies"),
        },
        "authenticated_inputs": {
            "v1_release": release_audit,
            "v1_protocol_lock_sha256": _sha256(v1_lock_bytes),
            "v1_heldout_receipt_sha256": _sha256(v1_opening_bytes),
            "v2_protocol_lock_sha256": v2_lock.sha256,
            "v2_heldout_receipt_sha256": _sha256(v2_opening_bytes),
            "v2_semantics_receipt_sha256": _sha256(semantics_bytes),
            "v2_recomputed_output_sha256": recomputed_v2_output_sha256,
            "v1_output_sha256": {
                relative: _sha256(content)
                for relative, content in sorted(v1_output_bytes.items())
            },
            "v2_output_sha256": {
                relative: _sha256(content)
                for relative, content in sorted(v2_output_bytes.items())
            },
        },
        "fit_audit": fit_audit,
        "evaluation_audit": evaluation_audit,
        "decision_audit": decision_audit,
        "artifacts": {
            "delta_rows_csv": CSV_OUTPUT.as_posix(),
            "delta_rows_csv_sha256": _sha256(csv_text.encode("utf-8")),
        },
    }
    json.dumps(payload, allow_nan=False)
    return payload, delta_rows


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _install_audit_pair(
    *,
    json_path: Path,
    json_bytes: bytes,
    csv_path: Path,
    csv_bytes: bytes,
) -> tuple[str, str]:
    """Install the referenced CSV before the JSON commit record."""

    csv_sha256 = protocol_v2.write_immutable_bytes_artifact_v2(
        csv_path,
        csv_bytes,
        label="multicity v1-v2 delta rows",
    )
    json_sha256 = protocol_v2.write_immutable_bytes_artifact_v2(
        json_path,
        json_bytes,
        label="multicity v1-v2 delta audit",
    )
    return json_sha256, csv_sha256


def write_delta_audit(repo_root: Path = ROOT) -> dict[str, object]:
    """Build twice, then immutably install the deterministic audit pair."""

    root = Path(repo_root).resolve()
    result = protocol_v2.canonical_result_root_v2(root / V2_RESULT_RELATIVE)
    payload, rows = build_delta_audit(root)
    csv_bytes = serialize_delta_csv(rows).encode("utf-8")
    json_bytes = _json_bytes(payload)
    json_path = protocol_v2.canonical_result_output_path_v2(result, JSON_OUTPUT)
    csv_path = protocol_v2.canonical_result_output_path_v2(result, CSV_OUTPUT)
    protocol_v2.preflight_immutable_bytes_artifact_v2(
        json_path,
        json_bytes,
        label="multicity v1-v2 delta audit",
    )
    protocol_v2.preflight_immutable_bytes_artifact_v2(
        csv_path,
        csv_bytes,
        label="multicity v1-v2 delta rows",
    )

    final_payload, final_rows = build_delta_audit(root)
    if _json_bytes(final_payload) != json_bytes or serialize_delta_csv(
        final_rows
    ).encode("utf-8") != csv_bytes:
        raise RuntimeError("multicity v1-v2 audit inputs changed before installation")
    json_sha256, csv_sha256 = _install_audit_pair(
        json_path=json_path,
        json_bytes=json_bytes,
        csv_path=csv_path,
        csv_bytes=csv_bytes,
    )
    return {
        "status": payload["status"],
        "classification": payload["decision_audit"]["v2_classification"],
        "json_sha256": json_sha256,
        "csv_sha256": csv_sha256,
        "fit_count": payload["fit_audit"]["total"],
        "evaluation_rows": payload["evaluation_audit"]["total_rows"],
        "winner_changed_rows": payload["evaluation_audit"][
            "winner_changed_rows"
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="authenticate and compare without writing audit artifacts",
    )
    args = parser.parse_args(argv)
    if args.check_only:
        payload, rows = build_delta_audit(args.repo_root)
        result = {
            "status": payload["status"],
            "classification": payload["decision_audit"]["v2_classification"],
            "fit_count": payload["fit_audit"]["total"],
            "evaluation_rows": len(rows),
            "winner_changed_rows": payload["evaluation_audit"][
                "winner_changed_rows"
            ],
        }
    else:
        result = write_delta_audit(args.repo_root)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
