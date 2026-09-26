"""Strict contracts for the primary-Warsaw protocol-v1/v2 delta audit."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _audit():
    return importlib.import_module("iclr_primary_warsaw_delta_audit_v2")


def _v2_fit(
    fit_id: str,
    weights: list[float],
    *,
    loss: float = 0.2,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "fit_id": fit_id,
        "config": {"fit_id": fit_id, "family": "outcome_frontier"},
        "training_only": True,
        "execution_mode": "serial",
        "arm": {
            "name": "outcome",
            "feature_names": [f"x{index}" for index in range(len(weights))],
            "init_name": "res",
        },
        "provenance": {"protocol_lock_sha256": "a" * 64},
        "result": {
            "best_loss": loss,
            "optimizer_best_loss": loss,
            "optimizer_best_weights": list(weights),
            "selected_weights": list(weights),
            "selection_source": "cmaes",
            "n_objective_evals": 200,
            "optimizer_config": {},
        },
    }


def _v1_fit_record(
    fit_id: str,
    weights: list[float] | None,
    *,
    loss: float | None = 0.3,
) -> dict[str, object]:
    return {
        "fit_id": fit_id,
        "family": "outcome_frontier",
        "source_member": "results/v1.json",
        "source_sha256": "b" * 64,
        "weight_status": (
            "recorded" if weights is not None else "not_recorded_in_protected_release"
        ),
        "weights": copy.deepcopy(weights),
        "train_objective": loss,
        "recorded_test_metrics": {},
    }


def _episode(
    *,
    fit_id: str = "fit/a",
    winners: tuple[str, ...] = ("p1",),
    worst_csd: float = 0.2,
) -> dict[str, object]:
    return {
        "fit_id": fit_id,
        "split": "temporal_2022",
        "series": "Poland/Warszawa/Bemowo",
        "scored_years": [2023],
        "worst_cohort": "age25-39|F",
        "worst_csd": worst_csd,
        "mean_csd": 0.1,
        "welfare": 100.0,
        "cost_welfare": 200.0,
        "exclusion": 0.05,
        "scored_outcomes": [{"year": 2023, "winners": list(winners)}],
    }


def test_strict_parsers_reject_duplicate_keys_nonfinite_and_duplicate_csv_headers() -> None:
    audit = _audit()

    with pytest.raises(RuntimeError, match="duplicate JSON key"):
        audit._load_json_bytes(b'{"x": 1, "x": 2}', label="duplicate")
    with pytest.raises(RuntimeError, match="finite"):
        audit._load_json_bytes(b'{"x": 1e309}', label="overflow")
    with pytest.raises(RuntimeError, match="duplicate CSV header"):
        audit._load_csv_bytes(b"x,x\n1,2\n", label="duplicate header")


def test_v1_mapping_covers_exact_locked_inventory_and_discloses_missing_weights() -> None:
    audit = _audit()
    protocol = importlib.import_module("iclr_primary_warsaw_protocol_v2")

    mapping = audit.v1_fit_mapping()
    inventory = protocol.locked_fit_inventory_v2()
    audit.validate_v1_fit_mapping(mapping, inventory)

    assert len(mapping) == 49
    assert [row["fit_id"] for row in mapping] == sorted(
        row["fit_id"] for row in inventory
    )
    unavailable = [row for row in mapping if row["weight_path"] is None]
    assert len(unavailable) == 7
    assert {row["family"] for row in unavailable} == {
        "outcome_loo",
        "static_support_floor",
    }


def test_v1_record_extraction_uses_exact_selector_and_preserves_absent_weights() -> None:
    audit = _audit()
    json_mapping = {
        "fit_id": "outcome_frontier/temporal_2022/target-1/seed-42",
        "family": "outcome_frontier",
        "source_member": "results/frontier.json",
        "selector": {
            "kind": "json_list_match",
            "container_path": ["points"],
            "field": "floor",
            "value": 1.0,
        },
        "weight_path": ["weights"],
        "train_objective_path": ["train_worst_csd"],
        "recorded_test_metric_paths": {"worst_csd": ["test_worst_csd"]},
    }
    content = json.dumps(
        {
            "points": [
                {
                    "floor": 1.0,
                    "weights": [0.1, 0.2],
                    "train_worst_csd": 0.3,
                    "test_worst_csd": 0.4,
                }
            ]
        }
    ).encode()

    record = audit.extract_v1_fit_record(json_mapping, content)

    assert record["weights"] == [0.1, 0.2]
    assert record["train_objective"] == 0.3
    assert record["recorded_test_metrics"] == {"worst_csd": 0.4}
    assert record["source_sha256"] == hashlib.sha256(content).hexdigest()

    csv_mapping = {
        **json_mapping,
        "fit_id": "static_support_floor/temporal_2022/kappa-1/seed-42",
        "family": "static_support_floor",
        "source_member": "results/identification.csv",
        "selector": {"kind": "csv_row", "field": "kappa", "value": "1.0"},
        "weight_path": None,
        "train_objective_path": ["train_worst_csd"],
        "recorded_test_metric_paths": {"worst_csd": ["test_worst_csd"]},
    }
    absent = audit.extract_v1_fit_record(
        csv_mapping,
        b"kappa,train_worst_csd,test_worst_csd\n1.0,0.11,0.12\n",
    )
    assert absent["weight_status"] == "not_recorded_in_protected_release"
    assert absent["weights"] is None
    assert absent["train_objective"] == 0.11


def test_v1_record_extraction_rejects_ambiguous_selectors() -> None:
    audit = _audit()
    mapping = {
        "fit_id": "fit/a",
        "family": "outcome_frontier",
        "source_member": "results/frontier.json",
        "selector": {
            "kind": "json_list_match",
            "container_path": ["points"],
            "field": "floor",
            "value": 1.0,
        },
        "weight_path": ["weights"],
        "train_objective_path": ["train_worst_csd"],
        "recorded_test_metric_paths": {},
    }
    content = json.dumps(
        {
            "points": [
                {"floor": 1.0, "weights": [0.1], "train_worst_csd": 0.2},
                {"floor": 1.0, "weights": [0.2], "train_worst_csd": 0.3},
            ]
        }
    ).encode()

    with pytest.raises(RuntimeError, match="exactly one"):
        audit.extract_v1_fit_record(mapping, content)


def test_fit_comparison_reports_exact_weight_and_loss_deltas_without_inventing_weights() -> None:
    audit = _audit()
    fit_id = "outcome_frontier/temporal_2022/target-1/seed-42"
    row = audit.compare_fit_record(
        _v1_fit_record(fit_id, [0.0, 0.5], loss=0.3),
        _v2_fit(fit_id, [0.25, 0.5], loss=0.2),
        v2_sha256="c" * 64,
    )

    assert row["weights_comparable"] is True
    assert row["weights_exact"] is False
    assert row["weight_linf_delta"] == 0.25
    assert row["train_objective_delta_v2_minus_v1"] == pytest.approx(-0.1)

    unavailable = audit.compare_fit_record(
        _v1_fit_record(fit_id, None, loss=None),
        _v2_fit(fit_id, [0.25, 0.5], loss=0.2),
        v2_sha256="c" * 64,
    )
    assert unavailable["weights_comparable"] is False
    assert unavailable["weights_exact"] is None
    assert unavailable["weight_linf_delta"] is None
    assert unavailable["train_objective_delta_v2_minus_v1"] is None


def test_fit_comparison_rejects_identity_dimension_and_json_type_aliases() -> None:
    audit = _audit()
    fit_id = "outcome_frontier/temporal_2022/target-1/seed-42"
    v1 = _v1_fit_record(fit_id, [0.0])
    v2 = _v2_fit(fit_id, [0.0, 1.0])

    with pytest.raises(RuntimeError, match="dimension"):
        audit.compare_fit_record(v1, v2, v2_sha256="c" * 64)
    v2 = _v2_fit("different", [0.0])
    with pytest.raises(RuntimeError, match="identity"):
        audit.compare_fit_record(v1, v2, v2_sha256="c" * 64)
    v2 = _v2_fit(fit_id, [0.0])
    v2["result"]["selected_weights"] = [0]
    with pytest.raises(RuntimeError, match="exact JSON floats"):
        audit.compare_fit_record(v1, v2, v2_sha256="c" * 64)


def test_complete_v2_matrix_is_reread_hash_checked_and_schema_validated(
    monkeypatch, tmp_path: Path
) -> None:
    audit = _audit()
    protocol = importlib.import_module("iclr_primary_warsaw_protocol_v2")
    train = importlib.import_module("iclr_primary_warsaw_train_v2")
    inventory = protocol.locked_fit_inventory_v2()
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    grid_path = protocol.canonical_grid_path_v2(result_root)
    grid_path.parent.mkdir(parents=True)
    grid_bytes = b'{"grid":true}\n'
    grid_path.write_bytes(grid_bytes)
    payloads: dict[str, dict[str, object]] = {}
    fit_sha: dict[str, str] = {}
    for spec in inventory:
        fit_id = str(spec["fit_id"])
        payload = {"fit_id": fit_id, "result": {"selected_weights": [0.0]}}
        content = (json.dumps(payload, sort_keys=True) + "\n").encode()
        path = protocol.canonical_fit_path_v2(result_root, spec)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        payloads[fit_id] = payload
        fit_sha[fit_id] = hashlib.sha256(content).hexdigest()

    lock_sha = "a" * 64
    monkeypatch.setattr(
        train,
        "verify_complete_matrix_v2",
        lambda **kwargs: {
            "schema_version": 2,
            "status": "complete",
            "protocol_lock_sha256": lock_sha,
            "grid_sha256": hashlib.sha256(grid_bytes).hexdigest(),
            "fit_count": 49,
            "fit_sha256": fit_sha,
        },
    )
    monkeypatch.setattr(
        protocol,
        "verify_primary_warsaw_protocol_lock_snapshot_v2",
        lambda *args, **kwargs: SimpleNamespace(sha256=lock_sha),
    )
    validated: list[str] = []

    def validate(payload, *, expected_spec, expected_protocol_lock_sha256):
        assert expected_protocol_lock_sha256 == lock_sha
        assert payload == payloads[str(expected_spec["fit_id"])]
        validated.append(str(expected_spec["fit_id"]))
        return payload

    monkeypatch.setattr(train, "validate_fit_payload_v2", validate)

    authenticated, reread = audit.authenticate_complete_v2_matrix(tmp_path)

    assert authenticated["fit_count"] == 49
    assert authenticated["grid_sha256"] == hashlib.sha256(grid_bytes).hexdigest()
    assert set(reread) == set(fit_sha)
    assert validated == sorted(fit_sha)


def test_evaluation_comparison_reports_every_winner_and_metric_change() -> None:
    audit = _audit()
    v1 = [_episode()]
    v2 = [
        _episode(
            winners=("p2",),
            worst_csd=0.15,
        )
    ]

    summary, rows = audit.compare_evaluation_rows(v1, v2)

    assert summary == {
        "total_rows": 1,
        "winner_changed_rows": 1,
        "metric_changed_rows": 1,
        "max_absolute_metric_delta": {
            "worst_csd": pytest.approx(0.05),
            "mean_csd": 0.0,
            "welfare": 0.0,
            "cost_welfare": 0.0,
            "exclusion": 0.0,
        },
    }
    assert rows[0]["winner_exact"] is False
    assert rows[0]["worst_csd_delta_v2_minus_v1"] == pytest.approx(-0.05)


def test_evaluation_comparison_rejects_missing_coordinates_and_noncanonical_winners() -> None:
    audit = _audit()
    with pytest.raises(RuntimeError, match="coordinate coverage"):
        audit.compare_evaluation_rows([_episode()], [])

    invalid = _episode(winners=("p2", "p1", "p1"))
    with pytest.raises(RuntimeError, match="winners are noncanonical"):
        audit.compare_evaluation_rows([invalid], [invalid])


def test_v2_evaluation_commit_hash_binds_receipt_rows_consumers_and_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    audit = _audit()
    evaluator = importlib.import_module("iclr_primary_warsaw_evaluate_v2")
    result_root = tmp_path / "results" / "iclr_primary_warsaw_v2"
    evaluation_root = result_root / "evaluation"
    evaluation_root.mkdir(parents=True)
    fit_sha = {
        str(row["fit_id"]): "b" * 64
        for row in importlib.import_module(
            "iclr_primary_warsaw_protocol_v2"
        ).locked_fit_inventory_v2()
    }
    matrix = {
        "schema_version": 2,
        "status": "complete",
        "protocol_lock_sha256": "a" * 64,
        "grid_sha256": "c" * 64,
        "fit_count": 49,
        "fit_sha256": fit_sha,
    }
    receipt_payload = {
        "schema_version": 2,
        "event": "evaluation replay started",
        "classification": "corrected post-hoc replay",
        "heldout_outcomes_already_known": True,
        "fresh_holdout": False,
        "preregistered": False,
        "disclosure": "Held-out outcomes were already known.",
        "protocol_lock_sha256": "a" * 64,
        "semantics_profile": "approval-set-first-occurrence-v2",
        "semantics_receipt_sha256": "d" * 64,
        "corpus_semantic_sha256": "e" * 64,
        "structural_gates_sha256": "f" * 64,
        "training_grid_sha256": "c" * 64,
        "fit_count": 49,
        "fit_sha256": fit_sha,
    }
    receipt_bytes = audit._json_bytes(receipt_payload)
    (result_root / "evaluation_replay_started.json").write_bytes(receipt_bytes)

    row = {
        "fit_id": next(iter(fit_sha)),
        "source_kind": "fit",
        "split": "temporal_2022",
        "view": "test",
        "scheme": "age_sex",
        "series": "Poland/Test/Unit",
        "scored_years": [2023],
        "year_outcomes": [
            {
                "year": 2023,
                "scored": True,
                "winners": ["1"],
                "spent": 1.0,
                "welfare": 2.0,
                "cost_welfare": 3.0,
                "exclusion": 0.0,
            }
        ],
        "metrics": {
            "worst_csd": 0.1,
            "worst_cohort": "age18-39|F",
            "mean_csd": 0.05,
            "welfare": 2.0,
            "cost_welfare": 3.0,
            "exclusion": 0.0,
        },
    }
    per_series = {
        "schema_version": 2,
        "semantics_profile": "approval-set-first-occurrence-v2",
        "rows": [row],
        "provenance": {"protocol_lock_sha256": "a" * 64},
    }
    consumers = {
        "schema_version": 2,
        "classification": "corrected post-hoc replay",
        "consumer_ids": ["synthetic"],
        "consumers": {"synthetic": {"values": {"x": "1"}}},
        "manuscript_macros": {"Alpha": "1.0000"},
        "provenance": {"protocol_lock_sha256": "a" * 64},
    }
    monkeypatch.setattr(
        evaluator, "validate_per_series_payload_v2", lambda payload, **kwargs: payload
    )
    monkeypatch.setattr(
        evaluator,
        "validate_paper_consumers_payload_v2",
        lambda payload, **kwargs: payload,
    )
    monkeypatch.setattr(
        evaluator,
        "PAPER_CONSUMER_CONTRACT_V2",
        {"synthetic": {"legacy_paths": ("results/synthetic.csv",)}},
    )
    content = {
        "evaluation/per_series.json": audit._json_bytes(per_series),
        "evaluation/per_series.csv": evaluator._per_series_csv_bytes_v2(per_series),
        "evaluation/paper_consumers.json": audit._json_bytes(consumers),
    }
    manifest = {
        "schema_version": 2,
        "classification": "corrected post-hoc replay",
        "files": {
            name: {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
            for name, value in sorted(content.items())
        },
        "provenance": {
            "protocol_lock_sha256": "a" * 64,
            "replay_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "training_grid_sha256": "c" * 64,
            "fit_sha256": fit_sha,
        },
    }
    content["evaluation/artifact_manifest.json"] = audit._json_bytes(manifest)
    summary = {
        "schema_version": 2,
        "status": "complete",
        "classification": "corrected post-hoc replay",
        "successful_process_exit_required_before_consumption": True,
        "fit_count": 49,
        "consumer_count": 1,
        "per_series_row_count": 1,
        "protocol_lock_sha256": "a" * 64,
        "replay_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "training_grid_sha256": "c" * 64,
        "fit_sha256": fit_sha,
        "output_sha256": {
            name: hashlib.sha256(value).hexdigest()
            for name, value in sorted(content.items())
        },
    }
    content["evaluation/summary.json"] = audit._json_bytes(summary)
    for relative, encoded in content.items():
        (result_root / relative).write_bytes(encoded)

    authenticated, rows, paper = audit.authenticate_v2_evaluation_commit(
        tmp_path, matrix
    )

    assert authenticated["evaluation_summary_sha256"] == hashlib.sha256(
        content["evaluation/summary.json"]
    ).hexdigest()
    assert rows == [row]
    assert paper["manuscript_macros"] == {"Alpha": "1.0000"}


def test_protected_evaluation_audit_compares_aggregates_and_discloses_winner_gap() -> None:
    audit = _audit()
    fit_id = "fit/a"
    v1 = _v1_fit_record(fit_id, [0.0])
    v1["recorded_test_metrics"] = {"worst_csd": 0.2, "welfare": 30.0}

    def row(series: str, worst: float, welfare: float, year: int) -> dict[str, object]:
        return {
            "fit_id": fit_id,
            "source_kind": "fit",
            "split": "temporal_2022",
            "view": "test",
            "scheme": "age_sex",
            "series": series,
            "scored_years": [year],
            "year_outcomes": [
                {
                    "year": year,
                    "scored": True,
                    "winners": [f"p{year}"],
                    "spent": 1.0,
                    "welfare": welfare,
                    "cost_welfare": 3.0,
                    "exclusion": 0.1,
                }
            ],
            "metrics": {
                "worst_csd": worst,
                "worst_cohort": "age18-39|F",
                "mean_csd": worst / 2,
                "welfare": welfare,
                "cost_welfare": 3.0,
                "exclusion": 0.1,
            },
        }

    summary, metric_rows, winner_rows = audit.compare_protected_evaluation(
        [v1],
        [row("series/b", 0.3, 20.0, 2024), row("series/a", 0.1, 10.0, 2023)],
        expected_fit_ids=[fit_id],
    )

    assert summary == {
        "fit_count": 1,
        "metric_coordinates": 5,
        "metric_comparable": 2,
        "metric_unavailable": 3,
        "metric_changed": 0,
        "winner_coordinates": 2,
        "winner_comparable": 0,
        "winner_unavailable": 2,
        "protected_winner_trace_status": "not_recorded_in_protected_release",
    }
    by_metric = {row["metric"]: row for row in metric_rows}
    assert by_metric["worst_csd"]["v2_value"] == pytest.approx(0.2)
    assert by_metric["worst_csd"]["delta_v2_minus_v1"] == pytest.approx(0.0)
    assert by_metric["welfare"]["v2_value"] == 30.0
    assert by_metric["mean_csd"]["status"] == "not_recorded_in_protected_release"
    assert [(row["series"], row["year"]) for row in winner_rows] == [
        ("series/a", 2023),
        ("series/b", 2024),
    ]
    assert all(row["v1_winners"] is None for row in winner_rows)
    assert all(row["winner_exact"] is None for row in winner_rows)


def test_manuscript_value_comparison_is_exact_and_complete() -> None:
    audit = _audit()
    summary, rows = audit.compare_manuscript_values(
        {"LearnedHeadCSD": "0.2000", "SeedSpread": "0.0100"},
        {"LearnedHeadCSD": "0.1900", "SeedSpread": "0.0100"},
    )

    assert summary == {"total": 2, "changed": 1, "exact": 1}
    assert [row["name"] for row in rows] == ["LearnedHeadCSD", "SeedSpread"]
    assert rows[0]["changed"] is True

    with pytest.raises(RuntimeError, match="coverage"):
        audit.compare_manuscript_values({"A": "1"}, {"B": "1"})


def test_typed_consumer_audit_rejects_missing_coverage_and_generic_values(
    monkeypatch,
) -> None:
    """Break caught: a protected consumer is omitted or remains a summary bag."""

    audit = _audit()
    legacy_contract = {
        "outcome.ablation": {"legacy_paths": ("results/iclr_ablation.csv",)},
        "transfer.lodz": {"legacy_paths": ("results/iclr_transfer.csv",)},
    }
    value_contract = {
        "outcome.ablation": {
            "schema": "outcome-ablation-table-v2",
            "lists": {"ablation_rows": 1},
        },
        "transfer.lodz": {
            "schema": "lodz-transfer-table-v2",
            "lists": {"transfer_rows": 1},
        },
    }
    monkeypatch.setattr(audit.evaluate_v2, "PAPER_CONSUMER_CONTRACT_V2", legacy_contract)
    monkeypatch.setattr(
        audit.evaluate_v2, "PAPER_CONSUMER_VALUE_CONTRACT_V2", value_contract
    )
    monkeypatch.setattr(
        audit,
        "CONSUMER_CSV_COMPARISON_SPECS",
        {
            consumer_id: audit.CONSUMER_CSV_COMPARISON_SPECS[consumer_id]
            for consumer_id in legacy_contract
        },
    )
    monkeypatch.setattr(audit, "CONSUMER_SPECIAL_COMPARISON_PATHS", {})

    complete = {
        "consumer_ids": ["outcome.ablation", "transfer.lodz"],
        "consumers": {
            "outcome.ablation": {
                "consumer_id": "outcome.ablation",
                "legacy_paths": ["results/iclr_ablation.csv"],
                "values": {
                    "schema": "outcome-ablation-table-v2",
                    "row_count": 1,
                    "ablation_rows": [
                        {
                            "ablated": "none (full model)",
                            "source_id": "fit/full",
                            "test_worst_csd": 0.2,
                            "test_welfare": 10.0,
                            "delta_vs_full": 0.0,
                        }
                    ],
                },
            },
            "transfer.lodz": {
                "consumer_id": "transfer.lodz",
                "legacy_paths": ["results/iclr_transfer.csv"],
                "values": {
                    "schema": "lodz-transfer-table-v2",
                    "row_count": 1,
                    "transfer_rows": [
                        {
                            "series": "Lodz",
                            "policy": "mes",
                            "source_id": "mes",
                            "worst_csd": 0.3,
                            "welfare": 11.0,
                            "exclusion": 0.1,
                            "worst_cohort": "age60+|F",
                        }
                    ],
                },
            },
        },
    }
    snapshots = {
        "results/iclr_ablation.csv": (
            b"ablated,test_worst_csd,test_welfare,delta_vs_full\n"
            b"none (full model),0.2,10.0,0.0\n"
        ),
        "results/iclr_transfer.csv": (
            b"series,policy,worst_csd,welfare,exclusion,worst_cohort\n"
            b"Lodz,mes,0.3,11.0,0.1,age60+|F\n"
        ),
    }

    summary, numeric_rows, consumer_rows = audit.compare_protected_consumers(
        snapshots, complete
    )
    assert summary["consumer_total"] == 2
    assert summary["consumer_compared"] == 2
    assert summary["numeric_coordinates"] == 6
    assert len(numeric_rows) == 6
    assert {row["consumer_id"] for row in consumer_rows} == {
        "outcome.ablation",
        "transfer.lodz",
    }

    missing = copy.deepcopy(complete)
    missing["consumer_ids"].remove("transfer.lodz")
    missing["consumers"].pop("transfer.lodz")
    with pytest.raises(RuntimeError, match="consumer coverage"):
        audit.compare_protected_consumers(snapshots, missing)

    generic = copy.deepcopy(complete)
    generic["consumers"]["outcome.ablation"]["values"] = {
        "rows": [],
        "summary": {},
    }
    with pytest.raises(RuntimeError, match="typed|schema"):
        audit.compare_protected_consumers(snapshots, generic)


def test_consumer_comparison_registry_covers_all_25_consumers_and_51_path_uses() -> None:
    audit = _audit()

    assert audit.validate_consumer_comparison_registry() == {
        "consumer_count": 25,
        "legacy_path_association_count": 51,
    }


def test_protected_ratio_parser_accepts_released_bare_form_and_rejects_bad_denominator() -> None:
    audit = _audit()

    assert audit.parse_protected_ratio("13/18") == (13, 18)
    assert audit.parse_protected_ratio(r"13/18 (72.2\%)") == (13, 18)
    with pytest.raises(RuntimeError, match="ratio"):
        audit.parse_protected_ratio("13 / 18")
    with pytest.raises(RuntimeError, match="denominator"):
        audit.parse_protected_ratio("13/0")
    with pytest.raises(RuntimeError, match="numerator"):
        audit.parse_protected_ratio("19/18")


def test_typed_consumer_numeric_mismatch_is_reported_at_exact_coordinate(
    monkeypatch,
) -> None:
    """Break caught: changed specialized values pass as a generic success count."""

    audit = _audit()
    monkeypatch.setattr(
        audit.evaluate_v2,
        "PAPER_CONSUMER_CONTRACT_V2",
        {"outcome.ablation": {"legacy_paths": ("results/iclr_ablation.csv",)}},
    )
    monkeypatch.setattr(
        audit.evaluate_v2,
        "PAPER_CONSUMER_VALUE_CONTRACT_V2",
        {
            "outcome.ablation": {
                "schema": "outcome-ablation-table-v2",
                "lists": {"ablation_rows": 1},
            }
        },
    )
    monkeypatch.setattr(
        audit,
        "CONSUMER_CSV_COMPARISON_SPECS",
        {"outcome.ablation": audit.CONSUMER_CSV_COMPARISON_SPECS["outcome.ablation"]},
    )
    monkeypatch.setattr(audit, "CONSUMER_SPECIAL_COMPARISON_PATHS", {})
    consumers = {
        "consumer_ids": ["outcome.ablation"],
        "consumers": {
            "outcome.ablation": {
                "consumer_id": "outcome.ablation",
                "legacy_paths": ["results/iclr_ablation.csv"],
                "values": {
                    "schema": "outcome-ablation-table-v2",
                    "row_count": 1,
                    "ablation_rows": [
                        {
                            "ablated": "none (full model)",
                            "source_id": "fit/full",
                            "test_worst_csd": 0.19,
                            "test_welfare": 10.0,
                            "delta_vs_full": 0.0,
                        }
                    ],
                },
            }
        },
    }
    snapshots = {
        "results/iclr_ablation.csv": (
            b"ablated,test_worst_csd,test_welfare,delta_vs_full\n"
            b"none (full model),0.2,10.0,0.0\n"
        )
    }

    summary, rows, _ = audit.compare_protected_consumers(snapshots, consumers)

    changed = [row for row in rows if row["changed"]]
    assert summary["numeric_changed"] == 1
    assert changed == [
        {
            "consumer_id": "outcome.ablation",
            "artifact_schema": "outcome-ablation-table-v2",
            "source_member": "results/iclr_ablation.csv",
            "table": "ablation_rows",
            "coordinate": "ablated=none (full model)",
            "field": "test_worst_csd",
            "v1_value": 0.2,
            "v2_value": 0.19,
            "delta_v2_minus_v1": pytest.approx(-0.01),
            "changed": True,
        }
    ]


def test_full_protected_macro_inventory_has_no_silent_exclusions() -> None:
    """Break caught: v1-only macros disappear instead of receiving a reason."""

    audit = _audit()
    protected = {
        "LearnedHeadCSD": "0.2000",
        "ExternalCorpusN": "42",
        "PrimaryLegacyOnly": "7",
    }
    corrected = {"LearnedHeadCSD": "0.1900"}
    coordinates = {
        "LearnedHeadCSD": {
            "consumer_id": "outcome.per_series",
            "coordinate": "headline",
        }
    }

    summary, rows = audit.classify_protected_manuscript_values(
        protected, corrected, coordinates
    )

    assert summary == {
        "protected_macro_total": 3,
        "compared": 1,
        "compared_changed": 1,
        "compared_exact": 0,
        "unaffected_non_primary": 1,
        "unavailable": 1,
    }
    assert [row["name"] for row in rows] == sorted(protected)
    assert {row["classification"] for row in rows} == {
        "compared",
        "unaffected_non_primary",
        "unavailable",
    }
    assert all(type(row["reason"]) is str and row["reason"] for row in rows)

    with pytest.raises(RuntimeError, match="macro coordinate coverage"):
        audit.classify_protected_manuscript_values(protected, corrected, {})


def test_seven_unserialized_legacy_fits_have_explicit_weight_and_winner_reasons() -> None:
    audit = _audit()
    unavailable = [
        {
            "fit_id": str(row["fit_id"]),
            "family": str(row["family"]),
            "weights_comparable": False,
        }
        for row in audit.v1_fit_mapping()
        if row["weight_path"] is None
    ]

    gaps = audit.explicit_legacy_fit_gaps(unavailable)

    assert len(gaps) == 7
    assert {row["family"] for row in gaps} == {
        "outcome_loo",
        "static_support_floor",
    }
    assert all(
        row["weight_status"] == "not_recorded_in_protected_release"
        and row["winner_trace_status"] == "not_recorded_in_protected_release"
        and row["reason"]
        for row in gaps
    )


def test_numbers_tex_parser_rejects_duplicates_and_unparsed_commands() -> None:
    audit = _audit()
    assert audit.parse_numbers_tex(
        b"% generated\n\\newcommand{\\Alpha}{1.0}\n\\newcommand{\\Beta}{x}\n"
    ) == {"Alpha": "1.0", "Beta": "x"}

    with pytest.raises(RuntimeError, match="duplicate manuscript macro"):
        audit.parse_numbers_tex(
            b"\\newcommand{\\Alpha}{1}\n\\newcommand{\\Alpha}{2}\n"
        )
    with pytest.raises(RuntimeError, match="unparsed manuscript macro"):
        audit.parse_numbers_tex(b"\\newcommand{broken}\n")


def test_protected_repository_release_authenticates_pdf_zip_and_v1_members() -> None:
    audit = _audit()
    repo_root = Path(__file__).resolve().parent.parent
    members = sorted({row["source_member"] for row in audit.v1_fit_mapping()})

    verified, snapshots = audit.verify_protected_v1_release(repo_root, members)

    assert verified["pdf_sha256"] == audit.PROTECTED_PDF_SHA256
    assert verified["reproducibility_zip_sha256"] == audit.PROTECTED_REPRO_SHA256
    assert verified["source_zip_sha256"] == audit.PROTECTED_SOURCE_SHA256
    assert verified["verified_member_count"] == len(members)
    assert set(snapshots) == set(members)
    assert all(hashlib.sha256(snapshots[name]).hexdigest() for name in snapshots)


def test_delta_csv_serialization_is_deterministic_and_schema_strict() -> None:
    audit = _audit()
    row_a = audit.compare_evaluation_rows([_episode(fit_id="fit/a")], [_episode(fit_id="fit/a")])[1][0]
    row_b = audit.compare_evaluation_rows([_episode(fit_id="fit/b")], [_episode(fit_id="fit/b")])[1][0]

    encoded = audit.serialize_delta_csv([row_b, row_a])

    assert encoded.count("\n") == 3
    assert encoded.splitlines()[1].startswith("fit/a,")
    assert encoded.splitlines()[2].startswith("fit/b,")
    malformed = dict(row_a)
    malformed["unexpected"] = True
    with pytest.raises(RuntimeError, match="row schema"):
        audit.serialize_delta_csv([malformed])


def test_delta_csv_serializes_protected_metric_availability_rows() -> None:
    audit = _audit()
    base = {
        "fit_id": "fit/b",
        "family": "outcome_frontier",
        "split": "temporal_2022",
        "metric": "welfare",
        "status": "comparable",
        "v1_value": 1.0,
        "v2_value": 2.0,
        "delta_v2_minus_v1": 1.0,
        "changed": True,
    }
    other = {**base, "fit_id": "fit/a", "status": "not_recorded_in_protected_release", "v1_value": None, "delta_v2_minus_v1": None, "changed": None}

    encoded = audit.serialize_delta_csv([base, other])

    assert encoded.splitlines()[0].startswith("fit_id,family,split,metric,status")
    assert encoded.splitlines()[1].startswith("fit/a,")
    assert encoded.splitlines()[2].startswith("fit/b,")


def test_install_pair_writes_csv_before_sealing_json(monkeypatch, tmp_path: Path) -> None:
    audit = _audit()
    calls: list[str] = []

    def record(path, content, *, label):
        calls.append(Path(path).name)
        return hashlib.sha256(content).hexdigest()

    monkeypatch.setattr(audit.protocol_v2, "write_immutable_bytes_artifact_v2", record)
    json_sha, csv_sha = audit._install_audit_pair(
        json_path=tmp_path / "audit.json",
        json_bytes=b"json",
        csv_path=tmp_path / "rows.csv",
        csv_bytes=b"csv",
    )

    assert calls == ["rows.csv", "audit.json"]
    assert json_sha == hashlib.sha256(b"json").hexdigest()
    assert csv_sha == hashlib.sha256(b"csv").hexdigest()


def test_write_path_recomputes_before_immutable_install(monkeypatch, tmp_path: Path) -> None:
    audit = _audit()
    calls: list[Path] = []
    payload = {
        "schema_version": 1,
        "status": "pass",
        "causal_attribution": {"duplicate_only_attribution_allowed": False},
        "fit_audit": {"total": 49},
        "evaluation_audit": {"total_rows": 1, "winner_changed_rows": 0},
        "consumer_audit": {"numeric_coordinates": 1},
        "manuscript_audit": {"protected_macro_total": 1, "compared_changed": 0},
    }

    def build(root):
        calls.append(Path(root))
        return copy.deepcopy(payload), []

    monkeypatch.setattr(audit, "build_delta_audit", build)
    monkeypatch.setattr(audit, "serialize_delta_csv", lambda rows: "csv")
    monkeypatch.setattr(audit, "_json_bytes", lambda value: b"json")
    monkeypatch.setattr(
        audit.protocol_v2,
        "preflight_immutable_bytes_artifact_v2",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        audit,
        "_install_audit_pair",
        lambda **kwargs: ("a" * 64, "b" * 64),
    )

    result = audit.write_delta_audit(tmp_path)

    assert calls == [tmp_path.resolve(), tmp_path.resolve()]
    assert result["status"] == "pass"
    assert result["fit_count"] == 49


def test_full_builder_assembles_all_authenticated_delta_layers(
    monkeypatch, tmp_path: Path
) -> None:
    audit = _audit()
    mapping = audit.v1_fit_mapping()
    fit_ids = [str(row["fit_id"]) for row in mapping]
    required_members = {
        str(row["source_member"]) for row in mapping
    } | {
        path
        for consumer in audit.evaluate_v2.PAPER_CONSUMER_CONTRACT_V2.values()
        for path in consumer["legacy_paths"]
    }
    snapshots = {member: b"protected" for member in required_members}
    release = {"pdf_sha256": "1" * 64, "verified_member_count": len(snapshots)}
    matrix = {
        "schema_version": 2,
        "status": "complete",
        "protocol_lock_sha256": "2" * 64,
        "grid_sha256": "3" * 64,
        "fit_count": 49,
        "fit_sha256": {fit_id: "4" * 64 for fit_id in fit_ids},
    }
    monkeypatch.setattr(
        audit,
        "verify_protected_v1_release",
        lambda root, members: (copy.deepcopy(release), copy.deepcopy(snapshots)),
    )
    monkeypatch.setattr(
        audit,
        "extract_v1_fit_record",
        lambda row, content: _v1_fit_record(str(row["fit_id"]), [0.0]),
    )
    monkeypatch.setattr(
        audit,
        "authenticate_complete_v2_matrix",
        lambda root: (copy.deepcopy(matrix), {fit_id: {} for fit_id in fit_ids}),
    )
    monkeypatch.setattr(
        audit,
        "compare_fit_record",
        lambda old, new, *, v2_sha256: {
            "fit_id": old["fit_id"],
            "weights_comparable": True,
            "weights_exact": True,
            "weight_linf_delta": 0.0,
            "v1_train_objective": 0.3,
            "v2_train_objective": 0.3,
            "train_objective_delta_v2_minus_v1": 0.0,
        },
    )
    consumers = {
        "manuscript_macros": {"Alpha": "2"},
        "consumer_ids": sorted(audit.evaluate_v2.PAPER_CONSUMER_CONTRACT_V2),
        "provenance": {
            "macro_coordinates": {
                "Alpha": {"source_id": "fit/a", "split": "temporal_2022"}
            }
        },
    }
    monkeypatch.setattr(
        audit,
        "authenticate_v2_evaluation_commit",
        lambda root, receipt: (
            {"evaluation_summary_sha256": "5" * 64},
            [],
            consumers,
        ),
    )
    metric_row = {
        "fit_id": fit_ids[0],
        "family": "outcome_frontier",
        "split": "temporal_2022",
        "metric": "welfare",
        "status": "comparable",
        "v1_value": 1.0,
        "v2_value": 2.0,
        "delta_v2_minus_v1": 1.0,
        "changed": True,
    }
    evaluation_summary = {
        "fit_count": 49,
        "metric_coordinates": 245,
        "metric_comparable": 195,
        "metric_unavailable": 50,
        "metric_changed": 1,
        "winner_coordinates": 1,
        "winner_comparable": 0,
        "winner_unavailable": 1,
        "protected_winner_trace_status": "not_recorded_in_protected_release",
    }
    monkeypatch.setattr(
        audit,
        "compare_protected_evaluation",
        lambda *args, **kwargs: (evaluation_summary, [metric_row], []),
    )
    monkeypatch.setattr(
        audit,
        "compare_protected_consumers",
        lambda *args, **kwargs: (
            {
                "consumer_total": 25,
                "consumer_compared": 25,
                "numeric_coordinates": 10,
                "numeric_changed": 2,
                "numeric_exact": 8,
            },
            [{"coordinate": "consumer/value"}],
            [{"consumer_id": "outcome.ablation"}],
        ),
    )
    monkeypatch.setattr(
        audit,
        "explicit_legacy_fit_gaps",
        lambda rows: [{"fit_id": f"gap/{index}"} for index in range(7)],
    )
    monkeypatch.setattr(
        audit,
        "_read_regular",
        lambda path, *, label: b"\\newcommand{\\Alpha}{1}\n\\newcommand{\\External}{9}\n",
    )
    monkeypatch.setattr(
        audit,
        "PROTECTED_NUMBERS_SHA256",
        hashlib.sha256(
            b"\\newcommand{\\Alpha}{1}\n\\newcommand{\\External}{9}\n"
        ).hexdigest(),
    )

    payload, csv_rows = audit.build_delta_audit(tmp_path)

    assert payload["status"] == "pass"
    assert payload["fit_audit"]["total"] == 49
    assert payload["evaluation_audit"]["metric_comparable"] == 195
    assert payload["consumer_audit"]["consumer_compared"] == 25
    assert payload["manuscript_audit"]["compared_changed"] == 1
    assert payload["manuscript_audit"]["unaffected_non_primary"] == 1
    assert payload["causal_attribution"]["duplicate_only_attribution_allowed"] is False
    assert csv_rows == [metric_row]
    assert payload["delta_csv"]["sha256"] == hashlib.sha256(
        audit.serialize_delta_csv(csv_rows).encode()
    ).hexdigest()


def test_payload_contract_forbids_duplicate_only_causal_attribution() -> None:
    audit = _audit()
    payload = audit.assemble_delta_payload(
        release_audit={"pdf_sha256": "a" * 64},
        authenticated_inputs={"v2_protocol_lock_sha256": "b" * 64},
        semantic_correction={"semantics_profile": "approval-set-first-occurrence-v2"},
        fit_audit={"total": 49},
        evaluation_audit={"total_rows": 1},
        consumer_audit={"consumer_total": 25},
        manuscript_audit={"total": 1},
        csv_sha256="c" * 64,
    )

    assert payload["status"] == "pass"
    assert payload["interpretation"] == "complete_corrected_lineage_rerun"
    assert payload["causal_attribution"] == {
        "duplicate_only_attribution_allowed": False,
        "reason": (
            "The corrected lineage changes both approval-set normalization and "
            "deterministic project ordering; observed deltas cannot be assigned "
            "to duplicate removal alone."
        ),
    }
    assert json.dumps(payload, allow_nan=False)
