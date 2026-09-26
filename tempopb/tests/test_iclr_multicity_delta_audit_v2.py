"""Contracts for the append-only multicity v1-to-v2 delta audit."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest


def _episode(
    *,
    series: str,
    winners: tuple[str, ...] = ("a",),
    worst_csd: float = 0.1,
) -> dict[str, object]:
    return {
        "series": series,
        "scored_years": [2022],
        "worst_csd": worst_csd,
        "worst_cohort": "age25-39|F",
        "mean_csd": worst_csd / 2,
        "welfare": 10.0,
        "cost_welfare": 20.0,
        "exclusion": 0.2,
        "years": [],
        "scored_outcomes": [{"year": 2022, "winners": list(winners)}],
    }


def _fit(weights: list[float], loss: float) -> dict[str, object]:
    return {
        "schema_version": 1,
        "config": {"split": "temporal_2022", "arm": "priority", "seed": 42},
        "arm": {"name": "priority", "feature_names": ["x"]},
        "execution_mode": "serial",
        "training_only": True,
        "result": {
            "best_weights": weights,
            "best_loss": loss,
            "initial_loss": 0.5,
            "n_evals": 8,
            "selection_source": "cmaes",
        },
    }


def _decision(observed: object = 0.1) -> dict[str, object]:
    return {
        "classification": "falsified",
        "conditions": {
            "temporal_macro_at_most_minus_0_010": {
                "passed": False,
                "observed": observed,
                "threshold": "<= -0.01",
            }
        },
        "gold_pass": False,
        "primary_seed": "42",
    }


def test_strict_json_rejects_duplicate_keys_and_exponent_overflow() -> None:
    audit = importlib.import_module("iclr_multicity_delta_audit_v2")

    with pytest.raises(RuntimeError, match="duplicate JSON key"):
        audit._load_json_bytes(b'{"value": 1, "value": 2}', label="duplicate")
    with pytest.raises(RuntimeError, match="finite"):
        audit._load_json_bytes(b'{"value": 1e309}', label="overflow")


def test_fit_delta_requires_matching_scientific_contract() -> None:
    audit = importlib.import_module("iclr_multicity_delta_audit_v2")
    v1 = _fit([0.0], 0.2)
    v2 = copy.deepcopy(v1)
    v2["schema_version"] = 2
    v2["result"]["best_weights"] = [0.25]
    v2["result"]["best_loss"] = 0.1

    row = audit.compare_fit_payloads(
        "temporal_2022/priority/seed-42.json",
        v1,
        v2,
        v1_sha256="a" * 64,
        v2_sha256="b" * 64,
    )

    assert row["weights_exact"] is False
    assert row["weight_linf_delta"] == 0.25
    assert row["best_loss_delta_v2_minus_v1"] == pytest.approx(-0.1)

    v2["config"]["seed"] = 1
    with pytest.raises(RuntimeError, match="fit contract differs"):
        audit.compare_fit_payloads(
            "temporal_2022/priority/seed-42.json",
            v1,
            v2,
            v1_sha256="a" * 64,
            v2_sha256="b" * 64,
        )

    v2 = copy.deepcopy(v1)
    v2["schema_version"] = 2
    v2["config"]["seed"] = True
    with pytest.raises(RuntimeError, match="fit contract differs"):
        audit.compare_fit_payloads(
            "temporal_2022/priority/seed-42.json",
            v1,
            v2,
            v1_sha256="a" * 64,
            v2_sha256="b" * 64,
        )


def test_per_series_delta_rejects_fixed_policy_winner_drift_outside_wola() -> None:
    audit = importlib.import_module("iclr_multicity_delta_audit_v2")
    v1 = {
        "temporal_2022": {
            "mes": [_episode(series="Poland/Warszawa/Bemowo")],
        }
    }
    v2 = copy.deepcopy(v1)
    v2["temporal_2022"]["mes"][0] = _episode(
        series="Poland/Warszawa/Bemowo",
        winners=("b",),
    )

    with pytest.raises(RuntimeError, match="fixed-policy winner drift"):
        audit.compare_per_series_payloads(v1, v2)


def test_per_series_delta_rejects_integer_float_year_alias() -> None:
    audit = importlib.import_module("iclr_multicity_delta_audit_v2")
    v1 = {
        "temporal_2022": {
            "mes": [_episode(series="Poland/Warszawa/Bemowo")],
        }
    }
    v2 = copy.deepcopy(v1)
    v2["temporal_2022"]["mes"][0]["scored_years"] = [2022.0]

    with pytest.raises(RuntimeError, match="scored years"):
        audit.compare_per_series_payloads(v1, v2)


def test_decision_delta_rejects_nested_schema_drift() -> None:
    audit = importlib.import_module("iclr_multicity_delta_audit_v2")
    v1 = _decision({"Poland/Warszawa": 0.1})
    v2 = copy.deepcopy(v1)
    v2["provenance"] = {}
    v2["conditions"]["temporal_macro_at_most_minus_0_010"]["observed"] = {
        "Poland/Gdynia": 0.1
    }

    with pytest.raises(RuntimeError, match="observed schema differs"):
        audit.compare_decision_payloads(v1, v2)


def test_decision_delta_requires_the_same_frozen_primary_seed() -> None:
    audit = importlib.import_module("iclr_multicity_delta_audit_v2")
    v1 = _decision()
    v2 = copy.deepcopy(v1)
    v2["provenance"] = {}
    v2["primary_seed"] = "99"

    with pytest.raises(RuntimeError, match="primary_seed differs"):
        audit.compare_decision_payloads(v1, v2)


def test_audit_pair_installs_referenced_csv_before_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audit = importlib.import_module("iclr_multicity_delta_audit_v2")
    calls = []

    def record(path, content, *, label):
        calls.append((Path(path).name, content, label))
        return "a" * 64 if Path(path).suffix == ".json" else "b" * 64

    monkeypatch.setattr(
        audit.protocol_v2,
        "write_immutable_bytes_artifact_v2",
        record,
    )
    json_sha, csv_sha = audit._install_audit_pair(
        json_path=tmp_path / "audit.json",
        json_bytes=b"json",
        csv_path=tmp_path / "rows.csv",
        csv_bytes=b"csv",
    )

    assert [name for name, _, _ in calls] == ["rows.csv", "audit.json"]
    assert json_sha == "a" * 64
    assert csv_sha == "b" * 64


def test_write_path_recomputes_both_builds_without_trusted_shortcut(
    tmp_path: Path,
    monkeypatch,
) -> None:
    audit = importlib.import_module("iclr_multicity_delta_audit_v2")
    calls = []
    payload = {
        "status": "pass",
        "decision_audit": {"v2_classification": "falsified"},
        "fit_audit": {"total": 24},
        "evaluation_audit": {"total_rows": 1650, "winner_changed_rows": 0},
    }

    def build(root, **kwargs):
        calls.append((Path(root), kwargs))
        return copy.deepcopy(payload), []

    monkeypatch.setattr(audit, "build_delta_audit", build)
    monkeypatch.setattr(
        audit.protocol_v2,
        "canonical_result_root_v2",
        lambda path: Path(path),
    )
    monkeypatch.setattr(
        audit.protocol_v2,
        "canonical_result_output_path_v2",
        lambda root, relative: Path(root) / relative,
    )
    monkeypatch.setattr(
        audit.protocol_v2,
        "preflight_immutable_bytes_artifact_v2",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(audit, "serialize_delta_csv", lambda rows: "csv")
    monkeypatch.setattr(audit, "_json_bytes", lambda value: b"json")
    monkeypatch.setattr(
        audit,
        "_install_audit_pair",
        lambda **kwargs: ("a" * 64, "b" * 64),
    )

    result = audit.write_delta_audit(tmp_path)

    assert len(calls) == 2
    assert calls[0][1] == {}
    assert calls[1][1] == {}
    assert result["status"] == "pass"


def test_repository_delta_audit_has_complete_authenticated_coverage() -> None:
    audit = importlib.import_module("iclr_multicity_delta_audit_v2")
    repo_root = Path(__file__).resolve().parent.parent

    payload, rows = audit.build_delta_audit(repo_root)

    assert payload["status"] == "pass"
    assert payload["interpretation"] == "complete_corrected_lineage_rerun"
    assert payload["fit_audit"]["total"] == 24
    assert payload["fit_audit"]["changed_weights"] == 2
    assert payload["fit_audit"]["negative_control_total"] == 5
    assert payload["fit_audit"]["negative_control_changed_weights"] == 2
    assert payload["evaluation_audit"]["total_rows"] == 1650
    assert payload["evaluation_audit"]["fixed_policy_rows_outside_wola"] == 592
    assert payload["evaluation_audit"]["fixed_policy_winner_drift_outside_wola"] == 0
    assert payload["evaluation_audit"]["winner_changed_rows"] == 26
    assert payload["decision_audit"]["v1_classification"] == "falsified"
    assert payload["decision_audit"]["v2_classification"] == "falsified"
    assert payload["decision_audit"]["condition_pass_status_changed"] == []
    assert len(rows) == 1650
    assert len({(row["split"], row["policy"], row["series"]) for row in rows}) == 1650

    encoded = audit.serialize_delta_csv(rows)
    assert encoded.count("\n") == 1651
    assert json.dumps(payload, allow_nan=False)
