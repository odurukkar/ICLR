"""Statistics and mechanical evidence decision for locked multi-city outputs."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from iclr_corpus import SeriesRef
from iclr_env import EnvConfig, EpisodeResult
from iclr_multicity_evaluate import (
    FIT_SOURCE_FILES,
    count_actuated_series,
    evaluate_series,
    evaluate_gold_gate,
    holm_adjust,
    paired_city_summary,
    required_fit_specs,
    required_fit_inventory,
    verify_fit_inventory,
    verify_lock_fit_inventory,
)
from iclr_multicity_protocol import build_protocol_lock
from iclr_multicity_train import build_multicity_arm
from parse_pb import PBInstance, Project, Vote


def _episode(series: str, csd: float, welfare: float, exclusion: float) -> EpisodeResult:
    return EpisodeResult(
        series=series,
        scored_years=(2023,),
        worst_csd=csd,
        worst_cohort="group",
        mean_csd=csd,
        welfare=welfare,
        cost_welfare=welfare,
        exclusion=exclusion,
        years=(),
    )


def test_paired_city_summary_reports_effect_counts_welfare_and_exclusion() -> None:
    learned = [
        _episode("Poland/Warszawa/A", 0.1, 98.0, 0.10),
        _episode("Poland/Warszawa/B", 0.3, 98.0, 0.12),
        _episode("Poland/Łódź/C", 0.2, 50.0, 0.20),
    ]
    baseline = [
        _episode("Poland/Warszawa/A", 0.2, 100.0, 0.10),
        _episode("Poland/Warszawa/B", 0.3, 100.0, 0.10),
        _episode("Poland/Łódź/C", 0.4, 50.0, 0.21),
    ]

    rows = paired_city_summary(
        learned, baseline, bootstrap_draws=2000, bootstrap_seed=7
    )

    warsaw = rows["Poland/Warszawa"]
    assert warsaw["n_series"] == 2
    assert warsaw["difference"] == pytest.approx(-0.05)
    assert (warsaw["wins"], warsaw["ties"], warsaw["losses"]) == (1, 1, 0)
    assert warsaw["welfare_ratio"] == pytest.approx(0.98)
    assert warsaw["exclusion_difference"] == pytest.approx(0.01)
    assert warsaw["ci_low"] <= warsaw["difference"] <= warsaw["ci_high"]
    assert "p_value" in warsaw
    assert rows["CITY_MACRO"]["difference"] == pytest.approx(-0.125)


def test_city_out_summary_omits_population_level_p_value() -> None:
    learned = [_episode("Poland/Gdynia/A", 0.1, 10.0, 0.1)]
    baseline = [_episode("Poland/Gdynia/A", 0.2, 10.0, 0.1)]

    rows = paired_city_summary(
        learned,
        baseline,
        bootstrap_draws=100,
        bootstrap_seed=1,
        include_inference=False,
    )

    assert "p_value" not in rows["Poland/Gdynia"]


def test_holm_adjustment_is_monotone_and_bounded() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.20})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.08, "c": 0.20})


def _gold_payload():
    city_row = {
        "difference": -0.02,
        "welfare_ratio": 0.99,
        "exclusion_difference": 0.001,
    }
    temporal_seeds = {
        str(seed): {
            "cities": {
                city: dict(city_row)
                for city in ("Poland/Warszawa", "Poland/Gdynia", "Poland/Łódź")
            },
            "city_macro_difference": -0.02 + offset,
        }
        for seed, offset in ((1, 0.0), (2, 0.001), (42, 0.002))
    }
    city_out_seeds = {
        str(seed): {
            "cities": {
                city: {"difference": -0.01}
                for city in ("Poland/Warszawa", "Poland/Gdynia", "Poland/Łódź")
            }
        }
        for seed in (1, 2, 42)
    }
    return {
        "containment_pass": True,
        "primary_seed": "42",
        "temporal": {
            "seeds": temporal_seeds,
            "priority_actuated_series": 30,
            "endowment_actuated_series": 20,
        },
        "city_out": {"seeds": city_out_seeds},
    }


def test_complete_gold_fixture_passes_every_condition() -> None:
    decision = evaluate_gold_gate(_gold_payload())
    assert decision["classification"] == "gold"
    assert decision["gold_pass"] is True
    assert all(row["passed"] for row in decision["conditions"].values())


def test_silver_gate_enforces_locked_effect_interval() -> None:
    payload = _gold_payload()
    payload["city_out"]["seeds"]["42"]["cities"]["Poland/Gdynia"][
        "difference"
    ] = 0.0
    for seed_row in payload["temporal"]["seeds"].values():
        seed_row["city_macro_difference"] = -0.007
    assert evaluate_gold_gate(payload)["classification"] == "silver"

    for seed_row in payload["temporal"]["seeds"].values():
        seed_row["city_macro_difference"] = -0.020
    assert evaluate_gold_gate(payload)["classification"] == "falsified"


@pytest.mark.parametrize(
    ("condition", "mutate"),
    (
        ("exact_mes_containment", lambda p: p.update(containment_pass=False)),
        (
            "temporal_negative_each_city",
            lambda p: p["temporal"]["seeds"]["42"]["cities"]["Poland/Gdynia"].update(
                difference=0.001
            ),
        ),
        (
            "temporal_macro_at_most_minus_0_010",
            lambda p: p["temporal"]["seeds"]["42"].update(
                city_macro_difference=-0.009
            ),
        ),
        (
            "city_out_negative_each_city_and_seed",
            lambda p: p["city_out"]["seeds"]["1"]["cities"]["Poland/Łódź"].update(
                difference=0.0
            ),
        ),
        (
            "welfare_ratio_each_city",
            lambda p: p["temporal"]["seeds"]["42"]["cities"]["Poland/Warszawa"].update(
                welfare_ratio=0.979
            ),
        ),
        (
            "exclusion_increase_each_city",
            lambda p: p["temporal"]["seeds"]["42"]["cities"]["Poland/Warszawa"].update(
                exclusion_difference=0.006
            ),
        ),
        (
            "temporal_negative_each_city_all_seeds",
            lambda p: p["temporal"]["seeds"]["2"]["cities"]["Poland/Gdynia"].update(
                difference=0.001
            ),
        ),
        (
            "temporal_macro_seed_spread",
            lambda p: p["temporal"]["seeds"]["2"].update(
                city_macro_difference=-0.013
            ),
        ),
        (
            "priority_actuation_greater_than_endowment",
            lambda p: p["temporal"].update(priority_actuated_series=20),
        ),
    ),
)
def test_each_gold_condition_can_fail_independently(condition, mutate) -> None:
    payload = deepcopy(_gold_payload())
    mutate(payload)
    decision = evaluate_gold_gate(payload)
    assert decision["gold_pass"] is False
    assert decision["conditions"][condition]["passed"] is False


def test_series_evaluation_records_only_scored_winner_fingerprints() -> None:
    ref = SeriesRef(
        "Poland/Test/Unit",
        (2022, 2023),
        (Path("2022.pb"), Path("2023.pb")),
    )
    instances = {
        year: PBInstance(
            path=f"test://{year}",
            meta={
                "budget": "1",
                "date_begin": f"{year}-01-01",
                "country": "Poland",
                "unit": "Test",
                "subunit": "Unit",
            },
            projects={"a": Project("a", 1.0, None)},
            votes=[Vote("0", ("a",), age=20, sex="F")],
        )
        for year in (2022, 2023)
    }

    learned = evaluate_series(
        ref,
        lambda inst, state: {"a"},
        (2023,),
        instances,
        EnvConfig(),
    )
    baseline = evaluate_series(
        ref,
        lambda inst, state: set(),
        (2023,),
        instances,
        EnvConfig(),
    )

    assert learned.scored_outcomes == ((2023, ("a",)),)
    assert baseline.scored_outcomes == ((2023, ()),)
    assert count_actuated_series([learned], [baseline]) == 1


def test_required_fit_matrix_has_24_prespecified_runs() -> None:
    specs = required_fit_specs()
    assert len(specs) == 24
    assert sum(spec["arm"] == "priority" for spec in specs) == 12
    assert all(spec["seed"] in (1, 2, 42) for spec in specs)


def test_required_fit_paths_are_split_arm_seed_addressed(tmp_path: Path) -> None:
    inventory = required_fit_inventory(tmp_path)
    assert len(inventory) == 24
    spec, path = inventory[0]
    assert path == (
        tmp_path
        / "fits"
        / str(spec["split"])
        / str(spec["arm"])
        / f"seed-{spec['seed']}.json"
    )


def test_lock_and_evaluator_require_the_same_fit_matrix(tmp_path: Path) -> None:
    tracked = tmp_path / "source.py"
    tracked.write_text("x = 1\n")
    payload = build_protocol_lock(tmp_path, {"source": tracked})
    verify_lock_fit_inventory(payload)

    payload["fit_inventory"].pop()
    with pytest.raises(RuntimeError, match="fit inventory differs"):
        verify_lock_fit_inventory(payload)


def test_fit_inventory_rejects_hyperparameter_drift(tmp_path: Path) -> None:
    spec = {"split": "temporal_2022", "arm": "priority", "seed": 1}
    path = tmp_path / "fit.json"
    arm = build_multicity_arm("priority")
    source_hashes = {name: f"hash-{name}" for name in FIT_SOURCE_FILES}
    lock_payload = {
        "tracked_files": {
            "artifact/corpus_manifest": {"sha256": "corpus-hash"},
            "split/temporal_2022": {"sha256": "split-hash"},
            **{
                f"source/{name}": {"sha256": digest}
                for name, digest in source_hashes.items()
            },
        }
    }
    payload = {
        "training_only": True,
        "execution_mode": "serial",
        "config": {
            **spec,
            "generations": 30,
            "popsize": None,
            "sigma0": 0.4,
            "bound": 10.0,
            "welfare_penalty": 2.0,
            "welfare_floor": 1.0,
        },
        "arm": {
            "name": "priority",
            "feature_names": list(arm.feature_names),
            "init_name": arm.init_name,
            "initial": arm.initial.tolist(),
        },
        "result": {"best_weights": [0.0] * 5},
        "provenance": {
            "protocol_lock_sha256": "lock-hash",
            "pabulib_commit": "2f4321fec84069f50abf35fb5e90852013d17070",
            "corpus_manifest_sha256": "corpus-hash",
            "split_sha256": "split-hash",
            "source_sha256": source_hashes,
        },
    }
    path.write_text(json.dumps(payload))
    verify_fit_inventory([(spec, path)], lock_payload, "lock-hash")

    payload["config"]["generations"] = 29
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="generations"):
        verify_fit_inventory([(spec, path)], lock_payload, "lock-hash")


def test_fit_inventory_rejects_missing_provenance(tmp_path: Path) -> None:
    spec = {"split": "temporal_2022", "arm": "priority", "seed": 1}
    arm = build_multicity_arm("priority")
    path = tmp_path / "fit.json"
    path.write_text(
        json.dumps(
                {
                    "training_only": True,
                    "execution_mode": "serial",
                    "config": {
                        **spec,
                        "generations": 30,
                        "popsize": None,
                        "sigma0": 0.4,
                        "bound": 10.0,
                        "welfare_penalty": 2.0,
                        "welfare_floor": 1.0,
                    },
                    "arm": {
                        "name": arm.name,
                        "feature_names": list(arm.feature_names),
                        "init_name": arm.init_name,
                        "initial": arm.initial.tolist(),
                    },
                    "result": {"best_weights": [0.0] * 5},
            }
        )
    )
    with pytest.raises(RuntimeError, match="provenance"):
        verify_fit_inventory([(spec, path)], {"tracked_files": {}}, "lock-hash")


def test_script_entrypoint_is_after_all_runtime_function_definitions() -> None:
    source = (Path(__file__).parents[1] / "src" / "iclr_multicity_evaluate.py").read_text()
    assert source.rfind('if __name__ == "__main__":') > source.index(
        "def evaluate_gold_gate"
    )
