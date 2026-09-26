"""Contract tests for the district-held-out analysis."""

from __future__ import annotations

import pytest

from iclr_cross_district import (
    boundary_feature_names,
    fold_block_summary,
    output_paths,
    paired_summary,
    validate_replay_summary,
    validate_feature_names,
    validate_fold_coverage,
    validate_run_config,
    validate_run_counts,
)
from iclr_train import TrainConfig


def test_training_config_records_optimizer_bound() -> None:
    assert TrainConfig().bound == 10.0


def test_output_paths_keep_sensitivity_results_separate() -> None:
    paths = output_paths("b40")
    assert paths[0].name == "iclr_cross_district_b40_per_series.csv"
    assert paths[1].name == "iclr_cross_district_b40_summary.csv"
    assert paths[2].name == "iclr_cross_district_b40_manifest.json"


def test_boundary_feature_names_reports_only_exact_box_hits() -> None:
    assert boundary_feature_names(
        [10.0, -9.9, -10.0, 0.0], ["a", "b", "c", "d"], bound=10.0
    ) == ["a", "c"]


def test_validate_replay_summary_checks_saved_fold_aggregate() -> None:
    rows = [
        {
            "worst_csd": 0.1,
            "mean_csd": 0.05,
            "welfare": 10.0,
            "cost_welfare": 8.0,
            "exclusion": 0.2,
        },
        {
            "worst_csd": 0.3,
            "mean_csd": 0.15,
            "welfare": 20.0,
            "cost_welfare": 12.0,
            "exclusion": 0.4,
        },
    ]
    saved = {
        "n_series": 2.0,
        "worst_csd": 0.2,
        "mean_csd": 0.1,
        "welfare": 30.0,
        "cost_welfare": 20.0,
        "exclusion": 0.3,
    }
    validate_replay_summary(rows, saved)
    with pytest.raises(ValueError, match="welfare"):
        validate_replay_summary(rows, {**saved, "welfare": 31.0})


def test_validate_fold_coverage_accepts_exact_partition() -> None:
    validate_fold_coverage(
        {"fold0": ["a", "c"], "fold1": ["b"]},
        expected_keys=["a", "b", "c"],
    )


@pytest.mark.parametrize(
    "memberships, message",
    [
        ({"fold0": ["a"], "fold1": ["a", "b"]}, "duplicated"),
        ({"fold0": ["a"]}, "missing"),
        ({"fold0": ["a", "b", "c"]}, "unexpected"),
    ],
)
def test_validate_fold_coverage_rejects_invalid_partitions(memberships, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_fold_coverage(memberships, expected_keys=["a", "b"])


def test_paired_summary_uses_exact_sign_flip_test() -> None:
    # Differences -1, -2, -3 give the hand-enumerable two-sided p=2/8=0.25.
    summary = paired_summary(
        learned=[0.0, 0.0, 0.0],
        baseline=[1.0, 2.0, 3.0],
        seed=7,
        n_boot=199,
    )

    assert summary["diff"] == pytest.approx(-2.0)
    assert summary["p_two_sided"] == pytest.approx(0.25)
    assert summary["wins"] == 3
    assert summary["n"] == 3
    assert summary["ci_lo"] <= summary["diff"] <= summary["ci_hi"]


def test_fold_block_summary_uses_folds_as_inference_units() -> None:
    # Per-series differences are [-1, -3, -2, -4], but the inferential units
    # are fold means [-2, -2, -4]. Their exact two-sided p-value is 2/8=0.25.
    summary = fold_block_summary(
        learned=[0.0, 0.0, 0.0, 0.0],
        baseline=[1.0, 3.0, 2.0, 4.0],
        folds=[0, 0, 1, 2],
    )

    assert summary["fold_mean_diff"] == pytest.approx(-8.0 / 3.0)
    assert summary["fold_p_two_sided"] == pytest.approx(0.25)
    assert summary["fold_wins"] == 3
    assert summary["n_folds"] == 3


def test_validate_run_config_accepts_primary_fit() -> None:
    validate_run_config(
        {
            "arm": "endowment",
            "split": "district_out_f2of5",
            "seed": 42,
            "generations": 25,
            "popsize": None,
            "sigma0": 0.4,
            "bound": 10.0,
            "init": "res",
            "welfare_penalty": 0.0,
            "welfare_floor": 1.0,
        },
        expected_split="district_out_f2of5",
        expected_seed=42,
        expected_generations=25,
    )


def test_validate_run_config_allows_missing_legacy_bound_only_when_explicit() -> None:
    config = {
        "arm": "endowment",
        "split": "district_out_f2of5",
        "seed": 42,
        "generations": 25,
        "popsize": None,
        "sigma0": 0.4,
        "init": "res",
        "welfare_penalty": 0.0,
        "welfare_floor": 1.0,
    }
    with pytest.raises(ValueError, match="bound"):
        validate_run_config(
            config,
            expected_split="district_out_f2of5",
            expected_seed=42,
            expected_generations=25,
        )

    validate_run_config(
        config,
        expected_split="district_out_f2of5",
        expected_seed=42,
        expected_generations=25,
        allow_missing_default_bound=True,
    )


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("arm", "outcome"),
        ("split", "district_out_f1of5"),
        ("seed", 7),
        ("generations", 1),
        ("bound", 40.0),
        ("init", "zeros"),
        ("welfare_penalty", 2.0),
    ],
)
def test_validate_run_config_rejects_nonprimary_fit(field, bad_value) -> None:
    config = {
        "arm": "endowment",
        "split": "district_out_f2of5",
        "seed": 42,
        "generations": 25,
        "popsize": None,
        "sigma0": 0.4,
        "bound": 10.0,
        "init": "res",
        "welfare_penalty": 0.0,
        "welfare_floor": 1.0,
    }
    config[field] = bad_value
    with pytest.raises(ValueError, match=field):
        validate_run_config(
            config,
            expected_split="district_out_f2of5",
            expected_seed=42,
            expected_generations=25,
        )


def test_validate_run_counts_accepts_exact_split_sizes() -> None:
    validate_run_counts(
        {"n_train_series": 15, "n_test_series": 4},
        expected_train=15,
        expected_test=4,
    )


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"n_train_series": 14, "n_test_series": 4}, "n_train_series"),
        ({"n_train_series": 15, "n_test_series": 3}, "n_test_series"),
    ],
)
def test_validate_run_counts_rejects_wrong_split_sizes(payload, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_run_counts(payload, expected_train=15, expected_test=4)


def test_validate_feature_names_requires_exact_order() -> None:
    validate_feature_names(["size", "deficit"], ["size", "deficit"])
    with pytest.raises(ValueError, match="feature_names"):
        validate_feature_names(["deficit", "size"], ["size", "deficit"])
