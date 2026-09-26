"""Behavioral tests for nested static anchors and contextual residuals."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest

from iclr_env import RolloutState
from iclr_policy import InstanceFeatures, clear_feature_cache, instance_features
from iclr_reviewer_checks import senior_tilt_policy
from parse_pb import PBInstance, Project, Vote

from iclr_residual_policy import (
    AGE_REFERENCE,
    AGE_SEX_REFERENCE,
    FEATURE_NAMES,
    AnchorSpec,
    FeatureScaler,
    anchor_logit_map,
    anchor_multipliers,
    anchor_policy,
    fit_context_scaler,
    residual_policy,
    standardized_context,
)


def _instance(path: str = "test://residual-primary", budget: float = 100.0) -> PBInstance:
    """All age-sex cells plus voters with missing age and sex."""
    return PBInstance(
        path=path,
        meta={"budget": str(budget)},
        projects={
            "p1": Project("p1", 2.0, None),
            "p2": Project("p2", 5.0, None),
            "p3": Project("p3", 11.0, None),
        },
        votes=[
            Vote("young-f", ("p1", "p2"), age=20, sex="F"),
            Vote("young-m", ("p1",), age=21, sex="M"),
            Vote("early-f", ("p2", "p3"), age=30, sex="F"),
            Vote("early-m", ("p3",), age=31, sex="M"),
            Vote("middle-f", ("p1", "p3"), age=45, sex="F"),
            Vote("middle-m", ("p1", "p2"), age=50, sex="M"),
            Vote("senior-f", ("p2",), age=65, sex="F"),
            Vote("senior-m", ("p1", "p2", "p3"), age=70, sex="M"),
            Vote("missing-age", ("p1",), age=None, sex="F"),
            Vote("missing-sex", ("p2",), age=70, sex=""),
        ],
    )


def _second_instance() -> PBInstance:
    return PBInstance(
        path="test://residual-second",
        meta={"budget": "80"},
        projects={
            "a": Project("a", 3.0, None),
            "b": Project("b", 7.0, None),
            "c": Project("c", 13.0, None),
        },
        votes=[
            Vote("a-f-1", ("a",), age=20, sex="F"),
            Vote("a-f-2", ("a", "b"), age=20, sex="F"),
            Vote("s-m", ("b", "c"), age=70, sex="M"),
            Vote("unknown", ("c",), age=None, sex="M"),
        ],
    )


def _constant_feature_instance() -> PBInstance:
    return PBInstance(
        path="test://residual-constant",
        meta={"budget": "20"},
        projects={"only": Project("only", 4.0, None)},
        votes=[
            Vote("young", ("only",), age=20, sex="F"),
            Vote("early", ("only",), age=30, sex="M"),
        ],
    )


def _complete_demographic_instance() -> PBInstance:
    inst = _instance(path="test://residual-complete")
    return PBInstance(
        path=inst.path,
        meta=inst.meta,
        projects=inst.projects,
        votes=inst.votes[:8],
    )


def _anchor(family: str, logits: tuple[float, ...] = ()) -> AnchorSpec:
    reference = {
        "mes": None,
        "senior": AGE_REFERENCE,
        "age": AGE_REFERENCE,
        "age_sex": AGE_SEX_REFERENCE,
    }[family]
    return AnchorSpec(family, logits, reference)


def _static_rows(instances: list[PBInstance]) -> np.ndarray:
    rows = []
    for inst in instances:
        features = instance_features(inst, scheme="age_sex")
        for index, _ in enumerate(features.cohort_order):
            rows.append(
                [features.static[name][index] for name in FEATURE_NAMES]
            )
    return np.asarray(rows, dtype=float)


def test_mes_anchor_returns_uniform_endowments() -> None:
    observed = anchor_policy(_anchor("mes"))(_instance(), RolloutState())

    assert observed == [10.0] * 10


@pytest.mark.parametrize("alpha", [0.0, 2.8, 999.0])
def test_senior_anchor_matches_existing_scalar_tilt(alpha: float) -> None:
    anchor = _anchor("senior", (math.log1p(alpha),))

    observed = anchor_policy(anchor)(_instance(), RolloutState())
    expected = senior_tilt_policy(alpha)(_instance(), RolloutState())

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)
    assert anchor.alpha == pytest.approx(alpha, abs=1e-12)


def test_age_anchor_contains_senior_anchor() -> None:
    senior_logit = math.log1p(2.8)
    observed = anchor_policy(_anchor("age", (0.0, 0.0, senior_logit)))(
        _instance(), RolloutState()
    )
    expected = anchor_policy(_anchor("senior", (senior_logit,)))(
        _instance(), RolloutState()
    )

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)


def test_age_sex_anchor_contains_age_anchor() -> None:
    age_logits = (0.2, -0.1, 0.4)
    observed = anchor_policy(
        _anchor("age_sex", (0.0, age_logits[0], age_logits[0], age_logits[1], age_logits[1], age_logits[2], age_logits[2]))
    )(_complete_demographic_instance(), RolloutState())
    expected = anchor_policy(_anchor("age", age_logits))(
        _complete_demographic_instance(), RolloutState()
    )

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    "anchor",
    [
        _anchor("senior", (0.4,)),
        _anchor("age", (0.2, -0.1, 0.4)),
        _anchor("age_sex", (0.0, 0.2, 0.2, -0.1, -0.1, 0.4, 0.4)),
    ],
)
def test_invalid_or_missing_demographics_use_unit_raw_multiplier(
    anchor: AnchorSpec,
) -> None:
    multipliers = anchor_multipliers(_instance(), anchor)

    assert multipliers[-2:].tolist() == [1.0, 1.0]


@pytest.mark.parametrize(
    "anchor",
    [
        _anchor("mes"),
        _anchor("senior", (math.log1p(2.8),)),
        _anchor("age", (0.1, -0.2, 0.3)),
        _anchor("age_sex", (0.0, 0.1, 0.1, -0.2, -0.2, 0.3, 0.3)),
    ],
)
def test_anchor_outputs_are_positive_and_budget_normalized(anchor: AnchorSpec) -> None:
    budget = 73.0
    observed = anchor_policy(anchor)(_instance(budget=budget), RolloutState())

    assert all(value > 0.0 for value in observed)
    assert abs(sum(observed) - budget) <= 1e-10 * max(1.0, budget)


def test_zero_residual_is_elementwise_identical_to_its_anchor() -> None:
    inst = _instance()
    anchor = _anchor("age_sex", (0.0, 0.2, 0.2, -0.1, -0.1, 0.4, 0.4))
    scaler = fit_context_scaler([inst])

    observed = residual_policy(anchor, scaler, np.zeros(len(FEATURE_NAMES)))(
        inst, RolloutState()
    )
    expected = anchor_policy(anchor)(inst, RolloutState())

    assert observed == expected


def test_residual_log_factor_and_final_factor_are_bounded() -> None:
    import iclr_residual_policy as policy_module

    log_factors = np.array(
        [policy_module._residual_log_factor(score) for score in (-50.0, 0.0, 50.0)]
    )

    assert np.all(log_factors >= -math.log(2.0))
    assert np.all(log_factors <= math.log(2.0))
    assert np.all(np.exp(log_factors) >= 0.5)
    assert np.all(np.exp(log_factors) <= 2.0)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda: _anchor("age", (0.0, 0.0)), "free logits"),
        (lambda: _anchor("senior", (float("nan"),)), "finite"),
        (lambda: residual_policy(_anchor("mes"), fit_context_scaler([_instance()]), (1.0,)), "weights"),
        (lambda: residual_policy(_anchor("mes"), fit_context_scaler([_instance()]), (0.0, 0.0, 0.0, float("inf"))), "finite"),
    ],
)
def test_invalid_parameter_dimensions_and_values_raise_value_error(operation, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        operation()


def test_nonpositive_budget_raises_for_an_election_with_voters() -> None:
    with pytest.raises(ValueError, match="positive budget"):
        anchor_policy(_anchor("mes"))(_instance(budget=0.0), RolloutState())


def test_zero_voter_election_returns_empty_before_budget_validation() -> None:
    empty = PBInstance(path="test://residual-empty", meta={"budget": "0"})

    assert anchor_policy(_anchor("mes"))(empty, RolloutState()) == []


def test_scaler_fits_only_supplied_election_cohort_rows_with_equal_weight() -> None:
    instances = [_instance(), _second_instance()]
    expected_rows = _static_rows(instances)

    scaler = fit_context_scaler(instances)

    np.testing.assert_allclose(scaler.mean, expected_rows.mean(axis=0), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(scaler.scale, expected_rows.std(axis=0, ddof=0), rtol=0.0, atol=1e-12)
    assert scaler.feature_names == FEATURE_NAMES
    assert scaler.row_count == len(expected_rows)
    assert scaler.instance_count == 2
    payload = {
        "clip": scaler.clip,
        "feature_names": list(scaler.feature_names),
        "instance_count": scaler.instance_count,
        "mean": scaler.mean.tolist(),
        "row_count": scaler.row_count,
        "scale": scaler.scale.tolist(),
    }
    expected_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    assert scaler.sha256 == expected_sha256


def test_constant_scaler_columns_use_unit_scale() -> None:
    scaler = fit_context_scaler([_constant_feature_instance()])

    assert scaler.scale.tolist() == [1.0] * len(FEATURE_NAMES)


def test_standardized_context_clips_each_feature_to_the_requested_bound() -> None:
    scaler = FeatureScaler(
        feature_names=FEATURE_NAMES,
        mean=np.zeros(len(FEATURE_NAMES)),
        scale=np.full(len(FEATURE_NAMES), 1e-9),
        clip=3.0,
    )

    transformed = standardized_context(_instance(), scaler)

    assert all(np.all(values >= -3.0) and np.all(values <= 3.0) for values in transformed.values())
    assert any(np.any(np.abs(values) == 3.0) for values in transformed.values())


def test_anchor_logit_map_has_fixed_reference_cells() -> None:
    assert anchor_logit_map(_anchor("age", (0.1, 0.2, 0.3)))[AGE_REFERENCE] == 0.0
    assert anchor_logit_map(
        _anchor("age_sex", (0.0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3))
    )[AGE_SEX_REFERENCE] == 0.0


def test_age_sex_logit_map_uses_the_exact_seven_free_cell_order() -> None:
    logits = (0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77)

    assert anchor_logit_map(_anchor("age_sex", logits)) == {
        "age<25|F": 0.0,
        "age<25|M": 0.11,
        "age25-39|F": 0.22,
        "age25-39|M": 0.33,
        "age40-59|F": 0.44,
        "age40-59|M": 0.55,
        "age60+|F": 0.66,
        "age60+|M": 0.77,
    }


def test_normalization_handles_a_finite_raw_sum_that_would_overflow() -> None:
    import iclr_residual_policy as policy_module

    observed = policy_module._budget_normalize(np.array([1e308, 1e308]), 1.0)

    assert all(math.isfinite(value) and value > 0.0 for value in observed)
    assert sum(observed) == 1.0


@pytest.mark.parametrize("budget", [float("inf"), float("nan")])
def test_nonempty_policy_rejects_nonfinite_budget(budget: float) -> None:
    with pytest.raises(ValueError, match="finite and positive budget"):
        anchor_policy(_anchor("mes"))(_instance(budget=budget), RolloutState())


def test_nonempty_policy_rejects_subnormal_budget_without_positive_shares() -> None:
    budget = float(np.nextafter(0.0, 1.0))

    with pytest.raises(ValueError, match="represent"):
        anchor_policy(_anchor("mes"))(_instance(budget=budget), RolloutState())


@pytest.mark.parametrize(
    ("family", "logits"),
    [
        ("senior", (1000.0,)),
        ("age", (1000.0, 0.0, 0.0)),
        ("age_sex", (1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ],
)
def test_extreme_finite_logits_raise_value_error_not_overflow(
    family: str, logits: tuple[float, ...]
) -> None:
    with pytest.raises(ValueError, match="representable"):
        anchor_policy(_anchor(family, logits))(_instance(), RolloutState())


def test_scaler_defensively_copies_and_freezes_its_arrays() -> None:
    mean = np.zeros(len(FEATURE_NAMES))
    scale = np.ones(len(FEATURE_NAMES))
    scaler = FeatureScaler(FEATURE_NAMES, mean, scale, clip=3.0)
    original_sha256 = scaler.sha256

    mean[0] = 99.0
    scale[1] = 99.0

    assert scaler.mean.tolist() == [0.0] * len(FEATURE_NAMES)
    assert scaler.scale.tolist() == [1.0] * len(FEATURE_NAMES)
    assert scaler.sha256 == original_sha256
    inst = _instance(path="test://residual-immutable-scaler")
    policy = residual_policy(_anchor("mes"), scaler, (0.4, -0.3, 0.2, -0.1))
    expected_policy_output = policy(inst, RolloutState())
    with pytest.raises(ValueError, match="read-only"):
        scaler.mean[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        scaler.scale[0] = 1.0
    with pytest.raises(ValueError, match="cannot set WRITEABLE"):
        scaler.mean.setflags(write=True)
    with pytest.raises(ValueError, match="cannot set WRITEABLE"):
        scaler.scale.setflags(write=True)
    assert scaler.sha256 == original_sha256
    assert policy(inst, RolloutState()) == expected_policy_output


def test_existing_residual_policy_is_immune_to_caller_weight_mutation() -> None:
    inst = _instance(path="test://residual-immutable-weights")
    weights = np.array([0.4, -0.3, 0.2, -0.1])
    policy = residual_policy(_anchor("mes"), fit_context_scaler([inst]), weights)
    expected = policy(inst, RolloutState())

    weights[:] = 100.0

    assert policy(inst, RolloutState()) == expected


def _same_path_payloads() -> tuple[PBInstance, PBInstance, PBInstance]:
    first = PBInstance(
        path="test://reused-path",
        meta={"budget": "10"},
        projects={"a": Project("a", 2.0, None)},
        votes=[Vote("first", ("a",), age=20, sex="F")],
    )
    second = PBInstance(
        path="test://reused-path",
        meta={"budget": "10"},
        projects={"a": Project("a", 3.0, None), "b": Project("b", 9.0, None)},
        votes=[
            Vote("second-young", ("a",), age=20, sex="F"),
            Vote("second-senior", ("b",), age=70, sex="M"),
        ],
    )
    expected_view = PBInstance(
        path="test://independent-expected-payload",
        meta=second.meta,
        projects=second.projects,
        votes=second.votes,
    )
    return first, second, expected_view


def test_scaler_and_context_ignore_stale_features_for_reused_paths() -> None:
    first, second, expected_view = _same_path_payloads()
    clear_feature_cache()
    instance_features(first, scheme="age_sex")
    expected_features = instance_features(expected_view, scheme="age_sex")
    expected_rows = np.asarray(
        [
            [expected_features.static[name][index] for name in FEATURE_NAMES]
            for index, _ in enumerate(expected_features.cohort_order)
        ],
        dtype=float,
    )

    scaler = fit_context_scaler([second])
    context = standardized_context(second, scaler)

    assert scaler.row_count == 2
    np.testing.assert_allclose(scaler.mean, expected_rows.mean(axis=0), rtol=0.0, atol=1e-12)
    assert tuple(context) == expected_features.cohort_order


def _features_with_first_static_values(values: np.ndarray) -> InstanceFeatures:
    cohorts = tuple(f"age<25|{'FM'[index]}" for index in range(len(values)))
    zeros = np.zeros(len(values))
    return InstanceFeatures(
        voter_cohorts=cohorts,
        cohort_sizes={cohort: 1 for cohort in cohorts},
        static={
            "share_deviation": values,
            "ballot_length_dev": zeros,
            "cost_focus_dev": zeros,
            "overlap_dev": zeros,
        },
        cohort_order=cohorts,
    )


def test_nonfinite_fitted_standard_deviation_uses_unit_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iclr_residual_policy as policy_module

    features = _features_with_first_static_values(np.array([-1e308, 1e308]))
    monkeypatch.setattr(policy_module, "instance_features", lambda inst, scheme: features)

    scaler = fit_context_scaler([_instance(path="test://overflow-scale")])

    assert scaler.mean[0] == 0.0
    assert scaler.scale[0] == 1.0


def test_nonfinite_source_feature_row_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iclr_residual_policy as policy_module

    features = _features_with_first_static_values(np.array([float("nan")]))
    monkeypatch.setattr(policy_module, "instance_features", lambda inst, scheme: features)

    with pytest.raises(ValueError, match="source feature rows"):
        fit_context_scaler([_instance(path="test://nan-source-feature")])


def test_nonfinite_fitted_mean_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iclr_residual_policy as policy_module

    features = _features_with_first_static_values(np.array([1e308, 1e308]))
    monkeypatch.setattr(policy_module, "instance_features", lambda inst, scheme: features)

    with pytest.raises(ValueError, match="fitted feature means"):
        fit_context_scaler([_instance(path="test://infinite-feature-mean")])


def test_nonzero_residual_is_positive_finite_and_budget_conserving() -> None:
    inst = _instance(path="test://residual-nonzero", budget=73.0)
    observed = residual_policy(
        _anchor("age", (0.1, -0.2, 0.3)),
        fit_context_scaler([inst]),
        (0.4, -0.3, 0.2, -0.1),
    )(inst, RolloutState())

    assert all(math.isfinite(value) and value > 0.0 for value in observed)
    assert abs(sum(observed) - inst.budget) <= 1e-10 * max(1.0, inst.budget)


def test_zero_voter_anchor_and_residual_policies_return_empty() -> None:
    empty = PBInstance(path="test://residual-empty-both", meta={"budget": "nan"})
    scaler = fit_context_scaler([_instance(path="test://residual-empty-scaler")])

    assert anchor_policy(_anchor("mes"))(empty, RolloutState()) == []
    assert residual_policy(_anchor("mes"), scaler, np.zeros(len(FEATURE_NAMES)))(
        empty, RolloutState()
    ) == []


@pytest.mark.parametrize(
    ("budget", "raw"),
    [
        (
            950314335.0358976,
            [
                0.04790619673228112,
                27.524743917546942,
                5.8679086473326,
                166.60274722097768,
                1.0594399106179484,
                0.024780909713169104,
                285.3433111081216,
            ],
        ),
        (133215364.42822711, [2.3210057100718986, 2.570743812668962, 11.338756475130744]),
        (
            348958226.8811893,
            [3.930178855689943, 0.1083197172006492, 11.073069070131199, 0.029873154110590223, 238.92687707915408],
        ),
    ],
)
def test_production_range_normalization_regressions_are_accepted(
    budget: float, raw: list[float]
) -> None:
    import iclr_residual_policy as policy_module

    observed = policy_module._budget_normalize(np.asarray(raw), budget)

    assert all(math.isfinite(value) and value > 0.0 for value in observed)
    assert abs(math.fsum(observed) - budget) <= 1e-10 * max(1.0, budget)


@pytest.mark.parametrize("raw", [[np.nextafter(0.0, 1.0), np.finfo(float).max], [np.finfo(float).max, np.nextafter(0.0, 1.0)]])
def test_full_finite_range_normalization_preserves_representable_shares(
    raw: list[float],
) -> None:
    import iclr_residual_policy as policy_module

    budget = float(np.finfo(float).max)
    observed = policy_module._budget_normalize(np.asarray(raw), budget)

    assert all(math.isfinite(value) and value > 0.0 for value in observed)
    assert abs(math.fsum(observed) - budget) <= 1e-10 * max(1.0, budget)


def test_max_anchor_with_positive_residual_log_factor_stays_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iclr_residual_policy as policy_module

    high_logit = math.log(np.finfo(float).max)
    inst = PBInstance(
        path="test://max-anchor-residual",
        meta={"budget": "1"},
        projects={"p": Project("p", 1.0, None)},
        votes=[Vote("young", ("p",), age=20, sex="F"), Vote("early", ("p",), age=30, sex="M")],
    )
    scaler = FeatureScaler(FEATURE_NAMES, np.zeros(4), np.ones(4), clip=3.0)
    monkeypatch.setattr(
        policy_module,
        "standardized_context",
        lambda instance, fitted: {"age<25|F": np.zeros(4), "age25-39|M": np.array([1.0, 0.0, 0.0, 0.0])},
    )

    observed = residual_policy(
        _anchor("age", (high_logit, 0.0, 0.0)), scaler, (1.0, 0.0, 0.0, 0.0)
    )(inst, RolloutState())

    assert policy_module._residual_log_factor(1.0) > 0.0
    assert all(math.isfinite(value) and value > 0.0 for value in observed)
    assert abs(math.fsum(observed) - 1.0) <= 1e-10


def _canonical_training_instances() -> list[PBInstance]:
    instances = []
    for index in range(40):
        instances.append(
            PBInstance(
                path=f"test://canonical-order-{index}",
                meta={"budget": "100"},
                projects={
                    "a": Project("a", float(index + 1), None),
                    "b": Project("b", float(83 - index), None),
                    "c": Project("c", float((index % 7) + 2), None),
                },
                votes=[
                    Vote("young", ("a", "c"), age=20, sex="F"),
                    Vote("early", ("a",), age=30, sex="M"),
                    Vote("middle", ("b",), age=50, sex="F"),
                    Vote("senior", ("b", "c"), age=70, sex="M"),
                ],
            )
        )
    return instances


def test_scaler_fit_is_bit_identical_under_population_permutations() -> None:
    instances = _canonical_training_instances()
    forward = fit_context_scaler(instances)
    reverse = fit_context_scaler(list(reversed(instances)))
    shuffled = fit_context_scaler(instances[::2] + instances[1::2])

    assert forward.mean.tobytes() == reverse.mean.tobytes() == shuffled.mean.tobytes()
    assert forward.scale.tobytes() == reverse.scale.tobytes() == shuffled.scale.tobytes()
    assert forward.sha256 == reverse.sha256 == shuffled.sha256


def test_scaler_fit_keeps_duplicate_instances_as_duplicate_rows() -> None:
    instances = _canonical_training_instances()[:2]
    base = fit_context_scaler(instances)
    duplicated = fit_context_scaler(instances + [instances[0]])

    assert base.row_count == 8
    assert duplicated.row_count == 12
    assert duplicated.instance_count == 3


@pytest.mark.parametrize("invalid", [float("inf"), float("-inf"), float("nan")])
def test_standardized_context_rejects_each_nonfinite_evaluation_feature(
    monkeypatch: pytest.MonkeyPatch, invalid: float
) -> None:
    import iclr_residual_policy as policy_module

    features = _features_with_first_static_values(np.array([invalid]))
    monkeypatch.setattr(policy_module, "instance_features", lambda inst, scheme: features)
    scaler = FeatureScaler(FEATURE_NAMES, np.zeros(4), np.ones(4), clip=3.0)

    with pytest.raises(ValueError, match="share_deviation.*age<25\\|F"):
        standardized_context(_instance(path="test://nonfinite-evaluation"), scaler)


def test_standardized_context_rejects_nonfinite_preclip_intermediate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iclr_residual_policy as policy_module

    features = _features_with_first_static_values(np.array([1e308]))
    monkeypatch.setattr(policy_module, "instance_features", lambda inst, scheme: features)
    scaler = FeatureScaler(
        FEATURE_NAMES,
        np.array([-1e308, 0.0, 0.0, 0.0]),
        np.ones(4),
        clip=3.0,
    )

    with pytest.raises(ValueError, match="standardized feature share_deviation.*age<25\\|F"):
        standardized_context(_instance(path="test://overflow-evaluation"), scaler)


@pytest.mark.parametrize("logs", [(0.0, -745.0), (-745.0, 0.0)])
def test_log_finalizer_accepts_positive_minimum_subnormal_shares(
    logs: tuple[float, float],
) -> None:
    import iclr_residual_policy as policy_module

    observed = policy_module._final_shares_from_logs(np.asarray(logs), 1.0)

    assert math.exp(-745.0) > 0.0
    assert all(math.isfinite(value) and value > 0.0 for value in observed)
    assert abs(math.fsum(observed) - 1.0) <= 1e-10


def test_checked_exp_uses_actual_round_to_zero_boundary() -> None:
    import iclr_residual_policy as policy_module

    assert policy_module._checked_exp(-745.0) == math.exp(-745.0)
    assert math.exp(-746.0) == 0.0
    with pytest.raises(ValueError, match="representable"):
        policy_module._checked_exp(-746.0)


def test_public_age_anchor_accepts_a_positive_subnormal_logit() -> None:
    inst = PBInstance(
        path="test://subnormal-age-anchor",
        meta={"budget": "1"},
        projects={"p": Project("p", 1.0, None)},
        votes=[Vote("young", ("p",), age=20, sex="F"), Vote("early", ("p",), age=30, sex="M")],
    )

    observed = anchor_policy(_anchor("age", (-745.0, 0.0, 0.0)))(inst, RolloutState())

    assert all(math.isfinite(value) and value > 0.0 for value in observed)
    assert abs(math.fsum(observed) - 1.0) <= 1e-10


def test_public_anchor_with_negative_residual_accepts_positive_subnormal_share(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import iclr_residual_policy as policy_module

    inst = PBInstance(
        path="test://subnormal-residual-anchor",
        meta={"budget": "1"},
        projects={"p": Project("p", 1.0, None)},
        votes=[Vote("young", ("p",), age=20, sex="F"), Vote("early", ("p",), age=30, sex="M")],
    )
    scaler = FeatureScaler(FEATURE_NAMES, np.zeros(4), np.ones(4), clip=3.0)
    monkeypatch.setattr(
        policy_module,
        "standardized_context",
        lambda instance, fitted: {"age<25|F": np.zeros(4), "age25-39|M": np.array([-1.0, 0.0, 0.0, 0.0])},
    )
    subnormal_logit = math.log(np.nextafter(0.0, 1.0))

    observed = residual_policy(
        _anchor("age", (subnormal_logit, 0.0, 0.0)), scaler, (1.0, 0.0, 0.0, 0.0)
    )(inst, RolloutState())

    assert policy_module._residual_log_factor(-1.0) < 0.0
    assert all(math.isfinite(value) and value > 0.0 for value in observed)
    assert abs(math.fsum(observed) - 1.0) <= 1e-10


@pytest.mark.parametrize("logit", [1e-16, -1e-16])
def test_senior_alpha_preserves_tiny_fitted_logits_exactly(logit: float) -> None:
    anchor = _anchor("senior", (logit,))

    assert anchor.alpha == math.expm1(logit)
    assert math.log1p(anchor.alpha) == anchor.free_logits[0]
