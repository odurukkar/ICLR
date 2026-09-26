"""Training-only tests for static residual-anchor fitting and selection."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pickle
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import pytest

import iclr_residual_train as residual_train_module
from iclr_corpus import SeriesRef, Split, leave_district_out
from iclr_env import EpisodeResult
from iclr_residual_policy import (
    AGE_REFERENCE,
    AGE_SEX_REFERENCE,
    AnchorSpec,
    fit_context_scaler,
)
from iclr_residual_train import (
    AnchorCandidate,
    AnchorFit,
    CityMetrics,
    DevelopmentFold,
    DevelopmentSeriesResult,
    FoldFit,
    RAY_AMPLITUDES,
    ResidualCandidate,
    ResidualMetrics,
    ResidualSeedFit,
    SeriesMetrics,
    _smoke_training_data,
    age_initializer,
    age_sex_initializer,
    anchor_search_config,
    cluster_city_bootstrap,
    development_cluster_key,
    evaluate_ray,
    evaluate_development_gate,
    fit_and_evaluate_development_fold,
    fit_residual_seed,
    residual_search_config,
    residual_training_loss,
    load_development_folds,
    run_development,
    score_static_candidate,
    score_residual_candidate,
    select_residual_candidate,
    select_primary_seed,
    select_ray_candidate,
    select_static_anchor,
    senior_anchor_grid,
    smoke_anchor_search_config,
    smoke_residual_search_config,
    soft_worst_city,
    summarize_city_metrics,
)
from iclr_wide_senior_audit import alpha_grid
from iclr_train import SeriesData
from parse_pb import PBInstance, Project, Vote


def _episode(
    series: str,
    csd: float | None,
    welfare: float,
    exclusion: float,
) -> EpisodeResult:
    return EpisodeResult(
        series=series,
        scored_years=(2022,),
        worst_csd=csd,
        worst_cohort="group" if csd is not None else None,
        mean_csd=csd,
        welfare=welfare,
        cost_welfare=welfare,
        exclusion=exclusion,
        years=(),
    )


def _anchor(family: str, logits: tuple[float, ...]) -> AnchorSpec:
    references = {
        "mes": None,
        "senior": AGE_REFERENCE,
        "age": AGE_REFERENCE,
        "age_sex": AGE_SEX_REFERENCE,
    }
    return AnchorSpec(family, logits, references[family])


def _candidate(
    family: str,
    logits: tuple[float, ...],
    objective: float,
    *,
    safe: bool = True,
) -> AnchorCandidate:
    return AnchorCandidate(
        anchor=_anchor(family, logits),
        objective=objective,
        metrics={"Poland/Test": CityMetrics(objective, 10.0, 0.1)},
        safe=safe,
        source="test",
        seed=None,
    )


def _anchor_payload_sha(anchor: AnchorSpec) -> str:
    payload = {
        "alpha": anchor.alpha,
        "family": anchor.family,
        "free_logits": list(anchor.free_logits),
        "reference_cell": anchor.reference_cell,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_soft_worst_city_is_stable_near_one_thousand_and_zero_for_zeros() -> None:
    expected = 1000.0 + 0.02 * math.log(
        (1.0 + math.exp((999.99 - 1000.0) / 0.02)) / 2.0
    )

    assert soft_worst_city([1000.0, 999.99]) == pytest.approx(expected)
    assert soft_worst_city([0.0, 0.0, 0.0]) == 0.0


def test_city_metrics_mean_series_per_city_without_pooling_city_sizes() -> None:
    episodes = [
        _episode(f"Poland/Large/{index}", 0.0, 2.0, 0.1)
        for index in range(10)
    ] + [_episode("Poland/Small/only", 1.0, 3.0, 0.3)]

    metrics = summarize_city_metrics(episodes)

    assert metrics == {
        "Poland/Large": CityMetrics(0.0, 20.0, 0.1),
        "Poland/Small": CityMetrics(1.0, 3.0, 0.3),
    }
    assert soft_worst_city([row.city_csd for row in metrics.values()]) != pytest.approx(
        1.0 / 11.0
    )


def test_welfare_safety_is_conjunctive_in_every_city(monkeypatch) -> None:
    episodes = [
        _episode("Poland/A/one", 0.1, 100.0, 0.1),
        _episode("Poland/B/one", 0.1, 98.9, 0.1),
    ]
    monkeypatch.setattr(
        "iclr_residual_train.rollout_static_training", lambda *args: episodes
    )
    mes = {
        "Poland/A": CityMetrics(0.2, 100.0, 0.1),
        "Poland/B": CityMetrics(0.2, 100.0, 0.1),
    }

    candidate = score_static_candidate(
        _anchor("mes", ()), [], anchor_search_config(), mes
    )

    assert candidate.safe is False
    assert math.isinf(candidate.objective)


def test_exclusion_safety_is_conjunctive_in_every_city(monkeypatch) -> None:
    episodes = [
        _episode("Poland/A/one", 0.1, 100.0, 0.1),
        _episode("Poland/B/one", 0.1, 100.0, 0.1000001),
    ]
    monkeypatch.setattr(
        "iclr_residual_train.rollout_static_training", lambda *args: episodes
    )
    mes = {
        "Poland/A": CityMetrics(0.2, 100.0, 0.1),
        "Poland/B": CityMetrics(0.2, 100.0, 0.1),
    }

    candidate = score_static_candidate(
        _anchor("mes", ()), [], anchor_search_config(), mes
    )

    assert candidate.safe is False
    assert math.isinf(candidate.objective)


def test_missing_city_csd_is_unsafe_with_infinite_optimizer_score(monkeypatch) -> None:
    monkeypatch.setattr(
        "iclr_residual_train.rollout_static_training",
        lambda *args: [_episode("Poland/A/one", None, 100.0, 0.1)],
    )

    candidate = score_static_candidate(
        _anchor("mes", ()),
        [],
        anchor_search_config(),
        {"Poland/A": CityMetrics(0.2, 100.0, 0.1)},
    )

    assert candidate.safe is False
    assert math.isinf(candidate.objective)


def test_same_city_infinite_welfare_is_unsafe_even_with_a_finite_row(
    monkeypatch,
) -> None:
    episodes = [
        _episode("Poland/A/finite", 0.1, 100.0, 0.1),
        _episode("Poland/A/infinite-welfare", 0.2, float("inf"), 0.1),
    ]
    monkeypatch.setattr(
        "iclr_residual_train.rollout_static_training", lambda *args: episodes
    )

    candidate = score_static_candidate(
        _anchor("mes", ()),
        [],
        anchor_search_config(),
        {"Poland/A": CityMetrics(0.2, 100.0, 0.1)},
    )

    assert candidate.safe is False
    assert math.isinf(candidate.objective)


def test_same_city_nan_exclusion_is_unsafe_even_with_a_finite_row(
    monkeypatch,
) -> None:
    episodes = [
        _episode("Poland/A/finite", 0.1, 100.0, 0.1),
        _episode("Poland/A/nan-exclusion", 0.2, 100.0, float("nan")),
    ]
    monkeypatch.setattr(
        "iclr_residual_train.rollout_static_training", lambda *args: episodes
    )

    candidate = score_static_candidate(
        _anchor("mes", ()),
        [],
        anchor_search_config(),
        {"Poland/A": CityMetrics(0.2, 100.0, 0.1)},
    )

    assert candidate.safe is False
    assert math.isinf(candidate.objective)


def test_mes_equal_to_its_baseline_is_an_explicit_safe_fallback(monkeypatch) -> None:
    episodes = [_episode("Poland/A/one", 0.2, 100.0, 0.1)]
    monkeypatch.setattr(
        "iclr_residual_train.rollout_static_training", lambda *args: episodes
    )
    mes = {"Poland/A": CityMetrics(0.2, 100.0, 0.1)}

    candidate = score_static_candidate(
        _anchor("mes", ()), [], anchor_search_config(), mes
    )

    assert candidate.anchor.family == "mes"
    assert candidate.safe is True
    assert math.isfinite(candidate.objective)


def test_anchor_candidate_metrics_are_a_defensive_immutable_snapshot() -> None:
    original = CityMetrics(0.2, 100.0, 0.1)
    changed = CityMetrics(999.0, -1.0, 999.0)
    source = {"Poland/A": original}
    candidate = AnchorCandidate(
        anchor=_anchor("mes", ()),
        objective=0.2,
        metrics=source,
        safe=True,
        source="test",
        seed=None,
    )

    source["Poland/A"] = changed

    assert candidate.metrics["Poland/A"] == original
    with pytest.raises(TypeError):
        candidate.metrics["Poland/A"] = changed


def test_anchor_fit_mes_metrics_are_defensive_immutable_and_not_aliased() -> None:
    original = CityMetrics(0.2, 100.0, 0.1)
    changed = CityMetrics(999.0, -1.0, 999.0)
    source = {"Poland/A": original}
    candidate = AnchorCandidate(
        anchor=_anchor("mes", ()),
        objective=0.2,
        metrics=source,
        safe=True,
        source="mes-fallback",
        seed=None,
    )
    fit = AnchorFit(candidate, (candidate,), source, anchor_search_config())

    source["Poland/A"] = changed

    assert candidate.metrics["Poland/A"] == original
    assert fit.mes_metrics["Poland/A"] == original
    assert fit.mes_metrics is not candidate.metrics
    with pytest.raises(TypeError):
        fit.mes_metrics["Poland/A"] = changed


def test_anchor_result_objects_remain_immutable_after_pickle_round_trip() -> None:
    metrics = {"Poland/A": CityMetrics(0.2, 100.0, 0.1)}
    candidate = AnchorCandidate(
        anchor=_anchor("mes", ()),
        objective=0.2,
        metrics=metrics,
        safe=True,
        source="mes-fallback",
        seed=None,
    )
    fit = AnchorFit(candidate, (candidate,), metrics, anchor_search_config())

    restored_candidate, restored_fit = pickle.loads(
        pickle.dumps((candidate, fit))
    )

    assert restored_candidate == candidate
    assert restored_fit == fit
    assert restored_fit.mes_metrics is not restored_fit.selected.metrics
    with pytest.raises(TypeError):
        restored_candidate.metrics["Poland/A"] = CityMetrics(0.0, 0.0, 0.0)
    with pytest.raises(TypeError):
        restored_fit.mes_metrics["Poland/A"] = CityMetrics(0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("owner", "pickle_round_trip"),
    (
        ("candidate", False),
        ("fit", False),
        ("candidate", True),
        ("fit", True),
    ),
    ids=(
        "candidate-before-pickle",
        "fit-before-pickle",
        "candidate-after-pickle",
        "fit-after-pickle",
    ),
)
def test_metrics_private_backing_slot_cannot_be_deleted(
    owner: str, pickle_round_trip: bool
) -> None:
    metrics = {"Poland/A": CityMetrics(0.2, 100.0, 0.1)}
    candidate = AnchorCandidate(
        anchor=_anchor("mes", ()),
        objective=0.2,
        metrics=metrics,
        safe=True,
        source="mes-fallback",
        seed=None,
    )
    fit = AnchorFit(candidate, (candidate,), metrics, anchor_search_config())
    if pickle_round_trip:
        candidate, fit = pickle.loads(pickle.dumps((candidate, fit)))
    public_metrics = candidate.metrics if owner == "candidate" else fit.mes_metrics
    expected = dict(public_metrics)

    with pytest.raises(AttributeError, match="immutable metrics cannot be modified"):
        del public_metrics._items

    assert dict(public_metrics) == expected


def test_rollout_static_training_passes_only_training_years_and_instances(
    monkeypatch,
) -> None:
    seen = []
    train_only = {2022: object()}
    row = SimpleNamespace(
        ref=SimpleNamespace(key="Poland/A/one", city="Poland/A", years=(2022, 2023)),
        train_years=(2022,),
        test_years=(2023,),
        train_only=train_only,
        all_years={2022: train_only[2022], 2023: object()},
    )

    def fake_rollout(ref, selector, score_years, cfg, instances):
        del selector, cfg
        seen.append((ref, score_years, instances))
        return _episode(ref.key, 0.2, 10.0, 0.1)

    monkeypatch.setattr("iclr_residual_train.rollout_selector", fake_rollout)

    from iclr_residual_train import rollout_static_training

    rollout_static_training(_anchor("mes", ()), [row])

    assert seen == [(row.ref, (2022,), train_only)]


def test_unsafe_candidate_with_lower_csd_cannot_win() -> None:
    unsafe = _candidate("mes", (), 0.01, safe=False)
    safe = _candidate("senior", (math.log(2.0),), 0.2)

    assert select_static_anchor([unsafe, safe]) is safe


def test_objective_tie_within_tolerance_prefers_fewer_free_dimensions() -> None:
    candidates = [
        _candidate("age_sex", (0.0,) * 7, 0.2),
        _candidate("age", (0.0,) * 3, 0.2),
        _candidate("senior", (0.0,), 0.20005),
    ]

    assert select_static_anchor(candidates).anchor.family == "senior"


def test_objective_difference_outside_tolerance_wins_before_family_size() -> None:
    lower = _candidate("age_sex", (0.0,) * 7, 0.2)
    smaller = _candidate("senior", (0.0,), 0.20011)

    assert select_static_anchor([smaller, lower]) is lower


def test_remaining_tie_prefers_squared_norm_then_full_payload_sha() -> None:
    smaller_norm = _candidate("age", (0.5, 0.0, 0.0), 0.2)
    larger_norm = _candidate("age", (0.6, 0.0, 0.0), 0.2)
    assert select_static_anchor([larger_norm, smaller_norm]) is smaller_norm

    same_norm_a = _candidate("age", (1.0, 0.0, 0.0), 0.2)
    same_norm_b = _candidate("age", (0.0, 1.0, 0.0), 0.2)
    expected = min(
        (same_norm_a, same_norm_b), key=lambda row: _anchor_payload_sha(row.anchor)
    )
    assert select_static_anchor([same_norm_b, same_norm_a]) is expected


def test_senior_grid_reuses_canonical_101_point_grid() -> None:
    grid = senior_anchor_grid()

    assert grid == alpha_grid()
    assert len(grid) == 101
    assert all(left < right for left, right in zip(grid, grid[1:]))
    assert grid[0] == 0.0
    assert 2.8 in grid
    assert 3.0 in grid
    assert grid[-1] == 999.0


def test_age_initializer_exactly_contains_best_safe_senior_candidate() -> None:
    ineligible = _candidate("senior", (math.log(5.0),), 0.01, safe=False)
    eligible = _candidate("senior", (math.log(2.5),), 0.2)

    initialized = age_initializer([ineligible, eligible])

    assert initialized == _anchor("age", (0.0, 0.0, math.log(2.5)))


def test_age_sex_initializer_copies_best_age_logits_into_both_sex_cells() -> None:
    best_age = _candidate("age", (0.2, -0.3, 0.4), 0.2)

    initialized = age_sex_initializer([best_age])

    assert initialized == _anchor(
        "age_sex", (0.0, 0.2, 0.2, -0.3, -0.3, 0.4, 0.4)
    )


def test_production_config_is_frozen_and_smoke_config_is_separate() -> None:
    production = anchor_search_config()
    smoke = smoke_anchor_search_config()

    assert production.seeds == (1, 2, 42)
    assert production.generations == 60
    assert production.popsize == 12
    assert production.sigma0 == 0.4
    assert production.bound == 6.0
    assert production.tau == 0.02
    assert production.welfare_floor == 0.99
    assert production.exclusion_delta == 0.0
    assert production.tie_tolerance == 1e-4
    assert smoke != production
    assert smoke.generations < production.generations
    assert smoke.popsize < production.popsize


def _residual_metrics(
    rows: tuple[tuple[str, str, float, float, float], ...],
) -> ResidualMetrics:
    series = tuple(SeriesMetrics(*row) for row in rows)
    cities = {}
    for city in sorted({row.city for row in series}):
        city_rows = [row for row in series if row.city == city]
        cities[city] = CityMetrics(
            city_csd=math.fsum(row.csd for row in city_rows) / len(city_rows),
            city_welfare=math.fsum(row.welfare for row in city_rows),
            city_exclusion=(
                math.fsum(row.exclusion for row in city_rows) / len(city_rows)
            ),
        )
    return ResidualMetrics(series, cities, ())


def _residual_candidate(
    weights: tuple[float, ...],
    loss: float,
    *,
    amplitude: float | None = None,
    safe: bool = True,
    metrics: ResidualMetrics | None = None,
    source: str = "test",
) -> ResidualCandidate:
    amplitude = amplitude if amplitude is not None else (0.0 if not any(weights) else 1.0)
    endpoint = (
        tuple(0.0 for _ in weights)
        if amplitude == 0.0
        else tuple(value / amplitude for value in weights)
    )
    metrics = metrics or _residual_metrics(
        (("Poland/A/one", "Poland/A", 0.2, 100.0, 0.1),)
    )
    if not safe:
        metrics = ResidualMetrics(
            metrics.series,
            metrics.cities,
            (*metrics.failed_safety, "unsafe test candidate"),
        )
        loss = float("inf")
    return ResidualCandidate(
        weights=weights,
        endpoint=endpoint,
        amplitude=amplitude,
        loss=loss,
        metrics=metrics,
        safe=safe,
        source=source,
    )


def test_every_endpoint_expands_to_exact_five_amplitudes_in_order() -> None:
    endpoint = (2.0, -2.0, 1.0, -1.0)

    def evaluator(weights):
        return _residual_candidate(tuple(weights), loss=0.0)

    candidates = evaluate_ray(endpoint, evaluator)

    assert RAY_AMPLITUDES == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert tuple(row.amplitude for row in candidates) == RAY_AMPLITUDES
    assert tuple(row.weights for row in candidates) == tuple(
        tuple(amplitude * value for value in endpoint)
        for amplitude in RAY_AMPLITUDES
    )
    assert candidates[0].weights == (0.0, 0.0, 0.0, 0.0)
    assert candidates[0].endpoint == (0.0, 0.0, 0.0, 0.0)


def test_equal_ray_losses_choose_smaller_amplitude() -> None:
    candidates = (
        _residual_candidate((0.75, 0.0, 0.0, 0.0), 0.1, amplitude=0.75),
        _residual_candidate((0.25, 0.0, 0.0, 0.0), 0.1, amplitude=0.25),
    )

    assert select_ray_candidate(candidates).amplitude == 0.25


@pytest.mark.parametrize(
    ("candidate_b", "anchor_b", "mes_b", "failed_fragment"),
    (
        ((0.2, 98.9, 0.1), (0.2, 100.0, 0.1), (0.2, 80.0, 0.2), "anchor welfare"),
        ((0.2, 98.9, 0.1), (0.2, 80.0, 0.2), (0.2, 100.0, 0.1), "MES welfare"),
        ((0.2, 100.0, 0.1001), (0.2, 80.0, 0.1), (0.2, 100.0, 0.2), "anchor exclusion"),
        ((0.2, 100.0, 0.1001), (0.2, 100.0, 0.2), (0.2, 80.0, 0.1), "MES exclusion"),
    ),
)
def test_residual_safety_is_every_city_against_anchor_and_mes(
    candidate_b,
    anchor_b,
    mes_b,
    failed_fragment: str,
) -> None:
    safe_a = ("Poland/A/one", "Poland/A", 0.2, 100.0, 0.1)
    candidate = _residual_metrics(
        (safe_a, ("Poland/B/one", "Poland/B", *candidate_b))
    )
    anchor = _residual_metrics(
        (safe_a, ("Poland/B/one", "Poland/B", *anchor_b))
    )
    mes = {
        "Poland/A": CityMetrics(0.2, 100.0, 0.1),
        "Poland/B": CityMetrics(*mes_b),
    }

    scored = score_residual_candidate(
        (0.1, 0.0, 0.0, 0.0),
        (0.1, 0.0, 0.0, 0.0),
        1.0,
        candidate,
        anchor,
        mes,
        "test-safety",
    )

    assert scored.safe is False
    assert math.isinf(scored.loss)
    assert any(failed_fragment in condition for condition in scored.metrics.failed_safety)
    assert set(scored.metrics.cities) == {"Poland/A", "Poland/B"}
    assert tuple(row.series for row in scored.metrics.series) == (
        "Poland/A/one",
        "Poland/B/one",
    )


def test_city_balanced_excess_is_zero_at_zero_weights() -> None:
    anchor = _residual_metrics(
        (
            ("Poland/A/one", "Poland/A", 0.2, 10.0, 0.1),
            ("Poland/A/two", "Poland/A", 0.4, 20.0, 0.2),
            ("Poland/B/one", "Poland/B", 0.8, 30.0, 0.3),
        )
    )

    assert residual_training_loss(anchor, anchor, (0.0, 0.0, 0.0, 0.0)) == 0.0


def test_residual_norm_penalty_is_exactly_one_e_minus_three_dot_w_w() -> None:
    anchor = _residual_metrics(
        (
            ("Poland/A/one", "Poland/A", 0.2, 10.0, 0.1),
            ("Poland/B/one", "Poland/B", 0.8, 30.0, 0.3),
        )
    )
    weights = (1.0, -2.0, 3.0, -4.0)

    assert residual_training_loss(anchor, anchor, weights) == 1e-3 * 30.0


@pytest.mark.parametrize(
    ("improvement", "expected_amplitude"),
    ((0.002, 1.0), (0.001999999999, 0.0)),
)
def test_nonzero_replacement_boundary_is_inclusive_only_at_point_zero_zero_two(
    improvement: float, expected_amplitude: float
) -> None:
    zero = _residual_candidate((0.0, 0.0, 0.0, 0.0), 0.0, amplitude=0.0)
    nonzero = _residual_candidate(
        (1.0, 0.0, 0.0, 0.0), -improvement, amplitude=1.0
    )

    assert select_residual_candidate((zero, nonzero)).amplitude == expected_amplitude


def test_unsafe_candidate_cannot_replace_zero() -> None:
    zero = _residual_candidate((0.0, 0.0, 0.0, 0.0), 0.0, amplitude=0.0)
    unsafe = _residual_candidate(
        (1.0, 0.0, 0.0, 0.0), -1.0, amplitude=1.0, safe=False
    )

    assert select_residual_candidate((unsafe, zero)) is zero


def test_zero_endpoint_all_five_amplitudes_selects_canonical_positive_zero() -> None:
    def evaluator(weights):
        return _residual_candidate(tuple(weights), loss=0.0, source="same-source")

    candidates = evaluate_ray((0.0, 0.0, 0.0, 0.0), evaluator)
    selected = select_residual_candidate(candidates)

    assert tuple(row.amplitude for row in candidates) == RAY_AMPLITUDES
    assert selected.amplitude == 0.0
    assert all(value == 0.0 for value in (*selected.weights, *selected.endpoint))
    assert all(
        math.copysign(1.0, value) == 1.0
        for value in (*selected.weights, *selected.endpoint)
    )


def test_lower_loss_noncanonical_zero_weight_row_cannot_replace_zero() -> None:
    zero = _valid_zero_candidate()
    noncanonical = _residual_candidate(
        (0.0, 0.0, 0.0, 0.0),
        -100.0,
        amplitude=0.75,
        source="adversarial-lower-loss",
    )

    assert select_residual_candidate((zero, noncanonical)) is zero


def test_primary_seed_uses_training_loss_then_amplitude_norm_and_seed() -> None:
    metrics_with_irrelevant_raw_csd = _residual_metrics(
        (("Poland/A/one", "Poland/A", 999.0, 100.0, 0.1),)
    )
    def seed_fit(seed: int, selected: ResidualCandidate) -> ResidualSeedFit:
        zero = _residual_candidate(
            (0.0, 0.0, 0.0, 0.0),
            0.0,
            amplitude=0.0,
            source=f"seed-{seed}-zero",
        )
        return ResidualSeedFit(seed, selected, (zero, selected), ())

    seed_fits = (
        seed_fit(
            42,
            _residual_candidate(
                (0.1, 0.0, 0.0, 0.0),
                -0.1,
                amplitude=0.5,
                metrics=metrics_with_irrelevant_raw_csd,
            ),
        ),
        seed_fit(
            2,
            _residual_candidate((0.2, 0.0, 0.0, 0.0), -0.1, amplitude=0.25),
        ),
        seed_fit(
            1,
            _residual_candidate((0.1, 0.0, 0.0, 0.0), -0.1, amplitude=0.25),
        ),
    )

    assert select_primary_seed(seed_fits) == 1

    strictly_lower_training_loss = seed_fit(
        42,
        _residual_candidate((3.0, 0.0, 0.0, 0.0), -0.100001, amplitude=1.0),
    )
    assert select_primary_seed(
        (strictly_lower_training_loss, seed_fits[1], seed_fits[2])
    ) == 42

    same_amplitude_and_norm = (
        seed_fit(
            42,
            _residual_candidate((0.1, 0.0, 0.0, 0.0), -0.1, amplitude=0.25),
        ),
        seed_fit(
            2,
            _residual_candidate((0.1, 0.0, 0.0, 0.0), -0.1, amplitude=0.25),
        ),
    )
    assert select_primary_seed(same_amplitude_and_norm) == 2


def test_fit_rejects_any_held_out_input_instead_of_ignoring_it() -> None:
    held_out = SimpleNamespace(
        ref=SimpleNamespace(key="Poland/A/one"),
        train_years=(2022,),
        test_years=(2023,),
        train_only={2022: object()},
        all_years={2022: object(), 2023: object()},
    )

    with pytest.raises(ValueError, match="held-out"):
        fit_residual_seed((held_out,), None, None, seed=1, workers=1)


def test_every_fitted_seed_archive_contains_an_exact_canonical_zero(monkeypatch) -> None:
    data = _smoke_training_data()
    mes = score_static_candidate(
        _anchor("mes", ()), data, smoke_anchor_search_config(), None
    )
    anchor_fit = AnchorFit(
        mes,
        (mes,),
        mes.metrics,
        smoke_anchor_search_config(),
    )
    scaler = fit_context_scaler(tuple(data[0].train_only.values()))
    monkeypatch.setattr(
        "iclr_residual_train.residual_search_config",
        smoke_residual_search_config,
    )

    for seed in residual_search_config().seeds:
        fit = fit_residual_seed(data, anchor_fit, scaler, seed, workers=1)
        zeros = [
            row
            for row in fit.archive
            if row.weights == (0.0, 0.0, 0.0, 0.0)
        ]
        assert len(zeros) == 1
        assert zeros[0].endpoint == (0.0, 0.0, 0.0, 0.0)
        assert zeros[0].amplitude == 0.0
        assert zeros[0].safe is True
        assert len(fit.archive) == 17
        assert tuple(row.amplitude for row in fit.archive[1:]) == (
            (0.25, 0.5, 0.75, 1.0) * 4
        )


def test_residual_result_payloads_are_defensive_immutable_and_pickle_safe() -> None:
    source_cities = {"Poland/A": CityMetrics(0.2, 100.0, 0.1)}
    metrics = ResidualMetrics(
        (SeriesMetrics("Poland/A/one", "Poland/A", 0.2, 100.0, 0.1),),
        source_cities,
        (),
    )
    candidate = _residual_candidate(
        (0.0, 0.0, 0.0, 0.0), 0.0, amplitude=0.0, metrics=metrics
    )
    fit = ResidualSeedFit(1, candidate, (candidate,), ({"generation": 0},))
    source_cities["Poland/A"] = CityMetrics(999.0, -1.0, 999.0)

    restored = pickle.loads(pickle.dumps(fit))

    assert restored == fit
    assert restored.archive[0].metrics.cities["Poland/A"] == CityMetrics(
        0.2, 100.0, 0.1
    )
    with pytest.raises(TypeError):
        restored.archive[0].metrics.cities["Poland/A"] = CityMetrics(0.0, 0.0, 0.0)
    with pytest.raises(TypeError):
        restored.optimizer_history[0]["generation"] = 999


def test_optimizer_history_is_defensive_copyable_pickleable_and_ordinary_json() -> None:
    zero = _valid_zero_candidate()
    source = {
        "generation": np.int64(1),
        "vector": np.array([1.0, 2.0]),
        "nested": [{"value": 0.5}, None, True, "ok"],
    }
    fit = ResidualSeedFit(1, zero, (zero,), (source,))
    source["vector"][0] = 999.0
    source["nested"][0]["value"] = 999.0

    assert fit.optimizer_history[0]["vector"] == (1.0, 2.0)
    assert fit.optimizer_history[0]["nested"][0]["value"] == 0.5

    for restored in (
        copy.copy(fit),
        copy.deepcopy(fit),
        pickle.loads(pickle.dumps(fit)),
    ):
        assert restored == fit
        nested = restored.optimizer_history[0]["nested"]
        with pytest.raises(TypeError):
            nested[0]["value"] = 999.0
        encoded = json.dumps(
            residual_train_module._residual_seed_payload(restored),
            sort_keys=True,
            allow_nan=False,
        )
        assert json.loads(encoded)["optimizer_history"][0]["vector"] == [1.0, 2.0]


class _MutableHistoryLeaf:
    def __init__(self) -> None:
        self.values: list[int] = []


@pytest.mark.parametrize(
    "unsupported",
    ({1, 2}, bytearray(b"mutable"), _MutableHistoryLeaf()),
    ids=("set", "bytearray", "custom-mutable"),
)
def test_optimizer_history_rejects_unsupported_leaves(unsupported: object) -> None:
    zero = _valid_zero_candidate()

    with pytest.raises(TypeError, match="optimizer history"):
        ResidualSeedFit(1, zero, (zero,), ({"unsupported": unsupported},))


@pytest.mark.parametrize(
    "nonfinite",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
        np.float64("nan"),
        np.array([1.0, np.inf]),
    ),
    ids=("nan", "positive-inf", "negative-inf", "numpy-nan", "array-inf"),
)
def test_optimizer_history_rejects_nonfinite_numeric_leaves(nonfinite: object) -> None:
    zero = _valid_zero_candidate()

    with pytest.raises(ValueError, match="finite optimizer history"):
        ResidualSeedFit(1, zero, (zero,), ({"nonfinite": nonfinite},))


@pytest.mark.parametrize(
    "value",
    (np.array(1.25), np.longdouble("1.25")),
    ids=("zero-dimensional-array", "longdouble"),
)
def test_optimizer_history_normalizes_finite_zero_dimensional_and_extended_float(
    value: object,
) -> None:
    zero = _valid_zero_candidate()

    fit = ResidualSeedFit(1, zero, (zero,), ({"value": value},))
    normalized = fit.optimizer_history[0]["value"]

    assert type(normalized) is float
    assert normalized == 1.25
    json.dumps(
        residual_train_module._residual_seed_payload(fit),
        sort_keys=True,
        allow_nan=False,
    )


@pytest.mark.parametrize(
    "value",
    (np.array(np.nan), np.longdouble(np.inf)),
    ids=("zero-dimensional-array", "longdouble"),
)
def test_optimizer_history_rejects_nonfinite_zero_dimensional_and_extended_float(
    value: object,
) -> None:
    zero = _valid_zero_candidate()

    with pytest.raises(ValueError, match="finite optimizer history"):
        ResidualSeedFit(1, zero, (zero,), ({"value": value},))


def test_optimizer_history_normalizes_builtin_numpy_scalars_and_matrix() -> None:
    zero = _valid_zero_candidate()
    fit = ResidualSeedFit(
        1,
        zero,
        (zero,),
        (
            {
                "bool": np.bool_(True),
                "integer": np.int64(7),
                "floating": np.float32(0.25),
                "matrix": np.array([[1, 2], [3, 4]], dtype=np.int64),
            },
        ),
    )
    record = fit.optimizer_history[0]

    assert type(record["bool"]) is bool
    assert type(record["integer"]) is int
    assert type(record["floating"]) is float
    assert record["matrix"] == ((1, 2), (3, 4))

    for restored in (
        copy.copy(fit),
        copy.deepcopy(fit),
        pickle.loads(pickle.dumps(fit)),
    ):
        assert restored == fit
        encoded = json.dumps(
            residual_train_module._residual_seed_payload(restored),
            sort_keys=True,
            allow_nan=False,
        )
        assert json.loads(encoded)["optimizer_history"][0]["matrix"] == [
            [1, 2],
            [3, 4],
        ]


@pytest.mark.parametrize(
    "value",
    (
        np.complex128(1.0 + 2.0j),
        np.clongdouble(1.0 + 2.0j),
        np.datetime64("2026-08-18"),
    ),
    ids=("complex", "extended-complex", "datetime"),
)
def test_optimizer_history_rejects_unsupported_numpy_scalars(value: object) -> None:
    zero = _valid_zero_candidate()

    with pytest.raises(TypeError, match="optimizer history"):
        ResidualSeedFit(1, zero, (zero,), ({"value": value},))


@pytest.mark.parametrize(
    "value",
    (
        np.datetime64("2026-08-18", "D"),
        np.datetime64("2026-08-18T00:00:00.000000001", "ns"),
        np.array("2026-08-18", dtype="datetime64[D]"),
        np.array("2026-08-18T00:00:00.000000001", dtype="datetime64[ns]"),
        np.array(["2026-08-18"], dtype="datetime64[D]"),
        np.array(
            ["2026-08-18T00:00:00.000000001"], dtype="datetime64[ns]"
        ),
        np.timedelta64(1, "D"),
        np.timedelta64(1, "ns"),
        np.array(1, dtype="timedelta64[D]"),
        np.array(1, dtype="timedelta64[ns]"),
        np.array([1], dtype="timedelta64[D]"),
        np.array([1], dtype="timedelta64[ns]"),
    ),
    ids=(
        "datetime-scalar-D",
        "datetime-scalar-ns",
        "datetime-zero-dimensional-D",
        "datetime-zero-dimensional-ns",
        "datetime-one-dimensional-D",
        "datetime-one-dimensional-ns",
        "timedelta-scalar-D",
        "timedelta-scalar-ns",
        "timedelta-zero-dimensional-D",
        "timedelta-zero-dimensional-ns",
        "timedelta-one-dimensional-D",
        "timedelta-one-dimensional-ns",
    ),
)
def test_optimizer_history_rejects_datetime_and_timedelta_before_unit_conversion(
    value: object,
) -> None:
    zero = _valid_zero_candidate()

    with pytest.raises(TypeError, match="optimizer history"):
        ResidualSeedFit(1, zero, (zero,), ({"value": value},))


@pytest.mark.parametrize(
    "value",
    (
        np.array(
            (np.datetime64("2026-08-18", "D"),),
            dtype=[("when", "datetime64[D]")],
        )[()],
        np.array(
            (np.datetime64("2026-08-18", "D"),),
            dtype=[("when", "datetime64[D]")],
        ),
        np.array(
            [(np.datetime64("2026-08-18", "D"),)],
            dtype=[("when", "datetime64[D]")],
        ),
        np.array(
            (np.datetime64("2026-08-18T00:00:00.000000001", "ns"),),
            dtype=[("when", "datetime64[ns]")],
        )[()],
        np.array(
            (np.datetime64("2026-08-18T00:00:00.000000001", "ns"),),
            dtype=[("when", "datetime64[ns]")],
        ),
        np.array(
            [(np.datetime64("2026-08-18T00:00:00.000000001", "ns"),)],
            dtype=[("when", "datetime64[ns]")],
        ),
        np.array(
            (np.timedelta64(1, "D"),), dtype=[("elapsed", "timedelta64[D]")]
        )[()],
        np.array(
            (np.timedelta64(1, "D"),), dtype=[("elapsed", "timedelta64[D]")]
        ),
        np.array(
            [(np.timedelta64(1, "D"),)],
            dtype=[("elapsed", "timedelta64[D]")],
        ),
        np.array(
            (np.timedelta64(1, "ns"),),
            dtype=[("elapsed", "timedelta64[ns]")],
        )[()],
        np.array(
            (np.timedelta64(1, "ns"),),
            dtype=[("elapsed", "timedelta64[ns]")],
        ),
        np.array(
            [(np.timedelta64(1, "ns"),)],
            dtype=[("elapsed", "timedelta64[ns]")],
        ),
        np.array((1, 2.5), dtype=[("count", "i8"), ("score", "f8")])[()],
        np.array((1, 2.5), dtype=[("count", "i8"), ("score", "f8")]),
        np.array([(1, 2.5)], dtype=[("count", "i8"), ("score", "f8")]),
    ),
    ids=(
        "datetime-record-D",
        "datetime-zero-dimensional-D",
        "datetime-one-dimensional-D",
        "datetime-record-ns",
        "datetime-zero-dimensional-ns",
        "datetime-one-dimensional-ns",
        "timedelta-record-D",
        "timedelta-zero-dimensional-D",
        "timedelta-one-dimensional-D",
        "timedelta-record-ns",
        "timedelta-zero-dimensional-ns",
        "timedelta-one-dimensional-ns",
        "numeric-record",
        "numeric-zero-dimensional",
        "numeric-one-dimensional",
    ),
)
def test_optimizer_history_rejects_structured_and_void_before_conversion(
    value: object,
) -> None:
    zero = _valid_zero_candidate()

    with pytest.raises(TypeError, match="optimizer history"):
        ResidualSeedFit(1, zero, (zero,), ({"value": value},))


def test_optimizer_history_preserves_supported_numpy_dtype_shapes() -> None:
    zero = _valid_zero_candidate()
    fit = ResidualSeedFit(
        1,
        zero,
        (zero,),
        (
            {
                "bool": np.bool_(True),
                "signed": np.int64(-2),
                "unsigned": np.uint64(9),
                "floating": np.longdouble("1.25"),
                "string": np.str_("ok"),
                "zero_dimensional": np.array(3, dtype=np.int16),
                "one_dimensional": np.array([4, 5], dtype=np.uint16),
                "multi_dimensional": np.array(
                    [[1.25, 2.5]], dtype=np.longdouble
                ),
                "object_array": np.array([1, "ok", None], dtype=object),
            },
        ),
    )
    record = fit.optimizer_history[0]

    assert record["bool"] is True
    assert record["signed"] == -2
    assert record["unsigned"] == 9
    assert record["floating"] == 1.25
    assert record["string"] == "ok"
    assert record["zero_dimensional"] == 3
    assert record["one_dimensional"] == (4, 5)
    assert record["multi_dimensional"] == ((1.25, 2.5),)
    assert record["object_array"] == (1, "ok", None)
    json.dumps(
        residual_train_module._residual_seed_payload(fit),
        sort_keys=True,
        allow_nan=False,
    )


def test_residual_production_config_is_frozen_and_smoke_is_reduced() -> None:
    production = residual_search_config()
    smoke = smoke_residual_search_config()

    assert production.seeds == (1, 2, 42)
    assert production.generations == 40
    assert production.popsize == 8
    assert production.sigma0 == 0.4
    assert production.bound == 3.0
    assert production.tau == 0.02
    assert production.welfare_floor == 0.99
    assert production.replacement_delta == 0.002
    assert production.norm_penalty == 1e-3
    assert smoke.seeds == production.seeds
    assert smoke.generations < production.generations
    assert smoke.popsize < production.popsize


def _anchor_fit_scaler_and_data():
    data = _smoke_training_data()
    mes = score_static_candidate(
        _anchor("mes", ()), data, smoke_anchor_search_config(), None
    )
    anchor_fit = AnchorFit(
        mes,
        (mes,),
        mes.metrics,
        smoke_anchor_search_config(),
    )
    scaler = fit_context_scaler(tuple(data[0].train_only.values()))
    return anchor_fit, scaler, data


def _valid_zero_candidate(*, source: str = "canonical-zero") -> ResidualCandidate:
    return _residual_candidate(
        (0.0, 0.0, 0.0, 0.0),
        0.0,
        amplitude=0.0,
        source=source,
    )


def _valid_nonzero_candidate(
    *, loss: float = -0.01, source: str = "nonzero"
) -> ResidualCandidate:
    metrics = _residual_metrics(
        (("Poland/A/one", "Poland/A", 0.2, 100.0, 0.1),)
    )
    return ResidualCandidate(
        weights=(0.25, 0.0, 0.0, 0.0),
        endpoint=(1.0, 0.0, 0.0, 0.0),
        amplitude=0.25,
        loss=loss,
        metrics=metrics,
        safe=True,
        source=source,
    )


def _valid_seed_fit(seed: int) -> ResidualSeedFit:
    zero = _valid_zero_candidate(source=f"seed-{seed}-zero")
    return ResidualSeedFit(seed, zero, (zero,), ())


def test_residual_metrics_reject_duplicate_series_ids() -> None:
    duplicate_rows = (
        SeriesMetrics("Poland/A/dup", "Poland/A", 0.1, 10.0, 0.1),
        SeriesMetrics("Poland/A/dup", "Poland/A", 0.2, 20.0, 0.2),
    )

    with pytest.raises(ValueError, match="duplicate residual series"):
        ResidualMetrics(
            duplicate_rows,
            {"Poland/A": CityMetrics(0.15, 30.0, 0.15)},
            (),
        )


def test_residual_metrics_reject_contradictory_raw_and_city_evidence() -> None:
    raw = (
        SeriesMetrics("Poland/A/one", "Poland/A", 0.2, 0.0, 1.0),
    )

    with pytest.raises(ValueError, match="city aggregates do not match raw series"):
        ResidualMetrics(
            raw,
            {"Poland/A": CityMetrics(0.2, 100.0, 0.0)},
            (),
        )


@pytest.mark.parametrize(
    ("episode", "expected"),
    (
        (
            _episode("Poland/A/one", None, float("nan"), float("inf")),
            ("missing CSD", "non-finite welfare", "non-finite exclusion"),
        ),
        (
            _episode("Poland/A/one", float("nan"), float("nan"), float("inf")),
            ("non-finite CSD", "non-finite welfare", "non-finite exclusion"),
        ),
    ),
)
def test_residual_metrics_retain_every_invalid_raw_field_independently(
    episode: EpisodeResult, expected: tuple[str, ...]
) -> None:
    metrics = residual_train_module._residual_metrics_from_episodes((episode,))

    for fragment in expected:
        assert any(fragment in failure for failure in metrics.failed_safety)
    assert len(metrics.failed_safety) == len(expected)


@pytest.mark.parametrize("field", ("csd", "welfare", "exclusion"))
def test_candidate_safety_rejects_and_retains_nonfinite_anchor_reference(
    field: str,
) -> None:
    candidate = _residual_metrics(
        (("Poland/A/one", "Poland/A", 0.2, 100.0, 0.1),)
    )
    values = {"csd": 0.2, "welfare": 100.0, "exclusion": 0.1}
    values[field] = float("nan")
    anchor = ResidualMetrics(
        (
            SeriesMetrics(
                "Poland/A/one",
                "Poland/A",
                values["csd"],
                values["welfare"],
                values["exclusion"],
            ),
        ),
        {},
        (),
    )

    scored = score_residual_candidate(
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        0.0,
        candidate,
        anchor,
        {"Poland/A": CityMetrics(0.2, 100.0, 0.1)},
        "nonfinite-anchor",
    )

    assert scored.safe is False
    assert math.isinf(scored.loss)
    assert any(
        "anchor" in failure.lower()
        and f"non-finite {field}" in failure.lower()
        for failure in scored.metrics.failed_safety
    )


@pytest.mark.parametrize(
    "mes",
    (
        CityMetrics(float("nan"), 100.0, 0.1),
        CityMetrics(0.2, float("nan"), 0.1),
        CityMetrics(0.2, 100.0, float("nan")),
    ),
    ids=("csd", "welfare", "exclusion"),
)
def test_candidate_safety_rejects_and_retains_nonfinite_mes_reference(
    mes: CityMetrics,
) -> None:
    metrics = _residual_metrics(
        (("Poland/A/one", "Poland/A", 0.2, 100.0, 0.1),)
    )

    scored = score_residual_candidate(
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        0.0,
        metrics,
        metrics,
        {"Poland/A": mes},
        "nonfinite-mes",
    )

    assert scored.safe is False
    assert math.isinf(scored.loss)
    assert any(
        "MES" in failure and "non-finite" in failure
        for failure in scored.metrics.failed_safety
    )


def test_residual_candidate_requires_weights_equal_scaled_endpoint() -> None:
    metrics = _residual_metrics(
        (("Poland/A/one", "Poland/A", 0.2, 100.0, 0.1),)
    )

    with pytest.raises(ValueError, match="weights must equal amplitude times endpoint"):
        ResidualCandidate(
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            0.25,
            0.0,
            metrics,
            True,
            "malformed",
        )


def test_residual_candidate_canonicalizes_every_signed_zero() -> None:
    candidate = ResidualCandidate(
        (-0.0, 0.0, -0.0, 0.0),
        (-0.0, 0.0, -0.0, 0.0),
        0.0,
        0.0,
        _residual_metrics(
            (("Poland/A/one", "Poland/A", 0.2, 100.0, 0.1),)
        ),
        True,
        "signed-zero",
    )

    assert all(math.copysign(1.0, value) == 1.0 for value in candidate.weights)
    assert all(math.copysign(1.0, value) == 1.0 for value in candidate.endpoint)


@pytest.mark.parametrize(
    ("safe", "loss", "failed_safety"),
    (
        (True, 0.0, ("raw failure",)),
        (True, float("inf"), ()),
        (False, -1.0, ("unsafe",)),
        (False, float("inf"), ()),
        (False, float("-inf"), ("unsafe",)),
    ),
)
def test_residual_candidate_enforces_safety_failure_and_loss_consistency(
    safe: bool, loss: float, failed_safety: tuple[str, ...]
) -> None:
    base = _residual_metrics(
        (("Poland/A/one", "Poland/A", 0.2, 100.0, 0.1),)
    )
    metrics = ResidualMetrics(base.series, base.cities, failed_safety)

    with pytest.raises(ValueError, match="candidate safety state is inconsistent"):
        ResidualCandidate(
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            0.0,
            loss,
            metrics,
            safe,
            "inconsistent",
        )


def test_residual_seed_fit_requires_nonempty_unique_archive_and_selected_member() -> None:
    zero = _valid_zero_candidate()
    nonzero = _valid_nonzero_candidate()

    with pytest.raises(ValueError, match="nonempty residual archive"):
        ResidualSeedFit(1, zero, (), ())
    with pytest.raises(ValueError, match="payload-unique"):
        ResidualSeedFit(1, zero, (zero, replace(zero, source="duplicate")), ())
    with pytest.raises(ValueError, match="exactly one canonical safe zero"):
        ResidualSeedFit(1, nonzero, (nonzero,), ())
    with pytest.raises(ValueError, match="selected candidate must occur in archive"):
        ResidualSeedFit(1, nonzero, (zero,), ())


def test_residual_seed_fit_rejects_selected_noncanonical_zero_weight_row() -> None:
    zero = _valid_zero_candidate()
    noncanonical = _residual_candidate(
        (0.0, 0.0, 0.0, 0.0),
        -1.0,
        amplitude=0.75,
        source="invalid-selected-zero-weight",
    )

    with pytest.raises(
        ValueError, match="selected zero-weight candidate must be canonical"
    ):
        ResidualSeedFit(1, noncanonical, (zero, noncanonical), ())


def test_select_primary_seed_rejects_duplicate_ids_and_invalid_selections() -> None:
    seed_one = _valid_seed_fit(1)
    duplicate_one = _valid_seed_fit(1)
    with pytest.raises(ValueError, match="duplicate residual seed"):
        select_primary_seed((seed_one, duplicate_one))

    invalid_selected = SimpleNamespace(
        safe=False,
        loss=float("inf"),
        amplitude=0.0,
        weights=(0.0, 0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="safe finite selection"):
        select_primary_seed((SimpleNamespace(seed=1, selected=invalid_selected),))

    nonfinite_selected = SimpleNamespace(
        safe=True,
        loss=float("nan"),
        amplitude=0.0,
        weights=(0.0, 0.0, 0.0, 0.0),
    )
    with pytest.raises(ValueError, match="safe finite selection"):
        select_primary_seed((SimpleNamespace(seed=1, selected=nonfinite_selected),))


def test_fold_fit_requires_exact_configured_seed_order_and_computed_primary() -> None:
    anchor_fit, scaler, _ = _anchor_fit_scaler_and_data()
    config = residual_search_config()
    seed_fits = tuple(_valid_seed_fit(seed) for seed in config.seeds)

    with pytest.raises(ValueError, match="seed order must equal configured seeds"):
        residual_train_module.FoldFit(
            anchor_fit, scaler, seed_fits[:-1], 1, config
        )
    with pytest.raises(ValueError, match="seed order must equal configured seeds"):
        residual_train_module.FoldFit(
            anchor_fit, scaler, tuple(reversed(seed_fits)), 1, config
        )
    with pytest.raises(ValueError, match="primary seed must equal training selection"):
        residual_train_module.FoldFit(
            anchor_fit, scaler, seed_fits, 42, config
        )


def test_evaluate_ray_zero_is_positive_and_payload_identical_for_endpoint_sign() -> None:
    def evaluator(weights):
        return _residual_candidate(tuple(weights), loss=0.0, source="same-source")

    negative = evaluate_ray((-1.0, -2.0, -3.0, -4.0), evaluator)[0]
    positive = evaluate_ray((1.0, 2.0, 3.0, 4.0), evaluator)[0]
    negative_payload = residual_train_module._residual_candidate_payload(negative)
    positive_payload = residual_train_module._residual_candidate_payload(positive)

    assert all(math.copysign(1.0, value) == 1.0 for value in negative.weights)
    assert all(math.copysign(1.0, value) == 1.0 for value in negative.endpoint)
    assert negative_payload == positive_payload
    assert residual_train_module.residual_candidate_sha256(negative) == (
        residual_train_module.residual_candidate_sha256(positive)
    )


def test_training_sanitizer_accepts_canonical_ref_year_superset() -> None:
    instance = _smoke_training_data()[0].train_only[2022]
    row = SeriesData(
        ref=SeriesRef("Poland/A/one", (2022, 2023), ()),
        train_years=(2022,),
        test_years=(),
        train_only={2022: instance},
        all_years={2022: instance},
    )

    sanitized = residual_train_module._training_only_data((row,))

    assert sanitized[0].ref == SeriesRef("Poland/A/one", (2022,), ())
    assert sanitized[0].test_years == ()
    assert set(sanitized[0].train_only) == {2022}
    assert set(sanitized[0].all_years) == {2022}


@pytest.mark.parametrize("violation", ("extra_all", "extra_train", "absent_ref"))
def test_training_sanitizer_rejects_nontraining_keys_and_absent_train_year(
    violation: str,
) -> None:
    instance = _smoke_training_data()[0].train_only[2022]
    row = SeriesData(
        ref=SeriesRef("Poland/A/one", (2022,), ()),
        train_years=(2022,),
        test_years=(),
        train_only={2022: instance},
        all_years={2022: instance},
    )
    if violation == "extra_all":
        row = replace(row, all_years={2022: instance, 2023: object()})
    elif violation == "extra_train":
        row = replace(
            row,
            train_only={2022: instance, 2023: object()},
            all_years={2022: instance, 2023: object()},
        )
    else:
        row = replace(row, ref=SeriesRef("Poland/A/one", (2021,), ()))

    with pytest.raises(ValueError, match="training|held-out"):
        residual_train_module._training_only_data((row,))


def _canonical_json_bytes(payload) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


@pytest.mark.parametrize("static_state", ("missing", "stale", "mismatched"))
def test_smoke_rejects_unauthenticated_static_before_residual_fit_or_write(
    tmp_path, monkeypatch, static_state: str
) -> None:
    anchor_fit, _, _ = _anchor_fit_scaler_and_data()
    static_path = tmp_path / "static-anchor.json"
    residual_path = tmp_path / "residual-fit.json"
    if static_state == "stale":
        static_path.write_bytes(b'{"stale":true}\n')
    elif static_state == "mismatched":
        static_path.write_bytes(_canonical_json_bytes({"schema_version": 1}))
    monkeypatch.setattr(residual_train_module, "SMOKE_ARTIFACT", static_path)
    monkeypatch.setattr(
        residual_train_module, "SMOKE_RESIDUAL_ARTIFACT", residual_path
    )
    monkeypatch.setattr(
        residual_train_module, "fit_static_anchor", lambda *args, **kwargs: anchor_fit
    )
    monkeypatch.setattr(
        residual_train_module,
        "_fit_residual_fold_with_config",
        lambda *args, **kwargs: pytest.fail("residual fitting must not start"),
    )

    with pytest.raises((FileNotFoundError, ValueError), match="static"):
        residual_train_module._run_smoke_residual(workers=1)

    assert residual_path.exists() is False


def test_smoke_authenticates_and_embeds_static_hash_before_payload_hash(
    tmp_path, monkeypatch
) -> None:
    anchor_fit, _, _ = _anchor_fit_scaler_and_data()
    static_path = tmp_path / "static-anchor.json"
    residual_path = tmp_path / "residual-fit.json"
    static_bytes = _canonical_json_bytes(residual_train_module._fit_payload(anchor_fit))
    static_path.write_bytes(static_bytes)
    static_sha = hashlib.sha256(static_bytes).hexdigest()
    monkeypatch.setattr(residual_train_module, "SMOKE_ARTIFACT", static_path)
    monkeypatch.setattr(
        residual_train_module, "SMOKE_RESIDUAL_ARTIFACT", residual_path
    )
    monkeypatch.setattr(
        residual_train_module, "fit_static_anchor", lambda *args, **kwargs: anchor_fit
    )

    def fake_fold(data, passed_anchor, scaler, workers, config):
        del data, workers
        seed_fits = tuple(_valid_seed_fit(seed) for seed in config.seeds)
        return residual_train_module.FoldFit(
            passed_anchor, scaler, seed_fits, 1, config
        )

    monkeypatch.setattr(
        residual_train_module, "_fit_residual_fold_with_config", fake_fold
    )

    assert residual_train_module._run_smoke_residual(workers=1) == 0
    payload = json.loads(residual_path.read_text())
    embedded_payload_sha = payload.pop("payload_sha256")

    assert payload["static_artifact_sha256"] == static_sha
    assert residual_train_module._canonical_sha256(payload) == embedded_payload_sha


def test_residual_payload_actuation_uses_policy_weights_not_amplitude() -> None:
    anchor_fit, scaler, _ = _anchor_fit_scaler_and_data()
    zero = _valid_zero_candidate()
    noncanonical = _residual_candidate(
        (0.0, 0.0, 0.0, 0.0),
        -1.0,
        amplitude=0.75,
        source="payload-zero-weight",
    )
    seed = SimpleNamespace(
        seed=1,
        selected=noncanonical,
        archive=(zero, noncanonical),
        optimizer_history=(),
    )
    fit = SimpleNamespace(
        anchor_fit=anchor_fit,
        scaler=scaler,
        seeds=(seed,),
        primary_seed=1,
        config=replace(smoke_residual_search_config(), seeds=(1,)),
    )

    payload = residual_train_module._residual_fit_payload(fit, "0" * 64)

    assert payload["synthetic_actuated"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (("tau", 0.03), ("welfare_floor", 0.98), ("norm_penalty", 0.002)),
)
def test_configured_fit_rejects_altered_frozen_scoring_constants(
    monkeypatch, field: str, value: float
) -> None:
    anchor_fit, scaler, data = _anchor_fit_scaler_and_data()
    config = replace(smoke_residual_search_config(), **{field: value})
    monkeypatch.setattr(
        residual_train_module,
        "minimize_batch",
        lambda *args, **kwargs: pytest.fail("invalid config reached optimizer"),
    )

    with pytest.raises(ValueError, match="frozen residual scoring constants"):
        residual_train_module._fit_residual_seed_with_config(
            data, anchor_fit, scaler, 1, 1, config
        )


def _assert_worker_state_is_clear() -> None:
    assert residual_train_module._RESIDUAL_WORKER_DATA == ()
    assert residual_train_module._RESIDUAL_WORKER_ANCHOR is None
    assert residual_train_module._RESIDUAL_WORKER_SCALER is None
    assert residual_train_module._RESIDUAL_WORKER_ANCHOR_METRICS is None
    assert residual_train_module._RESIDUAL_WORKER_MES_METRICS is None


def test_worker_initializer_failure_first_leaves_atomic_clear_state(monkeypatch) -> None:
    anchor_fit, scaler, data = _anchor_fit_scaler_and_data()
    monkeypatch.setattr(
        residual_train_module,
        "_residual_reference_metrics",
        lambda *args: (_ for _ in ()).throw(ValueError("forced reference failure")),
    )

    with pytest.raises(ValueError, match="forced reference failure"):
        residual_train_module._initialize_residual_worker(
            tuple(data), anchor_fit.selected.anchor, scaler
        )

    _assert_worker_state_is_clear()
    with pytest.raises(RuntimeError, match="not initialized"):
        residual_train_module._residual_worker_evaluate(
            ((0.0, 0.0, 0.0, 0.0), 0.0, "after-failure")
        )


def test_worker_success_then_failure_clears_prior_context(monkeypatch) -> None:
    anchor_fit, scaler, data = _anchor_fit_scaler_and_data()
    residual_train_module._initialize_residual_worker(
        tuple(data), anchor_fit.selected.anchor, scaler
    )
    monkeypatch.setattr(
        residual_train_module,
        "_residual_reference_metrics",
        lambda *args: (_ for _ in ()).throw(ValueError("forced replacement failure")),
    )

    with pytest.raises(ValueError, match="forced replacement failure"):
        residual_train_module._initialize_residual_worker(
            tuple(data), anchor_fit.selected.anchor, scaler
        )

    _assert_worker_state_is_clear()
    with pytest.raises(RuntimeError, match="not initialized"):
        residual_train_module._residual_worker_evaluate(
            ((0.0, 0.0, 0.0, 0.0), 0.0, "after-replacement-failure")
        )


def _development_instance(path: str, year: int) -> PBInstance:
    return PBInstance(
        path=path,
        meta={"budget": "1", "instance": f"synthetic_{year}"},
        projects={"p": Project("p", 1.0, None)},
        votes=[Vote("v", ("p",), age=40, sex="F")],
    )


def _development_series(
    key: str,
    year: int,
    *,
    scored: bool = False,
) -> SeriesData:
    instance = _development_instance(f"synthetic://{key}/{year}", year)
    return SeriesData(
        ref=SeriesRef(key, (year,), (Path(f"{key.replace('/', '_')}_{year}.pb"),)),
        train_years=() if scored else (year,),
        test_years=(year,) if scored else (),
        train_only={} if scored else {year: instance},
        all_years={year: instance},
    )


def _development_result(
    *,
    fold: str,
    seed: int,
    series: str,
    residual_csd: float = 0.8,
    anchor_csd: float = 1.0,
    residual_welfare: float = 100.0,
    anchor_welfare: float = 100.0,
    mes_welfare: float = 100.0,
    residual_exclusion: float = 0.1,
    anchor_exclusion: float = 0.1,
    mes_exclusion: float = 0.1,
    actuated: bool = False,
) -> DevelopmentSeriesResult:
    city = "/".join(series.split("/")[:2])
    return DevelopmentSeriesResult(
        fold=fold,
        seed=seed,
        series=series,
        city=city,
        cluster=development_cluster_key(series),
        residual={
            "csd": residual_csd,
            "welfare": residual_welfare,
            "exclusion": residual_exclusion,
        },
        anchor={
            "csd": anchor_csd,
            "welfare": anchor_welfare,
            "exclusion": anchor_exclusion,
        },
        mes={
            "csd": anchor_csd,
            "welfare": mes_welfare,
            "exclusion": mes_exclusion,
        },
        signatures={
            "residual": "actuated" if actuated else "anchor-equivalent",
            "anchor": "anchor-equivalent",
            "actuated": actuated,
        },
    )


def _passing_development_payload() -> dict[str, object]:
    city_rows = []
    cities = ("Poland/Gdynia", "Poland/Warszawa", "Poland/Łódź")
    for seed in (1, 2, 42):
        for city_index, city in enumerate(cities):
            for series_index in range(2):
                city_rows.append(
                    _development_result(
                        fold=f"city_out_{city.replace('/', '_')}",
                        seed=seed,
                        series=f"{city}/district-{series_index}",
                        actuated=seed == 42,
                    )
                )
    district_rows = [
        _development_result(
            fold=f"district_out_f{fold}of5",
            seed=seed,
            series=f"Poland/Warszawa/fold-{fold}",
            residual_csd=(
                0.8 if fold < 4 else (1.0 if seed == 42 else 1.1)
            ),
            actuated=seed == 42 and fold < 4,
        )
        for seed in (1, 2, 42)
        for fold in range(5)
    ]
    expected_scored_series = {
        **{
            f"city_out_{city.replace('/', '_')}": [
                f"{city}/district-{series_index}" for series_index in range(2)
            ]
            for city in cities
        },
        **{
            f"district_out_f{fold}of5": [f"Poland/Warszawa/fold-{fold}"]
            for fold in range(5)
        },
    }
    expected_safe_static = {
        fold: _development_safe_static_inventory_for_test()
        for fold in expected_scored_series
    }
    safe_static_results = {
        fold: _development_safe_static_results_for_test(series_ids)
        for fold, series_ids in expected_scored_series.items()
    }
    return {
        "schema_version": 1,
        "primary_seed": 42,
        "expected_scored_series": expected_scored_series,
        "expected_safe_static": expected_safe_static,
        "safe_static_results": safe_static_results,
        "city_results": city_rows,
        "district_results": district_rows,
    }


def _development_static_fit_for_test(
    config=None,
) -> dict[str, object]:
    candidate = _candidate("mes", (), 1.0)
    fit = AnchorFit(
        candidate,
        (candidate,),
        candidate.metrics,
        anchor_search_config() if config is None else config,
    )
    return residual_train_module._fit_payload(fit)


def _complete_development_static_fit_for_test(
    fold: DevelopmentFold,
    config=None,
) -> tuple[AnchorFit, dict[str, object]]:
    config = anchor_search_config() if config is None else config
    cities = sorted(
        {"/".join(row.ref.key.split("/")[:2]) for row in fold.train}
    )
    mes_metrics = {
        city: CityMetrics(1.0, 10.0, 0.1) for city in cities
    }
    unsafe_metrics = {
        city: CityMetrics(1.0, 0.0, 0.1) for city in cities
    }
    mes = AnchorCandidate(
        anchor=_anchor("mes", ()),
        objective=1.0,
        metrics=mes_metrics,
        safe=True,
        source="mes-fallback",
        seed=None,
    )
    candidates = [mes]
    candidates.extend(
        AnchorCandidate(
            anchor=_anchor("senior", (math.log1p(alpha),)),
            objective=1.0 if alpha == 0.0 else float("inf"),
            metrics=mes_metrics if alpha == 0.0 else unsafe_metrics,
            safe=alpha == 0.0,
            source=f"senior-grid-alpha-{alpha:.17g}",
            seed=None,
        )
        for alpha in senior_anchor_grid()
    )
    candidates.extend(
        (
            AnchorCandidate(
                anchor=_anchor("age", (0.0, 0.0, 0.0)),
                objective=1.0,
                metrics=mes_metrics,
                safe=True,
                source=f"age-seed-{config.seeds[0]}-initializer",
                seed=config.seeds[0],
            ),
            AnchorCandidate(
                anchor=_anchor("age_sex", (0.0,) * 7),
                objective=1.0,
                metrics=mes_metrics,
                safe=True,
                source=f"age_sex-seed-{config.seeds[0]}-initializer",
                seed=config.seeds[0],
            ),
        )
    )
    fit = AnchorFit(mes, tuple(candidates), mes_metrics, config)
    return fit, residual_train_module._fit_payload(fit)


def _complete_development_fold_fit_for_test(
    fold: DevelopmentFold,
    *,
    anchor_config=None,
    residual_config=None,
    primary_seed: int = 42,
) -> tuple[FoldFit, dict[str, object]]:
    anchor_config = (
        anchor_search_config() if anchor_config is None else anchor_config
    )
    residual_config = (
        residual_search_config() if residual_config is None else residual_config
    )
    anchor_fit, static_payload = _complete_development_static_fit_for_test(
        fold, anchor_config
    )
    training_series = tuple(sorted(row.ref.key for row in fold.train))
    rows = tuple(
        SeriesMetrics(
            series=series,
            city="/".join(series.split("/")[:2]),
            csd=1.0,
            welfare=10.0,
            exclusion=0.1,
        )
        for series in training_series
    )
    anchor_metrics = ResidualMetrics(
        rows,
        residual_train_module._canonical_city_metrics(rows),
        (),
    )
    zero = score_residual_candidate(
        (0.0,) * 4,
        (0.0,) * 4,
        0.0,
        anchor_metrics,
        anchor_metrics,
        anchor_fit.mes_metrics,
        "canonical-zero",
    )
    seed_fits = []
    for index, seed in enumerate(residual_config.seeds):
        reduction = 0.006 if seed == primary_seed else 0.004 + 0.0005 * index
        improved_rows = tuple(
            replace(row, csd=row.csd - reduction) for row in rows
        )
        improved_metrics = ResidualMetrics(
            improved_rows,
            residual_train_module._canonical_city_metrics(improved_rows),
            (),
        )
        endpoint = (0.16 + 0.08 * index, 0.0, 0.0, 0.0)
        amplitude = 0.25
        weights = tuple(amplitude * value for value in endpoint)
        improved = score_residual_candidate(
            weights,
            endpoint,
            amplitude,
            improved_metrics,
            anchor_metrics,
            anchor_fit.mes_metrics,
            f"seed-{seed}-improved",
        )
        seed_fits.append(
            ResidualSeedFit(
                seed,
                improved,
                (zero, improved),
                ({"generation": 0, "best_loss": improved.loss},),
            )
        )
    instances = tuple(
        row.train_only[year]
        for row in fold.train
        for year in sorted(row.train_only)
    )
    scaler = fit_context_scaler(instances)
    fit = FoldFit(anchor_fit, scaler, tuple(seed_fits), primary_seed, residual_config)
    return fit, {
        "primary_seed": fit.primary_seed,
        "static": static_payload,
        "residual": residual_train_module._residual_fit_payload(
            fit, residual_train_module._canonical_sha256(static_payload)
        ),
    }


def _complete_development_artifact_pair_for_test(
    fold: DevelopmentFold,
    result: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    _, fit_payload = _complete_development_fold_fit_for_test(fold)
    expected_safe_static = _development_safe_static_inventory_for_test()
    fold_input = residual_train_module._fold_input_payload(fold)
    common = {
        "schema_version": 2,
        "fold": fold.name,
        "kind": fold.kind,
        "source_manifest_sha256": fold.source_manifest_sha256,
        "provenance": dict(fold.provenance),
        "fold_input": fold_input,
        "fold_input_sha256": residual_train_module._canonical_sha256(fold_input),
        "primary_seed": 42,
        "expected_safe_static": expected_safe_static,
    }
    fit_record = residual_train_module._with_payload_sha(
        {**common, "fit": fit_payload}
    )
    series_record = residual_train_module._with_payload_sha(
        {
            **common,
            "fit_payload_sha256": fit_record["payload_sha256"],
            "result": result,
        }
    )
    return fit_record, series_record


def _complete_development_result_for_test(
    fold: DevelopmentFold,
) -> dict[str, object]:
    series_ids = [row.ref.key for row in fold.test]
    result = {
        "fold": fold.name,
        "kind": fold.kind,
        "primary_seed": 42,
        "fit_training_series": [row.ref.key for row in fold.train],
        "scored_series": series_ids,
        "series_results": [
            _development_result_payload_for_test(
                _development_result(
                    fold=fold.name,
                    seed=seed,
                    series=series,
                    residual_csd=1.0 if seed == 42 else 0.8,
                )
            )
            for seed in (1, 2, 42)
            for series in series_ids
        ],
        "safe_static": _development_safe_static_results_for_test(series_ids),
        "actuation": [],
    }
    _attach_exact_actuation_for_test(fold, result, set())
    return result


def _complete_fit_for_training_rows_for_test(
    rows,
    *,
    primary_seed: int = 42,
) -> FoldFit:
    fold = DevelopmentFold(
        "unit_training_fold", "city", tuple(rows), (), "f" * 64
    )
    fit, _ = _complete_development_fold_fit_for_test(
        fold, primary_seed=primary_seed
    )
    return fit


def _development_static_identity_for_test() -> str:
    return residual_train_module.anchor_payload_sha256(_anchor("mes", ()))


def _development_safe_static_inventory_for_test() -> list[dict[str, str]]:
    anchors = (
        _anchor("mes", ()),
        _anchor("senior", (0.0,)),
        _anchor("age", (0.0, 0.0, 0.0)),
        _anchor("age_sex", (0.0,) * 7),
    )
    return [
        {
            "identity": residual_train_module.anchor_payload_sha256(anchor),
            "family": anchor.family,
        }
        for anchor in anchors
    ]


def _development_safe_static_results_for_test(
    series_ids: list[str] | tuple[str, ...],
) -> dict[str, object]:
    return {
        row["identity"]: {
            "family": row["family"],
            "series": {
                series: {"csd": 1.0, "welfare": 100.0, "exclusion": 0.1}
                for series in series_ids
            },
        }
        for row in _development_safe_static_inventory_for_test()
    }


def _canonical_development_gate_folds() -> tuple[DevelopmentFold, ...]:
    city_series = {
        "city_out_Poland_Gdynia": (
            "Poland/Gdynia/district-0",
            "Poland/Gdynia/district-1",
        ),
        "city_out_Poland_Warszawa": (
            "Poland/Warszawa/district-0",
            "Poland/Warszawa/district-1",
        ),
        "city_out_Poland_Łódź": (
            "Poland/Łódź/district-0",
            "Poland/Łódź/district-1",
        ),
    }
    district_series = {
        f"district_out_f{fold}of5": (f"Poland/Warszawa/fold-{fold}",)
        for fold in range(5)
    }
    inventories = {**city_series, **district_series}
    return tuple(
        DevelopmentFold(
            fold,
            "city" if fold.startswith("city_") else "district",
            (_development_series(f"Poland/Training/fold-{index}", 2020),),
            tuple(_development_series(series, 2022, scored=True) for series in series_ids),
            f"{index + 1:x}" * 64,
            {"split_sha256": f"{index + 9:x}" * 64},
        )
        for index, (fold, series_ids) in enumerate(inventories.items())
    )


def _development_artifact_pair_for_test(
    fold: DevelopmentFold,
    result: dict[str, object],
    expected_safe_static: list[dict[str, str]],
) -> tuple[dict[str, object], dict[str, object]]:
    fold_input = residual_train_module._fold_input_payload(fold)
    fold_input_sha = residual_train_module._canonical_sha256(fold_input)
    _, fit_payload = _complete_development_fold_fit_for_test(
        fold, primary_seed=result["primary_seed"]
    )
    common = {
        "schema_version": 2,
        "fold": fold.name,
        "kind": fold.kind,
        "source_manifest_sha256": fold.source_manifest_sha256,
        "provenance": dict(fold.provenance),
        "fold_input": fold_input,
        "fold_input_sha256": fold_input_sha,
        "primary_seed": result["primary_seed"],
        "expected_safe_static": expected_safe_static,
    }
    fit_record = residual_train_module._with_payload_sha(
        {**common, "fit": fit_payload}
    )
    series_record = residual_train_module._with_payload_sha(
        {
            **common,
            "fit_payload_sha256": fit_record["payload_sha256"],
            "result": result,
        }
    )
    return fit_record, series_record


def _development_gate_inputs_for_test(
    payload: dict[str, object] | None = None,
) -> tuple[
    tuple[DevelopmentFold, ...],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
]:
    payload = _passing_development_payload() if payload is None else payload
    folds = _canonical_development_gate_folds()
    primary_seeds = payload.get("primary_seeds")
    fit_records = []
    series_records = []
    canonical_names = {fold.name for fold in folds}
    for fold in folds:
        primary_seed = (
            primary_seeds[fold.name]
            if primary_seeds is not None
            else payload["primary_seed"]
        )
        rows = (
            payload["city_results"]
            if fold.kind == "city"
            else payload["district_results"]
        )
        selected_rows = [
            _development_result_payload_for_test(row)
            if isinstance(row, DevelopmentSeriesResult)
            else copy.deepcopy(row)
            for row in rows
            if (
                isinstance(row, Mapping) and row.get("fold") == fold.name
            )
            or (
                isinstance(row, DevelopmentSeriesResult)
                and row.fold == fold.name
            )
        ]
        if fold.name == "district_out_f0of5":
            selected_rows.extend(
                _development_result_payload_for_test(row)
                if isinstance(row, DevelopmentSeriesResult)
                else copy.deepcopy(row)
                for row in (*payload["city_results"], *payload["district_results"])
                if (
                    row.get("fold")
                    if isinstance(row, Mapping)
                    else row.fold
                )
                not in canonical_names
            )
        result = {
            "fold": fold.name,
            "kind": fold.kind,
            "primary_seed": primary_seed,
            "fit_training_series": [row.ref.key for row in fold.train],
            "scored_series": [row.ref.key for row in fold.test],
            "series_results": selected_rows,
            "safe_static": copy.deepcopy(payload["safe_static_results"][fold.name]),
            "actuation": [],
        }
        actuated_series = {
            row["series"]
            for row in selected_rows
            if row["seed"] == primary_seed and row["signatures"]["actuated"]
        }
        _attach_exact_actuation_for_test(fold, result, actuated_series)
        expected_safe_static = _development_safe_static_inventory_for_test()
        fit_record, series_record = _development_artifact_pair_for_test(
            fold, result, expected_safe_static
        )
        fit_records.append(fit_record)
        series_records.append(series_record)
    return folds, tuple(fit_records), tuple(series_records)


def _trusted_development_gate_evidence(
    payload: dict[str, object] | None = None,
):
    return residual_train_module.DevelopmentGateEvidence(
        *_development_gate_inputs_for_test(payload)
    )


@pytest.mark.parametrize(
    ("series", "expected"),
    (
        ("Poland/Gdynia/place | large", "Poland/Gdynia/place"),
        ("Poland/Gdynia/place | SMALL", "Poland/Gdynia/place"),
        ("Poland/Gdynia/place__large", "Poland/Gdynia/place"),
        ("Poland/Gdynia/place__SMALL", "Poland/Gdynia/place"),
        ("Poland/Gdynia/place__large_projects", "Poland/Gdynia/place"),
        ("Poland/Gdynia/place__SMALL_PROJECTS", "Poland/Gdynia/place"),
        ("Poland/Gdynia/large | place", "Poland/Gdynia/large | place"),
        ("Poland/Gdynia/place__large_extra", "Poland/Gdynia/place__large_extra"),
    ),
)
def test_development_cluster_key_strips_only_terminal_pool_labels(
    series: str, expected: str
) -> None:
    assert development_cluster_key(series) == expected


def test_cluster_city_bootstrap_forms_cluster_means_before_equal_city_mean() -> None:
    rows = [
        _development_result(
            fold="city-a", seed=42, series="Poland/A/pool | large", residual_csd=0.0
        ),
        _development_result(
            fold="city-a", seed=42, series="Poland/A/pool | small", residual_csd=0.0
        ),
        _development_result(
            fold="city-a", seed=42, series="Poland/A/other", residual_csd=1.0
        ),
        _development_result(
            fold="city-b", seed=42, series="Poland/B/only", residual_csd=1.0
        ),
        _development_result(
            fold="city-c", seed=42, series="Poland/C/only", residual_csd=1.0
        ),
    ]

    result = cluster_city_bootstrap(rows, draws=20000, seed=20260818)

    assert result["mean"] == pytest.approx((-0.5 + 0.0 + 0.0) / 3.0)
    assert result["draws"] == 20000
    assert result["seed"] == 20260818
    assert result["series_count"] == 5
    assert result["cluster_count"] == 4
    assert result["city_count"] == 3
    json.dumps(result, allow_nan=False)


def test_development_public_records_are_defensive_restartable_and_json_ready() -> None:
    train = [_development_series("Poland/A/train", 2022)]
    test = [_development_series("Poland/B/test", 2023, scored=True)]
    fold = DevelopmentFold("city_out_Poland_B", "city", train, test, "a" * 64)
    source = {"csd": 0.8, "welfare": 100.0, "exclusion": 0.1}
    row = DevelopmentSeriesResult(
        "city_out_Poland_B",
        42,
        "Poland/B/test",
        "Poland/B",
        "Poland/B/test",
        source,
        {"csd": 1.0, "welfare": 100.0, "exclusion": 0.1},
        {"csd": 1.0, "welfare": 100.0, "exclusion": 0.1},
        {
            "residual": "actuated",
            "anchor": "anchor-equivalent",
            "actuated": True,
        },
    )
    train.append(_development_series("Poland/A/late", 2021))
    source["csd"] = 99.0

    assert len(fold.train) == 1
    assert row.residual["csd"] == 0.8
    with pytest.raises(TypeError):
        row.residual["csd"] = 1.0
    for copied in (copy.copy(row), copy.deepcopy(row), pickle.loads(pickle.dumps(row))):
        assert copied == row
    payload = asdict(row)
    json.dumps(payload, sort_keys=True, allow_nan=False)
    json.dumps(asdict(fold), sort_keys=True, allow_nan=False)


def test_development_fold_rejects_overlap_and_invalid_manifest_identity() -> None:
    row = _development_series("Poland/A/one", 2022)
    scored = _development_series("Poland/A/one", 2022, scored=True)
    with pytest.raises(ValueError, match="disjoint"):
        DevelopmentFold("bad", "city", (row,), (scored,), "a" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        DevelopmentFold("bad", "city", (row,), (), "not-a-hash")


def test_development_fold_defensively_copies_mutable_series_views() -> None:
    source = _development_series("Poland/A/one", 2022)
    fold = DevelopmentFold("fold", "city", (source,), (), "a" * 64)

    source.train_years = ()
    source.train_only.clear()

    assert fold.train[0].train_years == (2022,)
    assert set(fold.train[0].train_only) == {2022}


def test_evaluate_development_gate_passing_fixture_records_all_conditions() -> None:
    result = evaluate_development_gate(_trusted_development_gate_evidence())

    assert result["classification"] == "pass"
    assert result["failed_conditions"] == []
    assert set(result["conditions"]) == {
        "primary_improves_anchor",
        "all_held_cities_negative",
        "district_fold_majority",
        "dual_reference_safety",
        "all_seeds_negative",
        "actuation_coverage",
    }
    assert all(result["conditions"].values())
    assert result["thresholds"]["bootstrap_draws"] == 20000
    assert result["thresholds"]["bootstrap_seed"] == 20260818
    json.dumps(result, sort_keys=True, allow_nan=False)


def _replace_development_rows(
    payload: dict[str, object],
    predicate,
    **changes,
) -> dict[str, object]:
    cloned = copy.deepcopy(payload)
    cloned["city_results"] = [
        replace(row, **changes) if predicate(row) else row
        for row in cloned["city_results"]
    ]
    return cloned


@pytest.mark.parametrize(
    ("condition", "mutate"),
    (
        (
            "primary_improves_anchor",
            lambda payload: {
                **copy.deepcopy(payload),
                "city_results": [
                    replace(
                        row,
                        residual={
                            "csd": (
                                0.0 if row.series.endswith("district-0") else 1.9
                            ),
                            "welfare": 100.0,
                            "exclusion": 0.1,
                        },
                    )
                    if row.seed == 42
                    else row
                    for row in payload["city_results"]
                ],
            },
        ),
        (
            "all_held_cities_negative",
            lambda payload: _replace_development_rows(
                payload,
                lambda row: row.seed == 42 and row.city == "Poland/Gdynia",
                residual={"csd": 1.1, "welfare": 100.0, "exclusion": 0.1},
            ),
        ),
        (
            "district_fold_majority",
            lambda payload: {
                **copy.deepcopy(payload),
                "district_results": [
                    replace(
                        row,
                        residual={
                            "csd": (
                                1.0
                                if row.seed == 42
                                and row.fold in {
                                    "district_out_f3of5",
                                    "district_out_f4of5",
                                }
                                else row.residual["csd"]
                            ),
                            "welfare": 100.0,
                            "exclusion": 0.1,
                        },
                    )
                    for row in payload["district_results"]
                ],
            },
        ),
        (
            "dual_reference_safety",
            lambda payload: _replace_development_rows(
                payload,
                lambda row: row.seed == 42 and row.city == "Poland/Gdynia",
                residual={"csd": 0.8, "welfare": 98.9, "exclusion": 0.1},
            ),
        ),
        (
            "all_seeds_negative",
            lambda payload: _replace_development_rows(
                payload,
                lambda row: row.seed == 1,
                residual={"csd": 1.1, "welfare": 100.0, "exclusion": 0.1},
            ),
        ),
    ),
)
def test_each_development_gate_condition_fails_independently(
    condition: str, mutate
) -> None:
    result = evaluate_development_gate(
        _trusted_development_gate_evidence(mutate(_passing_development_payload()))
    )

    assert result["classification"] == "fail"
    assert result["failed_conditions"] == [condition]


def test_actuation_coverage_failure_uses_consistent_inert_primary_rows() -> None:
    payload = _passing_development_payload()
    payload["city_results"] = [
        replace(
            row,
            residual=(
                dict(row.anchor)
                if row.seed == 42 and row.city != "Poland/Gdynia"
                else row.residual
            ),
            signatures=(
                {
                    "residual": "anchor-equivalent",
                    "anchor": "anchor-equivalent",
                    "actuated": False,
                }
                if row.seed == 42 and row.city != "Poland/Gdynia"
                else row.signatures
            ),
        )
        for row in payload["city_results"]
    ]

    result = evaluate_development_gate(_trusted_development_gate_evidence(payload))

    assert result["conditions"]["actuation_coverage"] is False
    assert set(result["failed_conditions"]) == {
        "all_held_cities_negative",
        "actuation_coverage",
    }


def test_exactly_four_negative_district_folds_pass_and_three_fail() -> None:
    passing = _passing_development_payload()
    assert evaluate_development_gate(_trusted_development_gate_evidence(passing))["conditions"][
        "district_fold_majority"
    ] is True
    failing = copy.deepcopy(passing)
    failing["district_results"] = [
        replace(
            row,
            residual={
                "csd": (
                    1.0
                    if row.seed == 42
                    and row.fold in {
                        "district_out_f3of5",
                        "district_out_f4of5",
                    }
                    else row.residual["csd"]
                ),
                "welfare": 100.0,
                "exclusion": 0.1,
            },
        )
        for row in failing["district_results"]
    ]
    assert evaluate_development_gate(_trusted_development_gate_evidence(failing))["conditions"][
        "district_fold_majority"
    ] is False


def test_gate_uses_training_selected_primary_seed_independently_per_fold() -> None:
    payload = _passing_development_payload()
    selected = {
        "city_out_Poland_Gdynia": 1,
        "city_out_Poland_Warszawa": 2,
        "city_out_Poland_Łódź": 42,
        **{f"district_out_f{fold}of5": (1, 2, 42, 1, 2)[fold] for fold in range(5)},
    }
    payload.pop("primary_seed")
    payload["primary_seeds"] = selected
    for family in ("city_results", "district_results"):
        payload[family] = [
            replace(
                row,
                signatures={
                    "residual": (
                        "actuated" if row.seed == selected[row.fold]
                        and dict(row.residual) != dict(row.anchor)
                        else "anchor-equivalent"
                    ),
                    "anchor": "anchor-equivalent",
                    "actuated": row.seed == selected[row.fold]
                    and dict(row.residual) != dict(row.anchor),
                },
            )
            for row in payload[family]
        ]
    assert evaluate_development_gate(
        _trusted_development_gate_evidence(payload)
    )["classification"] == "pass"


def test_gate_safety_uses_city_aggregate_not_individual_series_welfare() -> None:
    payload = _passing_development_payload()
    payload["city_results"] = [
        replace(
            row,
            residual={
                "csd": 0.8,
                "welfare": 98.0 if row.series.endswith("district-0") else 102.0,
                "exclusion": 0.1,
            },
        )
        if row.seed == 42 and row.city == "Poland/Gdynia"
        else row
        for row in payload["city_results"]
    ]

    assert evaluate_development_gate(_trusted_development_gate_evidence(payload))["conditions"][
        "dual_reference_safety"
    ] is True


@pytest.mark.parametrize(
    "corruption",
    (
        "missing_city",
        "missing_seed",
        "missing_series",
        "missing_district",
        "duplicate",
        "unexpected_fold",
        "family",
    ),
)
def test_gate_rejects_incomplete_or_duplicate_development_evidence(
    corruption: str,
) -> None:
    payload = _passing_development_payload()
    if corruption == "missing_city":
        payload["city_results"] = [
            row for row in payload["city_results"] if row.city != "Poland/Gdynia"
        ]
    elif corruption == "missing_seed":
        payload["city_results"] = [
            row for row in payload["city_results"] if row.seed != 2
        ]
    elif corruption == "missing_series":
        target = payload["city_results"][0]
        payload["city_results"] = [
            row
            for row in payload["city_results"]
            if not (
                row.seed == target.seed
                and row.fold == target.fold
                and row.series == target.series
            )
        ]
    elif corruption == "missing_district":
        payload["district_results"] = payload["district_results"][:-1]
    elif corruption == "duplicate":
        payload["city_results"].append(payload["city_results"][0])
    elif corruption == "unexpected_fold":
        payload["city_results"][0] = replace(
            payload["city_results"][0], fold="city_out_Poland_Unexpected"
        )
    else:
        row = _development_result_payload_for_test(payload["city_results"][0])
        row["residual"].pop("exclusion")
        payload["city_results"][0] = row

    with pytest.raises(RuntimeError, match="development evidence"):
        evaluate_development_gate(_trusted_development_gate_evidence(payload))


def _development_result_payload_for_test(
    row: DevelopmentSeriesResult,
) -> dict[str, object]:
    return {
        "fold": row.fold,
        "seed": row.seed,
        "series": row.series,
        "city": row.city,
        "cluster": row.cluster,
        "residual": dict(row.residual),
        "anchor": dict(row.anchor),
        "mes": dict(row.mes),
        "signatures": dict(row.signatures),
    }


def _fit_development_fold_unchecked_for_test(
    fold: DevelopmentFold, *, workers: int
):
    return residual_train_module._fit_and_evaluate_development_fold(
        fold,
        workers,
        root=residual_train_module.DEVELOPMENT_ROOT,
        smoke=False,
        expected_anchor_config=anchor_search_config(),
        expected_residual_config=residual_search_config(),
    )


def test_load_development_folds_requires_exact_ordered_inventory(monkeypatch) -> None:
    city = tuple(
        DevelopmentFold(
            f"city_out_Poland_{name}", "city", (), (), str(index) * 64
        )
        for index, name in enumerate(("Gdynia", "Warszawa", "Łódź"), start=1)
    )
    district = tuple(
        DevelopmentFold(
            f"district_out_f{index}of5", "district", (), (), str(index + 4) * 64
        )
        for index in range(5)
    )
    monkeypatch.setattr(residual_train_module, "_load_city_development_folds", lambda: city)
    monkeypatch.setattr(
        residual_train_module, "_load_district_development_folds", lambda: district
    )

    assert tuple(
        row.name
        for row in residual_train_module._load_development_folds_unchecked()
    ) == tuple(
        row.name for row in (*city, *district)
    )
    monkeypatch.setattr(
        residual_train_module,
        "_load_district_development_folds",
        lambda: (*district[:-1], district[0]),
    )
    with pytest.raises(RuntimeError, match="fold inventory"):
        residual_train_module._load_development_folds_unchecked()


def test_fold_execution_never_sends_scored_rows_to_fitting(monkeypatch, tmp_path) -> None:
    train = _development_series("Poland/A/train", 2022)
    scored = _development_series("Poland/B/scored", 2023, scored=True)
    fold = DevelopmentFold("city_out_Poland_B", "city", (train,), (scored,), "a" * 64)
    seen = []

    def assert_training(rows):
        keys = tuple(row.ref.key for row in rows)
        seen.append(keys)
        assert keys == ("Poland/A/train",)

    monkeypatch.setattr(residual_train_module, "DEVELOPMENT_ROOT", tmp_path)
    monkeypatch.setattr(
        residual_train_module,
        "_fit_development_training",
        lambda rows, workers, smoke=False: (
            assert_training(rows),
            _complete_fit_for_training_rows_for_test(rows),
        )[1],
    )
    monkeypatch.setattr(
        residual_train_module,
        "_evaluate_development_scored",
        lambda passed_fold, fit, workers: _valid_fake_development_result(
            passed_fold
        ),
    )

    result = _fit_development_fold_unchecked_for_test(fold, workers=1)

    assert seen == [("Poland/A/train",)]
    assert result["scored_series"] == ["Poland/B/scored"]


def test_fold_restart_accepts_identical_and_rejects_divergent_artifacts(
    monkeypatch, tmp_path
) -> None:
    fold = DevelopmentFold(
        "city_out_Poland_B",
        "city",
        (_development_series("Poland/A/train", 2022),),
        (_development_series("Poland/B/scored", 2023, scored=True),),
        "a" * 64,
    )
    monkeypatch.setattr(residual_train_module, "DEVELOPMENT_ROOT", tmp_path)
    monkeypatch.setattr(
        residual_train_module,
        "_fit_development_training",
        lambda rows, *args, **kwargs: _complete_fit_for_training_rows_for_test(rows),
    )
    monkeypatch.setattr(
        residual_train_module,
        "_evaluate_development_scored",
        lambda passed, fit, workers: _valid_fake_development_result(passed),
    )

    first = _fit_development_fold_unchecked_for_test(fold, workers=1)
    second = _fit_development_fold_unchecked_for_test(fold, workers=1)
    assert first == second
    path = tmp_path / "folds" / fold.name / "per_series.json"
    path.write_text('{"divergent":true}\n')
    with pytest.raises(RuntimeError, match="divergent"):
        _fit_development_fold_unchecked_for_test(fold, workers=1)


def test_fold_restart_rejects_rehashed_but_wrong_fit_link(monkeypatch, tmp_path) -> None:
    fold = DevelopmentFold(
        "city_out_Poland_B",
        "city",
        (_development_series("Poland/A/train", 2022),),
        (_development_series("Poland/B/scored", 2023, scored=True),),
        "a" * 64,
    )
    monkeypatch.setattr(residual_train_module, "DEVELOPMENT_ROOT", tmp_path)
    monkeypatch.setattr(
        residual_train_module,
        "_fit_development_training",
        lambda rows, *args, **kwargs: _complete_fit_for_training_rows_for_test(rows),
    )
    monkeypatch.setattr(
        residual_train_module,
        "_evaluate_development_scored",
        lambda passed, fit, workers: _valid_fake_development_result(passed),
    )
    _fit_development_fold_unchecked_for_test(fold, workers=1)
    path = tmp_path / "folds" / fold.name / "per_series.json"
    payload = json.loads(path.read_text())
    payload["fit_payload_sha256"] = "f" * 64
    payload.pop("payload_sha256")
    payload["payload_sha256"] = residual_train_module._canonical_sha256(payload)
    path.write_bytes(_canonical_json_bytes(payload))

    with pytest.raises(RuntimeError, match="divergent"):
        _fit_development_fold_unchecked_for_test(fold, workers=1)


def test_run_development_delegates_only_to_guarded_task6_execution(
    monkeypatch, tmp_path: Path,
) -> None:
    expected = {"classification": "fail", "guarded": True}
    calls = []
    result_root = tmp_path / "results-development"
    analysis_root = tmp_path / "analysis-development"
    monkeypatch.setattr(residual_train_module, "DEVELOPMENT_ROOT", result_root)
    monkeypatch.setattr(
        residual_train_module,
        "DEVELOPMENT_ANALYSIS_ROOT",
        analysis_root,
        raising=False,
    )
    monkeypatch.setattr(
        residual_train_module,
        "_run_guarded_development",
        lambda workers: calls.append(workers) or expected,
        raising=False,
    )
    monkeypatch.setattr(
        residual_train_module,
        "load_development_folds",
        lambda: pytest.fail("legacy RED reached the production fold loader"),
    )
    monkeypatch.setattr(
        residual_train_module,
        "fit_and_evaluate_development_fold",
        lambda *args, **kwargs: pytest.fail("legacy RED reached the production fitter"),
    )
    monkeypatch.setattr(
        residual_train_module,
        "_load_city_development_folds",
        lambda: pytest.fail("legacy RED reached the city loader"),
    )
    monkeypatch.setattr(
        residual_train_module,
        "_load_district_development_folds",
        lambda: pytest.fail("legacy RED reached the district loader"),
    )
    monkeypatch.setattr(
        residual_train_module,
        "_fit_and_evaluate_development_fold",
        lambda *args, **kwargs: pytest.fail("legacy RED reached the private fitter"),
    )
    monkeypatch.setattr(
        residual_train_module,
        "_atomic_write",
        lambda *args, **kwargs: pytest.fail("legacy RED reached a production write"),
        raising=False,
    )

    result = run_development(workers=8)

    assert calls == [8]
    assert result == expected
    assert not result_root.exists()
    assert not analysis_root.exists()


def test_development_cli_has_fixed_production_and_separate_smoke_workers() -> None:
    smoke = residual_train_module.parse_args(["smoke-development", "--workers", "2"])
    prepare = residual_train_module.parse_args(
        ["prepare-development", "--workers", "8"]
    )
    production = residual_train_module.parse_args(["development", "--workers", "8"])
    aggregate = residual_train_module.parse_args(
        ["aggregate-development", "--verify-only"]
    )

    assert (smoke.command, smoke.workers) == ("smoke-development", 2)
    assert (prepare.command, prepare.workers) == ("prepare-development", 8)
    assert (production.command, production.workers) == ("development", 8)
    assert (aggregate.command, aggregate.verify_only) == (
        "aggregate-development",
        True,
    )
    with pytest.raises(SystemExit):
        residual_train_module.parse_args(["run-development", "--workers", "8"])
    with pytest.raises(SystemExit):
        residual_train_module.parse_args(["development", "--workers", "8", "--generations", "1"])


def test_smoke_development_uses_synthetic_folds_without_production_loader(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(residual_train_module, "SMOKE_DEVELOPMENT_ARTIFACT", tmp_path / "development.json")
    monkeypatch.setattr(
        residual_train_module,
        "load_development_folds",
        lambda: pytest.fail("smoke must not read production folds"),
    )
    monkeypatch.setattr(
        residual_train_module,
        "_execute_smoke_development_fold",
        lambda fold, workers: {
            "fold": fold.name,
            "kind": fold.kind,
            "primary_seed": 42,
            "amplitude": 0.0,
            "weights": [0.0, 0.0, 0.0, 0.0],
            "loss": 0.0,
            "series_count": len(fold.test),
            "actuated": False,
        },
    )

    result = residual_train_module._run_smoke_development(workers=1)
    payload = json.loads((tmp_path / "development.json").read_text())

    assert result == 0
    assert payload["classification"] == "smoke_only"
    assert payload["fold_kinds"] == ["city", "district"]
    assert "workers" not in payload


def _assert_fold_nested_mutations_fail(fold: DevelopmentFold) -> None:
    series = fold.train[0]
    instance = series.train_only[2022]
    project = instance.projects["p"]
    vote = instance.votes[0]
    mutations = (
        lambda: setattr(series, "train_years", ()),
        lambda: series.train_only.clear(),
        lambda: series.all_years.clear(),
        lambda: instance.meta.__setitem__("budget", "999"),
        lambda: instance.projects.clear(),
        lambda: setattr(project, "cost", 999.0),
        lambda: instance.votes.clear(),
        lambda: setattr(vote, "vid", "changed"),
    )
    for mutate in mutations:
        with pytest.raises((AttributeError, TypeError)):
            mutate()


def test_development_fold_is_recursively_immutable_after_all_round_trips() -> None:
    source = _development_series("Poland/A/one", 2022)
    fold = DevelopmentFold("fold", "city", (source,), (), "a" * 64)
    source.train_only[2022].meta["budget"] = "777"
    source.train_only[2022].projects.clear()
    source.train_only[2022].votes.clear()

    for snapshot in (
        fold,
        copy.copy(fold),
        copy.deepcopy(fold),
        pickle.loads(pickle.dumps(fold)),
    ):
        _assert_fold_nested_mutations_fail(snapshot)
        assert snapshot.train[0].train_only[2022].meta["budget"] == "1"
        assert tuple(snapshot.train[0].train_only[2022].projects) == ("p",)
        assert len(snapshot.train[0].train_only[2022].votes) == 1
        json.dumps(asdict(snapshot), sort_keys=True, allow_nan=False)


def test_development_fold_materializes_fresh_private_mutable_runtime_values() -> None:
    fold = DevelopmentFold(
        "fold",
        "city",
        (_development_series("Poland/A/one", 2022),),
        (_development_series("Poland/B/two", 2023, scored=True),),
        "a" * 64,
    )

    first_train = residual_train_module._materialize_development_series(fold.train[0])
    second_train = residual_train_module._materialize_development_series(fold.train[0])
    first_test = residual_train_module._materialize_development_series(fold.test[0])

    assert isinstance(first_train, SeriesData)
    assert isinstance(first_train.train_only[2022], PBInstance)
    assert first_train is not fold.train[0]
    assert first_train.train_only[2022] is not second_train.train_only[2022]
    assert first_test.all_years[2023] is not fold.test[0].all_years[2023]
    first_train.train_only[2022].meta["budget"] = "999"
    assert fold.train[0].train_only[2022].meta["budget"] == "1"


def test_smoke_development_materializes_training_before_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fold = residual_train_module._smoke_development_folds()[0]

    def stop_after_check(
        rows: tuple[SeriesData, ...], workers: int, *, smoke: bool
    ) -> None:
        assert isinstance(rows[0], SeriesData)
        assert isinstance(rows[0].train_only[2022], PBInstance)
        assert workers == 1
        assert smoke is True
        raise RuntimeError("training rows were materialized")

    monkeypatch.setattr(
        residual_train_module, "_fit_development_training", stop_after_check
    )
    with pytest.raises(RuntimeError, match="training rows were materialized"):
        residual_train_module._execute_smoke_development_fold(fold, workers=1)


def _synthetic_district_index() -> dict[str, SeriesRef]:
    return {
        f"Poland/Warszawa/district-{index:02d}": SeriesRef(
            f"Poland/Warszawa/district-{index:02d}", (2020,), ()
        )
        for index in range(19)
    }


def _split_rows_without_loading(
    split: Split, index: dict[str, SeriesRef]
) -> tuple[tuple[SeriesData, ...], tuple[SeriesData, ...], dict[str, str]]:
    def row(key: str, years: tuple[int, ...], scored: bool) -> SeriesData:
        instances = {
            year: _development_instance(f"synthetic://{key}/{year}", year)
            for year in index[key].years
        }
        return SeriesData(
            ref=index[key],
            train_years=() if scored else years,
            test_years=years if scored else (),
            train_only={} if scored else {year: instances[year] for year in years},
            all_years=instances if scored else {year: instances[year] for year in years},
        )

    return (
        tuple(row(key, years, False) for key, years in split.train),
        tuple(row(key, years, True) for key, years in split.test),
        {},
    )


@pytest.mark.parametrize(
    "corruption",
    ("renamed", "duplicated", "altered_years", "changed_partition"),
)
def test_district_loader_rejects_noncanonical_partition_before_instances(
    monkeypatch, tmp_path, corruption: str
) -> None:
    index = _synthetic_district_index()
    canonical = tuple(leave_district_out(index, fold, 5) for fold in range(5))
    for split in canonical:
        (tmp_path / f"{split.name}.json").write_text("{}")

    def supplied(name: str, split_dir: Path) -> Split:
        del split_dir
        fold = int(name.removeprefix("district_out_f").split("of5")[0])
        split = canonical[fold]
        if corruption == "renamed" and fold == 2:
            return replace(split, name="district_out_f0of5")
        if corruption == "duplicated":
            return canonical[0]
        if corruption == "altered_years" and fold == 2:
            key, _ = split.train[0]
            return replace(split, train=((key, (2021,)), *split.train[1:]))
        if corruption == "changed_partition" and fold == 2:
            moved = split.test[0]
            return replace(split, train=(*split.train, moved), test=split.test[1:])
        return split

    monkeypatch.setattr(residual_train_module, "DISTRICT_SPLIT_ROOT", tmp_path)
    monkeypatch.setattr(residual_train_module, "build_series_index", lambda cfg: index)
    monkeypatch.setattr(residual_train_module, "load_split", supplied)
    monkeypatch.setattr(
        residual_train_module,
        "_normalized_fold_views",
        lambda *args: pytest.fail("invalid split reached outcome-instance loading"),
    )

    with pytest.raises(RuntimeError, match="district"):
        residual_train_module._load_district_development_folds()


def test_district_loader_authenticates_all_five_membership_counts(
    monkeypatch, tmp_path
) -> None:
    index = _synthetic_district_index()
    canonical = tuple(leave_district_out(index, fold, 5) for fold in range(5))
    for split in canonical:
        (tmp_path / f"{split.name}.json").write_text("{}")
    monkeypatch.setattr(residual_train_module, "DISTRICT_SPLIT_ROOT", tmp_path)
    monkeypatch.setattr(residual_train_module, "build_series_index", lambda cfg: index)
    monkeypatch.setattr(
        residual_train_module,
        "load_split",
        lambda name, split_dir: canonical[
            int(name.removeprefix("district_out_f").split("of5")[0])
        ],
    )
    monkeypatch.setattr(residual_train_module, "_normalized_fold_views", _split_rows_without_loading)

    folds = residual_train_module._load_district_development_folds()
    test_counts = {key: 0 for key in index}
    train_counts = {key: 0 for key in index}
    for fold in folds:
        for row in fold.test:
            test_counts[row.ref.key] += 1
        for row in fold.train:
            train_counts[row.ref.key] += 1

    assert set(test_counts.values()) == {1}
    assert set(train_counts.values()) == {4}


@pytest.mark.parametrize(
    "corruption",
    (
        "consistent_city_omission",
        "district_primary_only",
        "extra_row",
        "wrong_fold_city",
        "missing_anchor",
        "missing_mes",
        "missing_safe_static_series",
        "extra_safe_static_candidate",
    ),
)
def test_gate_rejects_any_incomplete_or_extra_authenticated_inventory(
    monkeypatch, corruption: str
) -> None:
    payload = _passing_development_payload()
    if corruption == "consistent_city_omission":
        payload["city_results"] = [
            row
            for row in payload["city_results"]
            if row.series != "Poland/Łódź/district-1"
        ]
    elif corruption == "district_primary_only":
        payload["district_results"] = [
            row for row in payload["district_results"] if row.seed == 42
        ]
    elif corruption == "extra_row":
        payload["district_results"].append(
            _development_result(
                fold="district_out_extra",
                seed=7,
                series="Poland/Warszawa/extra",
            )
        )
    elif corruption == "wrong_fold_city":
        payload["city_results"][0] = replace(
            payload["city_results"][0], fold="city_out_Poland_Warszawa"
        )
    elif corruption in {"missing_anchor", "missing_mes"}:
        row = _development_result_payload_for_test(payload["city_results"][0])
        row.pop(corruption.removeprefix("missing_"))
        payload["city_results"][0] = row
    elif corruption == "missing_safe_static_series":
        fold = "city_out_Poland_Gdynia"
        candidate = _development_static_identity_for_test()
        payload["safe_static_results"][fold][candidate]["series"].pop(
            "Poland/Gdynia/district-1"
        )
    else:
        fold = "city_out_Poland_Gdynia"
        payload["safe_static_results"][fold]["unexpected"] = {
            "family": "age",
            "series": {},
        }
    monkeypatch.setattr(
        residual_train_module,
        "cluster_city_bootstrap",
        lambda *args, **kwargs: pytest.fail("invalid inventory reached inference"),
    )

    with pytest.raises(RuntimeError, match="development evidence"):
        evaluate_development_gate(_trusted_development_gate_evidence(payload))


@pytest.mark.parametrize(
    "omission",
    ("scored_series_and_claimed_inventory", "safe_static_and_claimed_inventory"),
)
def test_gate_rejects_self_consistent_caller_controlled_inventory_omissions(
    monkeypatch: pytest.MonkeyPatch, omission: str
) -> None:
    payload = _passing_development_payload()
    if omission == "scored_series_and_claimed_inventory":
        fold = "city_out_Poland_Łódź"
        series = "Poland/Łódź/district-1"
        payload["city_results"] = [
            row for row in payload["city_results"] if row.series != series
        ]
        payload["expected_scored_series"][fold].remove(series)
        for candidate in payload["safe_static_results"][fold].values():
            candidate["series"].pop(series)
    else:
        fold = "city_out_Poland_Gdynia"
        identity = _development_static_identity_for_test()
        payload["expected_safe_static"][fold] = []
        payload["safe_static_results"][fold].pop(identity)
    monkeypatch.setattr(
        residual_train_module,
        "cluster_city_bootstrap",
        lambda *args, **kwargs: pytest.fail(
            "caller-controlled inventory reached gate inference"
        ),
    )

    with pytest.raises(RuntimeError, match="trusted|evidence"):
        evaluate_development_gate(_trusted_development_gate_evidence(payload))


def test_gate_rejects_bare_self_authenticating_mapping_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        residual_train_module,
        "_coerce_development_result",
        lambda row: pytest.fail("bare mapping reached result parsing"),
    )

    with pytest.raises(RuntimeError, match="trusted"):
        evaluate_development_gate(_passing_development_payload())


def _rehash_development_record(record: dict[str, object]) -> None:
    record.pop("payload_sha256", None)
    record["payload_sha256"] = residual_train_module._canonical_sha256(record)


def _rehash_development_pair(
    fit_record: dict[str, object], series_record: dict[str, object]
) -> None:
    _rehash_development_record(fit_record)
    series_record["fit_payload_sha256"] = fit_record["payload_sha256"]
    _rehash_development_record(series_record)


def _repair_development_fit_links_for_test(
    fit_record: dict[str, object], series_record: dict[str, object]
) -> None:
    static = fit_record["fit"]["static"]
    residual = fit_record["fit"]["residual"]
    residual["static_artifact_sha256"] = residual_train_module._canonical_sha256(
        static
    )
    unsigned_residual = dict(residual)
    unsigned_residual.pop("payload_sha256", None)
    residual["payload_sha256"] = residual_train_module._canonical_sha256(
        unsigned_residual
    )
    _rehash_development_pair(fit_record, series_record)


def _insert_static_candidate_for_test(
    static: dict[str, object],
    family: str,
    logits: tuple[float, ...],
    source: str,
    seed: int,
    *,
    before_first_family: str | None = None,
) -> dict[str, object]:
    template = copy.deepcopy(
        next(
            row
            for row in static["candidates"]
            if row["anchor"]["family"] == family
        )
    )
    anchor = _anchor(family, logits)
    template["anchor"] = residual_train_module._anchor_payload(anchor)
    template["anchor_sha256"] = residual_train_module.anchor_payload_sha256(anchor)
    template["source"] = source
    template["seed"] = seed
    if before_first_family is None:
        insert_at = len(static["candidates"])
    else:
        insert_at = next(
            index
            for index, row in enumerate(static["candidates"])
            if row["anchor"]["family"] == before_first_family
        )
    static["candidates"].insert(insert_at, template)
    return template


def _set_development_exclusion_zero_for_test(
    fit_record: dict[str, object], series_record: dict[str, object]
) -> None:
    static = fit_record["fit"]["static"]
    for metrics in static["mes_metrics"].values():
        metrics["city_exclusion"] = 0.0
    for candidate in (*static["candidates"], static["selected"]):
        for metrics in candidate["metrics"].values():
            metrics["city_exclusion"] = 0.0

    residual = fit_record["fit"]["residual"]
    for metrics in residual["anchor"]["metrics"].values():
        metrics["city_exclusion"] = 0.0
    candidate_payloads = [residual["primary"]]
    for seed in residual["seeds"]:
        candidate_payloads.append(seed["selected"])
        candidate_payloads.extend(seed["archive"])
    for candidate in candidate_payloads:
        for row in candidate["metrics"]["series"]:
            row["exclusion"] = 0.0
        for metrics in candidate["metrics"]["cities"].values():
            metrics["city_exclusion"] = 0.0
    _repair_development_fit_links_for_test(fit_record, series_record)


def _sync_safe_static_context_for_test(
    fit_record: dict[str, object], series_record: dict[str, object]
) -> None:
    static = fit_record["fit"]["static"]
    safe_rows = [row for row in static["candidates"] if row["safe"]]
    inventory = [
        {"identity": row["anchor_sha256"], "family": row["anchor"]["family"]}
        for row in safe_rows
    ]
    fit_record["expected_safe_static"] = copy.deepcopy(inventory)
    series_record["expected_safe_static"] = copy.deepcopy(inventory)
    retained = series_record["result"]["safe_static"]
    series_template = copy.deepcopy(next(iter(retained.values()))["series"])
    series_record["result"]["safe_static"] = {
        row["anchor_sha256"]: {
            "family": row["anchor"]["family"],
            "series": copy.deepcopy(series_template),
        }
        for row in safe_rows
    }


def _retarget_residual_anchor_csd_for_test(
    fit_record: dict[str, object],
    series_record: dict[str, object],
    selected: dict[str, object],
    *,
    old_csd: float = 1.0,
) -> None:
    static = fit_record["fit"]["static"]
    static["selected"] = copy.deepcopy(selected)
    residual = fit_record["fit"]["residual"]
    residual["anchor"] = copy.deepcopy(selected)
    new_csd = next(iter(selected["metrics"].values()))["city_csd"]
    shift = old_csd - new_csd
    candidate_payloads = [residual["primary"]]
    for seed in residual["seeds"]:
        candidate_payloads.append(seed["selected"])
        candidate_payloads.extend(seed["archive"])
    for candidate in candidate_payloads:
        for row in candidate["metrics"]["series"]:
            row["csd"] -= shift
        for metrics in candidate["metrics"]["cities"].values():
            metrics["city_csd"] -= shift
    _sync_safe_static_context_for_test(fit_record, series_record)
    _repair_development_fit_links_for_test(fit_record, series_record)


def _set_static_candidate_evaluation_for_test(
    candidate: dict[str, object],
    *,
    csd: float,
    welfare: float,
    safe: bool,
) -> None:
    for metrics in candidate["metrics"].values():
        metrics["city_csd"] = csd
        metrics["city_welfare"] = welfare
    candidate["safe"] = safe
    candidate["objective"] = csd if safe else None


def _development_fold_with_numeric_zero_provenance_for_test() -> DevelopmentFold:
    base = _canonical_development_gate_folds()[0]
    return DevelopmentFold(
        base.name,
        base.kind,
        tuple(
            residual_train_module._materialize_development_series(row)
            for row in base.train
        ),
        tuple(
            residual_train_module._materialize_development_series(row)
            for row in base.test
        ),
        base.source_manifest_sha256,
        {**dict(base.provenance), "numeric_zero": 0.0},
    )


def _authenticate_complete_development_pair_for_test(
    fold: DevelopmentFold,
    fit_record: dict[str, object],
    series_record: dict[str, object],
) -> None:
    residual_train_module._authenticate_development_artifact_pair(
        fold,
        fit_record,
        series_record,
        anchor_search_config(),
    )


def test_trusted_gate_rejects_reviewer_mes_only_wrong_training_city_archive() -> None:
    folds, fit_records, series_records = _development_gate_inputs_for_test()
    fit_records = list(copy.deepcopy(fit_records))
    series_records = list(copy.deepcopy(series_records))
    fit_records[0]["fit"]["static"] = _development_static_fit_for_test()
    _rehash_development_pair(fit_records[0], series_records[0])

    with pytest.raises(RuntimeError, match="city|family|archive|static|training"):
        _authenticate_complete_development_pair_for_test(
            folds[0], fit_records[0], series_records[0]
        )


def test_trusted_gate_rejects_reviewer_impossible_mes_provenance() -> None:
    fold = _canonical_development_gate_folds()[0]
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )
    static = fit_record["fit"]["static"]
    static["candidates"][0]["source"] = "senior-grid-alpha-0"
    static["candidates"][0]["seed"] = 999
    static["selected"] = copy.deepcopy(static["candidates"][0])
    residual = fit_record["fit"]["residual"]
    residual["static_artifact_sha256"] = residual_train_module._canonical_sha256(
        static
    )
    residual["anchor"] = copy.deepcopy(static["selected"])
    unsigned_residual = dict(residual)
    unsigned_residual.pop("payload_sha256")
    residual["payload_sha256"] = residual_train_module._canonical_sha256(
        unsigned_residual
    )
    _rehash_development_pair(fit_record, series_record)

    with pytest.raises(RuntimeError, match="MES|source|seed|provenance|static"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


@pytest.mark.parametrize(
    "corruption",
    (
        "missing_mes",
        "missing_senior",
        "missing_age",
        "missing_age_sex",
        "missing_senior_point",
        "extra_senior_point",
        "wrong_age_seed",
        "wrong_age_sex_seed",
        "out_of_range_age_generation",
        "out_of_range_age_sex_generation",
    ),
)
def test_trusted_gate_authenticates_complete_static_search_provenance_matrix(
    corruption: str,
) -> None:
    fold = _canonical_development_gate_folds()[0]
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )
    static = fit_record["fit"]["static"]
    candidates = static["candidates"]
    if corruption.startswith("missing_") and corruption in {
        "missing_mes",
        "missing_senior",
        "missing_age",
        "missing_age_sex",
    }:
        family = corruption.removeprefix("missing_")
        static["candidates"] = [
            row for row in candidates if row["anchor"]["family"] != family
        ]
        static["families_present"] = [
            name for name in static["families_present"] if name != family
        ]
        if family == "mes":
            static["selected"] = copy.deepcopy(static["candidates"][0])
            static["mes_fallback"] = {"available": False, "safe": False}
    elif corruption == "missing_senior_point":
        static["candidates"] = [
            row
            for row in candidates
            if row["source"] != "senior-grid-alpha-999"
        ]
    elif corruption == "extra_senior_point":
        extra = copy.deepcopy(
            next(row for row in candidates if row["anchor"]["family"] == "senior")
        )
        extra_anchor = _anchor("senior", (math.log1p(1000.0),))
        extra["anchor"] = residual_train_module._anchor_payload(extra_anchor)
        extra["anchor_sha256"] = residual_train_module.anchor_payload_sha256(
            extra_anchor
        )
        extra["source"] = "senior-grid-alpha-1000"
        static["candidates"].append(extra)
    else:
        family = "age_sex" if "age_sex" in corruption else "age"
        row = next(
            candidate
            for candidate in candidates
            if candidate["anchor"]["family"] == family
        )
        if "wrong" in corruption:
            row["seed"] = 999
            row["source"] = f"{family}-seed-999-initializer"
        else:
            row["source"] = (
                f"{family}-seed-{row['seed']}-generation-"
                f"{anchor_search_config().generations}"
            )
    residual = fit_record["fit"]["residual"]
    residual["static_artifact_sha256"] = residual_train_module._canonical_sha256(
        static
    )
    unsigned_residual = dict(residual)
    unsigned_residual.pop("payload_sha256")
    residual["payload_sha256"] = residual_train_module._canonical_sha256(
        unsigned_residual
    )
    _rehash_development_pair(fit_record, series_record)

    with pytest.raises(RuntimeError, match="family|grid|seed|generation|MES|static"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


def test_trusted_gate_rejects_mes_equivalent_alpha_zero_with_contradictory_score() -> None:
    fold = _canonical_development_gate_folds()[0]
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )
    static = fit_record["fit"]["static"]
    mes = static["candidates"][0]
    alpha_zero = next(
        row for row in static["candidates"]
        if row["source"] == "senior-grid-alpha-0"
    )
    assert mes["anchor"]["free_logits"] == []
    assert alpha_zero["anchor"]["free_logits"] == [0.0]
    assert mes["safe"] is True
    unsafe = next(
        row for row in static["candidates"]
        if row["anchor"]["family"] == "senior"
        and row["source"] != "senior-grid-alpha-0"
    )
    alpha_zero["metrics"] = copy.deepcopy(unsafe["metrics"])
    alpha_zero["safe"] = False
    alpha_zero["objective"] = None
    _repair_development_fit_links_for_test(fit_record, series_record)

    with pytest.raises(RuntimeError, match="MES|alpha|equivalent|static"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


@pytest.mark.parametrize(
    "corruption",
    (
        "age_later_seed_initializer",
        "age_sex_later_seed_initializer",
        "wrong_age_initializer_anchor",
        "wrong_age_sex_initializer_anchor",
        "duplicate_age_source",
        "reversed_age_source_order",
        "age_initializer_not_first",
    ),
)
def test_trusted_gate_authenticates_nested_static_initializer_lineage(
    corruption: str,
) -> None:
    fold = _canonical_development_gate_folds()[0]
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )
    static = fit_record["fit"]["static"]
    age = next(
        row for row in static["candidates"]
        if row["anchor"]["family"] == "age"
    )
    age_sex = next(
        row for row in static["candidates"]
        if row["anchor"]["family"] == "age_sex"
    )
    if corruption == "age_later_seed_initializer":
        age["seed"] = 2
        age["source"] = "age-seed-2-initializer"
    elif corruption == "age_sex_later_seed_initializer":
        age_sex["seed"] = 2
        age_sex["source"] = "age_sex-seed-2-initializer"
    elif corruption == "wrong_age_initializer_anchor":
        anchor = _anchor("age", (0.0, 0.0, 0.25))
        age["anchor"] = residual_train_module._anchor_payload(anchor)
        age["anchor_sha256"] = residual_train_module.anchor_payload_sha256(anchor)
    elif corruption == "wrong_age_sex_initializer_anchor":
        anchor = _anchor("age_sex", (0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25))
        age_sex["anchor"] = residual_train_module._anchor_payload(anchor)
        age_sex["anchor_sha256"] = residual_train_module.anchor_payload_sha256(anchor)
    elif corruption == "duplicate_age_source":
        _insert_static_candidate_for_test(
            static,
            "age",
            (0.1, 0.0, 0.0),
            age["source"],
            age["seed"],
            before_first_family="age_sex",
        )
    elif corruption == "reversed_age_source_order":
        _insert_static_candidate_for_test(
            static,
            "age",
            (0.1, 0.0, 0.0),
            "age-seed-1-generation-1",
            1,
            before_first_family="age_sex",
        )
        _insert_static_candidate_for_test(
            static,
            "age",
            (0.2, 0.0, 0.0),
            "age-seed-1-generation-0",
            1,
            before_first_family="age_sex",
        )
    else:
        first_age = next(
            index for index, row in enumerate(static["candidates"])
            if row["anchor"]["family"] == "age"
        )
        row = _insert_static_candidate_for_test(
            static,
            "age",
            (0.1, 0.0, 0.0),
            "age-seed-1-generation-0",
            1,
            before_first_family="age_sex",
        )
        static["candidates"].remove(row)
        static["candidates"].insert(first_age, row)
    _repair_development_fit_links_for_test(fit_record, series_record)

    with pytest.raises(RuntimeError, match="initializer|lineage|source|order|static"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


def test_trusted_gate_rejects_seed_bearing_off_grid_senior_selected_root() -> None:
    fold = _canonical_development_gate_folds()[0]
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )
    static = fit_record["fit"]["static"]
    senior_logit = math.log1p(1000.0)
    forged = _insert_static_candidate_for_test(
        static,
        "senior",
        (senior_logit,),
        "senior-seed-1-forged",
        1,
        before_first_family="age",
    )
    _set_static_candidate_evaluation_for_test(
        forged, csd=0.5, welfare=10.0, safe=True
    )
    age = next(
        row for row in static["candidates"]
        if row["anchor"]["family"] == "age"
    )
    age_anchor = _anchor("age", (0.0, 0.0, senior_logit))
    age["anchor"] = residual_train_module._anchor_payload(age_anchor)
    age["anchor_sha256"] = residual_train_module.anchor_payload_sha256(age_anchor)
    _set_static_candidate_evaluation_for_test(
        age, csd=0.5, welfare=10.0, safe=True
    )
    age_sex = next(
        row for row in static["candidates"]
        if row["anchor"]["family"] == "age_sex"
    )
    age_sex_anchor = _anchor(
        "age_sex",
        (0.0, 0.0, 0.0, 0.0, 0.0, senior_logit, senior_logit),
    )
    age_sex["anchor"] = residual_train_module._anchor_payload(age_sex_anchor)
    age_sex["anchor_sha256"] = residual_train_module.anchor_payload_sha256(
        age_sex_anchor
    )
    _set_static_candidate_evaluation_for_test(
        age_sex, csd=0.5, welfare=10.0, safe=True
    )
    _retarget_residual_anchor_csd_for_test(
        fit_record, series_record, forged
    )

    with pytest.raises(RuntimeError, match="senior|grid|archive|provenance"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


@pytest.mark.parametrize(
    "corruption",
    (
        "missing",
        "reordered",
        "duplicated",
        "wrong_source",
        "wrong_seed",
        "off_grid_anchor",
    ),
)
def test_trusted_gate_authenticates_every_senior_grid_row(
    corruption: str,
) -> None:
    fold = _canonical_development_gate_folds()[0]
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )
    static = fit_record["fit"]["static"]
    senior_indices = [
        index for index, row in enumerate(static["candidates"])
        if row["anchor"]["family"] == "senior"
    ]
    target_index = senior_indices[1]
    if corruption == "missing":
        static["candidates"].pop(target_index)
    elif corruption == "reordered":
        other_index = senior_indices[2]
        static["candidates"][target_index], static["candidates"][other_index] = (
            static["candidates"][other_index],
            static["candidates"][target_index],
        )
    elif corruption == "duplicated":
        duplicate = copy.deepcopy(static["candidates"][target_index])
        first_age = next(
            index for index, row in enumerate(static["candidates"])
            if row["anchor"]["family"] == "age"
        )
        static["candidates"].insert(first_age, duplicate)
    elif corruption == "wrong_source":
        static["candidates"][target_index]["source"] = "senior-grid-alpha-forged"
    elif corruption == "wrong_seed":
        static["candidates"][target_index]["seed"] = 1
    else:
        anchor = _anchor("senior", (math.log1p(1000.0),))
        static["candidates"][target_index]["anchor"] = (
            residual_train_module._anchor_payload(anchor)
        )
        static["candidates"][target_index]["anchor_sha256"] = (
            residual_train_module.anchor_payload_sha256(anchor)
        )
    _repair_development_fit_links_for_test(fit_record, series_record)

    with pytest.raises(RuntimeError, match="senior|grid|candidate|archive"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


@pytest.mark.parametrize(
    ("transition", "child_state"),
    (
        ("senior_to_age", "safe_contradiction"),
        ("senior_to_age", "unsafe_contradiction"),
        ("age_to_age_sex", "safe_contradiction"),
        ("age_to_age_sex", "unsafe_contradiction"),
    ),
)
def test_trusted_gate_authenticates_nested_initializer_evaluations(
    transition: str, child_state: str
) -> None:
    fold = _canonical_development_gate_folds()[0]
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )
    static = fit_record["fit"]["static"]
    age = next(
        row for row in static["candidates"]
        if row["anchor"]["family"] == "age"
    )
    age_sex = next(
        row for row in static["candidates"]
        if row["anchor"]["family"] == "age_sex"
    )
    child = age if transition == "senior_to_age" else age_sex
    if child_state == "safe_contradiction":
        _set_static_candidate_evaluation_for_test(
            child, csd=0.5, welfare=10.0, safe=True
        )
        _retarget_residual_anchor_csd_for_test(
            fit_record, series_record, child
        )
    elif transition == "age_to_age_sex":
        _set_static_candidate_evaluation_for_test(
            age_sex, csd=1.0, welfare=0.0, safe=False
        )
        _sync_safe_static_context_for_test(fit_record, series_record)
        _repair_development_fit_links_for_test(fit_record, series_record)
    else:
        _set_static_candidate_evaluation_for_test(
            age, csd=1.0, welfare=0.0, safe=False
        )
        later_age = _insert_static_candidate_for_test(
            static,
            "age",
            (0.1, 0.0, 0.0),
            "age-seed-1-generation-0",
            1,
            before_first_family="age_sex",
        )
        _set_static_candidate_evaluation_for_test(
            later_age, csd=1.0, welfare=10.0, safe=True
        )
        age_sex_anchor = _anchor(
            "age_sex", (0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0)
        )
        age_sex["anchor"] = residual_train_module._anchor_payload(age_sex_anchor)
        age_sex["anchor_sha256"] = residual_train_module.anchor_payload_sha256(
            age_sex_anchor
        )
        _sync_safe_static_context_for_test(fit_record, series_record)
        _repair_development_fit_links_for_test(fit_record, series_record)

    with pytest.raises(RuntimeError, match="initializer|equivalent|evaluation|static"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


@pytest.mark.parametrize(
    "corruption",
    (
        "reviewer_forgery",
        "inner_digest",
        "static_link",
        "config",
        "anchor",
        "scaler_hash",
        "scaler_shape",
        "scaler_count",
        "seed_order",
        "seed_count",
        "archive_count",
        "candidate_vectors",
        "candidate_metrics_inventory",
        "candidate_metrics_aggregate",
        "selected_membership",
        "selected_selector",
        "primary_seed",
        "primary_payload",
        "synthetic_actuated",
    ),
)
def test_trusted_gate_authenticates_serialized_residual_fit_mutation_matrix(
    corruption: str,
) -> None:
    fold = _canonical_development_gate_folds()[0]
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )
    residual = fit_record["fit"]["residual"]
    if corruption == "reviewer_forgery":
        residual["training_only"] = False
        residual["static_artifact_sha256"] = "f" * 64
        residual["config"]["tau"] = 999.0
        residual["anchor"]["source"] = "forged"
        residual["scaler"]["sha256"] = "e" * 64
        residual["seeds"] = [
            {"seed": seed, "selected": {}, "archive": [], "archive_count": 0,
             "optimizer_history": []}
            for seed in (1, 2, 42)
        ]
        residual["primary"] = {}
    elif corruption == "inner_digest":
        residual["payload_sha256"] = "0" * 64
    elif corruption == "static_link":
        residual["static_artifact_sha256"] = "f" * 64
    elif corruption == "config":
        residual["config"]["replacement_delta"] = 0.0
    elif corruption == "anchor":
        residual["anchor"]["source"] = "forged-anchor"
    elif corruption == "scaler_hash":
        residual["scaler"]["sha256"] = "f" * 64
    elif corruption == "scaler_shape":
        residual["scaler"]["mean"].pop()
    elif corruption == "scaler_count":
        residual["scaler"]["row_count"] = True
    elif corruption == "seed_order":
        residual["seeds"][0], residual["seeds"][1] = (
            residual["seeds"][1], residual["seeds"][0]
        )
    elif corruption == "seed_count":
        residual["seeds"].pop()
    elif corruption == "archive_count":
        residual["seeds"][0]["archive_count"] += 1
    elif corruption == "candidate_vectors":
        residual["seeds"][0]["archive"][0]["weights"][0] = 0.1
    elif corruption == "candidate_metrics_inventory":
        residual["seeds"][0]["archive"][0]["metrics"]["series"][0][
            "series"
        ] = "Poland/Forged/series"
    elif corruption == "candidate_metrics_aggregate":
        city = next(
            iter(residual["seeds"][0]["archive"][0]["metrics"]["cities"])
        )
        residual["seeds"][0]["archive"][0]["metrics"]["cities"][city][
            "city_csd"
        ] = 0.5
    elif corruption == "selected_membership":
        residual["seeds"][0]["selected"]["source"] = "not-in-archive"
    elif corruption == "selected_selector":
        residual["seeds"][2]["selected"] = copy.deepcopy(
            residual["seeds"][2]["archive"][0]
        )
    elif corruption == "primary_seed":
        residual["primary_seed"] = 1
    elif corruption == "primary_payload":
        residual["primary"]["source"] = "forged-primary"
    else:
        residual["synthetic_actuated"] = False
    if corruption not in {"reviewer_forgery", "inner_digest"}:
        unsigned_residual = dict(residual)
        unsigned_residual.pop("payload_sha256")
        residual["payload_sha256"] = residual_train_module._canonical_sha256(
            unsigned_residual
        )
    _rehash_development_pair(fit_record, series_record)

    with pytest.raises(
        RuntimeError,
        match="residual|digest|static|config|anchor|scaler|seed|archive|candidate|primary|actuated|training",
    ):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


def test_lightweight_development_fit_requires_explicit_unit_only_carveout() -> None:
    lightweight = {"primary_seed": 42}
    with pytest.raises(RuntimeError, match="fit payload schema"):
        residual_train_module._validate_development_fit_payload(
            lightweight, 42, (), anchor_search_config()
        )

    residual_train_module._validate_development_fit_payload(
        lightweight,
        42,
        (),
        anchor_search_config(),
        allow_lightweight=True,
    )


def test_reconstructed_serialized_fold_fit_is_restartable_and_finite_json() -> None:
    fold = _canonical_development_gate_folds()[0]
    _, payload = _complete_development_fold_fit_for_test(fold)
    training_series = tuple(row.ref.key for row in fold.train)
    training_cities = ("Poland/Training",)
    anchor_fit, safe = residual_train_module._authenticated_static_fit(
        payload["static"], anchor_search_config(), training_cities
    )
    assert safe == _development_safe_static_inventory_for_test()
    reconstructed = residual_train_module._authenticated_residual_fit(
        payload["residual"],
        payload["static"],
        anchor_fit,
        residual_search_config(),
        training_series,
        training_cities,
    )
    static_sha = residual_train_module._canonical_sha256(payload["static"])

    for snapshot in (
        reconstructed,
        copy.copy(reconstructed),
        copy.deepcopy(reconstructed),
        pickle.loads(pickle.dumps(reconstructed)),
    ):
        rebuilt = residual_train_module._residual_fit_payload(snapshot, static_sha)
        assert _canonical_json_bytes(rebuilt) == _canonical_json_bytes(
            payload["residual"]
        )
        for seed_fit in snapshot.seeds:
            zero = next(
                candidate
                for candidate in seed_fit.archive
                if candidate.amplitude == 0.0
            )
            assert all(
                value == 0.0 and math.copysign(1.0, value) == 1.0
                for value in (*zero.weights, *zero.endpoint)
            )
        assert asdict(snapshot)["primary_seed"] == 42
        json.dumps(
            {
                "static": residual_train_module._fit_payload(snapshot.anchor_fit),
                "residual": residual_train_module._residual_fit_payload(
                    snapshot, static_sha
                ),
            },
            sort_keys=True,
            allow_nan=False,
        )


@pytest.mark.parametrize(
    "location",
    ("archive_zero", "selected_zero", "primary_zero_leaf"),
)
def test_development_fit_rejects_signed_zero_in_canonical_residual_vectors(
    location: str,
) -> None:
    fold = _canonical_development_gate_folds()[0]
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )
    residual = fit_record["fit"]["residual"]
    if location == "archive_zero":
        target = residual["seeds"][0]["archive"][0]
        index = 0
    elif location == "selected_zero":
        target = residual["seeds"][0]["selected"]
        index = 0
    else:
        target = residual["primary"]
        index = 1
    target["weights"][index] = -0.0
    target["endpoint"][index] = -0.0
    assert math.copysign(1.0, target["weights"][index]) == -1.0
    positive = copy.deepcopy(target)
    positive["weights"][index] = 0.0
    positive["endpoint"][index] = 0.0
    assert residual_train_module._canonical_sha256(target) != (
        residual_train_module._canonical_sha256(positive)
    )
    _repair_development_fit_links_for_test(fit_record, series_record)

    with pytest.raises(RuntimeError, match="canonical|payload|residual|zero"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


def test_development_fit_rejects_signed_zero_normalized_residual_metric_leaf() -> None:
    fold = _canonical_development_gate_folds()[0]
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )
    _set_development_exclusion_zero_for_test(fit_record, series_record)
    target = fit_record["fit"]["residual"]["seeds"][0]["archive"][0]
    city = next(iter(target["metrics"]["cities"]))
    target["metrics"]["cities"][city]["city_exclusion"] = -0.0
    assert math.copysign(
        1.0, target["metrics"]["cities"][city]["city_exclusion"]
    ) == -1.0
    _repair_development_fit_links_for_test(fit_record, series_record)

    with pytest.raises(RuntimeError, match="canonical|metric|payload|residual"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


def test_development_fit_preserves_signed_zero_scaler_mean_exactly() -> None:
    fold = _canonical_development_gate_folds()[0]
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )
    scaler_payload = fit_record["fit"]["residual"]["scaler"]
    scaler_payload["mean"][0] = -0.0
    scaler = residual_train_module.FeatureScaler(
        scaler_payload["feature_names"],
        scaler_payload["mean"],
        scaler_payload["scale"],
        scaler_payload["clip"],
        scaler_payload["row_count"],
        scaler_payload["instance_count"],
    )
    scaler_payload["sha256"] = scaler.sha256
    _repair_development_fit_links_for_test(fit_record, series_record)

    training_cities = ("Poland/Training",)
    anchor_fit, _ = residual_train_module._authenticated_static_fit(
        fit_record["fit"]["static"], anchor_search_config(), training_cities
    )
    reconstructed = residual_train_module._authenticated_residual_fit(
        fit_record["fit"]["residual"],
        fit_record["fit"]["static"],
        anchor_fit,
        residual_search_config(),
        tuple(row.ref.key for row in fold.train),
        training_cities,
    )
    rebuilt = residual_train_module._residual_fit_payload(
        reconstructed,
        residual_train_module._canonical_sha256(fit_record["fit"]["static"]),
    )
    assert math.copysign(1.0, rebuilt["scaler"]["mean"][0]) == -1.0
    assert _canonical_json_bytes(rebuilt) == _canonical_json_bytes(
        fit_record["fit"]["residual"]
    )


def test_development_fit_rejects_nonpositive_scaler_scale_signed_zero() -> None:
    fold = _canonical_development_gate_folds()[0]
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )
    scaler = fit_record["fit"]["residual"]["scaler"]
    scaler["scale"][0] = -0.0
    _repair_development_fit_links_for_test(fit_record, series_record)

    with pytest.raises(RuntimeError, match="scaler|residual"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


def test_trusted_gate_rejects_self_rehashed_nonproduction_static_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folds, fit_records, series_records = _development_gate_inputs_for_test()
    fit_records = list(copy.deepcopy(fit_records))
    series_records = list(copy.deepcopy(series_records))
    config = fit_records[0]["fit"]["static"]["config"]
    config["generations"] = 1
    config["seeds"] = [999]
    _rehash_development_pair(fit_records[0], series_records[0])
    monkeypatch.setattr(
        residual_train_module,
        "_validate_development_fold_result",
        lambda *args, **kwargs: pytest.fail(
            "altered production static config reached result conversion"
        ),
    )

    with pytest.raises(RuntimeError, match="config|static|fit"):
        residual_train_module.DevelopmentGateEvidence(
            folds, fit_records, series_records
        )


def test_trusted_gate_recomputes_unique_static_archive_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folds, fit_records, series_records = _development_gate_inputs_for_test()
    fit_records = list(copy.deepcopy(fit_records))
    series_records = list(copy.deepcopy(series_records))
    senior = _candidate("senior", (0.25,), 0.5)
    senior_payload = residual_train_module._candidate_payload(senior)
    senior_identity = _anchor_payload_sha(senior.anchor)
    static = fit_records[0]["fit"]["static"]
    assert static["selected"]["anchor"]["family"] == "mes"
    assert static["selected"]["objective"] == 1.0
    static["candidates"].append(senior_payload)
    static["families_present"] = ["mes", "senior"]
    expected = {"identity": senior_identity, "family": "senior"}
    fit_records[0]["expected_safe_static"].append(copy.deepcopy(expected))
    series_records[0]["expected_safe_static"].append(copy.deepcopy(expected))
    retained_mes = next(iter(series_records[0]["result"]["safe_static"].values()))
    series_records[0]["result"]["safe_static"][senior_identity] = {
        "family": "senior",
        "series": copy.deepcopy(retained_mes["series"]),
    }
    _rehash_development_pair(fit_records[0], series_records[0])
    monkeypatch.setattr(
        residual_train_module,
        "_validate_development_fold_result",
        lambda *args, **kwargs: pytest.fail(
            "nonoptimal exact archive-member selection reached result conversion"
        ),
    )

    with pytest.raises(RuntimeError, match="selected|winner|static"):
        residual_train_module.DevelopmentGateEvidence(
            folds, fit_records, series_records
        )


@pytest.mark.parametrize("corruption", ("safe", "objective"))
def test_trusted_gate_recomputes_static_candidate_semantics(
    monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    folds, fit_records, series_records = _development_gate_inputs_for_test()
    fit_records = list(copy.deepcopy(fit_records))
    series_records = list(copy.deepcopy(series_records))
    static = fit_records[0]["fit"]["static"]
    candidate = static["candidates"][0]
    selected = static["selected"]
    if corruption == "safe":
        candidate["safe"] = False
        candidate["objective"] = None
        selected["safe"] = False
        selected["objective"] = None
        static["mes_fallback"]["safe"] = False
        fit_records[0]["expected_safe_static"] = []
        series_records[0]["expected_safe_static"] = []
        series_records[0]["result"]["safe_static"] = {}
    else:
        candidate["objective"] = 0.5
        selected["objective"] = 0.5
    _rehash_development_pair(fit_records[0], series_records[0])
    monkeypatch.setattr(
        residual_train_module,
        "_validate_development_fold_result",
        lambda *args, **kwargs: pytest.fail(
            f"altered static {corruption} reached result conversion"
        ),
    )

    with pytest.raises(RuntimeError, match="safe|safety|objective|static"):
        residual_train_module.DevelopmentGateEvidence(
            folds, fit_records, series_records
        )


def test_development_fit_authentication_requires_explicit_smoke_config() -> None:
    fold = _canonical_development_gate_folds()[0]
    smoke_config = smoke_anchor_search_config()
    _, fit = _complete_development_fold_fit_for_test(
        fold,
        anchor_config=smoke_config,
        residual_config=smoke_residual_search_config(),
    )
    expected = _development_safe_static_inventory_for_test()

    residual_train_module._validate_development_fit_payload(
        fit,
        42,
        expected,
        smoke_config,
        smoke_residual_search_config(),
        tuple(row.ref.key for row in fold.train),
        ("Poland/Training",),
    )
    with pytest.raises(RuntimeError, match="config|static"):
        residual_train_module._validate_development_fit_payload(
            fit,
            42,
            expected,
            anchor_search_config(),
            smoke_residual_search_config(),
            tuple(row.ref.key for row in fold.train),
            ("Poland/Training",),
        )


def test_trusted_gate_rejects_self_rehashed_retained_anchor_identity_laundering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    folds, fit_records, series_records = _development_gate_inputs_for_test()
    fit_records = list(copy.deepcopy(fit_records))
    series_records = list(copy.deepcopy(series_records))
    old_identity = _development_static_identity_for_test()
    laundered_identity = "f" * 64
    candidate = fit_records[0]["fit"]["static"]["candidates"][0]
    assert candidate["anchor_sha256"] == old_identity
    candidate["anchor_sha256"] = laundered_identity
    fit_records[0]["fit"]["static"]["selected"][
        "anchor_sha256"
    ] = laundered_identity
    fit_records[0]["expected_safe_static"][0]["identity"] = laundered_identity
    series_records[0]["expected_safe_static"][0]["identity"] = laundered_identity
    safe_static = series_records[0]["result"]["safe_static"]
    safe_static[laundered_identity] = safe_static.pop(old_identity)
    _rehash_development_pair(fit_records[0], series_records[0])
    monkeypatch.setattr(
        residual_train_module,
        "_validate_development_fold_result",
        lambda *args, **kwargs: pytest.fail(
            "laundered retained anchor reached result conversion"
        ),
    )

    with pytest.raises(RuntimeError, match="anchor|static|candidate|fit"):
        residual_train_module.DevelopmentGateEvidence(
            folds, fit_records, series_records
        )


@pytest.mark.parametrize(
    "corruption",
    ("duplicate_candidate", "selected_absent", "selected_inconsistent"),
)
def test_development_fit_authentication_rejects_duplicate_or_unretained_selection(
    corruption: str,
) -> None:
    fold = _canonical_development_gate_folds()[0]
    _, fit = _complete_development_fold_fit_for_test(fold)
    expected = _development_safe_static_inventory_for_test()
    if corruption == "duplicate_candidate":
        fit["static"]["candidates"].append(
            copy.deepcopy(fit["static"]["candidates"][0])
        )
        expected.append(copy.deepcopy(expected[0]))
    elif corruption == "selected_absent":
        fit["static"]["selected"] = residual_train_module._candidate_payload(
            _candidate("senior", (0.25,), 1.0)
        )
    else:
        fit["static"]["selected"]["source"] = "not-the-retained-row"

    with pytest.raises(RuntimeError, match="candidate|selected|static|inventory"):
        residual_train_module._validate_development_fit_payload(
            fit,
            42,
            expected,
            anchor_search_config(),
            residual_search_config(),
            tuple(row.ref.key for row in fold.train),
            ("Poland/Training",),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("family", "unsupported"),
        ("free_logits", [0.0]),
        ("reference_cell", "age<25"),
        ("alpha", 0.0),
    ),
)
def test_development_fit_authentication_rejects_malformed_retained_anchors(
    field: str, value: object
) -> None:
    fold = _canonical_development_gate_folds()[0]
    _, fit = _complete_development_fold_fit_for_test(fold)
    candidate = fit["static"]["candidates"][0]
    candidate["anchor"][field] = value
    expected = [
        {
            "identity": candidate["anchor_sha256"],
            "family": candidate["anchor"]["family"],
        }
    ]

    with pytest.raises(RuntimeError, match="anchor|candidate|static"):
        residual_train_module._validate_development_fit_payload(
            fit,
            42,
            expected,
            anchor_search_config(),
            residual_search_config(),
            tuple(row.ref.key for row in fold.train),
            ("Poland/Training",),
        )


def test_development_fit_authentication_requires_exact_integer_static_schema() -> None:
    fold = _canonical_development_gate_folds()[0]
    _, fit = _complete_development_fold_fit_for_test(fold)
    fit["static"]["schema_version"] = True
    expected = _development_safe_static_inventory_for_test()

    with pytest.raises(RuntimeError, match="static fit schema"):
        residual_train_module._validate_development_fit_payload(
            fit,
            42,
            expected,
            anchor_search_config(),
            residual_search_config(),
            tuple(row.ref.key for row in fold.train),
            ("Poland/Training",),
        )


def test_development_artifact_rejects_boolean_nested_schema_with_stale_subhash() -> None:
    fold = _canonical_development_gate_folds()[0]
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )
    retained_sha = fit_record["fold_input_sha256"]
    for record in (fit_record, series_record):
        record["fold_input"]["schema_version"] = True
        assert record["fold_input_sha256"] == retained_sha
        assert record["fold_input_sha256"] != residual_train_module._canonical_sha256(
            record["fold_input"]
        )
    _rehash_development_pair(fit_record, series_record)

    with pytest.raises(RuntimeError, match="fold|schema|hash|artifact"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


def test_development_artifact_rejects_float_top_level_schema_alias() -> None:
    fold = _canonical_development_gate_folds()[0]
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )
    fit_record["schema_version"] = 2.0
    series_record["schema_version"] = 2.0
    _rehash_development_pair(fit_record, series_record)

    with pytest.raises(RuntimeError, match="schema|artifact"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


@pytest.mark.parametrize(
    "corruption",
    (
        "fit_top_bool",
        "series_top_bool",
        "fit_top_float",
        "series_top_float",
        "fit_nested_bool",
        "series_nested_bool",
        "fit_nested_float",
        "series_nested_float",
        "fit_outer_signed_zero",
        "series_outer_signed_zero",
        "fit_nested_signed_zero",
        "series_nested_signed_zero",
    ),
)
def test_development_artifact_authenticates_canonical_outer_numeric_context(
    corruption: str,
) -> None:
    if "signed_zero" in corruption:
        fold = _development_fold_with_numeric_zero_provenance_for_test()
    else:
        fold = _canonical_development_gate_folds()[0]
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )
    record = fit_record if corruption.startswith("fit_") else series_record
    if corruption.endswith("top_bool"):
        record["schema_version"] = True
    elif corruption.endswith("top_float"):
        record["schema_version"] = 2.0
    elif corruption.endswith("nested_bool"):
        record["fold_input"]["schema_version"] = True
    elif corruption.endswith("nested_float"):
        record["fold_input"]["schema_version"] = 1.0
    elif "outer_signed_zero" in corruption:
        record["provenance"]["numeric_zero"] = -0.0
        assert math.copysign(1.0, record["provenance"]["numeric_zero"]) == -1.0
    else:
        record["fold_input"]["provenance"]["numeric_zero"] = -0.0
        assert math.copysign(
            1.0, record["fold_input"]["provenance"]["numeric_zero"]
        ) == -1.0
    _rehash_development_pair(fit_record, series_record)

    with pytest.raises(RuntimeError, match="schema|fold|provenance|artifact|context"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


def test_development_artifact_valid_context_is_canonical_and_self_hashed() -> None:
    fold = _development_fold_with_numeric_zero_provenance_for_test()
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )
    expected_fold_input = residual_train_module._fold_input_payload(fold)
    expected_provenance = dict(fold.provenance)

    for record in (fit_record, series_record):
        assert record["fold_input_sha256"] == residual_train_module._canonical_sha256(
            record["fold_input"]
        )
        assert _canonical_json_bytes(record["fold_input"]) == (
            _canonical_json_bytes(expected_fold_input)
        )
        assert _canonical_json_bytes(record["provenance"]) == (
            _canonical_json_bytes(expected_provenance)
        )
    _authenticate_complete_development_pair_for_test(
        fold, fit_record, series_record
    )


@pytest.mark.parametrize(
    "corruption",
    (
        "changed_order",
        "too_few",
        "too_many",
        "duplicate_fold",
        "wrong_result_fold",
        "changed_fold_input",
        "static_not_in_fit",
        "boolean_primary_seed",
    ),
)
def test_trusted_gate_evidence_authenticates_fold_and_fit_records(
    corruption: str,
) -> None:
    folds, fit_records, series_records = _development_gate_inputs_for_test()
    folds = list(folds)
    fit_records = list(copy.deepcopy(fit_records))
    series_records = list(copy.deepcopy(series_records))
    if corruption == "changed_order":
        folds[0], folds[1] = folds[1], folds[0]
    elif corruption == "too_few":
        folds.pop()
        fit_records.pop()
        series_records.pop()
    elif corruption == "too_many":
        folds.append(folds[-1])
        fit_records.append(copy.deepcopy(fit_records[-1]))
        series_records.append(copy.deepcopy(series_records[-1]))
    elif corruption == "duplicate_fold":
        folds[1] = folds[0]
    elif corruption == "wrong_result_fold":
        series_records[0]["result"]["fold"] = folds[1].name
        _rehash_development_record(series_records[0])
    elif corruption == "changed_fold_input":
        fit_records[0]["fold_input_sha256"] = "f" * 64
        _rehash_development_record(fit_records[0])
    elif corruption == "boolean_primary_seed":
        fit_records[0]["primary_seed"] = True
        fit_records[0]["fit"]["primary_seed"] = True
        fit_records[0]["fit"]["residual"]["primary_seed"] = True
        _rehash_development_record(fit_records[0])
        series_records[0]["primary_seed"] = True
        series_records[0]["result"]["primary_seed"] = True
        series_records[0]["fit_payload_sha256"] = fit_records[0]["payload_sha256"]
        _rehash_development_record(series_records[0])
    else:
        old_identity = _development_static_identity_for_test()
        new_identity = "safe-not-derived-from-fit"
        series_records[0]["expected_safe_static"] = [
            {"identity": new_identity, "family": "mes"}
        ]
        candidate = series_records[0]["result"]["safe_static"].pop(old_identity)
        series_records[0]["result"]["safe_static"][new_identity] = candidate
        _rehash_development_record(series_records[0])

    with pytest.raises(RuntimeError, match="development|trusted|artifact|fold"):
        residual_train_module.DevelopmentGateEvidence(
            folds, fit_records, series_records
        )


def test_trusted_gate_evidence_is_recursive_immutable_restartable_and_json_ready() -> None:
    folds, fit_records, series_records = _development_gate_inputs_for_test()
    evidence = residual_train_module.DevelopmentGateEvidence(
        folds, fit_records, series_records
    )
    fit_records[0]["fold"] = "caller-mutated"
    series_records[0]["result"]["series_results"].clear()

    for snapshot in (
        evidence,
        copy.copy(evidence),
        copy.deepcopy(evidence),
        pickle.loads(pickle.dumps(evidence)),
    ):
        assert snapshot.fit_records[0]["fold"] == folds[0].name
        assert snapshot.city_results
        with pytest.raises((AttributeError, TypeError)):
            snapshot.fit_records[0]["fold"] = "mutated"
        with pytest.raises((AttributeError, TypeError)):
            snapshot.series_records[0]["result"]["series_results"].clear()
        json.dumps(asdict(snapshot), sort_keys=True, allow_nan=False)


def _valid_fake_development_result(fold: DevelopmentFold) -> dict[str, object]:
    series_results = [
        _development_result_payload_for_test(
            _development_result(
                fold=fold.name,
                seed=seed,
                series=row.ref.key,
                residual_csd=1.0 if seed == 42 else 0.8,
            )
        )
        for seed in (1, 2, 42)
        for row in fold.test
    ]
    result = {
        "fold": fold.name,
        "kind": fold.kind,
        "primary_seed": 42,
        "fit_training_series": [row.ref.key for row in fold.train],
        "scored_series": [row.ref.key for row in fold.test],
        "series_results": series_results,
        "safe_static": _development_safe_static_results_for_test(
            [row.ref.key for row in fold.test]
        ),
        "actuation": [],
    }
    _attach_exact_actuation_for_test(fold, result, set())
    return result


def _restart_test_fold(
    *,
    train_key: str = "Poland/A/train",
    scored_key: str = "Poland/B/scored",
    train_year: int = 2022,
    scored_year: int = 2023,
    kind: str = "city",
    split_sha: str = "b" * 64,
) -> DevelopmentFold:
    return DevelopmentFold(
        "city_out_Poland_B",
        kind,
        (_development_series(train_key, train_year),),
        (_development_series(scored_key, scored_year, scored=True),),
        "a" * 64,
        {
            "corpus_manifest_sha256": "a" * 64,
            "split_sha256": split_sha,
            "input_blob_sha256": {"blob": "c" * 64},
        },
    )


def test_development_fold_rejects_missing_actual_training_membership() -> None:
    row = _development_series("Poland/A/train", 2022)
    row.train_only = {}

    with pytest.raises(ValueError, match="training|train_only"):
        DevelopmentFold("city_out_Poland_B", "city", (row,), (), "a" * 64)


def _multi_year_development_series(
    key: str,
    ref_years: tuple[object, ...],
    train_years: tuple[object, ...],
    test_years: tuple[object, ...],
    train_instance_years: tuple[object, ...],
    all_instance_years: tuple[object, ...],
) -> SeriesData:
    instances = {
        year: _development_instance(f"synthetic://{key}/{year}", int(year))
        for year in set((*train_instance_years, *all_instance_years))
        if isinstance(year, int)
    }
    return SeriesData(
        ref=SeriesRef(key, ref_years, ()),
        train_years=train_years,
        test_years=test_years,
        train_only={year: instances[year] for year in train_instance_years},
        all_years={year: instances[year] for year in all_instance_years},
    )


def test_development_fold_accepts_valid_training_warmup_and_all_year_scoring() -> None:
    training = _multi_year_development_series(
        "Poland/A/train", (2020, 2021), (2020, 2021), (), (2020, 2021), (2020, 2021)
    )
    warmup = _multi_year_development_series(
        "Poland/B/warmup", (2020, 2021, 2022), (), (2021, 2022), (), (2020, 2021, 2022)
    )
    all_year = _multi_year_development_series(
        "Poland/C/all", (2020, 2021), (), (2020, 2021), (), (2020, 2021)
    )

    fold = DevelopmentFold(
        "city_out_Poland_B", "city", (training,), (warmup, all_year), "a" * 64
    )

    assert fold.train[0].train_years == (2020, 2021)
    assert fold.test[0].test_years == (2021, 2022)
    assert fold.test[1].test_years == (2020, 2021)


@pytest.mark.parametrize(
    ("role", "row"),
    (
        (
            "test",
            lambda: _multi_year_development_series(
                "Poland/B/reversed-scored",
                (2022, 2021),
                (),
                (2022, 2021),
                (),
                (2022, 2021),
            ),
        ),
        (
            "test",
            lambda: _multi_year_development_series(
                "Poland/B/reversed-warmup",
                (2020, 2019, 2021),
                (),
                (2021,),
                (),
                (2020, 2019, 2021),
            ),
        ),
        (
            "train",
            lambda: _multi_year_development_series(
                "Poland/A/reversed-training",
                (2021, 2020),
                (2021, 2020),
                (),
                (2021, 2020),
                (2021, 2020),
            ),
        ),
    ),
)
def test_development_fold_rejects_nonchronological_year_tuples(
    role: str, row
) -> None:
    train = (row(),) if role == "train" else ()
    test = (row(),) if role == "test" else ()

    with pytest.raises(ValueError, match="increasing|chronological"):
        DevelopmentFold("city_out_Poland_B", "city", train, test, "a" * 64)


def test_normalized_fold_rebuilds_fit_only_training_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = {
        "Poland/A/train": SeriesRef("Poland/A/train", (2020, 2021), ()),
        "Poland/B/scored": SeriesRef("Poland/B/scored", (2020, 2021), ()),
    }
    split = Split(
        "synthetic",
        (("Poland/A/train", (2020,)),),
        (("Poland/B/scored", (2021,)),),
    )

    def load(ref: SeriesRef, years: tuple[int, ...]):
        return {
            year: _development_instance(f"synthetic://{ref.key}/{year}", year)
            for year in years
        }

    monkeypatch.setattr(residual_train_module, "load_series", load)

    train, scored, _ = residual_train_module._normalized_fold_views(split, index)
    fold = DevelopmentFold("city_out_Poland_B", "city", train, scored, "a" * 64)

    assert fold.train[0].ref.years == (2020,)
    assert fold.test[0].ref.years == (2020, 2021)


@pytest.mark.parametrize(
    ("role", "row"),
    (
        (
            "train",
            lambda: _multi_year_development_series(
                "Poland/A/x", (2020,), (2020, 2020), (), (2020,), (2020,)
            ),
        ),
        (
            "train",
            lambda: _multi_year_development_series(
                "Poland/A/x", (2020,), (True,), (), (2020,), (2020,)
            ),
        ),
        (
            "train",
            lambda: _multi_year_development_series(
                "Poland/A/x", (2020,), (2020,), (2020,), (2020,), (2020,)
            ),
        ),
        (
            "train",
            lambda: _multi_year_development_series(
                "Poland/A/x", (2020,), (2020,), (), (), (2020,)
            ),
        ),
        (
            "train",
            lambda: _multi_year_development_series(
                "Poland/A/x", (2020,), (2020,), (), (2020,), (2020, 2021)
            ),
        ),
        (
            "train",
            lambda: _multi_year_development_series(
                "Poland/A/x", (2020, 2021), (2020,), (), (2020,), (2020,)
            ),
        ),
        (
            "test",
            lambda: _multi_year_development_series(
                "Poland/A/x", (2020,), (2020,), (2020,), (), (2020,)
            ),
        ),
        (
            "test",
            lambda: _multi_year_development_series(
                "Poland/A/x", (2020,), (), (2020,), (2020,), (2020,)
            ),
        ),
        (
            "test",
            lambda: _multi_year_development_series(
                "Poland/A/x", (2020,), (), (2020, 2020), (), (2020,)
            ),
        ),
        (
            "test",
            lambda: _multi_year_development_series(
                "Poland/A/x", (2020,), (), (), (), (2020,)
            ),
        ),
        (
            "test",
            lambda: _multi_year_development_series(
                "Poland/A/x", (2020, 2021), (), (2020,), (), (2020,)
            ),
        ),
        (
            "test",
            lambda: _multi_year_development_series(
                "Poland/A/x", (2022, 2021), (), (2021,), (), (2022, 2021)
            ),
        ),
        (
            "test",
            lambda: _multi_year_development_series(
                "Poland/A/x", (2020, 2022, 2021), (), (2020, 2021), (), (2020, 2022, 2021)
            ),
        ),
    ),
)
def test_development_fold_rejects_role_inconsistent_year_views(
    role: str, row
) -> None:
    train = (row(),) if role == "train" else ()
    test = (row(),) if role == "test" else ()

    with pytest.raises((TypeError, ValueError), match="year|training|scored|warm-up|role"):
        DevelopmentFold("city_out_Poland_B", "city", train, test, "a" * 64)


def test_fold_input_hash_includes_actual_train_only_membership_on_bypass() -> None:
    canonical = _restart_test_fold()
    malformed_row = replace(canonical.train[0], train_only={})
    malformed = object.__new__(DevelopmentFold)
    for name in (
        "name",
        "kind",
        "test",
        "source_manifest_sha256",
        "provenance",
    ):
        object.__setattr__(malformed, name, getattr(canonical, name))
    object.__setattr__(malformed, "train", (malformed_row,))

    assert residual_train_module._fold_input_sha256(malformed) != (
        residual_train_module._fold_input_sha256(canonical)
    )


def test_stale_resume_rejects_lower_level_train_only_membership_bypass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    canonical = _restart_test_fold()
    monkeypatch.setattr(residual_train_module, "DEVELOPMENT_ROOT", tmp_path)
    monkeypatch.setattr(
        residual_train_module,
        "_fit_development_training",
        lambda rows, *args, **kwargs: _complete_fit_for_training_rows_for_test(rows),
    )
    monkeypatch.setattr(
        residual_train_module,
        "_evaluate_development_scored",
        lambda fold, fit, workers: _valid_fake_development_result(fold),
    )
    _fit_development_fold_unchecked_for_test(canonical, workers=1)
    malformed_row = replace(canonical.train[0], train_only={})
    malformed = object.__new__(DevelopmentFold)
    for name in (
        "name",
        "kind",
        "test",
        "source_manifest_sha256",
        "provenance",
    ):
        object.__setattr__(malformed, name, getattr(canonical, name))
    object.__setattr__(malformed, "train", (malformed_row,))
    monkeypatch.setattr(
        residual_train_module,
        "_fit_development_training",
        lambda *args, **kwargs: pytest.fail("stale resume attempted a fresh fit"),
    )

    with pytest.raises(RuntimeError, match="divergent|fold input"):
        _fit_development_fold_unchecked_for_test(malformed, workers=1)


def _write_rehashed_payload(path: Path, mutate, *, indent: int | None = None) -> None:
    payload = json.loads(path.read_text())
    mutate(payload)
    payload.pop("payload_sha256", None)
    payload["payload_sha256"] = residual_train_module._canonical_sha256(payload)
    if indent is None:
        path.write_bytes(_canonical_json_bytes(payload))
    else:
        path.write_text(json.dumps(payload, indent=indent, sort_keys=True) + "\n")


@pytest.mark.parametrize(
    "changed_context",
    ("training", "scored", "provenance", "years", "kind"),
)
def test_resume_rejects_changed_current_fold_context(
    monkeypatch, tmp_path, changed_context: str
) -> None:
    original = _restart_test_fold()
    monkeypatch.setattr(residual_train_module, "DEVELOPMENT_ROOT", tmp_path)
    monkeypatch.setattr(
        residual_train_module,
        "_fit_development_training",
        lambda rows, *args, **kwargs: _complete_fit_for_training_rows_for_test(rows),
    )
    monkeypatch.setattr(
        residual_train_module,
        "_evaluate_development_scored",
        lambda fold, fit, workers: _valid_fake_development_result(fold),
    )
    _fit_development_fold_unchecked_for_test(original, workers=1)
    changes = {
        "training": {"train_key": "Poland/A/different"},
        "scored": {"scored_key": "Poland/C/different"},
        "provenance": {"split_sha": "d" * 64},
        "years": {"scored_year": 2024},
        "kind": {"kind": "district"},
    }

    with pytest.raises(RuntimeError, match="divergent|fold input"):
        _fit_development_fold_unchecked_for_test(
            _restart_test_fold(**changes[changed_context]), workers=1
        )


@pytest.mark.parametrize(
    "artifact_corruption",
    ("reformatted", "schema", "kind", "provenance", "year", "result_inventory"),
)
def test_resume_rejects_noncanonical_or_semantically_wrong_artifact(
    monkeypatch, tmp_path, artifact_corruption: str
) -> None:
    fold = _restart_test_fold()
    monkeypatch.setattr(residual_train_module, "DEVELOPMENT_ROOT", tmp_path)
    monkeypatch.setattr(
        residual_train_module,
        "_fit_development_training",
        lambda rows, *args, **kwargs: _complete_fit_for_training_rows_for_test(rows),
    )
    monkeypatch.setattr(
        residual_train_module,
        "_evaluate_development_scored",
        lambda passed, fit, workers: _valid_fake_development_result(passed),
    )
    _fit_development_fold_unchecked_for_test(fold, workers=1)
    fit_path = tmp_path / "folds" / fold.name / "fit.json"
    series_path = tmp_path / "folds" / fold.name / "per_series.json"
    if artifact_corruption == "reformatted":
        _write_rehashed_payload(series_path, lambda payload: None, indent=2)
    elif artifact_corruption == "result_inventory":
        _write_rehashed_payload(
            series_path,
            lambda payload: payload["result"].__setitem__("scored_series", []),
        )
    else:
        def mutate(payload):
            if artifact_corruption == "schema":
                payload["schema_version"] = 999
            elif artifact_corruption == "kind":
                payload["kind"] = "district"
            elif artifact_corruption == "provenance":
                payload["provenance"] = {"split_sha256": "f" * 64}
            else:
                payload["training_view"] = [{"series": "x", "years": [1900]}]
        _write_rehashed_payload(fit_path, mutate)

    with pytest.raises(RuntimeError, match="divergent|artifact|fold result"):
        _fit_development_fold_unchecked_for_test(fold, workers=1)


@pytest.mark.parametrize(
    (
        "focus",
        "failed_comparisons",
        "reference_welfare",
        "residual_welfare",
        "reference_exclusion",
        "residual_exclusion",
    ),
    (
        ("welfare_vs_anchor", ("welfare_vs_anchor", "welfare_vs_mes"), 100.0, 98.9, 0.1, 0.1),
        ("welfare_vs_mes", ("welfare_vs_anchor", "welfare_vs_mes"), 100.0, 98.8, 0.1, 0.1),
        ("exclusion_vs_anchor", ("exclusion_vs_anchor", "exclusion_vs_mes"), 100.0, 100.0, 0.1, 0.11),
        ("exclusion_vs_mes", ("exclusion_vs_anchor", "exclusion_vs_mes"), 100.0, 100.0, 0.1, 0.12),
    ),
)
def test_gate_retains_recomputable_dual_safety_records(
    focus: str,
    failed_comparisons: tuple[str, ...],
    reference_welfare: float,
    residual_welfare: float,
    reference_exclusion: float,
    residual_exclusion: float,
) -> None:
    payload = _passing_development_payload()
    payload["city_results"] = [
        replace(
            row,
            residual=(
                {
                    "csd": row.residual["csd"],
                    "welfare": residual_welfare,
                    "exclusion": residual_exclusion,
                }
                if row.seed == 42
                else row.residual
            ),
            anchor={
                "csd": row.anchor["csd"],
                "welfare": reference_welfare,
                "exclusion": reference_exclusion,
            },
            mes={
                "csd": row.mes["csd"],
                "welfare": reference_welfare,
                "exclusion": reference_exclusion,
            },
        )
        if row.fold == "city_out_Poland_Gdynia"
        else row
        for row in payload["city_results"]
    ]
    for candidate in payload["safe_static_results"]["city_out_Poland_Gdynia"].values():
        for metrics in candidate["series"].values():
            metrics["welfare"] = reference_welfare
            metrics["exclusion"] = reference_exclusion

    result = evaluate_development_gate(_trusted_development_gate_evidence(payload))
    record = result["observed"]["city_safety"]["city_out_Poland_Gdynia"]
    checks = record["comparisons"]

    assert checks["welfare_vs_anchor"] == (
        record["residual_welfare"] >= record["anchor_welfare_threshold"]
    )
    assert checks["welfare_vs_mes"] == (
        record["residual_welfare"] >= record["mes_welfare_threshold"]
    )
    assert checks["exclusion_vs_anchor"] == (
        record["residual_exclusion"] <= record["anchor_exclusion_threshold"]
    )
    assert checks["exclusion_vs_mes"] == (
        record["residual_exclusion"] <= record["mes_exclusion_threshold"]
    )
    assert checks[focus] is False
    assert all(checks[comparison] is False for comparison in failed_comparisons)
    assert record["failed_comparisons"] == list(failed_comparisons)
    assert result["conditions"]["dual_reference_safety"] == all(
        all(city["comparisons"].values())
        for city in result["observed"]["city_safety"].values()
    )


@pytest.mark.parametrize(
    ("series", "expected"),
    (
        ("Poland/Gdynia/x| large", "Poland/Gdynia/x"),
        ("Poland/Gdynia/x| SMALL", "Poland/Gdynia/x"),
        ("Poland/Gdynia/x | large", "Poland/Gdynia/x"),
        ("Poland/Gdynia/x | SMALL", "Poland/Gdynia/x"),
        ("Poland/Gdynia/x| large/child", "Poland/Gdynia/x| large/child"),
        ("Poland/Gdynia/x|| large", "Poland/Gdynia/x|"),
    ),
)
def test_development_cluster_key_handles_exact_pipe_spacing_forms(
    series: str, expected: str
) -> None:
    assert development_cluster_key(series) == expected


def _replace_static_anchor_for_test(
    row: dict[str, object], family: str, logits: tuple[float, ...]
) -> None:
    anchor = _anchor(family, logits)
    row["anchor"] = residual_train_module._anchor_payload(anchor)
    row["anchor_sha256"] = residual_train_module.anchor_payload_sha256(anchor)


def _set_exact_development_signatures_for_test(
    result: dict[str, object], actuated_series: set[str]
) -> None:
    primary_seed = result["primary_seed"]
    for row in result["series_results"]:
        actuated = row["seed"] == primary_seed and row["series"] in actuated_series
        row["signatures"] = {
            "residual": "actuated" if actuated else "anchor-equivalent",
            "anchor": "anchor-equivalent",
            "actuated": actuated,
        }


def _actuation_record_payload_for_test(
    instance: str, *, actuated: bool
) -> dict[str, object]:
    if not actuated:
        return {
            "instance": instance,
            "actuated": False,
            "first_grid_t": None,
            "refined_low": None,
            "refined_high": None,
            "csd_direction": None,
            "anchor_csd": 1.0,
            "changed_csd": None,
        }
    return {
        "instance": instance,
        "actuated": True,
        "first_grid_t": 1.0 / 64.0,
        "refined_low": 0.0,
        "refined_high": 1.0 / 16384.0,
        "csd_direction": "improves",
        "anchor_csd": 1.0,
        "changed_csd": 0.5,
    }


def _attach_exact_actuation_for_test(
    fold: DevelopmentFold,
    result: dict[str, object],
    actuated_series: set[str],
) -> None:
    entries = sorted(
        (
            row.all_years[year].path,
            row.ref.key,
        )
        for row in fold.test
        for year in row.test_years
    )
    result["actuation"] = [
        _actuation_record_payload_for_test(
            path, actuated=series in actuated_series
        )
        for path, series in entries
    ]
    _set_exact_development_signatures_for_test(result, actuated_series)


def _set_scored_metrics_equal_for_test(
    result: dict[str, object], *, seed: int, series: str | None = None
) -> None:
    for row in result["series_results"]:
        if row["seed"] == seed and (series is None or row["series"] == series):
            row["residual"] = copy.deepcopy(row["anchor"])


def _select_canonical_zero_for_scored_test(
    fit_record: dict[str, object], series_record: dict[str, object], seed: int
) -> None:
    residual = fit_record["fit"]["residual"]
    retained = next(row for row in residual["seeds"] if row["seed"] == seed)
    zero = next(
        candidate
        for candidate in retained["archive"]
        if candidate["amplitude"] == 0.0
        and all(value == 0.0 for value in candidate["weights"])
        and all(value == 0.0 for value in candidate["endpoint"])
    )
    retained["selected"] = copy.deepcopy(zero)
    retained["archive"] = [copy.deepcopy(zero)]
    retained["archive_count"] = 1
    _repair_development_fit_links_for_test(fit_record, series_record)


def _copy_selected_policy_weights_for_scored_test(
    fit_record: dict[str, object],
    series_record: dict[str, object],
    *,
    target_seed: int,
    source_seed: int,
) -> None:
    residual = fit_record["fit"]["residual"]
    target = next(row for row in residual["seeds"] if row["seed"] == target_seed)
    source = next(row for row in residual["seeds"] if row["seed"] == source_seed)
    old_selected = target["selected"]
    zero = next(
        candidate for candidate in target["archive"]
        if candidate["amplitude"] == 0.0
    )
    base_metrics = residual_train_module._authenticated_residual_metrics(
        old_selected["metrics"]
    )
    anchor_metrics = residual_train_module._authenticated_residual_metrics(
        zero["metrics"]
    )
    mes_metrics = {
        city: CityMetrics(**metrics)
        for city, metrics in fit_record["fit"]["static"]["mes_metrics"].items()
    }
    copied = score_residual_candidate(
        tuple(source["selected"]["weights"]),
        tuple(source["selected"]["endpoint"]),
        source["selected"]["amplitude"],
        base_metrics,
        anchor_metrics,
        mes_metrics,
        old_selected["source"],
    )
    copied_payload = residual_train_module._residual_candidate_payload(copied)
    target["archive"] = [
        copy.deepcopy(zero),
        copy.deepcopy(copied_payload),
    ]
    target["archive_count"] = 2
    target["selected"] = copy.deepcopy(copied_payload)
    _repair_development_fit_links_for_test(fit_record, series_record)


def _select_non_mes_static_anchor_for_scored_test(
    fit_record: dict[str, object], series_record: dict[str, object]
) -> tuple[str, str]:
    static = fit_record["fit"]["static"]
    senior = next(
        row
        for row in static["candidates"]
        if row["anchor"]["family"] == "senior"
        and row["anchor"]["free_logits"] == [math.log(2.0)]
    )
    _set_static_candidate_evaluation_for_test(
        senior, csd=0.5, welfare=10.0, safe=True
    )
    senior_candidate = residual_train_module._authenticated_static_candidate(senior)
    age = next(
        row for row in static["candidates"]
        if row["anchor"]["family"] == "age"
    )
    age_anchor = age_initializer((senior_candidate,))
    age["anchor"] = residual_train_module._anchor_payload(age_anchor)
    age["anchor_sha256"] = residual_train_module.anchor_payload_sha256(age_anchor)
    age["metrics"] = copy.deepcopy(senior["metrics"])
    age["safe"] = senior["safe"]
    age["objective"] = senior["objective"]
    age_candidate = residual_train_module._authenticated_static_candidate(age)
    age_sex = next(
        row for row in static["candidates"]
        if row["anchor"]["family"] == "age_sex"
    )
    age_sex_anchor = age_sex_initializer((age_candidate,))
    age_sex["anchor"] = residual_train_module._anchor_payload(age_sex_anchor)
    age_sex["anchor_sha256"] = residual_train_module.anchor_payload_sha256(
        age_sex_anchor
    )
    age_sex["metrics"] = copy.deepcopy(senior["metrics"])
    age_sex["safe"] = senior["safe"]
    age_sex["objective"] = senior["objective"]
    candidates = tuple(
        residual_train_module._authenticated_static_candidate(row)
        for row in static["candidates"]
    )
    winner = select_static_anchor(candidates, anchor_search_config().tie_tolerance)
    selected = next(
        row for row in static["candidates"]
        if row["anchor_sha256"]
        == residual_train_module.anchor_payload_sha256(winner.anchor)
    )
    _retarget_residual_anchor_csd_for_test(fit_record, series_record, selected)
    mes_identity = next(
        row["anchor_sha256"]
        for row in static["candidates"]
        if row["anchor"]["family"] == "mes"
    )
    return selected["anchor_sha256"], mes_identity


@pytest.mark.parametrize("loss_mode", ("finite_improvement", "all_infinite"))
def test_static_writer_deduplicates_every_cma_final_anchor_by_canonical_hash(
    monkeypatch: pytest.MonkeyPatch, loss_mode: str
) -> None:
    config = replace(
        anchor_search_config(), seeds=(1,), generations=2, popsize=4
    )
    initializer = _anchor("age", (0.0, 0.0, 0.0))
    metrics = {"Poland/Test": CityMetrics(1.0, 10.0, 0.1)}

    def synthetic_score(anchor, data, cfg, mes_metrics):
        del data, cfg, mes_metrics
        if loss_mode == "all_infinite":
            return AnchorCandidate(anchor, float("inf"), metrics, False, "raw", None)
        objective = math.fsum(
            (value - target) ** 2
            for value, target in zip(anchor.free_logits, (0.5, -0.25, 0.75))
        )
        return AnchorCandidate(anchor, objective, metrics, True, "raw", None)

    monkeypatch.setattr(
        residual_train_module, "score_static_candidate", synthetic_score
    )
    archive = residual_train_module._archive_cma_family(
        "age", initializer, (), config, metrics, workers=1
    )
    final = archive[-1]
    final_sha = residual_train_module.anchor_payload_sha256(final.anchor)
    earlier_shas = [
        residual_train_module.anchor_payload_sha256(row.anchor)
        for row in archive[:-1]
    ]
    retained = residual_train_module._deduplicate_candidates(archive)

    assert final.source == "age-seed-1-final"
    assert final_sha in earlier_shas
    if loss_mode == "all_infinite":
        assert final_sha == earlier_shas[0]
    else:
        assert final_sha in earlier_shas[1:]
    assert all(not row.source.endswith("-final") for row in retained)


def test_trusted_gate_rejects_reviewer_selected_out_of_bound_cma_root() -> None:
    fold = _canonical_development_gate_folds()[0]
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )
    static = fit_record["fit"]["static"]
    selected = _insert_static_candidate_for_test(
        static,
        "age",
        (7.0, 0.0, 0.0),
        "age-seed-1-generation-0",
        1,
        before_first_family="age_sex",
    )
    _set_static_candidate_evaluation_for_test(
        selected, csd=0.5, welfare=10.0, safe=True
    )
    age_sex = next(
        row for row in static["candidates"]
        if row["anchor"]["family"] == "age_sex"
    )
    _replace_static_anchor_for_test(
        age_sex, "age_sex", (0.0, 7.0, 7.0, 0.0, 0.0, 0.0, 0.0)
    )
    _set_static_candidate_evaluation_for_test(
        age_sex, csd=0.5, welfare=10.0, safe=True
    )
    _retarget_residual_anchor_csd_for_test(fit_record, series_record, selected)

    with pytest.raises(RuntimeError, match="bound|CMA|provenance|static"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


@pytest.mark.parametrize("family", ("age", "age_sex"))
@pytest.mark.parametrize(
    ("value", "accepted"),
    (
        (6.0, True),
        (-6.0, True),
        (math.nextafter(6.0, math.inf), False),
        (math.nextafter(-6.0, -math.inf), False),
    ),
)
def test_trusted_gate_authenticates_cma_generation_box_boundaries(
    family: str, value: float, accepted: bool
) -> None:
    fold = _canonical_development_gate_folds()[0]
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )
    static = fit_record["fit"]["static"]
    dimensions = 3 if family == "age" else 7
    logits = (value,) + (0.0,) * (dimensions - 1)
    inserted = _insert_static_candidate_for_test(
        static,
        family,
        logits,
        f"{family}-seed-1-generation-0",
        1,
        before_first_family="age_sex" if family == "age" else None,
    )
    _set_static_candidate_evaluation_for_test(
        inserted, csd=1.0, welfare=0.0, safe=False
    )
    _repair_development_fit_links_for_test(fit_record, series_record)

    if accepted:
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )
    else:
        with pytest.raises(RuntimeError, match="bound|CMA|provenance|static"):
            _authenticate_complete_development_pair_for_test(
                fold, fit_record, series_record
            )


@pytest.mark.parametrize("family", ("age", "age_sex"))
@pytest.mark.parametrize("value", (5.5, -5.5))
def test_trusted_gate_rejects_unreachable_retained_cma_final_sources(
    family: str, value: float
) -> None:
    fold = _canonical_development_gate_folds()[0]
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )
    static = fit_record["fit"]["static"]
    dimensions = 3 if family == "age" else 7
    inserted = _insert_static_candidate_for_test(
        static,
        family,
        (value,) + (0.0,) * (dimensions - 1),
        f"{family}-seed-1-final",
        1,
        before_first_family="age_sex" if family == "age" else None,
    )
    _set_static_candidate_evaluation_for_test(
        inserted, csd=1.0, welfare=0.0, safe=False
    )
    _repair_development_fit_links_for_test(fit_record, series_record)

    with pytest.raises(RuntimeError, match="final|lineage|source|provenance"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


def test_trusted_gate_rejects_reviewer_cross_family_policy_twin() -> None:
    fold = _canonical_development_gate_folds()[0]
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )
    static = fit_record["fit"]["static"]
    senior_logit = math.log(2.0)
    senior = next(
        row for row in static["candidates"]
        if row["source"] == "senior-grid-alpha-1"
    )
    selected = _insert_static_candidate_for_test(
        static,
        "age",
        (0.0, 0.0, senior_logit),
        "age-seed-1-generation-0",
        1,
        before_first_family="age_sex",
    )
    assert senior["safe"] is False
    _set_static_candidate_evaluation_for_test(
        selected, csd=0.5, welfare=10.0, safe=True
    )
    age_sex = next(
        row for row in static["candidates"]
        if row["anchor"]["family"] == "age_sex"
    )
    _replace_static_anchor_for_test(
        age_sex,
        "age_sex",
        (0.0, 0.0, 0.0, 0.0, 0.0, senior_logit, senior_logit),
    )
    _set_static_candidate_evaluation_for_test(
        age_sex, csd=0.5, welfare=10.0, safe=True
    )
    _retarget_residual_anchor_csd_for_test(fit_record, series_record, selected)

    with pytest.raises(RuntimeError, match="policy|twin|evaluation|static"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


@pytest.mark.parametrize(
    ("pair", "mode"),
    (
        ("senior_age", "safe_unsafe"),
        ("senior_age", "finite_safe"),
        ("age_age_sex", "safe_unsafe"),
        ("age_age_sex", "finite_safe"),
        ("senior_age_sex", "safe_unsafe"),
    ),
)
def test_trusted_gate_authenticates_global_static_policy_twins(
    pair: str, mode: str
) -> None:
    fold = _canonical_development_gate_folds()[0]
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )
    static = fit_record["fit"]["static"]
    selected = None
    if pair.startswith("senior"):
        value = math.log(2.0)
        parent = next(
            row for row in static["candidates"]
            if row["source"] == "senior-grid-alpha-1"
        )
        family = "age" if pair == "senior_age" else "age_sex"
        logits = (
            (0.0, 0.0, value)
            if family == "age"
            else (0.0, 0.0, 0.0, 0.0, 0.0, value, value)
        )
        selected = _insert_static_candidate_for_test(
            static,
            family,
            logits,
            f"{family}-seed-1-generation-0",
            1,
            before_first_family="age_sex" if family == "age" else None,
        )
    else:
        parent = _insert_static_candidate_for_test(
            static,
            "age",
            (0.2, 0.3, 0.4),
            "age-seed-1-generation-0",
            1,
            before_first_family="age_sex",
        )
        selected = _insert_static_candidate_for_test(
            static,
            "age_sex",
            (0.0, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4),
            "age_sex-seed-1-generation-0",
            1,
        )
    if mode == "finite_safe":
        _set_static_candidate_evaluation_for_test(
            parent, csd=1.2, welfare=10.0, safe=True
        )
    _set_static_candidate_evaluation_for_test(
        selected, csd=0.5, welfare=10.0, safe=True
    )
    if pair == "senior_age":
        age_sex = next(
            row for row in static["candidates"]
            if row["anchor"]["family"] == "age_sex"
        )
        value = math.log(2.0)
        _replace_static_anchor_for_test(
            age_sex,
            "age_sex",
            (0.0, 0.0, 0.0, 0.0, 0.0, value, value),
        )
        _set_static_candidate_evaluation_for_test(
            age_sex, csd=0.5, welfare=10.0, safe=True
        )
    _retarget_residual_anchor_csd_for_test(fit_record, series_record, selected)

    with pytest.raises(RuntimeError, match="policy|twin|evaluation|static"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


def test_trusted_gate_accepts_global_zero_policy_twins_with_exact_evaluations() -> None:
    fold = _canonical_development_gate_folds()[0]
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )

    _authenticate_complete_development_pair_for_test(
        fold, fit_record, series_record
    )


def test_trusted_gate_accepts_identically_evaluated_noninitializer_policy_twins() -> None:
    fold = _canonical_development_gate_folds()[0]
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, _complete_development_result_for_test(fold)
    )
    static = fit_record["fit"]["static"]
    senior = next(
        row for row in static["candidates"]
        if row["source"] == "senior-grid-alpha-1"
    )
    twin = _insert_static_candidate_for_test(
        static,
        "age",
        (0.0, 0.0, math.log(2.0)),
        "age-seed-1-generation-0",
        1,
        before_first_family="age_sex",
    )
    twin["metrics"] = copy.deepcopy(senior["metrics"])
    twin["safe"] = senior["safe"]
    twin["objective"] = senior["objective"]
    _repair_development_fit_links_for_test(fit_record, series_record)

    _authenticate_complete_development_pair_for_test(
        fold, fit_record, series_record
    )


@pytest.mark.parametrize("flipped", (False, True))
def test_trusted_gate_rejects_empty_scan_even_when_flags_flip_gate(
    flipped: bool,
) -> None:
    folds, fit_records, series_records = _development_gate_inputs_for_test()
    for record in series_records:
        _set_exact_development_signatures_for_test(record["result"], set())
        record["result"]["actuation"] = []
        _rehash_development_record(record)
    if flipped:
        for index in (0, 1):
            record = series_records[index]
            first_series = record["result"]["scored_series"][0]
            _set_exact_development_signatures_for_test(
                record["result"], {first_series}
            )
            _rehash_development_record(record)

    with pytest.raises(RuntimeError, match="actuation|scan|inventory"):
        residual_train_module.DevelopmentGateEvidence(
            folds, fit_records, series_records
        )


@pytest.mark.parametrize(
    "corruption",
    (
        "missing",
        "extra",
        "duplicate",
        "reordered",
        "wrong_instance",
        "missing_field",
        "extra_field",
        "boolean_numeric_alias",
        "integer_numeric_alias",
        "signed_zero",
        "contradictory_csd",
        "contradictory_bracket",
        "primary_mismatch",
        "nonprimary_true",
        "residual_label",
        "anchor_label",
        "signature_extra",
    ),
)
def test_development_artifact_authenticates_retained_actuation_linkage(
    corruption: str,
) -> None:
    fold = _canonical_development_gate_folds()[0]
    result = _complete_development_result_for_test(fold)
    first_series = result["scored_series"][0]
    _attach_exact_actuation_for_test(fold, result, {first_series})
    records = result["actuation"]
    if corruption == "missing":
        records.pop()
    elif corruption == "extra":
        extra = copy.deepcopy(records[-1])
        extra["instance"] += "/extra"
        records.append(extra)
    elif corruption == "duplicate":
        records.append(copy.deepcopy(records[-1]))
    elif corruption == "reordered":
        records.reverse()
    elif corruption == "wrong_instance":
        records[0]["instance"] += "/wrong"
    elif corruption == "missing_field":
        records[0].pop("changed_csd")
    elif corruption == "extra_field":
        records[0]["extra"] = None
    elif corruption == "boolean_numeric_alias":
        records[0]["first_grid_t"] = True
    elif corruption == "integer_numeric_alias":
        records[0]["anchor_csd"] = 1
    elif corruption == "signed_zero":
        records[-1]["anchor_csd"] = -0.0
    elif corruption == "contradictory_csd":
        records[0]["csd_direction"] = "worsens"
    elif corruption == "contradictory_bracket":
        records[0]["refined_low"] = 1.0 / 16384.0
        records[0]["refined_high"] = 3.0 / 16384.0
    elif corruption == "primary_mismatch":
        row = next(
            row for row in result["series_results"]
            if row["seed"] == 42 and row["series"] == first_series
        )
        row["signatures"] = {
            "residual": "anchor-equivalent",
            "anchor": "anchor-equivalent",
            "actuated": False,
        }
    elif corruption == "nonprimary_true":
        row = next(
            row for row in result["series_results"]
            if row["seed"] == 1 and row["series"] == first_series
        )
        row["signatures"] = {
            "residual": "actuated",
            "anchor": "anchor-equivalent",
            "actuated": True,
        }
    elif corruption == "residual_label":
        result["series_results"][0]["signatures"]["residual"] = "changed"
    elif corruption == "anchor_label":
        result["series_results"][0]["signatures"]["anchor"] = "same"
    else:
        result["series_results"][0]["signatures"]["extra"] = False
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, result
    )

    with pytest.raises(RuntimeError, match="actuation|signature|scan|record|result"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


@pytest.mark.parametrize("mode", ("all_inert", "mixed"))
def test_development_artifact_accepts_canonical_retained_actuation_linkage(
    mode: str,
) -> None:
    fold = _canonical_development_gate_folds()[0]
    result = _complete_development_result_for_test(fold)
    actuated = {result["scored_series"][0]} if mode == "mixed" else set()
    _attach_exact_actuation_for_test(fold, result, actuated)
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, result
    )

    _authenticate_complete_development_pair_for_test(
        fold, fit_record, series_record
    )


def test_development_artifact_rejects_extra_scored_result_row_field() -> None:
    fold = _canonical_development_gate_folds()[0]
    result = _complete_development_result_for_test(fold)
    result["series_results"][0]["extra"] = None
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, result
    )

    with pytest.raises(RuntimeError, match="result|row|schema|canonical"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


@pytest.mark.parametrize("reference", ("residual", "anchor", "mes"))
@pytest.mark.parametrize("alias", ("integer", "signed_zero"))
def test_development_artifact_rejects_scored_result_numeric_aliases(
    reference: str, alias: str
) -> None:
    fold = _canonical_development_gate_folds()[0]
    result = _complete_development_result_for_test(fold)
    row = result["series_results"][0]
    if alias == "integer":
        assert row[reference]["welfare"] == 100.0
        row[reference]["welfare"] = 100
    else:
        row[reference]["exclusion"] = -0.0
        assert math.copysign(1.0, row[reference]["exclusion"]) == -1.0
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, result
    )

    with pytest.raises(RuntimeError, match="result|metric|canonical|payload"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


def test_development_artifact_rejects_reordered_scored_result_rows() -> None:
    fold = _canonical_development_gate_folds()[0]
    result = _complete_development_result_for_test(fold)
    result["series_results"][0], result["series_results"][1] = (
        result["series_results"][1],
        result["series_results"][0],
    )
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, result
    )

    with pytest.raises(RuntimeError, match="result|order|inventory"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


@pytest.mark.parametrize("alias", ("integer", "signed_zero"))
def test_development_artifact_rejects_safe_static_scored_metric_aliases(
    alias: str,
) -> None:
    fold = _canonical_development_gate_folds()[0]
    result = _complete_development_result_for_test(fold)
    candidate = next(iter(result["safe_static"].values()))
    metrics = candidate["series"][result["scored_series"][0]]
    if alias == "integer":
        assert metrics["welfare"] == 100.0
        metrics["welfare"] = 100
    else:
        metrics["exclusion"] = -0.0
        assert math.copysign(1.0, metrics["exclusion"]) == -1.0
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, result
    )

    with pytest.raises(RuntimeError, match="safe-static|metric|canonical|payload"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


def test_development_artifact_accepts_canonical_scored_result_rows() -> None:
    fold = _canonical_development_gate_folds()[0]
    result = _complete_development_result_for_test(fold)
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, result
    )

    _authenticate_complete_development_pair_for_test(
        fold, fit_record, series_record
    )


def test_round9_fit_payload_returns_the_authenticated_fold_fit() -> None:
    fold = _canonical_development_gate_folds()[0]
    _, payload = _complete_development_fold_fit_for_test(fold)

    authenticated = residual_train_module._validate_development_fit_payload(
        payload,
        42,
        _development_safe_static_inventory_for_test(),
        anchor_search_config(),
        residual_search_config(),
        tuple(row.ref.key for row in fold.train),
        ("Poland/Training",),
    )

    assert isinstance(authenticated, FoldFit)
    assert authenticated.primary_seed == 42
    assert tuple(row.seed for row in authenticated.seeds) == (1, 2, 42)


@pytest.mark.parametrize("reference", ("anchor", "mes"))
@pytest.mark.parametrize(
    ("metric", "replacement"),
    (("csd", -9.0), ("welfare", 999.0), ("exclusion", 0.9)),
)
def test_round9_rejects_one_row_common_reference_metric_contradiction(
    reference: str, metric: str, replacement: float
) -> None:
    fold = _canonical_development_gate_folds()[0]
    result = _complete_development_result_for_test(fold)
    target = next(
        row
        for row in result["series_results"]
        if row["seed"] == 2 and row["series"] == result["scored_series"][0]
    )
    target[reference][metric] = replacement
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, result
    )

    with pytest.raises(RuntimeError, match="anchor|MES|reference|metric|scored"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


def test_round9_rejects_anchor_contradiction_before_all_seeds_gate_decision() -> None:
    folds, fit_records, series_records = _development_gate_inputs_for_test()
    fit_records = list(copy.deepcopy(fit_records))
    series_records = list(copy.deepcopy(series_records))
    target = next(
        row
        for row in series_records[0]["result"]["series_results"]
        if row["seed"] == 1
    )
    target["anchor"]["csd"] = -9.0
    _rehash_development_record(series_records[0])

    with pytest.raises(RuntimeError, match="anchor|reference|metric|scored"):
        residual_train_module.DevelopmentGateEvidence(
            folds, fit_records, series_records
        )


def test_round9_rejects_mes_contradiction_before_dual_safety_decision() -> None:
    folds, fit_records, series_records = _development_gate_inputs_for_test()
    fit_records = list(copy.deepcopy(fit_records))
    series_records = list(copy.deepcopy(series_records))
    target = next(
        row
        for row in series_records[0]["result"]["series_results"]
        if row["seed"] == 42
    )
    target["mes"]["welfare"] = 1000.0
    _rehash_development_record(series_records[0])

    with pytest.raises(RuntimeError, match="MES|reference|metric|scored"):
        residual_train_module.DevelopmentGateEvidence(
            folds, fit_records, series_records
        )


@pytest.mark.parametrize("alias", ("integer", "boolean", "signed_zero"))
def test_round9_metric_aliases_stop_before_relationship_comparison(
    alias: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = _canonical_development_gate_folds()[0]
    result = _complete_development_result_for_test(fold)
    metrics = result["series_results"][0]["residual"]
    if alias == "integer":
        metrics["welfare"] = 100
    elif alias == "boolean":
        metrics["welfare"] = True
    else:
        metrics["exclusion"] = -0.0
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, result
    )
    monkeypatch.setattr(
        residual_train_module,
        "_authenticate_development_scored_relationships",
        lambda *args, **kwargs: pytest.fail(
            "noncanonical metric alias reached relationship comparison"
        ),
    )

    with pytest.raises(
        RuntimeError, match="result|metric|canonical|payload|evidence|finite"
    ):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


def test_round9_rejects_exact_zero_selected_policy_scored_contradiction() -> None:
    fold = _canonical_development_gate_folds()[0]
    result = _complete_development_result_for_test(fold)
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, result
    )
    for seed in (1, 2):
        _select_canonical_zero_for_scored_test(fit_record, series_record, seed)

    with pytest.raises(RuntimeError, match="zero|anchor|policy|residual|scored"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


def test_round9_rejects_nonadjacent_equal_selected_policy_metric_mismatch() -> None:
    fold = _canonical_development_gate_folds()[0]
    result = _complete_development_result_for_test(fold)
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, result
    )
    _copy_selected_policy_weights_for_scored_test(
        fit_record, series_record, target_seed=1, source_seed=42
    )

    with pytest.raises(RuntimeError, match="equal|policy|weight|residual|scored"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


def test_round9_rejects_primary_scan_inert_changed_metrics() -> None:
    fold = _canonical_development_gate_folds()[0]
    result = _complete_development_result_for_test(fold)
    target = next(
        row
        for row in result["series_results"]
        if row["seed"] == 42 and row["series"] == result["scored_series"][0]
    )
    target["residual"]["csd"] = 0.75
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, result
    )

    with pytest.raises(RuntimeError, match="inert|actuation|anchor|residual|scored"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


@pytest.mark.parametrize(
    "mode", ("canonical_zero", "equal_nonadjacent", "actuated_primary", "distinct")
)
def test_round9_accepts_consistent_residual_policy_consequences(mode: str) -> None:
    fold = _canonical_development_gate_folds()[0]
    result = _complete_development_result_for_test(fold)
    first_series = result["scored_series"][0]
    if mode == "canonical_zero":
        _set_scored_metrics_equal_for_test(result, seed=1)
    elif mode == "equal_nonadjacent":
        _set_scored_metrics_equal_for_test(result, seed=1)
    elif mode == "actuated_primary":
        target = next(
            row
            for row in result["series_results"]
            if row["seed"] == 42 and row["series"] == first_series
        )
        target["residual"]["csd"] = 0.8
        _attach_exact_actuation_for_test(fold, result, {first_series})
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, result
    )
    if mode == "canonical_zero":
        _select_canonical_zero_for_scored_test(fit_record, series_record, 1)
    elif mode == "equal_nonadjacent":
        _copy_selected_policy_weights_for_scored_test(
            fit_record, series_record, target_seed=1, source_seed=42
        )

    _authenticate_complete_development_pair_for_test(
        fold, fit_record, series_record
    )


@pytest.mark.parametrize("reference", ("selected", "mes"))
def test_round9_rejects_safe_static_reference_metric_mismatch(
    reference: str,
) -> None:
    fold = _canonical_development_gate_folds()[0]
    result = _complete_development_result_for_test(fold)
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, result
    )
    selected_identity, mes_identity = _select_non_mes_static_anchor_for_scored_test(
        fit_record, series_record
    )
    identity = selected_identity if reference == "selected" else mes_identity
    first_series = result["scored_series"][0]
    series_record["result"]["safe_static"][identity]["series"][first_series][
        "csd"
    ] = 9.0
    _rehash_development_pair(fit_record, series_record)

    with pytest.raises(RuntimeError, match="static|anchor|MES|reference|metric"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


def test_round9_rejects_nonselected_nonadjacent_safe_static_policy_twin() -> None:
    fold = _canonical_development_gate_folds()[0]
    result = _complete_development_result_for_test(fold)
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, result
    )
    age_identity = next(
        row["anchor_sha256"]
        for row in fit_record["fit"]["static"]["candidates"]
        if row["anchor"]["family"] == "age" and row["safe"]
    )
    first_series = result["scored_series"][0]
    series_record["result"]["safe_static"][age_identity]["series"][first_series][
        "exclusion"
    ] = 0.2
    _rehash_development_pair(fit_record, series_record)

    with pytest.raises(RuntimeError, match="static|twin|policy|metric"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


def test_round9_accepts_complete_non_mes_selected_static_relationships() -> None:
    fold = _canonical_development_gate_folds()[0]
    result = _complete_development_result_for_test(fold)
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, result
    )
    selected_identity, mes_identity = _select_non_mes_static_anchor_for_scored_test(
        fit_record, series_record
    )

    assert selected_identity != mes_identity
    _authenticate_complete_development_pair_for_test(
        fold, fit_record, series_record
    )


def test_round9_safe_static_twin_grouping_ignores_identity_order() -> None:
    fold = _canonical_development_gate_folds()[0]
    result = _complete_development_result_for_test(fold)
    result["safe_static"] = dict(reversed(tuple(result["safe_static"].items())))
    fit_record, series_record = _complete_development_artifact_pair_for_test(
        fold, result
    )

    _authenticate_complete_development_pair_for_test(
        fold, fit_record, series_record
    )


def test_round9_fresh_writer_validates_result_with_authenticated_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = _canonical_development_gate_folds()[0]
    caller_fit, _ = _complete_development_fold_fit_for_test(fold)
    authenticated_fit = pickle.loads(pickle.dumps(caller_fit))
    result = _complete_development_result_for_test(fold)
    events = []

    monkeypatch.setattr(
        residual_train_module,
        "_fit_development_training",
        lambda *args, **kwargs: caller_fit,
    )
    monkeypatch.setattr(
        residual_train_module,
        "_evaluate_development_scored",
        lambda *args, **kwargs: copy.deepcopy(result),
    )

    def authenticate_fit(*args, **kwargs):
        events.append(("fit", authenticated_fit))
        return authenticated_fit

    def authenticate_result(payload, inner_fold, inventory, fit=None):
        del inner_fold, inventory
        events.append(("result", fit))
        return copy.deepcopy(payload)

    monkeypatch.setattr(
        residual_train_module,
        "_validate_development_fit_payload",
        authenticate_fit,
    )
    monkeypatch.setattr(
        residual_train_module,
        "_validate_development_fold_result",
        authenticate_result,
    )

    residual_train_module._fit_and_evaluate_development_fold(
        fold,
        1,
        root=tmp_path,
        smoke=True,
        expected_anchor_config=anchor_search_config(),
        expected_residual_config=residual_search_config(),
    )

    assert events == [
        ("fit", authenticated_fit),
        ("result", authenticated_fit),
    ]


def test_round9_writer_generated_synthetic_fold_authenticates_without_shortcut(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fold = _canonical_development_gate_folds()[0]
    fit, _ = _complete_development_fold_fit_for_test(fold)
    monkeypatch.setattr(
        residual_train_module,
        "_fit_development_training",
        lambda *args, **kwargs: fit,
    )

    result = residual_train_module._fit_and_evaluate_development_fold(
        fold,
        1,
        root=tmp_path,
        smoke=True,
        expected_anchor_config=anchor_search_config(),
        expected_residual_config=residual_search_config(),
    )
    fold_root = tmp_path / "folds" / fold.name
    fit_record = json.loads((fold_root / "fit.json").read_text(encoding="utf-8"))
    series_record = json.loads(
        (fold_root / "per_series.json").read_text(encoding="utf-8")
    )

    authenticated, _ = residual_train_module._authenticate_development_artifact_pair(
        fold,
        fit_record,
        series_record,
        anchor_search_config(),
        residual_search_config(),
    )
    assert authenticated == result


def _round10_all_zero_fold_fit_for_test(fold: DevelopmentFold) -> FoldFit:
    fit, _ = _complete_development_fold_fit_for_test(fold)
    seed_fits = []
    for retained in fit.seeds:
        zero = next(
            candidate
            for candidate in retained.archive
            if candidate.amplitude == 0.0
        )
        seed_fits.append(
            ResidualSeedFit(
                retained.seed,
                zero,
                (zero,),
                retained.optimizer_history,
            )
        )
    return FoldFit(fit.anchor_fit, fit.scaler, tuple(seed_fits), 1, fit.config)


def _round10_artifact_pair_for_fit(
    fold: DevelopmentFold,
    result: dict[str, object],
    fit: FoldFit,
) -> tuple[dict[str, object], dict[str, object]]:
    expected_safe_static = _development_safe_static_inventory_for_test()
    fold_input = residual_train_module._fold_input_payload(fold)
    common = {
        "schema_version": 2,
        "fold": fold.name,
        "kind": fold.kind,
        "source_manifest_sha256": fold.source_manifest_sha256,
        "provenance": dict(fold.provenance),
        "fold_input": fold_input,
        "fold_input_sha256": residual_train_module._canonical_sha256(fold_input),
        "primary_seed": fit.primary_seed,
        "expected_safe_static": expected_safe_static,
    }
    fit_record = residual_train_module._with_payload_sha(
        {**common, "fit": residual_train_module._fit_development_payload(fit)}
    )
    series_record = residual_train_module._with_payload_sha(
        {
            **common,
            "fit_payload_sha256": fit_record["payload_sha256"],
            "result": result,
        }
    )
    return fit_record, series_record


def _round10_result_for_fit(
    fold: DevelopmentFold,
    fit: FoldFit,
    actuated_series: set[str],
) -> dict[str, object]:
    result = _complete_development_result_for_test(fold)
    result["primary_seed"] = fit.primary_seed
    for row in result["series_results"]:
        row["residual"] = copy.deepcopy(row["anchor"])
    _attach_exact_actuation_for_test(fold, result, actuated_series)
    return result


@pytest.mark.parametrize("actuated_series_index", (0, 1))
def test_round10_rejects_complete_zero_primary_pair_with_retained_actuation(
    actuated_series_index: int,
) -> None:
    fold = _canonical_development_gate_folds()[0]
    fit = _round10_all_zero_fold_fit_for_test(fold)
    scored = tuple(row.ref.key for row in fold.test)
    result = _round10_result_for_fit(
        fold, fit, {scored[actuated_series_index]}
    )
    fit_record, series_record = _round10_artifact_pair_for_fit(
        fold, result, fit
    )

    with pytest.raises(RuntimeError, match="canonical.zero|actuation|actuated"):
        _authenticate_complete_development_pair_for_test(
            fold, fit_record, series_record
        )


def test_round10_rejects_complete_eight_fold_all_zero_false_actuation_gate() -> None:
    folds = _canonical_development_gate_folds()
    fit_records = []
    series_records = []
    for fold in folds:
        fit = _round10_all_zero_fold_fit_for_test(fold)
        actuated = {row.ref.key for row in fold.test}
        result = _round10_result_for_fit(fold, fit, actuated)
        fit_record, series_record = _round10_artifact_pair_for_fit(
            fold, result, fit
        )
        fit_records.append(fit_record)
        series_records.append(series_record)

    with pytest.raises(RuntimeError, match="canonical.zero|actuation|actuated"):
        evidence = residual_train_module.DevelopmentGateEvidence(
            folds, fit_records, series_records
        )
        decision = evaluate_development_gate(evidence)
        pytest.fail(
            "accepted impossible all-zero actuation evidence: "
            f"coverage={decision['conditions']['actuation_coverage']} "
            f"count={decision['observed']['actuated_series']} "
            f"cities={decision['observed']['actuated_cities']}"
        )


def test_round10_accepts_complete_zero_primary_pair_with_all_explicit_inert_rows() -> None:
    fold = _canonical_development_gate_folds()[0]
    fit = _round10_all_zero_fold_fit_for_test(fold)
    result = _round10_result_for_fit(fold, fit, set())
    fit_record, series_record = _round10_artifact_pair_for_fit(
        fold, result, fit
    )

    _authenticate_complete_development_pair_for_test(
        fold, fit_record, series_record
    )


def test_round10_accepts_authenticated_nonzero_primary_with_retained_actuation() -> None:
    fold = _canonical_development_gate_folds()[0]
    fit, _ = _complete_development_fold_fit_for_test(fold)
    first_series = fold.test[0].ref.key
    result = _complete_development_result_for_test(fold)
    target = next(
        row
        for row in result["series_results"]
        if row["seed"] == fit.primary_seed and row["series"] == first_series
    )
    target["residual"]["csd"] = 0.5
    _attach_exact_actuation_for_test(fold, result, {first_series})
    fit_record, series_record = _round10_artifact_pair_for_fit(
        fold, result, fit
    )

    _authenticate_complete_development_pair_for_test(
        fold, fit_record, series_record
    )


@pytest.mark.parametrize(
    "seed_order",
    ((1, 2, 42), (2, 1, 42), (2, 42, 1)),
)
def test_round10_zero_primary_check_is_independent_of_seed_position_and_mapping_order(
    seed_order: tuple[int, ...],
) -> None:
    fold = _canonical_development_gate_folds()[0]
    base = _round10_all_zero_fold_fit_for_test(fold)
    by_seed = {row.seed: row for row in base.seeds}
    config = replace(base.config, seeds=seed_order)
    fit = FoldFit(
        base.anchor_fit,
        base.scaler,
        tuple(by_seed[seed] for seed in seed_order),
        1,
        config,
    )
    result = _round10_result_for_fit(fold, fit, set())
    rows = tuple(
        residual_train_module._coerce_development_result(row)
        for row in result["series_results"]
    )
    safe_static = dict(reversed(tuple(result["safe_static"].items())))
    actuated_by_series = dict(
        reversed(
            tuple(
                (series, index == 0)
                for index, series in enumerate(result["scored_series"])
            )
        )
    )
    assert tuple(row.seed for row in fit.seeds).index(fit.primary_seed) == (
        seed_order.index(1)
    )
    assert all(
        residual_train_module._is_canonical_zero(row.selected)
        for row in fit.seeds
    )

    with pytest.raises(RuntimeError, match="canonical.zero|actuation|actuated"):
        residual_train_module._authenticate_development_scored_relationships(
            rows, safe_static, fit, actuated_by_series
        )


def test_round10_writer_generated_all_zero_fit_has_only_inert_scans_and_signatures() -> None:
    fold = _canonical_development_gate_folds()[0]
    fit = _round10_all_zero_fold_fit_for_test(fold)

    result = residual_train_module._evaluate_development_scored(fold, fit, 1)

    assert result["actuation"]
    assert all(record["actuated"] is False for record in result["actuation"])
    assert all(
        row["signatures"]["actuated"] is False
        for row in result["series_results"]
    )
