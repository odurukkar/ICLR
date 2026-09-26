"""Learned project utilities inside the Equal Shares payment kernel."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from iclr_env import RolloutState
from iclr_outcome import _score_vector
from iclr_priority_mes import (
    _minimum_affordable_price_numpy,
    fast_mes_with_endowments,
    priority_log_multipliers,
    priority_mes_outcome,
    priority_mes_selector,
    verify_priority_mes_corpus,
)
from iclr_corpus import SeriesRef
from parse_pb import PBInstance, Project, Vote
from rules import mes_with_endowments


def _competing_projects() -> PBInstance:
    """The first purchase makes the other project unaffordable."""

    return PBInstance(
        path="test://priority-mes/competing",
        meta={"budget": "10"},
        projects={
            "a": Project("a", 4.0, None),
            "b": Project("b", 4.0, None),
        },
        votes=[
            Vote(str(i), ("a", "b") if i < 5 else (("b",) if i == 5 else ()))
            for i in range(10)
        ],
    )


def _tie_instance(reverse: bool = False) -> PBInstance:
    projects = [
        ("a", Project("a", 5.0, None)),
        ("b", Project("b", 5.0, None)),
    ]
    if reverse:
        projects.reverse()
    return PBInstance(
        path=f"test://priority-mes/tie/{reverse}",
        meta={"budget": "5"},
        projects=dict(projects),
        votes=[Vote(str(i), ("a", "b")) for i in range(5)],
    )


def test_priority_rule_rejects_wrong_shape_and_nonfinite_weights() -> None:
    with pytest.raises(ValueError, match="expected 5 weights"):
        priority_mes_selector(np.zeros(4))
    with pytest.raises(ValueError, match="finite"):
        priority_mes_selector([0.0, 0.0, math.nan, 0.0, 0.0])


@pytest.mark.parametrize("n_supporters", (1, 2, 10, 100))
def test_numpy_affordability_matches_reference_arithmetic(n_supporters: int) -> None:
    from iclr_outcome import _minimum_affordable_price

    rng = np.random.default_rng(100 + n_supporters)
    balances = rng.uniform(0.0, 2.0, size=n_supporters)
    supporters = np.arange(n_supporters, dtype=np.int64)
    for share in (0.01, 0.25, 0.75, 1.0, 1.01):
        cost = float(balances.sum() * share)
        expected = _minimum_affordable_price(
            cost, supporters.tolist(), balances.tolist()
        )
        observed = _minimum_affordable_price_numpy(cost, supporters, balances)
        if expected is None:
            assert observed is None
        else:
            assert observed == pytest.approx(expected, abs=1e-12)


def test_log_multipliers_are_the_clipped_canonical_project_scores() -> None:
    inst = _competing_projects()
    state = RolloutState()
    weights = np.array([100.0, -100.0, 100.0, 0.0, 100.0])

    feats, observed = priority_log_multipliers(inst, state, weights)
    expected_feats, raw = _score_vector(inst, state, weights, "age_sex")

    assert feats.project_ids == expected_feats.project_ids
    assert np.array_equal(observed, np.clip(raw, -10.0, 10.0))


@pytest.mark.parametrize("completion", (False, True))
@pytest.mark.parametrize("reverse", (False, True))
def test_zero_weights_reproduce_standard_mes_exactly(
    completion: bool, reverse: bool
) -> None:
    inst = _tie_instance(reverse)

    expected = mes_with_endowments(inst, completion=completion)
    observed = priority_mes_selector(np.zeros(5), completion=completion)(
        inst, RolloutState()
    )

    assert observed == expected == {"a"}


def test_nonzero_utility_can_reverse_the_mes_affordability_priority() -> None:
    inst = _competing_projects()
    standard = priority_mes_selector(np.zeros(5), completion=False)(
        inst, RolloutState()
    )
    learned = priority_mes_selector([-10.0, 0.0, 0.0, 0.0, 0.0], completion=False)(
        inst, RolloutState()
    )

    assert standard == {"b"}
    assert learned == {"a"}


@pytest.mark.parametrize("completion", (False, True))
@pytest.mark.parametrize(
    "endowments",
    (
        [1.0] * 10,
        [2.0] * 5 + [0.0] * 5,
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.9, 4.5],
    ),
)
def test_fast_nonuniform_endowment_path_matches_reference_mes(
    completion: bool, endowments
) -> None:
    inst = _competing_projects()
    assert fast_mes_with_endowments(inst, endowments, completion=completion) == (
        mes_with_endowments(inst, endowments=endowments, completion=completion)
    )


def test_fast_endowment_path_rejects_invalid_balances() -> None:
    inst = _competing_projects()
    with pytest.raises(ValueError, match="expected 10 endowments"):
        fast_mes_with_endowments(inst, [1.0] * 9)
    with pytest.raises(ValueError, match="non-negative"):
        fast_mes_with_endowments(inst, [1.0] * 9 + [-1.0])
    with pytest.raises(ValueError, match="sum to the municipal budget"):
        fast_mes_with_endowments(inst, [2.0] * 10)


def test_payment_trace_charges_standard_equalized_price() -> None:
    inst = _competing_projects()

    outcome = priority_mes_outcome(
        inst,
        RolloutState(),
        [-10.0, 0.0, 0.0, 0.0, 0.0],
        completion=False,
    )

    assert outcome.payment_winners == ("a",)
    assert outcome.completion_winners == ()
    assert len(outcome.rounds) == 1
    round_ = outcome.rounds[0]
    assert round_.rho == pytest.approx(0.8)
    assert sum(round_.payments) == pytest.approx(4.0)
    assert all(payment == pytest.approx(min(1.0, round_.rho)) for payment in round_.payments)
    assert sum(inst.projects[pid].cost for pid in outcome.winners) <= inst.budget


@pytest.mark.parametrize(
    "inst",
    (
        PBInstance(path="test://empty", meta={"budget": "10"}),
        PBInstance(
            path="test://zero-budget",
            meta={"budget": "0"},
            projects={"a": Project("a", 1.0, None)},
            votes=[Vote("0", ("a",))],
        ),
    ),
)
def test_empty_or_zero_budget_election_returns_empty(inst: PBInstance) -> None:
    outcome = priority_mes_outcome(
        inst, RolloutState(), np.zeros(5), completion=True
    )
    assert outcome.winners == frozenset()
    assert outcome.rounds == ()


def test_structural_gate_counts_both_completion_modes(monkeypatch) -> None:
    inst = _competing_projects()
    ref = SeriesRef(
        key="Poland/Test/Unit",
        years=(2022,),
        paths=(Path("unused.pb"),),
    )
    monkeypatch.setattr(
        "iclr_priority_mes.load_series", lambda observed_ref: {2022: inst}
    )

    report = verify_priority_mes_corpus((ref,))

    assert report["n_series"] == 1
    assert report["n_elections"] == 1
    assert report["identity_checks"] == 2
    assert report["determinism_checks"] == 2
    assert report["budget_checks"] == 2
    assert report["status"] == "pass"
