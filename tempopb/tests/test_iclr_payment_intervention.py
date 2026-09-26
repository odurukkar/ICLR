"""Tests for the frozen-score payment-kernel intervention."""

from __future__ import annotations

import numpy as np
import pytest

from iclr_env import RolloutState
from iclr_outcome import payment_gated_score_selector, score_selector
from parse_pb import PBInstance, Project, Vote


def _high_cost_low_support_instance() -> PBInstance:
    """A direct cost-seeking score buys a project its supporters cannot fund."""
    return PBInstance(
        path="test://payment-gated-score",
        meta={"budget": "10"},
        projects={
            "expensive": Project("expensive", 10.0, None),
            "affordable": Project("affordable", 2.0, None),
        },
        votes=[
            Vote("0", ("expensive", "affordable"), age=20, sex="M"),
            Vote("1", ("expensive", "affordable"), age=20, sex="F"),
        ]
        + [Vote(str(i), ("affordable",), age=20, sex="M") for i in range(2, 8)]
        + [Vote("8", (), age=20, sex="F"), Vote("9", (), age=20, sex="F")],
    )


def test_payment_gate_changes_only_the_allocation_kernel_for_fixed_scores() -> None:
    inst = _high_cost_low_support_instance()
    # Positive cost-share weight ranks the budget-sized project first.
    weights = np.array([0.0, 0.0, 1.0, 0.0, 0.0])

    direct = score_selector(weights)(inst, RolloutState())
    gated = payment_gated_score_selector(weights, completion=False)(
        inst, RolloutState()
    )

    assert direct == {"expensive"}
    assert gated == {"affordable"}


def test_payment_gated_score_is_budget_feasible_with_standard_completion() -> None:
    inst = _high_cost_low_support_instance()
    weights = np.array([0.0, 0.0, 1.0, 0.0, 0.0])

    winners = payment_gated_score_selector(weights, completion=True)(
        inst, RolloutState()
    )

    assert sum(inst.projects[pid].cost for pid in winners) <= inst.budget
    assert winners == {"affordable"}


def test_payment_gated_score_rejects_wrong_weight_shape() -> None:
    with pytest.raises(ValueError, match="expected 5 weights"):
        payment_gated_score_selector(np.zeros(4))
