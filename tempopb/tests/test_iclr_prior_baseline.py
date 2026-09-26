"""Regression tests for published learned-rule baselines."""

from __future__ import annotations

from iclr_env import RolloutState
from iclr_outcome import llmrule_card_selector, llmrule_cost_selector
from parse_pb import PBInstance, Project, Vote


def test_llmrule_cost_selector_uses_published_nonlinear_score() -> None:
    """The published LLMRule score can prefer a cheaper, less popular project."""
    inst = PBInstance(
        path="test://llmrule-cost",
        meta={"budget": "10"},
        projects={
            "expensive": Project("expensive", 10.0, None),
            "balanced": Project("balanced", 4.0, None),
        },
        votes=[
            Vote(str(i), ("expensive", "balanced") if i < 5 else ("expensive",))
            for i in range(9)
        ]
        + [Vote("9", ())],
    )

    # sqrt(0.5)/(1 + 0.4) > sqrt(0.9)/(1 + 1.0), so the 4-unit project
    # is selected first and the 10-unit project no longer fits.
    assert llmrule_cost_selector()(inst, RolloutState()) == {"balanced"}


def test_llmrule_cost_selector_ignores_projects_without_support() -> None:
    inst = PBInstance(
        path="test://llmrule-zero-support",
        meta={"budget": "5"},
        projects={
            "supported": Project("supported", 5.0, None),
            "unsupported": Project("unsupported", 1.0, None),
        },
        votes=[Vote("0", ("supported",))],
    )

    assert llmrule_cost_selector()(inst, RolloutState()) == {"supported"}


def test_llmrule_card_selector_uses_published_cardinality_score() -> None:
    """The cardinality-satisfaction rule includes the published log factor."""
    inst = PBInstance(
        path="test://llmrule-card",
        meta={"budget": "10"},
        projects={
            "expensive": Project("expensive", 10.0, None),
            "balanced": Project("balanced", 4.0, None),
        },
        votes=[
            Vote(str(i), ("expensive", "balanced") if i < 6 else ("expensive",))
            for i in range(9)
        ]
        + [Vote("9", ())],
    )

    # 0.6/0.4*log(1.6) > 0.9/1.0*log(1.9), so the cheaper project wins.
    assert llmrule_card_selector()(inst, RolloutState()) == {"balanced"}
