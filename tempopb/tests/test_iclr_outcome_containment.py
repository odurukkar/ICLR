"""Independent containment checks for the direct project-score family."""

from __future__ import annotations

from iclr_env import RolloutState
from iclr_outcome import cost_effective_weights, score_selector
from parse_pb import PBInstance, Project, Vote
from rules import greedy_by_approvals_per_cost


def test_cost_effective_score_matches_independent_greedy_reference() -> None:
    instance = PBInstance(
        path="test://greedy-cost-containment",
        meta={"budget": "6"},
        projects={
            "a": Project("a", 5.0, None),
            "b": Project("b", 2.0, None),
            "c": Project("c", 4.0, None),
        },
        votes=[
            Vote("1", ("a", "b", "c")),
            Vote("2", ("a", "b", "c")),
            Vote("3", ("a", "c")),
            Vote("4", ("a",)),
        ],
    )

    reference = greedy_by_approvals_per_cost(instance)
    contained = score_selector(cost_effective_weights())(instance, RolloutState())

    assert reference == {"b", "c"}
    assert contained == reference
