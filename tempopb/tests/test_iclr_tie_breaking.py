"""Regression tests for the deterministic tie order stated in the paper."""

from parse_pb import PBInstance, Project, Vote
from rules import greedy_by_votes, mes_with_endowments


def _reverse_insertion_tie() -> PBInstance:
    return PBInstance(
        path="test://reverse-insertion-tie",
        meta={"budget": "10"},
        projects={
            "z": Project("z", 6.0, None),
            "a": Project("a", 6.0, None),
        },
        votes=[Vote("0", ("z",)), Vote("1", ("a",))],
    )


def test_greedy_count_uses_project_id_after_count_and_cost_ties() -> None:
    assert greedy_by_votes(_reverse_insertion_tie()) == {"a"}


def test_equal_shares_completion_uses_project_id_after_count_and_cost_ties() -> None:
    # Each supporter holds 5, so neither cost-6 project is payment-affordable;
    # completion must resolve the tied projects independently of dict order.
    assert mes_with_endowments(_reverse_insertion_tie(), completion=True) == {"a"}
