from collections.abc import Set

from cohorts import group_outcome
from parse_pb import PBInstance, Project, Vote


class OrderedSet(Set[str]):
    """Set semantics with an explicit iteration order for the regression."""

    def __init__(self, values):
        self._values = tuple(values)
        self._members = frozenset(values)

    def __contains__(self, value):
        return value in self._members

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._members)


def test_group_outcome_is_exactly_independent_of_winner_iteration_order():
    """Catches hash-order noise that can reorder tied optimizer candidates."""
    instance = PBInstance(
        path="test://cohort-order",
        meta={"budget": "10000000000000002"},
        projects={
            "large": Project("large", 1e16, None),
            "small-a": Project("small-a", 1.0, None),
            "small-b": Project("small-b", 1.0, None),
        },
        votes=[
            Vote("f", ("large", "small-a", "small-b"), age=70, sex="F"),
            Vote("m", ("large", "small-a", "small-b"), age=70, sex="M"),
        ],
    )

    forward = group_outcome(
        instance,
        OrderedSet(("large", "small-a", "small-b")),
        "age_sex",
    )
    reverse = group_outcome(
        instance,
        OrderedSet(("small-b", "small-a", "large")),
        "age_sex",
    )

    assert forward == reverse
