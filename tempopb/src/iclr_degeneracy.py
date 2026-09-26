"""Representative support can make CSD perfect while coverage vanishes."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, Tuple

from cohorts import cumulative_share_deficit, group_outcome
from parse_pb import PBInstance, Project, Vote
from run_experiments import exclusion_rate


COHORTS: Tuple[Tuple[int, str], ...] = (
    (20, "M"),
    (20, "F"),
    (30, "M"),
    (30, "F"),
    (50, "M"),
    (50, "F"),
    (70, "M"),
    (70, "F"),
)
RESULT_PATH = Path(__file__).resolve().parent.parent / "results" / "iclr_degeneracy.csv"


def representation_csd(
    group_shares: Mapping[str, float],
    project_group_shares: Sequence[Mapping[str, float]],
    cost_weights: Sequence[float],
) -> Dict[str, float]:
    """Evaluate CSD from funded-project representation ratios.

    ``cost_weights`` are funded costs divided by total funded cost. Each
    ``project_group_shares`` row gives the group composition of approvers for
    one funded project in the same order.
    """
    if not group_shares or any(share <= 0 for share in group_shares.values()):
        raise ValueError("group shares must be positive")
    if len(project_group_shares) != len(cost_weights) or not cost_weights:
        raise ValueError("project shares and cost weights must have equal nonzero length")
    if not math.isclose(sum(group_shares.values()), 1.0, abs_tol=1e-12):
        raise ValueError("group shares must sum to one")
    if any(weight < 0 for weight in cost_weights) or not math.isclose(
        sum(cost_weights), 1.0, abs_tol=1e-12
    ):
        raise ValueError("cost weights must be nonnegative and sum to one")
    groups = set(group_shares)
    for row in project_group_shares:
        if set(row) != groups or any(share < 0 for share in row.values()):
            raise ValueError("every project must provide nonnegative shares for every group")
        if not math.isclose(sum(row.values()), 1.0, abs_tol=1e-12):
            raise ValueError("each project's group shares must sum to one")

    return {
        group: 1.0
        - sum(
            weight * project_group_shares[index][group] / electorate_share
            for index, weight in enumerate(cost_weights)
        )
        for group, electorate_share in group_shares.items()
    }


def build_representative_support_instance(
    n_per_group: int,
    supporters_per_group: int,
) -> Tuple[PBInstance, Set[str]]:
    """Construct one representative funded project and excluded voters."""
    if n_per_group <= 0:
        raise ValueError("n_per_group must be positive")
    if not 0 < supporters_per_group <= n_per_group:
        raise ValueError("supporters_per_group must lie in [1, n_per_group]")

    budget = 100.0
    projects = {
        "representative": Project("representative", budget, None),
        # The dummy is individually feasible.  It remains unfunded because the
        # constructed winner already consumes the complete budget; the theorem
        # does not rely on admitting an invalid over-budget proposal.
        "dummy": Project("dummy", budget, None),
    }
    votes = []
    for group_index, (age, sex) in enumerate(COHORTS):
        for voter_index in range(n_per_group):
            supports_winner = voter_index < supporters_per_group
            votes.append(
                Vote(
                    vid=f"g{group_index}-v{voter_index}",
                    projects=("representative",) if supports_winner else ("dummy",),
                    age=age,
                    sex=sex,
                )
            )
    instance = PBInstance(
        path="representative-support-construction",
        meta={"budget": str(budget), "vote_type": "approval"},
        projects=projects,
        votes=votes,
    )
    return instance, {"representative"}


def evaluate_construction(n_per_group: int, supporters_per_group: int) -> Dict[str, float]:
    """Score the construction through the paper's production metric ledger."""
    instance, winners = build_representative_support_instance(
        n_per_group=n_per_group,
        supporters_per_group=supporters_per_group,
    )
    outcome = group_outcome(instance, winners, scheme="age_sex")
    deficits = [
        cumulative_share_deficit([outcome], cohort) for cohort in sorted(outcome)
    ]
    if not deficits or any(value is None for value in deficits):
        raise RuntimeError("construction produced an incomplete cohort ledger")
    return {
        "groups": len(outcome),
        "voters": len(instance.votes),
        "supporters": len(COHORTS) * supporters_per_group,
        "worst_abs_csd": max(abs(float(value)) for value in deficits),
        "exclusion": exclusion_rate(instance, winners),
    }


def construction_rows(group_sizes: Sequence[int]) -> List[Dict[str, float]]:
    """Evaluate a fixed eight-supporter construction as the electorate grows."""
    return [
        evaluate_construction(n_per_group=size, supporters_per_group=1)
        for size in group_sizes
    ]


def main() -> None:
    rows = construction_rows([10, 25, 50, 100])
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = ["groups", "voters", "supporters", "worst_abs_csd", "exclusion"]
    with RESULT_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {RESULT_PATH} ({len(rows)} constructions)")


if __name__ == "__main__":
    main()
