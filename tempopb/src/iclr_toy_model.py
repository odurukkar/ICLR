"""Executable counterexample to a discarded reachable-width formula.

An earlier draft capped one designated group at ``kappa`` and mistook the
resulting width for the full attainable range over all feasible endowment
vectors.  For unequal group sizes, capping the other group can produce a much
larger deficit.  The paper no longer states that theorem; this script preserves
the falsification that caused its removal.

The construction uses the production ``mes_with_endowments`` implementation.
It writes ``results/iclr_reachability_counterexample.csv`` so the negative
result remains executable and cannot be silently reintroduced.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from parse_pb import PBInstance, Project, Vote
from rules import mes_with_endowments

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "iclr_reachability_counterexample.csv"


def discarded_one_orientation_formula(alpha: float, kappa: float) -> float:
    """Value claimed by the discarded formula when group A is capped."""
    if kappa > 1.0 / alpha:
        kappa = 1.0 / alpha
    return alpha * (kappa - 1.0) / (1.0 - alpha)


def build_instance(
    n: int, alpha: float, n_proj: int, budget: float
) -> Tuple[PBInstance, List[str]]:
    """Separable two-group instance with equal-cost group-exclusive projects."""
    n_a = int(round(alpha * n))
    cost = budget / n_proj
    projects: Dict[str, Project] = {}
    for k in range(n_proj):
        projects[f"a{k}"] = Project(pid=f"a{k}", cost=cost, selected=False)
        projects[f"b{k}"] = Project(pid=f"b{k}", cost=cost, selected=False)
    votes: List[Vote] = []
    groups: List[str] = []
    a_ids = [f"a{k}" for k in range(n_proj)]
    b_ids = [f"b{k}" for k in range(n_proj)]
    for i in range(n):
        in_a = i < n_a
        votes.append(Vote(vid=str(i), projects=tuple(a_ids if in_a else b_ids)))
        groups.append("A" if in_a else "B")
    return (
        PBInstance(
            path=f"reachability-counterexample-alpha-{alpha}",
            meta={"budget": str(budget), "vote_type": "approval"},
            projects=projects,
            votes=votes,
        ),
        groups,
    )


def measured_csd(inst: PBInstance, groups: List[str], winners) -> float:
    """Worst-group CSD for one funded set, using the paper's accounting."""
    spent = sum(inst.projects[p].cost for p in winners)
    if spent <= 0:
        return 0.0
    utility: Dict[str, float] = {"A": 0.0, "B": 0.0}
    for pid in winners:
        approvers = [g for vote, g in zip(inst.votes, groups) if pid in vote.projects]
        if not approvers:
            continue
        share = inst.projects[pid].cost / len(approvers)
        for group in approvers:
            utility[group] += share
    worst = -np.inf
    for group in ("A", "B"):
        size = sum(1 for value in groups if value == group)
        entitlement = spent * size / len(groups)
        if entitlement > 0:
            worst = max(worst, (entitlement - utility[group]) / entitlement)
    return float(worst)


def evaluate_orientation(alpha: float, kappa: float, capped: str) -> float:
    """Measure the separable instance after capping group A or group B."""
    n, n_proj, budget = 800, 80, 8000.0
    inst, groups = build_instance(n=n, alpha=alpha, n_proj=n_proj, budget=budget)
    if capped == "A":
        kappa_a = kappa
        kappa_b = (1.0 - alpha * kappa_a) / (1.0 - alpha)
    elif capped == "B":
        kappa_b = kappa
        kappa_a = (1.0 - (1.0 - alpha) * kappa_b) / alpha
    else:
        raise ValueError(f"unknown capped group: {capped}")
    if min(kappa_a, kappa_b) < 0:
        raise ValueError("orientation is infeasible at this kappa")
    base = budget / n
    endowments = [base * (kappa_a if group == "A" else kappa_b) for group in groups]
    winners = mes_with_endowments(inst, endowments=endowments, completion=False)
    return measured_csd(inst, groups, winners)


def main() -> None:
    alpha, kappa = 1.0 / 8.0, 1.1
    old_value = discarded_one_orientation_formula(alpha, kappa)
    rows = []
    for capped in ("A", "B"):
        measured = evaluate_orientation(alpha, kappa, capped)
        rows.append(
            {
                "alpha": alpha,
                "kappa": kappa,
                "capped_group": capped,
                "discarded_formula": round(old_value, 6),
                "measured_worst_csd": round(measured, 6),
                "ratio_to_discarded_formula": round(measured / old_value, 2),
            }
        )
        logger.info(
            "cap %s: discarded formula %.4f, production mechanism %.4f",
            capped,
            old_value,
            measured,
        )

    if rows[1]["measured_worst_csd"] <= 40 * old_value:
        raise AssertionError("counterexample no longer falsifies the discarded formula")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
