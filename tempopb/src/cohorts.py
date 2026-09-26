"""Cohort assignment and group-level fairness metrics.

Cohorts are observable voter groups: age bracket x sex where the ballot
metadata provides them. Group utility is money-weighted attribution: each
funded project's cost is attributed to cohorts in proportion to their share
of the project's approvers. Entitlement is the cohort's share of the money
actually spent, so the metric isolates distributional skew from
underspending.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set

from parse_pb import PBInstance, Vote

logger = logging.getLogger(__name__)

AGE_BRACKETS = ((0, 24, "age<25"), (25, 39, "age25-39"), (40, 59, "age40-59"), (60, 200, "age60+"))


def age_bracket(age: Optional[int]) -> Optional[str]:
    if age is None or age <= 0 or age > 110:
        return None
    for lo, hi, label in AGE_BRACKETS:
        if lo <= age <= hi:
            return label
    return None


def sex_label(sex: str) -> Optional[str]:
    # Polish files use M/K (kobieta = female); others use M/F.
    if sex == "M":
        return "M"
    if sex in ("F", "K"):
        return "F"
    return None


def cohort_of(vote: Vote, scheme: str) -> Optional[str]:
    """scheme: 'age', 'sex', or 'age_sex'."""
    a = age_bracket(vote.age)
    s = sex_label(vote.sex)
    if scheme == "age":
        return a
    if scheme == "sex":
        return s
    if scheme == "age_sex":
        if a is None or s is None:
            return None
        return f"{a}|{s}"
    if scheme == "district":
        hood = vote.neighborhood.strip()
        if not hood or hood in {"-", "?", "unknown", "brak"}:
            return None
        return hood
    raise ValueError(f"unknown cohort scheme: {scheme}")


def group_outcome(
    inst: PBInstance,
    winners: Set[str],
    scheme: str = "age_sex",
) -> Dict[str, Dict[str, float]]:
    """Per-cohort spend attribution and entitlement for one election.

    Returns {cohort: {"utility": money attributed, "entitlement": share of
    spend, "voters": cohort size}}. Voters without cohort metadata are
    excluded from both sides of the ledger.
    """
    cohort_members: Dict[str, List[Vote]] = defaultdict(list)
    for v in inst.votes:
        c = cohort_of(v, scheme)
        if c is not None:
            cohort_members[c].append(v)
    n_covered = sum(len(m) for m in cohort_members.values())
    if n_covered == 0:
        return {}

    # Canonicalize floating-point accumulation so outcomes do not depend on
    # set/hash iteration order (which can otherwise perturb optimizer ranks).
    ordered_winners = tuple(sorted(winners))

    # approvers of each winning project, restricted to covered voters
    approvers: Dict[str, Dict[str, int]] = {
        p: defaultdict(int) for p in ordered_winners
    }
    approver_totals: Dict[str, int] = defaultdict(int)
    for c, members in cohort_members.items():
        for v in members:
            for p in v.projects:
                if p in winners:
                    approvers[p][c] += 1
                    approver_totals[p] += 1

    spent = sum(
        inst.projects[p].cost for p in ordered_winners if p in inst.projects
    )
    out: Dict[str, Dict[str, float]] = {}
    for c, members in cohort_members.items():
        utility = 0.0
        for p in ordered_winners:
            tot = approver_totals.get(p, 0)
            if tot > 0 and p in inst.projects:
                utility += inst.projects[p].cost * approvers[p][c] / tot
        out[c] = {
            "utility": utility,
            "entitlement": spent * len(members) / n_covered,
            "voters": float(len(members)),
        }
    return out


def historical_winners(inst: PBInstance) -> Set[str]:
    return {p.pid for p in inst.projects.values() if p.selected == 1}


def cumulative_share_deficit(
    yearly: Iterable[Dict[str, Dict[str, float]]],
    cohort: str,
) -> Optional[float]:
    """CSD over a series: (sum entitlements - sum utilities) / sum entitlements."""
    ent = util = 0.0
    seen = False
    for year_outcome in yearly:
        if cohort in year_outcome:
            ent += year_outcome[cohort]["entitlement"]
            util += year_outcome[cohort]["utility"]
            seen = True
    if not seen or ent <= 0:
        return None
    return (ent - util) / ent
