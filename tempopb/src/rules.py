"""Voting rules for counterfactual replay: greedy, MES, and Rollover Equal
Shares (RES).

All rules take a PBInstance with approval ballots and return a winner set.
MES is implemented directly (approval satisfaction, per-voter endowments)
so that RES can perturb endowments; pabutools serves as an external
cross-check in tests, not as the production implementation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from cohorts import cohort_of
from parse_pb import PBInstance

logger = logging.getLogger(__name__)


def greedy_by_votes(inst: PBInstance) -> Set[str]:
    """The rule most cities deploy: sort by approval count, fund if it fits."""
    counts: Dict[str, int] = {pid: 0 for pid in inst.projects}
    for v in inst.votes:
        for p in v.projects:
            if p in counts:
                counts[p] += 1
    remaining = inst.budget
    winners: Set[str] = set()
    for pid in sorted(
        counts, key=lambda p: (-counts[p], inst.projects[p].cost, p)
    ):
        cost = inst.projects[pid].cost
        if cost <= remaining and counts[pid] > 0:
            winners.add(pid)
            remaining -= cost
    return winners


def greedy_by_approvals_per_cost(inst: PBInstance) -> Set[str]:
    """Independently rank projects by approval count per unit cost.

    This is the reference implementation used to verify that the direct score
    family contains the cost-effectiveness baseline at ``v = e_2``. Ties use
    ascending cost and project id, matching the score selector's documented
    deterministic order without sharing its feature code.
    """
    counts: Dict[str, int] = {pid: 0 for pid in inst.projects}
    for vote in inst.votes:
        for pid in vote.projects:
            if pid in counts:
                counts[pid] += 1

    def priority(pid: str) -> tuple[float, float, str]:
        cost = float(inst.projects[pid].cost)
        ratio = counts[pid] / cost if cost > 0 else 0.0
        return (-ratio, cost, pid)

    remaining = float(inst.budget)
    winners: Set[str] = set()
    for pid in sorted(counts, key=priority):
        cost = float(inst.projects[pid].cost)
        if counts[pid] > 0 and cost <= remaining:
            winners.add(pid)
            remaining -= cost
    return winners


def mes_with_endowments(
    inst: PBInstance,
    endowments: Optional[List[float]] = None,
    completion: bool = True,
) -> Set[str]:
    """Method of Equal Shares, approval satisfaction, arbitrary endowments.

    A project p is rho-affordable if sum_{i approves p} min(b_i, rho) >= cost(p);
    the rule repeatedly buys the project with minimal rho, charging each
    approver min(b_i, rho). With equal endowments B/n this is standard MES.
    Optional completion: fill leftover budget greedily by approval count
    (documented in the paper; MES alone is not exhaustive).
    """
    n = len(inst.votes)
    if n == 0:
        return set()
    budgets = list(endowments) if endowments is not None else [inst.budget / n] * n

    approver_ids: Dict[str, List[int]] = {pid: [] for pid in inst.projects}
    for i, v in enumerate(inst.votes):
        for p in v.projects:
            if p in approver_ids:
                approver_ids[p].append(i)

    active = {
        pid for pid, sup in approver_ids.items()
        if sup and inst.projects[pid].cost > 0
    }
    winners: Set[str] = set()

    while active:
        best_pid, best_rho, best_key = None, None, None
        for pid in list(active):
            cost = inst.projects[pid].cost
            sup = approver_ids[pid]
            total_available = sum(budgets[i] for i in sup)
            if total_available < cost - 1e-9:
                active.discard(pid)  # can never afford: budgets only shrink
                continue
            # find minimal rho: approvers pay min(b_i, rho)
            bs = sorted((budgets[i] for i in sup))
            k = len(bs)
            paid_so_far = 0.0
            rho = None
            for j, b in enumerate(bs):
                # voters j..k-1 each would pay rho if rho <= their budget
                remaining_payers = k - j
                candidate = (cost - paid_so_far) / remaining_payers
                if candidate <= b + 1e-12:
                    rho = candidate
                    break
                paid_so_far += b
            if rho is None:
                active.discard(pid)
                continue
            # A set has hash-dependent iteration order.  Make exact-rho ties
            # reproducible with project cost and canonical project identifier.
            candidate_key = (rho, inst.projects[pid].cost, pid)
            if best_key is None or candidate_key < best_key:
                best_pid, best_rho, best_key = pid, rho, candidate_key
        if best_pid is None:
            break
        sup = approver_ids[best_pid]
        for i in sup:
            budgets[i] -= min(budgets[i], best_rho)
        winners.add(best_pid)
        active.discard(best_pid)

    if completion:
        counts = {pid: len(approver_ids[pid]) for pid in inst.projects}
        spent = sum(inst.projects[p].cost for p in winners)
        remaining = inst.budget - spent
        for pid in sorted(
            counts, key=lambda p: (-counts[p], inst.projects[p].cost, p)
        ):
            if pid in winners or counts[pid] == 0:
                continue
            cost = inst.projects[pid].cost
            if 0 < cost <= remaining:
                winners.add(pid)
                remaining -= cost
    return winners


def res_endowments(
    inst: PBInstance,
    group_deficits: Dict[str, float],
    lam: float,
    scheme: str = "age_sex",
) -> List[float]:
    """Rollover endowments: base share plus lambda x per-capita group deficit,
    rescaled so the total equals the year's budget (never overspends).
    Voters without cohort metadata get the base endowment.
    """
    n = len(inst.votes)
    base = inst.budget / n
    cohort_sizes: Dict[str, int] = {}
    voter_cohorts: List[Optional[str]] = []
    for v in inst.votes:
        c = cohort_of(v, scheme)
        voter_cohorts.append(c)
        if c is not None:
            cohort_sizes[c] = cohort_sizes.get(c, 0) + 1

    raw = []
    for c in voter_cohorts:
        boost = 0.0
        if c is not None and c in group_deficits and cohort_sizes.get(c):
            boost = lam * max(0.0, group_deficits[c]) / cohort_sizes[c]
        raw.append(base + boost)
    total = sum(raw)
    scale = inst.budget / total if total > 0 else 1.0
    return [b * scale for b in raw]
