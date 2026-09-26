"""Outcome-space policies: learn a project score, then fund greedily.

This arm drops the endowment channel. Instead of nudging what voters can pay
and letting Equal Shares decide, it scores projects directly and funds them in
score order subject to the budget. It therefore discards the approver-funded
payment phase retained by the endowment arm, and in exchange it is
not limited by how much leverage endowments have over money attribution.

The two arms together trace the paper's central object: what a rule can buy in
fairness once it stops being required to look like a proportional rule.

As in the endowment arm the family contains its baselines exactly:

    score = v . phi(p),  fund by descending score, ties by ascending cost/id

    v = (1, 0, 0, 0, 0)  ==  greedy by approval count (the deployed rule)
    v = (0, 1, 0, 0, 0)  ==  greedy by approvals per unit cost

The second is the cost-effectiveness baseline used in the audit literature and
prevents an apparent fairness gain from resting on omission of a strong welfare
reference.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from iclr_env import RolloutState, SelectorPolicy
from iclr_policy import instance_features
from parse_pb import PBInstance
from run_experiments import SCHEME

logger = logging.getLogger(__name__)

PROJECT_FEATURES: Tuple[str, ...] = (
    "approval_share",        # n(p)/n  -> v=e_1 is greedy by approval count
    "approvals_per_cost",    # cost-effectiveness -> v=e_2 is the utilitarian greedy
    "cost_share",            # c(p)/b; negative weight prefers cheap projects
    "deficit_weighted",      # share of p's approvers drawn from lagging cohorts
    "cohort_concentration",  # how unevenly p's support spreads across cohorts
)
N_PROJECT_FEATURES = len(PROJECT_FEATURES)

__all__ = [
    "PROJECT_FEATURES",
    "N_PROJECT_FEATURES",
    "ProjectFeatures",
    "project_features",
    "clear_project_cache",
    "score_selector",
    "payment_gated_score_selector",
    "greedy_equivalent_weights",
    "cost_effective_weights",
    "llmrule_cost_selector",
    "llmrule_card_selector",
]

# Keyed by (instance path, cohort scheme) -- see iclr_policy for why.
_CACHE: Dict[Tuple[str, str], "ProjectFeatures"] = {}


@dataclass(frozen=True)
class ProjectFeatures:
    """History-independent per-project features for one election."""

    project_ids: Tuple[str, ...]
    costs: np.ndarray
    static: np.ndarray          # (n_projects, N_PROJECT_FEATURES), history cols zero
    cohort_share: np.ndarray    # (n_projects, n_cohorts) share of approvers by cohort
    cohort_order: Tuple[str, ...]
    has_support: np.ndarray     # bool mask: at least one approver


def clear_project_cache() -> None:
    """Drop cached project features."""
    _CACHE.clear()


def project_features(inst: PBInstance, scheme: str = SCHEME) -> ProjectFeatures:
    """Compute (and cache) per-project features for one election."""
    cached = _CACHE.get((inst.path, scheme))
    if cached is not None:
        return cached

    feats = instance_features(inst, scheme)
    cohort_order = feats.cohort_order
    cohort_index = feats.cohort_index
    n_cohorts = len(cohort_order)

    pids = tuple(sorted(inst.projects))
    pid_index = {pid: i for i, pid in enumerate(pids)}
    n_p = len(pids)

    # Two different counts, deliberately. `counts_all` runs over every ballot,
    # matching what a city's greedy rule actually tallies, and is what the score
    # is built from. `by_cohort` runs only over demographically covered ballots,
    # matching the attribution ledger. Conflating them silently reorders
    # projects wherever coverage is partial.
    counts_all = np.zeros(n_p)
    counts_covered = np.zeros(n_p)
    by_cohort = np.zeros((n_p, n_cohorts))
    n_covered = 0
    for vote, cohort in zip(inst.votes, feats.voter_cohorts):
        covered = cohort is not None
        if covered:
            n_covered += 1
            col = cohort_index[cohort]
        for pid in vote.projects:
            row = pid_index.get(pid)
            if row is None:
                continue
            counts_all[row] += 1
            if covered:
                counts_covered[row] += 1
                by_cohort[row, col] += 1

    costs = np.array([inst.projects[pid].cost for pid in pids], dtype=float)
    budget = float(inst.budget) or 1.0
    n_all = float(len(inst.votes)) or 1.0

    approval_share = counts_all / n_all
    cost_share = costs / budget
    with np.errstate(divide="ignore", invalid="ignore"):
        per_cost = np.where(cost_share > 0, approval_share / cost_share, 0.0)
        shares = np.where(
            counts_covered[:, None] > 0, by_cohort / counts_covered[:, None], 0.0
        )
    # Normalize cost-effectiveness so its scale is comparable to approval_share.
    scale = per_cost.max() or 1.0
    per_cost_norm = per_cost / scale

    # Concentration: dispersion of a project's support across cohorts, 0 when the
    # support mirrors an equal split and 1 when it sits entirely in one cohort.
    if n_cohorts > 1:
        uniform = 1.0 / n_cohorts
        concentration = np.sqrt(
            ((shares - uniform) ** 2).sum(axis=1) / (1.0 - uniform)
        )
    else:
        concentration = np.zeros(n_p)

    static = np.zeros((n_p, N_PROJECT_FEATURES))
    static[:, 0] = approval_share
    static[:, 1] = per_cost_norm
    static[:, 2] = cost_share
    static[:, 4] = concentration

    result = ProjectFeatures(
        project_ids=pids,
        costs=costs,
        static=static,
        cohort_share=shares,
        cohort_order=cohort_order,
        has_support=counts_all > 0,
    )
    _CACHE[(inst.path, scheme)] = result
    return result


def greedy_equivalent_weights() -> np.ndarray:
    """Weights reproducing the deployed rule: greedy by approval count."""
    weights = np.zeros(N_PROJECT_FEATURES)
    weights[0] = 1.0
    return weights


def cost_effective_weights() -> np.ndarray:
    """Weights reproducing greedy by approvals per unit cost."""
    weights = np.zeros(N_PROJECT_FEATURES)
    weights[1] = 1.0
    return weights


def llmrule_cost_selector(scheme: str = SCHEME) -> SelectorPolicy:
    """Reproduce the published LLMRule example for cost satisfaction.

    Thach, Sha, and Chan (AAMAS 2026, DOI 10.65109/LGEP3560) report the
    learned priority score

        sqrt(approval_share) / (1 + cost_share).

    This frozen external rule is not refitted to CSD or to our corpus.  It is
    included to test the closest published automated-PB-rule baseline under
    the same longitudinal ledger as every other policy in the paper.
    """

    def _selector(inst: PBInstance, state: RolloutState) -> Set[str]:
        del state  # the published rule is history independent
        feats = project_features(inst, scheme)
        approval_share = feats.static[:, 0]
        cost_share = feats.static[:, 2]
        scores = np.sqrt(approval_share) / (1.0 + cost_share)

        order = sorted(
            (i for i in range(len(feats.project_ids)) if feats.has_support[i]),
            key=lambda i: (-scores[i], feats.costs[i], feats.project_ids[i]),
        )
        remaining = float(inst.budget)
        winners: Set[str] = set()
        for i in order:
            cost = feats.costs[i]
            if cost <= remaining:
                winners.add(feats.project_ids[i])
                remaining -= cost
        return winners

    return _selector


def llmrule_card_selector(scheme: str = SCHEME) -> SelectorPolicy:
    """Reproduce the published LLMRule example for count satisfaction.

    The AAMAS 2026 appendix gives executable code whose score simplifies to

        approval_share / cost_share * log(1 + approval_share).

    Unlike :func:`llmrule_cost_selector`, this rule was generated for the same
    approval-count welfare definition used in our main experiments.
    """

    def _selector(inst: PBInstance, state: RolloutState) -> Set[str]:
        del state
        feats = project_features(inst, scheme)
        approval_share = feats.static[:, 0]
        cost_share = feats.static[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.where(
                cost_share > 0,
                approval_share / cost_share * np.log1p(approval_share),
                0.0,
            )

        order = sorted(
            (i for i in range(len(feats.project_ids)) if feats.has_support[i]),
            key=lambda i: (-scores[i], feats.costs[i], feats.project_ids[i]),
        )
        remaining = float(inst.budget)
        winners: Set[str] = set()
        for i in order:
            cost = feats.costs[i]
            if cost <= remaining:
                winners.add(feats.project_ids[i])
                remaining -= cost
        return winners

    return _selector


def _deficit_weights(state: RolloutState, cohort_order: Sequence[str]) -> np.ndarray:
    """Normalized non-negative carried deficit per cohort, summing to one."""
    raw = np.array(
        [max(0.0, state.deficits.get(c, 0.0)) for c in cohort_order], dtype=float
    )
    total = raw.sum()
    if total <= 0:
        return np.zeros(len(cohort_order))
    return raw / total


def _score_vector(
    inst: PBInstance,
    state: RolloutState,
    weights: np.ndarray,
    scheme: str,
) -> Tuple[ProjectFeatures, np.ndarray]:
    """Return the canonical project features and scores for one policy call.

    Both the direct selector and the payment-kernel intervention use this exact
    helper. Their frozen-score comparison therefore changes the allocation
    kernel, not the learned parameters or feature computation.
    """
    feats = project_features(inst, scheme)
    matrix = feats.static.copy()
    if weights[3] != 0.0 and feats.cohort_order:
        matrix[:, 3] = feats.cohort_share @ _deficit_weights(
            state, feats.cohort_order
        )
    return feats, matrix @ weights


def score_selector(
    weights: Sequence[float],
    scheme: str = SCHEME,
    support_floor: Optional[float] = None,
) -> SelectorPolicy:
    """Build a selector that funds projects in learned-score order.

    Args:
        weights: One weight per entry of `PROJECT_FEATURES`.
        scheme: Cohort scheme.
        support_floor: If set to kappa, restrict funding to projects satisfying
            |A(p)|/n >= c(p)/(kappa*b) -- the Equal Shares support floor of
            Proposition 2, imposed here *without* the mechanism. This isolates
            the coverage constraint from proportional payment: if a policy still
            degenerates under it, the mechanism's protection is not the floor.

    Returns:
        A selector suitable for `iclr_env.rollout_selector`.

    Raises:
        ValueError: When the weight vector has the wrong length.
    """
    v = np.asarray(weights, dtype=float)
    if v.shape != (N_PROJECT_FEATURES,):
        raise ValueError(f"expected {N_PROJECT_FEATURES} weights, got {v.shape}")

    def _selector(inst: PBInstance, state: RolloutState) -> Set[str]:
        feats, scores = _score_vector(inst, state, v, scheme)
        eligible = feats.has_support
        if support_floor is not None and support_floor > 0:
            n = len(inst.votes) or 1
            counts = np.zeros(len(feats.project_ids))
            index = {pid: i for i, pid in enumerate(feats.project_ids)}
            for vote in inst.votes:
                for pid in vote.projects:
                    j = index.get(pid)
                    if j is not None:
                        counts[j] += 1
            floor_ok = (counts / n) >= (feats.costs / (support_floor * inst.budget)) - 1e-12
            eligible = eligible & floor_ok
        order = sorted(
            (i for i in range(len(feats.project_ids)) if eligible[i]),
            key=lambda i: (-scores[i], feats.costs[i], feats.project_ids[i]),
        )
        remaining = float(inst.budget)
        winners: Set[str] = set()
        for i in order:
            cost = feats.costs[i]
            if cost <= remaining:
                winners.add(feats.project_ids[i])
                remaining -= cost
        return winners

    return _selector


def _minimum_affordable_price(
    cost: float,
    supporter_ids: Sequence[int],
    balances: Sequence[float],
) -> Optional[float]:
    """Return the smallest equalized supporter charge, or ``None`` if infeasible."""
    if cost <= 0 or sum(balances[i] for i in supporter_ids) < cost - 1e-9:
        return None
    ordered = sorted(balances[i] for i in supporter_ids)
    paid_so_far = 0.0
    for offset, balance in enumerate(ordered):
        remaining_payers = len(ordered) - offset
        candidate = (cost - paid_so_far) / remaining_payers
        if candidate <= balance + 1e-12:
            return candidate
        paid_so_far += balance
    return None


def payment_gated_score_selector(
    weights: Sequence[float],
    scheme: str = SCHEME,
    completion: bool = True,
) -> SelectorPolicy:
    """Replay a fixed project score through an approver-funded payment kernel.

    This is a diagnostic intervention, not a new claim that the resulting rule
    is standard MES. It holds the outcome arm's five features and learned score
    vector fixed, but replaces direct municipal-budget greedy fill with the
    following kernel: start voters at uniform ``b/n`` balances; repeatedly pick
    the highest-scoring project whose supporters can collectively pay for it;
    and charge those supporters at the smallest equalized price. With
    ``completion=True``, any remainder is completed by approval count exactly as
    in the paper's MES implementation.

    Comparing this selector with :func:`score_selector` for the same frozen
    weights isolates the allocation-kernel intervention far more tightly than
    comparing separately trained endowment and project-score classes. Project
    ordering inside the payment phase is still score-based rather than MES's
    minimum-price ordering, so this control must not be described as MES.
    """
    v = np.asarray(weights, dtype=float)
    if v.shape != (N_PROJECT_FEATURES,):
        raise ValueError(f"expected {N_PROJECT_FEATURES} weights, got {v.shape}")

    def _selector(inst: PBInstance, state: RolloutState) -> Set[str]:
        feats, scores = _score_vector(inst, state, v, scheme)
        n_voters = len(inst.votes)
        if n_voters == 0 or inst.budget <= 0:
            return set()

        supporter_ids: Dict[str, List[int]] = {
            pid: [] for pid in feats.project_ids
        }
        for voter_id, vote in enumerate(inst.votes):
            for pid in vote.projects:
                if pid in supporter_ids:
                    supporter_ids[pid].append(voter_id)

        balances = [float(inst.budget) / n_voters] * n_voters
        active = {
            row
            for row, pid in enumerate(feats.project_ids)
            if feats.has_support[row] and feats.costs[row] > 0
        }
        winners: Set[str] = set()

        while active:
            affordable: List[Tuple[Tuple[float, float, str], int, float]] = []
            permanently_infeasible: List[int] = []
            for row in active:
                pid = feats.project_ids[row]
                rho = _minimum_affordable_price(
                    float(feats.costs[row]), supporter_ids[pid], balances
                )
                if rho is None:
                    permanently_infeasible.append(row)
                    continue
                affordable.append(
                    ((-float(scores[row]), float(feats.costs[row]), pid), row, rho)
                )
            active.difference_update(permanently_infeasible)
            if not affordable:
                break

            _, chosen, rho = min(affordable, key=lambda item: item[0])
            pid = feats.project_ids[chosen]
            for voter_id in supporter_ids[pid]:
                balances[voter_id] -= min(balances[voter_id], rho)
                if balances[voter_id] < 1e-12:
                    balances[voter_id] = 0.0
            winners.add(pid)
            active.discard(chosen)

        if completion:
            spent = sum(inst.projects[pid].cost for pid in winners)
            remaining = float(inst.budget) - spent
            order = sorted(
                (
                    row
                    for row, pid in enumerate(feats.project_ids)
                    if feats.has_support[row] and pid not in winners
                ),
                key=lambda row: (
                    -len(supporter_ids[feats.project_ids[row]]),
                    float(feats.costs[row]),
                    feats.project_ids[row],
                ),
            )
            for row in order:
                cost = float(feats.costs[row])
                if 0 < cost <= remaining + 1e-9:
                    winners.add(feats.project_ids[row])
                    remaining -= cost

        # This assertion protects the experimental intervention from silently
        # becoming infeasible after future completion changes.
        assert sum(inst.projects[pid].cost for pid in winners) <= inst.budget + 1e-6
        return winners

    return _selector
