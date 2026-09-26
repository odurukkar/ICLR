"""Learned endowment maps over cohort features.

The family is deliberately a strict superset of the hand-designed rule: a
voter's endowment is

    beta_i = base * (1 + sum_k w_k f_k(g(i))),   base = b_t / |N_t|

clamped at zero and rescaled to sum to the budget. The first feature is
    f_1(g) = max(0, D_{t-1}(g)) / (|g_t| * base)
so that w = (lambda, 0, ..., 0) reproduces Rollover Equal Shares *exactly*.
The hypothesis class can therefore match the hand-designed rule exactly. The
optimizer uses that rule only as its initial mean and does not automatically
retain it as a candidate, so fitted performance is always reported separately
from the replayed baseline.

Instance-only features are cached: they do not depend on the weights or on
carried history, so they are computed once per election and reused across every
policy evaluation during training.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cohorts import cohort_of
from iclr_env import EndowmentPolicy, RolloutState
from parse_pb import PBInstance
from run_experiments import SCHEME

logger = logging.getLogger(__name__)

FEATURE_NAMES: Tuple[str, ...] = (
    "deficit_per_capita",   # RES axis: w_1 = lambda reproduces RES exactly
    "deficit_normalized",   # signed deficit as a fraction of entitlement owed
    "share_deviation",      # cohort size relative to an equal split
    "ballot_length_dev",    # mean approvals of g relative to the electorate
    "cost_focus_dev",       # mean cost of g's approved projects, relative
    "overlap_dev",          # cosine overlap of g's approval profile with all
)
N_FEATURES = len(FEATURE_NAMES)

__all__ = [
    "FEATURE_NAMES",
    "N_FEATURES",
    "InstanceFeatures",
    "instance_features",
    "clear_feature_cache",
    "linear_policy",
    "res_equivalent_weights",
]

# Instance-only features keyed by instance path: independent of weights and of
# carried history, so they survive across the whole training run.
# Keyed by (instance path, cohort scheme): the features depend on the
# partition, so keying on path alone silently returns the wrong cohorts
# when more than one scheme is used in a process.
_CACHE: Dict[Tuple[str, str], "InstanceFeatures"] = {}


@dataclass(frozen=True)
class InstanceFeatures:
    """Per-election structure a policy may condition on, excluding history."""

    voter_cohorts: Tuple[Optional[str], ...]
    cohort_sizes: Dict[str, int]
    static: Dict[str, np.ndarray]  # feature name -> value per cohort (aligned)
    cohort_order: Tuple[str, ...]

    def static_vector(self, cohort: str) -> np.ndarray:
        """Static feature values for one cohort, in FEATURE_NAMES order."""
        idx = self.cohort_index.get(cohort)
        if idx is None:
            return np.zeros(N_FEATURES)
        return np.array(
            [
                0.0,  # deficit_per_capita: history-dependent, filled at rollout
                0.0,  # deficit_normalized: history-dependent
                self.static["share_deviation"][idx],
                self.static["ballot_length_dev"][idx],
                self.static["cost_focus_dev"][idx],
                self.static["overlap_dev"][idx],
            ]
        )

    @property
    def cohort_index(self) -> Dict[str, int]:
        return {c: i for i, c in enumerate(self.cohort_order)}


def clear_feature_cache() -> None:
    """Drop cached instance features. Call between corpora, not between runs."""
    _CACHE.clear()


def instance_features(inst: PBInstance, scheme: str = SCHEME) -> InstanceFeatures:
    """Compute (and cache) the history-independent cohort features of one election.

    Args:
        inst: The election.
        scheme: Cohort scheme.

    Returns:
        Cached `InstanceFeatures` for this instance.
    """
    cached = _CACHE.get((inst.path, scheme))
    if cached is not None:
        return cached

    voter_cohorts: List[Optional[str]] = []
    members: Dict[str, List[int]] = defaultdict(list)
    for i, vote in enumerate(inst.votes):
        cohort = cohort_of(vote, scheme)
        voter_cohorts.append(cohort)
        if cohort is not None:
            members[cohort].append(i)

    cohort_order = tuple(sorted(members))
    sizes = {c: len(members[c]) for c in cohort_order}
    n_covered = sum(sizes.values())
    k = len(cohort_order)

    # Population-level reference quantities over covered voters only.
    approve_counts: Dict[str, int] = defaultdict(int)
    total_len = 0
    total_cost = 0.0
    total_appr = 0
    for cohort in cohort_order:
        for i in members[cohort]:
            approvals = inst.votes[i].projects
            total_len += len(approvals)
            for pid in approvals:
                approve_counts[pid] += 1
                project = inst.projects.get(pid)
                if project is not None:
                    total_cost += project.cost
                    total_appr += 1
    mean_len = total_len / n_covered if n_covered else 0.0
    mean_cost = total_cost / total_appr if total_appr else 0.0

    all_freq = {
        pid: cnt / n_covered for pid, cnt in approve_counts.items()
    } if n_covered else {}
    all_norm = math.sqrt(sum(v * v for v in all_freq.values())) or 1.0

    share_dev = np.zeros(k)
    len_dev = np.zeros(k)
    cost_dev = np.zeros(k)
    overlap = np.zeros(k)

    for idx, cohort in enumerate(cohort_order):
        idxs = members[cohort]
        size = len(idxs)
        share_dev[idx] = (size / n_covered) - (1.0 / k) if n_covered and k else 0.0

        g_len = 0
        g_cost = 0.0
        g_appr = 0
        g_counts: Dict[str, int] = defaultdict(int)
        for i in idxs:
            approvals = inst.votes[i].projects
            g_len += len(approvals)
            for pid in approvals:
                g_counts[pid] += 1
                project = inst.projects.get(pid)
                if project is not None:
                    g_cost += project.cost
                    g_appr += 1
        len_dev[idx] = (g_len / size) / mean_len - 1.0 if size and mean_len else 0.0
        cost_dev[idx] = (g_cost / g_appr) / mean_cost - 1.0 if g_appr and mean_cost else 0.0

        dot = 0.0
        g_sq = 0.0
        for pid, cnt in g_counts.items():
            freq = cnt / size if size else 0.0
            g_sq += freq * freq
            dot += freq * all_freq.get(pid, 0.0)
        g_norm = math.sqrt(g_sq) or 1.0
        overlap[idx] = dot / (g_norm * all_norm)

    overlap = overlap - (overlap.mean() if k else 0.0)

    features = InstanceFeatures(
        voter_cohorts=tuple(voter_cohorts),
        cohort_sizes=sizes,
        static={
            "share_deviation": share_dev,
            "ballot_length_dev": len_dev,
            "cost_focus_dev": cost_dev,
            "overlap_dev": overlap,
        },
        cohort_order=cohort_order,
    )
    _CACHE[(inst.path, scheme)] = features
    return features


def res_equivalent_weights(lam: float) -> np.ndarray:
    """The weight vector reproducing Rollover Equal Shares at intensity lambda."""
    weights = np.zeros(N_FEATURES)
    weights[0] = lam
    return weights


def _cohort_feature_matrix(
    inst: PBInstance, state: RolloutState, feats: InstanceFeatures
) -> np.ndarray:
    """Feature matrix (n_cohorts x N_FEATURES) for the current year."""
    base = inst.budget / len(inst.votes) if inst.votes else 0.0
    matrix = np.zeros((len(feats.cohort_order), N_FEATURES))
    for idx, cohort in enumerate(feats.cohort_order):
        size = feats.cohort_sizes[cohort]
        deficit = state.deficits.get(cohort, 0.0)
        entitled = state.entitlements.get(cohort, 0.0)
        if size and base:
            matrix[idx, 0] = max(0.0, deficit) / (size * base)
        if entitled > 0:
            matrix[idx, 1] = float(np.clip(deficit / entitled, -1.0, 1.0))
        matrix[idx, 2] = feats.static["share_deviation"][idx]
        matrix[idx, 3] = feats.static["ballot_length_dev"][idx]
        matrix[idx, 4] = feats.static["cost_focus_dev"][idx]
        matrix[idx, 5] = feats.static["overlap_dev"][idx]
    return matrix


def linear_policy(weights: Sequence[float], scheme: str = SCHEME) -> EndowmentPolicy:
    """Build a policy from a weight vector over `FEATURE_NAMES`.

    The endowment is base * (1 + w . f(g)), clamped at zero and rescaled to the
    budget. `linear_policy(res_equivalent_weights(lam))` is RES at intensity
    lambda; `linear_policy(zeros)` is MES.

    Args:
        weights: One weight per feature in `FEATURE_NAMES` order.
        scheme: Cohort scheme.

    Returns:
        A policy callable suitable for `iclr_env.rollout`.

    Raises:
        ValueError: When the weight vector has the wrong length.
    """
    w = np.asarray(weights, dtype=float)
    if w.shape != (N_FEATURES,):
        raise ValueError(f"expected {N_FEATURES} weights, got {w.shape}")

    def _policy(inst: PBInstance, state: RolloutState) -> List[float]:
        n = len(inst.votes)
        if n == 0:
            return []
        base = inst.budget / n
        feats = instance_features(inst, scheme)
        matrix = _cohort_feature_matrix(inst, state, feats)
        multiplier = 1.0 + matrix @ w
        np.clip(multiplier, 0.0, None, out=multiplier)
        by_cohort = {c: multiplier[i] for i, c in enumerate(feats.cohort_order)}

        raw = [
            base * by_cohort.get(cohort, 1.0) if cohort is not None else base
            for cohort in feats.voter_cohorts
        ]
        total = sum(raw)
        if total <= 0:
            return [base] * n
        scale = inst.budget / total
        return [value * scale for value in raw]

    return _policy
