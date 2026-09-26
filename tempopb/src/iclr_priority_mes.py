"""Learn project utilities while retaining Equal Shares payments.

The learned vector changes only the priority among currently affordable
projects. Affordability prices and supporter charges are exactly the standard
approval-MES equations, and zero weights recover the repository's ordinary
MES implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, FrozenSet, List, Sequence, Set, Tuple

import numpy as np

from iclr_corpus import SeriesRef, load_series
from iclr_env import RolloutState, SelectorPolicy
from iclr_outcome import (
    N_PROJECT_FEATURES,
    ProjectFeatures,
    _score_vector,
)
from parse_pb import PBInstance
from rules import mes_with_endowments
from run_experiments import SCHEME

_SUPPORTER_CACHE: Dict[
    Tuple[str, Tuple[str, ...], int], Dict[str, np.ndarray]
] = {}


@dataclass(frozen=True)
class PaymentRound:
    """Auditable payment-phase record for one purchased project."""

    project_id: str
    rho: float
    priority: float
    supporter_ids: Tuple[int, ...]
    payments: Tuple[float, ...]


@dataclass(frozen=True)
class PriorityMESOutcome:
    """Winner set plus separate payment and completion traces."""

    winners: FrozenSet[str]
    payment_winners: Tuple[str, ...]
    completion_winners: Tuple[str, ...]
    rounds: Tuple[PaymentRound, ...]
    final_balances: Tuple[float, ...]


def _minimum_affordable_price_numpy(
    cost: float,
    supporter_ids: np.ndarray,
    balances: np.ndarray,
) -> float | None:
    """Vectorized equivalent of the standard equalized affordability price."""

    if cost <= 0 or supporter_ids.size == 0:
        return None
    supporter_balances = balances[supporter_ids]
    if float(np.sum(supporter_balances)) < cost - 1e-9:
        return None
    ordered = np.sort(supporter_balances)
    prefix_before = np.empty_like(ordered)
    prefix_before[0] = 0.0
    if ordered.size > 1:
        prefix_before[1:] = np.cumsum(ordered[:-1])
    remaining = np.arange(ordered.size, 0, -1, dtype=float)
    candidates = (cost - prefix_before) / remaining
    valid = np.flatnonzero(candidates <= ordered + 1e-12)
    if valid.size == 0:
        return None
    return float(candidates[int(valid[0])])


def _validated_weights(weights: Sequence[float]) -> np.ndarray:
    vector = np.asarray(weights, dtype=float)
    if vector.shape != (N_PROJECT_FEATURES,):
        raise ValueError(
            f"expected {N_PROJECT_FEATURES} weights, got {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("priority-MES weights must be finite")
    return vector


def _supporter_arrays(
    inst: PBInstance, project_ids: Tuple[str, ...]
) -> Dict[str, np.ndarray]:
    """Cache election-invariant supporter indices for repeated fitting calls."""

    key = (inst.path, project_ids, len(inst.votes))
    cached = _SUPPORTER_CACHE.get(key)
    if cached is not None:
        return cached
    mutable: Dict[str, List[int]] = {pid: [] for pid in project_ids}
    for voter_id, vote in enumerate(inst.votes):
        for pid in vote.projects:
            if pid in mutable:
                mutable[pid].append(voter_id)
    result = {
        pid: np.asarray(ids, dtype=np.int64) for pid, ids in mutable.items()
    }
    _SUPPORTER_CACHE[key] = result
    return result


def priority_log_multipliers(
    inst: PBInstance,
    state: RolloutState,
    weights: Sequence[float],
    scheme: str = SCHEME,
) -> Tuple[ProjectFeatures, np.ndarray]:
    """Return canonical project features and clipped log utility multipliers."""

    vector = _validated_weights(weights)
    feats, raw_scores = _score_vector(inst, state, vector, scheme)
    log_multipliers = np.clip(raw_scores, -10.0, 10.0)
    if not np.all(np.isfinite(log_multipliers)):
        raise ValueError("project utility multipliers must be finite")
    return feats, log_multipliers


def priority_mes_outcome(
    inst: PBInstance,
    state: RolloutState,
    weights: Sequence[float],
    scheme: str = SCHEME,
    completion: bool = True,
    endowments: Sequence[float] | None = None,
) -> PriorityMESOutcome:
    """Run utility-priority MES and return its payment audit trail."""

    vector = _validated_weights(weights)
    n_voters = len(inst.votes)
    if endowments is not None:
        initial = np.asarray(endowments, dtype=float)
        if initial.shape != (n_voters,):
            raise ValueError(
                f"expected {n_voters} endowments, got {initial.shape}"
            )
        if not np.all(np.isfinite(initial)) or np.any(initial < 0):
            raise ValueError("endowments must be finite and non-negative")
        tolerance = 1e-6 * max(1.0, abs(float(inst.budget)))
        if not math.isclose(
            float(initial.sum()), float(inst.budget), rel_tol=0.0, abs_tol=tolerance
        ):
            raise ValueError("endowments must sum to the municipal budget")
    else:
        initial = None
    if n_voters == 0 or inst.budget <= 0:
        return PriorityMESOutcome(frozenset(), (), (), (), tuple(0.0 for _ in inst.votes))

    feats, log_multipliers = priority_log_multipliers(
        inst, state, vector, scheme
    )
    supporter_ids = _supporter_arrays(inst, feats.project_ids)

    initial_balance = float(inst.budget) / n_voters
    balances = (
        initial.copy()
        if initial is not None
        else np.full(n_voters, initial_balance, dtype=float)
    )
    active: Set[int] = {
        row
        for row, pid in enumerate(feats.project_ids)
        if feats.has_support[row] and feats.costs[row] > 0
    }
    payment_winners: List[str] = []
    rounds: List[PaymentRound] = []

    while active:
        affordable: List[Tuple[Tuple[float, float, str], int, float]] = []
        infeasible: List[int] = []
        for row in active:
            pid = feats.project_ids[row]
            rho = _minimum_affordable_price_numpy(
                float(feats.costs[row]), supporter_ids[pid], balances
            )
            if rho is None:
                infeasible.append(row)
                continue
            multiplier = math.exp(float(log_multipliers[row]))
            key = (rho / multiplier, float(feats.costs[row]), pid)
            affordable.append((key, row, rho))
        active.difference_update(infeasible)
        if not affordable:
            break

        key, chosen, rho = min(affordable, key=lambda item: item[0])
        pid = feats.project_ids[chosen]
        id_array = supporter_ids[pid]
        payment_array = np.minimum(balances[id_array], rho)
        balances[id_array] -= payment_array
        balances[id_array] = np.where(balances[id_array] < 1e-12, 0.0, balances[id_array])
        ids = tuple(int(voter_id) for voter_id in id_array)
        payments = tuple(float(payment) for payment in payment_array)
        rounds.append(
            PaymentRound(
                project_id=pid,
                rho=float(rho),
                priority=float(key[0]),
                supporter_ids=ids,
                payments=payments,
            )
        )
        payment_winners.append(pid)
        active.discard(chosen)

    winners = set(payment_winners)
    completion_winners: List[str] = []
    if completion:
        spent = sum(inst.projects[pid].cost for pid in winners)
        remaining = float(inst.budget) - spent
        counts = {pid: int(ids.size) for pid, ids in supporter_ids.items()}
        for pid in sorted(
            counts,
            key=lambda project_id: (
                -counts[project_id],
                inst.projects[project_id].cost,
                project_id,
            ),
        ):
            if pid in winners or counts[pid] == 0:
                continue
            cost = float(inst.projects[pid].cost)
            if 0 < cost <= remaining:
                winners.add(pid)
                completion_winners.append(pid)
                remaining -= cost

    total_cost = sum(inst.projects[pid].cost for pid in winners)
    if total_cost > inst.budget + 1e-6:
        raise AssertionError(
            f"priority-MES exceeded budget: spent {total_cost}, budget {inst.budget}"
        )
    for round_ in rounds:
        cost = float(inst.projects[round_.project_id].cost)
        if not math.isclose(sum(round_.payments), cost, abs_tol=1e-6):
            raise AssertionError(
                f"payments for {round_.project_id} sum to {sum(round_.payments)}, "
                f"not cost {cost}"
            )

    return PriorityMESOutcome(
        winners=frozenset(winners),
        payment_winners=tuple(payment_winners),
        completion_winners=tuple(completion_winners),
        rounds=tuple(rounds),
        final_balances=tuple(float(balance) for balance in balances),
    )


def priority_mes_selector(
    weights: Sequence[float],
    scheme: str = SCHEME,
    completion: bool = True,
) -> SelectorPolicy:
    """Build a rollout selector for learned project utilities inside MES."""

    vector = _validated_weights(weights).copy()

    def _selector(inst: PBInstance, state: RolloutState) -> Set[str]:
        return set(
            priority_mes_outcome(
                inst,
                state,
                vector,
                scheme=scheme,
                completion=completion,
            ).winners
        )

    return _selector


def fast_mes_with_endowments(
    inst: PBInstance,
    endowments: Sequence[float],
    completion: bool = True,
) -> Set[str]:
    """Vectorized exact-priority MES for repeated nonuniform-endowment fitting."""

    return set(
        priority_mes_outcome(
            inst,
            RolloutState(),
            np.zeros(N_PROJECT_FEATURES),
            completion=completion,
            endowments=endowments,
        ).winners
    )


def verify_priority_mes_corpus(refs: Sequence[SeriesRef]) -> Dict[str, object]:
    """Run containment, determinism, payment, and budget gates on a corpus."""

    zero = np.zeros(N_PROJECT_FEATURES)
    n_elections = 0
    identity_checks = 0
    determinism_checks = 0
    budget_checks = 0
    payment_rounds = 0
    for ref in refs:
        instances = load_series(ref)
        for year in ref.years:
            inst = instances[year]
            n_elections += 1
            for completion in (False, True):
                expected = mes_with_endowments(inst, completion=completion)
                first = priority_mes_outcome(
                    inst,
                    RolloutState(),
                    zero,
                    completion=completion,
                )
                second = priority_mes_outcome(
                    inst,
                    RolloutState(),
                    zero,
                    completion=completion,
                )
                identity_checks += 1
                if set(first.winners) != expected:
                    raise RuntimeError(
                        f"MES containment failed for {ref.key} {year} "
                        f"completion={completion}"
                    )
                determinism_checks += 1
                if first != second:
                    raise RuntimeError(
                        f"determinism failed for {ref.key} {year} "
                        f"completion={completion}"
                    )
                spent = sum(inst.projects[pid].cost for pid in first.winners)
                budget_checks += 1
                if spent > inst.budget + 1e-6:
                    raise RuntimeError(
                        f"budget feasibility failed for {ref.key} {year}: {spent}"
                    )
                payment_rounds += len(first.rounds)
    return {
        "status": "pass",
        "n_series": len(refs),
        "n_elections": n_elections,
        "identity_checks": identity_checks,
        "determinism_checks": determinism_checks,
        "budget_checks": budget_checks,
        "payment_rounds_checked": payment_rounds,
        "completion_modes": [False, True],
        "zero_weights": zero.tolist(),
    }
