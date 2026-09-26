"""Rollout environment for endowment-space policies over repeated PB series.

A policy is any callable mapping (instance, carried cohort deficits) to a
per-voter endowment vector; the allocation procedure stays Equal Shares, so
every policy output is budget-feasible and payment-phase winners are funded by
charges to their approvers. This does not assert standard priceability:
endowments are nonuniform and greedy completion lies outside that accounting. Uniform
endowments recover MES exactly; the one-scalar RES map is provided as the
hand-designed baseline the learned map must beat.

Episodes support a warm-up prefix: deficits accumulate over observed years
under the policy's own outcomes, and only the scored suffix contributes to the
reported metrics. This is what makes the temporal split meaningful — a policy
deployed in 2023 has seen 2016-2022, it does not start cold.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from cohorts import cumulative_share_deficit, group_outcome, historical_winners
from iclr_corpus import SeriesRef, load_series
from parse_pb import PBInstance
from rules import greedy_by_votes, mes_with_endowments, res_endowments
from run_experiments import (
    SCHEME,
    approval_welfare,
    cost_welfare,
    exclusion_rate,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EnvConfig",
    "RolloutState",
    "YearResult",
    "EpisodeResult",
    "uniform_policy",
    "res_policy",
    "endowment_selector",
    "rollout",
    "rollout_selector",
    "rollout_reference",
    "aggregate",
]


@dataclass
class RolloutState:
    """History carried into year t, shared with the policy.

    `deficits` is the signed absolute deficit sum_{tau<t} (e - u) per cohort,
    matching the quantity RES rolls over. `entitlements` accumulates
    sum_{tau<t} e, which lets a policy normalize a deficit against the money
    the cohort was owed rather than against raw currency.
    """

    deficits: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    entitlements: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    year_index: int = 0

    def update(self, outcome: Mapping[str, Mapping[str, float]]) -> None:
        """Fold one year's cohort ledger into the carried history."""
        for cohort, vals in outcome.items():
            self.deficits[cohort] += vals["entitlement"] - vals["utility"]
            self.entitlements[cohort] += vals["entitlement"]
        self.year_index += 1


# An endowment policy maps (instance, history) to a per-voter endowment vector;
# the mechanism (Equal Shares) then turns that into winners. A selector skips
# the endowment channel and names the winner set directly. Both arms of the
# paper are selectors, which is why the ledger below is written against the
# more general type.
EndowmentPolicy = Callable[[PBInstance, RolloutState], List[float]]
SelectorPolicy = Callable[[PBInstance, RolloutState], Set[str]]


@dataclass(frozen=True)
class EnvConfig:
    """Fixed evaluation settings shared by every policy under comparison."""

    scheme: str = SCHEME
    completion: bool = True


@dataclass(frozen=True)
class YearResult:
    """Outcome of one election under one policy."""

    year: int
    n_winners: int
    spent: float
    welfare: float
    cost_welfare: float
    exclusion: float
    scored: bool


@dataclass(frozen=True)
class EpisodeResult:
    """Metrics for one series, computed over the scored years only."""

    series: str
    scored_years: Tuple[int, ...]
    worst_csd: Optional[float]
    worst_cohort: Optional[str]
    mean_csd: Optional[float]
    welfare: float
    cost_welfare: float
    exclusion: float
    years: Tuple[YearResult, ...]


def uniform_policy(inst: PBInstance, state: RolloutState) -> List[float]:
    """Equal endowments b/n for every voter. Recovers standard MES."""
    n = len(inst.votes)
    if n == 0:
        return []
    return [inst.budget / n] * n


def res_policy(lam: float, scheme: str = SCHEME) -> EndowmentPolicy:
    """The hand-designed one-scalar map: endowment bonus proportional to deficit.

    Args:
        lam: Rollover intensity in [0, 1]; lam = 0 reduces to `uniform_policy`.
        scheme: Cohort scheme passed through to `res_endowments`.

    Returns:
        A policy callable suitable for `rollout`.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"rollover intensity must lie in [0, 1], got {lam}")

    def _policy(inst: PBInstance, state: RolloutState) -> List[float]:
        return res_endowments(inst, dict(state.deficits), lam, scheme)

    return _policy


def _score_episode(
    series: str,
    scored: Dict[int, Dict[str, Dict[str, float]]],
    years: Sequence[YearResult],
    cfg: EnvConfig,
) -> EpisodeResult:
    """Reduce a rollout's per-year ledger to episode metrics."""
    scored_years = tuple(sorted(scored))
    cohorts = sorted({c for out in scored.values() for c in out})
    csds: Dict[str, float] = {}
    for cohort in cohorts:
        value = cumulative_share_deficit((scored[y] for y in scored_years), cohort)
        if value is not None:
            csds[cohort] = value

    worst_cohort = max(csds, key=lambda c: csds[c]) if csds else None
    scored_rows = [y for y in years if y.scored]
    n = len(scored_rows) or 1
    return EpisodeResult(
        series=series,
        scored_years=scored_years,
        worst_csd=csds[worst_cohort] if worst_cohort else None,
        worst_cohort=worst_cohort,
        mean_csd=sum(csds.values()) / len(csds) if csds else None,
        welfare=sum(y.welfare for y in scored_rows),
        cost_welfare=sum(y.cost_welfare for y in scored_rows),
        exclusion=sum(y.exclusion for y in scored_rows) / n,
        years=tuple(years),
    )


def endowment_selector(
    policy: EndowmentPolicy, cfg: Optional[EnvConfig] = None
) -> SelectorPolicy:
    """Wrap an endowment map into a selector by running Equal Shares on it."""
    cfg = cfg or EnvConfig()

    def _selector(inst: PBInstance, state: RolloutState) -> Set[str]:
        return mes_with_endowments(
            inst, endowments=policy(inst, state), completion=cfg.completion
        )

    return _selector


def rollout_selector(
    ref: SeriesRef,
    selector: SelectorPolicy,
    score_years: Optional[Sequence[int]] = None,
    cfg: Optional[EnvConfig] = None,
    instances: Optional[Dict[int, PBInstance]] = None,
) -> EpisodeResult:
    """Replay one series under any winner-selection rule, carrying its own history.

    This is the single ledger both arms of the paper are scored through, so an
    endowment-space policy and an outcome-space policy are never compared
    across two different accounting implementations.

    Args:
        ref: The series to replay.
        selector: Maps (instance, carried history) to a winner set.
        score_years: Years contributing to the reported metrics. Earlier years
            are still replayed as a warm-up so deficits carry in. Defaults to
            every year of the series.
        cfg: Evaluation settings.
        instances: Pre-parsed instances, to avoid re-parsing across many
            policy evaluations during training.

    Returns:
        Episode metrics over the scored years.
    """
    cfg = cfg or EnvConfig()
    insts = instances if instances is not None else load_series(ref)
    scored_set = set(score_years) if score_years is not None else set(ref.years)

    state = RolloutState()
    scored: Dict[int, Dict[str, Dict[str, float]]] = {}
    rows: List[YearResult] = []

    for year in ref.years:
        inst = insts.get(year)
        if inst is None:
            continue
        winners = selector(inst, state)
        outcome = group_outcome(inst, winners, cfg.scheme)
        state.update(outcome)

        is_scored = year in scored_set
        if is_scored:
            scored[year] = outcome
        rows.append(
            YearResult(
                year=year,
                n_winners=len(winners),
                spent=sum(inst.projects[p].cost for p in winners if p in inst.projects),
                welfare=approval_welfare(inst, winners),
                cost_welfare=cost_welfare(inst, winners),
                exclusion=exclusion_rate(inst, winners),
                scored=is_scored,
            )
        )

    return _score_episode(ref.key, scored, rows, cfg)


def rollout(
    ref: SeriesRef,
    policy: EndowmentPolicy,
    score_years: Optional[Sequence[int]] = None,
    cfg: Optional[EnvConfig] = None,
    instances: Optional[Dict[int, PBInstance]] = None,
) -> EpisodeResult:
    """Replay one series under an endowment map. Thin wrapper over the selector."""
    cfg = cfg or EnvConfig()
    return rollout_selector(
        ref,
        endowment_selector(policy, cfg),
        score_years=score_years,
        cfg=cfg,
        instances=instances,
    )


def rollout_reference(
    ref: SeriesRef,
    rule: str,
    score_years: Optional[Sequence[int]] = None,
    cfg: Optional[EnvConfig] = None,
    instances: Optional[Dict[int, PBInstance]] = None,
) -> EpisodeResult:
    """Replay a non-endowment reference rule: 'historical' or 'greedy'.

    These sit outside the endowment family and exist to anchor the frontier.

    Raises:
        ValueError: On an unknown rule name.
        RuntimeError: When a series has no recorded winners for 'historical'.
    """
    if rule not in {"historical", "greedy"}:
        raise ValueError(f"unknown reference rule: {rule!r}")
    cfg = cfg or EnvConfig()
    insts = instances if instances is not None else load_series(ref)
    scored_set = set(score_years) if score_years is not None else set(ref.years)

    scored: Dict[int, Dict[str, Dict[str, float]]] = {}
    rows: List[YearResult] = []
    for year in ref.years:
        inst = insts.get(year)
        if inst is None:
            continue
        if rule == "historical":
            winners = historical_winners(inst)
            if not winners:
                raise RuntimeError(f"{ref.key} {year}: no recorded winners")
        else:
            winners = greedy_by_votes(inst)
        outcome = group_outcome(inst, winners, cfg.scheme)
        is_scored = year in scored_set
        if is_scored:
            scored[year] = outcome
        rows.append(
            YearResult(
                year=year,
                n_winners=len(winners),
                spent=sum(inst.projects[p].cost for p in winners if p in inst.projects),
                welfare=approval_welfare(inst, winners),
                cost_welfare=cost_welfare(inst, winners),
                exclusion=exclusion_rate(inst, winners),
                scored=is_scored,
            )
        )
    return _score_episode(ref.key, scored, rows, cfg)


def aggregate(episodes: Sequence[EpisodeResult]) -> Dict[str, float]:
    """Mean metrics over episodes, skipping series with no scoreable cohort."""
    worst = [e.worst_csd for e in episodes if e.worst_csd is not None]
    mean = [e.mean_csd for e in episodes if e.mean_csd is not None]
    if not worst:
        return {"n_series": 0.0}
    return {
        "n_series": float(len(worst)),
        "worst_csd": sum(worst) / len(worst),
        "mean_csd": sum(mean) / len(mean) if mean else float("nan"),
        "welfare": sum(e.welfare for e in episodes),
        "cost_welfare": sum(e.cost_welfare for e in episodes),
        "exclusion": sum(e.exclusion for e in episodes) / len(episodes),
    }
