"""Replay learned selectors for analysis without dropping their history state.

Tables and figures often need winner identities rather than the aggregate
metrics returned by :func:`iclr_env.rollout_selector`. This helper mirrors that
function's warm-up semantics and returns only held-out selections. Centralizing
the loop prevents analysis scripts from accidentally evaluating every test
edition from a fresh ``RolloutState``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Sequence

from cohorts import group_outcome
from iclr_env import EnvConfig, RolloutState, SelectorPolicy
from parse_pb import PBInstance


@dataclass(frozen=True)
class HeldOutSelection:
    """One selected winner set after replaying its complete series prefix."""

    series: str
    year: int
    instance: PBInstance
    winners: FrozenSet[str]


def held_out_selections(
    selector: SelectorPolicy,
    data: Sequence,
    cfg: EnvConfig,
) -> list[HeldOutSelection]:
    """Replay each series and return winner sets for its held-out editions."""
    rows: list[HeldOutSelection] = []
    for series_data in data:
        state = RolloutState()
        test_years = set(series_data.test_years)
        for year in series_data.ref.years:
            instance = series_data.all_years.get(year)
            if instance is None:
                continue
            winners = frozenset(selector(instance, state))
            outcome = group_outcome(instance, winners, cfg.scheme)
            state.update(outcome)
            if year in test_years:
                rows.append(
                    HeldOutSelection(
                        series=series_data.ref.key,
                        year=year,
                        instance=instance,
                        winners=winners,
                    )
                )
    return rows
