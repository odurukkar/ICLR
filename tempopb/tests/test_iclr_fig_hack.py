"""Regression tests for the reward-hack figure's rollout semantics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from iclr_env import EnvConfig
from iclr_fig_hack import _collect
from parse_pb import PBInstance, Project, Vote


def _instance(path: str) -> PBInstance:
    return PBInstance(
        path=path,
        meta={"budget": "100", "vote_type": "approval"},
        projects={
            "early": Project("early", 10.0, None),
            "late": Project("late", 20.0, None),
        },
        votes=[
            Vote("v1", ("early", "late"), age=20, sex="M"),
            Vote("v2", ("early", "late"), age=20, sex="F"),
        ],
    )


def test_collect_warms_policy_state_before_held_out_year() -> None:
    """The plotted test winner must see the training-year rollout history."""
    data = [
        SimpleNamespace(
            ref=SimpleNamespace(key="synthetic/series", years=(2021, 2022)),
            test_years=(2022,),
            all_years={2021: _instance("figure-history-2021"),
                       2022: _instance("figure-history-2022")},
        )
    ]

    def history_sensitive_selector(inst, state):
        del inst
        return {"late"} if state.year_index else {"early"}

    x, y, spend = _collect(history_sensitive_selector, data, EnvConfig())

    assert x.tolist() == pytest.approx([1.0])
    assert y.tolist() == pytest.approx([0.2])
    assert spend.tolist() == pytest.approx([20.0])
