"""Shared regression tests for analysis-time policy replay."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from iclr_analysis_rollout import held_out_selections
from iclr_endowment_stats import _composition, _project_representation_tv
from iclr_env import EnvConfig
from iclr_theory_check import _outside_uniform_floor
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


def test_held_out_selections_replays_training_prefix() -> None:
    data = [
        SimpleNamespace(
            ref=SimpleNamespace(key="synthetic/series", years=(2021, 2022)),
            test_years=(2022,),
            all_years={2021: _instance("analysis-history-2021"),
                       2022: _instance("analysis-history-2022")},
        )
    ]

    def history_sensitive_selector(inst, state):
        del inst
        return {"late"} if state.year_index else {"early"}

    rows = held_out_selections(history_sensitive_selector, data, EnvConfig())

    assert len(rows) == 1
    assert rows[0].series == "synthetic/series"
    assert rows[0].year == 2022
    assert rows[0].winners == frozenset({"late"})


def test_composition_uses_replayed_held_out_winners() -> None:
    data = [
        SimpleNamespace(
            ref=SimpleNamespace(key="synthetic/series", years=(2021, 2022)),
            test_years=(2022,),
            all_years={2021: _instance("composition-history-2021"),
                       2022: _instance("composition-history-2022")},
        )
    ]

    def history_sensitive_selector(inst, state):
        del inst
        return {"late"} if state.year_index else {"early"}

    got = _composition(history_sensitive_selector, data, EnvConfig())

    assert got["mean_projects_funded"] == 1.0
    assert got["mean_cost_share"] == 0.2


def test_project_representation_uses_electorate_shares_not_equal_groups() -> None:
    """Catches confusing equal-group concentration with representation."""
    inst = PBInstance(
        path="representation-tv",
        meta={"budget": "100", "vote_type": "approval"},
        projects={"p": Project("p", 100.0, None)},
        votes=[
            Vote("m1", ("p",), age=20, sex="M"),
            Vote("m2", ("p",), age=20, sex="M"),
            Vote("m3", (), age=20, sex="M"),
            Vote("f1", ("p",), age=20, sex="F"),
        ],
    )

    # Electorate=(3/4,1/4), supporters=(2/3,1/3), so TV=1/12.
    assert _project_representation_tv(inst, "p") == pytest.approx(1.0 / 12.0)


def test_floor_statistic_uses_replayed_held_out_winners() -> None:
    def floor_instance(path: str) -> PBInstance:
        return PBInstance(
            path=path,
            meta={"budget": "100", "vote_type": "approval"},
            projects={
                "early": Project("early", 10.0, None),
                "late": Project("late", 75.0, None),
            },
            votes=[
                Vote("v1", ("early", "late"), age=20, sex="M"),
                Vote("v2", ("early",), age=20, sex="F"),
            ],
        )

    data = [
        SimpleNamespace(
            ref=SimpleNamespace(key="synthetic/series", years=(2021, 2022)),
            test_years=(2022,),
            all_years={2021: floor_instance("floor-history-2021"),
                       2022: floor_instance("floor-history-2022")},
        )
    ]

    def history_sensitive_selector(inst, state):
        del inst
        return {"late"} if state.year_index else {"early"}

    got = _outside_uniform_floor(history_sensitive_selector, data, EnvConfig())

    assert got == {"pct_projects": 100.0, "pct_budget": 100.0}
