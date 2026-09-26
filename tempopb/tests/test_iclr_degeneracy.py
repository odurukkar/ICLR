"""Regression and ledger tests for the representative-support theorem."""

from __future__ import annotations

import pytest

def test_representation_identity_is_zero_for_representative_projects() -> None:
    """Catches using raw supporter shares instead of representation ratios."""
    from iclr_degeneracy import representation_csd

    got = representation_csd(
        group_shares={"A": 0.25, "B": 0.75},
        project_group_shares=[
            {"A": 0.25, "B": 0.75},
            {"A": 0.25, "B": 0.75},
        ],
        cost_weights=[0.2, 0.8],
    )

    assert got == pytest.approx({"A": 0.0, "B": 0.0}, abs=1e-12)


def test_representation_identity_tracks_skew_by_group() -> None:
    """Catches dropping the entitlement-share normalization."""
    from iclr_degeneracy import representation_csd

    got = representation_csd(
        group_shares={"A": 0.5, "B": 0.5},
        project_group_shares=[{"A": 0.75, "B": 0.25}],
        cost_weights=[1.0],
    )

    assert got == pytest.approx({"A": -0.5, "B": 0.5}, abs=1e-12)


def test_representative_support_has_zero_csd_and_high_exclusion() -> None:
    """Catches evaluating the construction outside the production ledger."""
    from iclr_degeneracy import evaluate_construction

    got = evaluate_construction(n_per_group=100, supporters_per_group=1)

    assert got["groups"] == 8
    assert got["voters"] == 800
    assert got["supporters"] == 8
    assert got["worst_abs_csd"] == pytest.approx(0.0, abs=1e-12)
    assert got["exclusion"] == pytest.approx(0.99, abs=1e-12)


def test_representative_support_uses_only_individually_feasible_projects() -> None:
    """Catches relying on a project whose cost exceeds the PB budget."""
    from iclr_degeneracy import build_representative_support_instance

    instance, winners = build_representative_support_instance(
        n_per_group=100, supporters_per_group=1
    )

    assert winners == {"representative"}
    assert all(project.cost <= instance.budget for project in instance.projects.values())
    assert sum(instance.projects[pid].cost for pid in winners) <= instance.budget


def test_group_ledger_conditions_on_demographically_covered_voters() -> None:
    """Paper notation must not put uncovered ballots in the CSD denominator."""
    from cohorts import cumulative_share_deficit, group_outcome
    from parse_pb import PBInstance, Project, Vote

    instance = PBInstance(
        path="test://covered-ledger",
        meta={"budget": "9"},
        projects={"p": Project("p", 9.0, None)},
        votes=[
            Vote("m", ("p",), age=20, sex="M"),
            Vote("f", ("p",), age=20, sex="F"),
            Vote("uncovered", ("p",)),
        ],
    )

    outcome = group_outcome(instance, {"p"}, "age_sex")

    assert set(outcome) == {"age<25|M", "age<25|F"}
    for values in outcome.values():
        assert values["entitlement"] == pytest.approx(4.5)
        assert values["utility"] == pytest.approx(4.5)
    assert cumulative_share_deficit([outcome], "age<25|M") == pytest.approx(0.0)


def test_construction_rows_approach_total_exclusion() -> None:
    """Catches changing the asymptotic construction or its reporting unit."""
    from iclr_degeneracy import construction_rows

    rows = construction_rows([10, 25, 50, 100])

    assert [row["voters"] for row in rows] == [80, 200, 400, 800]
    assert [row["supporters"] for row in rows] == [8, 8, 8, 8]
    assert [row["exclusion"] for row in rows] == pytest.approx(
        [0.9, 0.96, 0.98, 0.99]
    )
    assert max(row["worst_abs_csd"] for row in rows) < 1e-12
