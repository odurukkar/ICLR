"""Regression tests for the frozen no-refit multi-city stress test."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from iclr_corpus import SeriesRef
from iclr_env import EpisodeResult


def test_multicity_transfer_module_exists() -> None:
    """The stress test must be a replayable artifact, not a one-off command."""
    assert importlib.util.find_spec("iclr_multicity_transfer") is not None


def test_stress_selection_includes_short_series_and_excludes_training_city() -> None:
    """A future min-run shortcut must not cherry-pick only longer transfers."""
    from iclr_multicity_transfer import select_stress_series

    index = {
        "Poland/Warszawa/A": SeriesRef(
            "Poland/Warszawa/A", (2022, 2023), (Path("w22.pb"), Path("w23.pb"))
        ),
        "Poland/Gdynia/one": SeriesRef(
            "Poland/Gdynia/one", (2024,), (Path("g24.pb"),)
        ),
        "Poland/Łódź/three": SeriesRef(
            "Poland/Łódź/three",
            (2023, 2024, 2025),
            (Path("l23.pb"), Path("l24.pb"), Path("l25.pb")),
        ),
    }

    selected = select_stress_series(index)

    assert [ref.key for ref in selected] == [
        "Poland/Gdynia/one",
        "Poland/Łódź/three",
    ]


def _episode(
    series: str,
    years: tuple[int, ...],
    worst_csd: float,
    welfare: float,
    exclusion: float,
) -> EpisodeResult:
    return EpisodeResult(
        series=series,
        scored_years=years,
        worst_csd=worst_csd,
        worst_cohort="g",
        mean_csd=worst_csd / 2,
        welfare=welfare,
        cost_welfare=2 * welfare,
        exclusion=exclusion,
        years=(),
    )


def test_scope_aggregation_uses_series_not_elections_as_analysis_unit() -> None:
    """A two-edition series must not receive twice the inferential weight."""
    from iclr_multicity_transfer import aggregate_scope_rows

    episodes = {
        "mes": [
            _episode("long", (2023, 2024), 0.1, 100.0, 0.2),
            _episode("short", (2024,), 0.5, 50.0, 0.6),
        ],
        "learned-endowment": [
            _episode("long", (2023, 2024), 0.05, 98.0, 0.1),
            _episode("short", (2024,), 0.35, 49.0, 0.5),
        ],
    }

    rows = {
        row["policy"]: row
        for row in aggregate_scope_rows("city", "Poland/Test", episodes)
    }

    assert rows["mes"] == {
        "row_type": "city",
        "scope": "Poland/Test",
        "policy": "mes",
        "n_series": 2,
        "n_elections": 3,
        "worst_csd": 0.3,
        "welfare": 150.0,
        "cost_welfare": 300.0,
        "exclusion": 0.4,
        "delta_vs_mes": 0.0,
        "welfare_ratio_vs_mes": 1.0,
        "exclusion_delta_vs_mes": 0.0,
    }
    assert rows["learned-endowment"]["worst_csd"] == 0.2
    assert rows["learned-endowment"]["delta_vs_mes"] == -0.1
    assert rows["learned-endowment"]["welfare_ratio_vs_mes"] == 147.0 / 150.0
    assert rows["learned-endowment"]["exclusion_delta_vs_mes"] == -0.1


def test_frozen_rows_use_series_metadata_year_and_deduplicate_files() -> None:
    """Filename years can disagree with parsed metadata, as in Lublin."""
    from iclr_multicity_transfer import frozen_instance_rows

    lublin = SeriesRef(
        "Poland/Lublin/hard", (2019,), (Path("Poland_Lublin_2020_hard.pb"),)
    )
    duplicate = SeriesRef(
        "Poland/Lublin/hard", (2019,), (Path("Poland_Lublin_2020_hard.pb"),)
    )

    assert frozen_instance_rows((lublin, duplicate)) == [
        ("Poland_Lublin_2020_hard.pb", "Poland/Lublin/hard", 2019)
    ]


def test_multicity_macros_preserve_city_breakdown_and_mixed_result() -> None:
    """The paper must not hide city ties behind an improved pooled mean."""
    from gen_iclr_numbers import _multicity_macros

    rows = []
    for row_type, scope, n_series, n_elections, values in (
        (
            "city",
            "Poland/Gdynia",
            10,
            13,
            {
                "mes": (0.20, 1.0),
                "learned-endowment": (0.20, 1.0),
                "learned-outcome-f1.00": (0.19, 0.97),
            },
        ),
        (
            "all",
            "all-non-Warsaw",
            17,
            21,
            {
                "mes": (0.21, 1.0),
                "learned-endowment": (0.20, 0.994),
                "learned-outcome-f1.00": (0.18, 0.91),
            },
        ),
    ):
        for policy, (csd, welfare_ratio) in values.items():
            rows.append(
                {
                    "row_type": row_type,
                    "scope": scope,
                    "policy": policy,
                    "n_series": str(n_series),
                    "n_elections": str(n_elections),
                    "worst_csd": str(csd),
                    "delta_vs_mes": str(csd - values["mes"][0]),
                    "welfare_ratio_vs_mes": str(welfare_ratio),
                    "exclusion_delta_vs_mes": "0.0",
                }
            )

    macros = _multicity_macros(rows)

    assert macros["MultiGdyniaEndowDiff"] == "+0.0000"
    assert macros["MultiAllSeriesN"] == "17"
    assert macros["MultiAllElectionN"] == "21"
    assert macros["MultiAllEndowGain"] == "0.0100"
    assert macros["MultiAllEndowWelfareCostPct"] == "0.6\\%"
    assert macros["MultiAllOutcomeGain"] == "0.0300"
