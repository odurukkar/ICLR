"""Paper generators must consume only the declared temporal seed artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import gen_iclr_appendix
import gen_iclr_numbers
import iclr_fig_dynamics
import iclr_fig_forest


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _endowment_run(
    value: float,
    elapsed_sec: float,
    *,
    seed: int = 42,
    split: str = "temporal_2022",
) -> dict[str, object]:
    return {
        "config": {"split": split, "seed": seed},
        "best_weights": [1.25, -0.5],
        "elapsed_sec": elapsed_sec,
        "learned": {"test": {"worst_csd": value}},
    }


def _outcome_frontier(
    seed: int,
    *,
    weights: list[float] | None = None,
    elapsed_sec: float = 60.0,
) -> dict[str, object]:
    return {
        "split": "temporal_2022",
        "seed": seed,
        "points": [
            {
                "floor": 1.0,
                "weights": weights or [float(seed), -float(seed)],
                "test_worst_csd": seed / 100.0,
                "elapsed_sec": elapsed_sec,
            }
        ],
    }


def _declared_endowment_paths(root: Path) -> tuple[Path, ...]:
    return (
        root / "iclr_train" / "run_main_seed42.json",
        root / "iclr_train" / "run_endow_seed1.json",
        root / "iclr_train" / "run_endow_seed2.json",
    )


def _declared_frontier_paths(root: Path) -> tuple[Path, ...]:
    return tuple(
        root
        / "iclr_frontier"
        / f"frontier_outcome_temporal_2022_seed{seed}.json"
        for seed in (1, 2, 3, 42)
    )


def test_endowment_seed_summary_ignores_wildcard_stale_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _declared_endowment_paths(tmp_path)
    for path, seed, value in zip(
        paths, (42, 1, 2), (0.16, 0.13, 0.15), strict=True
    ):
        _write_json(path, _endowment_run(value, 120.0, seed=seed))
    (tmp_path / "iclr_train" / "run_endow_seed99.json").write_text("not json")
    monkeypatch.setattr(
        gen_iclr_numbers,
        "ENDOWMENT_SEED_PATHS",
        paths,
        raising=False,
    )

    assert gen_iclr_numbers._endowment_seed_summary_macros() == {
        "EndowNumSeeds": "3",
        "EndowSeedLo": "0.1300",
        "EndowSeedHi": "0.1600",
        "EndowSeedSpread": "0.0300",
    }


def test_endowment_seed_summary_fails_closed_when_a_declared_run_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _declared_endowment_paths(tmp_path)
    _write_json(paths[0], _endowment_run(0.16, 120.0))
    _write_json(paths[1], _endowment_run(0.13, 120.0, seed=1))
    _write_json(
        tmp_path / "iclr_train" / "run_endow_seed99.json",
        _endowment_run(0.01, 1.0, seed=99),
    )
    monkeypatch.setattr(
        gen_iclr_numbers,
        "ENDOWMENT_SEED_PATHS",
        paths,
        raising=False,
    )

    with pytest.raises(
        FileNotFoundError,
        match=r"required endowment seed artifacts missing: .*run_endow_seed2\.json",
    ):
        gen_iclr_numbers._endowment_seed_summary_macros()


def test_runtime_macros_ignore_stale_seed_and_split_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    endowment_paths = _declared_endowment_paths(tmp_path)
    for path, seed, elapsed in zip(
        endowment_paths, (42, 1, 2), (120.0, 240.0, 360.0), strict=True
    ):
        _write_json(path, _endowment_run(0.1, elapsed, seed=seed))
    frontier_paths = _declared_frontier_paths(tmp_path)
    for path, seed, elapsed in zip(
        frontier_paths,
        (1, 2, 3, 42),
        (60.0, 120.0, 180.0, 240.0),
        strict=True,
    ):
        _write_json(path, _outcome_frontier(seed, elapsed_sec=elapsed))

    (tmp_path / "iclr_train" / "run_endow_seed99.json").write_text("not json")
    (
        tmp_path
        / "iclr_frontier"
        / "frontier_outcome_temporal_2022_seed99.json"
    ).write_text("not json")
    (
        tmp_path
        / "iclr_frontier"
        / "frontier_outcome_district_seed42.json"
    ).write_text("not json")

    monkeypatch.setattr(gen_iclr_numbers, "RESULTS", tmp_path)
    monkeypatch.setattr(
        gen_iclr_numbers,
        "ENDOWMENT_SEED_PATHS",
        endowment_paths,
        raising=False,
    )
    monkeypatch.setattr(gen_iclr_numbers, "FRONTIER_SEED_PATHS", frontier_paths)

    assert gen_iclr_numbers._runtime_macros() == {
        "EndowRuntimeLoMin": "2",
        "EndowRuntimeHiMin": "6",
        "OutcomeRuntimeLoMin": "1",
        "OutcomeRuntimeHiMin": "4",
    }


@pytest.mark.parametrize("module", (iclr_fig_forest, iclr_fig_dynamics))
def test_figure_policy_weights_ignore_stale_split_frontiers(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    main_run = tmp_path / "iclr_train" / "run_main_seed42.json"
    canonical = (
        tmp_path
        / "iclr_frontier"
        / "frontier_outcome_temporal_2022_seed42.json"
    )
    stale = (
        tmp_path
        / "iclr_frontier"
        / "frontier_outcome_district_seed42.json"
    )
    _write_json(main_run, _endowment_run(0.1, 120.0))
    _write_json(canonical, _outcome_frontier(42, weights=[4.2, -4.2]))
    stale.write_text("not json")

    monkeypatch.setattr(module, "RESULTS", tmp_path)
    monkeypatch.setattr(gen_iclr_numbers, "PRIMARY_FRONTIER", canonical)

    endowment_weights, outcome_weights = module._load_policy_weights()

    assert np.array_equal(endowment_weights, np.array([1.25, -0.5]))
    assert np.array_equal(outcome_weights, np.array([4.2, -4.2]))


@pytest.mark.parametrize("module", (iclr_fig_forest, iclr_fig_dynamics))
def test_figure_policy_weights_fail_closed_without_the_declared_frontier(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    main_run = tmp_path / "iclr_train" / "run_main_seed42.json"
    missing = (
        tmp_path
        / "iclr_frontier"
        / "frontier_outcome_temporal_2022_seed42.json"
    )
    stale = (
        tmp_path
        / "iclr_frontier"
        / "frontier_outcome_district_seed42.json"
    )
    _write_json(main_run, _endowment_run(0.1, 120.0))
    _write_json(stale, _outcome_frontier(42, weights=[9.9]))

    monkeypatch.setattr(module, "RESULTS", tmp_path)
    monkeypatch.setattr(gen_iclr_numbers, "PRIMARY_FRONTIER", missing)

    with pytest.raises(
        FileNotFoundError,
        match=r"required primary frontier artifact missing: .*seed42\.json",
    ):
        module._load_policy_weights()


def test_appendix_seed_table_ignores_stale_split_frontiers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _declared_frontier_paths(tmp_path)
    for path, seed in zip(paths, (1, 2, 3, 42), strict=True):
        _write_json(path, _outcome_frontier(seed))
    (
        tmp_path
        / "iclr_frontier"
        / "frontier_outcome_district_seed99.json"
    ).write_text("not json")

    monkeypatch.setattr(gen_iclr_appendix, "RESULTS", tmp_path)
    monkeypatch.setattr(gen_iclr_numbers, "FRONTIER_SEED_PATHS", paths)

    table = gen_iclr_appendix._seed_table()

    assert "seed 1 & seed 2 & seed 3 & seed 42" in table
    assert "seed 99" not in table


def test_appendix_seed_table_fails_closed_when_a_declared_frontier_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _declared_frontier_paths(tmp_path)
    for path, seed in zip(paths[:3], (1, 2, 3), strict=True):
        _write_json(path, _outcome_frontier(seed))
    _write_json(
        tmp_path
        / "iclr_frontier"
        / "frontier_outcome_district_seed42.json",
        _outcome_frontier(42),
    )

    monkeypatch.setattr(gen_iclr_appendix, "RESULTS", tmp_path)
    monkeypatch.setattr(gen_iclr_numbers, "FRONTIER_SEED_PATHS", paths)

    with pytest.raises(
        FileNotFoundError,
        match=r"required temporal-2022 frontier seed artifacts missing: .*seed42\.json",
    ):
        gen_iclr_appendix._seed_table()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("split", "district", "expected split temporal_2022"),
        ("seed", 99, "expected seed 42"),
    ),
)
def test_primary_frontier_rejects_mismatched_internal_metadata(
    field: str,
    value: object,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    canonical = (
        tmp_path
        / "iclr_frontier"
        / "frontier_outcome_temporal_2022_seed42.json"
    )
    payload = _outcome_frontier(42)
    payload[field] = value
    _write_json(canonical, payload)
    monkeypatch.setattr(gen_iclr_numbers, "PRIMARY_FRONTIER", canonical)

    with pytest.raises(ValueError, match=message):
        gen_iclr_numbers._load_primary_frontier()


def test_frontier_seed_sweep_rejects_payload_from_the_wrong_seed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _declared_frontier_paths(tmp_path)
    for path, seed in zip(paths, (1, 2, 3, 42), strict=True):
        _write_json(path, _outcome_frontier(seed))
    _write_json(paths[1], _outcome_frontier(99))
    monkeypatch.setattr(gen_iclr_numbers, "FRONTIER_SEED_PATHS", paths)

    with pytest.raises(ValueError, match=r"seed2\.json: expected seed 2, found 99"):
        gen_iclr_numbers._load_frontier_seed_sweep()


@pytest.mark.parametrize(
    ("split", "seed", "message"),
    (
        ("district", 1, "expected split temporal_2022"),
        ("temporal_2022", 99, "expected seed 1"),
    ),
)
def test_endowment_seed_loader_rejects_mismatched_internal_metadata(
    split: str,
    seed: int,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _declared_endowment_paths(tmp_path)
    _write_json(paths[0], _endowment_run(0.16, 120.0, seed=42))
    _write_json(paths[1], _endowment_run(0.13, 120.0, seed=seed, split=split))
    _write_json(paths[2], _endowment_run(0.15, 120.0, seed=2))
    monkeypatch.setattr(gen_iclr_numbers, "ENDOWMENT_SEED_PATHS", paths)

    with pytest.raises(ValueError, match=message):
        gen_iclr_numbers._load_endowment_seed_runs()


@pytest.mark.parametrize("floors", ([0.95], [1.0, 1.0]))
def test_exact_frontier_point_rejects_missing_or_duplicate_floor(
    floors: list[float],
) -> None:
    frontier = {"points": [{"floor": floor} for floor in floors]}

    with pytest.raises(
        ValueError,
        match=rf"expected exactly one frontier point at floor 1\.00; found {len(floors) if floors == [1.0, 1.0] else 0}",
    ):
        gen_iclr_numbers._exact_frontier_point(frontier, 1.0)


@pytest.mark.parametrize("module", (iclr_fig_forest, iclr_fig_dynamics))
def test_figure_policy_weights_require_the_exact_headline_floor(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    main_run = tmp_path / "iclr_train" / "run_main_seed42.json"
    canonical = (
        tmp_path
        / "iclr_frontier"
        / "frontier_outcome_temporal_2022_seed42.json"
    )
    payload = _outcome_frontier(42)
    payload["points"][0]["floor"] = 0.99
    _write_json(main_run, _endowment_run(0.1, 120.0))
    _write_json(canonical, payload)

    monkeypatch.setattr(module, "RESULTS", tmp_path)
    monkeypatch.setattr(gen_iclr_numbers, "PRIMARY_FRONTIER", canonical)

    with pytest.raises(
        ValueError,
        match=r"expected exactly one frontier point at floor 1\.00; found 0",
    ):
        module._load_policy_weights()
