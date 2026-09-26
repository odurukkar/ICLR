"""Regression tests for disjoint train/test series splits."""

from __future__ import annotations

from pathlib import Path

from iclr_corpus import SeriesRef, Split
from iclr_env import EnvConfig
from iclr_train import (
    SeriesData,
    TrainConfig,
    _evaluate,
    _load_series_data,
    _objective,
)


def _ref(key: str, years: tuple[int, ...]) -> SeriesRef:
    return SeriesRef(
        key=key,
        years=years,
        paths=tuple(Path(f"{key}-{year}.pb") for year in years),
    )


def test_loader_includes_series_that_exist_only_in_test(monkeypatch) -> None:
    """A district-out fold must not silently drop every held-out district."""
    train_ref = _ref("city/train", (2019, 2020))
    test_ref = _ref("city/test", (2021, 2022))
    split = Split(
        name="district-out",
        train=((train_ref.key, train_ref.years),),
        test=((test_ref.key, test_ref.years),),
    )
    index = {train_ref.key: train_ref, test_ref.key: test_ref}

    monkeypatch.setattr(
        "iclr_train._parse_all",
        lambda ref: {year: object() for year in ref.years},
    )
    data = _load_series_data(split, index)

    assert [row.ref.key for row in data] == ["city/test", "city/train"]
    by_key = {row.ref.key: row for row in data}
    assert by_key["city/train"].train_years == (2019, 2020)
    assert by_key["city/train"].test_years == ()
    assert by_key["city/test"].train_years == ()
    assert by_key["city/test"].test_years == (2021, 2022)


def test_training_objective_excludes_test_only_series(monkeypatch) -> None:
    """Held-out districts must not enter the optimizer as empty episodes."""
    train_ref = _ref("city/train", (2019,))
    test_ref = _ref("city/test", (2020,))
    data = [
        SeriesData(train_ref, (2019,), (), {2019: object()}, {2019: object()}),
        SeriesData(test_ref, (), (2020,), {}, {2020: object()}),
    ]
    seen = []

    def fake_rollout(ref, selector, score_years, cfg, instances):
        seen.append((ref.key, tuple(score_years)))
        return object()

    monkeypatch.setattr("iclr_train.rollout_selector", fake_rollout)
    monkeypatch.setattr(
        "iclr_train.aggregate",
        lambda episodes: {
            "n_series": float(len(episodes)),
            "worst_csd": 0.2,
            "welfare": 1.0,
        },
    )

    loss = _objective(
        lambda inst, state: set(),
        data,
        EnvConfig(),
        TrainConfig(),
        mes_welfare=1.0,
    )

    assert loss == 0.2
    assert seen == [("city/train", (2019,))]


def test_train_evaluation_excludes_test_only_series(monkeypatch) -> None:
    """Train metrics must use the same nonempty training-series population."""
    train_ref = _ref("city/train", (2019,))
    test_ref = _ref("city/test", (2020,))
    data = [
        SeriesData(train_ref, (2019,), (), {2019: object()}, {2019: object()}),
        SeriesData(test_ref, (), (2020,), {}, {2020: object()}),
    ]
    seen = []

    def fake_rollout(ref, selector, score_years, cfg, instances):
        seen.append((ref.key, tuple(score_years)))
        return object()

    monkeypatch.setattr("iclr_train.rollout_selector", fake_rollout)
    monkeypatch.setattr(
        "iclr_train.aggregate",
        lambda episodes: {"n_series": float(len(episodes))},
    )

    stats = _evaluate(lambda inst, state: set(), data, EnvConfig(), on_test=False)

    assert stats["n_series"] == 1.0
    assert seen == [("city/train", (2019,))]
