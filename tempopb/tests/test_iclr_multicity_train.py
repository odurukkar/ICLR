"""City-balanced fitting protocol for the locked multi-city experiment."""

from __future__ import annotations

import hashlib
import numpy as np
import pytest

import iclr_multicity_train as train_module
from iclr_cmaes import CMAESResult
from iclr_corpus import SeriesRef, Split
from iclr_env import EnvConfig, EpisodeResult, RolloutState
from iclr_train import SeriesData
from iclr_multicity_train import (
    EXECUTION_MODE,
    MultiCityTrainConfig,
    build_multicity_arm,
    authorize_fit,
    city_balanced_metric,
    city_balanced_training_loss,
    fit_payload,
    fit_output_path,
    fit_one,
    parse_train_args,
    load_training_series_data,
    rollout_training,
    select_best_with_initial,
    training_summary,
    write_immutable_fit,
)
from pathlib import Path
from parse_pb import PBInstance, Project, Vote


def _episode(series: str, worst: float, welfare: float) -> EpisodeResult:
    return EpisodeResult(
        series=series,
        scored_years=(2022,),
        worst_csd=worst,
        worst_cohort="group",
        mean_csd=worst,
        welfare=welfare,
        cost_welfare=welfare,
        exclusion=0.0,
        years=(),
    )


def test_city_balanced_metric_is_mean_of_city_means_not_pooled_mean() -> None:
    episodes = [
        _episode("Poland/Warszawa/A", 0.0, 10.0),
        _episode("Poland/Warszawa/B", 0.0, 10.0),
        _episode("Poland/Łódź/C", 1.0, 10.0),
    ]

    assert city_balanced_metric(episodes, "worst_csd") == pytest.approx(0.5)
    assert np.mean([episode.worst_csd for episode in episodes]) == pytest.approx(
        1.0 / 3.0
    )


def test_welfare_hinge_is_balanced_across_cities() -> None:
    episodes = [
        _episode("Poland/Warszawa/A", 0.0, 10.0),
        _episode("Poland/Warszawa/B", 0.0, 10.0),
        _episode("Poland/Łódź/C", 1.0, 5.0),
    ]

    loss = city_balanced_training_loss(
        episodes,
        mes_city_welfare={"Poland/Warszawa": 20.0, "Poland/Łódź": 10.0},
        welfare_penalty=2.0,
        welfare_floor=1.0,
    )

    # Fairness = (0 + 1)/2 = .5. Penalty = 2 * mean(0, .5) = .5.
    assert loss == pytest.approx(1.0)


def test_missing_city_baseline_or_empty_episodes_is_rejected() -> None:
    with pytest.raises(ValueError, match="no scoreable training episodes"):
        city_balanced_training_loss([], {}, 2.0, 1.0)
    with pytest.raises(ValueError, match="missing MES welfare"):
        city_balanced_training_loss(
            [_episode("Poland/Gdynia/A", 0.2, 3.0)], {}, 2.0, 1.0
        )


@pytest.mark.parametrize(
    ("name", "n_features", "init_name"),
    (
        ("priority", 5, "mes"),
        ("endowment", 6, "res-1.0"),
        ("outcome", 5, "greedy-cost"),
    ),
)
def test_matched_arms_have_frozen_exact_baseline_initializations(
    name: str, n_features: int, init_name: str
) -> None:
    arm = build_multicity_arm(name)

    assert arm.n_features == n_features
    assert arm.init_name == init_name
    assert arm.initial.shape == (n_features,)
    if name == "priority":
        assert np.array_equal(arm.initial, np.zeros(5))


def test_endowment_arm_uses_verified_fast_mes_kernel(monkeypatch) -> None:
    inst = PBInstance(
        path="test://fast-endowment-arm",
        meta={"budget": "1"},
        projects={"a": Project("a", 1.0, None)},
        votes=[Vote("0", ("a",), age=20, sex="F")],
    )
    seen = []

    def fake_fast(observed_inst, endowments, completion):
        seen.append((observed_inst.path, tuple(endowments), completion))
        return {"a"}

    monkeypatch.setattr("iclr_multicity_train.fast_mes_with_endowments", fake_fast)
    arm = build_multicity_arm("endowment")
    winners = arm.selector(arm.initial, EnvConfig())(inst, RolloutState())

    assert winners == {"a"}
    assert seen == [(inst.path, (1.0,), True)]


def test_fit_payload_cannot_contain_heldout_metrics() -> None:
    config = MultiCityTrainConfig(split="temporal_2022", arm="priority", seed=1)
    arm = build_multicity_arm("priority")

    payload = fit_payload(
        config=config,
        arm=arm,
        best_weights=np.ones(arm.n_features),
        best_loss=0.1,
        n_evals=8,
        history=[{"generation": 0, "best_f": 0.1}],
        training_summary={"loss": 0.1},
    )

    serialized = repr(payload).casefold()
    assert payload["training_only"] is True
    assert "test" not in serialized
    assert "heldout" not in serialized
    assert payload["execution_mode"] == "serial"
    assert EXECUTION_MODE == "serial"


def test_exact_initial_baseline_is_retained_as_an_evaluated_candidate() -> None:
    result = CMAESResult(best_x=np.array([2.0]), best_f=0.3, n_evals=8)

    weights, loss, source = select_best_with_initial(
        np.array([0.0]), 0.2, result
    )

    assert np.array_equal(weights, np.array([0.0]))
    assert loss == 0.2
    assert source == "initial"

    weights, loss, source = select_best_with_initial(
        np.array([0.0]), 0.4, result
    )
    assert np.array_equal(weights, np.array([2.0]))
    assert loss == 0.3
    assert source == "cmaes"


def test_rollout_training_never_scores_rows_without_fit_years(monkeypatch) -> None:
    train_ref = SeriesRef("Poland/Test/train", (2022,), (Path("train.pb"),))
    score_ref = SeriesRef("Poland/Test/score", (2023,), (Path("score.pb"),))
    data = [
        SeriesData(train_ref, (2022,), (), {}, {}),
        SeriesData(score_ref, (), (2023,), {}, {}),
    ]
    seen = []

    def fake_rollout(ref, selector, score_years, cfg, instances):
        seen.append((ref.key, tuple(score_years)))
        return _episode(ref.key, 0.1, 1.0)

    monkeypatch.setattr("iclr_multicity_train.rollout_selector", fake_rollout)

    episodes = rollout_training(lambda inst, state: set(), data, EnvConfig())

    assert len(episodes) == 1
    assert seen == [("Poland/Test/train", (2022,))]


def test_training_loader_parses_only_frozen_fit_years(monkeypatch) -> None:
    train_ref = SeriesRef(
        "Poland/Warszawa/Wola",
        (2022, 2023),
        (Path("train.pb"), Path("score.pb")),
    )
    score_ref = SeriesRef(
        "Poland/Gdynia/Cisowa | large",
        (2022, 2023),
        (Path("other-train.pb"), Path("other-score.pb")),
    )
    split = Split(
        name="city_out_Poland_Gdynia",
        train=((train_ref.key, (2022,)),),
        test=((score_ref.key, (2023,)),),
    )
    seen = []

    def fake_load(ref, years):
        seen.append((ref.key, tuple(years)))
        return {year: object() for year in years}

    monkeypatch.setattr("iclr_multicity_train.load_series", fake_load)

    rows = load_training_series_data(
        split, {train_ref.key: train_ref, score_ref.key: score_ref}
    )

    assert seen == [(train_ref.key, (2022,))]
    assert len(rows) == 1
    assert rows[0].ref.key == train_ref.key
    assert rows[0].train_years == (2022,)
    assert rows[0].test_years == ()
    assert tuple(rows[0].all_years) == (2022,)


def test_training_summary_preserves_each_city_and_macro_balance() -> None:
    episodes = [
        _episode("Poland/Warszawa/A", 0.1, 10.0),
        _episode("Poland/Warszawa/B", 0.3, 10.0),
        _episode("Poland/Łódź/C", 0.2, 5.0),
    ]

    summary = training_summary(
        episodes,
        {"Poland/Warszawa": 20.0, "Poland/Łódź": 10.0},
        welfare_penalty=2.0,
        welfare_floor=1.0,
    )

    assert summary["cities"]["Poland/Warszawa"]["worst_csd"] == pytest.approx(0.2)
    assert summary["cities"]["Poland/Łódź"]["welfare_ratio"] == pytest.approx(0.5)
    assert summary["city_macro_worst_csd"] == pytest.approx(0.2)
    assert summary["loss"] == pytest.approx(0.7)


def test_fit_output_path_is_split_arm_seed_addressed(tmp_path: Path) -> None:
    config = MultiCityTrainConfig(
        split="city_out_Poland_Gdynia", arm="priority", seed=42
    )
    assert fit_output_path(tmp_path, config) == (
        tmp_path
        / "fits"
        / "city_out_Poland_Gdynia"
        / "priority"
        / "seed-42.json"
    )


def test_fit_one_evaluates_mes_and_initial_without_score_view(
    monkeypatch,
) -> None:
    ref = SeriesRef("Poland/Warszawa/Wola", (2022,), (Path("train.pb"),))
    data = [SeriesData(ref, (2022,), (), {}, {})]
    calls = []

    def fake_rollout(selector, observed_data, cfg):
        calls.append(tuple(row.train_years for row in observed_data))
        if len(calls) == 1:
            return [_episode(ref.key, 0.3, 10.0)]
        return [_episode(ref.key, 0.2, 10.0)]

    def fake_minimize(objective, x0, cfg):
        assert objective(np.asarray(x0)) == pytest.approx(0.2)
        return CMAESResult(
            best_x=np.ones_like(x0),
            best_f=0.25,
            history=[{"generation": 0, "best_f": 0.25}],
            n_evals=4,
        )

    monkeypatch.setattr("iclr_multicity_train.rollout_training", fake_rollout)
    monkeypatch.setattr("iclr_multicity_train.minimize", fake_minimize)

    payload = fit_one(
        MultiCityTrainConfig(
            split="temporal_2022", arm="priority", seed=1, generations=1
        ),
        data,
        EnvConfig(),
    )

    assert payload["result"]["selection_source"] == "initial"
    assert payload["result"]["n_evals"] == 5
    assert payload["training_only"] is True
    assert all(years == ((2022,),) for years in calls)


def test_smoke_cli_uses_one_generation_and_four_candidates() -> None:
    args, config = parse_train_args(
        [
            "--split",
            "temporal_2022",
            "--arm",
            "priority",
            "--seed",
            "1",
            "--smoke",
        ]
    )

    assert args.smoke is True
    assert config.generations == 1
    assert config.popsize == 4
    assert config.split == "temporal_2022"
    assert config.arm == "priority"


def test_smoke_cli_rejects_arbitrary_output_paths(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_train_args(
            [
                "--split",
                "temporal_2022",
                "--arm",
                "priority",
                "--seed",
                "1",
                "--smoke",
                "--output",
                str(tmp_path / "protocol_lock.json"),
            ]
        )


def test_primary_fit_requires_protocol_lock_but_smoke_does_not(tmp_path: Path) -> None:
    config = MultiCityTrainConfig(split="temporal_2022", arm="priority", seed=1)
    authorize_fit(config, smoke=True, result_root=tmp_path, repo_root=tmp_path)
    with pytest.raises(RuntimeError, match="protocol lock"):
        authorize_fit(config, smoke=False, result_root=tmp_path, repo_root=tmp_path)


def test_fit_provenance_keeps_the_locked_manifest_digest_after_redaction(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "corpus_manifest.json"
    manifest.write_text('{"destination":"data/pb_multicity"}\n')
    lock_payload = {
        "tracked_files": {
            "artifact/corpus_manifest": {
                "path": "results/iclr_multicity/corpus_manifest.json",
                "sha256": "a" * 64,
            }
        }
    }

    assert (
        train_module.canonical_corpus_manifest_sha256(lock_payload, manifest)
        == "a" * 64
    )
    assert train_module.canonical_corpus_manifest_sha256(None, manifest) != "a" * 64


def test_fit_provenance_keeps_locked_source_digests_after_runtime_amendments() -> None:
    locked = {
        name: hashlib.sha256(f"locked:{name}".encode()).hexdigest()
        for name in train_module.FIT_SOURCE_FILES
    }
    lock_payload = {
        "tracked_files": {
            f"source/{name}": {"sha256": digest}
            for name, digest in locked.items()
        }
    }

    assert train_module._source_hashes(lock_payload) == locked


def test_fit_provenance_rejects_missing_locked_source_digests() -> None:
    with pytest.raises(RuntimeError, match="canonical fit-source digest"):
        train_module._source_hashes({"tracked_files": {}})


@pytest.mark.parametrize(
    "lock_payload",
    (
        {},
        {"tracked_files": {}},
        {"tracked_files": {"artifact/corpus_manifest": {"sha256": "bad"}}},
    ),
)
def test_fit_provenance_rejects_a_missing_or_malformed_canonical_manifest_digest(
    tmp_path: Path,
    lock_payload: dict,
) -> None:
    manifest = tmp_path / "corpus_manifest.json"
    manifest.write_text("{}\n")

    with pytest.raises(RuntimeError, match="canonical corpus-manifest digest"):
        train_module.canonical_corpus_manifest_sha256(lock_payload, manifest)


def test_fit_artifact_is_idempotent_but_not_overwritable(tmp_path: Path) -> None:
    path = tmp_path / "fit.json"
    first = write_immutable_fit(path, {"a": 1})
    second = write_immutable_fit(path, {"a": 1})
    assert first == second
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        write_immutable_fit(path, {"a": 2})
