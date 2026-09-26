"""Training-only pipeline for the locked three-city policy comparison.

This module never evaluates the scored 2023+ suffix. It fits one parameter
vector on the `Split.train` view, using an equal-weight mean of city means and
a city-balanced welfare hinge relative to ordinary Equal Shares.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from iclr_cmaes import CMAESConfig, CMAESResult, minimize
from iclr_corpus import SeriesRef, Split, load_series
from iclr_env import (
    EnvConfig,
    EpisodeResult,
    SelectorPolicy,
    endowment_selector,
    rollout_selector,
    uniform_policy,
)
from iclr_outcome import (
    N_PROJECT_FEATURES,
    PROJECT_FEATURES,
    cost_effective_weights,
    score_selector,
)
from iclr_policy import FEATURE_NAMES, N_FEATURES, linear_policy, res_equivalent_weights
from iclr_multicity_protocol import (
    PABULIB_COMMIT,
    FIT_SOURCE_FILES,
    _sha256,
    load_canonical_index_from_manifest,
    load_frozen_split,
    verify_multicity_protocol_lock,
    write_json_artifact,
)
from iclr_priority_mes import fast_mes_with_endowments, priority_mes_selector
from iclr_train import SeriesData

EXECUTION_MODE = "serial"
PRIMARY_SEEDS: Tuple[int, ...] = (1, 2, 42)
ROOT = Path(__file__).resolve().parent.parent
RESULT_ROOT = ROOT / "results" / "iclr_multicity"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiCityTrainConfig:
    """Every prespecified hyperparameter that determines one fit."""

    split: str
    arm: str
    seed: int
    generations: int = 30
    popsize: int | None = None
    sigma0: float = 0.4
    bound: float = 10.0
    welfare_penalty: float = 2.0
    welfare_floor: float = 1.0


@dataclass(frozen=True)
class MultiCityArm:
    """One matched learned class and its exact baseline initialization."""

    name: str
    n_features: int
    feature_names: Tuple[str, ...]
    init_name: str
    initial: np.ndarray

    def selector(self, weights: Sequence[float], cfg: EnvConfig) -> SelectorPolicy:
        if self.name == "priority":
            return priority_mes_selector(
                weights, scheme=cfg.scheme, completion=cfg.completion
            )
        if self.name == "endowment":
            policy = linear_policy(weights, cfg.scheme)

            def _selector(inst, state):
                return fast_mes_with_endowments(
                    inst,
                    policy(inst, state),
                    completion=cfg.completion,
                )

            return _selector
        if self.name == "outcome":
            return score_selector(weights, scheme=cfg.scheme)
        raise ValueError(f"unknown multi-city arm: {self.name!r}")


def build_multicity_arm(name: str) -> MultiCityArm:
    """Return one of the three prespecified learned comparison classes."""

    if name == "priority":
        return MultiCityArm(
            name="priority",
            n_features=N_PROJECT_FEATURES,
            feature_names=PROJECT_FEATURES,
            init_name="mes",
            initial=np.zeros(N_PROJECT_FEATURES),
        )
    if name == "endowment":
        return MultiCityArm(
            name="endowment",
            n_features=N_FEATURES,
            feature_names=FEATURE_NAMES,
            init_name="res-1.0",
            initial=res_equivalent_weights(1.0),
        )
    if name == "outcome":
        return MultiCityArm(
            name="outcome",
            n_features=N_PROJECT_FEATURES,
            feature_names=PROJECT_FEATURES,
            init_name="greedy-cost",
            initial=cost_effective_weights(),
        )
    raise ValueError(f"unknown multi-city arm: {name!r}")


def _city(series: str) -> str:
    parts = series.split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else series


def city_balanced_metric(
    episodes: Sequence[EpisodeResult], field: str
) -> float:
    """Mean of city-level series means for one episode field."""

    by_city: Dict[str, list[float]] = defaultdict(list)
    for episode in episodes:
        value = getattr(episode, field)
        if value is not None and np.isfinite(float(value)):
            by_city[_city(episode.series)].append(float(value))
    if not by_city:
        raise ValueError("no scoreable training episodes")
    return float(np.mean([np.mean(values) for values in by_city.values()]))


def city_balanced_training_loss(
    episodes: Sequence[EpisodeResult],
    mes_city_welfare: Mapping[str, float],
    welfare_penalty: float,
    welfare_floor: float,
) -> float:
    """City-balanced CSD plus the prespecified city-balanced welfare hinge."""

    fairness = city_balanced_metric(episodes, "worst_csd")
    candidate_welfare: Dict[str, float] = defaultdict(float)
    for episode in episodes:
        if episode.worst_csd is not None:
            candidate_welfare[_city(episode.series)] += float(episode.welfare)

    hinges = []
    for city, welfare in sorted(candidate_welfare.items()):
        baseline = mes_city_welfare.get(city)
        if baseline is None or baseline <= 0:
            raise ValueError(f"missing MES welfare for {city}")
        ratio = welfare / float(baseline)
        hinges.append(max(0.0, welfare_floor - ratio))
    if not hinges:
        raise ValueError("no scoreable training episodes")
    return float(fairness + welfare_penalty * np.mean(hinges))


def fit_payload(
    *,
    config: MultiCityTrainConfig,
    arm: MultiCityArm,
    best_weights: Sequence[float],
    best_loss: float,
    n_evals: int,
    history: Sequence[Mapping[str, object]],
    training_summary: Mapping[str, object],
) -> Dict[str, object]:
    """Build a serialization payload containing training information only."""

    return {
        "schema_version": 1,
        "training_only": True,
        "execution_mode": EXECUTION_MODE,
        "config": asdict(config),
        "arm": {
            "name": arm.name,
            "feature_names": list(arm.feature_names),
            "init_name": arm.init_name,
            "initial": arm.initial.tolist(),
        },
        "result": {
            "best_weights": np.asarray(best_weights, dtype=float).tolist(),
            "best_loss": float(best_loss),
            "n_evals": int(n_evals),
            "history": [dict(row) for row in history],
        },
        "training_summary": dict(training_summary),
    }


def rollout_training(
    selector: SelectorPolicy,
    data: Sequence[SeriesData],
    cfg: EnvConfig,
) -> list[EpisodeResult]:
    """Replay only fitting years; rows with no fitting view are excluded."""

    return [
        rollout_selector(
            row.ref,
            selector,
            score_years=row.train_years,
            cfg=cfg,
            instances=row.train_only,
        )
        for row in data
        if row.train_years
    ]


def load_training_series_data(
    split: Split, index: Mapping[str, SeriesRef]
) -> list[SeriesData]:
    """Parse only frozen fit years; scored-city files are never opened here."""

    rows = []
    for key, fit_years in split.train:
        if not fit_years:
            continue
        ref = index[key]
        instances = load_series(ref, years=fit_years)
        missing = sorted(set(fit_years) - set(instances))
        if missing:
            raise RuntimeError(f"missing fit years for {key}: {missing}")
        rows.append(
            SeriesData(
                ref=ref,
                train_years=tuple(fit_years),
                test_years=(),
                train_only=dict(instances),
                all_years=dict(instances),
            )
        )
    return rows


def select_best_with_initial(
    initial: np.ndarray,
    initial_loss: float,
    result: CMAESResult,
) -> Tuple[np.ndarray, float, str]:
    """Retain the exact baseline when no sampled candidate improves on it."""

    if initial_loss <= result.best_f:
        return np.asarray(initial, dtype=float).copy(), float(initial_loss), "initial"
    return result.best_x.copy(), float(result.best_f), "cmaes"


def training_summary(
    episodes: Sequence[EpisodeResult],
    mes_city_welfare: Mapping[str, float],
    welfare_penalty: float,
    welfare_floor: float,
) -> Dict[str, object]:
    """Report each fitting city and the exact macro objective components."""

    grouped: Dict[str, list[EpisodeResult]] = defaultdict(list)
    for episode in episodes:
        if episode.worst_csd is not None:
            grouped[_city(episode.series)].append(episode)
    if not grouped:
        raise ValueError("no scoreable training episodes")
    cities: Dict[str, Dict[str, float]] = {}
    for city, rows in sorted(grouped.items()):
        baseline = mes_city_welfare.get(city)
        if baseline is None or baseline <= 0:
            raise ValueError(f"missing MES welfare for {city}")
        welfare = sum(float(row.welfare) for row in rows)
        cities[city] = {
            "n_series": len(rows),
            "worst_csd": float(np.mean([row.worst_csd for row in rows])),
            "welfare": welfare,
            "mes_welfare": float(baseline),
            "welfare_ratio": welfare / float(baseline),
        }
    return {
        "cities": cities,
        "city_macro_worst_csd": city_balanced_metric(episodes, "worst_csd"),
        "loss": city_balanced_training_loss(
            episodes,
            mes_city_welfare,
            welfare_penalty,
            welfare_floor,
        ),
    }


def fit_output_path(root: Path, config: MultiCityTrainConfig) -> Path:
    """Deterministic artifact path for one split/arm/seed fit."""

    return (
        Path(root)
        / "fits"
        / config.split
        / config.arm
        / f"seed-{config.seed}.json"
    )


def fit_one(
    config: MultiCityTrainConfig,
    data: Sequence[SeriesData],
    cfg: EnvConfig | None = None,
) -> Dict[str, object]:
    """Fit one arm on one frozen training view and return its artifact payload."""

    cfg = cfg or EnvConfig()
    arm = build_multicity_arm(config.arm)
    mes_episodes = rollout_training(
        endowment_selector(uniform_policy, cfg), data, cfg
    )
    mes_city_welfare: Dict[str, float] = defaultdict(float)
    for episode in mes_episodes:
        if episode.worst_csd is not None:
            mes_city_welfare[_city(episode.series)] += float(episode.welfare)

    def objective(weights: np.ndarray) -> float:
        episodes = rollout_training(arm.selector(weights, cfg), data, cfg)
        return city_balanced_training_loss(
            episodes,
            mes_city_welfare,
            config.welfare_penalty,
            config.welfare_floor,
        )

    initial_loss = objective(arm.initial)
    result = minimize(
        objective,
        arm.initial,
        CMAESConfig(
            sigma0=config.sigma0,
            popsize=config.popsize,
            generations=config.generations,
            seed=config.seed,
            bound=config.bound,
        ),
    )
    best_weights, best_loss, source = select_best_with_initial(
        arm.initial, initial_loss, result
    )
    best_episodes = rollout_training(arm.selector(best_weights, cfg), data, cfg)
    summary = training_summary(
        best_episodes,
        mes_city_welfare,
        config.welfare_penalty,
        config.welfare_floor,
    )
    payload = fit_payload(
        config=config,
        arm=arm,
        best_weights=best_weights,
        best_loss=best_loss,
        n_evals=result.n_evals + 1,
        history=result.history,
        training_summary=summary,
    )
    payload["result"]["selection_source"] = source
    payload["result"]["initial_loss"] = float(initial_loss)
    return payload


def parse_train_args(
    argv: Sequence[str] | None = None,
) -> Tuple[argparse.Namespace, MultiCityTrainConfig]:
    """Parse CLI options and materialize the exact run configuration."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--arm", choices=("priority", "endowment", "outcome"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--popsize", type=int)
    parser.add_argument("--sigma0", type=float, default=0.4)
    parser.add_argument("--bound", type=float, default=10.0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    generations = 1 if args.smoke else args.generations
    popsize = 4 if args.smoke else args.popsize
    return args, MultiCityTrainConfig(
        split=args.split,
        arm=args.arm,
        seed=args.seed,
        generations=generations,
        popsize=popsize,
        sigma0=args.sigma0,
        bound=args.bound,
    )


def authorize_fit(
    config: MultiCityTrainConfig,
    *,
    smoke: bool,
    result_root: Path,
    repo_root: Path,
) -> Mapping[str, object] | None:
    """Require an unchanged protocol lock for every non-smoke fit."""

    if smoke:
        return None
    lock_path = Path(result_root) / "protocol_lock.json"
    if not lock_path.is_file():
        raise RuntimeError(f"protocol lock is required before fitting: {lock_path}")
    payload = verify_multicity_protocol_lock(
        lock_path,
        repo_root,
        result_root,
        Path(repo_root) / "data" / "pb_multicity",
    )
    spec = {"split": config.split, "arm": config.arm, "seed": config.seed}
    if spec not in payload.get("fit_inventory", []):
        raise RuntimeError(f"fit is outside the locked inventory: {spec}")
    protocol = payload["protocol"]
    expected = {
        "generations": protocol["generations"],
        "bound": protocol["coefficient_bound"],
        "sigma0": protocol["sigma0"],
        "welfare_penalty": protocol["objective"]["welfare_penalty"],
        "welfare_floor": protocol["objective"]["welfare_floor_ratio"],
    }
    observed = {
        "generations": config.generations,
        "bound": config.bound,
        "sigma0": config.sigma0,
        "welfare_penalty": config.welfare_penalty,
        "welfare_floor": config.welfare_floor,
    }
    if observed != expected:
        raise RuntimeError(
            f"fit hyperparameters differ from protocol lock: {observed} != {expected}"
        )
    return payload


def write_immutable_fit(path: Path, payload: Mapping[str, object]) -> str:
    """Write once; identical reruns are allowed, divergent overwrites are not."""

    path = Path(path)
    canonical = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != canonical:
            raise RuntimeError(f"refusing to overwrite locked fit artifact: {path}")
        return _sha256(path)
    return write_json_artifact(path, payload)


def _source_hashes(
    lock_payload: Mapping[str, object] | None = None,
) -> Dict[str, str]:
    """Return current smoke hashes or canonical locked fit-source hashes."""

    source_dir = Path(__file__).resolve().parent
    if lock_payload is None:
        return {name: _sha256(source_dir / name) for name in FIT_SOURCE_FILES}
    tracked = lock_payload.get("tracked_files")
    if not isinstance(tracked, dict):
        raise RuntimeError("protocol lock has no canonical fit-source digests")
    hashes: Dict[str, str] = {}
    for name in FIT_SOURCE_FILES:
        row = tracked.get(f"source/{name}")
        digest = row.get("sha256") if isinstance(row, dict) else None
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise RuntimeError(
                f"protocol lock has no canonical fit-source digest for {name}"
            )
        hashes[name] = digest
    return hashes


def canonical_corpus_manifest_sha256(
    lock_payload: Mapping[str, object] | None,
    corpus_manifest: Path,
) -> str:
    """Return the locked manifest digest used as canonical fit provenance."""

    if lock_payload is None:
        return _sha256(Path(corpus_manifest))
    tracked = lock_payload.get("tracked_files")
    row = (
        tracked.get("artifact/corpus_manifest")
        if isinstance(tracked, dict)
        else None
    )
    digest = row.get("sha256") if isinstance(row, dict) else None
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError("protocol lock has no canonical corpus-manifest digest")
    return digest


def main(argv: Sequence[str] | None = None) -> None:
    """Fit one training-only artifact; scored suffixes remain unopened."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args, config = parse_train_args(argv)
    lock_payload = authorize_fit(
        config,
        smoke=args.smoke,
        result_root=RESULT_ROOT,
        repo_root=ROOT,
    )
    split_path = RESULT_ROOT / "splits" / f"{config.split}.json"
    corpus_manifest = RESULT_ROOT / "corpus_manifest.json"
    if not split_path.is_file() or not corpus_manifest.is_file():
        raise RuntimeError("frozen corpus and split artifacts must exist before fitting")
    index = load_canonical_index_from_manifest(
        corpus_manifest, ROOT / "data" / "pb_multicity"
    )
    split = load_frozen_split(split_path, index)
    data = load_training_series_data(split, index)
    payload = fit_one(config, data, EnvConfig())
    payload["provenance"] = {
        **(
            {"protocol_lock_sha256": _sha256(RESULT_ROOT / "protocol_lock.json")}
            if lock_payload is not None
            else {}
        ),
        "pabulib_commit": PABULIB_COMMIT,
        "corpus_manifest_sha256": canonical_corpus_manifest_sha256(
            lock_payload, corpus_manifest
        ),
        "split_sha256": _sha256(split_path),
        "source_sha256": _source_hashes(lock_payload),
    }
    if args.smoke:
        output = (
            RESULT_ROOT
            / "smoke"
            / config.split
            / config.arm
            / f"seed-{config.seed}.json"
        )
    else:
        output = fit_output_path(RESULT_ROOT, config)
    digest = write_immutable_fit(output, payload)
    logger.info("wrote training-only fit %s", output)
    print(f"output={output}")
    print(f"sha256={digest}")
    print(f"best_loss={payload['result']['best_loss']:.8f}")


if __name__ == "__main__":
    main()
