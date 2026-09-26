"""Fit either paper policy family by CMA-ES and score it on held-out years.

Training scores only the training years; evaluation warms the policy up over
those same years and then scores the held-out suffix, so a learned map is
judged on editions it never optimized against.

The default initial mean is an exact in-space hand-designed reference: RES(1)
for the endowment arm and greedy approvals-per-cost for the outcome arm. See
`iclr_verify_policy.py` and `iclr_verify_outcome.py` for the executable
containment checks. The CMA-ES result does not automatically retain its initial
mean as a candidate, so executable baselines are evaluated and reported
separately.

Usage:
    uv run python src/iclr_train.py --split temporal_2022 --generations 40
    uv run python src/iclr_train.py --smoke      # fast wiring check
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from iclr_cmaes import CMAESConfig, minimize
from iclr_corpus import Split, SeriesRef, build_series_index, load_split
from iclr_env import (
    EnvConfig,
    EpisodeResult,
    SelectorPolicy,
    aggregate,
    endowment_selector,
    res_policy,
    rollout_reference,
    rollout_selector,
    uniform_policy,
)
from iclr_outcome import (
    N_PROJECT_FEATURES,
    PROJECT_FEATURES,
    cost_effective_weights,
    greedy_equivalent_weights,
    llmrule_card_selector,
    llmrule_cost_selector,
    score_selector,
)
from iclr_policy import (
    FEATURE_NAMES,
    N_FEATURES,
    linear_policy,
    res_equivalent_weights,
)
from parse_pb import PBInstance

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "iclr_train"
BASELINE_LAMBDAS: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Arm:
    """One of the paper's two policy families, behind a common interface."""

    name: str
    n_features: int
    feature_names: Tuple[str, ...]
    inits: Dict[str, np.ndarray]

    def selector(self, weights: np.ndarray, cfg: EnvConfig) -> SelectorPolicy:
        if self.name == "endowment":
            return endowment_selector(linear_policy(weights), cfg)
        return score_selector(weights)


def build_arm(name: str) -> Arm:
    """Look up an arm by name.

    Raises:
        ValueError: On an unknown arm name.
    """
    if name == "endowment":
        return Arm(
            name="endowment",
            n_features=N_FEATURES,
            feature_names=FEATURE_NAMES,
            inits={
                "res": res_equivalent_weights(1.0),
                "zeros": np.zeros(N_FEATURES),
            },
        )
    if name == "outcome":
        return Arm(
            name="outcome",
            n_features=N_PROJECT_FEATURES,
            feature_names=PROJECT_FEATURES,
            inits={
                "res": cost_effective_weights(),   # cost-effectiveness baseline start
                "zeros": greedy_equivalent_weights(),
            },
        )
    raise ValueError(f"unknown arm: {name!r}")


@dataclass(frozen=True)
class TrainConfig:
    """Everything that determines a training run's result."""

    split: str = "temporal_2022"
    arm: str = "endowment"  # "endowment" | "outcome"
    seed: int = 42
    generations: int = 40
    popsize: Optional[int] = None
    sigma0: float = 0.4
    bound: float = 10.0
    # Legacy serialized key: ``res`` selects the arm-specific reference start
    # (RES(1) for endowment; approvals-per-cost for outcome), while ``zeros``
    # selects the other exact in-space reference.  Keep these values stable so
    # historical run artifacts remain replayable.
    init: str = "res"  # "res" | "zeros"
    welfare_penalty: float = 0.0
    # Legacy field/CLI name: this is the soft penalty target relative to MES
    # welfare on the same years, not a hard feasibility floor.
    welfare_floor: float = 1.0


@dataclass
class SeriesData:
    """Pre-parsed instances for one series, split into train and eval views."""

    ref: SeriesRef
    train_years: Tuple[int, ...]
    test_years: Tuple[int, ...]
    train_only: Dict[int, PBInstance]
    all_years: Dict[int, PBInstance]


def _load_series_data(split: Split, index: Dict[str, SeriesRef]) -> List[SeriesData]:
    """Parse every series once; training re-uses these objects for all evaluations."""
    train_map = dict(split.train)
    test_map = dict(split.test)
    data: List[SeriesData] = []
    for key in sorted(set(train_map) | set(test_map)):
        train_years = tuple(train_map.get(key, ()))
        test_years = tuple(test_map.get(key, ()))
        ref = index[key]
        all_years = {y: inst for y, inst in _parse_all(ref).items()}
        data.append(
            SeriesData(
                ref=ref,
                train_years=train_years,
                test_years=test_years,
                train_only={y: all_years[y] for y in train_years if y in all_years},
                all_years=all_years,
            )
        )
    logger.info("pre-parsed %d series", len(data))
    return data


def _parse_all(ref: SeriesRef) -> Dict[int, PBInstance]:
    from iclr_corpus import load_series  # local import keeps module import cheap

    return load_series(ref)


def _objective(
    selector: SelectorPolicy,
    data: Sequence[SeriesData],
    cfg: EnvConfig,
    train_cfg: TrainConfig,
    mes_welfare: float,
) -> float:
    """Mean worst-cohort CSD with an optional soft welfare-target penalty.

    Fairness alone is a degenerate objective: an optimizer is happy to wreck
    utilitarian welfare to shave a deficit. When ``welfare_penalty > 0``, the
    hinge term penalizes shortfall below ``welfare_floor`` but does not impose a
    hard constraint. Unpenalized fits leave it at zero and report welfare
    alongside the loss instead.
    """
    episodes = [
        rollout_selector(
            d.ref, selector, score_years=d.train_years, cfg=cfg, instances=d.train_only
        )
        for d in data
        if d.train_years
    ]
    stats = aggregate(episodes)
    if not stats.get("n_series"):
        return float("inf")
    loss = stats["worst_csd"]
    if train_cfg.welfare_penalty > 0 and mes_welfare > 0:
        ratio = stats["welfare"] / mes_welfare
        loss += train_cfg.welfare_penalty * max(0.0, train_cfg.welfare_floor - ratio)
    return float(loss)


def _evaluate(
    selector: SelectorPolicy,
    data: Sequence[SeriesData],
    cfg: EnvConfig,
    on_test: bool,
) -> Dict[str, float]:
    """Score a selector on train years, or warm up on them and score test years."""
    episodes: List[EpisodeResult] = []
    for d in data:
        if on_test:
            if not d.test_years:
                continue
            episodes.append(
                rollout_selector(
                    d.ref, selector, score_years=d.test_years, cfg=cfg,
                    instances=d.all_years,
                )
            )
        else:
            if not d.train_years:
                continue
            episodes.append(
                rollout_selector(
                    d.ref, selector, score_years=d.train_years, cfg=cfg,
                    instances=d.train_only,
                )
            )
    return aggregate(episodes)


def _baselines(data: Sequence[SeriesData], cfg: EnvConfig) -> Dict[str, Dict[str, float]]:
    """Reference points the learned map has to beat, on both views.

    Both arms are scored against the same set, so the frontier plot compares
    like with like: the endowment family's baselines (MES, RES) and the outcome
    family's (both greedies) appear in every run.
    """
    out: Dict[str, Dict[str, float]] = {}
    endowment_rules = [
        ("mes", uniform_policy),
        *[(f"res-{lam}", res_policy(lam)) for lam in BASELINE_LAMBDAS if lam > 0],
    ]
    for name, policy in endowment_rules:
        selector = endowment_selector(policy, cfg)
        out[f"{name}/train"] = _evaluate(selector, data, cfg, on_test=False)
        out[f"{name}/test"] = _evaluate(selector, data, cfg, on_test=True)

    for name, weights in (
        ("greedy-count", greedy_equivalent_weights()),
        ("greedy-cost", cost_effective_weights()),
    ):
        selector = score_selector(weights)
        out[f"{name}/train"] = _evaluate(selector, data, cfg, on_test=False)
        out[f"{name}/test"] = _evaluate(selector, data, cfg, on_test=True)

    for name, external_selector in (
        ("llmrule-cost", llmrule_cost_selector()),
        ("llmrule-card", llmrule_card_selector()),
    ):
        out[f"{name}/train"] = _evaluate(
            external_selector, data, cfg, on_test=False
        )
        out[f"{name}/test"] = _evaluate(
            external_selector, data, cfg, on_test=True
        )

    for name in ("historical", "greedy"):
        for view, on_test in (("train", False), ("test", True)):
            episodes = []
            for d in data:
                years = d.test_years if on_test else d.train_years
                if not years:
                    continue
                episodes.append(
                    rollout_reference(
                        d.ref, name, score_years=years, cfg=cfg,
                        instances=d.all_years if on_test else d.train_only,
                    )
                )
            out[f"{name}/{view}"] = aggregate(episodes)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="temporal_2022")
    parser.add_argument("--arm", choices=("endowment", "outcome"), default="endowment")
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--popsize", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma0", type=float, default=0.4)
    parser.add_argument(
        "--bound", type=float, default=10.0,
        help="symmetric box for weights; raise it to test whether a fit that "
             "railed at the boundary is an artifact of the box",
    )
    parser.add_argument(
        "--init", choices=("res", "zeros"), default="res",
        help="legacy arm-specific start key: 'res' is RES(1) for the endowment "
             "arm and greedy approvals-per-cost for the outcome arm; 'zeros' "
             "is uniform MES for endowment and greedy approval count for outcome",
    )
    parser.add_argument("--welfare-penalty", type=float, default=0.0)
    parser.add_argument(
        "--welfare-floor", type=float, default=1.0,
        help="legacy name for the soft welfare target relative to MES; this is "
             "used inside a hinge penalty, not as a hard constraint",
    )
    parser.add_argument("--smoke", action="store_true", help="3 generations, 4 series")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    train_cfg = TrainConfig(
        split=args.split,
        arm=args.arm,
        seed=args.seed,
        generations=3 if args.smoke else args.generations,
        popsize=4 if args.smoke else args.popsize,
        sigma0=args.sigma0,
        bound=args.bound,
        init=args.init,
        welfare_penalty=args.welfare_penalty,
        welfare_floor=args.welfare_floor,
    )
    env_cfg = EnvConfig()
    arm = build_arm(train_cfg.arm)

    index = build_series_index()
    split = load_split(train_cfg.split)
    data = _load_series_data(split, index)
    if args.smoke:
        data = data[:4]
        logger.warning("SMOKE MODE: %d series, %d generations", len(data), train_cfg.generations)

    mes_train = _evaluate(
        endowment_selector(uniform_policy, env_cfg), data, env_cfg, on_test=False
    )
    logger.info(
        "arm=%s | MES on train years: worst_csd=%.4f welfare=%.1f",
        arm.name, mes_train["worst_csd"], mes_train["welfare"],
    )

    x0 = arm.inits[train_cfg.init]
    t0 = time.time()
    n_evals = {"count": 0}

    def objective(weights: np.ndarray) -> float:
        n_evals["count"] += 1
        return _objective(
            arm.selector(weights, env_cfg), data, env_cfg, train_cfg, mes_train["welfare"]
        )

    logger.info("init %s -> %s (loss %.6f)", train_cfg.init, x0, objective(x0))

    result = minimize(
        objective,
        x0,
        CMAESConfig(
            sigma0=train_cfg.sigma0,
            popsize=train_cfg.popsize,
            generations=train_cfg.generations,
            seed=train_cfg.seed,
            bound=train_cfg.bound,
        ),
    )
    railed = [
        name for name, w in zip(arm.feature_names, result.best_x)
        if abs(abs(w) - train_cfg.bound) < 1e-6
    ]
    if railed:
        logger.warning(
            "weights railed at the +/-%.1f box: %s -- rerun with a larger "
            "--bound before trusting this fit", train_cfg.bound, ", ".join(railed),
        )
    elapsed = time.time() - t0

    best_selector = arm.selector(result.best_x, env_cfg)
    learned_train = _evaluate(best_selector, data, env_cfg, on_test=False)
    learned_test = _evaluate(best_selector, data, env_cfg, on_test=True)
    baselines = _baselines(data, env_cfg)

    payload = {
        "config": asdict(train_cfg),
        "smoke": args.smoke,
        "n_series": len(data),
        "n_train_series": sum(bool(d.train_years) for d in data),
        "n_test_series": sum(bool(d.test_years) for d in data),
        "elapsed_sec": round(elapsed, 1),
        "n_objective_evals": n_evals["count"],
        "feature_names": list(arm.feature_names),
        "best_weights": result.best_x.tolist(),
        "best_train_loss": result.best_f,
        "learned": {"train": learned_train, "test": learned_test},
        "baselines": baselines,
        "history": result.history,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or (
        "smoke" if args.smoke
        else f"{arm.name}_{train_cfg.split}_seed{train_cfg.seed}"
    )
    out_path = OUT_DIR / f"run_{tag}.json"
    out_path.write_text(json.dumps(payload, indent=2))

    logger.info("=" * 68)
    logger.info(
        "arm=%s weights: %s",
        arm.name, dict(zip(arm.feature_names, np.round(result.best_x, 4))),
    )
    logger.info("%-14s %10s %14s", "policy", "train", "test(held-out)")
    for label, key in [
        ("greedy-count", "greedy-count"),
        ("greedy-cost", "greedy-cost"),
        ("MES", "mes"),
        ("RES(1.0)", "res-1.0"),
    ]:
        logger.info(
            "%-14s %10.4f %14.4f",
            label,
            baselines[f"{key}/train"].get("worst_csd", float("nan")),
            baselines[f"{key}/test"].get("worst_csd", float("nan")),
        )
    logger.info(
        "%-14s %10.4f %14.4f", "LEARNED",
        learned_train.get("worst_csd", float("nan")),
        learned_test.get("worst_csd", float("nan")),
    )
    logger.info("wrote %s (%.0fs, %d evals)", out_path, elapsed, n_evals["count"])


if __name__ == "__main__":
    main()
