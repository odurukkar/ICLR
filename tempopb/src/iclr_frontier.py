"""Trace the fairness-welfare frontier for one policy arm.

An unconstrained fairness objective finds a degenerate corner: it funds
demographically balanced projects regardless of how many voters want them,
driving the deficit toward zero while welfare falls below the deployed rule.
The interesting object is therefore not a single learned policy but the curve
swept out as the soft welfare target changes.

Each grid point re-runs CMA-ES with

    loss = worst_cohort_CSD + penalty * max(0, target - welfare / welfare_MES)

so a point is a soft-penalty fit targeting `floor` times the welfare Equal Shares
delivers on the same years. The target is not a hard feasibility constraint.
The serialized result key and CLI option retain the legacy name ``floor`` for
compatibility with the frozen artifacts.
Hand-designed rules are plotted in the same plane as references.

Usage:
    uv run python src/iclr_frontier.py --arm outcome --generations 25
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from iclr_cmaes import CMAESConfig, minimize
from iclr_corpus import build_series_index, load_split
from iclr_env import EnvConfig, endowment_selector, uniform_policy
from iclr_train import (
    TrainConfig,
    _baselines,
    _evaluate,
    _load_series_data,
    _objective,
    build_arm,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "iclr_frontier"
DEFAULT_FLOORS: Tuple[float, ...] = (0.0, 0.85, 0.95, 1.0, 1.02, 1.05)
PENALTY = 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("endowment", "outcome"), default="outcome")
    parser.add_argument("--split", default="temporal_2022")
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--penalty", type=float, default=PENALTY)
    parser.add_argument(
        "--floors", type=float, nargs="*", default=list(DEFAULT_FLOORS)
    )
    args = parser.parse_args()

    env_cfg = EnvConfig()
    arm = build_arm(args.arm)
    index = build_series_index()
    split = load_split(args.split)
    data = _load_series_data(split, index)

    mes_train = _evaluate(
        endowment_selector(uniform_policy, env_cfg), data, env_cfg, on_test=False
    )
    baselines = _baselines(data, env_cfg)
    logger.info(
        "arm=%s split=%s series=%d MES train welfare=%.0f",
        arm.name, args.split, len(data), mes_train["welfare"],
    )

    rows: List[dict] = []
    points: List[dict] = []
    for floor in args.floors:
        train_cfg = TrainConfig(
            split=args.split,
            arm=arm.name,
            seed=args.seed,
            generations=args.generations,
            sigma0=0.4,
            init="res",
            welfare_penalty=0.0 if floor <= 0 else args.penalty,
            welfare_floor=floor,
        )
        t0 = time.time()

        def objective(weights: np.ndarray) -> float:
            return _objective(
                arm.selector(weights, env_cfg),
                data, env_cfg, train_cfg, mes_train["welfare"],
            )

        result = minimize(
            objective,
            arm.inits["res"],
            CMAESConfig(
                sigma0=0.4, generations=args.generations, seed=args.seed
            ),
        )
        selector = arm.selector(result.best_x, env_cfg)
        train_stats = _evaluate(selector, data, env_cfg, on_test=False)
        test_stats = _evaluate(selector, data, env_cfg, on_test=True)
        elapsed = time.time() - t0

        row = {
            "arm": arm.name,
            "floor": floor,
            "penalty": train_cfg.welfare_penalty,
            "train_worst_csd": round(train_stats["worst_csd"], 6),
            "train_welfare_ratio": round(
                train_stats["welfare"] / mes_train["welfare"], 6
            ),
            "test_worst_csd": round(test_stats["worst_csd"], 6),
            "test_mean_csd": round(test_stats["mean_csd"], 6),
            "test_welfare": round(test_stats["welfare"], 1),
            "test_exclusion": round(test_stats["exclusion"], 6),
            "elapsed_sec": round(elapsed, 1),
        }
        rows.append(row)
        points.append({**row, "weights": result.best_x.tolist()})
        logger.info(
            "floor=%.2f -> test worst_csd=%.4f welfare=%.0f exclusion=%.4f (%.0fs)",
            floor, test_stats["worst_csd"], test_stats["welfare"],
            test_stats["exclusion"], elapsed,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{arm.name}_{args.split}_seed{args.seed}"
    with (OUT_DIR / f"frontier_{stem}.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT_DIR / f"frontier_{stem}.json").write_text(
        json.dumps(
            {
                "arm": arm.name,
                "split": args.split,
                "seed": args.seed,
                "penalty": args.penalty,
                "feature_names": list(arm.feature_names),
                "mes_train_welfare": mes_train["welfare"],
                "baselines": baselines,
                "points": points,
            },
            indent=2,
        )
    )
    logger.info("wrote %s/frontier_%s.{csv,json}", OUT_DIR, stem)


if __name__ == "__main__":
    main()
