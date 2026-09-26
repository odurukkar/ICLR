"""Feature ablation and per-series failure cases for the outcome arm.

The leave-one-feature-out analysis refits with feature k forced to zero and
reports the held-out change. The companion per-series output records every
district, including the four where the headline policy loses, so the aggregate
result can be audited rather than read in isolation.

Writes results/iclr_ablation.csv and results/iclr_per_series.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from iclr_cmaes import CMAESConfig, minimize
from iclr_corpus import build_series_index, load_split
from iclr_env import (
    EnvConfig,
    endowment_selector,
    res_policy,
    rollout_selector,
    uniform_policy,
)
from iclr_outcome import (
    PROJECT_FEATURES,
    cost_effective_weights,
    greedy_equivalent_weights,
    score_selector,
)
from iclr_train import TrainConfig, _evaluate, _load_series_data, _objective, build_arm

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FRONTIER_DIR = ROOT / "results" / "iclr_frontier"
ABLATION_CSV = ROOT / "results" / "iclr_ablation.csv"
PER_SERIES_CSV = ROOT / "results" / "iclr_per_series.csv"


def _masked_objective_factory(arm, data, env_cfg, train_cfg, mes_welfare, mask):
    """Objective with the masked feature pinned to zero weight."""

    def objective(weights: np.ndarray) -> float:
        w = np.asarray(weights, dtype=float).copy()
        w[mask] = 0.0
        return _objective(
            arm.selector(w, env_cfg), data, env_cfg, train_cfg, mes_welfare
        )

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="temporal_2022")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--floor", type=float, default=1.0)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--penalty", type=float, default=2.0)
    args = parser.parse_args()

    env_cfg = EnvConfig()
    arm = build_arm("outcome")
    index = build_series_index()
    data = _load_series_data(load_split(args.split), index)
    mes_train = _evaluate(
        endowment_selector(uniform_policy, env_cfg), data, env_cfg, on_test=False
    )
    train_cfg = TrainConfig(
        split=args.split, arm="outcome", seed=args.seed,
        generations=args.generations, welfare_penalty=args.penalty,
        welfare_floor=args.floor,
    )

    # ---- 1. per-series table at the headline operating point ----
    path = FRONTIER_DIR / f"frontier_outcome_{args.split}_seed{args.seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"run iclr_frontier.py first: {path}")
    frontier = json.loads(path.read_text())
    point = min(frontier["points"], key=lambda p: abs(p["floor"] - args.floor))
    best_w = np.array(point["weights"], dtype=float)
    logger.info("per-series table at floor=%.2f", point["floor"])

    rules = {
        "greedy-count": score_selector(greedy_equivalent_weights()),
        "greedy-cost": score_selector(cost_effective_weights()),
        "mes": endowment_selector(uniform_policy, env_cfg),
        "res-1.0": endowment_selector(res_policy(1.0), env_cfg),
        "learned": score_selector(best_w),
    }
    rows: List[dict] = []
    for d in data:
        if not d.test_years:
            continue
        row: Dict[str, object] = {
            "series": d.ref.key,
            "test_years": "|".join(str(y) for y in d.test_years),
        }
        for name, sel in rules.items():
            ep = rollout_selector(
                d.ref, sel, score_years=d.test_years, cfg=env_cfg,
                instances=d.all_years,
            )
            row[name] = round(ep.worst_csd, 6) if ep.worst_csd is not None else None
            if name == "learned":
                row["worst_cohort"] = ep.worst_cohort
        learned, mes = row.get("learned"), row.get("mes")
        row["learned_minus_mes"] = (
            round(learned - mes, 6) if learned is not None and mes is not None else None
        )
        row["learned_wins"] = int(row["learned_minus_mes"] < 0) if row["learned_minus_mes"] is not None else None
        rows.append(row)

    rows.sort(key=lambda r: (r["learned_minus_mes"] is None, r["learned_minus_mes"]))
    with PER_SERIES_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    losses = [r for r in rows if r["learned_wins"] == 0]
    logger.info(
        "per-series: %d/%d wins, %d losses -> %s",
        len(rows) - len(losses), len(rows), len(losses),
        ", ".join(r["series"].split("/")[-1] for r in losses) or "none",
    )
    logger.info("wrote %s", PER_SERIES_CSV)

    # ---- 2. leave-one-feature-out ablation ----
    ab_rows: List[dict] = []
    full_test = _evaluate(score_selector(best_w), data, env_cfg, on_test=True)
    ab_rows.append(
        {
            "ablated": "none (full model)",
            "test_worst_csd": round(full_test["worst_csd"], 6),
            "test_welfare": round(full_test["welfare"], 1),
            "delta_vs_full": 0.0,
        }
    )
    for k, name in enumerate(PROJECT_FEATURES):
        objective = _masked_objective_factory(
            arm, data, env_cfg, train_cfg, mes_train["welfare"], k
        )
        result = minimize(
            objective,
            cost_effective_weights(),
            CMAESConfig(sigma0=0.4, generations=args.generations, seed=args.seed),
        )
        w = result.best_x.copy()
        w[k] = 0.0
        stats = _evaluate(score_selector(w), data, env_cfg, on_test=True)
        delta = stats["worst_csd"] - full_test["worst_csd"]
        ab_rows.append(
            {
                "ablated": name,
                "test_worst_csd": round(stats["worst_csd"], 6),
                "test_welfare": round(stats["welfare"], 1),
                "delta_vs_full": round(delta, 6),
            }
        )
        logger.info(
            "ablate %-22s test worst_csd=%.4f (%+.4f vs full)",
            name, stats["worst_csd"], delta,
        )

    with ABLATION_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ab_rows[0]))
        writer.writeheader()
        writer.writerows(ab_rows)
    logger.info("wrote %s", ABLATION_CSV)


if __name__ == "__main__":
    main()
