"""Paired significance for the frontier's headline contrasts.

The frontier script reports means. A mean over 18 correlated Warsaw districts is
not evidence on its own, and "that improvement is within noise" is the single
most common reason results like these get rejected. This script re-evaluates
every frontier point per series on the held-out years, uses a paired percentile
bootstrap for confidence intervals, and tests the paired mean with an exact
sign-flip randomization distribution.

Reports the contrast, its 95% CI, an exact two-sided p-value, and the win count,
so a claim can be stated at exactly the strength the data supports and no more.

Writes results/iclr_significance.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from iclr_corpus import build_series_index, load_split
from iclr_env import (
    EnvConfig,
    SelectorPolicy,
    endowment_selector,
    res_policy,
    rollout_selector,
    uniform_policy,
)
from iclr_outcome import (
    cost_effective_weights,
    greedy_equivalent_weights,
    llmrule_card_selector,
    llmrule_cost_selector,
    score_selector,
)
from iclr_train import _load_series_data, build_arm
from iclr_stats import exact_paired_sign_flip_pvalue

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
FRONTIER_DIR = ROOT / "results" / "iclr_frontier"
OUT_CSV = ROOT / "results" / "iclr_significance.csv"
N_BOOT = 10_000


def _per_series(
    selector: SelectorPolicy, data, cfg: EnvConfig
) -> Dict[str, Optional[float]]:
    """Worst-cohort CSD on held-out years, one entry per series."""
    out: Dict[str, Optional[float]] = {}
    for d in data:
        if not d.test_years:
            continue
        episode = rollout_selector(
            d.ref, selector, score_years=d.test_years, cfg=cfg, instances=d.all_years
        )
        out[d.ref.key] = episode.worst_csd
    return out


def _paired_bootstrap(
    a: Sequence[float], b: Sequence[float], seed: int = 0, n_boot: int = N_BOOT
) -> Tuple[float, float, float, float, int]:
    """Return paired mean, bootstrap CI, exact sign-flip p, and wins."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[~np.isnan(diff)]
    n = diff.size
    if n == 0:
        return (float("nan"),) * 4 + (0,)
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, n, (n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = exact_paired_sign_flip_pvalue(diff)
    return float(diff.mean()), float(lo), float(hi), p, int((diff < 0).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", default="outcome")
    parser.add_argument("--split", default="temporal_2022")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path = FRONTIER_DIR / f"frontier_{args.arm}_{args.split}_seed{args.seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"run iclr_frontier.py first: {path}")
    frontier = json.loads(path.read_text())

    cfg = EnvConfig()
    arm = build_arm(args.arm)
    index = build_series_index()
    data = _load_series_data(load_split(args.split), index)

    baselines: Dict[str, SelectorPolicy] = {
        "greedy-count": score_selector(greedy_equivalent_weights()),
        "greedy-cost": score_selector(cost_effective_weights()),
        "llmrule-cost": llmrule_cost_selector(),
        "llmrule-card": llmrule_card_selector(),
        "mes": endowment_selector(uniform_policy, cfg),
        "res-1.0": endowment_selector(res_policy(1.0), cfg),
    }
    base_scores = {
        name: _per_series(sel, data, cfg) for name, sel in baselines.items()
    }
    keys = sorted(base_scores["mes"])
    logger.info("scored %d baselines on %d series", len(baselines), len(keys))

    rows: List[dict] = []
    for point in frontier["points"]:
        weights = np.array(point["weights"], dtype=float)
        learned = _per_series(arm.selector(weights, cfg), data, cfg)
        for name, scores in base_scores.items():
            a = [learned[k] for k in keys]
            b = [scores[k] for k in keys]
            mean, lo, hi, p, wins = _paired_bootstrap(a, b, seed=args.seed)
            rows.append(
                {
                    "arm": args.arm,
                    "floor": point["floor"],
                    "baseline": name,
                    "learned_mean": round(float(np.nanmean(a)), 6),
                    "baseline_mean": round(float(np.nanmean(b)), 6),
                    "diff": round(mean, 6),
                    "ci_lo": round(lo, 6),
                    "ci_hi": round(hi, 6),
                    "p_two_sided": round(p, 5),
                    "wins": wins,
                    "n": len(keys),
                }
            )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("=" * 86)
    logger.info(
        "%-8s %-13s %9s %9s %9s %20s %8s %6s",
        "floor", "baseline", "learned", "base", "diff", "95% CI", "p", "wins",
    )
    logger.info("-" * 86)
    for row in rows:
        logger.info(
            "%-8.2f %-13s %9.4f %9.4f %+9.4f  [%+.4f,%+.4f] %8.4f %4d/%d",
            row["floor"], row["baseline"], row["learned_mean"], row["baseline_mean"],
            row["diff"], row["ci_lo"], row["ci_hi"], row["p_two_sided"],
            row["wins"], row["n"],
        )
    logger.info("=" * 86)
    logger.info("negative diff = learned policy has LOWER (better) worst-cohort CSD")
    logger.info("wrote %s", OUT_CSV)


if __name__ == "__main__":
    main()
