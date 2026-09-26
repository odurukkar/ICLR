"""Does the result survive redefining the groups?

Our formulation applies to any partition of the electorate, but the fitted
objective uses one partition: age bracket crossed with sex. This analysis asks
whether the measured finding is an artifact of that choice.

This re-scores the *same fitted policies* under coarser partitions of the same
voters -- age alone (4 groups) and sex alone (2 groups) -- against the same
baselines. The policies are not refitted: they were trained under age x sex, so
this is also a small transfer test across the definition of the objective.

Requires the scheme-aware feature cache. Keying those caches on instance path
alone silently returns features for whichever partition was computed first,
which would make this experiment quietly meaningless.

Writes results/iclr_scheme_robustness.csv.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

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
from iclr_outcome import cost_effective_weights, score_selector
from iclr_policy import linear_policy
from iclr_train import _load_series_data
from iclr_stats import exact_paired_sign_flip_pvalue

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_CSV = RESULTS / "iclr_scheme_robustness.csv"
SCHEMES: Tuple[str, ...] = ("age_sex", "age", "sex")
N_BOOT = 10_000


def _per_series(selector: SelectorPolicy, data, cfg: EnvConfig) -> Dict[str, float]:
    return {
        d.ref.key: rollout_selector(
            d.ref, selector, score_years=d.test_years, cfg=cfg, instances=d.all_years
        ).worst_csd
        for d in data
        if d.test_years
    }


def _boot(
    a: List[float],
    b: List[float],
    seed: int = 42,
    n_boot: int = N_BOOT,
) -> Tuple[float, float, int]:
    diff = np.array(a, dtype=float) - np.array(b, dtype=float)
    diff = diff[~np.isnan(diff)]
    # Keep the shared helper signature; this table reports no interval and
    # therefore needs no bootstrap draw.
    del seed, n_boot
    p = exact_paired_sign_flip_pvalue(diff)
    return float(diff.mean()), float(min(p, 1.0)), int((diff < 0).sum())


def main() -> None:
    index = build_series_index()
    data = _load_series_data(load_split("temporal_2022"), index)

    endow_w = np.array(
        json.loads((RESULTS / "iclr_train" / "run_main_seed42.json").read_text())["best_weights"]
    )
    front = sorted((RESULTS / "iclr_frontier").glob("frontier_outcome_*seed42.json"))
    pts = json.loads(front[0].read_text())["points"]
    outcome_w = np.array(min(pts, key=lambda p: abs(p["floor"] - 1.0))["weights"])

    rows: List[dict] = []
    for scheme in SCHEMES:
        cfg = EnvConfig(scheme=scheme)
        policies: List[Tuple[str, SelectorPolicy]] = [
            ("mes", endowment_selector(uniform_policy, cfg)),
            ("res-1.0", endowment_selector(res_policy(1.0), cfg)),
            ("greedy-cost", score_selector(cost_effective_weights(), scheme)),
            ("learned-endowment", endowment_selector(linear_policy(endow_w, scheme), cfg)),
            ("learned-outcome", score_selector(outcome_w, scheme)),
        ]
        scores = {name: _per_series(sel, data, cfg) for name, sel in policies}
        keys = sorted(scores["mes"])
        n_groups = len({
            c for d in data for y in d.test_years
            for c in _cohorts(d.all_years.get(y), scheme)
        })
        logger.info("--- scheme=%s (%d groups) ---", scheme, n_groups)
        for name in ("mes", "res-1.0", "greedy-cost", "learned-endowment", "learned-outcome"):
            mean_csd = float(np.nanmean([scores[name][k] for k in keys]))
            row = {"scheme": scheme, "n_groups": n_groups, "policy": name,
                   "worst_csd": round(mean_csd, 6)}
            if name.startswith("learned"):
                diff, p, wins = _boot(
                    [scores[name][k] for k in keys], [scores["mes"][k] for k in keys]
                )
                row.update({"vs_mes_diff": round(diff, 6), "vs_mes_p": round(p, 5),
                            "vs_mes_wins": wins, "n": len(keys)})
                logger.info("  %-18s %.4f   vs MES %+.4f (p=%.4f, %d/%d)",
                            name, mean_csd, diff, p, wins, len(keys))
            else:
                logger.info("  %-18s %.4f", name, mean_csd)
            rows.append(row)

    fields = ["scheme", "n_groups", "policy", "worst_csd", "vs_mes_diff",
              "vs_mes_p", "vs_mes_wins", "n"]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s", OUT_CSV)


def _cohorts(inst, scheme: str):
    if inst is None:
        return []
    from cohorts import cohort_of
    return {c for c in (cohort_of(v, scheme) for v in inst.votes) if c is not None}


if __name__ == "__main__":
    main()
