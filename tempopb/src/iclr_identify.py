"""Static-floor control: how much does the one-shot support floor explain?

The two learned classes differ in more than their allocation kernels. This
control asks whether a scalar support restriction alone repairs the unpenalized
direct policy's coverage failure.

This runs the discriminating experiment. Optimize the outcome family with **no
welfare penalty** but under the Equal Shares support floor of Proposition 2
imposed directly (|A(p)|/n >= c(p)/(kappa*b)). That gives coverage without
proportional payment.

The result measures how much a one-shot floor repairs; it cannot by itself
identify the full mechanism effect. The separate frozen-score intervention
adds iterative payment while holding learned parameters fixed.

Writes results/iclr_identification.csv.
"""
from __future__ import annotations
import csv, json, logging
from pathlib import Path
from typing import List
import numpy as np
from iclr_cmaes import CMAESConfig, minimize
from iclr_corpus import build_series_index, load_split
from iclr_env import EnvConfig, aggregate, endowment_selector, rollout_selector, uniform_policy
from iclr_outcome import N_PROJECT_FEATURES, cost_effective_weights, score_selector
from iclr_train import TrainConfig, _evaluate, _load_series_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "iclr_identification.csv"
# These are fixed diagnostic floors reported in the manuscript.  Do not
# key an outcome-policy control to the realized kappa of a separately fitted
# endowment policy: that statistic changes when the rollout analysis changes.
KAPPAS = (1.0, 2.0)

def main() -> None:
    cfg = EnvConfig()
    data = _load_series_data(load_split("temporal_2022"), build_series_index())
    mes = _evaluate(endowment_selector(uniform_policy, cfg), data, cfg, on_test=False)
    rows: List[dict] = []
    for kappa in KAPPAS:
        def objective(w: np.ndarray, k=kappa) -> float:
            sel = score_selector(w, support_floor=k)
            eps = [rollout_selector(d.ref, sel, score_years=d.train_years, cfg=cfg,
                                    instances=d.train_only) for d in data]
            st = aggregate(eps)
            return float(st["worst_csd"]) if st.get("n_series") else float("inf")
        res = minimize(objective, cost_effective_weights(),
                       CMAESConfig(sigma0=0.4, generations=25, seed=42))
        sel = score_selector(res.best_x, support_floor=kappa)
        tr = _evaluate(sel, data, cfg, on_test=False)
        te = _evaluate(sel, data, cfg, on_test=True)
        rows.append({"kappa": kappa, "welfare_floor": "none",
                     "train_worst_csd": round(tr["worst_csd"], 6),
                     "test_worst_csd": round(te["worst_csd"], 6),
                     "test_welfare": round(te["welfare"], 1),
                     "test_exclusion": round(te["exclusion"], 6),
                     "welfare_ratio_vs_mes": round(te["welfare"] / mes["welfare"], 4)})
        logger.info("kappa=%.2f -> test csd=%.4f welfare=%.0f excl=%.4f",
                    kappa, te["worst_csd"], te["welfare"], te["exclusion"])
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    logger.info("wrote %s", OUT)

if __name__ == "__main__":
    main()
