"""History-free control: does the *history* channel earn its place?

The worst group is the same demographic cell in 13 of 18 series, so the learned
map may be acting mainly as a fixed per-group tilt. RES(lambda) does not settle
this: it tilts by *realized*
deficit, so it is a history rule, not a group-identity rule.

The discriminating control is the endowment family restricted to its
history-independent features -- group size, ballot length, cost focus, approval
overlap -- with both deficit features forced to zero. If that recovers most of
the improvement, the temporal framing is weakened and should be softened.

Writes results/iclr_static_tilt.csv.
"""
from __future__ import annotations
import csv, json, logging
from pathlib import Path
from typing import List
import numpy as np
from iclr_cmaes import CMAESConfig, minimize
from iclr_corpus import build_series_index, load_split
from iclr_env import EnvConfig, aggregate, endowment_selector, rollout_selector, uniform_policy
from iclr_policy import FEATURE_NAMES, N_FEATURES, linear_policy, res_equivalent_weights
from iclr_train import _evaluate, _load_series_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "iclr_static_tilt.csv"
HISTORY_IDX = (0, 1)   # deficit_per_capita, deficit_normalized

def main() -> None:
    cfg = EnvConfig()
    data = _load_series_data(load_split("temporal_2022"), build_series_index())
    mask = np.ones(N_FEATURES); mask[list(HISTORY_IDX)] = 0.0
    logger.info("static features: %s",
                [n for i, n in enumerate(FEATURE_NAMES) if mask[i]])

    def objective(w: np.ndarray) -> float:
        sel = endowment_selector(linear_policy(np.asarray(w) * mask), cfg)
        eps = [rollout_selector(d.ref, sel, score_years=d.train_years, cfg=cfg,
                                instances=d.train_only) for d in data]
        st = aggregate(eps)
        return float(st["worst_csd"]) if st.get("n_series") else float("inf")

    res = minimize(objective, np.zeros(N_FEATURES),
                   CMAESConfig(sigma0=0.4, generations=30, seed=42))
    w_static = res.best_x * mask
    rows: List[dict] = []
    for name, w in (("static-only (no history)", w_static),
                    ("MES", np.zeros(N_FEATURES)),
                    ("RES(1.0)", res_equivalent_weights(1.0))):
        te = _evaluate(endowment_selector(linear_policy(w), cfg), data, cfg, on_test=True)
        rows.append({"policy": name,
                     "test_worst_csd": round(te["worst_csd"], 6),
                     "test_welfare": round(te["welfare"], 1),
                     "test_exclusion": round(te["exclusion"], 6)})
        logger.info("%-26s test csd=%.4f welfare=%.0f excl=%.4f",
                    name, te["worst_csd"], te["welfare"], te["exclusion"])
    full = json.loads((ROOT / "results" / "iclr_train" / "run_main_seed42.json").read_text())
    rows.append({"policy": "full (with history)",
                 "test_worst_csd": round(full["learned"]["test"]["worst_csd"], 6),
                 "test_welfare": round(full["learned"]["test"]["welfare"], 1),
                 "test_exclusion": round(full["learned"]["test"]["exclusion"], 6)})
    with OUT.open("w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(rows[0])); w_.writeheader(); w_.writerows(rows)
    logger.info("wrote %s", OUT)

if __name__ == "__main__":
    main()
