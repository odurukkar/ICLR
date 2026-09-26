"""Cross-city transfer: policies fit on Warsaw, evaluated on Łódź.

The `temporal_2022` split trains on 18 Warsaw district series through 2022.
Łódź enters Pabulib only in 2023 and is excluded from that split entirely, so
the policies learned there have seen **no Łódź ballots, no Łódź projects, and no
Łódź year**. Replaying them on Łódź 2023-2025 is therefore a genuine transfer
test across both city and time, and it needs no retraining.

The Łódź series is a single citywide election an order of magnitude larger than
a Warsaw district (~65k ballots per edition against ~4k), so this is also a
scale-transfer test. One series is one series: report it as a case study, never
as evidence of general external validity.

Writes results/iclr_transfer.csv.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from iclr_corpus import CorpusConfig, build_series_index, load_series
from iclr_env import (
    EnvConfig,
    SelectorPolicy,
    endowment_selector,
    res_policy,
    rollout_reference,
    rollout_selector,
    uniform_policy,
)
from iclr_outcome import (
    cost_effective_weights,
    greedy_equivalent_weights,
    score_selector,
)
from iclr_policy import linear_policy

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_CSV = RESULTS / "iclr_transfer.csv"
HELD_OUT_CITY = "Poland/Łódź"


def _learned_policies(cfg: EnvConfig) -> List[Tuple[str, SelectorPolicy]]:
    """Policies fit on Warsaw only, loaded from their training artifacts."""
    out: List[Tuple[str, SelectorPolicy]] = []

    endow = RESULTS / "iclr_train" / "run_main_seed42.json"
    if endow.exists():
        w = np.array(json.loads(endow.read_text())["best_weights"])
        out.append(("learned-endowment", endowment_selector(linear_policy(w), cfg)))

    for path in sorted((RESULTS / "iclr_frontier").glob("frontier_outcome_*seed42.json")):
        pts = json.loads(path.read_text())["points"]
        for floor in (1.0, 0.85):
            best = min(pts, key=lambda p: abs(p["floor"] - floor))
            out.append(
                (f"learned-outcome-f{floor:.2f}", score_selector(np.array(best["weights"])))
            )
        break
    return out


def main() -> None:
    cfg = EnvConfig()
    index = build_series_index()
    targets = {k: r for k, r in index.items() if r.city == HELD_OUT_CITY}
    if not targets:
        raise RuntimeError(f"no series for held-out city {HELD_OUT_CITY!r}")

    rules: List[Tuple[str, Optional[SelectorPolicy]]] = [
        ("greedy-count", score_selector(greedy_equivalent_weights())),
        ("greedy-cost", score_selector(cost_effective_weights())),
        ("mes", endowment_selector(uniform_policy, cfg)),
        ("res-1.0", endowment_selector(res_policy(1.0), cfg)),
        *_learned_policies(cfg),
    ]

    rows: List[dict] = []
    for key, ref in sorted(targets.items()):
        instances = load_series(ref)
        n_ballots = sum(len(i.votes) for i in instances.values())
        logger.info(
            "%s: years=%s, %d ballots (trained policies have seen none of it)",
            key, list(ref.years), n_ballots,
        )
        hist = rollout_reference(ref, "historical", cfg=cfg, instances=instances)
        rows.append(
            {
                "series": key, "policy": "historical",
                "worst_csd": round(hist.worst_csd, 6),
                "welfare": round(hist.welfare, 1),
                "exclusion": round(hist.exclusion, 6),
                "worst_cohort": hist.worst_cohort,
            }
        )
        for name, sel in rules:
            ep = rollout_selector(ref, sel, cfg=cfg, instances=instances)
            rows.append(
                {
                    "series": key, "policy": name,
                    "worst_csd": round(ep.worst_csd, 6) if ep.worst_csd else None,
                    "welfare": round(ep.welfare, 1),
                    "exclusion": round(ep.exclusion, 6),
                    "worst_cohort": ep.worst_cohort,
                }
            )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("=" * 72)
    logger.info("CROSS-CITY TRANSFER -- Warsaw-trained policies on %s", HELD_OUT_CITY)
    logger.info("%-24s %11s %13s %11s", "policy", "worst_csd", "welfare", "exclusion")
    logger.info("-" * 72)
    mes_row = next((r for r in rows if r["policy"] == "mes"), None)
    for r in rows:
        mark = ""
        if mes_row and r["worst_csd"] is not None and r["policy"].startswith("learned"):
            mark = "  better" if r["worst_csd"] < mes_row["worst_csd"] else "  WORSE"
        logger.info(
            "%-24s %11.4f %13.0f %11.4f%s",
            r["policy"], r["worst_csd"], r["welfare"], r["exclusion"], mark,
        )
    logger.info("=" * 72)
    logger.info("Single series: a case study, not external validity.")
    logger.info("wrote %s", OUT_CSV)


if __name__ == "__main__":
    main()
