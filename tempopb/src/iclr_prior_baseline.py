"""Replay the closest published learned PB rule under the frozen protocol.

Thach, Sha, and Chan (AAMAS 2026, DOI 10.65109/LGEP3560) publish
LLM-generated priority rules for approval ballots with cost and count
satisfaction:

    cost:  sqrt(approval_share) / (1 + cost_share)
    count: approval_share / cost_share * log(1 + approval_share).

This script does not refit that rule.  It evaluates the frozen formula on the
same train/test split and through the same longitudinal ledger used by both
learned families in the paper.

Writes results/iclr_llmrule_baseline.csv, including per-series and aggregate
rows for both train and held-out views.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List

from iclr_corpus import build_series_index, load_split
from iclr_env import EnvConfig, aggregate, rollout_selector
from iclr_outcome import llmrule_card_selector, llmrule_cost_selector
from iclr_train import _load_series_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "results" / "iclr_llmrule_baseline.csv"
SOURCE_DOI = "10.65109/LGEP3560"
RULES = (
    (
        "llmrule-cost",
        llmrule_cost_selector,
        "sqrt(approval_share)/(1+cost_share)",
    ),
    (
        "llmrule-card",
        llmrule_card_selector,
        "approval_share/cost_share*log1p(approval_share)",
    ),
)


def main() -> None:
    cfg = EnvConfig()
    data = _load_series_data(load_split("temporal_2022"), build_series_index())
    rows: List[dict] = []

    for rule, selector_factory, score in RULES:
        selector = selector_factory()
        for view in ("train", "test"):
            episodes = []
            for item in data:
                years = item.train_years if view == "train" else item.test_years
                if not years:
                    continue
                instances = item.train_only if view == "train" else item.all_years
                episode = rollout_selector(
                    item.ref,
                    selector,
                    score_years=years,
                    cfg=cfg,
                    instances=instances,
                )
                episodes.append(episode)
                rows.append(
                    {
                        "rule": rule,
                        "series": item.ref.key,
                        "view": view,
                        "worst_csd": round(float(episode.worst_csd), 6),
                        "mean_csd": round(float(episode.mean_csd), 6),
                        "welfare": round(float(episode.welfare), 1),
                        "cost_welfare": round(float(episode.cost_welfare), 1),
                        "exclusion": round(float(episode.exclusion), 6),
                        "n_series": 1,
                        "source_doi": SOURCE_DOI,
                        "score": score,
                    }
                )

            summary = aggregate(episodes)
            rows.append(
                {
                    "rule": rule,
                    "series": "__aggregate__",
                    "view": view,
                    "worst_csd": round(float(summary["worst_csd"]), 6),
                    "mean_csd": round(float(summary["mean_csd"]), 6),
                    "welfare": round(float(summary["welfare"]), 1),
                    "cost_welfare": round(float(summary["cost_welfare"]), 1),
                    "exclusion": round(float(summary["exclusion"]), 6),
                    "n_series": int(summary["n_series"]),
                    "source_doi": SOURCE_DOI,
                    "score": score,
                }
            )
            logger.info(
                "%s/%s: worst_csd=%.4f welfare=%.0f exclusion=%.4f (%d series)",
                rule,
                view,
                summary["worst_csd"],
                summary["welfare"],
                summary["exclusion"],
                summary["n_series"],
            )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s", OUT_CSV)


if __name__ == "__main__":
    main()
