"""Why does the learned policy lose on some districts?

"14 of 18 series" invites the question the paper must answer: what do the four
losing districts have in common? Reporting failures without explaining them is
only half a defence.

This computes per-series characteristics that plausibly govern transfer --
which cohort is worst, how cohesive cohorts are, ballot-length dispersion, how
hard the baseline finds the series -- and relates each to the learned policy's
per-series margin over Equal Shares.

Writes results/iclr_failure_analysis.csv.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional

import numpy as np

from iclr_corpus import build_series_index, load_split
from iclr_policy import instance_features
from iclr_train import _load_series_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PER_SERIES = ROOT / "results" / "iclr_per_series.csv"
OUT_CSV = ROOT / "results" / "iclr_failure_analysis.csv"


def _series_traits(series_data, scheme: str = "age_sex") -> Dict[str, float]:
    """Structural traits of one series, averaged over its test-year elections."""
    overlaps: List[float] = []
    len_disp: List[float] = []
    n_voters: List[int] = []
    for year in series_data.test_years:
        inst = series_data.all_years.get(year)
        if inst is None:
            continue
        feats = instance_features(inst, scheme)
        if not feats.cohort_order:
            continue
        # cohort cohesion: spread of per-cohort overlap with the electorate.
        # `overlap_dev` is already mean-centred, so its dispersion measures how
        # differently cohorts vote from one another.
        overlaps.append(float(np.std(feats.static["overlap_dev"])))
        len_disp.append(float(np.std(feats.static["ballot_length_dev"])))
        n_voters.append(len(inst.votes))
    return {
        "cohort_divergence": mean(overlaps) if overlaps else float("nan"),
        "ballot_len_dispersion": mean(len_disp) if len_disp else float("nan"),
        "mean_voters": mean(n_voters) if n_voters else float("nan"),
        "n_test_years": len(series_data.test_years),
    }


def main() -> None:
    if not PER_SERIES.exists():
        raise FileNotFoundError(f"run iclr_ablation.py first: {PER_SERIES}")
    with PER_SERIES.open() as fh:
        per = {r["series"]: r for r in csv.DictReader(fh)}

    index = build_series_index()
    data = _load_series_data(load_split("temporal_2022"), index)

    rows: List[dict] = []
    for d in data:
        if not d.test_years or d.ref.key not in per:
            continue
        rec = per[d.ref.key]
        traits = _series_traits(d)
        rows.append(
            {
                "series": d.ref.key.split("/")[-1],
                "learned_minus_mes": float(rec["learned_minus_mes"]),
                "loses": int(rec["learned_wins"] == "0"),
                "worst_cohort": rec.get("worst_cohort", ""),
                "is_senior_worst": int("60+" in str(rec.get("worst_cohort", ""))),
                "mes_csd": float(rec["mes"]),
                **{k: round(v, 5) for k, v in traits.items()},
            }
        )

    rows.sort(key=lambda r: r["learned_minus_mes"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    wins = [r for r in rows if not r["loses"]]
    losses = [r for r in rows if r["loses"]]
    logger.info("=" * 72)
    logger.info("%d wins vs %d losses", len(wins), len(losses))
    logger.info("-" * 72)
    logger.info("%-24s %10s %10s", "trait", "wins", "losses")
    for key in ("mes_csd", "cohort_divergence", "ballot_len_dispersion",
                "mean_voters", "is_senior_worst"):
        w = mean(r[key] for r in wins)
        l = mean(r[key] for r in losses)
        logger.info("%-24s %10.4f %10.4f   %s", key, w, l,
                    "<-- differs" if abs(w - l) > 0.25 * (abs(w) + 1e-9) else "")

    # Correlation of each trait with the margin (negative margin = learned wins)
    logger.info("-" * 72)
    logger.info("Pearson r with margin (learned - MES); positive r = trait predicts failure")
    margins = np.array([r["learned_minus_mes"] for r in rows])
    for key in ("mes_csd", "cohort_divergence", "ballot_len_dispersion",
                "mean_voters", "is_senior_worst"):
        vals = np.array([r[key] for r in rows], dtype=float)
        if np.std(vals) < 1e-12:
            continue
        r = float(np.corrcoef(vals, margins)[0, 1])
        logger.info("  %-24s r = %+.3f", key, r)
    logger.info("=" * 72)
    logger.info("losing series: %s", ", ".join(r["series"] for r in losses))
    logger.info("wrote %s", OUT_CSV)


if __name__ == "__main__":
    main()
