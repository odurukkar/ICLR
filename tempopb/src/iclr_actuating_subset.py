"""Secondary analysis: what the fresh-city study says where the surface acts.

The locked decision is falsified, and it should stay falsified: a protocol is
worth nothing if its outcome is renegotiated afterwards. But the decision is
driven by tie mass rather than by an observed adverse effect. Two thirds of the
external series return exactly the Equal Shares allocation, and those zeros
pull every city mean toward zero and widen every interval mechanically.

This reports the same contrast restricted to the series where the endowment
surface actually changes an allocation. It is descriptive and secondary by
construction: the subset is defined by an outcome (whether the allocation
moved), so it cannot support an inferential claim, and it does not amend the
locked decision. It exists so a reader can tell "the method was tested and
failed" apart from "the protocol could not test the method."

Writes results/iclr_actuating_subset.csv.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
EVALUATION = RESULTS / "iclr_external_support_set_amendment" / "evaluation"
PER_SERIES = EVALUATION / "per_series.json"
OUT = RESULTS / "iclr_actuating_subset.csv"
SEEDS = ("1", "2", "42")
PRIMARY_SEED = "42"
TOLERANCE = 1e-9


def _by_series(rows: Sequence[Mapping[str, object]]) -> Dict[str, Mapping[str, object]]:
    return {str(row["series"]): row for row in rows}


def actuating_rows(payload: Mapping[str, object] | None = None) -> List[Dict[str, object]]:
    """Per seed: how many series move, and the contrast among those that do."""
    if payload is None:
        payload = json.loads(PER_SERIES.read_text(encoding="utf-8"))
    baseline = _by_series(payload["mes"])
    rows: List[Dict[str, object]] = []
    for seed in SEEDS:
        learned = _by_series(payload[f"endowment/seed-{seed}"])
        shared = sorted(set(learned) & set(baseline))
        diffs = []
        for key in shared:
            a, b = learned[key].get("worst_csd"), baseline[key].get("worst_csd")
            if a is None or b is None:
                continue
            diffs.append((key, float(a) - float(b)))
        moved = [d for _, d in diffs if abs(d) > TOLERANCE]
        inert = len(diffs) - len(moved)
        wins = sum(1 for d in moved if d < 0)
        losses = sum(1 for d in moved if d > 0)
        rows.append({
            "seed": seed,
            "n_series": len(diffs),
            "n_inert": inert,
            "n_actuating": len(moved),
            "wins_where_actuating": wins,
            "losses_where_actuating": losses,
            "mean_all_series": float(np.mean([d for _, d in diffs])) if diffs else float("nan"),
            "mean_where_actuating": float(np.mean(moved)) if moved else float("nan"),
        })
        logger.info(
            "seed %-2s: %d/%d actuate; mean over all %+.4f, mean where actuating %+.4f (%d win / %d lose)",
            seed, len(moved), len(diffs), rows[-1]["mean_all_series"],
            rows[-1]["mean_where_actuating"], wins, losses,
        )
    return rows


def main() -> None:
    rows = actuating_rows()
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote %s", OUT)


if __name__ == "__main__":
    main()
