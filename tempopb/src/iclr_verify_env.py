"""Phase-1 gate: the ICLR env must reproduce the frozen replay numbers.

Replays the full primary corpus through `iclr_env` under uniform endowments
(= MES) and the one-scalar RES map, and compares the per-series worst-cohort
CSD against `results/ceiling.csv`, which carries the mes/res10 columns from the
original pipeline. Any per-series disagreement above tolerance is a bug in the
environment, not a modelling choice, and Phase 2 must not start until this
passes.

Also freezes the three evaluation splits.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List

from iclr_corpus import (
    build_series_index,
    leave_city_out,
    leave_district_out,
    load_series,
    save_split,
    temporal_split,
)
from iclr_env import EnvConfig, res_policy, rollout, rollout_reference, uniform_policy

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CEILING_CSV = ROOT / "results" / "ceiling.csv"
OUT_CSV = ROOT / "results" / "iclr_env_verification.csv"
TOL = 5e-4


def _reference_rows() -> Dict[str, Dict[str, float]]:
    """Frozen per-series worst-cohort CSD from the original pipeline."""
    if not CEILING_CSV.exists():
        raise FileNotFoundError(f"missing reference results: {CEILING_CSV}")
    out: Dict[str, Dict[str, float]] = {}
    with CEILING_CSV.open() as fh:
        for row in csv.DictReader(fh):
            out[row["series"]] = {
                "hist": float(row["hist_csd"]),
                "mes": float(row["mes_csd"]),
                "res10": float(row["res10_csd"]),
            }
    return out


def main() -> None:
    cfg = EnvConfig()
    index = build_series_index()
    reference = _reference_rows()
    logger.info("index=%d series, reference=%d series", len(index), len(reference))

    rows: List[dict] = []
    failures: List[str] = []
    res10 = res_policy(1.0)

    for i, (key, ref) in enumerate(sorted(index.items()), start=1):
        if key not in reference:
            logger.warning("[%d] %s not in reference results, skipping", i, key)
            continue
        instances = load_series(ref)
        got = {
            "hist": rollout_reference(ref, "historical", cfg=cfg, instances=instances),
            "mes": rollout(ref, uniform_policy, cfg=cfg, instances=instances),
            "res10": rollout(ref, res10, cfg=cfg, instances=instances),
        }
        row = {"series": key, "years": len(ref.years)}
        for name, episode in got.items():
            want = reference[key][name]
            have = episode.worst_csd
            delta = float("nan") if have is None else abs(have - want)
            row[f"{name}_ref"] = round(want, 6)
            row[f"{name}_env"] = None if have is None else round(have, 6)
            row[f"{name}_delta"] = None if have is None else round(delta, 6)
            if have is None or delta > TOL:
                failures.append(f"{key}/{name}: ref={want:.6f} env={have} delta={delta}")
        rows.append(row)
        logger.info(
            "[%d/%d] %s mes ref=%.4f env=%.4f",
            i, len(index), key, reference[key]["mes"], got["mes"].worst_csd or float("nan"),
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with OUT_CSV.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        logger.info("wrote %s (%d series)", OUT_CSV, len(rows))

    for split in (
        temporal_split(index),
        leave_city_out(index, "Poland/Łódź"),
        *(leave_district_out(index, f) for f in range(5)),
    ):
        save_split(split)
        logger.info(
            "split %-22s train=%d test=%d", split.name, len(split.train), len(split.test)
        )

    if failures:
        logger.error("VERIFICATION FAILED (%d mismatches):", len(failures))
        for line in failures:
            logger.error("  %s", line)
        raise SystemExit(1)
    logger.info("VERIFICATION PASSED: env reproduces frozen replay numbers")


if __name__ == "__main__":
    main()
