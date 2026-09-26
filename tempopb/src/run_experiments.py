"""RQ1 audit + RQ2 counterfactual replay over all usable longitudinal series.

Outputs (results/):
  audit_long.csv     one row per (series, year, rule, cohort): utility,
                     entitlement, underfunded flag
  series_summary.csv one row per (series, rule): mean/max CSD, welfare,
                     exclusion rate
  persistence.csv    underfunding persistence stats (historical outcomes)

Rules replayed: historical (recorded winners), greedy, mes, res-05, res-10.
RES carries each cohort's cumulative deficit (from its own outcomes) into
the next year's endowments.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

from cohorts import cohort_of, group_outcome, historical_winners
from parse_pb import PBInstance, parse_pb_file
from rules import greedy_by_votes, mes_with_endowments, res_endowments

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "pb"
RESULTS_DIR = Path(
    os.environ.get(
        "TEMPOPB_RESULTS_DIR",
        Path(__file__).resolve().parent.parent / "results",
    )
)

SCHEME = "age_sex"
MIN_RUN = 3
MIN_DEMO_COVERAGE = 0.5
MAX_VOTERS = 300_000  # skip the giant citywide instances in the main pass
RULES = ("historical", "greedy", "mes", "res-025", "res-05", "res-075", "res-10")
LAMBDAS = {"res-025": 0.25, "res-05": 0.5, "res-075": 0.75, "res-10": 1.0}


def demo_coverage(inst: PBInstance) -> float:
    if not inst.votes:
        return 0.0
    n = sum(1 for v in inst.votes if cohort_of(v, SCHEME) is not None)
    return n / len(inst.votes)


def approval_welfare(inst: PBInstance, winners: Set[str]) -> float:
    wset = winners
    return float(sum(len(set(v.projects) & wset) for v in inst.votes))


def cost_welfare(inst: PBInstance, winners: Set[str]) -> float:
    """Cost-weighted satisfaction: money on funded projects each voter
    approved, summed over voters."""
    return float(sum(
        sum(inst.projects[p].cost for p in set(v.projects) & winners
            if p in inst.projects)
        for v in inst.votes
    ))


def exclusion_rate(inst: PBInstance, winners: Set[str]) -> float:
    if not inst.votes:
        return 0.0
    excluded = sum(1 for v in inst.votes if not set(v.projects) & winners)
    return excluded / len(inst.votes)


def longest_run_years(years: List[int]) -> List[int]:
    ys = sorted(set(years))
    best: List[int] = []
    cur: List[int] = []
    for y in ys:
        if cur and y == cur[-1] + 1:
            cur.append(y)
        else:
            cur = [y]
        if len(cur) > len(best):
            best = list(cur)
    return best


def winners_for_rule(
    inst: PBInstance, rule: str, deficits: Dict[str, float]
) -> Optional[Set[str]]:
    if rule == "historical":
        w = historical_winners(inst)
        return w or None
    if rule == "greedy":
        return greedy_by_votes(inst)
    if rule == "mes":
        return mes_with_endowments(inst)
    if rule in LAMBDAS:
        endow = res_endowments(inst, deficits, LAMBDAS[rule], SCHEME)
        return mes_with_endowments(inst, endowments=endow)
    raise ValueError(rule)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # ---- build the series index ----
    by_series: Dict[str, Dict[int, Path]] = defaultdict(dict)
    for path in sorted(DATA_DIR.glob("*.pb")):
        try:
            inst = parse_pb_file(path)
        except Exception as exc:  # noqa: BLE001
            logger.error("parse failed %s: %s", path.name, exc)
            continue
        if inst.vote_type != "approval" or inst.year is None:
            continue
        if not historical_winners(inst):
            continue
        if len(inst.votes) == 0 or len(inst.votes) > MAX_VOTERS:
            continue
        if demo_coverage(inst) < MIN_DEMO_COVERAGE:
            continue
        by_series[inst.series_key()][inst.year] = path

    usable = {
        key: years for key, years in by_series.items()
        if len(longest_run_years(list(years))) >= MIN_RUN
    }
    logger.info("usable series (run >= %d): %d", MIN_RUN, len(usable))

    audit_rows: List[dict] = []
    summary_rows: List[dict] = []
    transitions: List[tuple] = []  # (under_t, under_t1) historical only

    for s_idx, (key, years_map) in enumerate(sorted(usable.items())):
        run_years = longest_run_years(list(years_map))
        insts = {y: parse_pb_file(years_map[y]) for y in run_years}
        logger.info("[%d/%d] %s years=%s", s_idx + 1, len(usable), key, run_years)

        for rule in RULES:
            t0 = time.time()
            deficits: Dict[str, float] = defaultdict(float)
            yearly_outcomes: Dict[int, Dict[str, Dict[str, float]]] = {}
            welfare_sum = spent_sum = cost_welfare_sum = 0.0
            excl: List[float] = []
            skip_series = False
            for y in run_years:
                inst = insts[y]
                winners = winners_for_rule(inst, rule, deficits)
                if winners is None:
                    skip_series = True
                    break
                outcome = group_outcome(inst, winners, SCHEME)
                yearly_outcomes[y] = outcome
                welfare_sum += approval_welfare(inst, winners)
                cost_welfare_sum += cost_welfare(inst, winners)
                spent_sum += sum(
                    inst.projects[p].cost for p in winners if p in inst.projects
                )
                excl.append(exclusion_rate(inst, winners))
                for c, vals in outcome.items():
                    deficits[c] += vals["entitlement"] - vals["utility"]
                    audit_rows.append(
                        {
                            "series": key, "year": y, "rule": rule, "cohort": c,
                            "voters": int(vals["voters"]),
                            "utility": round(vals["utility"], 2),
                            "entitlement": round(vals["entitlement"], 2),
                            "underfunded": int(vals["utility"] < vals["entitlement"]),
                        }
                    )
            if skip_series:
                continue

            cohorts_all = sorted({c for o in yearly_outcomes.values() for c in o})
            csds = []
            for c in cohorts_all:
                ent = sum(o[c]["entitlement"] for o in yearly_outcomes.values() if c in o)
                utl = sum(o[c]["utility"] for o in yearly_outcomes.values() if c in o)
                if ent > 0:
                    csds.append((ent - utl) / ent)
            if not csds:
                continue
            summary_rows.append(
                {
                    "series": key, "rule": rule, "n_years": len(run_years),
                    "mean_csd": round(sum(csds) / len(csds), 4),
                    "max_csd": round(max(csds), 4),
                    "welfare": round(welfare_sum, 1),
                    "welfare_cost": round(cost_welfare_sum, 1),
                    "spent": round(spent_sum, 2),
                    "mean_exclusion": round(sum(excl) / len(excl), 4),
                    "runtime_s": round(time.time() - t0, 1),
                }
            )

            if rule == "historical":
                for c in cohorts_all:
                    flags = [
                        int(yearly_outcomes[y][c]["utility"] < yearly_outcomes[y][c]["entitlement"])
                        for y in run_years if c in yearly_outcomes[y]
                    ]
                    for a, b in zip(flags, flags[1:]):
                        transitions.append((a, b))

    with open(RESULTS_DIR / "audit_long.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(audit_rows[0].keys()))
        w.writeheader()
        w.writerows(audit_rows)
    with open(RESULTS_DIR / "series_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    n11 = sum(1 for a, b in transitions if a == 1 and b == 1)
    n1x = sum(1 for a, b in transitions if a == 1)
    n_1 = sum(1 for _, b in transitions if b == 1)
    with open(RESULTS_DIR / "persistence.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["transitions", "P(under_t1|under_t)", "base_rate P(under_t1)"])
        w.writerow([
            len(transitions),
            round(n11 / n1x, 4) if n1x else "",
            round(n_1 / len(transitions), 4) if transitions else "",
        ])
    logger.info(
        "persistence: P(under|under)=%s vs base=%s over %d transitions",
        round(n11 / n1x, 3) if n1x else "n/a",
        round(n_1 / len(transitions), 3) if transitions else "n/a",
        len(transitions),
    )
    logger.info("wrote %d audit rows, %d summary rows", len(audit_rows), len(summary_rows))


if __name__ == "__main__":
    sys.exit(main())
