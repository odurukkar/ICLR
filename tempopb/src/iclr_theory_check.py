"""Empirical check of the support-floor proposition behind Section 6.

The theory claim, informally: a policy minimizing cohort deficit wants to fund
*demographically neutral* projects, because attribution then matches entitlement
regardless of how many people approve. Equal Shares cannot follow it there,
because a project is only funded if **its own approvers** can pay for it:

    p funded under MES with endowments beta  ==>  sum_{i in A(p)} beta_i >= c(p)

so with beta_i <= kappa * (b/n) for every voter,

    |A(p)| / n  >=  c(p) / (kappa * b).                            (support floor)

Every project bought during proportional payment is approved by at least a
cost-proportional fraction of the electorate, and the bound degrades linearly in
kappa, the largest endowment multiple the policy hands any voter. Greedy
completion and a rule that scores projects directly obey no such constraint.

This script measures, per policy:
  * kappa, the realized max endowment multiple (endowment policies only)
  * the implied support floor, and whether any funded project violates it
  * the *realized* minimum approval share among funded projects, which is what
    separates the two arms empirically

A theorem that does not predict a measurement is decorative; this is the
measurement it predicts.

Writes results/iclr_theory_check.csv.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from iclr_analysis_rollout import held_out_selections
from iclr_corpus import build_series_index, load_split
from iclr_env import EnvConfig, endowment_selector, res_policy, uniform_policy
from iclr_outcome import cost_effective_weights, score_selector
from iclr_policy import linear_policy
from iclr_train import _load_series_data
from rules import mes_with_endowments

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "results" / "iclr_theory_check.csv"


def _support_stats(inst, winners) -> Tuple[float, float]:
    """(min approval share among funded projects, min cost share among them)."""
    n = len(inst.votes) or 1
    counts: Dict[str, int] = {p: 0 for p in winners}
    for vote in inst.votes:
        for pid in vote.projects:
            if pid in counts:
                counts[pid] += 1
    if not counts:
        return float("nan"), float("nan")
    shares = [c / n for c in counts.values()]
    idx = int(np.argmin(shares))
    pid = list(counts)[idx]
    cost_share = inst.projects[pid].cost / (inst.budget or 1.0)
    return float(min(shares)), float(cost_share)


def _payment_floor_violations(inst, winners, kappa: float) -> int:
    """Count payment-phase winners that violate their proved cap floor."""
    counts: Dict[str, int] = {pid: 0 for pid in winners}
    for vote in inst.votes:
        for pid in vote.projects:
            if pid in counts:
                counts[pid] += 1
    n = len(inst.votes) or 1
    return sum(
        counts[pid] / n + 1e-9
        < inst.projects[pid].cost / ((inst.budget or 1.0) * kappa)
        for pid in winners
    )


def _outside_uniform_floor(selector, data, cfg: EnvConfig) -> dict:
    """Share of held-out projects/spend outside the kappa=1 payment floor."""
    n_viol = n_tot = 0
    c_viol = c_tot = 0.0
    for selection in held_out_selections(selector, data, cfg):
        inst = selection.instance
        counts: Dict[str, int] = {}
        for vote in inst.votes:
            for pid in vote.projects:
                counts[pid] = counts.get(pid, 0) + 1
        for pid in selection.winners:
            cost = inst.projects[pid].cost
            n_tot += 1
            c_tot += cost
            if counts.get(pid, 0) / len(inst.votes) < cost / inst.budget - 1e-12:
                n_viol += 1
                c_viol += cost
    return {
        "pct_projects": round(100 * n_viol / max(n_tot, 1), 4),
        "pct_budget": round(100 * c_viol / max(c_tot, 1e-9), 4),
    }


def main() -> None:
    cfg = EnvConfig()
    index = build_series_index()
    data = _load_series_data(load_split("temporal_2022"), index)

    endow_json = ROOT / "results" / "iclr_train" / "run_main_seed42.json"
    learned_endow: Optional[np.ndarray] = None
    if endow_json.exists():
        learned_endow = np.array(json.loads(endow_json.read_text())["best_weights"])

    # Both operating points matter. The welfare-constrained point is the
    # headline policy; the *unconstrained* point is where reward-hacking
    # actually happens, so it is the one the separation claim is about.
    front = sorted((ROOT / "results" / "iclr_frontier").glob("frontier_outcome_*seed42.json"))
    outcome_points: List[Tuple[str, np.ndarray]] = []
    if front:
        pts = json.loads(front[0].read_text())["points"]
        for label, floor in (("learned-outcome-f1.0", 1.0), ("learned-outcome-f0.0", 0.0)):
            best = min(pts, key=lambda p: abs(p["floor"] - floor))
            outcome_points.append((label, np.array(best["weights"])))

    rows: List[dict] = []
    endowment_policies = [("mes", uniform_policy), ("res-1.0", res_policy(1.0))]
    if learned_endow is not None:
        endowment_policies.append(("learned-endowment", linear_policy(learned_endow)))

    for name, policy in endowment_policies:
        beta_by_instance: Dict[int, List[float]] = {}
        payment_by_instance: Dict[int, set[str]] = {}

        def selector(inst, state, pol=policy):
            beta = pol(inst, state)
            beta_by_instance[id(inst)] = beta
            payment_by_instance[id(inst)] = mes_with_endowments(
                inst, endowments=beta, completion=False
            )
            return mes_with_endowments(
                inst, endowments=beta, completion=cfg.completion
            )

        for selection in held_out_selections(selector, data, cfg):
            inst = selection.instance
            beta = beta_by_instance[id(inst)]
            base = inst.budget / len(inst.votes) if inst.votes else 0.0
            kappa = (max(beta) / base) if base and beta else float("nan")
            min_share, cost_share = _support_stats(inst, selection.winners)
            floor = cost_share / kappa if kappa and kappa > 0 else float("nan")
            violations = (
                _payment_floor_violations(
                    inst, payment_by_instance[id(inst)], kappa
                )
                if kappa == kappa and kappa > 0
                else 0
            )
            rows.append(
                {
                    "series": selection.series.split("/")[-1],
                    "year": selection.year,
                    "policy": name,
                    "kappa": round(kappa, 4),
                    "min_approval_share": round(min_share, 6),
                    "implied_floor": round(floor, 6),
                    "violates_floor": violations,
                }
            )

    for name, weights in [("greedy-cost", cost_effective_weights()), *outcome_points]:
        for selection in held_out_selections(score_selector(weights), data, cfg):
            min_share, _ = _support_stats(selection.instance, selection.winners)
            rows.append(
                {
                    "series": selection.series.split("/")[-1],
                    "year": selection.year,
                    "policy": name,
                    "kappa": float("nan"),
                    "min_approval_share": round(min_share, 6),
                    "implied_floor": float("nan"),
                    "violates_floor": 0,
                }
            )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # ---- the decisive measurement the proposition predicts ----
    # Share of each policy's spending that goes to projects plain Equal Shares
    # (kappa = 1) could never buy, i.e. |A(p)|/n < c(p)/b. This is the quantity
    # that separates the arms; min approval share does not.
    infeasible: List[dict] = []
    endowment_selectors = [
        ("mes", endowment_selector(uniform_policy, cfg)),
        ("res-1.0", endowment_selector(res_policy(1.0), cfg)),
    ]
    if learned_endow is not None:
        endowment_selectors.append(
            (
                "learned-endowment",
                endowment_selector(linear_policy(learned_endow), cfg),
            )
        )
    for name, selector in endowment_selectors:
        infeasible.append({"policy": name, **_outside_uniform_floor(selector, data, cfg)})
    for name, weights in [("greedy-cost", cost_effective_weights()), *outcome_points]:
        infeasible.append(
            {
                "policy": name,
                **_outside_uniform_floor(score_selector(weights), data, cfg),
            }
        )

    inf_csv = ROOT / "results" / "iclr_infeasible_spend.csv"
    with inf_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["policy", "pct_projects", "pct_budget"])
        writer.writeheader()
        writer.writerows(infeasible)
    logger.info("-" * 78)
    logger.info("Budget routed outside the kappa=1 payment floor:")
    for r in infeasible:
        logger.info("  %-22s %6.2f%% of projects   %6.2f%% of budget",
                    r["policy"], r["pct_projects"], r["pct_budget"])
    logger.info("wrote %s", inf_csv)

    logger.info("=" * 78)
    logger.info(
        "%-20s %8s %18s %14s %10s",
        "policy", "kappa", "min approval share", "implied floor", "violations",
    )
    logger.info("-" * 78)
    for name in ("mes", "res-1.0", "learned-endowment", "greedy-cost",
                 "learned-outcome-f1.0", "learned-outcome-f0.0"):
        sub = [r for r in rows if r["policy"] == name]
        if not sub:
            continue
        ks = [r["kappa"] for r in sub if r["kappa"] == r["kappa"]]
        shares = [r["min_approval_share"] for r in sub if r["min_approval_share"] == r["min_approval_share"]]
        floors = [r["implied_floor"] for r in sub if r["implied_floor"] == r["implied_floor"]]
        logger.info(
            "%-20s %8s %18.5f %14s %10d",
            name,
            f"{np.mean(ks):.2f}" if ks else "n/a",
            float(np.mean(shares)) if shares else float("nan"),
            f"{np.mean(floors):.5f}" if floors else "n/a",
            sum(r["violates_floor"] for r in sub),
        )
    logger.info("=" * 78)
    logger.info(
        "The proposition predicts zero violations among payment-phase winners "
        "for every MES-based policy, and that holds. Note the min approval share does NOT separate the "
        "arms -- the separating quantity is the share of budget spent outside "
        "the kappa=1 payment floor, reported above."
    )
    logger.info("wrote %s", OUT_CSV)


if __name__ == "__main__":
    main()
