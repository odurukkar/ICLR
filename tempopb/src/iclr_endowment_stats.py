"""Persist the last two paper tables that were only ever computed ad hoc.

Two numbers reached the draft from throwaway scripts, which violates the rule
that every figure in the paper resolves to a committed script writing into
results/. This closes that:

1. **Endowment-arm significance** -- the learned endowment map against Equal
   Shares and Rollover Equal Shares, with paired-bootstrap confidence intervals
   and exact sign-flip tests over series on held-out editions. Cited in the
   abstract and Section 5.
2. **Allocation composition** -- how many projects each policy funds, their mean
   cost share, cohort concentration, and supporter/electorate representation
   distance. This shows the unpenalized direct policy funds *few and large* projects
   with more representative support rather than simply unpopular ones.

Writes results/iclr_endow_significance.csv,
results/iclr_two_arm_per_series.csv, and results/iclr_composition.csv.
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
from iclr_env import (
    EnvConfig,
    SelectorPolicy,
    endowment_selector,
    res_policy,
    rollout_selector,
    uniform_policy,
)
from iclr_outcome import (
    cost_effective_weights,
    llmrule_card_selector,
    llmrule_cost_selector,
    project_features,
    score_selector,
)
from iclr_policy import instance_features, linear_policy
from iclr_train import _load_series_data
from iclr_stats import exact_paired_sign_flip_pvalue

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SIG_CSV = RESULTS / "iclr_endow_significance.csv"
COMP_CSV = RESULTS / "iclr_composition.csv"
PER_SERIES_CSV = RESULTS / "iclr_two_arm_per_series.csv"
N_BOOT = 10_000


def _per_series(selector: SelectorPolicy, data, cfg: EnvConfig) -> Dict[str, float]:
    return {
        d.ref.key: rollout_selector(
            d.ref, selector, score_years=d.test_years, cfg=cfg, instances=d.all_years
        ).worst_csd
        for d in data
        if d.test_years
    }


def _bootstrap(
    a: List[float],
    b: List[float],
    seed: int = 42,
    n_boot: int = N_BOOT,
) -> Tuple[float, float, float, float, int]:
    diff = np.array(a, dtype=float) - np.array(b, dtype=float)
    diff = diff[~np.isnan(diff)]
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, diff.size, (n_boot, diff.size))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = exact_paired_sign_flip_pvalue(diff)
    return float(diff.mean()), float(lo), float(hi), float(min(p, 1.0)), int((diff < 0).sum())


def _project_representation_tv(inst, pid: str, scheme: str = "age_sex") -> float:
    """TV distance between a project's supporter and electorate compositions."""
    projects = project_features(inst, scheme)
    row = {project_id: i for i, project_id in enumerate(projects.project_ids)}.get(pid)
    if row is None:
        return float("nan")
    supporter_share = projects.cohort_share[row]
    if supporter_share.sum() <= 0:
        return float("nan")
    election = instance_features(inst, scheme)
    electorate_share = np.array(
        [election.cohort_sizes[group] for group in projects.cohort_order],
        dtype=float,
    )
    electorate_share /= electorate_share.sum()
    return float(0.5 * np.abs(supporter_share - electorate_share).sum())


def _composition(selector: SelectorPolicy, data, cfg: EnvConfig) -> dict:
    """Summarize projects selected after replaying each complete series prefix."""
    n_win: List[int] = []
    cost_share: List[float] = []
    concentration: List[float] = []
    representation_cost = 0.0
    representation_weighted_tv = 0.0
    for selection in held_out_selections(selector, data, cfg):
        feats = project_features(selection.instance, cfg.scheme)
        index = {pid: i for i, pid in enumerate(feats.project_ids)}
        idx = [index[pid] for pid in selection.winners if pid in index]
        if not idx:
            continue
        n_win.append(len(idx))
        cost_share.append(float(np.mean(feats.static[idx, 2])))
        concentration.append(float(np.mean(feats.static[idx, 4])))
        for pid in selection.winners:
            tv = _project_representation_tv(selection.instance, pid, cfg.scheme)
            if not np.isfinite(tv):
                continue
            cost = float(selection.instance.projects[pid].cost)
            representation_cost += cost
            representation_weighted_tv += cost * tv
    return {
        "mean_projects_funded": round(float(np.mean(n_win)), 2),
        "mean_cost_share": round(float(np.mean(cost_share)), 5),
        "mean_concentration": round(float(np.mean(concentration)), 5),
        "cost_weighted_representation_tv": round(
            representation_weighted_tv / representation_cost, 5
        ) if representation_cost else float("nan"),
    }


def main() -> None:
    cfg = EnvConfig()
    data = _load_series_data(load_split("temporal_2022"), build_series_index())

    endow_path = RESULTS / "iclr_train" / "run_main_seed42.json"
    if not endow_path.exists():
        raise FileNotFoundError(f"train the endowment arm first: {endow_path}")
    endow_w = np.array(json.loads(endow_path.read_text())["best_weights"])

    front = sorted((RESULTS / "iclr_frontier").glob("frontier_outcome_*seed42.json"))
    outcome: Dict[str, np.ndarray] = {}
    if front:
        for point in json.loads(front[0].read_text())["points"]:
            if point["floor"] in (0.0, 1.0):
                outcome[f"outcome-f{point['floor']:.2f}"] = np.array(point["weights"])

    # ---- 1. endowment significance ----
    learned = _per_series(endowment_selector(linear_policy(endow_w), cfg), data, cfg)
    keys = sorted(learned)
    rows: List[dict] = []
    reference_scores: Dict[str, Dict[str, float]] = {}
    for name, sel in (
        ("mes", endowment_selector(uniform_policy, cfg)),
        ("res-1.0", endowment_selector(res_policy(1.0), cfg)),
        ("llmrule-cost", llmrule_cost_selector()),
        ("llmrule-card", llmrule_card_selector()),
    ):
        base = _per_series(sel, data, cfg)
        reference_scores[name] = base
        mean, lo, hi, p, wins = _bootstrap([learned[k] for k in keys], [base[k] for k in keys])
        rows.append(
            {
                "contrast": f"learned-endowment vs {name}",
                "learned_mean": round(float(np.mean([learned[k] for k in keys])), 6),
                "baseline_mean": round(float(np.mean([base[k] for k in keys])), 6),
                "diff": round(mean, 6), "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
                "p_two_sided": round(p, 5), "wins": wins, "n": len(keys),
            }
        )
        logger.info(
            "learned-endowment vs %-8s diff=%+.4f CI [%+.4f,%+.4f] p=%.4f wins %d/%d",
            name, mean, lo, hi, p, wins, len(keys),
        )
    # Report the paired class-level contrast at the headline point. A
    # nonsignificant result is not treated as evidence of equivalence.
    out_scores: Optional[Dict[str, float]] = None
    if "outcome-f1.00" in outcome:
        out_scores = _per_series(score_selector(outcome["outcome-f1.00"]), data, cfg)
        mean, lo, hi, p, wins = _bootstrap(
            [learned[k] for k in keys], [out_scores[k] for k in keys]
        )
        rows.append(
            {
                "contrast": "learned-endowment vs learned-outcome",
                "learned_mean": round(float(np.mean([learned[k] for k in keys])), 6),
                "baseline_mean": round(float(np.mean([out_scores[k] for k in keys])), 6),
                "diff": round(mean, 6), "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
                "p_two_sided": round(p, 5), "wins": wins, "n": len(keys),
            }
        )
        logger.info(
            "learned-endowment vs learned-outcome diff=%+.4f CI [%+.4f,%+.4f] p=%.4f wins %d/%d",
            mean, lo, hi, p, wins, len(keys),
        )

    with SIG_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Store every paired district effect used by the evidence plot. Keeping
    # this separate from aggregate significance prevents a plot from silently
    # recomputing a different rollout or baseline.
    if out_scores is not None:
        mes_scores = reference_scores["mes"]
        per_series = []
        for key in keys:
            endow_delta = learned[key] - mes_scores[key]
            outcome_delta = out_scores[key] - mes_scores[key]
            per_series.append(
                {
                    "series": key,
                    "mes": round(mes_scores[key], 6),
                    "learned_endowment": round(learned[key], 6),
                    "learned_outcome": round(out_scores[key], 6),
                    "endowment_minus_mes": round(endow_delta, 6),
                    "outcome_minus_mes": round(outcome_delta, 6),
                    "endowment_wins": int(endow_delta < 0),
                    "outcome_wins": int(outcome_delta < 0),
                }
            )
        per_series.sort(key=lambda row: float(row["outcome_minus_mes"]))
        with PER_SERIES_CSV.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(per_series[0]))
            writer.writeheader()
            writer.writerows(per_series)

    # ---- 2. allocation composition ----
    sels: List[Tuple[str, SelectorPolicy]] = [
        ("mes", endowment_selector(uniform_policy, cfg)),
        ("learned-endowment", endowment_selector(linear_policy(endow_w), cfg)),
        ("greedy-cost", score_selector(cost_effective_weights())),
        ("llmrule-cost", llmrule_cost_selector()),
        ("llmrule-card", llmrule_card_selector()),
        *[(k, score_selector(w)) for k, w in sorted(outcome.items())],
    ]
    comp: List[dict] = []
    for name, sel in sels:
        comp.append({"policy": name, **_composition(sel, data, cfg)})
        logger.info(
            "%-20s funded=%5.1f  cost share=%.4f  concentration=%.4f  rep-TV=%.4f",
            name, comp[-1]["mean_projects_funded"],
            comp[-1]["mean_cost_share"], comp[-1]["mean_concentration"],
            comp[-1]["cost_weighted_representation_tv"],
        )
    with COMP_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(comp[0]))
        writer.writeheader()
        writer.writerows(comp)

    logger.info("wrote %s, %s, and %s", SIG_CSV, PER_SERIES_CSV, COMP_CSV)


if __name__ == "__main__":
    main()
