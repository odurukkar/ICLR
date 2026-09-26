"""Three checks a reviewer asked for, none of which needs a new frontier sweep.

**Matched welfare target.** The headline endowment point sits at soft target
0.99 and the headline direct point at 1.00, so the paragraph comparing them
compares different operating points. This replays both arms at 1.00 and reports
the paired series contrast that comparison actually needs.

**Attribution coverage.** Definition 1 sums attribution only over funded
projects with at least one demographically covered approver, while entitlement
uses the whole funded set. When some funded project has no covered approver,
every group carries an artificial deficit that edition, and its size depends on
how much budget the policy sends to such projects. This measures that exposure
per policy and re-runs the headline contrast on the editions where it cannot
occur.

**A one-parameter demographic tilt.** The worst-served group is 60+ in most
series and the history-free control keeps the whole gain, so the obvious null
model is a single constant multiplying the 60+ endowment. R ES does not test
this: it tilts by realized deficit, not by group identity. This fits that one
scalar on the training editions and scores it held out.

Writes results/iclr_matched_target.csv, results/iclr_attribution_coverage.csv
and results/iclr_senior_tilt.csv.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cohorts import cohort_of, group_outcome
from iclr_corpus import build_series_index, load_series, load_split
from iclr_env import (
    EnvConfig,
    EpisodeResult,
    RolloutState,
    SelectorPolicy,
    endowment_selector,
    rollout_selector,
)
from iclr_outcome import score_selector
from iclr_policy import linear_policy
from iclr_stats import exact_paired_sign_flip_pvalue
from parse_pb import PBInstance

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FRONTIER = RESULTS / "iclr_frontier"
SPLIT = "temporal_2022"
SCHEME = "age_sex"
MATCHED_TARGET = 1.00
SENIOR_PREFIX = "age60+"
TILT_GRID = tuple(round(0.05 * step, 2) for step in range(0, 61))  # 0.00 .. 3.00
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260812

OUT_MATCHED = RESULTS / "iclr_matched_target.csv"
OUT_COVERAGE = RESULTS / "iclr_attribution_coverage.csv"
OUT_TILT = RESULTS / "iclr_senior_tilt.csv"


def _frontier_point(path: Path, target: float) -> Dict[str, object]:
    payload = json.loads(path.read_text())
    point = min(payload["points"], key=lambda p: abs(p["floor"] - target))
    if abs(point["floor"] - target) > 1e-9:
        raise RuntimeError(f"{path.name} has no point at soft target {target}")
    return point


def _paired(values: Sequence[float]) -> Tuple[float, float, float, float, int]:
    diff = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = diff[rng.integers(0, diff.size, size=(BOOTSTRAP_DRAWS, diff.size))]
    means = draws.mean(axis=1)
    return (
        float(diff.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
        exact_paired_sign_flip_pvalue(diff),
        int(np.count_nonzero(diff < 0)),
    )


def _episodes(
    refs, selector: SelectorPolicy, test_years: Dict[str, Sequence[int]], cfg: EnvConfig
) -> List[EpisodeResult]:
    return [
        rollout_selector(ref, selector, score_years=test_years[ref.key], cfg=cfg)
        for ref in refs
    ]


def _test_view(split_name: str = SPLIT):
    index = build_series_index()
    split = load_split(split_name)
    test_years = {key: years for key, years in split.test}
    refs = [index[key] for key in sorted(test_years)]
    return refs, test_years


def matched_target_contrast(cfg: Optional[EnvConfig] = None) -> Dict[str, object]:
    """Compare both learned surfaces at the same soft welfare target."""
    cfg = cfg or EnvConfig()
    refs, test_years = _test_view()
    endow = _frontier_point(
        FRONTIER / f"frontier_endowment_{SPLIT}_seed42.json", MATCHED_TARGET
    )
    direct = _frontier_point(
        FRONTIER / f"frontier_outcome_{SPLIT}_seed42.json", MATCHED_TARGET
    )
    endow_eps = _episodes(
        refs, endowment_selector(linear_policy(np.array(endow["weights"])), cfg),
        test_years, cfg,
    )
    direct_eps = _episodes(
        refs, score_selector(np.array(direct["weights"])), test_years, cfg
    )
    diff = [
        a.worst_csd - b.worst_csd
        for a, b in zip(endow_eps, direct_eps)
        if a.worst_csd is not None and b.worst_csd is not None
    ]
    mean, low, high, pvalue, wins = _paired(diff)
    row = {
        "soft_target": MATCHED_TARGET,
        "n_series": len(diff),
        "endowment_csd": float(np.mean([e.worst_csd for e in endow_eps])),
        "direct_csd": float(np.mean([e.worst_csd for e in direct_eps])),
        "difference": mean,
        "ci_low": low,
        "ci_high": high,
        "p_value": pvalue,
        "endowment_wins": wins,
        "endowment_welfare": float(sum(e.welfare for e in endow_eps)),
        "direct_welfare": float(sum(e.welfare for e in direct_eps)),
        "endowment_exclusion": float(np.mean([e.exclusion for e in endow_eps])),
        "direct_exclusion": float(np.mean([e.exclusion for e in direct_eps])),
    }
    with OUT_MATCHED.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    logger.info(
        "matched target %.2f: endowment %.4f vs direct %.4f (diff %+.4f, p=%.4f, %d/%d)",
        MATCHED_TARGET, row["endowment_csd"], row["direct_csd"], mean, pvalue,
        wins, len(diff),
    )
    return row


def _uncovered_exposure(
    inst: PBInstance, winners: Sequence[str], scheme: str = SCHEME
) -> Tuple[int, float, float]:
    """Funded projects with no covered approver, and the budget they absorb."""
    approvers: Dict[str, int] = {}
    for vote in inst.votes:
        if cohort_of(vote, scheme) is None:
            continue
        for project in vote.projects:
            approvers[project] = approvers.get(project, 0) + 1
    uncovered = 0
    spend = 0.0
    total = 0.0
    for project in winners:
        item = inst.projects.get(project)
        if item is None:
            continue
        total += item.cost
        if approvers.get(project, 0) == 0:
            uncovered += 1
            spend += item.cost
    return uncovered, spend, total


def attribution_coverage(cfg: Optional[EnvConfig] = None) -> List[Dict[str, object]]:
    """Measure how exposed each policy is to the uncovered-project artefact."""
    cfg = cfg or EnvConfig()
    refs, test_years = _test_view()
    policies: List[Tuple[str, SelectorPolicy]] = [
        ("mes", endowment_selector(linear_policy(np.zeros(6)), cfg)),
    ]
    endow = json.loads((RESULTS / "iclr_train" / "run_main_seed42.json").read_text())
    policies.append(
        ("learned-endowment",
         endowment_selector(linear_policy(np.array(endow["best_weights"])), cfg))
    )
    direct = _frontier_point(
        FRONTIER / f"frontier_outcome_{SPLIT}_seed42.json", MATCHED_TARGET
    )
    policies.append(("learned-direct", score_selector(np.array(direct["weights"]))))
    corner = _frontier_point(FRONTIER / f"frontier_outcome_{SPLIT}_seed42.json", 0.0)
    policies.append(("learned-direct-unpenalized", score_selector(np.array(corner["weights"]))))

    rows: List[Dict[str, object]] = []
    clean_series: Dict[str, set] = {}
    for name, selector in policies:
        editions = 0
        affected = 0
        uncovered_projects = 0
        funded_projects = 0
        uncovered_spend = 0.0
        total_spend = 0.0
        clean = set()
        for ref in refs:
            instances = load_series(ref)
            state = RolloutState()
            series_clean = True
            for year in ref.years:
                inst = instances.get(year)
                if inst is None:
                    continue
                winners = selector(inst, state)
                state.update(group_outcome(inst, winners, cfg.scheme))
                if year not in set(test_years[ref.key]):
                    continue
                count, spend, total = _uncovered_exposure(inst, winners)
                editions += 1
                funded_projects += len(winners)
                uncovered_projects += count
                uncovered_spend += spend
                total_spend += total
                if count:
                    affected += 1
                    series_clean = False
            if series_clean:
                clean.add(ref.key)
        clean_series[name] = clean
        rows.append({
            "policy": name,
            "scored_editions": editions,
            "editions_with_uncovered_funded_project": affected,
            "funded_projects": funded_projects,
            "uncovered_funded_projects": uncovered_projects,
            "uncovered_budget_share": (uncovered_spend / total_spend) if total_spend else 0.0,
        })
        logger.info(
            "%-28s %3d/%3d editions affected, %.4f%% of budget",
            name, affected, editions, 100 * rows[-1]["uncovered_budget_share"],
        )

    intersect = set(ref.key for ref in refs)
    for name in ("mes", "learned-endowment"):
        intersect &= clean_series[name]
    rows.append({
        "policy": "series_clean_for_mes_and_learned_endowment",
        "scored_editions": len(intersect),
        "editions_with_uncovered_funded_project": 0,
        "funded_projects": 0,
        "uncovered_funded_projects": 0,
        "uncovered_budget_share": 0.0,
    })
    with OUT_COVERAGE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def senior_tilt_policy(alpha: float, scheme: str = SCHEME):
    """beta_i = (b/n)(1 + alpha) for 60+ voters, (b/n) otherwise, rescaled."""

    def _policy(inst: PBInstance, state: RolloutState) -> List[float]:
        del state
        n = len(inst.votes)
        if n == 0:
            return []
        base = inst.budget / n
        raw = []
        for vote in inst.votes:
            cohort = cohort_of(vote, scheme)
            senior = cohort is not None and cohort.startswith(SENIOR_PREFIX)
            raw.append(base * (1.0 + alpha) if senior else base)
        total = sum(raw)
        if total <= 0:
            return [base] * n
        scale = inst.budget / total
        return [value * scale for value in raw]

    return _policy


def senior_tilt(cfg: Optional[EnvConfig] = None) -> List[Dict[str, object]]:
    """Fit one scalar on the training editions and score it held out."""
    cfg = cfg or EnvConfig()
    index = build_series_index()
    split = load_split(SPLIT)
    train_years = {key: years for key, years in split.train}
    test_years = {key: years for key, years in split.test}
    refs = [index[key] for key in sorted(test_years)]

    rows: List[Dict[str, object]] = []
    for alpha in TILT_GRID:
        selector = endowment_selector(senior_tilt_policy(alpha), cfg)
        train = [
            rollout_selector(ref, selector, score_years=train_years[ref.key], cfg=cfg)
            for ref in refs
        ]
        train_csd = float(np.mean([e.worst_csd for e in train if e.worst_csd is not None]))
        rows.append({"alpha": alpha, "train_worst_csd": train_csd})
        logger.info("alpha %.2f train %.4f", alpha, train_csd)

    best = min(rows, key=lambda r: r["train_worst_csd"])
    selector = endowment_selector(senior_tilt_policy(best["alpha"]), cfg)
    test = _episodes(refs, selector, test_years, cfg)
    mes = _episodes(
        refs, endowment_selector(linear_policy(np.zeros(6)), cfg), test_years, cfg
    )
    diff = [a.worst_csd - b.worst_csd for a, b in zip(test, mes)]
    mean, low, high, pvalue, wins = _paired(diff)
    best.update({
        "test_worst_csd": float(np.mean([e.worst_csd for e in test])),
        "mes_worst_csd": float(np.mean([e.worst_csd for e in mes])),
        "difference": mean,
        "ci_low": low,
        "ci_high": high,
        "p_value": pvalue,
        "wins": wins,
        "n_series": len(diff),
        "test_welfare": float(sum(e.welfare for e in test)),
        "test_exclusion": float(np.mean([e.exclusion for e in test])),
    })
    with OUT_TILT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(best))
        writer.writeheader()
        writer.writerow(best)
        writer.writerow({"alpha": "GRID", "train_worst_csd": len(TILT_GRID)})
    logger.info(
        "senior tilt alpha*=%.2f: test %.4f vs MES %.4f (diff %+.4f, p=%.4f, %d/%d)",
        best["alpha"], best["test_worst_csd"], best["mes_worst_csd"], mean,
        pvalue, wins, len(diff),
    )
    return [best]


def main() -> None:
    matched_target_contrast()
    attribution_coverage()
    senior_tilt()


if __name__ == "__main__":
    main()


OUT_GRANULARITY = RESULTS / "iclr_instance_granularity.csv"


def instance_granularity() -> List[Dict[str, object]]:
    """Median projects per scored election, per corpus.

    Actuation depends on how many projects an edition offers: with few
    projects, no endowment vector reorders the Equal Shares purchase sequence.
    Reporting this for the fitting corpus and each external corpus lets a
    reader check whether inaction tracks instance granularity rather than the
    support-set projection.
    """
    rows: List[Dict[str, object]] = []

    refs, test_years = _test_view()
    counts = []
    for ref in refs:
        instances = load_series(ref)
        counts.extend(
            len(instances[year].projects)
            for year in test_years[ref.key]
            if year in instances
        )
    rows.append({
        "corpus": "warsaw_fitting",
        "scored_elections": len(counts),
        "median_projects": float(np.median(counts)),
        "mean_projects": float(np.mean(counts)),
    })

    manifest = RESULTS / "iclr_external_support_set" / "corpus_manifest.json"
    external_dir = ROOT / "data" / "pb_external_validation"
    if manifest.is_file() and external_dir.is_dir():
        payload = json.loads(manifest.read_text())
        counts = []
        for entry in payload["files"]:
            if int(entry["year"]) < 2023:
                continue
            path = external_dir / entry["name"]
            if not path.is_file():
                continue
            projects = 0
            section = None
            header = False
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                stripped = line.strip()
                if stripped in ("META", "PROJECTS", "VOTES"):
                    section = stripped
                    header = True
                    continue
                if section == "VOTES":
                    break
                if section == "PROJECTS":
                    if header:
                        header = False
                        continue
                    if stripped:
                        projects += 1
            counts.append(projects)
        if counts:
            rows.append({
                "corpus": "external_projected",
                "scored_elections": len(counts),
                "median_projects": float(np.median(counts)),
                "mean_projects": float(np.mean(counts)),
            })

    with OUT_GRANULARITY.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        logger.info(
            "%-20s n=%3d median=%.1f mean=%.1f",
            row["corpus"], row["scored_elections"], row["median_projects"],
            row["mean_projects"],
        )
    return rows
