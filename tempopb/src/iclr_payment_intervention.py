"""Evaluate a frozen learned project score with and without payment gating.

The paper's two trained arms differ in learned object and feature map as well as
allocation procedure. This analysis removes that confound for a narrower
question: for an already learned five-feature project score, what changes when
the exact same scores are replayed through an approver-funded payment kernel?

No model is refit. For each requested soft-target point from the frozen
outcome frontier, the script evaluates:

* ``direct`` -- the original municipal-budget greedy fill;
* ``static-floor`` -- the same direct fill after the one-shot support filter
  from Proposition 2 at kappa=1;
* ``payment+completion`` -- score-prioritized approver payment followed by the
  paper's approval-count completion; and
* ``payment-only`` -- the same payment phase without completion.

The payment-prioritized rule is a diagnostic and is not standard MES because
it orders affordable projects by the frozen learned score instead of minimum
price. Outputs are series-level, paired, and fully tied to the source frontier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from iclr_corpus import build_series_index, load_split
from iclr_env import EnvConfig, EpisodeResult, SelectorPolicy, aggregate, rollout_selector
from iclr_outcome import payment_gated_score_selector, score_selector
from iclr_stats import exact_paired_sign_flip_pvalue
from iclr_train import SeriesData, _load_series_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FRONTIER = (
    ROOT / "results" / "iclr_frontier" / "frontier_outcome_temporal_2022_seed42.json"
)
SUMMARY_CSV = ROOT / "results" / "iclr_payment_intervention_summary.csv"
PER_SERIES_CSV = ROOT / "results" / "iclr_payment_intervention_per_series.csv"
MANIFEST_JSON = ROOT / "results" / "iclr_payment_intervention_manifest.json"
N_BOOT = 10_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matching_point(points: Sequence[Mapping[str, object]], floor: float) -> Mapping[str, object]:
    matches = [point for point in points if abs(float(point["floor"]) - floor) < 1e-12]
    if len(matches) != 1:
        raise ValueError(f"expected one frontier point at floor {floor}, found {len(matches)}")
    return matches[0]


def _test_episodes(
    selector: SelectorPolicy,
    data: Sequence[SeriesData],
    cfg: EnvConfig,
) -> List[EpisodeResult]:
    return [
        rollout_selector(
            district.ref,
            selector,
            score_years=district.test_years,
            cfg=cfg,
            instances=district.all_years,
        )
        for district in data
        if district.test_years
    ]


def _budget_total(data: Sequence[SeriesData]) -> float:
    return float(
        sum(
            district.all_years[year].budget
            for district in data
            for year in district.test_years
        )
    )


def _contrast(
    values: Sequence[float],
    direct: Sequence[float],
    *,
    seed: int,
    n_boot: int,
) -> Dict[str, float | int]:
    a = np.asarray(values, dtype=float)
    b = np.asarray(direct, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError(f"invalid paired vectors: {a.shape} and {b.shape}")
    diff = a - b
    rng = np.random.default_rng(seed)
    draws = diff[rng.integers(0, diff.size, size=(n_boot, diff.size))].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "diff_vs_direct": float(diff.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_two_sided": exact_paired_sign_flip_pvalue(diff),
        "wins_vs_direct": int(np.count_nonzero(diff < 0)),
        "n": int(diff.size),
    }


def _validate_frozen_direct(point: Mapping[str, object], stats: Mapping[str, float]) -> None:
    expected = {
        "worst_csd": float(point["test_worst_csd"]),
        "welfare": float(point["test_welfare"]),
        "exclusion": float(point["test_exclusion"]),
    }
    for key, target in expected.items():
        tolerance = 5e-6 if key != "welfare" else 0.5
        if abs(float(stats[key]) - target) > tolerance:
            raise ValueError(
                f"frozen direct replay mismatch for {key}: "
                f"recomputed {stats[key]} vs frontier {target}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
    parser.add_argument("--floors", type=float, nargs="+", default=[0.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()

    frontier = json.loads(args.frontier.read_text())
    if frontier.get("arm") != "outcome":
        raise ValueError(f"expected outcome frontier, got {frontier.get('arm')!r}")
    split_name = str(frontier["split"])
    cfg = EnvConfig()
    data = _load_series_data(load_split(split_name), build_series_index())
    test_budget = _budget_total(data)
    summary_rows: List[Dict[str, object]] = []
    per_series_rows: List[Dict[str, object]] = []

    for floor in args.floors:
        point = _matching_point(frontier["points"], floor)
        weights = np.asarray(point["weights"], dtype=float)
        policies = {
            "direct": score_selector(weights),
            "static-floor": score_selector(weights, support_floor=1.0),
            "payment+completion": payment_gated_score_selector(
                weights, completion=True
            ),
            "payment-only": payment_gated_score_selector(
                weights, completion=False
            ),
        }
        by_kernel: Dict[str, List[EpisodeResult]] = {
            name: _test_episodes(selector, data, cfg)
            for name, selector in policies.items()
        }
        direct_stats = aggregate(by_kernel["direct"])
        _validate_frozen_direct(point, direct_stats)
        direct_by_series = {
            episode.series: float(episode.worst_csd)
            for episode in by_kernel["direct"]
            if episode.worst_csd is not None
        }
        direct_exclusion = {
            episode.series: float(episode.exclusion)
            for episode in by_kernel["direct"]
        }
        direct_welfare = float(direct_stats["welfare"])

        for kernel, episodes in by_kernel.items():
            stats = aggregate(episodes)
            spent = float(sum(year.spent for episode in episodes for year in episode.years if year.scored))
            values = [
                float(episode.worst_csd)
                for episode in episodes
                if episode.worst_csd is not None
            ]
            direct_values = [direct_by_series[episode.series] for episode in episodes if episode.worst_csd is not None]
            exclusion_contrast = _contrast(
                [float(episode.exclusion) for episode in episodes],
                [direct_exclusion[episode.series] for episode in episodes],
                seed=args.seed,
                n_boot=args.n_boot,
            )
            summary_rows.append(
                {
                    "floor": floor,
                    "kernel": kernel,
                    "worst_csd": stats["worst_csd"],
                    "mean_csd": stats["mean_csd"],
                    "welfare": stats["welfare"],
                    "exclusion": stats["exclusion"],
                    "spent": spent,
                    "budget_utilization": spent / test_budget,
                    "welfare_ratio_vs_direct": float(stats["welfare"]) / direct_welfare,
                    **_contrast(
                        values,
                        direct_values,
                        seed=args.seed,
                        n_boot=args.n_boot,
                    ),
                    "exclusion_diff_vs_direct": exclusion_contrast["diff_vs_direct"],
                    "exclusion_ci_lo": exclusion_contrast["ci_lo"],
                    "exclusion_ci_hi": exclusion_contrast["ci_hi"],
                    "exclusion_p_two_sided": exclusion_contrast["p_two_sided"],
                    "exclusion_wins_vs_direct": exclusion_contrast["wins_vs_direct"],
                }
            )
            for episode in episodes:
                per_series_rows.append(
                    {
                        "floor": floor,
                        "kernel": kernel,
                        "series": episode.series,
                        "n_years": len(episode.scored_years),
                        "worst_csd": episode.worst_csd,
                        "mean_csd": episode.mean_csd,
                        "welfare": episode.welfare,
                        "exclusion": episode.exclusion,
                        "spent": sum(year.spent for year in episode.years if year.scored),
                    }
                )

    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with PER_SERIES_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_series_rows[0]))
        writer.writeheader()
        writer.writerows(per_series_rows)

    manifest = {
        "analysis": "frozen project-score allocation-kernel intervention",
        "source_frontier": str(args.frontier.relative_to(ROOT)),
        "source_sha256": _sha256(args.frontier),
        "split": split_name,
        "floors": args.floors,
        "n_series": int(summary_rows[0]["n"]),
        "bootstrap_replicates": args.n_boot,
        "bootstrap_seed": args.seed,
        "hypothesis_test": "exact paired two-sided sign-flip over series",
        "warning": (
            "payment-prioritized score is a diagnostic, not standard MES; "
            "it orders affordable projects by the frozen score. Later score "
            "values may diverge because each rollout carries its own endogenous "
            "deficit history, although weights and feature definitions are fixed"
        ),
        "outputs": {
            "summary": str(SUMMARY_CSV.relative_to(ROOT)),
            "per_series": str(PER_SERIES_CSV.relative_to(ROOT)),
        },
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    logger.info(
        "%-22s %8s %10s %10s %9s %10s %20s %8s",
        "kernel", "floor", "CSD", "welfare", "excl", "d.direct", "95% CI", "p",
    )
    for row in summary_rows:
        logger.info(
            "%-22s %8.2f %10.4f %10.0f %9.4f %+10.4f [%+.4f,%+.4f] %8.4f",
            row["kernel"], row["floor"], row["worst_csd"], row["welfare"],
            row["exclusion"], row["diff_vs_direct"], row["ci_lo"], row["ci_hi"],
            row["p_two_sided"],
        )
    logger.info("wrote %s, %s, and %s", SUMMARY_CSV, PER_SERIES_CSV, MANIFEST_JSON)


if __name__ == "__main__":
    main()
