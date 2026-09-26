"""Test whether the learned endowment map beats the fitted 60+ scalar control.

This is a post-lock diagnostic.  It does not refit either policy and writes
outside ``results/`` so the frozen manuscript artifacts remain untouched.
Both policies are replayed on the same temporal-2022 held-out series.

Writes:
    analysis-output/ml-contribution-audit/per_series.csv
    analysis-output/ml-contribution-audit/summary.json
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from iclr_corpus import build_series_index, load_split
from iclr_env import EnvConfig, endowment_selector, rollout_selector
from iclr_policy import linear_policy
from iclr_reviewer_checks import senior_tilt_policy
from iclr_stats import exact_paired_sign_flip_pvalue
from iclr_train import _load_series_data


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUTPUT = ROOT / "analysis-output" / "ml-contribution-audit"
FULL_RUN = RESULTS / "iclr_train" / "run_main_seed42.json"
SENIOR_TILT = RESULTS / "iclr_senior_tilt.csv"
SPLIT = "temporal_2022"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260813


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_summary(
    learned: Sequence[float], static: Sequence[float]
) -> Dict[str, float | int]:
    """Summarize paired learned-minus-static differences."""
    a = np.asarray(learned, dtype=float)
    b = np.asarray(static, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("paired inputs must be non-empty vectors of equal length")
    diff = a - b
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = diff[
        rng.integers(0, diff.size, size=(BOOTSTRAP_DRAWS, diff.size))
    ].mean(axis=1)
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else float("nan")
    wins = int(np.count_nonzero(diff < 0))
    losses = int(np.count_nonzero(diff > 0))
    ties = int(diff.size - wins - losses)
    return {
        "n_series": int(diff.size),
        "learned_mean": float(a.mean()),
        "static_60plus_mean": float(b.mean()),
        "difference": float(diff.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "p_two_sided_exact_sign_flip": float(
            exact_paired_sign_flip_pvalue(diff)
        ),
        "paired_cohens_dz": float(diff.mean() / sd) if sd > 0 else float("nan"),
        "median_difference": float(np.median(diff)),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def main() -> None:
    full_payload = json.loads(FULL_RUN.read_text())
    weights = np.asarray(full_payload["best_weights"], dtype=float)
    with SENIOR_TILT.open(newline="") as handle:
        tilt_rows = list(csv.DictReader(handle))
    alpha = float(tilt_rows[0]["alpha"])

    cfg = EnvConfig()
    data = _load_series_data(load_split(SPLIT), build_series_index())
    selectors = {
        "learned": endowment_selector(linear_policy(weights), cfg),
        "static_60plus": endowment_selector(senior_tilt_policy(alpha), cfg),
    }

    rows: List[Dict[str, object]] = []
    for item in data:
        if not item.test_years:
            continue
        episodes = {
            name: rollout_selector(
                item.ref,
                selector,
                score_years=item.test_years,
                cfg=cfg,
                instances=item.all_years,
            )
            for name, selector in selectors.items()
        }
        learned_csd = episodes["learned"].worst_csd
        static_csd = episodes["static_60plus"].worst_csd
        if learned_csd is None or static_csd is None:
            raise RuntimeError(f"missing CSD for {item.ref.key}")
        rows.append(
            {
                "series": item.ref.key,
                "test_years": "|".join(str(year) for year in item.test_years),
                "learned_csd": float(learned_csd),
                "static_60plus_csd": float(static_csd),
                "learned_minus_static": float(learned_csd - static_csd),
                "learned_welfare": int(episodes["learned"].welfare),
                "static_60plus_welfare": int(episodes["static_60plus"].welfare),
                "learned_exclusion": float(episodes["learned"].exclusion),
                "static_60plus_exclusion": float(
                    episodes["static_60plus"].exclusion
                ),
            }
        )

    summary = paired_summary(
        [float(row["learned_csd"]) for row in rows],
        [float(row["static_60plus_csd"]) for row in rows],
    )
    summary.update(
        {
            "contrast": "frozen learned endowment map minus fitted 60+ scalar tilt",
            "metric": "mean per-series worst-cohort CSD; lower is better",
            "split": SPLIT,
            "alpha": alpha,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "learned_welfare": int(
                sum(int(row["learned_welfare"]) for row in rows)
            ),
            "static_60plus_welfare": int(
                sum(int(row["static_60plus_welfare"]) for row in rows)
            ),
            "learned_exclusion": float(
                np.mean([float(row["learned_exclusion"]) for row in rows])
            ),
            "static_60plus_exclusion": float(
                np.mean([float(row["static_60plus_exclusion"]) for row in rows])
            ),
            "inputs": {
                str(FULL_RUN.relative_to(ROOT)): _sha256(FULL_RUN),
                str(SENIOR_TILT.relative_to(ROOT)): _sha256(SENIOR_TILT),
                f"results/iclr_splits/{SPLIT}.json": _sha256(
                    RESULTS / "iclr_splits" / f"{SPLIT}.json"
                ),
            },
            "evidence_boundary": (
                "Post-lock paired diagnostic on 18 Warsaw series. It does not "
                "establish cross-city transfer or independent replication."
            ),
        }
    )

    # Catch replay drift before publishing the diagnostic.
    expected_learned = float(full_payload["learned"]["test"]["worst_csd"])
    expected_static = float(tilt_rows[0]["test_worst_csd"])
    if not np.isclose(summary["learned_mean"], expected_learned, atol=1e-12):
        raise RuntimeError("learned-map replay does not match the frozen aggregate")
    if not np.isclose(summary["static_60plus_mean"], expected_static, atol=1e-12):
        raise RuntimeError("60+ tilt replay does not match the frozen aggregate")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "per_series.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
