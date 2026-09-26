"""Audit whether a wider 60+ scalar explains the contextual Warsaw result.

The published scalar search stopped at alpha=3.  This post-lock exploratory
audit keeps every old grid point and adds a logarithmic tail through alpha=999.
It selects policies using training editions only, both unconstrained and under
predeclared welfare/exclusion safeguards, then evaluates the frozen choices on
the same held-out series.  It does not modify ``results/``.
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from iclr_corpus import build_series_index, load_split
from iclr_env import EnvConfig, aggregate, endowment_selector, rollout_selector, uniform_policy
from iclr_ml_contribution_audit import paired_summary
from iclr_policy import linear_policy
from iclr_reviewer_checks import senior_tilt_policy
from iclr_static_demographic_audit import static_age_lookup_policy
from iclr_train import SeriesData, _load_series_data


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "analysis-output" / "ml-contribution-audit"
HISTORY_FREE = OUTPUT / "history_free_seed42_g30.json"
AGE_LOOKUP = OUTPUT / "static_age_lookup_seed42_g30.json"
SPLIT = "temporal_2022"
WELFARE_FLOOR = 0.99
EXCLUSION_DELTA_MAX = 0.0

_WORKER_DATA: List[SeriesData] = []
_WORKER_CFG = EnvConfig()


def alpha_grid() -> tuple[float, ...]:
    """Published 0:.05:3 grid plus a log-spaced tail to alpha=999."""

    old = [round(0.05 * step, 2) for step in range(61)]
    tail = [float(multiplier - 1.0) for multiplier in np.geomspace(1.0, 1000.0, 41)]
    return tuple(sorted(set(old + tail)))


def select_training_candidate(
    rows: Sequence[Mapping[str, float]],
    *,
    welfare_floor: float,
    exclusion_delta_max: float,
) -> Dict[str, float]:
    """Choose lowest training CSD among candidates passing both safeguards."""

    eligible = [
        row
        for row in rows
        if float(row["welfare_ratio"]) >= welfare_floor
        and float(row["exclusion_delta"]) <= exclusion_delta_max
    ]
    if not eligible:
        raise ValueError("no scalar candidate passes both training safeguards")
    return dict(min(eligible, key=lambda row: (float(row["worst_csd"]), float(row["alpha"]))))


def _training_metrics(
    alpha: float, data: Sequence[SeriesData], cfg: EnvConfig
) -> Dict[str, float]:
    selector = endowment_selector(senior_tilt_policy(alpha), cfg)
    episodes = [
        rollout_selector(
            row.ref,
            selector,
            score_years=row.train_years,
            cfg=cfg,
            instances=row.train_only,
        )
        for row in data
        if row.train_years
    ]
    stats = aggregate(episodes)
    return {
        "alpha": float(alpha),
        "worst_csd": float(stats["worst_csd"]),
        "welfare": float(stats["welfare"]),
        "exclusion": float(stats["exclusion"]),
    }


def _worker_init() -> None:
    global _WORKER_DATA, _WORKER_CFG
    split = load_split(SPLIT)
    _WORKER_DATA = _load_series_data(split, build_series_index())
    _WORKER_CFG = EnvConfig()


def _worker_metrics(alpha: float) -> Dict[str, float]:
    return _training_metrics(alpha, _WORKER_DATA, _WORKER_CFG)


def _evaluate_selected(
    alphas: Mapping[str, float],
    history_free_weights: np.ndarray,
    age_logits: np.ndarray,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    cfg = EnvConfig()
    data = _load_series_data(load_split(SPLIT), build_series_index())
    selectors = {
        **{
            name: endowment_selector(senior_tilt_policy(alpha), cfg)
            for name, alpha in alphas.items()
        },
        "history_free": endowment_selector(linear_policy(history_free_weights), cfg),
        "age_lookup": endowment_selector(static_age_lookup_policy(age_logits), cfg),
    }
    rows: List[Dict[str, object]] = []
    for item in data:
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
        if any(episode.worst_csd is None for episode in episodes.values()):
            raise RuntimeError(f"missing CSD for {item.ref.key}")
        row: Dict[str, object] = {
            "series": item.ref.key,
            "test_years": "|".join(str(year) for year in item.test_years),
        }
        for name, episode in episodes.items():
            row[f"{name}_csd"] = float(episode.worst_csd)
            row[f"{name}_welfare"] = int(episode.welfare)
            row[f"{name}_exclusion"] = float(episode.exclusion)
        rows.append(row)

    def values(name: str) -> List[float]:
        return [float(row[f"{name}_csd"]) for row in rows]

    contrasts = {
        f"history_free_minus_{name}": paired_summary(
            values("history_free"), values(name)
        )
        for name in alphas
    }
    contrasts["age_lookup_minus_scalar_wide"] = paired_summary(
        values("age_lookup"), values("scalar_wide")
    )
    metrics = {
        name: {
            "mean_csd": float(np.mean(values(name))),
            "welfare": int(sum(int(row[f"{name}_welfare"]) for row in rows)),
            "mean_exclusion": float(
                np.mean([float(row[f"{name}_exclusion"]) for row in rows])
            ),
        }
        for name in selectors
    }
    return rows, {"contrasts": contrasts, "policy_metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cfg = EnvConfig()
    parent_data = _load_series_data(load_split(SPLIT), build_series_index())
    mes_selector = endowment_selector(uniform_policy, cfg)
    mes_episodes = [
        rollout_selector(
            row.ref,
            mes_selector,
            score_years=row.train_years,
            cfg=cfg,
            instances=row.train_only,
        )
        for row in parent_data
        if row.train_years
    ]
    mes = aggregate(mes_episodes)

    grid = alpha_grid()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init) as pool:
        rows = list(pool.map(_worker_metrics, grid))
    for row in rows:
        row["welfare_ratio"] = float(row["welfare"] / mes["welfare"])
        row["exclusion_delta"] = float(row["exclusion"] - mes["exclusion"])
    rows.sort(key=lambda row: float(row["alpha"]))

    wide = dict(min(rows, key=lambda row: (float(row["worst_csd"]), float(row["alpha"]))))
    safe = select_training_candidate(
        rows,
        welfare_floor=WELFARE_FLOOR,
        exclusion_delta_max=EXCLUSION_DELTA_MAX,
    )
    old = dict(
        min(
            (row for row in rows if float(row["alpha"]) <= 3.0),
            key=lambda row: (float(row["worst_csd"]), float(row["alpha"])),
        )
    )

    history_payload = json.loads(HISTORY_FREE.read_text())
    age_payload = json.loads(AGE_LOOKUP.read_text())
    test_rows, test_summary = _evaluate_selected(
        {
            "scalar_old_grid": float(old["alpha"]),
            "scalar_wide": float(wide["alpha"]),
            "scalar_safeguarded": float(safe["alpha"]),
        },
        np.asarray(history_payload["summary"]["best_weights"], dtype=float),
        np.asarray(age_payload["summary"]["best_logits"], dtype=float),
    )
    summary: Dict[str, object] = {
        "classification": "post-lock exploratory scalar-range audit",
        "split": SPLIT,
        "grid_size": len(grid),
        "grid_max_alpha": max(grid),
        "mes_training": mes,
        "training_selection": {
            "old_grid": old,
            "wide_unconstrained": wide,
            "wide_safeguarded": safe,
        },
        "safeguards": {
            "welfare_ratio_min": WELFARE_FLOOR,
            "exclusion_delta_max": EXCLUSION_DELTA_MAX,
        },
        **test_summary,
        "evidence_boundary": (
            "Post-lock exploratory audit on already-known Warsaw outcomes. "
            "Selections use training editions only, but the analysis family "
            "was designed after inspecting prior held-out results."
        ),
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "wide_senior_grid_training.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT / "wide_senior_selected_per_series.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_rows[0]))
        writer.writeheader()
        writer.writerows(test_rows)
    (OUTPUT / "wide_senior_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
