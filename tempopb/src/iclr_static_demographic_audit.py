"""Fit a complexity-matched static age lookup and audit contextual learning.

This post-lock exploratory control assigns one positive multiplier to each of
the four prespecified age brackets.  It uses no ballot context, history, city,
or series identity, and exactly contains the fitted 60+ scalar family.  The
fit sees training editions only.  Frozen manuscript files under ``results/``
are read but never modified; outputs go to
``analysis-output/ml-contribution-audit``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from cohorts import AGE_BRACKETS, age_bracket
from iclr_cmaes import CMAESConfig
from iclr_corpus import build_series_index, load_series, load_split
from iclr_env import EnvConfig, RolloutState, aggregate, endowment_selector, rollout_selector
from iclr_history_free_audit import minimize_batch
from iclr_ml_contribution_audit import paired_summary
from iclr_reviewer_checks import senior_tilt_policy
from iclr_train import SeriesData, _load_series_data
from parse_pb import PBInstance


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "analysis-output" / "ml-contribution-audit"
SENIOR_TILT = ROOT / "results" / "iclr_senior_tilt.csv"
HISTORY_FREE = OUTPUT / "history_free_seed42_g30.json"
SPLIT = "temporal_2022"
AGE_LABELS = tuple(label for _, _, label in AGE_BRACKETS)
logger = logging.getLogger(__name__)

_WORKER_DATA: List[SeriesData] = []
_WORKER_CFG = EnvConfig()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def static_age_lookup_policy(log_multipliers: Sequence[float]):
    """Return a history-free four-bracket age endowment policy.

    Missing or invalid ages receive the uniform raw multiplier.  Positive
    factors are parameterized in log space and all raw endowments are rescaled
    to the election budget.
    """

    logits = np.asarray(log_multipliers, dtype=float)
    if logits.shape != (len(AGE_LABELS),):
        raise ValueError(
            f"expected four age logits, got shape {logits.shape}"
        )
    factors = {
        label: float(math.exp(value))
        for label, value in zip(AGE_LABELS, logits)
    }

    def _policy(inst: PBInstance, state: RolloutState) -> List[float]:
        del state
        n = len(inst.votes)
        if n == 0:
            return []
        base = inst.budget / n
        raw = [
            base * factors.get(age_bracket(vote.age), 1.0)
            for vote in inst.votes
        ]
        total = sum(raw)
        if total <= 0:
            return [base] * n
        scale = inst.budget / total
        return [value * scale for value in raw]

    return _policy


def _load_training_only(split_name: str) -> List[SeriesData]:
    split = load_split(split_name)
    index = build_series_index()
    rows: List[SeriesData] = []
    for key, years in split.train:
        ref = index[key]
        instances = load_series(ref, years=years)
        rows.append(
            SeriesData(
                ref=ref,
                train_years=tuple(years),
                test_years=(),
                train_only=dict(instances),
                all_years=dict(instances),
            )
        )
    return rows


def _training_loss(
    logits: Sequence[float], data: Sequence[SeriesData], cfg: EnvConfig
) -> float:
    selector = endowment_selector(static_age_lookup_policy(logits), cfg)
    episodes = [
        rollout_selector(
            row.ref,
            selector,
            score_years=row.train_years,
            cfg=cfg,
            instances=row.train_only,
        )
        for row in data
    ]
    return float(aggregate(episodes)["worst_csd"])


def _worker_init(split_name: str) -> None:
    global _WORKER_DATA, _WORKER_CFG
    _WORKER_DATA = _load_training_only(split_name)
    _WORKER_CFG = EnvConfig()


def _worker_objective(logits: Sequence[float]) -> float:
    return _training_loss(logits, _WORKER_DATA, _WORKER_CFG)


def _ordered_pool_map(
    pool: ProcessPoolExecutor, population: np.ndarray
) -> List[float]:
    return list(pool.map(_worker_objective, population.tolist()))


def _evaluate_test(
    age_logits: np.ndarray,
    history_free_weights: np.ndarray,
    alpha: float,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    from iclr_policy import linear_policy

    cfg = EnvConfig()
    data = _load_series_data(load_split(SPLIT), build_series_index())
    selectors = {
        "age_lookup": endowment_selector(
            static_age_lookup_policy(age_logits), cfg
        ),
        "history_free": endowment_selector(
            linear_policy(history_free_weights), cfg
        ),
        "static_60plus": endowment_selector(senior_tilt_policy(alpha), cfg),
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
        rows.append(
            {
                "series": item.ref.key,
                "test_years": "|".join(str(year) for year in item.test_years),
                "age_lookup_csd": float(episodes["age_lookup"].worst_csd),
                "history_free_csd": float(episodes["history_free"].worst_csd),
                "static_60plus_csd": float(episodes["static_60plus"].worst_csd),
                "age_lookup_welfare": int(episodes["age_lookup"].welfare),
                "history_free_welfare": int(episodes["history_free"].welfare),
                "static_60plus_welfare": int(episodes["static_60plus"].welfare),
                "age_lookup_exclusion": float(episodes["age_lookup"].exclusion),
                "history_free_exclusion": float(episodes["history_free"].exclusion),
                "static_60plus_exclusion": float(episodes["static_60plus"].exclusion),
            }
        )

    def values(name: str) -> List[float]:
        return [float(row[f"{name}_csd"]) for row in rows]

    summary: Dict[str, object] = {
        "age_lookup_minus_static_60plus": paired_summary(
            values("age_lookup"), values("static_60plus")
        ),
        "history_free_minus_age_lookup": paired_summary(
            values("history_free"), values("age_lookup")
        ),
        "policy_metrics": {
            name: {
                "mean_csd": float(np.mean(values(name))),
                "welfare": int(sum(int(row[f"{name}_welfare"]) for row in rows)),
                "mean_exclusion": float(
                    np.mean([float(row[f"{name}_exclusion"]) for row in rows])
                ),
            }
            for name in selectors
        },
        "evidence_boundary": (
            "Post-lock exploratory audit on 18 Warsaw series after the held-out "
            "outcomes were already known. It cannot support a new confirmatory "
            "claim or cross-city transfer claim."
        ),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--bound", type=float, default=3.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with SENIOR_TILT.open(newline="") as handle:
        alpha = float(next(csv.DictReader(handle))["alpha"])
    initial = np.zeros(len(AGE_LABELS))
    initial[-1] = math.log1p(alpha)
    config = CMAESConfig(
        sigma0=0.4,
        generations=args.generations,
        seed=args.seed,
        bound=args.bound,
    )
    population = config.popsize or (4 + int(3 * math.log(len(AGE_LABELS))))
    workers = args.workers or population
    parent_data = _load_training_only(SPLIT)
    parent_cfg = EnvConfig()
    initial_loss = _training_loss(initial, parent_data, parent_cfg)
    logger.info(
        "fitting static age lookup with seed=%d, generations=%d, workers=%d",
        args.seed,
        args.generations,
        workers,
    )
    logger.info("contained 60+ scalar train loss=%.6f", initial_loss)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(SPLIT,),
    ) as pool:
        result = minimize_batch(
            lambda xs: _ordered_pool_map(pool, xs),
            initial,
            config,
        )
    if initial_loss <= result.best_f:
        best_logits = initial
        best_loss = initial_loss
        selection_source = "contained_60plus_initial"
    else:
        best_logits = result.best_x
        best_loss = result.best_f
        selection_source = "cmaes"

    history_free_payload = json.loads(HISTORY_FREE.read_text())
    history_free_weights = np.asarray(
        history_free_payload["summary"]["best_weights"], dtype=float
    )
    rows, summary = _evaluate_test(best_logits, history_free_weights, alpha)
    summary.update(
        {
            "classification": "post-lock exploratory strong-baseline audit",
            "split": SPLIT,
            "seed": args.seed,
            "generations": args.generations,
            "workers": workers,
            "bound": args.bound,
            "population": population,
            "alpha": alpha,
            "initial_logits": initial.tolist(),
            "initial_train_loss": initial_loss,
            "best_logits": best_logits.tolist(),
            "best_multipliers": np.exp(best_logits).tolist(),
            "best_train_loss": best_loss,
            "selection_source": selection_source,
            "n_objective_evals": result.n_evals + 1,
            "inputs": {
                str(SENIOR_TILT.relative_to(ROOT)): _sha256(SENIOR_TILT),
                str(HISTORY_FREE.relative_to(ROOT)): _sha256(HISTORY_FREE),
                f"results/iclr_splits/{SPLIT}.json": _sha256(
                    ROOT / "results" / "iclr_splits" / f"{SPLIT}.json"
                ),
                "src/cohorts.py": _sha256(ROOT / "src" / "cohorts.py"),
                "src/iclr_static_demographic_audit.py": _sha256(
                    ROOT / "src" / "iclr_static_demographic_audit.py"
                ),
            },
        }
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = f"static_age_lookup_seed{args.seed}_g{args.generations}"
    with (OUTPUT / f"{stem}_per_series.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {"summary": summary, "history": result.history}
    (OUTPUT / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
