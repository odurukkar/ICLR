"""Refit and audit the history-free learned endowment control.

The original control saved only aggregate metrics, and its source accumulated
floating-point values in hash-dependent set order.  Its exact optimizer path
therefore cannot be reconstructed from the aggregate alone.  This post-lock
runner uses the same model, split, objective, optimizer seed, and budget after
canonicalizing that accumulation order.  It records the new deterministic
weights and produces the missing paired comparison against the fitted 60+
scalar.  Candidate evaluations within each generation run in separate
processes; the optimizer updates and objective are otherwise identical to
``iclr_cmaes.minimize``.

All outputs go to ``analysis-output/ml-contribution-audit``.  Frozen files in
``results/`` are read but never modified.
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
from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np

from iclr_cmaes import CMAESConfig, CMAESResult
from iclr_corpus import build_series_index, load_series, load_split
from iclr_env import EnvConfig, aggregate, endowment_selector, rollout_selector
from iclr_ml_contribution_audit import paired_summary
from iclr_policy import FEATURE_NAMES, N_FEATURES, linear_policy
from iclr_reviewer_checks import senior_tilt_policy
from iclr_train import SeriesData, _load_series_data


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "analysis-output" / "ml-contribution-audit"
SENIOR_TILT = ROOT / "results" / "iclr_senior_tilt.csv"
HISTORICAL_CONTROL = ROOT / "results" / "iclr_static_tilt.csv"
HISTORICAL_LOG = ROOT / "results" / "iclr_static_tilt_log.txt"
SPLIT = "temporal_2022"
HISTORY_IDX = (0, 1)
logger = logging.getLogger(__name__)

_WORKER_DATA: List[SeriesData] = []
_WORKER_CFG = EnvConfig()
_WORKER_MASK = np.ones(N_FEATURES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Dict[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def minimize_batch(
    evaluate_batch: Callable[[np.ndarray], Sequence[float]],
    x0: np.ndarray,
    cfg: CMAESConfig,
) -> CMAESResult:
    """CMA-ES with ordered batch objective evaluation.

    The algebra and RNG calls mirror ``iclr_cmaes.minimize``.  Only the line
    that evaluates the population is replaced by one ordered batch call.
    """
    mean = np.asarray(x0, dtype=float).copy()
    if mean.ndim != 1:
        raise ValueError(f"x0 must be 1-D, got shape {mean.shape}")
    n = mean.size
    rng = np.random.default_rng(cfg.seed)

    lam = cfg.popsize or (4 + int(3 * math.log(n)))
    mu = lam // 2
    raw_w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    recombination = raw_w / raw_w.sum()
    mu_eff = 1.0 / np.sum(recombination**2)

    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c_s = (mu_eff + 2) / (n + mu_eff + 5)
    c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(
        1 - c_1,
        2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff),
    )
    d_s = 1 + 2 * max(0.0, math.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_s
    chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

    p_c = np.zeros(n)
    p_s = np.zeros(n)
    cov = np.eye(n)
    sigma = cfg.sigma0
    result = CMAESResult(best_x=mean.copy(), best_f=float("inf"))

    for generation in range(cfg.generations):
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-14, None)
        sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        zs = rng.standard_normal((lam, n))
        xs = np.clip(mean + sigma * zs @ sqrt_cov.T, -cfg.bound, cfg.bound)
        fs = np.asarray(evaluate_batch(xs), dtype=float)
        if fs.shape != (lam,):
            raise ValueError(f"batch evaluator returned shape {fs.shape}, expected {(lam,)}")
        result.n_evals += lam

        order = np.argsort(fs)
        xs, fs, zs = xs[order], fs[order], zs[order]
        if fs[0] < result.best_f:
            result.best_f = float(fs[0])
            result.best_x = xs[0].copy()

        old_mean = mean.copy()
        mean = recombination @ xs[:mu]
        inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        step = (mean - old_mean) / sigma
        p_s = (1 - c_s) * p_s + math.sqrt(
            c_s * (2 - c_s) * mu_eff
        ) * (inv_sqrt @ step)
        h_sig = float(
            np.linalg.norm(p_s)
            / math.sqrt(1 - (1 - c_s) ** (2 * (generation + 1)))
            / chi_n
            < 1.4 + 2 / (n + 1)
        )
        p_c = (1 - c_c) * p_c + h_sig * math.sqrt(
            c_c * (2 - c_c) * mu_eff
        ) * step
        artmp = (xs[:mu] - old_mean) / sigma
        cov = (
            (1 - c_1 - c_mu) * cov
            + c_1
            * (
                np.outer(p_c, p_c)
                + (1 - h_sig) * c_c * (2 - c_c) * cov
            )
            + c_mu * (artmp.T @ np.diag(recombination) @ artmp)
        )
        cov = np.triu(cov) + np.triu(cov, 1).T
        sigma *= math.exp((c_s / d_s) * (np.linalg.norm(p_s) / chi_n - 1))

        record = {
            "generation": generation,
            "best_f": float(fs[0]),
            "median_f": float(np.median(fs)),
            "sigma": float(sigma),
            "best_x": xs[0].tolist(),
            "mean_x": mean.tolist(),
        }
        result.history.append(record)
        logger.info(
            "gen %3d best=%.6f median=%.6f sigma=%.4f",
            generation,
            fs[0],
            np.median(fs),
            sigma,
        )
    return result


def _worker_init(split_name: str, mask: Sequence[float]) -> None:
    """Load training-only instances once in each candidate worker."""
    global _WORKER_DATA, _WORKER_CFG, _WORKER_MASK
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
    _WORKER_DATA = rows
    _WORKER_CFG = EnvConfig()
    _WORKER_MASK = np.asarray(mask, dtype=float)


def _worker_objective(weights: Sequence[float]) -> float:
    vector = np.asarray(weights, dtype=float) * _WORKER_MASK
    selector = endowment_selector(linear_policy(vector), _WORKER_CFG)
    episodes = [
        rollout_selector(
            row.ref,
            selector,
            score_years=row.train_years,
            cfg=_WORKER_CFG,
            instances=row.train_only,
        )
        for row in _WORKER_DATA
    ]
    return float(aggregate(episodes)["worst_csd"])


def _ordered_pool_map(
    pool: ProcessPoolExecutor, population: np.ndarray
) -> List[float]:
    return list(pool.map(_worker_objective, population.tolist()))


def _evaluate_test(weights: np.ndarray, alpha: float) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    cfg = EnvConfig()
    data = _load_series_data(load_split(SPLIT), build_series_index())
    policies = {
        "history_free": endowment_selector(linear_policy(weights), cfg),
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
            for name, selector in policies.items()
        }
        a = episodes["history_free"]
        b = episodes["static_60plus"]
        if a.worst_csd is None or b.worst_csd is None:
            raise RuntimeError(f"missing CSD for {item.ref.key}")
        rows.append(
            {
                "series": item.ref.key,
                "test_years": "|".join(str(year) for year in item.test_years),
                "history_free_csd": float(a.worst_csd),
                "static_60plus_csd": float(b.worst_csd),
                "history_free_minus_static": float(a.worst_csd - b.worst_csd),
                "history_free_welfare": int(a.welfare),
                "static_60plus_welfare": int(b.welfare),
                "history_free_exclusion": float(a.exclusion),
                "static_60plus_exclusion": float(b.exclusion),
            }
        )
    summary: Dict[str, object] = paired_summary(
        [float(row["history_free_csd"]) for row in rows],
        [float(row["static_60plus_csd"]) for row in rows],
    )
    summary.update(
        {
            "contrast": "history-free contextual map minus fitted 60+ scalar tilt",
            "metric": "mean per-series worst-cohort CSD; lower is better",
            "history_free_welfare": sum(
                int(row["history_free_welfare"]) for row in rows
            ),
            "static_60plus_welfare": sum(
                int(row["static_60plus_welfare"]) for row in rows
            ),
            "history_free_exclusion": float(
                np.mean([float(row["history_free_exclusion"]) for row in rows])
            ),
            "static_60plus_exclusion": float(
                np.mean([float(row["static_60plus_exclusion"]) for row in rows])
            ),
            "evidence_boundary": (
                "Post-lock paired diagnostic on 18 Warsaw series. It does not "
                "establish cross-city transfer or independent replication."
            ),
        }
    )
    # Rename the generic key emitted by paired_summary for clarity.
    summary["history_free_mean"] = summary.pop("learned_mean")
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    mask = np.ones(N_FEATURES)
    mask[list(HISTORY_IDX)] = 0.0
    config = CMAESConfig(
        sigma0=0.4,
        generations=args.generations,
        seed=args.seed,
    )
    population = config.popsize or (4 + int(3 * math.log(N_FEATURES)))
    workers = args.workers or population
    logger.info(
        "fitting %s with seed=%d, generations=%d, workers=%d",
        [name for index, name in enumerate(FEATURE_NAMES) if mask[index]],
        args.seed,
        args.generations,
        workers,
    )
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(SPLIT, mask.tolist()),
    ) as pool:
        result = minimize_batch(
            lambda xs: _ordered_pool_map(pool, xs),
            np.zeros(N_FEATURES),
            config,
        )
    weights = result.best_x * mask

    with SENIOR_TILT.open(newline="") as handle:
        alpha = float(next(csv.DictReader(handle))["alpha"])
    rows, summary = _evaluate_test(weights, alpha)
    summary.update(
        {
            "seed": args.seed,
            "generations": args.generations,
            "workers": workers,
            "best_train_loss": result.best_f,
            "best_weights": weights.tolist(),
            "n_objective_evals": result.n_evals,
            "alpha": alpha,
            "split": SPLIT,
        }
    )
    with HISTORICAL_CONTROL.open(newline="") as handle:
        historical_rows = list(csv.DictReader(handle))
    historical = next(
        row
        for row in historical_rows
        if row["policy"] == "static-only (no history)"
    )
    protocol: Dict[str, object] = {
        "classification": "deterministic post-lock refit",
        "split": SPLIT,
        "objective": "training mean per-series worst-cohort CSD",
        "feature_names": list(FEATURE_NAMES),
        "masked_feature_indices": list(HISTORY_IDX),
        "masked_feature_names": [FEATURE_NAMES[index] for index in HISTORY_IDX],
        "seed": args.seed,
        "generations": args.generations,
        "sigma0": config.sigma0,
        "bound": config.bound,
        "population": population,
    }
    source_paths = (
        ROOT / "src" / "cohorts.py",
        ROOT / "src" / "iclr_cmaes.py",
        ROOT / "src" / "iclr_env.py",
        ROOT / "src" / "iclr_history_free_audit.py",
        ROOT / "src" / "iclr_policy.py",
        ROOT / "src" / "iclr_static_tilt.py",
        ROOT / "results" / "iclr_splits" / f"{SPLIT}.json",
        SENIOR_TILT,
        HISTORICAL_CONTROL,
        HISTORICAL_LOG,
    )
    summary.update(
        {
            "protocol": protocol,
            "protocol_sha256": _canonical_sha256(protocol),
            "source_sha256": {
                str(path.relative_to(ROOT)): _sha256(path)
                for path in source_paths
            },
            "historical_control": {
                "reported_test_mean": float(historical["test_worst_csd"]),
                "reported_test_welfare": float(historical["test_welfare"]),
                "reported_test_exclusion": float(historical["test_exclusion"]),
                "status": (
                    "historical aggregate only; not an equality target because "
                    "the original weights were not saved and the original "
                    "floating-point accumulation was hash-order dependent"
                ),
            },
        }
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = f"history_free_seed{args.seed}_g{args.generations}"
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
