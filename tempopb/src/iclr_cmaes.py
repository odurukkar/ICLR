"""Minimal CMA-ES in numpy.

Self-contained on purpose: the project has no scipy or cma dependency, and a
paper whose optimizer is 120 auditable lines is easier to reproduce than one
that pins a third-party evolution-strategy release. The implementation follows
the standard (mu/mu_w, lambda)-CMA-ES with cumulative step-size adaptation
(Hansen, The CMA Evolution Strategy: A Tutorial).

Objectives here are noise-free deterministic replays, so no reevaluation or
noise handling is included.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["CMAESConfig", "CMAESResult", "minimize"]


@dataclass(frozen=True)
class CMAESConfig:
    """Strategy parameters. Defaults follow the standard tutorial settings."""

    sigma0: float = 0.5
    popsize: Optional[int] = None  # None -> 4 + floor(3 ln n)
    generations: int = 40
    seed: int = 42
    bound: float = 10.0  # symmetric box the samples are clipped into


@dataclass
class CMAESResult:
    """Best point found plus the full generation history."""

    best_x: np.ndarray
    best_f: float
    history: List[dict] = field(default_factory=list)
    n_evals: int = 0


def minimize(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    cfg: Optional[CMAESConfig] = None,
    on_generation: Optional[Callable[[int, dict], None]] = None,
) -> CMAESResult:
    """Minimize `objective` from `x0` with CMA-ES.

    Args:
        objective: Deterministic scalar function of a weight vector.
        x0: Initial mean.
        cfg: Strategy parameters.
        on_generation: Optional callback receiving (generation, record).

    Returns:
        The best point found and per-generation history.

    Raises:
        ValueError: When x0 is not one-dimensional.
    """
    cfg = cfg or CMAESConfig()
    mean = np.asarray(x0, dtype=float).copy()
    if mean.ndim != 1:
        raise ValueError(f"x0 must be 1-D, got shape {mean.shape}")
    n = mean.size
    rng = np.random.default_rng(cfg.seed)

    lam = cfg.popsize or (4 + int(3 * math.log(n)))
    mu = lam // 2
    raw_w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w = raw_w / raw_w.sum()
    mu_eff = 1.0 / np.sum(w**2)

    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c_s = (mu_eff + 2) / (n + mu_eff + 5)
    c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
    d_s = 1 + 2 * max(0.0, math.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_s
    chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

    p_c = np.zeros(n)
    p_s = np.zeros(n)
    cov = np.eye(n)
    sigma = cfg.sigma0

    result = CMAESResult(best_x=mean.copy(), best_f=float("inf"))

    for gen in range(cfg.generations):
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-14, None)
        sqrt_cov = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        zs = rng.standard_normal((lam, n))
        xs = np.clip(mean + sigma * zs @ sqrt_cov.T, -cfg.bound, cfg.bound)
        fs = np.array([objective(x) for x in xs])
        result.n_evals += lam

        order = np.argsort(fs)
        xs, fs, zs = xs[order], fs[order], zs[order]
        if fs[0] < result.best_f:
            result.best_f = float(fs[0])
            result.best_x = xs[0].copy()

        old_mean = mean.copy()
        mean = w @ xs[:mu]

        inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        step = (mean - old_mean) / sigma
        p_s = (1 - c_s) * p_s + math.sqrt(c_s * (2 - c_s) * mu_eff) * (inv_sqrt @ step)
        h_sig = float(
            np.linalg.norm(p_s)
            / math.sqrt(1 - (1 - c_s) ** (2 * (gen + 1)))
            / chi_n
            < 1.4 + 2 / (n + 1)
        )
        p_c = (1 - c_c) * p_c + h_sig * math.sqrt(c_c * (2 - c_c) * mu_eff) * step

        artmp = (xs[:mu] - old_mean) / sigma
        cov = (
            (1 - c_1 - c_mu) * cov
            + c_1 * (np.outer(p_c, p_c) + (1 - h_sig) * c_c * (2 - c_c) * cov)
            + c_mu * (artmp.T @ np.diag(w) @ artmp)
        )
        cov = np.triu(cov) + np.triu(cov, 1).T  # enforce symmetry
        sigma *= math.exp((c_s / d_s) * (np.linalg.norm(p_s) / chi_n - 1))

        record = {
            "generation": gen,
            "best_f": float(fs[0]),
            "median_f": float(np.median(fs)),
            "sigma": float(sigma),
            "best_x": xs[0].tolist(),
            "mean_x": mean.tolist(),
        }
        result.history.append(record)
        if on_generation is not None:
            on_generation(gen, record)
        logger.info(
            "gen %3d  best=%.6f  median=%.6f  sigma=%.4f",
            gen, fs[0], np.median(fs), sigma,
        )

    return result
