"""Shared paired-inference helpers for the ICLR analysis.

Confidence intervals use a paired percentile bootstrap.  Hypothesis tests use
an exact paired sign-flip randomization distribution, which imposes the null by
enumerating every sign assignment of the observed paired differences.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def exact_paired_sign_flip_pvalue(differences: Sequence[float]) -> float:
    """Return the exact two-sided sign-flip p-value for a paired mean.

    The test assumes paired differences are sign-exchangeable under the null.
    The paper has 18 series, so all ``2**18`` assignments are enumerated.  Work
    is chunked to avoid constructing a large dense sign matrix at once.
    """
    diff = np.asarray(differences, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan")
    if diff.size > 24:
        raise ValueError(
            "exact sign-flip enumeration is limited to 24 paired differences"
        )

    observed = abs(float(diff.mean()))
    total = 1 << diff.size
    extreme = 0
    shifts = np.arange(diff.size, dtype=np.uint64)
    for start in range(0, total, 32_768):
        masks = np.arange(start, min(start + 32_768, total), dtype=np.uint64)[:, None]
        bits = ((masks >> shifts) & 1).astype(np.int8)
        signs = 2 * bits - 1
        randomized_means = (signs * diff).mean(axis=1)
        extreme += int(
            np.count_nonzero(np.abs(randomized_means) >= observed - 1e-15)
        )
    return extreme / total


def monte_carlo_paired_sign_flip_pvalue(
    differences: Sequence[float], draws: int = 200_000, seed: int = 0
) -> float:
    """Sampled version of the same test, for more pairs than can be enumerated.

    Exhaustive enumeration costs ``2**n``; the cross-city corpus has more series
    than that allows. This draws sign vectors uniformly instead and applies the
    standard ``(1 + extreme) / (1 + draws)`` estimator, which never returns zero
    and stays valid as a randomization p-value. It assumes the same null as
    :func:`exact_paired_sign_flip_pvalue`: sign-exchangeable paired differences.

    Args:
        differences: Paired per-series differences.
        draws: Number of random sign assignments.
        seed: Seed fixing the draw, so the reported value is reproducible.

    Returns:
        The two-sided p-value estimate.
    """
    diff = np.asarray(differences, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan")

    observed = abs(float(diff.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = draws
    while remaining > 0:
        block = min(remaining, 20_000)
        signs = rng.integers(0, 2, size=(block, diff.size), dtype=np.int8) * 2 - 1
        means = (signs * diff).mean(axis=1)
        extreme += int(np.count_nonzero(np.abs(means) >= observed - 1e-15))
        remaining -= block
    return (1 + extreme) / (1 + draws)
