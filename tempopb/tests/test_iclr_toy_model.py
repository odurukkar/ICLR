"""Regression for a discarded reachability theorem outside the paper release."""

from __future__ import annotations

import pytest

from iclr_toy_model import (
    build_instance,
    discarded_one_orientation_formula,
    measured_csd,
)
from rules import mes_with_endowments


def test_old_width_formula_misses_reverse_cap_orientation() -> None:
    """Catches treating one cap orientation as the whole attainable set."""
    alpha, kappa = 0.125, 1.1
    inst, groups = build_instance(n=800, alpha=alpha, n_proj=80, budget=8000.0)
    kappa_b = kappa
    kappa_a = (1.0 - (1.0 - alpha) * kappa_b) / alpha
    beta = [10.0 * (kappa_a if group == "A" else kappa_b) for group in groups]

    winners = mes_with_endowments(inst, endowments=beta, completion=False)
    measured = measured_csd(inst, groups, winners)

    assert measured == pytest.approx(0.7, abs=1e-12)
    old_value = discarded_one_orientation_formula(alpha, kappa)
    assert old_value == pytest.approx(1.0 / 70.0)
    assert measured > 40 * old_value
