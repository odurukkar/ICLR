"""Regression tests for paired uncertainty and hypothesis testing."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "p_value",
    [
        pytest.param(
            lambda a, b: __import__("iclr_significance")._paired_bootstrap(
                a, b, seed=7, n_boot=99
            )[3],
            id="frontier",
        ),
        pytest.param(
            lambda a, b: __import__("iclr_endowment_stats")._bootstrap(
                list(a), list(b), seed=7, n_boot=99
            )[3],
            id="endowment",
        ),
        pytest.param(
            lambda a, b: __import__("iclr_scheme_robustness")._boot(
                list(a), list(b), seed=7, n_boot=99
            )[1],
            id="group-partition",
        ),
    ],
)
def test_p_value_is_exact_paired_sign_flip(p_value) -> None:
    """P-values must impose the paired null, not tail-count raw bootstraps."""
    # Paired differences are -1, -2, -3.  Of all 2^3 sign assignments, only
    # the all-positive and all-negative assignments have |mean| >= 2, so the
    # hand-derived exact two-sided p-value is 2/8 = 0.25.
    actual = p_value([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])

    assert actual == pytest.approx(0.25)


def test_exact_test_returns_one_when_every_paired_difference_is_zero() -> None:
    """A point mass at the null must not be reported as significant."""
    from iclr_significance import _paired_bootstrap

    _, _, _, p_value, _ = _paired_bootstrap(
        [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], seed=7, n_boot=99
    )

    assert p_value == pytest.approx(1.0)
