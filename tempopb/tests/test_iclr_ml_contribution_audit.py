import numpy as np


def test_paired_summary_uses_learned_minus_static_direction():
    from iclr_ml_contribution_audit import paired_summary

    result = paired_summary([0.1, 0.2, 0.3], [0.2, 0.2, 0.4])
    assert np.isclose(result["difference"], -0.06666666666666667)
    assert result["wins"] == 2
    assert result["ties"] == 1
    assert result["losses"] == 0


def test_paired_summary_rejects_unpaired_inputs():
    from iclr_ml_contribution_audit import paired_summary

    try:
        paired_summary([0.1], [0.1, 0.2])
    except ValueError as exc:
        assert "equal length" in str(exc)
    else:
        raise AssertionError("unpaired inputs must fail")


def test_batch_cmaes_matches_sequential_cmaes():
    from iclr_cmaes import CMAESConfig, minimize
    from iclr_history_free_audit import minimize_batch

    config = CMAESConfig(generations=5, seed=7, sigma0=0.3)
    objective = lambda value: float(np.sum((value - 0.25) ** 2))
    sequential = minimize(objective, np.zeros(3), config)
    batched = minimize_batch(
        lambda population: [objective(row) for row in population],
        np.zeros(3),
        config,
    )

    assert sequential.n_evals == batched.n_evals
    assert sequential.best_f == batched.best_f
    assert np.array_equal(sequential.best_x, batched.best_x)
    assert sequential.history == batched.history
