import numpy as np


def test_alpha_grid_extends_the_old_range_and_preserves_old_points():
    from iclr_wide_senior_audit import alpha_grid

    grid = alpha_grid()

    assert 0.0 in grid
    assert 2.8 in grid
    assert 3.0 in grid
    assert max(grid) >= 999.0
    assert np.all(np.diff(grid) > 0)


def test_training_only_selection_applies_both_safeguards():
    from iclr_wide_senior_audit import select_training_candidate

    rows = [
        {"alpha": 10.0, "worst_csd": 0.10, "welfare_ratio": 0.98, "exclusion_delta": -0.01},
        {"alpha": 20.0, "worst_csd": 0.11, "welfare_ratio": 1.00, "exclusion_delta": 0.01},
        {"alpha": 30.0, "worst_csd": 0.12, "welfare_ratio": 0.99, "exclusion_delta": 0.00},
        {"alpha": 40.0, "worst_csd": 0.13, "welfare_ratio": 1.01, "exclusion_delta": -0.01},
    ]

    selected = select_training_candidate(
        rows, welfare_floor=0.99, exclusion_delta_max=0.0
    )

    assert selected["alpha"] == 30.0
