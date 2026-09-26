import math

import numpy as np
import pytest

from iclr_env import RolloutState
from iclr_reviewer_checks import senior_tilt_policy
from parse_pb import PBInstance, Vote


def _instance() -> PBInstance:
    return PBInstance(
        path="test://static-age-lookup",
        meta={"budget": "500"},
        votes=[
            Vote("young", (), age=20, sex="F"),
            Vote("early", (), age=30, sex="M"),
            Vote("middle", (), age=50, sex="F"),
            Vote("senior", (), age=70, sex="M"),
            Vote("missing", (), age=None, sex=""),
        ],
    )


def test_zero_age_logits_reproduce_uniform_endowments():
    from iclr_static_demographic_audit import static_age_lookup_policy

    observed = static_age_lookup_policy(np.zeros(4))(_instance(), RolloutState())

    assert observed == [100.0] * 5


def test_age_lookup_exactly_contains_the_fitted_senior_tilt_family():
    from iclr_static_demographic_audit import static_age_lookup_policy

    alpha = 2.8
    logits = np.array([0.0, 0.0, 0.0, math.log1p(alpha)])
    lookup = static_age_lookup_policy(logits)(_instance(), RolloutState())
    scalar = senior_tilt_policy(alpha)(_instance(), RolloutState())

    assert np.allclose(lookup, scalar, rtol=0.0, atol=1e-12)


def test_age_lookup_rejects_the_wrong_number_of_logits():
    from iclr_static_demographic_audit import static_age_lookup_policy

    with pytest.raises(ValueError, match="four age logits"):
        static_age_lookup_policy(np.zeros(3))
