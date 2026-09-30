"""Known-buggy variants of laplace_bounded_count_v1, for auditor calibration.

Each function has the same signature and claims eps=1.0 pure DP, but is wrong.
`EXPECTED` records whether the simple threshold auditor should catch it
(part of the calibration story: silence is not safety).
"""
import numpy as np

EXPECTED = {
    "private_count_wrong_scale": "detectable",       # true privacy ~4*eps
    "private_count_no_noise": "detectable",          # deterministic output
    "private_count_mild_wrong_scale": "detectable",  # true privacy 2*eps (harder)
}


def private_count_wrong_scale(data, epsilon, rng):
    """BUG: Laplace scale 1/(4*eps) instead of 1/eps -> only ~4*eps-DP."""
    true_count = sum(1 for x in data if x == 1)
    return float(true_count + rng.laplace(scale=1.0 / (4.0 * epsilon)))


def private_count_no_noise(data, epsilon, rng):
    """BUG: forgets the noise entirely -> not DP for any finite epsilon."""
    return float(sum(1 for x in data if x == 1))


def private_count_mild_wrong_scale(data, epsilon, rng):
    """BUG: scale 1/(2*eps) -> 2*eps-DP; a subtler calibration probe."""
    true_count = sum(1 for x in data if x == 1)
    return float(true_count + rng.laplace(scale=1.0 / (2.0 * epsilon)))
