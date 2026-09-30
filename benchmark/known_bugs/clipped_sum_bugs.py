"""Known-buggy variants of clipped_bounded_sum_v1, for auditor calibration."""
import numpy as np

B = 10.0

EXPECTED = {
    "private_sum_no_clipping": "detectable",     # unbounded sensitivity; use pairs w/ huge values
    "private_sum_wrong_scale": "detectable",     # scale 1/eps instead of B/eps -> ~B*eps-DP
    "private_sum_deterministic_empty": "detectable",  # returns exact 0.0 on empty data
}


def private_sum_no_clipping(data, epsilon, rng):
    """BUG: skips clipping; a single value of 100 gives sensitivity 100, not B=10."""
    return float(sum(float(x) for x in data) + rng.laplace(scale=B / epsilon))


def private_sum_wrong_scale(data, epsilon, rng):
    """BUG: uses count sensitivity (1) instead of sum sensitivity (B) for the scale."""
    clipped = [min(max(float(x), 0.0), B) for x in data]
    return float(sum(clipped) + rng.laplace(scale=1.0 / epsilon))


def private_sum_deterministic_empty(data, epsilon, rng):
    """BUG: special-cases empty input with a deterministic exact 0.0 release."""
    if len(data) == 0:
        return 0.0
    clipped = [min(max(float(x), 0.0), B) for x in data]
    return float(sum(clipped) + rng.laplace(scale=B / epsilon))
