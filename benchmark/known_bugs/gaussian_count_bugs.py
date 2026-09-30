"""Known-buggy variants of gaussian_bounded_count_v1, for auditor calibration.

This task is marked `partially_auditable`: a pure-epsilon threshold auditor
cannot cleanly falsify an (eps, delta) claim, so several of these are expected
to be MISSED. That is the point -- the calibration table has to show where the
auditor is blind, otherwise `not_falsified` cannot be interpreted.
"""
import math

EXPECTED = {
    # uses the Laplace/pure-DP formula sigma = 1/eps, ignoring delta entirely.
    # At eps=1, delta=1e-5 the correct sigma is ~4.84, so this is ~4.8x too
    # little noise. Large enough that even a pure-eps auditor should see it.
    "private_count_gaussian_pure_dp_sigma": "detectable",
    # drops the sqrt(2 ln(1.25/delta)) factor but keeps a plausible shape
    "private_count_gaussian_ignores_delta": "detectable",
    # sqrt applied to the wrong sub-expression: sqrt(2) * ln(1.25/delta) instead
    # of sqrt(2 * ln(1.25/delta)). This gives sigma ~16.0, i.e. MORE noise than
    # required -> still private, must not be flagged.
    "private_count_gaussian_sqrt_misplaced": "likely_missed",
    # variance used where standard deviation belongs: scale = sigma^2. At these
    # parameters sigma^2 ~ 23.4 > sigma, so again over-noised and private.
    "private_count_gaussian_variance_for_sigma": "likely_missed",
}


def private_count_gaussian_pure_dp_sigma(data, epsilon, delta, rng):
    """BUG: sigma = 1/epsilon -- the pure-DP calibration, delta unused."""
    true_count = sum(1 for x in data if x == 1)
    return float(true_count + rng.normal(scale=1.0 / epsilon))


def private_count_gaussian_ignores_delta(data, epsilon, delta, rng):
    """BUG: sigma = sqrt(2)/epsilon -- keeps a sqrt(2) but drops the log term."""
    true_count = sum(1 for x in data if x == 1)
    return float(true_count + rng.normal(scale=math.sqrt(2.0) / epsilon))


def private_count_gaussian_sqrt_misplaced(data, epsilon, delta, rng):
    """BUG: sqrt(2) * ln(1.25/delta) instead of sqrt(2 * ln(1.25/delta)).

    Over-noises (sigma ~16 vs ~4.84), so it remains private. Utility bug only.
    """
    true_count = sum(1 for x in data if x == 1)
    sigma = math.sqrt(2.0) * math.log(1.25 / delta) / epsilon
    return float(true_count + rng.normal(scale=sigma))


def private_count_gaussian_variance_for_sigma(data, epsilon, delta, rng):
    """BUG: passes the variance where numpy expects the standard deviation."""
    true_count = sum(1 for x in data if x == 1)
    sigma = math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon
    return float(true_count + rng.normal(scale=sigma ** 2))
