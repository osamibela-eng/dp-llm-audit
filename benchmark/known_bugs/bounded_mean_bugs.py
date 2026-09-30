"""Known-buggy variants of bounded_mean_v1, for auditor calibration.

Same signature and the same eps=1.0 pure-DP claim as the reference; each is
wrong in a way drawn from the task's `common_bugs` list.
"""
import numpy as np

EXPECTED = {
    # sensitivity 1 instead of 1/n -> noise n times too LARGE. This is
    # over-noising: it is BAD FOR UTILITY but still private, so a DP auditor
    # must NOT flag it. Included deliberately as a false-positive probe.
    "private_mean_sum_sensitivity": "likely_missed",
    # divides sensitivity by n-1 -> slightly too little noise, ~n/(n-1) * eps.
    # Very subtle at large n; a genuine detection-power probe.
    "private_mean_off_by_one_n": "likely_missed",
    # no noise at all -> deterministic, not DP for any finite epsilon
    "private_mean_no_noise": "detectable",
    # noise scale uses epsilon instead of 1/epsilon: at eps=1 these coincide
    # (the epsilon trap, AUDITOR_NOTES section 6), so this is only visible at
    # the second audit epsilon.
    "private_mean_inverted_epsilon": "detectable_at_second_eps",
}


def private_mean_sum_sensitivity(data, epsilon, rng):
    """BUG: treats the mean as having sensitivity 1 (the SUM's sensitivity).

    Adds n times more noise than needed. Still epsilon-DP -- strictly more
    private -- so a correct auditor should stay silent. Utility is destroyed.
    """
    n = len(data)
    if n == 0:
        return 0.0
    true_mean = float(np.sum(np.asarray(data, dtype=float))) / n
    return float(true_mean + rng.laplace(scale=1.0 / epsilon))


def private_mean_off_by_one_n(data, epsilon, rng):
    """BUG: scale 1/((n-1)*epsilon) -> true privacy is n/(n-1) * epsilon."""
    n = len(data)
    if n == 0:
        return 0.0
    denom = max(n - 1, 1)
    true_mean = float(np.sum(np.asarray(data, dtype=float))) / n
    return float(true_mean + rng.laplace(scale=1.0 / (denom * epsilon)))


def private_mean_no_noise(data, epsilon, rng):
    """BUG: returns the exact mean -> not DP for any finite epsilon."""
    n = len(data)
    if n == 0:
        return 0.0
    return float(np.sum(np.asarray(data, dtype=float))) / n


def private_mean_inverted_epsilon(data, epsilon, rng):
    """BUG: scale epsilon/n instead of 1/(n*epsilon).

    At epsilon = 1 the two coincide and the bug is INVISIBLE; at the second
    audit epsilon (0.3) the noise is ~11x too small.
    """
    n = len(data)
    if n == 0:
        return 0.0
    true_mean = float(np.sum(np.asarray(data, dtype=float))) / n
    return float(true_mean + rng.laplace(scale=epsilon / n))
