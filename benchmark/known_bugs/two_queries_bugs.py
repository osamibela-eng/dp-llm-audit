"""Known-buggy variants of two_queries_split_budget_v1, for auditor calibration."""
import numpy as np

EXPECTED = {
    # full epsilon on each of two queries -> the joint release is 2*eps-DP.
    # A factor-2 overspend; detectable, but it is the SUBTLEST of the
    # under-noising bugs in the suite and a good power probe.
    "private_count_and_sum_full_budget_each": "detectable",
    # splits the SENSITIVITY instead of the budget: scale (1/2)/eps. That is
    # half the required noise per query on top of no composition accounting,
    # so the joint release is 4*eps-DP.
    "private_count_and_sum_split_sensitivity": "detectable",
    # draws ONE Laplace value and adds it to both outputs. The pair is then
    # perfectly correlated, which LEAKS: the difference (sum - count) becomes
    # deterministic given the data.
    "private_count_and_sum_shared_noise": "detectable",
    # over-splits into eps/4 each -> 0.5*eps-DP overall. Wrong, but MORE
    # private than claimed; the auditor must not flag it.
    "private_count_and_sum_over_split": "likely_missed",
}


def private_count_and_sum_full_budget_each(data, epsilon, rng):
    """BUG: no composition accounting -- each query gets the full epsilon."""
    scale = 1.0 / epsilon
    noisy_count = float(len(data) + rng.laplace(scale=scale))
    total = float(np.sum(np.asarray(data, dtype=float))) if data else 0.0
    noisy_sum = float(total + rng.laplace(scale=scale))
    return (noisy_count, noisy_sum)


def private_count_and_sum_split_sensitivity(data, epsilon, rng):
    """BUG: halves the sensitivity rather than the budget."""
    scale = 0.5 / epsilon
    noisy_count = float(len(data) + rng.laplace(scale=scale))
    total = float(np.sum(np.asarray(data, dtype=float))) if data else 0.0
    noisy_sum = float(total + rng.laplace(scale=scale))
    return (noisy_count, noisy_sum)


def private_count_and_sum_shared_noise(data, epsilon, rng):
    """BUG: one noise draw reused for both outputs -> the difference is exact."""
    scale = 2.0 / epsilon
    noise = rng.laplace(scale=scale)
    total = float(np.sum(np.asarray(data, dtype=float))) if data else 0.0
    return (float(len(data) + noise), float(total + noise))


def private_count_and_sum_over_split(data, epsilon, rng):
    """BUG: budget split four ways for two queries -> over-noised but private."""
    scale = 4.0 / epsilon
    noisy_count = float(len(data) + rng.laplace(scale=scale))
    total = float(np.sum(np.asarray(data, dtype=float))) if data else 0.0
    noisy_sum = float(total + rng.laplace(scale=scale))
    return (noisy_count, noisy_sum)
