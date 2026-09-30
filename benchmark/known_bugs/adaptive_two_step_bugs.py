"""Known-buggy variants of adaptive_two_step_v1, for auditor calibration."""
import numpy as np

EXPECTED = {
    # branches on the RAW count. The noisy releases look identical in
    # distribution to the reference on most inputs; what leaks is WHICH branch
    # was taken, and that is only visible on data engineered to sit right at
    # the branch boundary. Hard for a generic auditor, and a good calibration
    # probe for the pair-design lesson in AUDITOR_NOTES section 5.
    "private_adaptive_release_branches_on_raw": "likely_missed",
    # full epsilon on each step -> 2*eps-DP
    "private_adaptive_release_full_budget_each": "detectable",
    # no noise on step 1, only on step 2
    "private_adaptive_release_step1_clear": "detectable",
    # over-splits into eps/4 per step -> over-noised, still private
    "private_adaptive_release_over_split": "likely_missed",
}


def private_adaptive_release_branches_on_raw(data, epsilon, rng):
    """BUG: the branch is decided by the TRUE count, not the released one.

    The choice of second query becomes an un-paid-for, data-dependent release.
    Two neighbouring datasets straddling the boundary produce second components
    drawn from different query distributions.
    """
    eps_step = epsilon / 2.0
    scale = 1.0 / eps_step
    high = sum(1 for x in data if x >= 0.5)
    step1 = float(high + rng.laplace(scale=scale))
    if high >= 3:                                     # WRONG: raw count
        total = float(np.sum(np.asarray(data, dtype=float))) if data else 0.0
        step2 = float(total + rng.laplace(scale=scale))
    else:
        low = sum(1 for x in data if x < 0.5)
        step2 = float(low + rng.laplace(scale=scale))
    return (step1, step2)


def private_adaptive_release_full_budget_each(data, epsilon, rng):
    """BUG: each step spends the full epsilon -> 2*eps-DP."""
    scale = 1.0 / epsilon
    high = sum(1 for x in data if x >= 0.5)
    step1 = float(high + rng.laplace(scale=scale))
    if step1 >= 3.0:
        total = float(np.sum(np.asarray(data, dtype=float))) if data else 0.0
        step2 = float(total + rng.laplace(scale=scale))
    else:
        low = sum(1 for x in data if x < 0.5)
        step2 = float(low + rng.laplace(scale=scale))
    return (step1, step2)


def private_adaptive_release_step1_clear(data, epsilon, rng):
    """BUG: step 1 released exactly; only step 2 is noised."""
    high = sum(1 for x in data if x >= 0.5)
    step1 = float(high)                               # WRONG: no noise
    if step1 >= 3.0:
        total = float(np.sum(np.asarray(data, dtype=float))) if data else 0.0
        step2 = float(total + rng.laplace(scale=1.0 / epsilon))
    else:
        low = sum(1 for x in data if x < 0.5)
        step2 = float(low + rng.laplace(scale=1.0 / epsilon))
    return (step1, step2)


def private_adaptive_release_over_split(data, epsilon, rng):
    """BUG: eps/4 per step for two steps -> over-noised, still private."""
    eps_step = epsilon / 4.0
    scale = 1.0 / eps_step
    high = sum(1 for x in data if x >= 0.5)
    step1 = float(high + rng.laplace(scale=scale))
    if step1 >= 3.0:
        total = float(np.sum(np.asarray(data, dtype=float))) if data else 0.0
        step2 = float(total + rng.laplace(scale=scale))
    else:
        low = sum(1 for x in data if x < 0.5)
        step2 = float(low + rng.laplace(scale=scale))
    return (step1, step2)
