"""Known-buggy variants of repeated_release_v1, for auditor calibration."""
import numpy as np

K = 4

EXPECTED = {
    # full epsilon on each of four releases -> 4*eps-DP. The headline
    # composition bug and the one most likely to appear in model output.
    "private_repeated_count_full_budget_each": "detectable",
    # noise drawn once and copied: the four outputs are identical, so the
    # release is really a single eps/4 answer repeated. It is MORE private
    # than claimed, not less -- the auditor should stay silent, and only the
    # functional test ("four independent draws") catches it.
    "private_repeated_count_shared_noise": "likely_missed",
    # divides by K in the wrong direction: scale K/eps is correct, this uses
    # eps/K as the SCALE -> at eps=1, K=4 that is scale 0.25 vs the correct
    # 4.0, i.e. 16x too little noise.
    "private_repeated_count_inverted_split": "detectable",
    # splits by K-1 instead of K -> 4/3 * eps overall. Subtle power probe.
    "private_repeated_count_off_by_one_k": "likely_missed",
}


def private_repeated_count_full_budget_each(data, epsilon, rng):
    """BUG: each of the four releases spends the full epsilon."""
    scale = 1.0 / epsilon
    true_count = sum(1 for x in data if x == 1)
    return [float(true_count + rng.laplace(scale=scale)) for _ in range(K)]


def private_repeated_count_shared_noise(data, epsilon, rng):
    """BUG: one draw, four copies -- not four independent releases."""
    scale = float(K) / epsilon
    true_count = sum(1 for x in data if x == 1)
    noise = rng.laplace(scale=scale)
    return [float(true_count + noise) for _ in range(K)]


def private_repeated_count_inverted_split(data, epsilon, rng):
    """BUG: uses eps/K as the Laplace SCALE instead of K/eps."""
    scale = epsilon / K
    true_count = sum(1 for x in data if x == 1)
    return [float(true_count + rng.laplace(scale=scale)) for _ in range(K)]


def private_repeated_count_off_by_one_k(data, epsilon, rng):
    """BUG: divides the budget by K-1 -> total spend 4/3 * epsilon."""
    scale = float(K - 1) / epsilon
    true_count = sum(1 for x in data if x == 1)
    return [float(true_count + rng.laplace(scale=scale)) for _ in range(K)]
