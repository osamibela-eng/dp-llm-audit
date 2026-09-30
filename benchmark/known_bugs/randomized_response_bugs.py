"""Known-buggy variants of randomized_response_v1, for auditor calibration."""
import math

EXPECTED = {
    # truthful with p = 0.9 -> true epsilon = ln(9) = 2.197, about 2x the claim
    "randomized_response_too_truthful": "detectable",
    # never flips -> deterministic passthrough, not DP at any finite epsilon
    "randomized_response_deterministic": "detectable",
    # uses e^eps/(1+e^eps) as the FLIP probability instead of the truth
    # probability. The output distribution is the mirror image, so the privacy
    # loss is identical (still eps-DP) but the utility is inverted. A DP
    # auditor should NOT flag it -- a functional test should.
    "randomized_response_flipped_formula": "likely_missed",
    # p = eps/(1+eps) instead of e^eps/(1+e^eps): at eps = ln(3) = 1.0986 this
    # gives p = 0.523 (over-noised, still private). Probe for over-noising.
    "randomized_response_missing_exp": "likely_missed",
}


def randomized_response_too_truthful(data, epsilon, rng):
    """BUG: hardcoded 0.9 truth probability -> true epsilon = ln(9)."""
    bit = int(data[0])
    if rng.random() < 0.9:
        return int(bit)
    return int(1 - bit)


def randomized_response_deterministic(data, epsilon, rng):
    """BUG: returns the raw bit -> no privacy at all."""
    return int(data[0])


def randomized_response_flipped_formula(data, epsilon, rng):
    """BUG: treats e^eps/(1+e^eps) as the FLIP probability.

    Truthful with probability 1/(1+e^eps) instead. Still eps-DP by symmetry,
    so the auditor is expected to stay silent; the functional test catches it.
    """
    bit = int(data[0])
    p_flip = math.exp(epsilon) / (1.0 + math.exp(epsilon))
    if rng.random() < p_flip:
        return int(1 - bit)
    return int(bit)


def randomized_response_missing_exp(data, epsilon, rng):
    """BUG: p = eps/(1+eps), forgetting to exponentiate -> over-noised."""
    bit = int(data[0])
    p_truth = epsilon / (1.0 + epsilon)
    if rng.random() < p_truth:
        return int(bit)
    return int(1 - bit)
