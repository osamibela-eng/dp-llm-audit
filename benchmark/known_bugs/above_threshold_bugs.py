"""Known-buggy variants of above_threshold_v1, for auditor calibration.

Every variant here corresponds to a mistake that appeared in the *published*
literature and was catalogued by Lyu, Su & Li (VLDB 2017). They are included
because they are the strongest available test of whether an empirical auditor
can see real-world DP errors, as opposed to toy ones.
"""
import math

THRESHOLD = 4.0


def _queries(data):
    ones = sum(1 for x in data if x == 1)
    zeros = sum(1 for x in data if x == 0)
    return [float(ones), float(zeros), float(len(data))]


EXPECTED = {
    # CORRECTED after calibration (2026-08-20). All four of these were first
    # predicted "detectable" on the reasoning that each is a genuine DP
    # violation. Measurement disagrees, and the reason is instructive: the harm
    # from every one of them GROWS WITH THE NUMBER OF QUERIES, and this task
    # instance answers only three. A search over neighbouring pairs found these
    # maximum demonstrable epsilon lower bounds against a claim of 1.0:
    #
    #     no_query_noise         1.04   (barely clears - kept "detectable")
    #     fresh_threshold_noise  0.70
    #     missing_factor_two     0.58
    #     returns_value          0.56
    #
    # So three of the four are genuinely wrong yet UNFALSIFIABLE at this task
    # size. Relabelled rather than explained away: predictions are corrected by
    # measurement, which is the entire point of keeping a calibration table.
    # See AUDITOR_NOTES section 9.
    "above_threshold_no_query_noise": "detectable",
    "above_threshold_fresh_threshold_noise": "likely_missed",
    "above_threshold_missing_factor_two": "likely_missed",
    "above_threshold_returns_value": "likely_missed",
    # splits the budget across all three queries as if each cost eps/3. This is
    # unnecessary but conservative -> OVER-noised and still private, so the
    # auditor must stay silent. False-positive probe.
    "above_threshold_over_splits_budget": "likely_missed",
}


def above_threshold_fresh_threshold_noise(data, epsilon, rng):
    """BUG: redraws the threshold noise inside the loop."""
    eps1 = eps2 = epsilon / 2.0
    for i, q in enumerate(_queries(data), start=1):
        rho = rng.laplace(scale=1.0 / eps1)               # WRONG: per query
        nu = rng.laplace(scale=2.0 / eps2)
        if q + nu >= THRESHOLD + rho:
            return int(i)
    return 0


def above_threshold_no_query_noise(data, epsilon, rng):
    """BUG: compares the TRUE query value against a noisy threshold."""
    rho = rng.laplace(scale=2.0 / epsilon)
    for i, q in enumerate(_queries(data), start=1):
        if q >= THRESHOLD + rho:                          # WRONG: q not noisy
            return int(i)
    return 0


def above_threshold_missing_factor_two(data, epsilon, rng):
    """BUG: query noise Delta/eps2 rather than 2*Delta/eps2."""
    eps1 = eps2 = epsilon / 2.0
    rho = rng.laplace(scale=1.0 / eps1)
    for i, q in enumerate(_queries(data), start=1):
        nu = rng.laplace(scale=1.0 / eps2)                # WRONG: missing 2
        if q + nu >= THRESHOLD + rho:
            return int(i)
    return 0


def above_threshold_returns_value(data, epsilon, rng):
    """BUG: releases the noisy value that crossed, not just its index.

    Returns an int to keep the declared output type, but the value carries far
    more information about the data than the index does.
    """
    eps1 = eps2 = epsilon / 2.0
    rho = rng.laplace(scale=1.0 / eps1)
    for q in _queries(data):
        nu = rng.laplace(scale=2.0 / eps2)
        if q + nu >= THRESHOLD + rho:
            return int(round(q + nu))                     # WRONG: the value
    return 0


def above_threshold_over_splits_budget(data, epsilon, rng):
    """BUG: divides the budget by the number of queries as well.

    Unnecessary -- AboveThreshold does not degrade with query count -- so this
    is merely over-noised. Still epsilon-DP; must not be falsified.
    """
    k = 3
    eps1 = epsilon / 2.0
    eps2 = (epsilon / 2.0) / k
    rho = rng.laplace(scale=1.0 / eps1)
    for i, q in enumerate(_queries(data), start=1):
        nu = rng.laplace(scale=2.0 / eps2)
        if q + nu >= THRESHOLD + rho:
            return int(i)
    return 0
