"""Known-buggy variants of exponential_mechanism_v1, for auditor calibration."""
import math

CATEGORIES = ("A", "B", "C", "D")

EXPECTED = {
    # drops the factor of 2 -> exp(eps*u) instead of exp(eps*u/2). The classic
    # exponential-mechanism error; true guarantee is 2*eps-DP.
    "private_select_missing_factor_two": "detectable",
    # returns the argmax of the weights -> deterministic selection, no privacy
    "private_select_argmax": "detectable",
    # normalises utilities to [0,1] before exponentiating, which rescales the
    # effective epsilon by the count range and destroys the calibration
    "private_select_normalized_utility": "detectable",
    # uses eps*u/4 -> HALF the intended epsilon. Over-noised, still private,
    # must not be flagged.
    "private_select_over_damped": "likely_missed",
}


def _counts(data):
    c = {k: 0 for k in CATEGORIES}
    for x in data:
        if x in c:
            c[x] += 1
    return c


def _sample(probs, rng):
    u = rng.random()
    acc = 0.0
    for c, p in zip(CATEGORIES, probs):
        acc += p
        if u <= acc:
            return str(c)
    return str(CATEGORIES[-1])


def private_select_missing_factor_two(data, epsilon, rng):
    """BUG: exp(eps * u) with no /2 -> 2*epsilon-DP."""
    counts = _counts(data)
    ex = [epsilon * counts[c] for c in CATEGORIES]
    m = max(ex)
    w = [math.exp(e - m) for e in ex]
    t = sum(w)
    return _sample([x / t for x in w], rng)


def private_select_argmax(data, epsilon, rng):
    """BUG: deterministic argmax -- ignores the mechanism entirely."""
    counts = _counts(data)
    best, best_c = -1, CATEGORIES[0]
    for c in CATEGORIES:
        if counts[c] > best:
            best, best_c = counts[c], c
    return str(best_c)


def private_select_normalized_utility(data, epsilon, rng):
    """BUG: min-max normalises utilities before exponentiating.

    The exponent range becomes [0, eps/2] regardless of the true count gap, so
    the mechanism is miscalibrated in a data-dependent way.
    """
    counts = _counts(data)
    vals = [counts[c] for c in CATEGORIES]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1
    ex = [epsilon * ((v - lo) / span) / 2.0 for v in vals]
    m = max(ex)
    w = [math.exp(e - m) for e in ex]
    t = sum(w)
    return _sample([x / t for x in w], rng)


def private_select_over_damped(data, epsilon, rng):
    """BUG: exp(eps*u/4) -> effectively eps/2-DP. Over-private, not a leak."""
    counts = _counts(data)
    ex = [epsilon * counts[c] / 4.0 for c in CATEGORIES]
    m = max(ex)
    w = [math.exp(e - m) for e in ex]
    t = sum(w)
    return _sample([x / t for x in w], rng)
