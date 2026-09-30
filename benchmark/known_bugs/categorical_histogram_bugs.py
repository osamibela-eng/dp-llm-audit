"""Known-buggy variants of categorical_histogram_v1, for auditor calibration."""
import numpy as np

CATEGORIES = ("A", "B", "C")

EXPECTED = {
    "private_histogram_support_leak": "detectable",   # zero bins released exactly
    "private_histogram_wrong_scale": "detectable",    # scale 1/(3*eps) -> 3*eps-DP
    "private_histogram_no_noise": "detectable",       # deterministic
}


def private_histogram_support_leak(data, epsilon, rng):
    """BUG: adds noise only to non-zero bins; empty bins are released as exact 0.0,
    so the *support* of the data leaks deterministically."""
    counts = {c: 0 for c in CATEGORIES}
    for x in data:
        if x in counts:
            counts[x] += 1
    return {c: (float(counts[c] + rng.laplace(scale=1.0 / epsilon)) if counts[c] > 0
                else 0.0)
            for c in CATEGORIES}


def private_histogram_wrong_scale(data, epsilon, rng):
    """BUG: splits the budget as if bins were sequentially composed... backwards:
    uses scale 1/(3*eps), i.e. LESS noise than required -> ~3*eps-DP."""
    counts = {c: 0 for c in CATEGORIES}
    for x in data:
        if x in counts:
            counts[x] += 1
    return {c: float(counts[c] + rng.laplace(scale=1.0 / (3.0 * epsilon)))
            for c in CATEGORIES}


def private_histogram_no_noise(data, epsilon, rng):
    """BUG: returns exact counts."""
    counts = {c: 0 for c in CATEGORIES}
    for x in data:
        if x in counts:
            counts[x] += 1
    return {c: float(counts[c]) for c in CATEGORIES}
