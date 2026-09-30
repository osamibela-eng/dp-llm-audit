"""Known-buggy variants of report_noisy_max_v1, for auditor calibration.

NOTE the third variant: half-scale RNM is truly non-private at the claimed eps
(it is ~2*eps-DP), but its worst observable likelihood ratio on small count
gaps stays below e^eps for many pair choices -- we EXPECT the threshold
auditor to miss it. That is a feature: it demonstrates, inside the calibration
table, why "not falsified" must never be reported as "safe".
"""
import numpy as np

CATEGORIES = ("A", "B", "C", "D")

EXPECTED = {
    "private_argmax_noise_after_selection": "detectable",   # deterministic argmax
    "private_argmax_shared_noise": "detectable",            # same draw added to all
    "private_argmax_half_scale": "likely_missed",           # 2*eps-DP, subtle
}


def private_argmax_noise_after_selection(data, epsilon, rng):
    """BUG: selects the argmax on TRUE counts, then adds noise only to the
    (discarded) count value. The released label is deterministic."""
    counts = [sum(1 for x in data if x == c) for c in CATEGORIES]
    winner = int(np.argmax(counts))
    _noisy_count = counts[winner] + rng.laplace(scale=2.0 / epsilon)  # unused
    return CATEGORIES[winner]


def private_argmax_shared_noise(data, epsilon, rng):
    """BUG: draws ONE Laplace noise value and adds it to every count; the
    ordering is unchanged, so this is exactly the deterministic argmax."""
    counts = [sum(1 for x in data if x == c) for c in CATEGORIES]
    z = rng.laplace(scale=2.0 / epsilon)
    noisy = [c + z for c in counts]
    return CATEGORIES[int(np.argmax(noisy))]


def private_argmax_half_scale(data, epsilon, rng):
    """BUG: uses Laplace(1/eps) instead of Laplace(2/eps) -> only 2*eps-DP."""
    counts = [sum(1 for x in data if x == c) for c in CATEGORIES]
    noisy = [c + rng.laplace(scale=1.0 / epsilon) for c in counts]
    return CATEGORIES[int(np.argmax(noisy))]
