"""Known-buggy variants of empty_dataset_count_v1, for auditor calibration.

These are SUPPORT bugs rather than calibration bugs: the output domain differs
between neighbouring datasets, so the violation is not a matter of too little
noise. It is the same shape of failure as the nullability channel in the DP
synthetic-data literature -- an event with probability exactly zero on one side
and positive probability on the other.
"""

EXPECTED = {
    # returns exactly 0.0 for the empty dataset. [] and [1] are neighbours, so
    # observing 0.0 is a perfect distinguisher. Should be caught easily by an
    # exact-value event, and is the cleanest support-violation probe here.
    "private_count_safe_zero_on_empty": "detectable",
    # raises on empty -> deterministic observable difference between neighbours
    "private_count_safe_raises_on_empty": "detectable",
    # rounds the noise to an int only in the empty case: the output lands on
    # the integer lattice exactly when the dataset is empty
    "private_count_safe_rounds_on_empty": "detectable",
    # noise added, but the result is clamped at 0 for every input. Clamping is
    # post-processing applied uniformly, so it stays eps-DP -- a false-positive
    # probe that looks superficially like the zero-on-empty bug.
    "private_count_safe_clamps_at_zero": "likely_missed",
}


def private_count_safe_zero_on_empty(data, epsilon, rng):
    """BUG: special-cases the empty dataset to a deterministic 0.0."""
    if not data:
        return 0.0                                   # perfect distinguisher
    true_count = sum(1 for x in data if x == 1)
    return float(true_count + rng.laplace(scale=1.0 / epsilon))


def private_count_safe_raises_on_empty(data, epsilon, rng):
    """BUG: raises on the empty dataset."""
    if not data:
        raise ValueError("empty dataset")
    true_count = sum(1 for x in data if x == 1)
    return float(true_count + rng.laplace(scale=1.0 / epsilon))


def private_count_safe_rounds_on_empty(data, epsilon, rng):
    """BUG: rounds to an integer only when the dataset is empty."""
    true_count = sum(1 for x in data if x == 1)
    noisy = true_count + rng.laplace(scale=1.0 / epsilon)
    if not data:
        return float(round(noisy))                   # integer lattice iff empty
    return float(noisy)


def private_count_safe_clamps_at_zero(data, epsilon, rng):
    """BUG: clamps negatives to zero for EVERY input.

    A uniform post-processing step, so still epsilon-DP. It piles probability
    mass onto exactly 0.0, which superficially resembles the zero-on-empty bug
    but is applied identically to both neighbours and therefore does not leak.
    """
    true_count = sum(1 for x in data if x == 1)
    noisy = true_count + rng.laplace(scale=1.0 / epsilon)
    return float(max(0.0, noisy))
