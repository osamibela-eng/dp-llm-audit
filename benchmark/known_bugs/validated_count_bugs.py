"""Known-buggy variants of validated_clipped_count_v1, for auditor calibration.

The interesting failure here is not noise calibration -- it is CONTROL FLOW.
A validator that raises, or that logs, creates a deterministic side channel
that no amount of Laplace noise on the return value can close.
"""

EXPECTED = {
    # raises on the first invalid record. The exception is a deterministic
    # function of the data, so [5,'oops'] and [5] are perfectly distinguishable.
    # Whether an auditor SEES this depends on whether it treats an exception as
    # an observable outcome or as a runtime failure to be discarded -- which is
    # precisely the coverage question this task exists to expose.
    "private_valid_count_raises_on_invalid": "detectable",
    # counts everything, valid or not -> still eps-DP (sensitivity is still 1),
    # just answering the wrong question. Utility bug, not a privacy bug.
    "private_valid_count_counts_everything": "likely_missed",
    # accepts booleans as integers: True is counted as a valid record. Wrong
    # answer, unchanged sensitivity -> not a privacy violation.
    "private_valid_count_accepts_bools": "likely_missed",
    # no noise at all
    "private_valid_count_no_noise": "detectable",
}


def _is_valid_strict(x):
    if isinstance(x, bool):
        return False
    return isinstance(x, int) and 0 <= x <= 100


def private_valid_count_raises_on_invalid(data, epsilon, rng):
    """BUG: validation by exception -> a deterministic data-dependent channel."""
    valid = 0
    for x in data:
        if not isinstance(x, int) or isinstance(x, bool):
            raise ValueError(f"invalid record: {x!r}")   # leaks the record too
        if 0 <= x <= 100:
            valid += 1
    return float(valid + rng.laplace(scale=1.0 / epsilon))


def private_valid_count_counts_everything(data, epsilon, rng):
    """BUG: skips validation entirely. Wrong answer, still private."""
    return float(len(data) + rng.laplace(scale=1.0 / epsilon))


def private_valid_count_accepts_bools(data, epsilon, rng):
    """BUG: isinstance(True, int) is True, so booleans slip through."""
    valid = sum(1 for x in data if isinstance(x, int) and 0 <= x <= 100)
    return float(valid + rng.laplace(scale=1.0 / epsilon))


def private_valid_count_no_noise(data, epsilon, rng):
    """BUG: returns the exact validated count."""
    return float(sum(1 for x in data if _is_valid_strict(x)))
