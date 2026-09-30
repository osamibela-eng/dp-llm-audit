"""Reference implementation: task validated_clipped_count_v1 (correct, eps-DP)."""


def _is_valid(x):
    """True iff x is an integer in [0, 100].

    Booleans are excluded explicitly: in Python `isinstance(True, int)` is True,
    so a naive check silently counts True as the integer 1. That is a real
    validation bug, not a pedantic one.

    Floats are rejected even when integral -- the spec says integers -- and the
    check never raises, whatever the record contains.
    """
    if isinstance(x, bool):
        return False
    if not isinstance(x, int):
        return False
    return 0 <= x <= 100


def private_valid_count(data, epsilon, rng):
    """DP count of valid records, with validation that cannot leak.

    Sensitivity is 1: adding or removing one record changes the count of valid
    records by at most 1, whether or not that record was valid.

    The privacy-relevant requirement is that validation is TOTAL and SILENT.
    Raising on a malformed record turns the exception into a deterministic,
    un-noised channel: an observer who sees a crash learns that the dataset
    contained a record of that shape. The same applies to printing or logging
    anything derived from record contents. Validation here is a pure predicate
    with no side effects and no failure mode.
    """
    valid = sum(1 for x in data if _is_valid(x))
    return float(valid + rng.laplace(loc=0.0, scale=1.0 / epsilon))
