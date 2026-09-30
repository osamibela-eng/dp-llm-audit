"""Reference implementation: task empty_dataset_count_v1 (correct, eps-DP)."""


def private_count_safe(data, epsilon, rng):
    """DP count of 1s, with the empty dataset on the ORDINARY code path.

    There is deliberately no `if not data:` branch here, and its absence is the
    whole content of the task.

    Under add/remove-one adjacency [] and [1] are neighbours, so the empty
    dataset is not a degenerate case to be handled specially -- it is one side
    of a pair the guarantee must cover. Any special case that returns a
    deterministic value (0, or 0.0, or an integer instead of a float) makes the
    two neighbours perfectly distinguishable: seeing exactly 0.0 tells you the
    input was empty, at any epsilon.

    Rounding only in the empty case is the same bug wearing a different hat:
    the output DOMAIN differs between neighbours, which is a support difference
    rather than a distribution difference, and no epsilon covers it.
    """
    true_count = sum(1 for x in data if x == 1)
    return float(true_count + rng.laplace(loc=0.0, scale=1.0 / epsilon))
