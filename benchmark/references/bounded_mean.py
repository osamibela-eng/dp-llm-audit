"""Reference implementation: task bounded_mean_v1 (correct, eps-DP)."""
import numpy as np


def private_mean(data, epsilon, rng):
    """DP mean of values in [0, 1] under replace-one adjacency.

    Under replace-one the length n is fixed, so replacing a single record can
    move the mean by at most 1/n (the range of the domain divided by n).
    Laplace noise with scale 1/(n * epsilon) therefore gives pure epsilon-DP.

    The two common ways to get this wrong are to use the SUM sensitivity (1)
    instead of the mean sensitivity (1/n), and to divide by the wrong n.
    """
    n = len(data)
    if n == 0:
        # Spec guarantees non-empty; return 0.0 rather than raising so that the
        # auditor sees a total function.
        return 0.0
    true_mean = float(np.sum(np.asarray(data, dtype=float))) / n
    return float(true_mean + rng.laplace(loc=0.0, scale=1.0 / (n * epsilon)))
