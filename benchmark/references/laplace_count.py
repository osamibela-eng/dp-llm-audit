"""Reference implementation: task laplace_bounded_count_v1 (correct, eps-DP)."""
import numpy as np


def private_count(data, epsilon, rng):
    """Count of records equal to 1 in a binary list, under add/remove-one adjacency.

    Sensitivity of the count is 1, so Laplace noise with scale 1/epsilon
    gives pure epsilon-DP.
    """
    true_count = sum(1 for x in data if x == 1)
    return float(true_count + rng.laplace(loc=0.0, scale=1.0 / epsilon))
