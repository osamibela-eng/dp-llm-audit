"""Reference implementation: task clipped_bounded_sum_v1 (correct, eps-DP).

Input values are arbitrary non-negative floats and MAY exceed the bound B;
the spec requires clipping each value to [0, B] before summing, giving
add/remove sensitivity B.
"""
import numpy as np

B = 10.0


def private_sum(data, epsilon, rng):
    clipped = [min(max(float(x), 0.0), B) for x in data]
    return float(sum(clipped) + rng.laplace(loc=0.0, scale=B / epsilon))
