"""E8 A1 reference: clipped_bounded_sum_v1 with identifiers renamed.

Clip to [0, 10], sensitivity 10 under add/remove-one, scale 10/epsilon.
"""
import numpy as np


def aggregate_bounded_values(readings, epsilon, rng):
    capped = [min(max(float(x), 0.0), 10.0) for x in readings]
    total = float(sum(capped)) if capped else 0.0
    return float(total + rng.laplace(loc=0.0, scale=10.0 / epsilon))
