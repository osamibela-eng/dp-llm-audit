"""E8 A2 reference: clipped sum with the clip range WIDENED to [0, 30].

Sensitivity is the clip bound, so it rises from 10 to 30 and the Laplace scale
is 30/epsilon.

The direction is deliberate. Narrowing the range to [0, 3] would leave a
memorised clip-to-10 implementation over-noised, hence still private, and the
audit would have nothing to find. Widening it leaves that same memorised
implementation under-noised by a factor of 3, which is a genuine violation of
the claim and one the auditor can detect.
"""
import numpy as np


def private_sum_capped(data, epsilon, rng):
    clipped = [min(max(float(x), 0.0), 30.0) for x in data]
    total = float(sum(clipped)) if clipped else 0.0
    return float(total + rng.laplace(loc=0.0, scale=30.0 / epsilon))
