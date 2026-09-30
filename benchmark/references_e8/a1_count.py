"""E8 A1 reference: laplace_bounded_count_v1 with identifiers renamed.

Semantically identical to benchmark/references/laplace_count.py. Sensitivity 1
under add/remove-one, Laplace scale 1/epsilon.
"""
import numpy as np


def count_positive_flags(records, epsilon, rng):
    flagged = sum(1 for r in records if r == 1)
    return float(flagged + rng.laplace(loc=0.0, scale=1.0 / epsilon))
