"""E8 A1 reference: bounded_mean_v1 with identifiers renamed.

Replace-one adjacency on values in [0, 1] gives the mean sensitivity 1/n, so
the Laplace scale is 1/(n*epsilon).
"""
import numpy as np


def average_scores(scores, epsilon, rng):
    n = len(scores)
    if n == 0:
        raise ValueError("scores must be non-empty")
    mean = float(sum(float(x) for x in scores)) / n
    return float(mean + rng.laplace(loc=0.0, scale=1.0 / (n * epsilon)))
