"""E8 A2 reference: bounded mean with the value range widened to [0, 4].

Under replace-one, changing one record moves the mean by at most 4/n rather
than 1/n, so the Laplace scale is 4/(n*epsilon).

A memorised bounded-mean implementation assumes values in [0, 1] and uses
1/(n*epsilon), under-noising by a factor of 4.
"""
import numpy as np


def private_average(data, epsilon, rng):
    n = len(data)
    if n == 0:
        raise ValueError("data must be non-empty")
    mean = float(sum(float(x) for x in data)) / n
    sensitivity = 4.0 / n
    return float(mean + rng.laplace(loc=0.0, scale=sensitivity / epsilon))
