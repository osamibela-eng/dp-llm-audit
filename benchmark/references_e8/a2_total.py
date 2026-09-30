"""E8 A2 reference: sum over records drawn from {0..5}.

The perturbation widens the record domain from {0,1} to {0,...,5} and asks for
the SUM. Under add/remove-one a single record can carry 5, so the sensitivity
is 5, not 1, and the Laplace scale is 5/epsilon.

A memorised Laplace count uses scale 1/epsilon and therefore under-noises by a
factor of 5. That is the signature this task is built to expose.
"""
import numpy as np


def private_total(data, epsilon, rng):
    total = float(sum(data)) if data else 0.0
    sensitivity = 5.0
    return float(total + rng.laplace(loc=0.0, scale=sensitivity / epsilon))
