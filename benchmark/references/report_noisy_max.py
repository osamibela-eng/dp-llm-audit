"""Reference implementation: task report_noisy_max_v1 (correct, eps-DP).

Input is a list of category labels from {A, B, C, D}. The mechanism computes
the four counts, adds independent Laplace(2/epsilon) noise to each count
(counts have sensitivity 1 under add/remove-one; the classic Report Noisy Max
analysis gives pure epsilon-DP with scale 2*Delta/epsilon), and returns the
label of the noisy argmax. Only the winning LABEL is released - never the
noisy counts themselves.
"""
import numpy as np

CATEGORIES = ("A", "B", "C", "D")


def private_argmax(data, epsilon, rng):
    counts = [sum(1 for x in data if x == c) for c in CATEGORIES]
    noisy = [c + rng.laplace(loc=0.0, scale=2.0 / epsilon) for c in counts]
    return CATEGORIES[int(np.argmax(noisy))]
