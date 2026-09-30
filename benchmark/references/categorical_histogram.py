"""Reference implementation: task categorical_histogram_v1 (correct, eps-DP).

Fixed public category set {A, B, C}; add/remove-one adjacency changes exactly
one bin by 1, so the histogram has L1 sensitivity 1 and per-bin Laplace noise
with scale 1/epsilon gives pure epsilon-DP. Noise must be added to EVERY bin,
including empty ones (a common bug is leaving zero bins deterministic).
Invalid labels are ignored (counted toward no bin) rather than raising,
so control flow does not depend on individual records.
"""
import numpy as np

CATEGORIES = ("A", "B", "C")


def private_histogram(data, epsilon, rng):
    counts = {c: 0 for c in CATEGORIES}
    for x in data:
        if x in counts:
            counts[x] += 1
    return {c: float(counts[c] + rng.laplace(loc=0.0, scale=1.0 / epsilon))
            for c in CATEGORIES}
