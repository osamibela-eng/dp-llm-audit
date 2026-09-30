"""Reference implementation: task above_threshold_v1 (correct, eps-DP).

This is the AboveThreshold / Sparse Vector Technique template. It is included
because it is the DP primitive with the worst track record in the literature:
Lyu, Su and Li (VLDB 2017) catalogue several *published* variants that are not
actually differentially private. The failure modes are subtle and all of them
look reasonable, which makes it the sharpest probe in this benchmark.
"""
import math

THRESHOLD = 4.0        # public constant, not data-dependent


def _queries(data):
    """The three fixed counting queries, each of sensitivity 1."""
    ones = sum(1 for x in data if x == 1)
    zeros = sum(1 for x in data if x == 0)
    return [float(ones), float(zeros), float(len(data))]


def above_threshold(data, epsilon, rng):
    """Return the index (1-based) of the first query to privately exceed T, else 0.

    Correct construction (Dwork & Roth Alg. 1; Lyu-Su-Li 2017 'Alg. 1'):

        rho  ~ Lap(Delta / eps1)          drawn ONCE, added to the threshold
        nu_i ~ Lap(2 * Delta / eps2)      drawn FRESH for each query
        halt at the first i with  q_i(D) + nu_i >= T + rho

    with eps = eps1 + eps2. Splitting evenly, eps1 = eps2 = eps/2, so

        threshold scale = 1 / (eps/2) = 2/eps
        query scale     = 2 / (eps/2) = 4/eps

    Three things carry the proof and are each a separate way to get it wrong:

    * The threshold noise is drawn ONCE. Re-drawing it per query destroys the
      argument entirely -- this is the single most common published error.
    * The query noise is TWICE the threshold noise (2*Delta, not Delta). The
      factor of 2 is what pays for the unbounded number of below-threshold
      answers.
    * Only the INDEX is released, never the noisy value that crossed. Returning
      the value is a strictly larger release than the analysis covers, and is
      the error that makes the 'Sparse Vector' variants in several papers
      non-private.

    Because the mechanism halts at the first crossing, the guarantee does not
    degrade with the number of queries -- that is the whole point of the
    technique, and why over-splitting the budget across queries (a plausible
    looking 'fix') is unnecessary rather than merely wasteful.
    """
    eps1 = epsilon / 2.0
    eps2 = epsilon / 2.0
    sensitivity = 1.0

    rho = rng.laplace(loc=0.0, scale=sensitivity / eps1)          # ONCE
    noisy_threshold = THRESHOLD + rho

    for i, q in enumerate(_queries(data), start=1):
        nu = rng.laplace(loc=0.0, scale=2.0 * sensitivity / eps2)  # fresh
        if q + nu >= noisy_threshold:
            return int(i)                                          # index only
    return 0
