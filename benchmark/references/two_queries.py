"""Reference implementation: task two_queries_split_budget_v1 (correct, eps-DP)."""
import numpy as np


def private_count_and_sum(data, epsilon, rng):
    """Release a noisy count AND a noisy sum under a single epsilon budget.

    Two queries are answered on the same data, so by SEQUENTIAL COMPOSITION the
    privacy costs add. Splitting the budget evenly gives each query eps/2, and
    each has sensitivity 1 under add/remove-one (adding a record changes the
    count by 1, and changes the sum by at most 1 because values lie in [0, 1]).

    Laplace scale is therefore 1/(eps/2) = 2/eps for BOTH releases.

    The two classic errors are (a) spending the full epsilon on each query,
    which yields 2*eps-DP overall, and (b) splitting the SENSITIVITY rather
    than the budget, which changes nothing about composition.
    """
    eps_each = epsilon / 2.0
    scale = 1.0 / eps_each

    noisy_count = float(len(data) + rng.laplace(loc=0.0, scale=scale))
    noisy_sum = float(float(np.sum(np.asarray(data, dtype=float))) if data else 0.0)
    noisy_sum = float(noisy_sum + rng.laplace(loc=0.0, scale=scale))
    return (noisy_count, noisy_sum)
