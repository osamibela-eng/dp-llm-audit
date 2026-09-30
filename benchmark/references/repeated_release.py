"""Reference implementation: task repeated_release_v1 (correct, eps-DP)."""
import numpy as np

K = 4          # number of independent releases required by the spec


def private_repeated_count(data, epsilon, rng):
    """Four independent noisy counts whose TOTAL release is epsilon-DP.

    Four releases of the same query on the same data compose sequentially, so
    each release gets eps/4 and Laplace scale 4/eps. Sensitivity of the count
    is 1 under add/remove-one.

    Two failure modes this guards against:
      - spending the full epsilon per release (a 4x overspend, K*eps-DP), and
      - drawing the noise ONCE and returning four copies, which looks noisy but
        collapses to a single release and lets an attacker average nothing away
        while the spec promises four independent answers.
    """
    eps_each = epsilon / K
    scale = 1.0 / eps_each
    true_count = sum(1 for x in data if x == 1)
    return [float(true_count + rng.laplace(loc=0.0, scale=scale)) for _ in range(K)]
