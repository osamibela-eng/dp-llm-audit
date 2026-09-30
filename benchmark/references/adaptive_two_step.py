"""Reference implementation: task adaptive_two_step_v1 (correct, eps-DP)."""
import numpy as np


def private_adaptive_release(data, epsilon, rng):
    """Two-step adaptive release: the second query depends on the first OUTPUT.

    Budget: eps/2 per step, so scale 2/eps for both. Each query has
    sensitivity 1 (a count under add/remove-one; the sum of values in [0,1]
    likewise changes by at most 1).

    The property that makes this correct is subtle and is the whole point of
    the task: the branch is taken on the NOISY step-1 release, never on the
    raw count. Branching on the noisy value is post-processing -- the analyst
    already holds it -- so adaptive (sequential) composition applies and the
    total is eps-DP. Branching on the RAW count would make the CHOICE of the
    second query itself a data-dependent release that no noise pays for, and
    the identity of the branch leaks.

    Both branches release a sensitivity-1 quantity at the same scale, so the
    accounting does not depend on which branch is taken.
    """
    eps_step = epsilon / 2.0
    scale = 1.0 / eps_step                       # = 2/epsilon

    high = sum(1 for x in data if x >= 0.5)
    step1 = float(high + rng.laplace(loc=0.0, scale=scale))

    if step1 >= 3.0:                             # branch on the NOISY value
        total = float(np.sum(np.asarray(data, dtype=float))) if data else 0.0
        step2 = float(total + rng.laplace(loc=0.0, scale=scale))
    else:
        low = sum(1 for x in data if x < 0.5)
        step2 = float(low + rng.laplace(loc=0.0, scale=scale))

    return (step1, step2)
