"""Reference implementation: task private_topk_v1 (correct, eps-DP)."""
import numpy as np

CATEGORIES = ("A", "B", "C", "D")


def private_top2(data, epsilon, rng):
    """Top-2 selection by PEELING: Report Noisy Max twice on a shrinking set.

    Two adaptive selections on the same data compose sequentially, so each
    round gets eps/2. Each round is a Report Noisy Max over the remaining
    candidates with Laplace(2*Delta/eps_round) noise, matching the convention
    in report_noisy_max_v1 (Delta = 1 for counts under add/remove-one).

        eps_round = eps / 2   ->   scale = 2 / (eps/2) = 4 / eps

    Three properties this gets right and the buggy variants do not:

    1. FRESH noise in round 2. Reusing round 1's draws makes the second
       selection a deterministic function of the first, which both breaks the
       composition argument and correlates the two outputs.
    2. The winner is REMOVED from the candidate set, not merely down-weighted,
       so the two outputs are always distinct as the spec requires.
    3. Only the labels are released -- never the noisy counts. Releasing the
       noisy scores would be a strictly larger release than the analysis covers.
    """
    eps_round = epsilon / 2.0
    scale = 2.0 / eps_round                     # = 4/epsilon

    remaining = list(CATEGORIES)
    winners = []
    for _ in range(2):
        counts = [sum(1 for x in data if x == c) for c in remaining]
        noisy = [c + rng.laplace(loc=0.0, scale=scale) for c in counts]
        idx = int(np.argmax(noisy))
        winners.append(str(remaining[idx]))
        remaining.pop(idx)                      # peel: never selectable again
    return winners
