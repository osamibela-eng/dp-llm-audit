"""Known-buggy variants of private_topk_v1, for auditor calibration."""
import numpy as np

CATEGORIES = ("A", "B", "C", "D")

EXPECTED = {
    # full epsilon in each of the two peeling rounds -> 2*eps-DP
    "private_top2_full_budget_each": "detectable",
    # no noise at all: returns the two true most-frequent labels
    "private_top2_sorts_true_counts": "detectable",
    # round 2 reuses round 1's noise draws instead of drawing fresh
    "private_top2_reuses_noise": "detectable",
    # splits the budget four ways instead of two -> over-noised, still private
    "private_top2_over_split": "likely_missed",
}


def _counts(data, cats):
    return [sum(1 for x in data if x == c) for c in cats]


def private_top2_full_budget_each(data, epsilon, rng):
    """BUG: each peeling round spends the whole budget."""
    scale = 2.0 / epsilon
    remaining, winners = list(CATEGORIES), []
    for _ in range(2):
        noisy = [c + rng.laplace(scale=scale) for c in _counts(data, remaining)]
        i = int(np.argmax(noisy))
        winners.append(str(remaining[i]))
        remaining.pop(i)
    return winners


def private_top2_sorts_true_counts(data, epsilon, rng):
    """BUG: no noise -- returns the two genuinely most frequent labels."""
    counts = _counts(data, CATEGORIES)
    order = np.argsort(counts)[::-1]
    return [str(CATEGORIES[int(order[0])]), str(CATEGORIES[int(order[1])])]


def private_top2_reuses_noise(data, epsilon, rng):
    """BUG: draws noise once and reuses it for the second round.

    The second selection becomes a deterministic function of the first round's
    draws, so the two rounds are not independent and composition does not hold.
    """
    eps_round = epsilon / 2.0
    scale = 2.0 / eps_round
    noise = {c: rng.laplace(scale=scale) for c in CATEGORIES}   # drawn ONCE
    remaining, winners = list(CATEGORIES), []
    for _ in range(2):
        noisy = [cnt + noise[c] for c, cnt in zip(remaining, _counts(data, remaining))]
        i = int(np.argmax(noisy))
        winners.append(str(remaining[i]))
        remaining.pop(i)
    return winners


def private_top2_over_split(data, epsilon, rng):
    """BUG: budget split four ways for two rounds -> over-noised but private."""
    eps_round = epsilon / 4.0
    scale = 2.0 / eps_round
    remaining, winners = list(CATEGORIES), []
    for _ in range(2):
        noisy = [c + rng.laplace(scale=scale) for c in _counts(data, remaining)]
        i = int(np.argmax(noisy))
        winners.append(str(remaining[i]))
        remaining.pop(i)
    return winners
