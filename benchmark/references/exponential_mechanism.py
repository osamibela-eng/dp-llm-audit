"""Reference implementation: task exponential_mechanism_v1 (correct, eps-DP)."""
import math

CATEGORIES = ("A", "B", "C", "D")


def private_select(data, epsilon, rng):
    """Exponential mechanism over four categories, utility = count.

    Sample category c with probability proportional to exp(eps * u(D,c) / (2*Du))
    where u(D,c) is the count of c and Du = 1 is its sensitivity. The factor of
    2 is what makes this epsilon-DP rather than 2*epsilon-DP, and dropping it is
    the single most common error in hand-written implementations.

    Overflow guard: subtract the maximum exponent before exponentiating. This is
    a pure shift of the weights and does not change the distribution.
    """
    counts = {c: 0 for c in CATEGORIES}
    for x in data:
        if x in counts:
            counts[x] += 1

    sensitivity = 1.0
    exponents = [epsilon * counts[c] / (2.0 * sensitivity) for c in CATEGORIES]
    m = max(exponents)
    weights = [math.exp(e - m) for e in exponents]      # shift-invariant
    total = sum(weights)
    probs = [w / total for w in weights]

    u = rng.random()
    acc = 0.0
    for c, p in zip(CATEGORIES, probs):
        acc += p
        if u <= acc:
            return str(c)
    return str(CATEGORIES[-1])                          # floating-point tail
