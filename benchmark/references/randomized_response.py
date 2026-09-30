"""Reference implementation: task randomized_response_v1 (correct, eps-DP local)."""
import math


def randomized_response(data, epsilon, rng):
    """Randomized response on a single bit, pure epsilon-DP in the local model.

    Report the true bit with probability p = e^eps / (1 + e^eps), else the flip.
    The likelihood ratio between the two possible inputs is p / (1 - p) = e^eps
    for either output, which is exactly the epsilon-DP condition.

    At eps = ln(3), p = 3/4.
    """
    bit = int(data[0])
    p_truth = math.exp(epsilon) / (1.0 + math.exp(epsilon))
    if rng.random() < p_truth:
        return int(bit)
    return int(1 - bit)
