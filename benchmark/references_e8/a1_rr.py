"""E8 A1 reference: randomized_response_v1 with identifiers renamed.

Truthful with probability e^eps / (1 + e^eps). Parameterised by epsilon, so the
code is independent of the particular claim.
"""
import math


def obfuscate_bit(record, epsilon, rng):
    bit = int(record[0])
    p = math.exp(epsilon) / (1.0 + math.exp(epsilon))
    return int(bit) if rng.random() < p else int(1 - bit)
