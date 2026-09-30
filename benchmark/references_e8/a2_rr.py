"""E8 A2 reference: randomized response claimed at ln(1.5) rather than ln(3).

The correct implementation is parameterised by epsilon and is therefore
identical in form to the canonical one: report truthfully with probability
e^eps / (1 + e^eps), which at eps = ln(1.5) is 3/5.

Why the claim was lowered rather than raised. Raising it to ln(9) would make
the canonical "truthful with probability 3/4" constant MORE private than
claimed, so a model reproducing it from memory would pass and the task would
discriminate nothing. Lowering the claim to ln(1.5) makes that same memorised
constant an overspend of exactly

    e^(ln 3 - ln 1.5) = 2,

which is a real violation and one the auditor can find.
"""
import math


def private_bit_report(data, epsilon, rng):
    bit = int(data[0])
    p = math.exp(epsilon) / (1.0 + math.exp(epsilon))
    return int(bit) if rng.random() < p else int(1 - bit)
