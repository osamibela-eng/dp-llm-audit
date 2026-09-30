"""Functional tests for exponential_mechanism_v1."""
import math
from collections import Counter

import numpy as np

from benchmark.references.exponential_mechanism import private_select

N = 8000


def test_interface():
    rng = np.random.default_rng(0)
    out = private_select(["A", "B"], 1.0, rng)
    assert isinstance(out, str) and out in {"A", "B", "C", "D"}


def test_handles_empty():
    rng = np.random.default_rng(0)
    assert private_select([], 1.0, rng) in {"A", "B", "C", "D"}


def test_is_randomized():
    """Every category must remain reachable -- an argmax implementation fails."""
    rng = np.random.default_rng(0)
    data = ["A"] * 6 + ["B"] * 2
    outs = {private_select(data, 1.0, rng) for _ in range(400)}
    assert len(outs) > 1, "deterministic selection -> not the exponential mechanism"


def test_prefers_the_most_common_category():
    rng = np.random.default_rng(0)
    data = ["A"] * 8 + ["B"] * 2 + ["C"]
    counts = Counter(private_select(data, 1.0, rng) for _ in range(N))
    assert counts["A"] > counts["B"] > counts["C"]


def test_matches_closed_form_distribution():
    """Probabilities must follow exp(eps*u/2), normalised -- the factor of 2 matters.

    With counts A=4, B=2, C=0, D=0 at eps=1 the correct weights are
    exp(2), exp(1), exp(0), exp(0); dropping the /2 would give exp(4), exp(2), 1, 1
    and a visibly different distribution for A.
    """
    rng = np.random.default_rng(0)
    data = ["A"] * 4 + ["B"] * 2
    eps = 1.0
    w = [math.exp(eps * u / 2.0) for u in (4, 2, 0, 0)]
    expected_a = w[0] / sum(w)

    counts = Counter(private_select(data, eps, rng) for _ in range(N))
    observed_a = counts["A"] / N
    assert abs(observed_a - expected_a) < 0.03, (
        f"P(A) = {observed_a:.3f}, expected {expected_a:.3f} "
        "(a missing factor of 2 would push this much higher)")


def test_larger_epsilon_concentrates_on_the_winner():
    rng = np.random.default_rng(1)
    data = ["A"] * 5 + ["B"] * 3
    lo = sum(private_select(data, 0.2, rng) == "A" for _ in range(N)) / N
    hi = sum(private_select(data, 3.0, rng) == "A" for _ in range(N)) / N
    assert hi > lo + 0.1


def test_no_overflow_at_large_counts():
    """Naive exp() overflows without the max-subtraction guard."""
    rng = np.random.default_rng(0)
    data = ["A"] * 5000
    assert private_select(data, 10.0, rng) in {"A", "B", "C", "D"}
