"""Functional tests for randomized_response_v1."""
import math

import numpy as np

from benchmark.references.randomized_response import randomized_response

EPS = math.log(3.0)          # truthful with probability 3/4
N = 6000


def test_interface():
    rng = np.random.default_rng(0)
    out = randomized_response([1], EPS, rng)
    assert isinstance(out, int)
    assert out in (0, 1)


def test_is_randomized():
    rng = np.random.default_rng(0)
    outs = {randomized_response([1], EPS, rng) for _ in range(200)}
    assert outs == {0, 1}, "both outcomes must be reachable from a single input"


def test_truth_probability_matches_epsilon():
    """At eps = ln(3) the true bit should be reported about 3/4 of the time."""
    rng = np.random.default_rng(0)
    ones = sum(randomized_response([1], EPS, rng) for _ in range(N))
    p_truth = ones / N
    assert abs(p_truth - 0.75) < 0.02, f"expected ~0.75 truthful, got {p_truth:.3f}"


def test_symmetric_in_the_input_bit():
    """Reporting 0 given 0 should be as likely as reporting 1 given 1."""
    rng = np.random.default_rng(1)
    p1 = sum(randomized_response([1], EPS, rng) for _ in range(N)) / N
    p0 = sum(1 - randomized_response([0], EPS, rng) for _ in range(N)) / N
    assert abs(p1 - p0) < 0.03


def test_larger_epsilon_is_more_truthful():
    rng = np.random.default_rng(2)
    lo = sum(randomized_response([1], 0.2, rng) for _ in range(N)) / N
    hi = sum(randomized_response([1], 3.0, rng) for _ in range(N)) / N
    assert hi > lo + 0.15, "truthfulness must increase with epsilon"
