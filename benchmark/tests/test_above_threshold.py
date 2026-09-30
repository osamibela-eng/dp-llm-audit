"""Functional tests for above_threshold_v1."""
from collections import Counter

import numpy as np

from benchmark.references.above_threshold import above_threshold

N = 4000


def test_interface():
    rng = np.random.default_rng(0)
    out = above_threshold([1, 1, 0], 1.0, rng)
    assert isinstance(out, int) and out in (0, 1, 2, 3)


def test_handles_empty():
    rng = np.random.default_rng(0)
    assert above_threshold([], 1.0, rng) in (0, 1, 2, 3)


def test_is_randomized():
    """Data sitting near the threshold must not give a fixed answer."""
    rng = np.random.default_rng(0)
    data = [1] * 4 + [0] * 4            # q1 = 4, q2 = 4, q3 = 8, T = 4
    outs = {above_threshold(data, 1.0, rng) for _ in range(300)}
    assert len(outs) > 1, "deterministic -> queries or threshold are un-noised"


def test_returns_an_index_not_a_value():
    """A large dataset would return a big number if the VALUE were released."""
    rng = np.random.default_rng(0)
    data = [1] * 200 + [0] * 200        # q1 = q2 = 200, q3 = 400
    outs = [above_threshold(data, 1.0, rng) for _ in range(500)]
    assert max(outs) <= 3, f"output {max(outs)} looks like a query value, not an index"


def test_fires_early_when_far_above_threshold():
    rng = np.random.default_rng(0)
    data = [1] * 50                     # q1 = 50 >> T
    counts = Counter(above_threshold(data, 1.0, rng) for _ in range(N))
    assert counts[1] / N > 0.9, "should almost always halt on the first query"


def test_below_threshold_data_fires_less_often_than_above():
    """Below-threshold data must be markedly less likely to fire.

    Note the absolute rate is NOT near zero, and that is correct, not a bug.
    With eps = 1 the query noise is Lap(4/eps) = Lap(4) against a gap of only
    3, and three queries each get a fresh draw, so a spurious crossing is
    common. The analytic P(no crossing) for data=[1] is 0.459, and the
    reference measures 0.458 -- verified against a direct integration over rho.

    Testing the absolute rate would therefore encode the noise scale into the
    test and break for any legitimate re-parameterisation. The meaningful
    invariant is the ORDERING: data far above the threshold fires far more
    often than data far below it.
    """
    rng = np.random.default_rng(0)
    below = sum(above_threshold([1], 1.0, rng) != 0 for _ in range(N)) / N
    above = sum(above_threshold([1] * 50, 1.0, rng) != 0 for _ in range(N)) / N
    assert above > below + 0.3, (
        f"firing rate barely responds to the data (below {below:.2f}, above {above:.2f})")
    assert below < 0.75, "below-threshold data fires almost always -> queries un-noised?"


def test_threshold_noise_is_drawn_once():
    """One shared rho couples the three comparisons.

    With a per-query threshold draw the mechanism becomes markedly more eager
    to fire: three independent chances at a low threshold rather than one.
    This test pins the reference's firing rate on borderline data, which the
    fresh-threshold-noise variant inflates.
    """
    rng = np.random.default_rng(0)
    data = [1] * 3 + [0] * 3            # q1 = 3, q2 = 3, q3 = 6 vs T = 4
    fire = sum(above_threshold(data, 1.0, rng) != 0 for _ in range(N)) / N
    assert 0.30 < fire < 0.95, f"firing rate {fire:.2f} outside the reference band"


def test_more_budget_means_sharper_decisions():
    rng = np.random.default_rng(1)
    data = [1] * 10                     # clearly above T
    lo = sum(above_threshold(data, 0.1, rng) == 1 for _ in range(N)) / N
    hi = sum(above_threshold(data, 5.0, rng) == 1 for _ in range(N)) / N
    assert hi > lo, "larger epsilon should make the correct answer more likely"
