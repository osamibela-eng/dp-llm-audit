"""Functional tests for two_queries_split_budget_v1."""
import numpy as np

from benchmark.references.two_queries import private_count_and_sum

N = 4000


def test_interface():
    rng = np.random.default_rng(0)
    out = private_count_and_sum([0.5, 0.25], 1.0, rng)
    assert isinstance(out, tuple) and len(out) == 2
    assert all(isinstance(x, float) for x in out)


def test_handles_empty():
    rng = np.random.default_rng(0)
    c, s = private_count_and_sum([], 1.0, rng)
    assert isinstance(c, float) and isinstance(s, float)


def test_centers_on_truth():
    rng = np.random.default_rng(0)
    data = [1.0, 0.5, 0.25, 0.25]          # n = 4, sum = 2.0
    outs = np.array([private_count_and_sum(data, 1.0, rng) for _ in range(N)])
    assert abs(outs[:, 0].mean() - 4.0) < 0.2
    assert abs(outs[:, 1].mean() - 2.0) < 0.2


def test_budget_is_split_not_duplicated():
    """Each query must use scale 2/eps, not 1/eps.

    Laplace(scale=b) has variance 2b^2. With eps=1 the correct per-query scale
    is 2.0 -> variance 8.0. Spending the full budget on each query would give
    scale 1.0 -> variance 2.0, so this test separates the two.
    """
    rng = np.random.default_rng(0)
    outs = np.array([private_count_and_sum([0.5] * 5, 1.0, rng) for _ in range(N)])
    for j, name in enumerate(("count", "sum")):
        var = outs[:, j].var()
        assert 5.5 < var < 11.0, f"{name} variance {var:.2f} inconsistent with scale 2/eps"


def test_the_two_noises_are_independent():
    """Shared-noise implementations make (sum - count) deterministic."""
    rng = np.random.default_rng(0)
    outs = np.array([private_count_and_sum([0.5] * 5, 1.0, rng) for _ in range(N)])
    diff_var = (outs[:, 1] - outs[:, 0]).var()
    assert diff_var > 1.0, "difference has no spread -> the two draws are shared"
    corr = np.corrcoef(outs[:, 0], outs[:, 1])[0, 1]
    assert abs(corr) < 0.15, f"noises look correlated (r={corr:.2f})"
