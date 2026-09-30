"""Functional tests for bounded_mean_v1.

Interface, semantic, and reference-distribution levels. Functional tests MAY
seed the rng; audit runs never do (auditors/AUDITOR_NOTES.md section 4).
"""
import numpy as np

from benchmark.references.bounded_mean import private_mean

N = 4000


def test_interface():
    rng = np.random.default_rng(0)
    out = private_mean([0.5, 0.25], 1.0, rng)
    assert isinstance(out, float)


def test_is_noisy():
    rng = np.random.default_rng(0)
    outs = {private_mean([0.5, 0.25, 0.75], 1.0, rng) for _ in range(20)}
    assert len(outs) > 1, "output must be randomized"


def test_centers_on_true_mean():
    rng = np.random.default_rng(0)
    data = [1.0] * 6 + [0.0] * 6          # true mean 0.5, n = 12
    samples = np.array([private_mean(data, 1.0, rng) for _ in range(N)])
    assert abs(samples.mean() - 0.5) < 0.02


def test_noise_scales_with_one_over_n():
    """Sensitivity is 1/n, so the noise variance must fall like 1/n^2.

    This is the test that separates the reference from the sum-sensitivity bug:
    doubling n should cut the standard deviation roughly in half.
    """
    rng = np.random.default_rng(0)
    small = np.array([private_mean([0.5] * 10, 1.0, rng) for _ in range(N)])
    large = np.array([private_mean([0.5] * 20, 1.0, rng) for _ in range(N)])
    ratio = small.std() / large.std()
    assert 1.5 < ratio < 2.6, f"std should roughly halve when n doubles, got ratio {ratio:.2f}"


def test_variance_matches_scale():
    """Laplace(scale=b) has variance 2b^2; here b = 1/(n*eps)."""
    rng = np.random.default_rng(0)
    n, eps = 10, 1.0
    b = 1.0 / (n * eps)
    samples = np.array([private_mean([0.5] * n, eps, rng) for _ in range(N)])
    expected = 2.0 * b * b
    assert 0.6 * expected < samples.var() < 1.6 * expected
