"""Functional tests for clipped_bounded_sum_v1."""
import numpy as np

from benchmark.references.clipped_sum import private_sum

N = 4000


def test_interface_and_empty():
    rng = np.random.default_rng(0)
    assert isinstance(private_sum([1.0, 2.0], 1.0, rng), float)
    assert isinstance(private_sum([], 1.0, rng), float)


def test_clipping_applied():
    rng = np.random.default_rng(0)
    # value 1000 must be clipped to 10: mean of output should be ~10, not ~1000
    samples = np.array([private_sum([1000.0], 1.0, rng) for _ in range(N)])
    assert abs(samples.mean() - 10.0) < 1.0


def test_noise_scale():
    rng = np.random.default_rng(0)
    samples = np.array([private_sum([5.0, 5.0], 1.0, rng) for _ in range(N)])
    assert abs(samples.mean() - 10.0) < 1.0
    # Laplace(scale=10) variance = 200; accept [140, 270]
    assert 140 < samples.var() < 270
