"""Functional tests for gaussian_bounded_count_v1."""
import math

import numpy as np

from benchmark.references.gaussian_count import private_count_gaussian

N = 4000
EPS, DELTA = 1.0, 1e-5
SIGMA = math.sqrt(2.0 * math.log(1.25 / DELTA)) / EPS      # ~4.844


def test_interface():
    rng = np.random.default_rng(0)
    out = private_count_gaussian([1, 0, 1], EPS, DELTA, rng)
    assert isinstance(out, float)


def test_handles_empty():
    rng = np.random.default_rng(0)
    assert isinstance(private_count_gaussian([], EPS, DELTA, rng), float)


def test_is_noisy():
    rng = np.random.default_rng(0)
    outs = {private_count_gaussian([1, 0, 1], EPS, DELTA, rng) for _ in range(20)}
    assert len(outs) > 1


def test_centers_on_true_count():
    rng = np.random.default_rng(0)
    data = [1] * 7 + [0] * 5
    samples = np.array([private_count_gaussian(data, EPS, DELTA, rng) for _ in range(N)])
    assert abs(samples.mean() - 7) < 0.35


def test_sigma_matches_classical_calibration():
    """Standard deviation must match sqrt(2 ln(1.25/delta)) / eps.

    This is the test that separates the reference from both the pure-DP sigma
    (too small) and the misplaced-sqrt variant (too large).
    """
    rng = np.random.default_rng(0)
    samples = np.array([private_count_gaussian([1] * 5, EPS, DELTA, rng) for _ in range(N)])
    assert 0.85 * SIGMA < samples.std() < 1.15 * SIGMA, (
        f"std {samples.std():.2f} inconsistent with classical sigma {SIGMA:.2f}")


def test_smaller_delta_means_more_noise():
    rng = np.random.default_rng(1)
    loose = np.array([private_count_gaussian([1] * 5, EPS, 1e-3, rng) for _ in range(N)])
    tight = np.array([private_count_gaussian([1] * 5, EPS, 1e-9, rng) for _ in range(N)])
    assert tight.std() > loose.std(), "a tighter delta requires more noise"
