"""Functional tests for laplace_bounded_count_v1.

Three levels: interface, semantic, reference-distribution. Functional tests may
seed the rng for reproducibility; AUDIT runs must never be seeded (see
auditors/AUDITOR_NOTES.md).
"""
import numpy as np
import pytest

from benchmark.references.laplace_count import private_count

N = 4000


def test_interface():
    rng = np.random.default_rng(0)
    out = private_count([1, 0, 1], 1.0, rng)
    assert isinstance(out, float)


def test_handles_empty():
    rng = np.random.default_rng(0)
    out = private_count([], 1.0, rng)
    assert isinstance(out, float)


def test_is_noisy():
    rng = np.random.default_rng(0)
    outs = {private_count([1, 0, 1], 1.0, rng) for _ in range(20)}
    assert len(outs) > 1, "output must be randomized"


def test_unbiased_and_correct_scale():
    rng = np.random.default_rng(0)
    data = [1] * 7 + [0] * 5
    samples = np.array([private_count(data, 1.0, rng) for _ in range(N)])
    assert abs(samples.mean() - 7) < 0.15, "noisy count should center on the true count"
    # Laplace(scale=1) has variance 2; accept [1.5, 2.6] at N=4000
    assert 1.5 < samples.var() < 2.6, "noise variance inconsistent with scale 1/epsilon"
