"""Functional tests for the tier-3 tasks.

adaptive_two_step_v1, validated_clipped_count_v1, empty_dataset_count_v1 and
subsampled_count_v1 share a file because each has few but pointed tests.
"""
import math

import numpy as np
import pytest

from benchmark.references.adaptive_two_step import private_adaptive_release
from benchmark.references.empty_dataset_count import private_count_safe
from benchmark.references.subsampled_count import private_subsampled_count
from benchmark.references.validated_count import private_valid_count

N = 4000


# ---------------------------------------------------------------- adaptive
def test_adaptive_interface():
    rng = np.random.default_rng(0)
    out = private_adaptive_release([0.9, 0.1], 1.0, rng)
    assert isinstance(out, tuple) and len(out) == 2
    assert all(isinstance(x, float) for x in out)


def test_adaptive_budget_is_split():
    """eps/2 per step -> scale 2/eps -> variance 2*(2)^2 = 8 at eps=1."""
    rng = np.random.default_rng(0)
    outs = np.array([private_adaptive_release([0.9] * 5, 1.0, rng) for _ in range(N)])
    assert 5.5 < outs[:, 0].var() < 11.0


def test_adaptive_branch_is_reachable_both_ways():
    """Data near the branch boundary must sometimes take each path.

    If the branch were taken on the RAW count it would be fixed for fixed data;
    branching on the noisy release makes it random.
    """
    rng = np.random.default_rng(0)
    data = [0.9] * 3                       # true high count = 3, right at the edge
    seconds = [private_adaptive_release(data, 1.0, rng)[1] for _ in range(600)]
    assert len(set(np.round(seconds, 6))) > 1


# --------------------------------------------------------------- validated
def test_validated_never_raises_on_junk():
    rng = np.random.default_rng(0)
    junk = [5, "oops", None, 3.7, {"a": 1}, [1, 2], object(), -4, 101, 50]
    out = private_valid_count(junk, 1.0, rng)
    assert isinstance(out, float)


def test_validated_counts_only_valid_integers():
    rng = np.random.default_rng(0)
    data = [0, 50, 100, 101, -1, "7", 7.0, None, True]   # valid: 0, 50, 100
    samples = np.array([private_valid_count(data, 1.0, rng) for _ in range(N)])
    assert abs(samples.mean() - 3.0) < 0.2


def test_validated_rejects_booleans():
    """isinstance(True, int) is True in Python -- a real validation trap."""
    rng = np.random.default_rng(0)
    samples = np.array([private_valid_count([True, False], 1.0, rng) for _ in range(N)])
    assert abs(samples.mean() - 0.0) < 0.2


def test_validated_is_noisy():
    rng = np.random.default_rng(0)
    outs = {private_valid_count([1, 2, 3], 1.0, rng) for _ in range(20)}
    assert len(outs) > 1


# ------------------------------------------------------------------- empty
def test_empty_uses_the_same_noisy_path():
    """The empty dataset must not produce a deterministic output."""
    rng = np.random.default_rng(0)
    outs = {private_count_safe([], 1.0, rng) for _ in range(200)}
    assert len(outs) > 1, "empty input gives a fixed value -> special-cased"


def test_empty_is_not_rounded():
    """Rounding only when empty would put every output on the integer lattice."""
    rng = np.random.default_rng(0)
    outs = [private_count_safe([], 1.0, rng) for _ in range(200)]
    assert any(abs(o - round(o)) > 1e-9 for o in outs)


def test_empty_centers_on_zero():
    rng = np.random.default_rng(0)
    samples = np.array([private_count_safe([], 1.0, rng) for _ in range(N)])
    assert abs(samples.mean()) < 0.15


def test_empty_does_not_raise():
    rng = np.random.default_rng(0)
    assert isinstance(private_count_safe([], 1.0, rng), float)


# -------------------------------------------------------------- subsampled
def test_subsampled_interface():
    rng = np.random.default_rng(0)
    assert isinstance(private_subsampled_count([1, 0, 1], 0.5, rng), float)


def test_subsampled_centers_on_q_times_count():
    """Poisson sampling at q = 0.5 halves the expected count."""
    rng = np.random.default_rng(0)
    data = [1] * 40
    samples = np.array([private_subsampled_count(data, 0.5, rng) for _ in range(N)])
    assert abs(samples.mean() - 20.0) < 1.0


def test_subsampled_base_budget_is_looser_than_the_claim():
    """eps0 = ln(1 + (e^eps - 1)/q) must exceed eps -- amplification pays the gap.

    At eps = 0.5, q = 0.5 the base budget is 0.8318. A variant that runs the
    base mechanism at eps0 = eps would show visibly MORE noise than this.
    """
    eps, q = 0.5, 0.5
    eps0 = math.log(1.0 + (math.exp(eps) - 1.0) / q)
    assert eps0 > eps
    assert abs(eps0 - 0.8318) < 1e-3

    rng = np.random.default_rng(0)
    samples = np.array([private_subsampled_count([1] * 10, eps, rng) for _ in range(N)])
    # variance = sampling variance (n*q*(1-q) = 2.5) + Laplace variance 2/eps0^2
    expected = 10 * q * (1 - q) + 2.0 / (eps0 ** 2)
    assert 0.6 * expected < samples.var() < 1.6 * expected


def test_subsampled_uses_the_supplied_rng():
    """Two runs with identically seeded Generators must agree."""
    a = private_subsampled_count([1] * 20, 0.5, np.random.default_rng(123))
    b = private_subsampled_count([1] * 20, 0.5, np.random.default_rng(123))
    assert a == b, "output not reproducible from the supplied rng -> other randomness used"
