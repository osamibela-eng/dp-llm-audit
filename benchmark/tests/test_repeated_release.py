"""Functional tests for repeated_release_v1."""
import numpy as np

from benchmark.references.repeated_release import private_repeated_count

N = 3000


def test_interface():
    rng = np.random.default_rng(0)
    out = private_repeated_count([1, 0, 1], 1.0, rng)
    assert isinstance(out, list) and len(out) == 4
    assert all(isinstance(x, float) for x in out)


def test_centers_on_true_count():
    rng = np.random.default_rng(0)
    data = [1] * 6 + [0] * 4
    outs = np.array([private_repeated_count(data, 1.0, rng) for _ in range(N)])
    assert abs(outs.mean() - 6.0) < 0.35


def test_releases_are_independent():
    """The four values must be four fresh draws, not one draw copied."""
    rng = np.random.default_rng(0)
    outs = np.array([private_repeated_count([1] * 5, 1.0, rng) for _ in range(N)])
    within = np.array([row.std() for row in outs])
    assert within.mean() > 0.5, "the four releases barely differ -> shared noise"
    corr = np.corrcoef(outs[:, 0], outs[:, 1])[0, 1]
    assert abs(corr) < 0.15, f"releases 0 and 1 are correlated (r={corr:.2f})"


def test_budget_split_across_four_releases():
    """Each release gets eps/4, so scale = 4/eps -> variance 2*(4)^2 = 32."""
    rng = np.random.default_rng(0)
    outs = np.array([private_repeated_count([1] * 5, 1.0, rng) for _ in range(N)])
    var = outs[:, 0].var()
    assert 22.0 < var < 45.0, f"variance {var:.1f} inconsistent with scale 4/eps"


def test_more_budget_means_less_noise():
    rng = np.random.default_rng(1)
    tight = np.array([private_repeated_count([1] * 5, 0.5, rng)[0] for _ in range(N)])
    loose = np.array([private_repeated_count([1] * 5, 4.0, rng)[0] for _ in range(N)])
    assert loose.var() < tight.var()
