"""Functional tests for categorical_histogram_v1."""
import numpy as np

from benchmark.references.categorical_histogram import private_histogram

N = 3000


def test_interface_keys_and_empty():
    rng = np.random.default_rng(0)
    out = private_histogram(["A", "B"], 1.0, rng)
    assert set(out.keys()) == {"A", "B", "C"}
    assert all(isinstance(v, float) for v in out.values())
    out_empty = private_histogram([], 1.0, rng)
    assert set(out_empty.keys()) == {"A", "B", "C"}


def test_ignores_unknown_labels():
    rng = np.random.default_rng(0)
    out = private_histogram(["A", "Z", "Q"], 1.0, rng)  # must not raise
    assert set(out.keys()) == {"A", "B", "C"}


def test_all_bins_noisy_including_zero_bins():
    rng = np.random.default_rng(0)
    c_values = {private_histogram(["A", "A"], 1.0, rng)["C"] for _ in range(20)}
    assert len(c_values) > 1, "zero-count bins must also receive noise"


def test_means_match_counts():
    rng = np.random.default_rng(0)
    data = ["A"] * 5 + ["B"] * 2
    sa = np.array([private_histogram(data, 1.0, rng)["A"] for _ in range(N)])
    sb = np.array([private_histogram(data, 1.0, rng)["B"] for _ in range(N)])
    assert abs(sa.mean() - 5) < 0.2 and abs(sb.mean() - 2) < 0.2
