"""Functional tests for report_noisy_max_v1."""
import numpy as np

from benchmark.references.report_noisy_max import private_argmax, CATEGORIES

N = 3000


def test_interface_and_edge_cases():
    rng = np.random.default_rng(0)
    assert private_argmax(["A", "B", "A"], 1.0, rng) in CATEGORIES
    assert private_argmax([], 1.0, rng) in CATEGORIES          # empty: any category ok
    assert private_argmax(["A", "B"], 1.0, rng) in CATEGORIES  # tie: must not raise


def test_output_is_randomized():
    rng = np.random.default_rng(0)
    outs = {private_argmax(["A", "A", "B", "B"], 1.0, rng) for _ in range(50)}
    assert len(outs) > 1, "selection must be randomized on close counts"


def test_clear_winner_usually_wins():
    rng = np.random.default_rng(0)
    data = ["A"] * 20 + ["B"] * 2
    wins = sum(private_argmax(data, 1.0, rng) == "A" for _ in range(N))
    assert wins / N > 0.95, "a clear winner should be selected with high probability"
