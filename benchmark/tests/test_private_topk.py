"""Functional tests for private_topk_v1."""
from collections import Counter

import numpy as np

from benchmark.references.private_topk import private_top2

N = 4000


def test_interface():
    rng = np.random.default_rng(0)
    out = private_top2(["A", "B"], 1.0, rng)
    assert isinstance(out, list) and len(out) == 2
    assert all(isinstance(x, str) and x in {"A", "B", "C", "D"} for x in out)


def test_outputs_are_distinct():
    """Peeling must remove the winner; the same label twice is a contract break."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        a, b = private_top2(["A"] * 5 + ["B"] * 3 + ["C"], 1.0, rng)
        assert a != b


def test_handles_empty():
    rng = np.random.default_rng(0)
    out = private_top2([], 1.0, rng)
    assert len(out) == 2 and out[0] != out[1]


def test_is_randomized():
    """A true-count sort would always return the same pair."""
    rng = np.random.default_rng(0)
    data = ["A"] * 5 + ["B"] * 4 + ["C"] * 3
    seen = {tuple(private_top2(data, 1.0, rng)) for _ in range(300)}
    assert len(seen) > 1, "deterministic output -> counts were sorted, not selected"


def test_prefers_the_true_top_two():
    rng = np.random.default_rng(0)
    data = ["A"] * 12 + ["B"] * 9 + ["C"] * 1
    first = Counter(private_top2(data, 1.0, rng)[0] for _ in range(N))
    assert first["A"] > first["B"] > first["C"]


def test_rounds_use_independent_noise():
    """Reused noise couples the rounds: given round 1, round 2 becomes fixed.

    With genuinely fresh draws the second winner varies even when the first is
    held constant.
    """
    rng = np.random.default_rng(0)
    data = ["A"] * 6 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5
    seconds = [b for a, b in (private_top2(data, 1.0, rng) for _ in range(N)) if a == "A"]
    assert len(set(seconds)) > 1, "second winner is determined by the first -> shared noise"


def test_more_budget_sharpens_selection():
    rng = np.random.default_rng(1)
    data = ["A"] * 8 + ["B"] * 6 + ["C"] * 2 + ["D"]
    lo = sum(private_top2(data, 0.2, rng)[0] == "A" for _ in range(N)) / N
    hi = sum(private_top2(data, 5.0, rng)[0] == "A" for _ in range(N)) / N
    assert hi > lo + 0.1
