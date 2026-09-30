"""E6: derive a repair diagnosis mechanically from the audit result.

The objection this kills is that our negative repair result used weak feedback.
R2 hands the model a counterexample and nothing else; a reviewer can reasonably
say that a counterexample without an interpretation is a thin signal, and that
richer feedback might have worked.

R3 tests that. The feedback is a counterexample PLUS a diagnosis, and the
credibility of the whole arm rests on the diagnosis being derived rather than
written. Everything below is computed from three signals that a tool already has:

  1. support mismatch / determinism   the smoke test already detects an output
                                      that never varies, and a counterexample
                                      whose lower bound is 1.0 or whose upper
                                      bound is 0.0 is a support violation
  2. under-noising magnitude          the empirical epsilon lower bound against
                                      the claim gives a multiplicative shortfall
  3. composition structure            how many values the task releases, read
                                      off the declared output type

There are no per-task hints, no human-written explanations, and nothing that
depends on having read the program. If a future reader suspects we hand-tuned the
feedback, this file is the answer: the templates below are the whole vocabulary.

    from repair.diagnose import diagnose
    text = diagnose(task, contract, counterexample)
"""
from __future__ import annotations

import math
from typing import Any

#: How many values each declared output type releases in one call. Composition
#: applies whenever this exceeds one, and it is read from the contract rather
#: than from any knowledge of the mechanism.
RELEASES = {"float": 1, "int": 1, "str": 1, "dict": None, "tuple": 2, "list": None}


def _support_violation(ce: dict) -> bool:
    """An event with probability pinned at 1 on one side and 0 on the other.

    No amount of noise repairs this: the two neighbours are perfectly
    distinguishable, so the claim fails at every epsilon.
    """
    p1 = float(ce.get("p1_lower_bound", 0.0))
    p2 = float(ce.get("p2_upper_bound", 1.0))
    return p1 >= 0.999 or p2 <= 1e-6


def _shortfall(ce: dict, claimed_eps: float) -> float | None:
    """exp(observed eps lower bound - claimed eps): how many times too little noise."""
    try:
        eps_lb = float(ce["empirical_eps_lower_bound"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(eps_lb):
        return None
    return math.exp(max(0.0, eps_lb - claimed_eps))


def diagnose(task: dict, contract: Any, ce: dict | None) -> str:
    """Template-generated diagnosis. Returns '' when no signal fires."""
    if not ce:
        return ""

    claimed_eps = float(task.get("claimed_epsilon") or 0.0)
    lines: list[str] = []

    # ---- signal 1: support ------------------------------------------------
    if _support_violation(ce):
        lines.append(
            "SUPPORT VIOLATION. On the two neighbouring datasets above, one "
            "output event has probability essentially 1 on one side and "
            "essentially 0 on the other. The two datasets are therefore "
            "perfectly distinguishable and no choice of noise scale repairs "
            "this. Look for a branch, an early return, or a noise scale that "
            "collapses to zero, on one of these two inputs.")

    # ---- signal 2: magnitude ---------------------------------------------
    factor = _shortfall(ce, claimed_eps)
    if factor is not None and factor > 1.5 and not _support_violation(ce):
        lines.append(
            f"NOISE MAGNITUDE. The observed privacy loss on this event is about "
            f"{factor:.1f} times the claimed bound of exp({claimed_eps:g}). That "
            f"is consistent with a noise scale that is too small by roughly this "
            f"factor, or with a sensitivity constant that is too small by it. "
            f"Check the value substituted for the sensitivity, and check whether "
            f"epsilon divides the scale or multiplies it.")

    # ---- signal 3: composition -------------------------------------------
    n_rel = RELEASES.get(getattr(contract, "output_type", "float"), None)
    if n_rel is None or n_rel > 1:
        how = (f"{n_rel} values" if n_rel else
               "more than one value (a dictionary or list of releases)")
        lines.append(
            f"COMPOSITION. This task releases {how} from the same data in a "
            f"single call. Each release consumes part of the privacy budget, so "
            f"the per-release epsilon must be smaller than the claimed total. "
            f"Check that the budget is divided across releases rather than spent "
            f"in full on each.")

    if not lines:
        return ""
    return ("=== MECHANICAL DIAGNOSIS ===\n"
            "The following was derived automatically from the audit result. It "
            "was not written by a person and does not depend on reading your "
            "code.\n\n" + "\n\n".join(f"- {t}" for t in lines))


def signals_fired(task: dict, contract: Any, ce: dict | None) -> list[str]:
    """Which signals fired, for reporting how often each contributes."""
    if not ce:
        return []
    out = []
    if _support_violation(ce):
        out.append("support")
    f = _shortfall(ce, float(task.get("claimed_epsilon") or 0.0))
    if f is not None and f > 1.5 and not _support_violation(ce):
        out.append("magnitude")
    n_rel = RELEASES.get(getattr(contract, "output_type", "float"), None)
    if n_rel is None or n_rel > 1:
        out.append("composition")
    return out
