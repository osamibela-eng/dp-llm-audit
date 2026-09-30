"""
simple_auditor -- a self-contained statistical falsifier for (epsilon, delta)-DP claims.

WHAT IT IS
  A dependency-light (numpy + scipy only) black-box DP auditor in the spirit of
  StatDP/DP-Sniper: it runs a mechanism many times on fixed neighboring dataset
  pairs, searches a family of output events, and uses Clopper-Pearson confidence
  bounds with Bonferroni correction to decide whether the observed output
  probabilities are statistically inconsistent with the claimed (eps, delta)-DP
  guarantee.

WHAT A RESULT MEANS (put this in the paper):
  - status == "falsified":  with probability >= 1 - alpha over the audit's own
    randomness, the mechanism violates the claimed guarantee at the reported
    (D, D', event). This is sound evidence of a bug.
  - status == "not_falsified":  the mechanism SURVIVED this audit budget over
    this family of pairs and events. It is NOT certified DP.
  - status == "unsupported":  output type/shape not auditable by this tool.

DESIGN DECISIONS (documented for the paper's audit-policy section):
  - Pilot/confirm split: candidate events are chosen from a small pilot sample,
    then tested on a FRESH confirmation sample, so event selection does not
    invalidate the confidence bounds.
  - Bonferroni over all (pair, direction, event) tests at overall level alpha.
  - Events are coarse (thresholds and exact values on canonicalized outputs).
    Floating-point/bit-level leakage is deliberately OUT OF SCOPE.
  - The auditor supplies the RNG. Audit runs must never be seeded by the
    harness; pass seed=... only to reproduce a specific audit run end-to-end.

Mechanism contract:  mech(data: list, epsilon: float, rng: np.random.Generator) -> output
  where output is a number, a str/bool category, a list/tuple of numbers,
  or a dict mapping str -> number.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
from scipy.stats import beta


# ----------------------------------------------------------------------------- utils

def cp_lower(k: int, n: int, a: float) -> float:
    """Clopper-Pearson lower confidence bound for a binomial proportion."""
    if k <= 0:
        return 0.0
    return float(beta.ppf(a, k, n - k + 1))


def cp_upper(k: int, n: int, a: float) -> float:
    """Clopper-Pearson upper confidence bound for a binomial proportion."""
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - a, k + 1, n - k))


class UnsupportedOutput(Exception):
    pass


def _canonicalize(out: Any) -> tuple:
    """Map a mechanism output to a flat tuple of floats, or ('CAT', label)."""
    if isinstance(out, (str, bool)):
        return ("CAT", str(out))
    if isinstance(out, dict):
        try:
            items = sorted(out.items())
        except TypeError:
            raise UnsupportedOutput("dict with unsortable keys")
        return tuple(float(v) for _, v in items)
    if isinstance(out, (list, tuple, np.ndarray)):
        arr = np.asarray(out).ravel()
        if arr.dtype.kind in "OUS":
            # Non-numeric sequence, e.g. the ordered top-k labels ['A', 'B'].
            # Treat the whole sequence as a single composite category. Order is
            # preserved deliberately: for a top-k release, ['A','B'] and
            # ['B','A'] are different outputs and a mechanism that confuses them
            # is leaking rank information, so they must not be pooled.
            try:
                return ("CAT", "|".join(str(x) for x in arr.tolist()))
            except Exception as exc:                      # pragma: no cover
                raise UnsupportedOutput(f"unstringifiable sequence: {exc}")
        return tuple(float(x) for x in arr)
    if isinstance(out, (int, float, np.integer, np.floating)):
        return (float(out),)
    raise UnsupportedOutput(f"output type {type(out)!r}")


def _sample(mech: Callable, data: list, eps: float, n: int,
            rng: np.random.Generator) -> list[tuple]:
    outs = []
    for _ in range(n):
        outs.append(_canonicalize(mech(list(data), eps, rng)))
    return outs


def _to_matrix(samples: list[tuple]):
    """Return ('cat', np.array[str]) or ('num', 2d float array). Raises if mixed/ragged."""
    if any(s and s[0] == "CAT" for s in samples):
        if not all(s and s[0] == "CAT" for s in samples):
            raise UnsupportedOutput("mixed categorical/numeric outputs")
        return "cat", np.array([s[1] for s in samples])
    d = len(samples[0])
    if any(len(s) != d for s in samples):
        raise UnsupportedOutput("ragged output dimensions")
    return "num", np.array(samples, dtype=float)


# ----------------------------------------------------------------------------- events

@dataclass
class Event:
    desc: str
    fn: Callable[[Any], np.ndarray]  # matrix -> boolean vector


def _pilot_events(kind: str, pilots: list, max_thresholds: int = 12,
                  exact_min_freq: float = 0.03) -> list[Event]:
    """Build candidate events from pilot samples only (both sides pooled)."""
    events: list[Event] = []
    if kind == "cat":
        values = sorted(set(v for p in pilots for v in p))
        for v in values:
            events.append(Event(f"output == {v!r}", lambda M, v=v: M == v))
        return events

    pooled = np.vstack(pilots)
    d = pooled.shape[1]
    for i in range(d):
        col = pooled[:, i]
        qs = np.unique(np.quantile(col, np.linspace(0.05, 0.95, max_thresholds)))
        for t in qs:
            events.append(Event(f"out[{i}] >= {t:.6g}", lambda M, i=i, t=t: M[:, i] >= t))
            events.append(Event(f"out[{i}] <= {t:.6g}", lambda M, i=i, t=t: M[:, i] <= t))
        # exact-value events catch point masses (no-noise / support-leak bugs)
        vals, counts = np.unique(np.round(col, 10), return_counts=True)
        for v, c in zip(vals, counts):
            if c / len(col) >= exact_min_freq:
                events.append(Event(f"out[{i}] == {v:.6g}",
                                    lambda M, i=i, v=v: np.round(M[:, i], 10) == v))
    return events


# ----------------------------------------------------------------------------- result

@dataclass
class Counterexample:
    d1: list
    d2: list
    event: str
    p1_lb: float
    p2_ub: float
    eps_lower_bound: float
    n_confirm: int

    def as_dict(self) -> dict:
        return {"D": self.d1, "D_prime": self.d2, "event": self.event,
                "p1_lower_bound": self.p1_lb, "p2_upper_bound": self.p2_ub,
                "empirical_eps_lower_bound": self.eps_lower_bound,
                "n_confirm_samples": self.n_confirm}


@dataclass
class AuditResult:
    status: str                       # falsified | not_falsified | unsupported | error
    claimed_eps: float
    claimed_delta: float
    counterexample: Optional[Counterexample] = None
    tests_run: int = 0
    elapsed_s: float = 0.0
    detail: str = ""
    config: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- main

def falsify(mech: Callable,
            pairs: list[tuple[list, list]],
            claimed_eps: float,
            claimed_delta: float = 0.0,
            n_confirm: int = 50_000,
            n_pilot: int = 4_000,
            alpha: float = 0.01,
            seed: Optional[int] = None) -> AuditResult:
    """Attempt to statistically falsify the claim that `mech` is
    (claimed_eps, claimed_delta)-DP with respect to the given neighboring pairs.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    e_eps = math.exp(claimed_eps)

    try:
        # ---- pilot phase: choose events per pair (independent of confirm data)
        per_pair_events: list[list[Event]] = []
        kinds = []
        for d1, d2 in pairs:
            p1 = _sample(mech, d1, claimed_eps, n_pilot, rng)
            p2 = _sample(mech, d2, claimed_eps, n_pilot, rng)
            k1, m1 = _to_matrix(p1)
            k2, m2 = _to_matrix(p2)
            if k1 != k2:
                raise UnsupportedOutput("output kind differs across neighbors")
            kinds.append(k1)
            per_pair_events.append(_pilot_events(k1, [m1, m2] if k1 == "num" else [m1, m2]))

        total_tests = sum(2 * len(ev) for ev in per_pair_events)  # 2 directions
        if total_tests == 0:
            return AuditResult("unsupported", claimed_eps, claimed_delta,
                               detail="no candidate events", elapsed_s=time.time() - t0)
        a_adj = alpha / total_tests / 2.0  # /2: lower+upper bound per test

        # ---- confirm phase: fresh samples, Bonferroni-corrected CP bounds
        best: Optional[Counterexample] = None
        for (d1, d2), events, kind in zip(pairs, per_pair_events, kinds):
            s1 = _sample(mech, d1, claimed_eps, n_confirm, rng)
            s2 = _sample(mech, d2, claimed_eps, n_confirm, rng)
            _, M1 = _to_matrix(s1)
            _, M2 = _to_matrix(s2)
            for ev in events:
                k_a = int(ev.fn(M1).sum())
                k_b = int(ev.fn(M2).sum())
                for (ka, kb, da, db) in ((k_a, k_b, d1, d2), (k_b, k_a, d2, d1)):
                    p1_lb = cp_lower(ka, n_confirm, a_adj)
                    p2_ub = cp_upper(kb, n_confirm, a_adj)
                    if p1_lb > e_eps * p2_ub + claimed_delta:
                        num = max(p1_lb - claimed_delta, 1e-300)
                        eps_lb = math.log(num / max(p2_ub, 1e-300))
                        if best is None or eps_lb > best.eps_lower_bound:
                            best = Counterexample(list(da), list(db), ev.desc,
                                                  p1_lb, p2_ub, eps_lb, n_confirm)

        cfg = {"n_confirm": n_confirm, "n_pilot": n_pilot, "alpha": alpha,
               "total_tests": total_tests, "seed": seed}
        if best is not None:
            return AuditResult("falsified", claimed_eps, claimed_delta, best,
                               total_tests, time.time() - t0, config=cfg)
        return AuditResult("not_falsified", claimed_eps, claimed_delta, None,
                           total_tests, time.time() - t0, config=cfg)

    except UnsupportedOutput as ex:
        return AuditResult("unsupported", claimed_eps, claimed_delta,
                           detail=str(ex), elapsed_s=time.time() - t0)
    except Exception as ex:  # generated code misbehaving mid-audit
        return AuditResult("error", claimed_eps, claimed_delta,
                           detail=f"{type(ex).__name__}: {ex}", elapsed_s=time.time() - t0)
