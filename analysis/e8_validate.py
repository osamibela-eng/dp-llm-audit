"""Validate the E8 perturbations BEFORE spending anything on generation.

E8 asks whether a frontier model's clean sweep is competence or recall. It
answers that by perturbing a DP-relevant constant so the canonical answer is
numerically wrong. That design has two ways to be silently broken, and both
would produce a publishable-looking result that means nothing:

  1. The perturbed task is ill-posed, so even a CORRECT implementation fails.
     Then a model failure is our fault, not the model's.
  2. The perturbation runs in the wrong direction, so a memorised answer is
     over-noised rather than under-noised. It stays private, the audit finds
     nothing, and the task discriminates nothing while appearing to work.

This script checks both, mechanically:

  REFERENCE  every reference must audit NOT FALSIFIED.
  MEMORISED  a deliberately memorised answer (the unperturbed constant) must
             audit FALSIFIED on every A2 task.

If either check fails, the affected task is not fit to use.

    python analysis/e8_validate.py
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auditors.simple_auditor import falsify  # noqa: E402
from sandbox.runner import load_mechanism  # noqa: E402

# The memorised answer for each A2 task: the canonical, unperturbed
# implementation a model would produce from recall. Each must be FALSIFIED, or
# the perturbation runs in the wrong direction and discriminates nothing.
# Keys must match the ids in tasks_e8.yaml exactly; a typo here shows up as
# "no memorised variant defined" rather than as a silent pass.
MEMORISED_SRC = {
    # sum over {0..5} but noised as if sensitivity were 1
    "e8_a2_count": '''
import numpy as np
def private_total(data, epsilon, rng):
    total = float(sum(data)) if data else 0.0
    return float(total + rng.laplace(loc=0.0, scale=1.0 / epsilon))
''',
    # clips to 30 correctly but uses the memorised sensitivity 10
    "e8_a2_sum": '''
import numpy as np
def private_sum_capped(data, epsilon, rng):
    clipped = [min(max(float(x), 0.0), 30.0) for x in data]
    total = float(sum(clipped)) if clipped else 0.0
    return float(total + rng.laplace(loc=0.0, scale=10.0 / epsilon))
''',
    # mean over [0,4] noised as if the range were [0,1]
    "e8_a2_mean": '''
import numpy as np
def private_average(data, epsilon, rng):
    n = len(data)
    mean = float(sum(float(x) for x in data)) / n
    return float(mean + rng.laplace(loc=0.0, scale=1.0 / (n * epsilon)))
''',
    # the canonical truthful-with-probability-3/4 constant
    "e8_a2_rr": '''
import math
def private_bit_report(data, epsilon, rng):
    bit = int(data[0])
    p = 0.75
    return int(bit) if rng.random() < p else int(1 - bit)
''',
}


def audit_source(src, task, n_confirm, seed):
    try:
        fn = load_mechanism(src, task["function_name"])
    except Exception as e:                                   # noqa: BLE001
        return "load_fail", f"{type(e).__name__}: {e}"
    try:
        res = falsify(fn, [tuple(p) for p in task["auditor"]["pairs"]],
                      float(task["claimed_epsilon"]),
                      float(task.get("claimed_delta") or 0.0),
                      n_confirm=n_confirm, seed=seed)
    except Exception as e:                                   # noqa: BLE001
        return "audit_error", f"{type(e).__name__}: {e}"
    eps_lb = None
    if res.counterexample:
        eps_lb = res.counterexample.eps_lower_bound
    return res.status, eps_lb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-confirm", type=int, default=30_000)
    a = ap.parse_args()

    tasks = yaml.safe_load((ROOT / "benchmark" / "tasks_e8.yaml")
                           .read_text(encoding="utf-8"))["tasks"]

    # Catch key typos here rather than letting them surface as a task that
    # silently has no direction check.
    a2_ids = {t["id"] for t in tasks if t["arm"] == "A2"}
    stray = set(MEMORISED_SRC) - a2_ids
    if stray:
        sys.exit(f"MEMORISED_SRC has keys matching no A2 task: {sorted(stray)}")

    print("=" * 76)
    print("E8 VALIDATION  (must pass before any generation is paid for)")
    print("=" * 76)

    bad = []

    print("\n--- CHECK 1: every reference must survive its own audit ---\n")
    for t in tasks:
        src = (ROOT / t["reference"]).read_text(encoding="utf-8")
        status, extra = audit_source(src, t, a.n_confirm, seed=880_001)
        ok = status == "not_falsified"
        mark = "ok " if ok else "FAIL"
        print(f"  [{mark}] {t['id']:16s} {t['arm']}  {status}"
              + (f"   {extra}" if extra else ""))
        if not ok:
            bad.append((t["id"], f"reference audited {status}"))

    print("\n--- CHECK 2: the memorised answer must be CAUGHT on every A2 task ---\n")
    for t in tasks:
        if t["arm"] != "A2":
            continue
        src = MEMORISED_SRC.get(t["id"])
        if src is None:
            print(f"  [SKIP] {t['id']:16s} no memorised variant defined")
            bad.append((t["id"], "no memorised variant to validate direction"))
            continue
        status, extra = audit_source(src, t, a.n_confirm, seed=880_002)
        ok = status == "falsified"
        mark = "ok " if ok else "FAIL"
        detail = f"   empirical eps lb {extra:.2f}" if isinstance(extra, float) else ""
        print(f"  [{mark}] {t['id']:16s} memorised answer -> {status}{detail}")
        if not ok:
            bad.append((t["id"],
                        f"memorised answer audited {status}; the perturbation "
                        f"does not discriminate"))

    print("\n" + "=" * 76)
    if bad:
        print(f"  {len(bad)} PROBLEM(S). These tasks are not fit to use:\n")
        for tid, why in bad:
            print(f"    {tid:16s} {why}")
        print("\n  Fix or drop them before generating. A task that cannot catch")
        print("  the memorised answer contributes nothing but noise, and a task")
        print("  whose reference fails would blame the model for our error.")
        sys.exit(1)

    print("  All references audit clean, and every A2 perturbation catches the")
    print("  memorised answer. The arm is fit to run.")


if __name__ == "__main__":
    main()
