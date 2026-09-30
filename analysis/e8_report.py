"""E8: competence or recall?

The audit result alone cannot answer this. "Not falsified" means the program
satisfies its claim, and a program that adds far too much noise satisfies its
claim comfortably while computing the wrong query. So a clean sweep on the
perturbed tasks is consistent with two very different behaviours:

  the model recomputed the sensitivity from the new spec        (competence)
  the model reproduced a memorised constant and got lucky       (recall)

This script separates them by MEASURING what each generated program actually
does, rather than inferring it from a null result.

For each A2 program we estimate the effective noise scale empirically: run the
mechanism many times on a fixed input, take the spread of the output, and invert
the known noise family to recover the scale the program used. Then compare it
against two reference points fixed in advance:

  correct    the scale the perturbed spec requires
  memorised  the scale the ORIGINAL, unperturbed spec required

A program sitting at `correct` tracked the change. A program sitting at
`memorised` reproduced the textbook answer. Anything far above `correct` is
over-noised: privacy-safe, but not evidence of understanding.

    python analysis/e8_report.py
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import statistics
import sys

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sandbox.runner import load_mechanism  # noqa: E402

OUT = ROOT / "results" / "processed"

# For each A2 task: a probe dataset, the scale the perturbed spec requires, and
# the scale a memorised (unperturbed) answer would use. Both at epsilon = the
# task's claimed epsilon. Laplace has std = scale * sqrt(2).
PROBES = {
    "e8_a2_count": {
        "data": [5, 5, 5, 5, 5],
        "correct": lambda eps, n: 5.0 / eps,
        "memorised": lambda eps, n: 1.0 / eps,
        "kind": "laplace",
    },
    "e8_a2_sum": {
        "data": [30.0, 30.0, 30.0],
        "correct": lambda eps, n: 30.0 / eps,
        "memorised": lambda eps, n: 10.0 / eps,
        "kind": "laplace",
    },
    "e8_a2_mean": {
        "data": [4.0, 0.0, 4.0, 0.0],
        "correct": lambda eps, n: 4.0 / (n * eps),
        "memorised": lambda eps, n: 1.0 / (n * eps),
        "kind": "laplace",
    },
    "e8_a2_rr": {
        "data": [1],
        "correct": lambda eps, n: math.exp(eps) / (1.0 + math.exp(eps)),
        "memorised": lambda eps, n: 0.75,
        "kind": "bernoulli",
    },
}


def measure(fn, task, spec, draws=4000):
    """Empirically recover the constant the program uses."""
    rng = np.random.default_rng(4242)
    data = spec["data"]
    eps = float(task["claimed_epsilon"])
    outs = []
    for _ in range(draws):
        try:
            outs.append(fn(list(data), eps, rng))
        except Exception:                                    # noqa: BLE001
            return None
    if spec["kind"] == "laplace":
        vals = [float(o) for o in outs]
        # Laplace(scale) has standard deviation scale * sqrt(2)
        return statistics.pstdev(vals) / math.sqrt(2.0)
    # bernoulli: fraction reporting the true bit
    true_bit = int(data[0])
    return sum(1 for o in outs if int(o) == true_bit) / len(outs)


def classify(observed, correct, memorised, kind):
    """Which reference point is the program closer to, in log space?"""
    if observed is None:
        return "unmeasurable"
    if kind == "bernoulli":
        # both are probabilities; compare directly
        d_cor = abs(observed - correct)
        d_mem = abs(observed - memorised)
        if d_cor <= 0.03:
            return "correct"
        if d_mem <= 0.03:
            return "memorised"
        return "over_noised" if observed < correct else "other"
    if observed <= 0:
        return "unmeasurable"
    r_cor = observed / correct
    r_mem = observed / memorised
    if 0.75 <= r_cor <= 1.33:
        return "correct"
    if 0.75 <= r_mem <= 1.33:
        return "memorised"
    return "over_noised" if r_cor > 1.33 else "under_noised"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes", default=str(OUT / "e8_outcomes.jsonl"))
    ap.add_argument("--draws", type=int, default=4000)
    a = ap.parse_args()

    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks_e8.yaml")
                            .read_text(encoding="utf-8"))["tasks"]}
    rows = [json.loads(l) for l in
            pathlib.Path(a.outcomes).read_text(encoding="utf-8").splitlines()
            if l.strip()]

    print("=" * 76)
    print("E8  IS THE FRONTIER MODEL'S CLEAN SWEEP COMPETENCE OR RECALL?")
    print("=" * 76)

    # ---- part 1: audit outcomes by arm -----------------------------------
    by_arm = collections.defaultdict(collections.Counter)
    for r in rows:
        arm = tasks[r["task_id"]]["arm"]
        by_arm[arm][r["outcome"]] += 1

    print("\n  audit outcomes by arm")
    print(f"    {'arm':6s} {'n':>4s} {'falsified':>10s} {'not falsified':>14s} "
          f"{'did not run':>12s}")
    for arm in ("A1", "A2"):
        c = by_arm[arm]
        n = sum(c.values())
        broke = n - c["falsified"] - c["not_falsified"]
        print(f"    {arm:6s} {n:4d} {c['falsified']:10d} {c['not_falsified']:14d} "
              f"{broke:12d}")

    print("\n  A1 changes only names, so a failure there would mean the model is")
    print("  sensitive to surface form. A2 changes a DP constant, so a failure")
    print("  there would mean the model reproduced the textbook answer.")

    # ---- part 2: what constant did each A2 program actually use? ---------
    print("\n  " + "-" * 72)
    print("  MEASURED behaviour of the A2 programs")
    print("  (a null audit is not enough: an over-noised program passes too)")

    verdicts = collections.Counter()
    per_task = collections.defaultdict(collections.Counter)
    per_model = collections.defaultdict(collections.Counter)
    for r in rows:
        tid = r["task_id"]
        spec = PROBES.get(tid)
        if not spec:
            continue
        task = tasks[tid]
        try:
            fn = load_mechanism(r["code"], task["function_name"])
            eps = float(task["claimed_epsilon"])
            n = len(spec["data"])
            obs = measure(fn, task, spec, draws=a.draws)
            v = classify(obs, spec["correct"](eps, n), spec["memorised"](eps, n),
                         spec["kind"])
        except Exception:                                    # noqa: BLE001
            # Local 7B programs often do not run at all; a crash during
            # measurement is recorded, not allowed to abort the report.
            v = "unmeasurable"
        verdicts[v] += 1
        per_task[tid][v] += 1
        per_model[r.get("model", "?")][v] += 1

    print(f"\n    {'task':16s} {'correct':>8s} {'memorised':>10s} "
          f"{'over-noised':>12s} {'other':>7s}")
    for tid in sorted(per_task):
        c = per_task[tid]
        other = sum(v for k, v in c.items()
                    if k not in ("correct", "memorised", "over_noised"))
        print(f"    {tid:16s} {c['correct']:8d} {c['memorised']:10d} "
              f"{c['over_noised']:12d} {other:7d}")

    tot = sum(verdicts.values())
    print(f"\n    total {tot}: " + "  ".join(f"{k}={v}" for k, v in
                                             verdicts.most_common()))

    # ---- verdict ---------------------------------------------------------
    print("\n" + "=" * 76)
    mem = verdicts["memorised"]
    cor = verdicts["correct"]
    if tot == 0:
        print("  nothing measurable")
    elif mem == 0 and cor > 0:
        print(f"  No A2 program used the memorised constant. {cor}/{tot} used the")
        print("  constant the perturbed spec requires.")
        print("\n  On these four families the model recomputed sensitivity from the")
        print("  specification rather than reproducing a remembered implementation.")
        print("  That is evidence against the memorisation explanation for E5, and")
        print("  it is measured behaviour rather than an inference from a null.")
    elif mem > 0:
        print(f"  {mem}/{tot} A2 programs used the MEMORISED constant.")
        print("  Where those programs were not falsified, the audit lacked the")
        print("  power to see it; where they were, the perturbation did its job.")
    print("\n  Scope: four task families, one model. The composition and Gaussian")
    print("  families are excluded because the auditor cannot detect the relevant")
    print("  error there at all, so nothing could have been concluded from them.")

    # The summary is named after the outcomes file, so running this on a second
    # corpus (e8b_outcomes.jsonl, the local models) cannot overwrite the
    # original e8_summary.json the paper already reports.
    stem = pathlib.Path(a.outcomes).name.replace("_outcomes.jsonl", "")
    summary = OUT / f"{stem}_summary.json"
    summary.write_text(json.dumps({
        "by_arm": {k: dict(v) for k, v in by_arm.items()},
        "measured": dict(verdicts),
        "per_task": {k: dict(v) for k, v in per_task.items()},
        "per_model": {k: dict(v) for k, v in per_model.items()},
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote {summary}")


if __name__ == "__main__":
    main()
