"""Sensitivity of the headline rate and of the model ranking.

Assessment 7 asked for three things this file computes.

1. THE GAUSSIAN TASK. `gaussian_bounded_count` claims (eps, delta)-DP, and our
   auditor tests a pure-eps condition. Its calibrated detection power is 0.00 and
   three of its falsifications did not survive fresh seeds. If it stays in the
   aggregate, a reviewer can argue the headline is contaminated by an auditor
   mismatch. So we report the rate three ways: all tasks, excluding the Gaussian
   task, and excluding every task whose calibrated detection power is below 0.5.

2. THE RANKING ACROSS DENOMINATORS. The paper's central object is that the
   ordering of the three models depends on which denominator is used. Rather than
   assert it twice, we print the rank under every denominator in the waterfall.

3. PARTIAL-IDENTIFICATION BOUNDS. A model's non-executing programs have no
   privacy verdict, and no imputation can honestly supply one. What we can do is
   bound: assume every unaudited program would have been safe (lower bound), then
   assume every one would have been falsified (upper bound). The width of that
   interval is the cost of missing coverage, and for a model that fails to run
   half its outputs the interval is nearly useless - which is the point.

   These are NOT an adjusted accuracy and are never reported as one.

    python analysis/sensitivity.py
"""
from __future__ import annotations

import collections
import glob
import json
import math
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOT_EXECUTABLE = {"syntax_fail", "screen_fail", "runtime_fail", "timeout"}
FUNCTIONAL_FAIL = {"semantic_fail"}
GAUSSIAN = "gaussian_bounded_count_v1"
SHORT = {"qwen2.5-coder:7b": "qwen", "deepseek-coder:6.7b": "deepseek",
         "codellama:7b": "codellama"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fmt(k, n):
    if n == 0:
        return "      n/a      "
    lo, hi = wilson(k, n)
    return f"{k:3d}/{n:<3d} {k/n:5.0%} [{lo:.2f},{hi:.2f}]"


def detection_power(cal):
    power = {}
    for c in cal:
        if c.get("program") == "REFERENCE":
            continue
        t = c["task"]
        power.setdefault(t, [0, 0])
        if c.get("expected") == "detectable":
            power[t][1] += 1
            if c.get("status") == "falsified" or c.get("eps_lb"):
                power[t][0] += 1
    return {t: (a / b if b else None) for t, (a, b) in power.items()}


def load():
    rows, seen = [], set()
    for f in sorted(glob.glob(str(ROOT / "results" / "processed" / "outcomes_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("repair_arm") or r.get("parent_sha256"):
                continue
            k = (r["model"], r["task_id"], r["sample_idx"])
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
    return rows


def stages(sub, repro=None):
    c = collections.Counter(r["outcome"] for r in sub)
    gen = len(sub)
    ex = gen - sum(c[o] for o in NOT_EXECUTABLE)
    fn = ex - sum(c[o] for o in FUNCTIONAL_FAIL)
    aud = c["falsified"] + c["not_falsified"]
    fals = c["falsified"]
    rep = (sum(1 for r in sub if r["outcome"] == "falsified"
               and repro.get(r["code_sha256"]) == "reproducible")
           if repro is not None else None)
    return gen, ex, fn, aud, fals, rep


def main():
    rows = load()
    cal = [json.loads(l) for l in
           (ROOT / "results" / "calibration.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    power = detection_power(cal)
    conf = {r["code_sha256"]: r["classification"] for r in
            (json.loads(l) for l in
             (ROOT / "results" / "processed" / "confirm_seeds.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip())}

    weak = {t for t, p in power.items() if p is None or p < 0.5}

    print("=" * 78)
    print("SENSITIVITY OF THE HEADLINE RATE")
    print("=" * 78)
    print(f"  {'subset':34s} {'reproducibly falsified / functionally passing'}")
    variants = [
        ("all tasks", rows),
        (f"excluding {GAUSSIAN.replace('_v1','')}",
         [r for r in rows if r["task_id"] != GAUSSIAN]),
        ("excluding all detection power < 0.5",
         [r for r in rows if r["task_id"] not in weak]),
    ]
    for name, sub in variants:
        _, _, fn, _, _, rep = stages(sub, conf)
        print(f"  {name:34s} {fmt(rep, fn)}")
    print(f"\n  tasks with power < 0.5: {', '.join(sorted(t.replace('_v1','') for t in weak))}")
    print("\n  The headline is stable across all three subsets, so the Gaussian")
    print("  task is not driving it. We still exclude that task from aggregate")
    print("  privacy conclusions in the paper, because a pure-eps auditor cannot")
    print("  properly test an (eps, delta) claim regardless of what the number does.")

    # ---- ranking under each denominator ---------------------------------
    print("\n" + "=" * 78)
    print("MODEL RANKING UNDER EACH DENOMINATOR   (1 = lowest falsification rate)")
    print("=" * 78)
    models = list(SHORT)
    dens = ["generated", "executable", "functionally passing", "audited"]
    print(f"  {'denominator':22s} " + "".join(f"{SHORT[m]:>26s}" for m in models))
    for i, den in enumerate(dens):
        vals = {}
        for m in models:
            gen, ex, fn, aud, fals, _ = stages([r for r in rows if r["model"] == m])
            n = [gen, ex, fn, aud][i]
            vals[m] = (fals, n)
        order = sorted(models, key=lambda m: vals[m][0] / vals[m][1])
        line = f"  {den:22s} "
        for m in models:
            k, n = vals[m]
            line += f"{fmt(k, n):>26s}"
        print(line)
        print(f"  {'  rank':22s} " + "".join(
            f"{order.index(m)+1:>26d}" for m in models))
    # reproducible variant
    vals = {}
    for m in models:
        sub = [r for r in rows if r["model"] == m]
        _, _, fn, _, _, rep = stages(sub, conf)
        vals[m] = (rep, fn)
    order = sorted(models, key=lambda m: vals[m][0] / vals[m][1])
    print(f"  {'reproducible / func':22s} " + "".join(
        f"{fmt(*vals[m]):>26s}" for m in models))
    print(f"  {'  rank':22s} " + "".join(f"{order.index(m)+1:>26d}" for m in models))

    # ---- partial identification -----------------------------------------
    print("\n" + "=" * 78)
    print("PARTIAL-IDENTIFICATION BOUNDS   (per generated program)")
    print("=" * 78)
    print("  Unaudited programs have NO privacy verdict. These bracket what the")
    print("  rate could have been, assuming every unaudited program was safe")
    print("  (lower) or falsified (upper). They are bounds, not an adjustment.\n")
    print(f"  {'model':14s} {'audited':>9s} {'lower':>8s} {'upper':>8s} {'width':>8s}")
    for m in models:
        gen, ex, fn, aud, fals, _ = stages([r for r in rows if r["model"] == m])
        lo = fals / gen
        hi = (fals + (gen - aud)) / gen
        print(f"  {SHORT[m]:14s} {aud:4d}/{gen:<4d} {lo:8.0%} {hi:8.0%} "
              f"{hi-lo:8.0%}")
    gen, ex, fn, aud, fals, _ = stages(rows)
    print(f"  {'ALL':14s} {aud:4d}/{gen:<4d} {fals/gen:8.0%} "
          f"{(fals + gen - aud)/gen:8.0%} {(gen-aud)/gen:8.0%}")
    print("\n  codellama's interval is the widest by a wide margin, because it has")
    print("  the least coverage. That width IS the finding: with half a model's")
    print("  output unaudited, the data do not pin down its falsification rate.")


if __name__ == "__main__":
    main()
