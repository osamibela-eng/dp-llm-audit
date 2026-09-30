"""Outcome waterfall and falsification rates with explicit denominators.

Review feedback (2026-08-21) was blunt about this: "over half of auditable code is
falsified" is accurate but insufficient, because a reviewer will immediately ask
whether the falsified programs were already semantically broken. Every rate here
therefore carries the denominator it was computed over, and the waterfall is
printed so the denominators cannot be confused.

    generated -> executable -> functionally passing -> audited -> falsified

Wilson score intervals rather than normal approximation: at n = 50 and p near 0
or 1 the normal interval runs outside [0,1] and understates uncertainty.

    python analysis/report_rates.py
    python analysis/report_rates.py --exclude-zero-coverage --min-power 0.5
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Outcome vocabulary, partitioned by the stage at which each program stopped.
NOT_EXECUTABLE = {"syntax_fail", "screen_fail", "runtime_fail", "timeout"}
NOT_AUDITED = {"auditor_unsupported", "auditor_error"}
FUNCTIONAL_FAIL = {"semantic_fail"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def fmt(k: int, n: int) -> str:
    if n == 0:
        return "     n/a      "
    lo, hi = wilson(k, n)
    return f"{k:3d}/{n:<3d} {k/n:5.0%} [{lo:.2f},{hi:.2f}]"


def detection_power(cal_rows) -> dict:
    """caught / (variants predicted detectable), per task."""
    power = {}
    for c in cal_rows:
        if c.get("program") == "REFERENCE":
            continue
        t = c["task"]
        power.setdefault(t, [0, 0])
        if c.get("expected") == "detectable":
            power[t][1] += 1
            if c.get("status") == "falsified" or c.get("eps_lb"):
                power[t][0] += 1
    return {t: (a / b if b else None) for t, (a, b) in power.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(ROOT / "results" / "processed" / "outcomes_*.jsonl"))
    ap.add_argument("--exclude-zero-coverage", action="store_true",
                    help="drop tasks with no audited programs (e.g. private_topk)")
    ap.add_argument("--min-power", type=float, default=None,
                    help="restrict to tasks whose calibrated detection power is >= this")
    a = ap.parse_args()

    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml").read_text(encoding="utf-8"))["tasks"]}
    cal = [json.loads(l) for l in
           (ROOT / "results" / "calibration.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    power = detection_power(cal)

    rows = []
    for f in sorted(glob.glob(a.glob)):
        rows += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    if not rows:
        raise SystemExit(f"no outcome rows matched {a.glob}")

    # ---- filters, applied before any rate is computed -------------------
    dropped = []
    if a.exclude_zero_coverage:
        audited_by_task = collections.Counter(
            r["task_id"] for r in rows if r["outcome"] in ("falsified", "not_falsified"))
        zero = {t for t in tasks if audited_by_task[t] == 0}
        if zero:
            dropped.append(f"zero audit coverage: {sorted(zero)}")
            rows = [r for r in rows if r["task_id"] not in zero]
    if a.min_power is not None:
        weak = {t for t in tasks if (power.get(t) is None or power[t] < a.min_power)}
        if weak:
            dropped.append(f"detection power < {a.min_power}: {sorted(weak)}")
            rows = [r for r in rows if r["task_id"] not in weak]

    models = sorted({r["model"] for r in rows})

    # ---- waterfall ------------------------------------------------------
    print("=" * 78)
    print("OUTCOME WATERFALL  (every rate below names its denominator)")
    print("=" * 78)
    for m in models + ["ALL MODELS"]:
        sub = rows if m == "ALL MODELS" else [r for r in rows if r["model"] == m]
        c = collections.Counter(r["outcome"] for r in sub)
        gen = len(sub)
        executable = gen - sum(c[o] for o in NOT_EXECUTABLE)
        func_ok = executable - sum(c[o] for o in FUNCTIONAL_FAIL)
        audited = c["falsified"] + c["not_falsified"]
        fals = c["falsified"]
        print(f"\n{m}")
        print(f"  generated              {gen:4d}")
        print(f"  -> executable          {executable:4d}   ({executable/gen:.0%} of generated)")
        print(f"  -> functionally passing{func_ok:4d}   ({func_ok/gen:.0%} of generated)")
        print(f"  -> audited             {audited:4d}   ({audited/gen:.0%} of generated)")
        print(f"  -> FALSIFIED           {fals:4d}")
        print(f"     falsified / generated            {fmt(fals, gen)}")
        print(f"     falsified / executable           {fmt(fals, executable)}")
        print(f"     falsified / functionally passing {fmt(fals, func_ok)}")
        print(f"     falsified / audited              {fmt(fals, audited)}")
        if c:
            print("     unaudited: " + ", ".join(
                f"{o}={c[o]}" for o in sorted(NOT_AUDITED | NOT_EXECUTABLE | FUNCTIONAL_FAIL) if c[o]))

    # ---- per tier -------------------------------------------------------
    print("\n" + "=" * 78)
    print("BY TIER   (falsified / audited, Wilson 95%)")
    print("=" * 78)
    for tier in (1, 2, 3):
        sub = [r for r in rows if tasks[r["task_id"]]["tier"] == tier]
        f = sum(1 for r in sub if r["outcome"] == "falsified")
        n = f + sum(1 for r in sub if r["outcome"] == "not_falsified")
        print(f"  tier {tier}:  {fmt(f, n)}")

    # ---- per task -------------------------------------------------------
    print("\n" + "=" * 78)
    print("BY TASK   (falsified / audited, Wilson 95%)   power = calibrated detection")
    print("=" * 78)
    print(f"  {'task':32s} {'tier':4s} {'power':>6s}  falsified / audited")
    for t in sorted(tasks, key=lambda x: (tasks[x]["tier"], x)):
        sub = [r for r in rows if r["task_id"] == t]
        if not sub:
            continue
        f = sum(1 for r in sub if r["outcome"] == "falsified")
        n = f + sum(1 for r in sub if r["outcome"] == "not_falsified")
        pw = power.get(t)
        pws = f"{pw:.2f}" if pw is not None else "  n/a"
        print(f"  {t:32s} {tasks[t]['tier']:<4} {pws:>6s}  {fmt(f, n)}")

    if dropped:
        print("\nEXCLUDED FROM THE ABOVE:")
        for d in dropped:
            print(f"  - {d}")

    print("\nNote: 'not_falsified' is not evidence of correctness. It means the program")
    print("survived this audit budget over these neighbouring pairs and output events.")


if __name__ == "__main__":
    main()
