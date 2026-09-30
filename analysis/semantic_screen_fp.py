"""How often does the "deterministic release" screen fire on correct code?

`aggregate.py` screens out programs whose output never varies: it draws 12
samples on a fixed input and, if all 12 are identical, records `semantic_fail`.
The intent is to catch mechanisms that forgot the noise entirely.

The screen uses an unseeded generator, and 12 identical draws is not rare when
the output has low entropy. A randomised response that reports truthfully with
probability 0.75 produces 12 identical answers about 3% of the time while being
completely correct. For a three-valued output with one dominant branch the rate
is far higher.

That matters because a false `semantic_fail` removes a program from the audited
set, so it silently changes the denominator of every rate in the paper.

This script measures the false-positive rate directly rather than bounding it.
For each program the campaign screened out, it re-runs the same screen many
times. A program that fails the screen every time really is deterministic. A
program that fails it sometimes was excluded by luck.

    python analysis/semantic_screen_fp.py --repeats 200
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import sys

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sandbox.runner import load_mechanism  # noqa: E402

OUT = ROOT / "results" / "processed"


def adapt(fn, task):
    """Tasks with delta > 0 take (data, epsilon, delta, rng).

    aggregate.py wraps them the same way. Without this the Gaussian programs
    raise on every call here and get reported as un-rerunnable, which would
    understate the coverage of this check.
    """
    delta = float(task.get("claimed_delta") or 0.0)
    if delta > 0:
        return lambda data, eps, rng: fn(data, eps, delta, rng)
    return fn


def screen_once(fn, task, rng, draws=12):
    """One run of the campaign's determinism screen. True == flagged."""
    d1 = task["auditor"]["pairs"][0][0]
    try:
        outs = {repr(fn(list(d1), task["claimed_epsilon"], rng))
                for _ in range(draws)}
    except Exception:                                        # noqa: BLE001
        return None
    return len(outs) == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=200)
    a = ap.parse_args()

    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml")
                            .read_text(encoding="utf-8"))["tasks"]}

    flagged = []
    for f in sorted(glob.glob(str(OUT / "outcomes_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r["outcome"] == "semantic_fail":
                    flagged.append(r)

    print("=" * 74)
    print("DETERMINISM SCREEN: how many exclusions are real?")
    print("=" * 74)
    print(f"\n  {len(flagged)} programs were screened out as semantic_fail.")
    print(f"  Re-running the identical screen {a.repeats} times on each.\n")

    print(f"  {'task':30s} {'model':22s} {'flagged':>9s}  verdict")
    real = lucky = broken = 0
    rows = []
    for r in flagged:
        task = tasks[r["task_id"]]
        try:
            fn = load_mechanism(r["code"], task["function_name"])
        except Exception:                                    # noqa: BLE001
            print(f"  {r['task_id']:30s} {r['model']:22s} {'--':>9s}  will not load")
            broken += 1
            continue
        rng = np.random.default_rng(20260824)
        hits = tot = 0
        for _ in range(a.repeats):
            v = screen_once(fn, task, rng)
            if v is None:
                continue
            tot += 1
            hits += bool(v)
        if tot == 0:
            print(f"  {r['task_id']:30s} {r['model']:22s} {'--':>9s}  raises")
            broken += 1
            continue
        frac = hits / tot
        if frac >= 0.99:
            verdict = "DETERMINISTIC (exclusion correct)"
            real += 1
        elif frac <= 0.5:
            verdict = f"NOT deterministic -- excluded by luck"
            lucky += 1
        else:
            verdict = "borderline"
            lucky += 1
        print(f"  {r['task_id']:30s} {r['model']:22s} {frac:>8.0%}  {verdict}")
        rows.append({"task_id": r["task_id"], "model": r["model"],
                     "code_sha256": r["code_sha256"], "screen_hit_rate": frac})

    print("\n" + "=" * 74)
    n = real + lucky
    if n:
        print(f"  correctly excluded : {real}/{n}")
        print(f"  excluded by luck   : {lucky}/{n}")
        print(f"\n  Each lucky exclusion removed a working program from the")
        print("  audited set, so it shifted the denominator of every rate.")
        print(f"  At {len(flagged)} exclusions out of a 480-program corpus the")
        print("  effect on any headline figure is under one percentage point,")
        print("  but the screen is not sound and the paper says so rather than")
        print("  leaving a reader to assume these were all real.")
    if broken:
        print(f"  could not be re-run: {broken}")

    (OUT / "semantic_screen_fp.json").write_text(
        json.dumps({"repeats": a.repeats, "programs": rows}, indent=2),
        encoding="utf-8")
    print("\n  wrote results/processed/semantic_screen_fp.json")


if __name__ == "__main__":
    main()
