"""E5: does the failure survive a much stronger generator?

Every program in the main study came from a 7B open-weights model on a laptop.
The obvious objection is that this is a finding about small models rather than
about code generation, and that a capable model would simply get these tasks
right. E5 answers it by running the identical 16 tasks through a hosted frontier
model and auditing with the same auditor, the same published pairs and the same
thresholds.

Reported separately from the local models, always. The two populations differ in
model family, in serving stack and in sampling defaults, so a pooled "generated
code fails X% of the time" figure would be an average over an arbitrary model
mix and would mean nothing. Adjacent columns, never one number.

The frontier model is also NOT a fourth point in the pre-specified pairwise model
comparison. Those tests were fixed over the three local models before the data
existed; adding a model afterwards and re-running them is the practice the error
analysis rules out.

    python analysis/e5_frontier.py
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "processed"
NOT_EXECUTABLE = ("syntax_fail", "runtime_fail", "api_error", "timeout")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(path):
    rows, seen = [], set()
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("repair_arm") or r.get("parent_sha256"):
            continue                     # repaired programs are not the corpus
        k = (r["model"], r["task_id"], r["sample_idx"])
        if k in seen:
            continue
        seen.add(k)
        rows.append(r)
    return rows


def waterfall(rows, label):
    c = collections.Counter(r["outcome"] for r in rows)
    gen = len(rows)
    ex = gen - sum(c[o] for o in NOT_EXECUTABLE)
    fn = ex - c["semantic_fail"]
    aud = c["falsified"] + c["not_falsified"]
    fals = c["falsified"]

    print(f"\n  {label}")
    print(f"    generated                {gen:4d}")
    print(f"    executable               {ex:4d}   ({ex / gen:.0%} of generated)")
    print(f"    functionally correct     {fn:4d}")
    print(f"    auditable                {aud:4d}")
    print(f"    FALSIFIED                {fals:4d}")
    print()
    for den, name in ((gen, "generated"), (ex, "executable"),
                      (fn, "functional"), (aud, "audited")):
        if den:
            lo, hi = wilson(fals, den)
            print(f"      per {name:11s} {fals}/{den} = {fals / den:5.0%}  "
                  f"95% CI [{lo:.2f}, {hi:.2f}]")
    return {"generated": gen, "executable": ex, "functional": fn,
            "audited": aud, "falsified": fals, "outcomes": dict(c)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier", default=str(OUT / "e5_frontier_outcomes.jsonl"))
    ap.add_argument("--local-glob", default=str(OUT / "outcomes_*.jsonl"))
    a = ap.parse_args()

    fpath = pathlib.Path(a.frontier)
    if not fpath.exists():
        sys.exit(f"{fpath} not found; run the E5 audit first")
    rows = load(fpath)

    models = {r["model"] for r in rows}
    if len(models) != 1:
        sys.exit(f"expected exactly one frontier model, found {sorted(models)}")
    model = models.pop()

    tasks = [t["id"] for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml")
                            .read_text(encoding="utf-8"))["tasks"]]

    print("=" * 74)
    print("E5  A FRONTIER MODEL ON THE SAME BENCHMARK")
    print("=" * 74)
    print(f"\n  model: {model}")

    per_task = collections.Counter(r["task_id"] for r in rows)
    empty = [t for t in tasks if per_task[t] == 0]
    if empty:
        sys.exit(f"REFUSING: {len(empty)} task(s) have no programs "
                 f"({', '.join(empty[:4])}...). A pooled rate over a partial "
                 f"corpus describes the tasks the run reached, not the model. "
                 f"See analysis/coverage.py.")
    counts = sorted(set(per_task[t] for t in tasks))
    print(f"  coverage: all {len(tasks)} tasks, "
          f"{counts[0] if len(counts) == 1 else f'{counts[0]}-{counts[-1]}'} "
          f"samples each")

    stats = waterfall(rows, f"{model} (E5)")

    # ---- per task --------------------------------------------------------
    print("\n  " + "-" * 70)
    print(f"  {'task':32s} {'n':>3s} {'falsified':>10s}")
    for t in tasks:
        tr = [r for r in rows if r["task_id"] == t]
        f = sum(1 for r in tr if r["outcome"] == "falsified")
        print(f"  {t:32s} {len(tr):3d} {f:10d}")

    # ---- side by side, explicitly not pooled -----------------------------
    import glob as _glob
    local = []
    for p in sorted(_glob.glob(a.local_glob)):
        local.extend(load(p))
    if local:
        print("\n  " + "-" * 70)
        print("  Side by side with the local models. These are ADJACENT columns,")
        print("  never summed: different model family, serving stack and sampling")
        print("  defaults. A pooled rate here would be an average over an")
        print("  arbitrary model mix.")
        lstats = waterfall(local, f"local models pooled (n={len({r['model'] for r in local})})")

        print("\n  " + "-" * 70)
        fa = stats["falsified"] / stats["audited"] if stats["audited"] else float("nan")
        la = lstats["falsified"] / lstats["audited"] if lstats["audited"] else float("nan")
        print(f"  falsified per audited: frontier {fa:.0%}   local {la:.0%}")
        print("\n  No significance test is reported for this difference and none")
        print("  should be. The pairwise model comparisons were pre-specified over")
        print("  the three local models before any data existed. Adding a fourth")
        print("  model afterwards and testing it against them is exactly the")
        print("  practice the error-control section rules out.")

    # ---- what a zero can and cannot mean --------------------------------
    if stats["falsified"] == 0 and stats["audited"]:
        # The auditor is not omniscient. Calibration against planted bugs puts
        # its per-program detection power at roughly 0.65-0.76. Zero detections
        # therefore does not mean zero violations; it bounds them. P(catch none
        # | V violations) = (1 - power)^V, so we can say which V are ruled out.
        print("\n  " + "-" * 70)
        print("  WHAT ZERO DETECTIONS MEANS")
        print(f"    {stats['audited']} programs audited, none falsified.")
        print("\n    This is 'no violation found', not 'correct'. The auditor's")
        print("    measured detection power on planted bugs is 0.65-0.76 per")
        print("    program, so a zero bounds the true violation count rather")
        print("    than establishing it:")
        print(f"\n      {'true violations':>16s}  {'P(we would see none)':>22s}")
        for power in (0.65, 0.76):
            print(f"      at power {power:.2f}:")
            for v in (1, 2, 3, 5, 10):
                print(f"      {v:>16d}  {(1 - power) ** v:>22.4f}")
            break
        print("\n    So a handful of undetected violations is entirely")
        print("    consistent with this result. What it rules out decisively is")
        print("    anything close to the local models' rate: at 61% of audited,")
        print(f"    roughly {round(0.61 * stats['audited'])} of these programs would")
        print("    violate, and the probability of catching none of those is")
        print("    indistinguishable from zero.")
        print("\n    The auditor is also weakest exactly where composition is")
        print("    involved, which is where it missed planted bugs. Frontier")
        print("    programs on those tasks carry the most residual uncertainty.")

    (OUT / "e5_frontier_summary.json").write_text(
        json.dumps({"model": model, **stats}, indent=2), encoding="utf-8")
    print(f"\n  wrote results/processed/e5_frontier_summary.json")


if __name__ == "__main__":
    main()
