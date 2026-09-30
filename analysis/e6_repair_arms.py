"""E6: does richer repair feedback change the outcome?

v1 reported that handing a model its own counterexample (R2) did not repair the
program more often than simply asking it to try again (R0). The obvious
objection is that a counterexample on its own is thin feedback, and that a
model given an interpretation of the counterexample would have done better.

R3 tests exactly that. It is R2 plus a diagnosis, and the diagnosis is
generated from three automatable signals rather than written by hand. See
repair/diagnose.py; those templates are the entire vocabulary. If we had
hand-written per-task hints the arm would prove nothing, because the hints
would carry our understanding of the bug rather than the tool's.

All three arms run on the same pre-registered 51 programs, so every comparison
is paired and the test is exact McNemar on the discordant pairs.

A note on what "repaired" means here. A repaired program has to survive the
audit, so runtime failures and syntax failures count against the arm. That is
the honest accounting: a repair that does not run is not a repair.

    python analysis/e6_repair_arms.py
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auditors.contract import load_contracts  # noqa: E402
from repair.diagnose import signals_fired  # noqa: E402

OUT = ROOT / "results" / "processed"
ARMS = ("R0", "R2", "R3")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on the discordant counts.

    Under the null the b discordant pairs favouring one arm are Binomial(b+c,
    1/2). Exact rather than chi-square because the discordant counts here are
    small enough that the asymptotic test would be optimistic.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load_repair(path) -> dict:
    """(arm, parent_sha256) -> outcome row, deduplicated by newest."""
    by = {}
    for f in sorted(glob.glob(str(path))):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            arm, parent = r.get("repair_arm"), r.get("parent_sha256")
            if not arm or not parent:
                continue
            by[(arm, parent)] = r
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repair", default=str(OUT / "repair_*outcomes*.jsonl"))
    a = ap.parse_args()

    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml")
                            .read_text(encoding="utf-8"))["tasks"]}
    contracts = load_contracts()

    sub_rows = [json.loads(l) for l in
                (OUT / "repair_subset.jsonl").read_text(encoding="utf-8").splitlines()
                if l.strip()]
    header = next((r for r in sub_rows if r.get("_header")), {})
    subset = [r for r in sub_rows if not r.get("_header")]
    parents = [r["code_sha256"] for r in subset]
    parent_row = {r["code_sha256"]: r for r in subset}

    by = load_repair(a.repair)
    present = {arm: sum(1 for p in parents if (arm, p) in by) for arm in ARMS}

    print("=" * 78)
    print("E6  REPAIR UNDER THREE FEEDBACK CONDITIONS")
    print("=" * 78)
    if header.get("rule"):
        print(f"\n  pre-registered selection rule: {header['rule']}")
    print(f"  programs in the subset: {len(parents)}")
    print("  coverage: " + "  ".join(f"{k} {v}/{len(parents)}"
                                     for k, v in present.items()))

    missing = [arm for arm, n in present.items() if n < len(parents)]
    if missing:
        print(f"\n  INCOMPLETE: {', '.join(missing)} has not finished. "
              f"Comparisons below are restricted to programs all arms cover.")

    # Restrict to the programs every arm covers, or the pairing is broken.
    paired = [p for p in parents if all((arm, p) in by for arm in ARMS)]
    if not paired:
        print("\n  nothing to compare yet.")
        return

    def repaired(arm, p):
        return by[(arm, p)]["outcome"] == "not_falsified"

    print(f"\n  paired programs: {len(paired)}")
    print("\n  " + "-" * 74)
    print(f"  {'arm':6s} {'repaired':>10s} {'rate':>8s}  {'95% CI':>16s}   "
          f"{'still falsified':>15s} {'did not run':>12s}")
    for arm in ARMS:
        k = sum(1 for p in paired if repaired(arm, p))
        fal = sum(1 for p in paired if by[(arm, p)]["outcome"] == "falsified")
        broke = len(paired) - k - fal
        lo, hi = wilson(k, len(paired))
        print(f"  {arm:6s} {k:10d} {k/len(paired):8.0%}  [{lo:.2f}, {hi:.2f}]   "
              f"{fal:15d} {broke:12d}")

    print("\n  'did not run' is a syntax or runtime failure in the repaired code.")
    print("  It counts against the arm: a repair that does not execute is not")
    print("  a repair.")

    # ---- paired tests -----------------------------------------------------
    print("\n  " + "-" * 74)
    print("  exact McNemar on discordant pairs")
    print(f"  {'comparison':14s} {'only left':>10s} {'only right':>11s} {'p':>10s}")
    comparisons = [("R2", "R0"), ("R3", "R0"), ("R3", "R2")]
    results = {}
    for left, right in comparisons:
        b = sum(1 for p in paired if repaired(left, p) and not repaired(right, p))
        c = sum(1 for p in paired if repaired(right, p) and not repaired(left, p))
        p_val = mcnemar_exact(b, c)
        results[(left, right)] = (b, c, p_val)
        print(f"  {left+' vs '+right:14s} {b:10d} {c:11d} {p_val:10.3f}")

    print("\n  Three comparisons on one family. Under Holm the smallest p would")
    print("  need to clear 0.05/3 = 0.017 to be called significant.")
    smallest = min(results.values(), key=lambda x: x[2])
    if smallest[2] > 0.017:
        print("  None of them do.")

    # ---- what this design could actually have detected --------------------
    # Every arm repairs almost nothing, so the comparison sits on a floor. A
    # null result here is easy to over-read: "feedback does not help" and "we
    # could not have seen it help unless it helped enormously" look identical
    # in the table above. Say which one this is.
    need = None
    for b in range(1, 30):
        if mcnemar_exact(b, 0) < 0.0167:
            need = b
            break
    best_n = max(sum(1 for p in paired if repaired(arm, p)) for arm in ARMS)
    print("\n  " + "-" * 74)
    print("  POWER. This is a floor effect, and it limits what the nulls mean.")
    print(f"    best arm repaired {best_n} of {len(paired)} programs")
    print(f"    to reach the Holm threshold an arm would have to repair {need}")
    print(f"    programs the other missed while missing none of the other's")
    print("\n  So the honest statement is not 'feedback makes no difference'.")
    print("  It is: richer feedback produces no LARGE improvement, and this")
    print("  design could not have detected a small one. What the arms do")
    print("  establish is the level: every condition leaves these models")
    print(f"  failing {1 - best_n / len(paired):.0%} of the time, so the barrier is")
    print("  not the thinness of the feedback.")

    # ---- did the diagnosis reach the programs that needed it? -------------
    print("\n  " + "-" * 74)
    print("  R3 broken down by which signal fired")
    groups = collections.defaultdict(lambda: [0, 0])
    nosig = []
    for p in paired:
        row = parent_row[p]
        sig = signals_fired(tasks[row["task_id"]], contracts[row["task_id"]],
                            row.get("counterexample"))
        key = "+".join(sig) if sig else "(none)"
        groups[key][1] += 1
        if repaired("R3", p):
            groups[key][0] += 1
        if not sig:
            nosig.append(p)
    for key, (k, n) in sorted(groups.items(), key=lambda x: -x[1][1]):
        print(f"    {key:24s} {k}/{n} repaired")

    if nosig:
        print(f"\n  On {len(nosig)} program(s) no signal fired, so R3's prompt was")
        print("  byte-identical to R2's. Those are not evidence either way and")
        print("  the comparison below excludes them.")
        strict = [p for p in paired if p not in set(nosig)]
        if strict:
            b = sum(1 for p in strict if repaired("R3", p) and not repaired("R2", p))
            c = sum(1 for p in strict if repaired("R2", p) and not repaired("R3", p))
            print(f"  R3 vs R2 on the {len(strict)} programs that did get a diagnosis: "
                  f"{b} vs {c}, p = {mcnemar_exact(b, c):.3f}")

    # ---- per task ---------------------------------------------------------
    print("\n  " + "-" * 74)
    print("  by task")
    per = collections.defaultdict(lambda: collections.Counter())
    for p in paired:
        tid = parent_row[p]["task_id"]
        per[tid]["n"] += 1
        for arm in ARMS:
            if repaired(arm, p):
                per[tid][arm] += 1
    print(f"    {'task':32s} {'n':>3s} {'R0':>5s} {'R2':>5s} {'R3':>5s}")
    for tid, c in sorted(per.items()):
        print(f"    {tid:32s} {c['n']:3d} {c['R0']:5d} {c['R2']:5d} {c['R3']:5d}")

    # ---- what to write ----------------------------------------------------
    print("\n" + "=" * 78)
    best = max(ARMS, key=lambda arm: sum(1 for p in paired if repaired(arm, p)))
    counts = {arm: sum(1 for p in paired if repaired(arm, p)) for arm in ARMS}
    print(f"  raw counts: " + "  ".join(f"{a}={counts[a]}" for a in ARMS))
    if smallest[2] > 0.017:
        print("  No arm separates from another. Read with the power note above:")
        print("  the claim this supports is that richer feedback does not rescue")
        print("  these models, not that feedback is worthless in general. The")
        print("  barrier is capability, and whether it is a capability floor or a")
        print("  small-model floor is what the frontier arm is for.")
    else:
        print(f"  {best} separates. Report it with the exact p and the Holm threshold.")

    (OUT / "e6_repair_arms.json").write_text(json.dumps({
        "paired_programs": len(paired),
        "repaired": counts,
        "mcnemar": {f"{l}_vs_{r}": {"only_left": b, "only_right": c, "p": pv}
                    for (l, r), (b, c, pv) in results.items()},
        "no_signal_programs": len(nosig),
    }, indent=2), encoding="utf-8")
    print(f"\n  wrote results/processed/e6_repair_arms.json")


if __name__ == "__main__":
    main()
