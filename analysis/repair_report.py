"""Paired R0 vs R2 repair analysis.

Assessment 6 fixed the success criterion, and it is stricter than "the auditor
stopped complaining":

    successful repair = passes functional tests
                      AND not re-falsified
                      AND introduces no new failure (no regression to
                          syntax/runtime/timeout/unsupported)

A program that silently becomes unauditable, or that stops executing, is not
repaired. Counting it as a success would let a model "fix" a privacy violation by
breaking the code, which is exactly the direction this benchmark exists to
detect.

The comparison is PAIRED: both arms were run over the identical pre-specified
subset (`analysis/select_repair_subset.py`), so each program contributes one R0
outcome and one R2 outcome. McNemar's exact test on the discordant pairs is the
right test - an unpaired chi-square would throw away the pairing and understate
the evidence.

Also reports repair rate BY ERROR FAMILY, because the taxonomy found a mean of
2.5 defects per program. The pre-registered expectation, recorded before these
results were read:

    Counterexample feedback improves local correction, but multi-defect programs
    limit complete repair.

If R2 beats R0 on single-defect programs and not on multi-defect ones, that is
the mechanism. If R2 does not beat R0 anywhere, that is a real negative result
about counterexample feedback and is reported as one.

    python analysis/repair_report.py
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
UNSUPPORTED = {"auditor_unsupported", "auditor_error"}
FUNCTIONAL_FAIL = {"semantic_fail"}

ROW_ORDER = ["parses", "executes", "passes functional tests", "not re-falsified",
             "still falsified", "new regression", "timeout / non-termination",
             "auditor unsupported"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on discordant counts b and c."""
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load_jsonl(pattern):
    out = []
    for f in sorted(glob.glob(str(pattern))):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                out.append(json.loads(line))
    return out


def main():
    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml").read_text(encoding="utf-8"))["tasks"]}

    # the pre-specified subset (skip its header record)
    subset = [r for r in load_jsonl(ROOT / "results" / "processed" / "repair_subset.jsonl")
              if not r.get("_header")]
    parents = {r["code_sha256"]: r for r in subset}
    print(f"pre-specified subset: {len(parents)} falsified programs\n")

    # repaired outcomes, keyed by (arm, parent hash)
    outcomes = load_jsonl(ROOT / "results" / "processed" / "repair_outcomes.jsonl")
    by_arm = {}
    for r in outcomes:
        arm = r.get("repair_arm") or (r["sample_idx"].split(":")[-1]
                                      if isinstance(r.get("sample_idx"), str) else None)
        if arm not in ("R0", "R2"):
            continue
        by_arm[(arm, r.get("parent_sha256"))] = r

    have = {a: sum(1 for (arm, _) in by_arm if arm == a) for a in ("R0", "R2")}
    print(f"re-audited: R0 {have['R0']}, R2 {have['R2']}")

    paired = [h for h in parents if ("R0", h) in by_arm and ("R2", h) in by_arm]
    print(f"programs with BOTH arms complete: {len(paired)}   <- all analysis below\n")
    if not paired:
        raise SystemExit("no paired programs yet - re-run once both arms finish")

    def classify(rec) -> dict:
        o = rec["outcome"]
        return {
            "parses": o != "syntax_fail",
            "executes": o not in NOT_EXECUTABLE,
            "passes functional tests": o not in NOT_EXECUTABLE | FUNCTIONAL_FAIL,
            "not re-falsified": o == "not_falsified",
            "still falsified": o == "falsified",
            "new regression": o in NOT_EXECUTABLE | UNSUPPORTED,
            "timeout / non-termination": o == "timeout",
            "auditor unsupported": o in UNSUPPORTED,
        }

    def success(rec) -> bool:
        c = classify(rec)
        return c["passes functional tests"] and c["not re-falsified"] and not c["new regression"]

    # ---- the review's table ---------------------------------------------
    print("=" * 72)
    print(f"REPAIR OUTCOMES   (paired, n = {len(paired)} per arm)")
    print("=" * 72)
    print(f"  {'outcome':30s} {'R0':>12s} {'R2':>12s}")
    for row in ROW_ORDER:
        r0 = sum(1 for h in paired if classify(by_arm[("R0", h)])[row])
        r2 = sum(1 for h in paired if classify(by_arm[("R2", h)])[row])
        print(f"  {row:30s} {r0:5d} {r0/len(paired):5.0%} {r2:5d} {r2/len(paired):5.0%}")

    s0 = {h for h in paired if success(by_arm[("R0", h)])}
    s2 = {h for h in paired if success(by_arm[("R2", h)])}
    print("\n  " + "-" * 62)
    lo0, hi0 = wilson(len(s0), len(paired))
    lo2, hi2 = wilson(len(s2), len(paired))
    print(f"  {'SUCCESSFUL REPAIR':30s} {len(s0):5d} {len(s0)/len(paired):5.0%} "
          f"{len(s2):5d} {len(s2)/len(paired):5.0%}")
    print(f"  {'  95% CI':30s} [{lo0:.2f},{hi0:.2f}]   [{lo2:.2f},{hi2:.2f}]")
    print("  functional pass AND not re-falsified AND no new failure")

    # ---- McNemar ---------------------------------------------------------
    b = len(s2 - s0)          # R2 succeeded, R0 did not
    c = len(s0 - s2)          # R0 succeeded, R2 did not
    both = len(s0 & s2)
    neither = len(paired) - both - b - c
    p = mcnemar_exact(b, c)
    print("\n" + "=" * 72)
    print("PAIRED COMPARISON   (McNemar exact, two-sided)")
    print("=" * 72)
    print(f"  both arms repaired it        {both:4d}")
    print(f"  R2 only                      {b:4d}   <- counterexample helped")
    print(f"  R0 only                      {c:4d}   <- counterexample hurt")
    print(f"  neither                      {neither:4d}")
    print(f"\n  discordant pairs {b + c}, p = {p:.4f}")
    if not math.isnan(p):
        if p < 0.05:
            better = "R2" if b > c else "R0"
            print(f"  -> {better} is better on this subset at alpha = 0.05")
        else:
            print("  -> no significant difference between the arms on this subset.")
            print("     With this many discordant pairs the test cannot resolve a")
            print("     difference smaller than roughly the observed gap; report as")
            print("     inconclusive, not as evidence of equivalence.")

    # ---- by defect count -------------------------------------------------
    lab_path = ROOT / "results" / "processed" / "taxonomy_annotator_labels.jsonl"
    if lab_path.exists():
        labs = {r["code_sha256"]: r for r in load_jsonl(lab_path)}
        print("\n" + "=" * 72)
        print("BY DEFECT COUNT   (the pre-registered mechanism)")
        print("=" * 72)
        print(f"  {'defects':10s} {'n':>4s} {'R0 repaired':>14s} {'R2 repaired':>14s}")
        groups = collections.defaultdict(list)
        for h in paired:
            k = len(labs[h]["labels"]) if h in labs else None
            if k is not None:
                groups["1 defect" if k == 1 else f"{k} defects"].append(h)
        for key in sorted(groups, key=lambda x: int(x.split()[0])):
            g = groups[key]
            r0 = sum(1 for h in g if h in s0)
            r2 = sum(1 for h in g if h in s2)
            print(f"  {key:10s} {len(g):4d} {r0:8d} {r0/len(g):5.0%} {r2:8d} {r2/len(g):5.0%}")

        # ---- by family ---------------------------------------------------
        print("\n" + "=" * 72)
        print("BY ERROR FAMILY   (a program counts once per family it carries)")
        print("=" * 72)
        fam = collections.defaultdict(list)
        for h in paired:
            for f in labs.get(h, {}).get("labels", []):
                fam[f].append(h)
        print(f"  {'family':26s} {'n':>4s} {'R0':>12s} {'R2':>12s}")
        for f in sorted(fam, key=lambda x: -len(fam[x])):
            g = fam[f]
            r0 = sum(1 for h in g if h in s0)
            r2 = sum(1 for h in g if h in s2)
            print(f"  {f:26s} {len(g):4d} {r0:5d} {r0/len(g):5.0%} {r2:5d} {r2/len(g):5.0%}")

    # ---- where the arms disagree ----------------------------------------
    if b or c:
        print("\n" + "=" * 72)
        print("DISCORDANT PROGRAMS")
        print("=" * 72)
        for h in sorted(s2 - s0):
            t = parents[h]["task_id"]
            print(f"  R2 only  {t:30s} {parents[h]['model']:22s} "
                  f"{by_arm[('R0', h)]['outcome']}")
        for h in sorted(s0 - s2):
            t = parents[h]["task_id"]
            print(f"  R0 only  {t:30s} {parents[h]['model']:22s} "
                  f"{by_arm[('R2', h)]['outcome']}")


if __name__ == "__main__":
    main()
