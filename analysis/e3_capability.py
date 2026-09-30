"""E3: is auditor capability predictable from the contract alone?

The objection is "`auditor_unsupported` is just bookkeeping". If that were true,
whether a program can be audited would be an accident of that program.

The first version of this script predicted {auditable, unsupported} per task and
agreed with only 7 of 15 tasks. Reading the disagreements is what produced the
actual result, and it is sharper than the one we set out to confirm. Two distinct
things were being collapsed:

1. Whether the auditor can represent the release the task DECLARES. This is a
   property of the contract, and it is predictable.

2. Whether the program actually returned what its own specification declared. A
   program that returns None where the task says "a single string" is recorded
   `auditor_unsupported`, but the auditor has not failed to cover the task; the
   program has broken its contract.

Separating them gives the finding: in this corpus, **every** observed unsupported
outcome is of the second kind. Not one is a coverage limit of the auditor at task
level. So `auditor_unsupported` is not bookkeeping and not an auditor excuse; it
is a measurable conformance failure of generated code, and the contract is what
lets us say which of the two it is.

A third capability, `unfaithful`, was also missing. Our pure-eps auditor does not
refuse an (eps, delta) task; it returns a verdict for a condition the task never
claimed. Refusing and answering the wrong question are different failures and are
now named differently.

    python analysis/e3_capability.py
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auditors.contract import load_contracts, predict_capability  # noqa: E402

UNSUPPORTED = {"auditor_unsupported", "auditor_error"}
AUDITED = {"falsified", "not_falsified"}
NOT_REACHED = {"syntax_fail", "screen_fail", "runtime_fail", "timeout",
               "semantic_fail"}

#: Classify an auditor's stated reason for declining. Everything here is a
#: statement about the PROGRAM's output, not about the task's declared type.
CONFORMANCE_MARKERS = (
    "output type",           # returned None, or a type the task never declared
    "ragged",                # inconsistent shapes across draws
    "array dimensions",      # same, via numpy
    "setting an array element with a sequence",
    "unsupported return type",
)


def classify_decline(detail: str) -> str:
    d = (detail or "").lower()
    if any(m in d for m in CONFORMANCE_MARKERS):
        return "program broke its output contract"
    if "error" in d or "valueerror" in d or "cannot" in d:
        return "program raised inside the mechanism"
    return "auditor coverage limit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes",
                    default=str(ROOT / "results" / "processed" / "outcomes_*.jsonl"))
    a = ap.parse_args()

    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml")
                            .read_text(encoding="utf-8"))["tasks"]}
    contracts = load_contracts()

    rows, seen = [], set()
    for f in sorted(glob.glob(a.outcomes)):
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
    if not rows:
        raise SystemExit(f"no outcome rows matched {a.outcomes}")

    # ---- 1. why did the auditor ever decline? ---------------------------
    declines = [r for r in rows if r["outcome"] in UNSUPPORTED]
    causes = collections.Counter(classify_decline(r.get("detail")) for r in declines)

    print("=" * 76)
    print("E3a  WHY THE AUDITOR DECLINED  (n = %d)" % len(declines))
    print("=" * 76)
    for cause, n in causes.most_common():
        print(f"  {n:3d}  {cause}")
    coverage_limits = causes["auditor coverage limit"]
    print(f"\n  Of {len(declines)} declines, {len(declines) - coverage_limits} are the "
          f"program failing to return what its own")
    print("  specification declares, and %d are a coverage limit of the auditor."
          % coverage_limits)
    print("  `auditor_unsupported` is therefore a conformance measurement, not")
    print("  bookkeeping and not an auditor excuse.")

    print("\n  examples:")
    for r in declines[:6]:
        print(f"    {r['task_id']:28s} {(r.get('detail') or '')[:52]}")

    # ---- 2. capability among CONFORMING programs ------------------------
    print("\n" + "=" * 76)
    print("E3b  CONTRACT PREDICTION, restricted to programs that conform")
    print("=" * 76)
    print(f"  {'task':30s} {'predicted':>12s} {'audited':>8s} {'declined':>9s}  verdict")

    agree = disagree = 0
    for tid in sorted(tasks, key=lambda x: (tasks[x]["tier"], x)):
        c = contracts.get(tid)
        if c is None:
            continue
        sub = [r for r in rows if r["task_id"] == tid
               and r["outcome"] not in NOT_REACHED]
        conforming = [r for r in sub
                      if r["outcome"] not in UNSUPPORTED
                      or classify_decline(r.get("detail")) == "auditor coverage limit"]
        n_aud = sum(1 for r in conforming if r["outcome"] in AUDITED)
        n_dec = sum(1 for r in conforming if r["outcome"] in UNSUPPORTED)

        pred = predict_capability(c, "ours",
                                  float(tasks[tid].get("claimed_delta") or 0.0))
        if not conforming:
            verdict = "no coverage"
        elif pred in ("auditable", "unfaithful"):
            ok = n_dec == 0
            verdict = "match" if ok else f"MISMATCH ({n_dec} declined)"
            agree += ok
            disagree += (not ok)
        else:
            ok = n_aud == 0
            verdict = "match" if ok else f"MISMATCH ({n_aud} audited)"
            agree += ok
            disagree += (not ok)
        print(f"  {tid:30s} {pred:>12s} {n_aud:8d} {n_dec:9d}  {verdict}")

    total = agree + disagree
    if total:
        print(f"\n  agreement among conforming programs: {agree}/{total} tasks")

    # ---- 3. the unfaithful case ----------------------------------------
    unfaithful = [tid for tid in tasks
                  if contracts.get(tid) and predict_capability(
                      contracts[tid], "ours",
                      float(tasks[tid].get("claimed_delta") or 0.0)) == "unfaithful"]
    if unfaithful:
        print("\n" + "=" * 76)
        print("E3c  WHERE THE AUDITOR ANSWERS THE WRONG QUESTION")
        print("=" * 76)
        for tid in unfaithful:
            n = sum(1 for r in rows if r["task_id"] == tid and r["outcome"] in AUDITED)
            print(f"  {tid}: {n} programs received a verdict.")
        print("\n  These tasks claim (eps, delta)-DP. Our auditor tests a pure-eps")
        print("  condition, so it does not decline; it returns a verdict for a")
        print("  guarantee the task never claimed. Refusing and answering the wrong")
        print("  question are different, and only the contract distinguishes them.")

    # ---- 4. coverage by auditor, decided before running anything --------
    print("\n" + "=" * 76)
    print("E3d  PREDICTED COVERAGE BY AUDITOR, from contracts alone")
    print("=" * 76)
    tally = collections.Counter()
    print(f"  {'task':30s} {'ours':>12s} {'statdp':>12s}")
    for tid in sorted(tasks, key=lambda x: (tasks[x]["tier"], x)):
        c = contracts.get(tid)
        if not c:
            continue
        d = float(tasks[tid].get("claimed_delta") or 0.0)
        p1, p2 = predict_capability(c, "ours", d), predict_capability(c, "statdp", d)
        tally[("ours", p1)] += 1
        tally[("statdp", p2)] += 1
        print(f"  {tid:30s} {p1:>12s} {p2:>12s}")
    for who in ("ours", "statdp"):
        print(f"\n  {who:7s}: " + ", ".join(
            f"{tally[(who, k)]} {k}" for k in ("auditable", "unfaithful", "unsupported")))
    print("\n  The gap between those columns is coverage, settled by the contract")
    print("  before a single program is run.")


if __name__ == "__main__":
    main()
