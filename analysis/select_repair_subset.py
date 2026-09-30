"""Pre-specify the repair subset, before either repair arm is run.

Assessment 5 was specific about this and the specificity is the point. Running
repair on "the falsified programs" and reporting a success rate invites the
obvious objection: the set was chosen after the outcomes were known, so any
convenient subset could have been picked. The defence is to fix the sample by a
rule, commit it to the repository, and run both arms over the identical list.

The rule, in full:

  size        50 falsified programs
  strata      tier 1 (basic mechanisms)          20
              tier 2 (composition, selection)    20
              tier 3 (adaptive, edge cases)      10
  balance     within each stratum, allocate as evenly as possible across the
              three models, then across tasks within a model, so no single task
              or model dominates a stratum
  order       ties broken by code_sha256, which is content-derived and so is
              arbitrary with respect to the outcome but fully reproducible
  eligibility outcome == falsified AND a counterexample is present, because R2
              cannot be constructed without one and an unpaired comparison
              would defeat the design

Both arms then run against this file:

    python analysis/select_repair_subset.py
    python repair/repair_runner.py --arm R0 --outcomes results/processed/repair_subset.jsonl ...
    python repair/repair_runner.py --arm R2 --outcomes results/processed/repair_subset.jsonl ...

Deviations from the intended stratum sizes are printed and written into the
file's header record rather than silently absorbed — if tier 3 yields only 31
falsified programs across all models, that is a fact about the benchmark and
belongs in the paper, not in a rounding decision.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

TARGETS = {1: 20, 2: 20, 3: 10}


def load(pattern: str) -> list[dict]:
    rows, seen = [], set()
    for f in sorted(glob.glob(pattern)):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r["model"], r["task_id"], r["sample_idx"])
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
    return rows


def round_robin(pool: list[dict], want: int) -> list[dict]:
    """Take `want` items, cycling model -> task so neither dominates.

    Within a (model, task) bucket, order by code_sha256: content-derived, hence
    arbitrary with respect to the outcome, hence not a choice.
    """
    buckets = collections.defaultdict(list)
    for r in pool:
        buckets[(r["model"], r["task_id"])].append(r)
    for b in buckets.values():
        b.sort(key=lambda r: r["code_sha256"])

    # Cycle models evenly; inside a model, cycle its tasks evenly.
    by_model = collections.defaultdict(list)
    for (model, task), b in buckets.items():
        by_model[model].append([task, b])
    for lst in by_model.values():
        lst.sort(key=lambda x: x[0])

    picked, exhausted = [], False
    while len(picked) < want and not exhausted:
        exhausted = True
        for model in sorted(by_model):
            tasks = by_model[model]
            for entry in tasks:
                if entry[1]:
                    picked.append(entry[1].pop(0))
                    exhausted = False
                    break          # one per model per sweep, then next model
            if len(picked) >= want:
                break
    return picked[:want]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(ROOT / "results" / "processed" / "outcomes_*.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results" / "processed" / "repair_subset.jsonl"))
    a = ap.parse_args()

    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml").read_text(encoding="utf-8"))["tasks"]}
    rows = load(a.glob)

    eligible = [r for r in rows if r["outcome"] == "falsified" and r.get("counterexample")]
    no_ce = [r for r in rows if r["outcome"] == "falsified" and not r.get("counterexample")]

    print(f"falsified programs        {sum(1 for r in rows if r['outcome']=='falsified')}")
    print(f"  with a counterexample   {len(eligible)}   (eligible)")
    if no_ce:
        print(f"  without                 {len(no_ce)}   EXCLUDED — R2 cannot be built")

    selected, notes = [], []
    for tier, want in TARGETS.items():
        pool = [r for r in eligible if tasks[r["task_id"]]["tier"] == tier]
        got = round_robin(pool, want)
        selected += got
        status = "" if len(got) == want else f"  <-- only {len(pool)} available"
        if len(got) != want:
            notes.append(f"tier {tier}: wanted {want}, took {len(got)} "
                         f"(pool was {len(pool)})")
        print(f"  tier {tier}: {len(got):3d} / {want}{status}")

    print(f"\nselected {len(selected)}")
    print("  by model: " + ", ".join(
        f"{m}={n}" for m, n in sorted(collections.Counter(r["model"] for r in selected).items())))
    tc = collections.Counter(r["task_id"] for r in selected)
    print(f"  distinct tasks: {len(tc)}   most from one task: {max(tc.values())}")

    out = pathlib.Path(a.out)
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "_header": True,
            "rule": "50 falsified programs; tier strata 20/20/10; round-robin over "
                    "model then task; ties by code_sha256; eligibility requires a "
                    "counterexample so R0 and R2 are paired over an identical set",
            "targets": {str(k): v for k, v in TARGETS.items()},
            "selected": len(selected),
            "deviations": notes or None,
            "eligible_pool": len(eligible),
        }) + "\n")
        for r in selected:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}  (first line is a header record)")


if __name__ == "__main__":
    main()
