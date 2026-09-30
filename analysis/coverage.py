"""Show, per task, how many usable programs a model actually produced.

A quota-limited run does not fail loudly. It produces a corpus that looks fine
in aggregate and is badly shaped underneath: the tasks it reached have full
samples and the tasks it never got to have none. A single rate computed over
that is a rate over whichever tasks happened to come first in the file, which
is not a property of the model at all.

This exists so that no partial corpus gets used without someone seeing its
shape first.

    python analysis/coverage.py --model gemini-3.6-flash
    python analysis/coverage.py                     # every model found
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--expect", type=int, default=None,
                    help="samples per task the design calls for")
    ap.add_argument("--raw-glob", default=str(ROOT / "results" / "raw" / "gen_*.jsonl"))
    a = ap.parse_args()

    tasks = [t["id"] for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml")
                            .read_text(encoding="utf-8"))["tasks"]]

    ok = collections.defaultdict(lambda: collections.Counter())
    bad = collections.defaultdict(lambda: collections.Counter())
    seen = collections.defaultdict(set)
    for f in sorted(glob.glob(a.raw_glob)):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = r.get("model") or "?"
            if a.model and m != a.model:
                continue
            key = (r.get("task_id"), r.get("sample_idx"))
            if r.get("code"):
                if key in seen[m]:
                    continue        # a retry that succeeded twice; count once
                seen[m].add(key)
                ok[m][r["task_id"]] += 1
            else:
                bad[m][r.get("task_id")] += 1

    if not ok and not bad:
        print("no generations found")
        return

    for m in sorted(set(ok) | set(bad)):
        total = sum(ok[m].values())
        covered = sum(1 for t in tasks if ok[m][t] > 0)
        per = [ok[m][t] for t in tasks]
        expect = a.expect or (max(per) if per else 0)

        print("=" * 68)
        print(f"{m}")
        print(f"  usable programs: {total}   tasks covered: {covered}/{len(tasks)}"
              f"   failed requests: {sum(bad[m].values())}")
        print()
        for t in tasks:
            n = ok[m][t]
            mark = "" if n else "   <-- NONE"
            print(f"    {t:32s} {n:3d} {'#' * n}{mark}")

        if covered < len(tasks):
            print()
            print(f"  UNUSABLE AS A RATE. {len(tasks) - covered} task(s) have no")
            print("  programs at all, so any percentage computed here describes")
            print("  only the tasks the run happened to reach, not the model.")
            print("  Finish the coverage before reporting anything from it.")
        elif len(set(per)) > 1:
            print()
            print(f"  Coverage is complete but uneven ({min(per)}-{max(per)} per")
            print("  task). Per-task rates are fine; a single pooled rate would")
            print("  silently weight the tasks with more samples more heavily.")
        else:
            print()
            print(f"  Balanced: {expect} samples for every task.")


if __name__ == "__main__":
    main()
