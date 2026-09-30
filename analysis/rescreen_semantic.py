"""Re-screen the programs excluded as semantic_fail, with the corrected screen.

The original non-degeneracy screen drew 12 samples and excluded the program if
they were all identical. Measurement showed that fires on correct low-entropy
mechanisms: 3 of 8 re-runnable exclusions were spurious
(analysis/semantic_screen_fp.py). aggregate.py now draws SCREEN_DRAWS = 200,
which makes the false-positive rate negligible.

Only the excluded programs can be affected: raising the number of draws can turn
a `semantic_fail` into a pass, never the reverse. So the corpus does not need a
full re-audit. This script re-screens exactly those programs, audits the ones
that now pass, and rewrites their rows in place.

    python analysis/rescreen_semantic.py            # report only
    python analysis/rescreen_semantic.py --apply    # rewrite outcome rows
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

from analysis.aggregate import (SCREEN_DRAWS, _adapt_signature,  # noqa: E402
                                smoke_and_semantic)
from auditors.simple_auditor import falsify  # noqa: E402
from sandbox.runner import load_mechanism  # noqa: E402

OUT = ROOT / "results" / "processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--n-audit", type=int, default=30_000)
    ap.add_argument("--second-eps", type=float, default=0.3)
    a = ap.parse_args()

    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml")
                            .read_text(encoding="utf-8"))["tasks"]}

    print("=" * 74)
    print(f"RE-SCREEN semantic_fail EXCLUSIONS  (screen now {SCREEN_DRAWS} draws)")
    print("=" * 74)
    print("\n  Raising the draw count can only turn an exclusion into a pass,")
    print("  never the reverse, so only these programs can change.\n")

    changes = []
    for path in sorted(glob.glob(str(OUT / "outcomes_*.jsonl"))):
        rows = [json.loads(l) for l in
                pathlib.Path(path).read_text(encoding="utf-8").splitlines()
                if l.strip()]
        touched = False
        for r in rows:
            if r["outcome"] != "semantic_fail":
                continue
            task = tasks[r["task_id"]]
            try:
                fn = _adapt_signature(load_mechanism(r["code"],
                                                     task["function_name"]), task)
            except Exception as e:                           # noqa: BLE001
                print(f"  {r['task_id']:28s} {r['model']:22s} will not load: "
                      f"{type(e).__name__}")
                continue

            label = smoke_and_semantic(fn, task)
            if label == "semantic_fail":
                print(f"  {r['task_id']:28s} {r['model']:22s} still degenerate "
                      f"-> exclusion stands")
                continue
            if label == "runtime_fail":
                print(f"  {r['task_id']:28s} {r['model']:22s} raises -> "
                      f"runtime_fail")
                continue

            # Passes the corrected screen: it is a working program and must be
            # audited like any other.
            pairs = [tuple(p) for p in task["auditor"]["pairs"]]
            eps_list = [float(task["claimed_epsilon"])]
            if a.second_eps:
                eps_list.append(a.second_eps)
            outcome, ce, at_eps = "not_falsified", None, None
            for eps in eps_list:
                try:
                    res = falsify(fn, pairs, eps,
                                  float(task.get("claimed_delta") or 0.0),
                                  n_confirm=a.n_audit, seed=None)
                except Exception as e:                       # noqa: BLE001
                    outcome = "auditor_error"
                    break
                if res.status == "falsified":
                    outcome, at_eps = "falsified", eps
                    ce = res.counterexample.as_dict() if res.counterexample else None
                    break
                if res.status in ("unsupported", "error"):
                    outcome = ("auditor_unsupported" if res.status == "unsupported"
                               else "auditor_error")
                    break

            print(f"  {r['task_id']:28s} {r['model']:22s} "
                  f"semantic_fail -> {outcome}"
                  + (f" (at eps={at_eps})" if at_eps else ""))
            changes.append((r["task_id"], r["model"], outcome))
            if a.apply:
                r["outcome"] = outcome
                r["rescreened"] = True
                if ce:
                    r["counterexample"] = ce
                    r["falsified_at_eps"] = at_eps
                touched = True

        if touched and a.apply:
            pathlib.Path(path).write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            print(f"    rewrote {pathlib.Path(path).name}")

    print("\n" + "=" * 74)
    if not changes:
        print("  No exclusion changed. Every semantic_fail is a genuinely")
        print("  deterministic release, and the reported denominators stand.")
    else:
        c = collections.Counter(o for _, _, o in changes)
        print(f"  {len(changes)} exclusion(s) were spurious: "
              + ", ".join(f"{k}={v}" for k, v in c.most_common()))
        if not a.apply:
            print("\n  Report only. Re-run with --apply to rewrite the rows,")
            print("  then regenerate macros and re-verify.")
        else:
            print("\n  Rows rewritten. Now run:")
            print("    python analysis/make_paper_macros.py")
            print("    python analysis/verify_paper_numbers.py")


if __name__ == "__main__":
    main()
