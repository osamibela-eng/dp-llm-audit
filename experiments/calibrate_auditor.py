"""Auditor calibration: run the simple auditor on every reference implementation
(expect: not_falsified) and every known-bug variant (record which are caught).

This produces the paper's tool-calibration table for the starter tasks.

Usage:
    python experiments/calibrate_auditor.py                 # fast (n=30k)
    python experiments/calibrate_auditor.py --n 100000      # tighter
    python experiments/calibrate_auditor.py --seed 1        # reproducible audit
"""
import argparse
import importlib.util
import json
import pathlib
import sys
import time

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auditors.simple_auditor import falsify  # noqa: E402


def load_function(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30_000, help="confirm samples per side")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="results/calibration.jsonl")
    args = ap.parse_args()

    tasks = yaml.safe_load((ROOT / "benchmark" / "tasks.yaml").read_text())["tasks"]
    rows, t_start = [], time.time()

    for task in tasks:
        if not task.get("reference"):
            continue
        pairs = [tuple(p) for p in task["auditor"]["pairs"]]
        eps, delta = task["claimed_epsilon"], task["claimed_delta"]

        # --- reference (expected: not_falsified)
        ref_path = ROOT / task["reference"]
        fn = load_function(ref_path, task["function_name"])
        r = falsify(fn, pairs, eps, delta, n_confirm=args.n, seed=args.seed)
        rows.append(dict(task=task["id"], program="REFERENCE", expected="not_falsified",
                         status=r.status, eps_lb=(r.counterexample.eps_lower_bound
                                                  if r.counterexample else None),
                         elapsed=round(r.elapsed_s, 1)))
        print(f"[{task['id']:32s}] REFERENCE               -> {r.status:14s} "
              f"({r.elapsed_s:.1f}s)")

        # --- known bugs
        if task.get("bugs"):
            bug_path = ROOT / task["bugs"]
            spec = importlib.util.spec_from_file_location(bug_path.stem, bug_path)
            bug_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bug_mod)
            for bug_name, expected in bug_mod.EXPECTED.items():
                bfn = getattr(bug_mod, bug_name)
                r = falsify(bfn, pairs, eps, delta, n_confirm=args.n, seed=args.seed)
                ce = r.counterexample.as_dict() if r.counterexample else None
                rows.append(dict(task=task["id"], program=bug_name, expected=expected,
                                 status=r.status,
                                 eps_lb=(ce["empirical_eps_lower_bound"] if ce else None),
                                 counterexample=ce, elapsed=round(r.elapsed_s, 1)))
                mark = "CAUGHT " if r.status == "falsified" else "missed "
                print(f"[{task['id']:32s}] {bug_name:28s} -> {mark}"
                      f"(expected {expected}, {r.elapsed_s:.1f}s"
                      + (f", eps_lb={ce['empirical_eps_lower_bound']:.2f})" if ce else ")"))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    caught = sum(1 for r in rows if r["program"] != "REFERENCE" and r["status"] == "falsified")
    bugs = sum(1 for r in rows if r["program"] != "REFERENCE")
    fp = sum(1 for r in rows if r["program"] == "REFERENCE" and r["status"] == "falsified")
    print("\n=== calibration summary ===")
    print(f"known bugs caught : {caught}/{bugs}")
    print(f"false positives   : {fp}/{sum(1 for r in rows if r['program']=='REFERENCE')} references flagged")
    print(f"total time        : {time.time()-t_start:.0f}s   -> {out}")


if __name__ == "__main__":
    main()
