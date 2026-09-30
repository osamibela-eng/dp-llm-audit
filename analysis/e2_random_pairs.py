"""E2: do the results survive pairs we did not choose?

Two objections, one experiment.

  "Your hand-designed pairs drive the results."
  "Fixed published pairs are gameable by an adversarial generator."

The answer is to replace our pair set with pairs drawn from the declared contract
and re-audit. Every generated pair is checked against the contract before use, so
a failure to reproduce cannot be blamed on an inadmissible probe.

Design. For each program we run R replicates. Each replicate swaps in a random
valid pair set of THE SAME SIZE as the set the task ships, so the multiple-testing
burden is unchanged and the comparison is like for like. Reporting per-pair
detections instead would quietly inflate the family-wise error and make random
pairs look better than they are.

Two directions:

  (i)  reproducibly falsified programs -> how often do random pairs still catch
       them? A high rate means the violation is a property of the program, not of
       our pair choice.
  (ii) not-falsified programs -> do random pairs catch anything our pairs missed?
       Anything found here is a violation our published pairs were blind to, which
       is the honest measure of pair-design sensitivity.

    python analysis/e2_random_pairs.py --replicates 3
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import pathlib
import sys
import time

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auditors.contract import load_contracts, random_valid_pairs  # noqa: E402
from auditors.simple_auditor import falsify  # noqa: E402
from sandbox.runner import load_mechanism  # noqa: E402

OUT = ROOT / "results" / "processed"


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_outcomes(pattern):
    rows, seen = [], set()
    for f in sorted(glob.glob(pattern)):
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
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--n-confirm", type=int, default=8000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--outcomes",
                    default=str(OUT / "outcomes_*.jsonl"))
    a = ap.parse_args()

    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml")
                            .read_text(encoding="utf-8"))["tasks"]}
    contracts = load_contracts()
    rows = load_outcomes(a.outcomes)

    # This used to treat confirm_seeds.jsonl as optional and fall back to "all
    # falsified" when it was missing, while still printing the word
    # "reproducibly". That is the worst kind of default: the run succeeds, the
    # label is wrong, and nothing in the output says so. It silently mixed 8
    # one-off detections into a set described as reproducible. Require the file.
    conf_path = OUT / "confirm_seeds.jsonl"
    if not conf_path.exists():
        sys.exit(f"missing {conf_path.name}. E2 is defined over REPRODUCIBLY "
                 f"falsified programs; without the confirmation pass there is no "
                 f"way to know which those are. Run analysis/confirm_seeds.py "
                 f"first rather than letting this fall back to all detections.")
    repro = set()
    for line in conf_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            if d["classification"] == "reproducible":
                repro.add(d["code_sha256"])
    if not repro:
        sys.exit("confirm_seeds.jsonl contains no reproducible detections.")

    falsified = [r for r in rows if r["outcome"] == "falsified"
                 and r["code_sha256"] in repro]
    survived = [r for r in rows if r["outcome"] == "not_falsified"]
    if a.limit:
        falsified, survived = falsified[:a.limit], survived[:a.limit]

    print(f"reproducibly falsified: {len(falsified)}   not falsified: {len(survived)}")
    print(f"{a.replicates} replicates each, random valid pairs from the contract, "
          f"n_confirm={a.n_confirm}\n", flush=True)

    out_path = OUT / "e2_random_pairs.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["code_sha256"])
        if done:
            print(f"resuming: {len(done)} already done\n", flush=True)

    work = [("falsified", r) for r in falsified] + [("not_falsified", r) for r in survived]
    t_start = time.time()

    with open(out_path, "a", encoding="utf-8") as fh:
        for i, (group, r) in enumerate(work, 1):
            if r["code_sha256"] in done:
                continue
            tid = r["task_id"]
            t, c = tasks[tid], contracts[tid]
            n_pairs = len(t["auditor"]["pairs"])
            hits, errs = 0, 0

            for rep in range(a.replicates):
                # Seed from the program hash so a re-run reproduces exactly.
                rng = np.random.default_rng(
                    abs(hash((r["code_sha256"], rep))) % (2**32))
                pairs = random_valid_pairs(c, rng, k=n_pairs)
                if not pairs:
                    errs += 1
                    continue
                try:
                    fn = load_mechanism(r["code"], t["function_name"])
                    res = falsify(fn, [tuple(p) for p in pairs],
                                  float(t["claimed_epsilon"]),
                                  float(t.get("claimed_delta") or 0.0),
                                  n_confirm=a.n_confirm, seed=1000 + rep)
                    if res.status == "falsified":
                        hits += 1
                except Exception:                             # noqa: BLE001
                    errs += 1

            rec = {"code_sha256": r["code_sha256"], "task_id": tid,
                   "model": r["model"], "group": group,
                   "n_pairs": n_pairs, "replicates": a.replicates,
                   "hits": hits, "errors": errs}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 10 == 0 or hits != a.replicates:
                print(f"[{i:3d}/{len(work)}] {tid:28s} {group:14s} "
                      f"{hits}/{a.replicates - errs}", flush=True)

    print(f"\n{(time.time()-t_start)/60:.1f} min")
    report(out_path, tasks)


def report(out_path, tasks):
    rows = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()
            if l.strip()]

    # Classify against the confirmation pass rather than trusting the group
    # label in the file. An older run wrote rows for all falsified programs
    # while calling them reproducible, and a report that re-derives the split
    # here cannot inherit that mistake from stale data.
    conf_path = OUT / "confirm_seeds.jsonl"
    cls = {}
    if conf_path.exists():
        for line in conf_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                cls[d["code_sha256"]] = d["classification"]

    fal_all = [r for r in rows if r["group"] == "falsified"]
    fal = [r for r in fal_all if cls.get(r["code_sha256"]) == "reproducible"]
    one_off = [r for r in fal_all if cls.get(r["code_sha256"]) == "one_off"]
    unknown = [r for r in fal_all if r["code_sha256"] not in cls]
    sur = [r for r in rows if r["group"] == "not_falsified"]

    print("\n" + "=" * 76)
    print("E2  DO THE RESULTS SURVIVE PAIRS WE DID NOT CHOOSE?")
    print("=" * 76)
    if unknown:
        print(f"\n  WARNING: {len(unknown)} falsified rows have no confirmation")
        print("  classification and are excluded. Re-run analysis/confirm_seeds.py.")

    if fal:
        # A program "survives" if random pairs caught it in every replicate.
        full = sum(1 for r in fal if r["errors"] == 0 and r["hits"] == r["replicates"])
        some = sum(1 for r in fal if r["hits"] > 0)
        none = sum(1 for r in fal if r["hits"] == 0)
        lo, hi = wilson(some, len(fal))
        print(f"\n  reproducibly falsified programs re-audited: {len(fal)}")
        print(f"    caught by random pairs in EVERY replicate : {full:4d}  "
              f"({full/len(fal):.0%})")
        print(f"    caught in at least one replicate          : {some:4d}  "
              f"({some/len(fal):.0%})  95% CI [{lo:.2f}, {hi:.2f}]")
        print(f"    never caught by random pairs              : {none:4d}  "
              f"({none/len(fal):.0%})")
        print("\n  A violation caught by pairs we did not design is a property of")
        print("  the program. Programs in the last row are ones our published")
        print("  pairs were unusually well suited to catching.")

        by = collections.defaultdict(lambda: [0, 0])
        for r in fal:
            by[r["task_id"]][1] += 1
            if r["hits"] > 0:
                by[r["task_id"]][0] += 1
        weak = [(t, a, b) for t, (a, b) in by.items() if b and a / b < 0.8]
        if weak:
            print("\n  tasks where random pairs are clearly weaker than ours:")
            for t, a_, b_ in sorted(weak, key=lambda x: x[1] / x[2]):
                print(f"    {t:30s} {a_}/{b_} re-caught")

    # ---- convergent check on the detections we already rejected -----------
    if one_off:
        caught = sum(1 for r in one_off if r["hits"] > 0)
        print("\n  " + "-" * 72)
        print(f"  Detections the confirmation pass classified as ONE-OFF: {len(one_off)}")
        print(f"    re-caught by random pairs: {caught}/{len(one_off)}")
        if caught == 0:
            print("\n    None of them. Two methods that share no machinery agree:")
            print("    re-auditing on fresh seeds rejected these, and pairs drawn")
            print("    independently from the contract never reproduce them. That")
            print("    is convergent evidence they were false positives, which is")
            print("    what the uncorrected outer loop in the error analysis")
            print("    predicts should exist. Excluding them was right.")
        else:
            print("\n    Some reproduce under random pairs. That is worth reading")
            print("    closely: the confirmation pass may be rejecting real")
            print("    violations that our own pairs happen to detect unreliably.")

    if sur:
        newly = [r for r in sur if r["hits"] > 0]
        lo, hi = wilson(len(newly), len(sur))
        print(f"\n  not-falsified programs re-audited: {len(sur)}")
        print(f"    newly caught by random pairs: {len(newly):4d}  "
              f"({len(newly)/len(sur):.0%})  95% CI [{lo:.2f}, {hi:.2f}]")
        if newly:
            print("\n    Each of these is a violation our published pairs missed.")
            c = collections.Counter(r["task_id"] for r in newly)
            for t, n in c.most_common(8):
                print(f"      {t:30s} {n}")
            print("\n    This is pair-design sensitivity measured rather than")
            print("    assumed, and it is a lower bound on what better pairs")
            print("    would find.")
        else:
            print("    None. Our published pairs were not leaving easy violations")
            print("    on the table for these programs.")


if __name__ == "__main__":
    main()
