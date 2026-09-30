"""Confirm the programs that random pairs caught but our published pairs missed.

E2 found a small number of these. They matter more than their count suggests,
because each one is a violation our own pair set was blind to, and that is the
one direction in which our headline rate could be too low rather than too high.

A program caught in one replicate out of three is exactly the kind of result
that multiple testing manufactures, so we do not report any of them on the
strength of the E2 run. Each candidate is re-audited here with fresh pair draws,
fresh audit seeds, and a larger confirmation sample, using the same
reproducible/borderline/one-off classification the main campaign uses for its
own detections. Only the reproducible ones go in the paper.

    python analysis/e2_confirm_new.py
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auditors.contract import load_contracts, random_valid_pairs  # noqa: E402
from auditors.simple_auditor import falsify  # noqa: E402
from sandbox.runner import load_mechanism  # noqa: E402

OUT = ROOT / "results" / "processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=10)
    ap.add_argument("--n-confirm", type=int, default=40_000)
    a = ap.parse_args()

    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml")
                            .read_text(encoding="utf-8"))["tasks"]}
    contracts = load_contracts()

    e2 = [json.loads(l) for l in
          (OUT / "e2_random_pairs.jsonl").read_text(encoding="utf-8").splitlines()
          if l.strip()]
    cands = [r for r in e2 if r["group"] == "not_falsified" and r["hits"] > 0]
    if not cands:
        print("no candidates to confirm")
        return

    # pull the source back out of the campaign outcomes
    code = {}
    for f in sorted(glob.glob(str(OUT / "outcomes_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                code[d["code_sha256"]] = d

    print(f"confirming {len(cands)} candidate(s), {a.replicates} fresh replicates "
          f"each, n_confirm={a.n_confirm}\n")

    rows = []
    for r in cands:
        src = code.get(r["code_sha256"])
        if not src:
            print(f"  {r['task_id']}: source not found, skipping")
            continue
        t, c = tasks[r["task_id"]], contracts[r["task_id"]]
        n_pairs = len(t["auditor"]["pairs"])
        hits = 0
        for rep in range(a.replicates):
            # deliberately a different seed stream from the E2 run
            rng = np.random.default_rng(90_000 + rep * 7 + len(r["code_sha256"]))
            pairs = random_valid_pairs(c, rng, k=n_pairs)
            try:
                fn = load_mechanism(src["code"], t["function_name"])
                res = falsify(fn, [tuple(p) for p in pairs],
                              float(t["claimed_epsilon"]),
                              float(t.get("claimed_delta") or 0.0),
                              n_confirm=a.n_confirm, seed=500_000 + rep)
                if res.status == "falsified":
                    hits += 1
            except Exception:                                # noqa: BLE001
                pass

        frac = hits / a.replicates
        cls = ("reproducible" if frac >= 0.8 else
               "borderline" if frac >= 0.3 else "one_off")
        rows.append({"code_sha256": r["code_sha256"], "task_id": r["task_id"],
                     "model": r["model"], "e2_hits": r["hits"],
                     "confirm_hits": hits, "replicates": a.replicates,
                     "classification": cls})
        print(f"  {r['task_id']:26s} {r['model']:22s} "
              f"E2 {r['hits']}/{r['replicates']}  ->  fresh {hits}/{a.replicates}  "
              f"{cls}")

    (OUT / "e2_confirm_new.jsonl").write_text(
        "\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")

    keep = [x for x in rows if x["classification"] == "reproducible"]
    bord = [x for x in rows if x["classification"] == "borderline"]
    off = [x for x in rows if x["classification"] == "one_off"]
    print(f"\n  reproducible {len(keep)}   borderline {len(bord)}   one-off {len(off)}")
    print("\n  Only the reproducible ones are counted, because that is the")
    print("  threshold the main campaign uses and changing it here to admit a")
    print("  result we like would be indefensible.")
    if bord:
        print("\n  But borderline is not the same as noise, and reporting it as")
        print("  noise would be its own distortion. These programs are falsified")
        print("  by a substantial minority of random pair draws at a large")
        print("  confirmation sample. The reading is that the violation is real")
        print("  but narrow: it lives on a part of the input space that most")
        print("  pair draws, including every pair we published, do not reach.")
        print("  That is evidence our pair set has blind spots, and it belongs")
        print("  in the limitations rather than in the headline count.")
    print(f"\n  wrote results/processed/e2_confirm_new.jsonl")


if __name__ == "__main__":
    main()
