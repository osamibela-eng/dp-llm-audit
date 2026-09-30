"""Cross-auditor comparison: our counterexample auditor vs StatDP.

Assessment 5 asks what the second auditor buys. A single falsification rate from
a single tool is not evidence that the tool is measuring the property; two
independent tools agreeing is much better evidence, and — more usefully for the
paper — the *disagreements* say where each tool is blind.

Reports four things the review named:

  overlap                 both tools falsify the same program
  single-tool detections  each tool alone, which is the complementarity claim
  cost                    wall-clock seconds per program, both tools
  false positives         both tools against the 16 reference implementations

and breaks the agreement down by task, since the tools differ structurally
rather than randomly: our auditor is given hand-designed neighbouring pairs and
tests a fixed output partition, whereas StatDP searches for its own event and
generates its own databases.

Two limits, recorded rather than worked around:

  * StatDP generates numeric databases of a fixed size, so tasks whose input is
    categorical or whose semantics depend on a specific input shape are outside
    its reach. Those are reported as `unsupported`, NOT as `not falsified` —
    collapsing them would inflate our auditor's apparent advantage.
  * StatDP is run at ONE_DIFFER adjacency. The default (ALL_DIFFER) falsifies
    correct references, which is a mismatch with the add/remove-one adjacency
    every task in this benchmark claims. See AUDITOR_NOTES §8.

    python analysis/cross_auditor.py --iterations 20000
    python analysis/cross_auditor.py --limit 20        # smoke test
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import statistics
import sys
import time

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
# analysis/coverage.py shadows the `coverage` package when this directory is on
# sys.path, which crashes numba (imported by StatDP). See e9_prior_bugs.py.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if pathlib.Path(p or ".").resolve() != _HERE]
sys.path.insert(0, str(ROOT))

from auditors import statdp_adapter  # noqa: E402

AUDITED = ("falsified", "not_falsified")


def load_outcomes(pattern: str) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    # Same dedupe key aggregate.py uses; pilot and main runs can overlap.
    seen, out = set(), []
    for r in rows:
        k = (r["model"], r["task_id"], r["sample_idx"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def run_statdp(source: str, func: str, eps: float, iters: int, size: int,
               cores: int | None = None):
    """Returns (verdict, p_value, seconds). verdict in falsified/not_falsified/unsupported."""
    t0 = time.time()
    try:
        p, _ = statdp_adapter.audit_source(source, func, eps, input_size=size,
                                           event_iterations=iters,
                                           detect_iterations=iters, cores=cores)
    except Exception as e:
        return "unsupported", None, round(time.time() - t0, 1), f"{type(e).__name__}: {e}"
    dt = round(time.time() - t0, 1)
    if p is None:
        return "unsupported", None, dt, "no event found"
    return ("falsified" if p < 0.05 else "not_falsified"), p, dt, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(ROOT / "results" / "processed" / "outcomes_*.jsonl"))
    ap.add_argument("--iterations", type=int, default=20_000,
                    help="event AND detect iterations. StatDP's default is 50k; 20k keeps "
                         "a full sweep tractable and is recorded with the result.")
    ap.add_argument("--input-size", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cores", type=int, default=None,
                    help="cap StatDP worker processes. Needed when this sweep runs "
                         "alongside model generation: StatDP will otherwise take "
                         "every core and starve the generation server.")
    ap.add_argument("--references", action="store_true",
                    help="audit the 16 references instead — the false-positive check")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml").read_text(encoding="utf-8"))["tasks"]}

    if a.references:
        work = []
        for tid, t in tasks.items():
            src = ROOT / "benchmark" / "references" / f"{pathlib.Path(t['reference']).name}"
            if not src.exists():
                src = ROOT / t["reference"]
            work.append({"task_id": tid, "model": "REFERENCE", "sample_idx": 0,
                         "code": src.read_text(encoding="utf-8"), "outcome": "not_falsified"})
        default_out = ROOT / "results" / "processed" / "cross_auditor_references.jsonl"
    else:
        work = [r for r in load_outcomes(a.glob) if r["outcome"] in AUDITED]
        default_out = ROOT / "results" / "processed" / "cross_auditor.jsonl"

    if a.limit:
        work = work[:a.limit]
    out_path = pathlib.Path(a.out or default_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: these sweeps are long and get interrupted.
    done = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done.add((d["model"], d["task_id"], d["sample_idx"]))
        if done:
            print(f"resuming: {len(done)} already cross-audited", flush=True)

    print(f"{len(work)} programs, StatDP at {a.iterations} iterations, "
          f"ONE_DIFFER adjacency\n", flush=True)

    with open(out_path, "a", encoding="utf-8") as fh:
        for i, r in enumerate(work, 1):
            key = (r["model"], r["task_id"], r["sample_idx"])
            if key in done:
                continue
            t = tasks[r["task_id"]]
            verdict, p, secs, err = run_statdp(r["code"], t["function_name"],
                                               float(t["claimed_epsilon"]),
                                               a.iterations, a.input_size, a.cores)
            rec = {**key_fields(r), "ours": r["outcome"], "statdp": verdict,
                   "statdp_p": p, "statdp_seconds": secs, "statdp_note": err,
                   "iterations": a.iterations, "input_size": a.input_size}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            mark = "  <-- disagree" if (verdict in AUDITED and verdict != r["outcome"]) else ""
            print(f"[{i:3d}/{len(work)}] {r['task_id']:30s} ours={r['outcome']:14s} "
                  f"statdp={verdict:14s} {secs:6.1f}s{mark}", flush=True)

    summarise(out_path, tasks)


def key_fields(r: dict) -> dict:
    return {"task_id": r["task_id"], "model": r["model"], "sample_idx": r["sample_idx"],
            "tier": None}


def summarise(path: pathlib.Path, tasks: dict):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return
    for r in rows:
        r["tier"] = tasks[r["task_id"]]["tier"]

    both = [r for r in rows if r["statdp"] in AUDITED]
    unsupported = [r for r in rows if r["statdp"] == "unsupported"]

    print("\n" + "=" * 74)
    print("CROSS-AUDITOR AGREEMENT   (only programs BOTH tools could audit)")
    print("=" * 74)
    print(f"  cross-audited attempts        {len(rows)}")
    print(f"  StatDP could not audit        {len(unsupported)}"
          f"   ({len(unsupported)/len(rows):.0%})  -> excluded below, not counted as 'not falsified'")
    print(f"  comparable pairs              {len(both)}")

    if both:
        c = collections.Counter((r["ours"], r["statdp"]) for r in both)
        b = c[("falsified", "falsified")]
        ours_only = c[("falsified", "not_falsified")]
        sd_only = c[("not_falsified", "falsified")]
        neither = c[("not_falsified", "not_falsified")]
        n = len(both)
        print(f"\n  both falsify                  {b:4d}   ({b/n:.0%})")
        print(f"  ours only                     {ours_only:4d}   ({ours_only/n:.0%})")
        print(f"  StatDP only                   {sd_only:4d}   ({sd_only/n:.0%})")
        print(f"  neither                       {neither:4d}   ({neither/n:.0%})")
        agree = (b + neither) / n
        print(f"\n  raw agreement                 {agree:.0%}")
        union = b + ours_only + sd_only
        if union:
            print(f"  union falsified               {union:4d}"
                  f"   (ours alone finds {(b+ours_only)/union:.0%} of the union)")

    secs = [r["statdp_seconds"] for r in rows if r.get("statdp_seconds")]
    if secs:
        print(f"\n  StatDP cost                   {sum(secs)/len(secs):.1f}s per program "
              f"(median {statistics.median(secs):.1f}s, total {sum(secs)/60:.0f} min)")

    print("\n  by task:")
    print(f"    {'task':30s} {'tier':>4s}  both  ours  sdp  neither  unsup")
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        by[r["task_id"]][(r["ours"], r["statdp"])] += 1
    for tid in sorted(by, key=lambda x: (tasks[x]["tier"], x)):
        c = by[tid]
        print(f"    {tid:30s} {tasks[tid]['tier']:>4}  "
              f"{c[('falsified','falsified')]:4d}  {c[('falsified','not_falsified')]:4d}  "
              f"{c[('not_falsified','falsified')]:3d}  {c[('not_falsified','not_falsified')]:7d}  "
              f"{c[('falsified','unsupported')]+c[('not_falsified','unsupported')]:5d}")

    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
