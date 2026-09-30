"""Fresh-seed confirmation: re-audit every falsified program K more times.

Assessment 5 asks for this and the reason is that the headline number is a count
of statistical rejections. Each falsification is a Clopper-Pearson bound that
cleared a Bonferroni-corrected threshold, so at alpha = 0.01 across hundreds of
tests some rejections are expected to be seed luck even when the program is
fine. A rate built from those would be inflated in exactly the way a reviewer
will suspect.

Re-running with fresh seeds separates the two cases directly. The audit is not
deterministic given the program: the pilot phase picks events from fresh samples
and the confirm phase draws again, so an independent seed is an independent
attempt to falsify. Programs are classified by how many of K attempts succeed:

  reproducible   falsified in ALL K re-runs        -> report as falsified
  borderline     falsified in some but not all     -> report separately, never
                                                      silently as either
  one_off        falsified in NONE                 -> the original was seed luck

The borderline class matters more than it looks. A program near the detection
boundary is not a false positive; it is a real violation whose magnitude is
close to what this audit budget can resolve. Merging it into either neighbour
would misstate the finding, so it gets its own column in the paper.

    python analysis/confirm_seeds.py --k 3
    python analysis/confirm_seeds.py --k 3 --n-confirm 30000 --limit 20
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib
import sys
import time

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auditors.simple_auditor import falsify           # noqa: E402
from sandbox.runner import ScreenError, load_mechanism  # noqa: E402

# Seeds are fixed constants rather than drawn from the clock: the point is a
# fresh seed per re-run, not an unreproducible one.
SEEDS = [20260821, 777, 31337, 4242, 90210, 13, 2718281, 1618033]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(ROOT / "results" / "processed" / "outcomes_*.jsonl"))
    ap.add_argument("--k", type=int, default=3, help="re-runs per program")
    ap.add_argument("--n-confirm", type=int, default=30_000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "processed" / "confirm_seeds.jsonl"))
    a = ap.parse_args()

    if a.k > len(SEEDS):
        raise SystemExit(f"--k at most {len(SEEDS)} (extend SEEDS to go higher)")

    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml").read_text(encoding="utf-8"))["tasks"]}
    work = [r for r in load(a.glob) if r["outcome"] == "falsified"]
    if a.limit:
        work = work[:a.limit]

    out_path = pathlib.Path(a.out)
    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                done[(d["model"], d["task_id"], d["sample_idx"])] = d
        if done:
            print(f"resuming: {len(done)} already confirmed")

    print(f"{len(work)} falsified programs x {a.k} fresh seeds, "
          f"n_confirm={a.n_confirm}\n", flush=True)

    t_start = time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, r in enumerate(work, 1):
            key = (r["model"], r["task_id"], r["sample_idx"])
            if key in done:
                continue
            task = tasks[r["task_id"]]
            pairs = task["auditor"]["pairs"]
            hits, eps_lbs, errs = 0, [], []
            for s in SEEDS[:a.k]:
                try:
                    fn = load_mechanism(r["code"], task["function_name"])
                    res = falsify(fn, pairs, float(task["claimed_epsilon"]),
                                  float(task.get("claimed_delta") or 0.0),
                                  n_confirm=a.n_confirm, seed=s)
                except (ScreenError, Exception) as e:      # noqa: BLE001
                    errs.append(f"{type(e).__name__}: {e}")
                    continue
                if res.status == "falsified":
                    hits += 1
                    ce = getattr(res, "counterexample", None)
                    if ce is not None:
                        eps_lbs.append(getattr(ce, "empirical_eps_lower_bound", None))

            attempts = a.k - len(errs)
            if attempts == 0:
                cls = "error"
            elif hits == attempts:
                cls = "reproducible"
            elif hits == 0:
                cls = "one_off"
            else:
                cls = "borderline"

            rec = {"task_id": r["task_id"], "model": r["model"],
                   "sample_idx": r["sample_idx"], "code_sha256": r["code_sha256"],
                   "tier": task["tier"], "k": a.k, "attempts": attempts,
                   "falsified_in": hits, "classification": cls,
                   "eps_lb_reruns": [e for e in eps_lbs if e is not None],
                   "errors": errs or None, "n_confirm": a.n_confirm}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            flag = "" if cls == "reproducible" else f"   <-- {cls.upper()}"
            print(f"[{i:3d}/{len(work)}] {r['task_id']:30s} {r['model']:22s} "
                  f"{hits}/{attempts}{flag}", flush=True)

    print(f"\n{(time.time()-t_start)/60:.1f} min")
    summarise(out_path, tasks)


def summarise(path: pathlib.Path, tasks: dict):
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return
    c = collections.Counter(r["classification"] for r in rows)
    n = len(rows)
    print("\n" + "=" * 70)
    print("FRESH-SEED CONFIRMATION")
    print("=" * 70)
    for k in ("reproducible", "borderline", "one_off", "error"):
        if c[k]:
            print(f"  {k:14s} {c[k]:4d}   ({c[k]/n:.0%})")
    print(f"  {'total':14s} {n:4d}")

    if c["one_off"]:
        print(f"\n  {c['one_off']} original falsification(s) did not reproduce at any fresh")
        print("  seed. Those must be removed from the headline rate, and the")
        print("  corrected rate reported with this check named as its reason.")
    else:
        print("\n  Every original falsification reproduced at least once. The rate is")
        print("  not an artefact of a lucky seed.")

    print("\n  by tier:")
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        by[r["tier"]][r["classification"]] += 1
    for tier in sorted(by):
        cc = by[tier]
        tot = sum(cc.values())
        print(f"    tier {tier}:  reproducible {cc['reproducible']:3d}/{tot:<3d}  "
              f"borderline {cc['borderline']:2d}  one_off {cc['one_off']:2d}")

    print("\n  tasks with any non-reproducible falsification:")
    bad = collections.Counter(r["task_id"] for r in rows
                              if r["classification"] in ("borderline", "one_off"))
    if not bad:
        print("    none")
    for t, k in bad.most_common():
        print(f"    {t:32s} {k}")


if __name__ == "__main__":
    main()
