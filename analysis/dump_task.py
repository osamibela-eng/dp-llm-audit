"""Print the reference and every falsified program for one task, for annotation.

Support tool for the taxonomy pass. Prints the spec, the reference, and each
falsified program with its content hash, which is the key the label file uses.

    python analysis/dump_task.py clipped_bounded_sum_v1
"""
from __future__ import annotations

import glob
import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: dump_task.py <task_id> [--no-ref]")
    tid = sys.argv[1]
    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml").read_text(encoding="utf-8"))["tasks"]}
    t = tasks[tid]

    if "--no-ref" not in sys.argv:
        print("=" * 78)
        print(f"TASK {tid}   tier {t['tier']}   eps={t['claimed_epsilon']} "
              f"delta={t.get('claimed_delta') or 0}   adjacency={t['adjacency']}")
        print("=" * 78)
        print(t["spec"].strip())
        ref = ROOT / "benchmark" / "references" / pathlib.Path(t["reference"]).name
        if not ref.exists():
            ref = ROOT / t["reference"]
        print("\n--- REFERENCE " + "-" * 62)
        print(ref.read_text(encoding="utf-8").strip())

    rows, seen = [], set()
    for f in sorted(glob.glob(str(ROOT / "results" / "processed" / "outcomes_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r["model"], r["task_id"], r["sample_idx"])
            if k in seen or r["task_id"] != tid or r["outcome"] != "falsified":
                continue
            seen.add(k)
            rows.append(r)

    for i, r in enumerate(rows, 1):
        print(f"\n--- [{i}] key={r['code_sha256'][:12]} " + "-" * 50)
        print(r["code"].strip())


if __name__ == "__main__":
    main()
