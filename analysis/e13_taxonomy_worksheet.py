"""E13 -- blind worksheet for the two-annotator failure taxonomy.

Reviewer 64A asked which categories of bugs occur and why. The taxonomy rules
(docs/TAXONOMY_RULES.md) were committed before any program was labelled; the
earlier single-annotator labels were produced with knowledge of the outcomes and
cover only the 240-program corpus, so they cannot support an agreement claim.

This script writes one worksheet item per REPRODUCIBLY falsified program in the
480-program corpus, following the blinding protocol in TAXONOMY_RULES.md: each
item carries the task specification, the reference implementation and the
generated code -- and NOT the model name, the audit outcome or the
counterexample. Items are shuffled under a fixed seed and keyed by an opaque id.

    python analysis/e13_taxonomy_worksheet.py
"""
from __future__ import annotations

import glob
import json
import pathlib
import random

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "processed" / "e13_worksheet.jsonl"
KEY = ROOT / "results" / "processed" / "e13_worksheet_key.jsonl"   # unblinding map, not given to annotators


def main():
    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml").read_text(encoding="utf-8"))["tasks"]}
    rows, seen = [], set()
    for f in sorted(glob.glob(str(ROOT / "results" / "processed" / "outcomes_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r["model"], r["task_id"], str(r["sample_idx"]))
            if k not in seen:
                seen.add(k)
                rows.append(r)
    conf = {}
    for line in open(ROOT / "results" / "processed" / "confirm_seeds.jsonl", encoding="utf-8"):
        c = json.loads(line)
        conf[(c["model"], c["task_id"], str(c["sample_idx"]))] = c["classification"]
    repro = [r for r in rows if r["outcome"] == "falsified"
             and conf.get((r["model"], r["task_id"], str(r["sample_idx"]))) == "reproducible"]
    if len(repro) != 160:
        raise SystemExit(f"expected 160 reproducible programs, found {len(repro)}")

    random.Random(20260929).shuffle(repro)
    with open(OUT, "w", encoding="utf-8") as w, open(KEY, "w", encoding="utf-8") as k:
        for i, r in enumerate(repro, 1):
            t = tasks[r["task_id"]]
            ref = (ROOT / t["reference"]).read_text(encoding="utf-8")
            item = {"item": f"P{i:03d}", "task_id": r["task_id"],
                    "claimed_epsilon": t["claimed_epsilon"], "claimed_delta": t["claimed_delta"],
                    "adjacency": t["adjacency"], "spec": t["spec"].strip(),
                    "reference": ref, "code": r["code"]}
            w.write(json.dumps(item) + "\n")
            k.write(json.dumps({"item": item["item"], "model": r["model"],
                                "task_id": r["task_id"], "sample_idx": r["sample_idx"],
                                "code_sha256": r["code_sha256"]}) + "\n")
    print(f"wrote {len(repro)} items -> {OUT.name} (key: {KEY.name})")


if __name__ == "__main__":
    main()
