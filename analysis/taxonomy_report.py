"""Taxonomy results: family distribution, co-occurrence, and the RQ2 breakdown.

Reads the single-annotator labels from `analysis/annotator_pass.py`. Read that
file's provenance note before using any number here: the labels are NOT an
independent second annotation, and the pre-label comparison at the bottom is a
consistency check, not inter-annotator reliability.

    python analysis/taxonomy_report.py
"""
from __future__ import annotations

import collections
import glob
import itertools
import json
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

FAMILIES = ["F1_sensitivity", "F2_calibration", "F3_adjacency", "F4_composition",
            "F5_selection", "F6_data_dependent_flow", "F7_boundary",
            "F8_randomness", "F9_api_misuse", "F10_claim_mismatch"]
SHORT = {f: f.split("_")[0] for f in FAMILIES}


def bar(n: int, total: int, width: int = 34) -> str:
    return "#" * round(width * n / total) if total else ""


def main():
    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml").read_text(encoding="utf-8"))["tasks"]}

    labels = {}
    for line in (ROOT / "results" / "processed" / "taxonomy_annotator_labels.jsonl"
                 ).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            labels[r["code_sha256"]] = r

    # model / tier come from the outcome rows
    meta, seen = {}, set()
    for f in sorted(glob.glob(str(ROOT / "results" / "processed" / "outcomes_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r["model"], r["task_id"], r["sample_idx"])
            if k in seen or r["outcome"] != "falsified":
                continue
            seen.add(k)
            meta[r["code_sha256"]] = r

    n = len(labels)
    print("=" * 76)
    print(f"ERROR TAXONOMY over {n} falsified programs   (multi-label)")
    print("single annotator, rules published before labelling - see annotator_pass.py")
    print("=" * 76)

    cnt = collections.Counter(f for r in labels.values() for f in r["labels"])
    print(f"\n  {'family':26s} {'n':>4s} {'% of programs':>14s}")
    for f in FAMILIES + ["ambiguous"]:
        print(f"  {f:26s} {cnt[f]:4d} {cnt[f]/n:13.0%}  {bar(cnt[f], n)}")

    per = collections.Counter(len(r["labels"]) for r in labels.values())
    total_labels = sum(len(r["labels"]) for r in labels.values())
    print(f"\n  labels per program: mean {total_labels/n:.2f}, "
          + ", ".join(f"{k}->{v}" for k, v in sorted(per.items())))

    # ---- co-occurrence ---------------------------------------------------
    print("\n" + "=" * 76)
    print("CO-OCCURRENCE   (pairs appearing in >= 4 programs)")
    print("=" * 76)
    pairs = collections.Counter()
    for r in labels.values():
        for a, b in itertools.combinations(sorted(set(r["labels"])), 2):
            pairs[(a, b)] += 1
    for (a, b), k in pairs.most_common():
        if k < 4:
            break
        print(f"  {SHORT.get(a,a):>4s} + {SHORT.get(b,b):<4s} {k:4d}   "
              f"({k/n:.0%} of falsified programs)")

    # ---- single-cause programs ------------------------------------------
    solo = [(h, r) for h, r in labels.items() if len(r["labels"]) == 1]
    print("\n" + "=" * 76)
    print(f"SINGLE-CAUSE PROGRAMS ({len(solo)})   one defect, nothing else wrong")
    print("=" * 76)
    for f in FAMILIES:
        these = [(h, r) for h, r in solo if r["labels"] == [f]]
        if these:
            print(f"  {f:26s} {len(these):3d}   "
                  + ", ".join(sorted({r['task_id'].replace('_v1', '') for _, r in these})))

    # ---- by tier ---------------------------------------------------------
    print("\n" + "=" * 76)
    print("BY TIER   (% of that tier's falsified programs carrying the family)")
    print("=" * 76)
    tier_n = collections.Counter(tasks[r["task_id"]]["tier"] for r in labels.values())
    print(f"  {'family':26s}" + "".join(f"{'tier '+str(t):>10s}" for t in (1, 2, 3)))
    for f in FAMILIES:
        row = f"  {f:26s}"
        for t in (1, 2, 3):
            k = sum(1 for r in labels.values()
                    if tasks[r["task_id"]]["tier"] == t and f in r["labels"])
            row += f"{k:4d} {k/tier_n[t]:4.0%}" if tier_n[t] else "     -"
        print(row)
    print("  " + "-" * 56)
    print(f"  {'falsified in tier':26s}" + "".join(f"{tier_n[t]:6d}    " for t in (1, 2, 3)))

    # ---- by model --------------------------------------------------------
    print("\n" + "=" * 76)
    print("BY MODEL")
    print("=" * 76)
    models = sorted({meta[h]["model"] for h in labels if h in meta})
    mn = collections.Counter(meta[h]["model"] for h in labels if h in meta)
    print(f"  {'family':26s}" + "".join(f"{m.split(':')[0][:12]:>14s}" for m in models))
    for f in FAMILIES:
        row = f"  {f:26s}"
        for m in models:
            k = sum(1 for h, r in labels.items()
                    if h in meta and meta[h]["model"] == m and f in r["labels"])
            row += f"{k:6d} {k/mn[m]:5.0%}" if mn[m] else "      -"
        print(row)
    print("  " + "-" * 68)
    print(f"  {'falsified':26s}" + "".join(f"{mn[m]:8d}      " for m in models))

    # ---- consistency vs the syntactic pre-labeller -----------------------
    pre_path = ROOT / "results" / "processed" / "taxonomy_prelabels.jsonl"
    if pre_path.exists():
        pre = {json.loads(l)["code_sha256"]: json.loads(l)
               for l in pre_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        print("\n" + "=" * 76)
        print("PRE-LABELLER vs ANNOTATOR   *** consistency check, NOT reliability ***")
        print("=" * 76)
        print("  Both were produced by the same author. This measures whether the")
        print("  syntactic rules capture what a reading of the code finds - it is not")
        print("  evidence that the taxonomy is reproducible across people.\n")
        print(f"  {'family':26s} {'annot':>6s} {'script':>7s} {'both':>6s} "
              f"{'script missed':>14s}")
        for f in FAMILIES:
            a = {h for h, r in labels.items() if f in r["labels"]}
            s = {h for h, r in pre.items() if f in r["prelabels"]}
            both = a & s
            print(f"  {f:26s} {len(a):6d} {len(s):7d} {len(both):6d} "
                  f"{len(a - s):14d}")
        only_script = sum(1 for h, r in pre.items()
                          if h in labels
                          and set(r["prelabels"]) - set(labels[h]["labels"]))
        print(f"\n  programs where the script asserted a family the annotator did not: "
              f"{only_script}")


if __name__ == "__main__":
    main()
