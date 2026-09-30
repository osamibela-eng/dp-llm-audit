"""Freeze the experimental artefacts: hash every input the results depend on.

Assessment 5 asks for a freeze before the remaining analysis is run, and the
reason is specific rather than ceremonial. The repair experiment and the
fresh-seed confirmations both re-invoke models and auditors, so from this point
on there are two ways for a number in the paper to become unreproducible:

  1. a benchmark task, reference or auditor is edited after the programs that
     were audited against it, and
  2. a generation file is appended to (--resume does exactly this) after a rate
     has been computed from it.

Neither is visible in a diff of the results. Both are visible here, because the
manifest pins the *content* of every dependency:

  benchmark/tasks.yaml, references/, known_bugs/   what was asked and expected
  auditors/                                        how falsification was decided
  sandbox/, generators/prompts/                    how programs were run and asked for
  results/raw/*.jsonl                              the generations themselves
  results/calibration.jsonl                        detection power

Writes results/FREEZE.json. Re-running with --check re-hashes and reports any
drift, which is what makes the freeze worth having: it fails loudly instead of
silently producing different numbers.

    python analysis/freeze.py            # write the manifest
    python analysis/freeze.py --check    # verify nothing has moved
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]

# (label, glob) pairs. Order is the order they appear in the manifest.
TARGETS = [
    ("benchmark_tasks",   ["benchmark/tasks.yaml"]),
    ("references",        ["benchmark/references/*.py"]),
    ("known_bugs",        ["benchmark/known_bugs/*.py"]),
    ("tests",             ["benchmark/tests/*.py"]),
    ("auditors",          ["auditors/*.py"]),
    ("sandbox",           ["sandbox/*.py"]),
    ("prompts",           ["generators/prompts/*.txt"]),
    ("generation_code",   ["generators/*.py"]),
    ("analysis_code",     ["analysis/*.py"]),
    ("generations",       ["results/raw/gen_*.jsonl"]),
    ("calibration",       ["results/calibration.jsonl"]),
    ("outcomes",          ["results/processed/outcomes_*.jsonl"]),
]


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonl_len(p: pathlib.Path) -> int | None:
    """Record counts for JSONL so an accidental append is legible, not just a
    changed hash."""
    if p.suffix != ".jsonl":
        return None
    with open(p, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def collect() -> dict:
    out = {}
    for label, patterns in TARGETS:
        entries = {}
        for pat in patterns:
            for p in sorted(ROOT.glob(pat)):
                rel = p.relative_to(ROOT).as_posix()
                rec = {"sha256": sha256(p), "bytes": p.stat().st_size}
                n = jsonl_len(p)
                if n is not None:
                    rec["records"] = n
                entries[rel] = rec
        out[label] = entries
    return out


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="compare against the existing manifest instead of writing one")
    ap.add_argument("--stamp", default=None,
                    help="freeze date, ISO. Passed in rather than read from the clock "
                         "so re-running is idempotent.")
    a = ap.parse_args()

    path = ROOT / "results" / "FREEZE.json"
    current = collect()

    if a.check:
        if not path.exists():
            raise SystemExit(f"no manifest at {path} — run without --check first")
        frozen = json.loads(path.read_text(encoding="utf-8"))["files"]
        drift = []
        for label in set(frozen) | set(current):
            was, now = frozen.get(label, {}), current.get(label, {})
            for f in sorted(set(was) | set(now)):
                if f not in was:
                    drift.append(f"ADDED    {f}")
                elif f not in now:
                    drift.append(f"REMOVED  {f}")
                elif was[f]["sha256"] != now[f]["sha256"]:
                    dr = ""
                    if "records" in was[f] and "records" in now[f]:
                        dr = f"  ({was[f]['records']} -> {now[f]['records']} records)"
                    drift.append(f"CHANGED  {f}{dr}")
        if drift:
            print(f"{len(drift)} file(s) differ from the freeze:\n")
            for d in drift:
                print("  " + d)
            print("\nAny rate computed before this drift must be recomputed, or the")
            print("drift explained in docs/FINDINGS.md.")
            raise SystemExit(1)
        n = sum(len(v) for v in current.values())
        print(f"clean — all {n} frozen files match")
        return

    manifest = {
        "frozen_at": a.stamp,
        "git_head": git_head(),
        "note": "Analysis freeze taken after generation completed for all three "
                "models and before the repair experiment and fresh-seed "
                "confirmation audits. Verify with: python analysis/freeze.py --check",
        "files": current,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {path.relative_to(ROOT)}")
    for label, entries in current.items():
        recs = sum(e.get("records", 0) for e in entries.values())
        extra = f", {recs} records" if recs else ""
        print(f"  {label:18s} {len(entries):3d} files{extra}")


if __name__ == "__main__":
    main()
