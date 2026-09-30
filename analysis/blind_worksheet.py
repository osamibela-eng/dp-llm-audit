"""Human-readable blind worksheet for taxonomy annotation, and its parser.

The JSONL worksheet is the machine-readable twin; this is the one a person
actually fills in. Forty programs is real work, and asking an annotator to edit
code embedded in JSON string escapes would add transcription errors on top of the
judgement errors we are trying to measure.

Blinding, concretely. The worksheet carries the task spec, the claimed parameters,
the adjacency and the code. It does NOT carry the model name, the outcome, the
counterexample, or the script's pre-label. Programs are shuffled under a fixed
seed so their order carries no information either. A `key` comment retains the
content hash, which reveals nothing about any of the above and is what lets the
scorer line the two sets of labels up afterwards.

    python analysis/blind_worksheet.py --make 40     # write docs/BLIND_WORKSHEET.md
    ...fill in the LABELS lines...
    python analysis/blind_worksheet.py --parse       # -> taxonomy_human_labels.jsonl
    python analysis/prelabel_taxonomy.py --score     # agreement against the pre-labels
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import random
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.prelabel_taxonomy import FAMILIES, NEEDS_HUMAN  # noqa: E402

MD_PATH = ROOT / "docs" / "BLIND_WORKSHEET.md"
OUT_PATH = ROOT / "results" / "processed" / "taxonomy_human_labels.jsonl"

SHORT = {f.split("_")[0]: f for f in FAMILIES}      # "F1" -> "F1_sensitivity"

HEADER = """# Blind taxonomy worksheet

Fill in the **LABELS** line under each program. Nothing else needs editing.

- Write family codes separated by commas: `F1, F6`. Short codes are enough.
- **Multi-label is expected.** A program can be wrong in more than one way, and
  recording only the most obvious one loses the rest.
- Write `ambiguous` if the defect fits none of the ten families, and say why in NOTE.
- Add `over_noising` to **MARKERS** if the defect makes the mechanism *more*
  private than claimed rather than less. That is an implementation error with a
  utility cost, not a privacy failure, and it must not be pooled with leaks.
- Leave LABELS blank to skip a program; skipped programs are excluded from the
  agreement statistic rather than counted as disagreement.

The decision rules are in `docs/TAXONOMY_RULES.md`. They were written before any
program was labelled and should be read first.

| code | family | |
|---|---|---|
| F1 | sensitivity | noise scale derived from the wrong sensitivity |
| F2 | calibration | right sensitivity, wrong map from (sensitivity, eps) to a noise parameter |
| F3 | adjacency | consistent, but assumes a different neighbouring relation than the task states |
| F4 | composition | multiple releases whose budget does not compose to the claimed total |
| F5 | selection | max / argmax / top-k step unnoised or noised incorrectly |
| F6 | data_dependent_flow | branch, loop bound, exception or early return visible in the output |
| F7 | boundary | correct on typical input, guarantee fails at an edge of the domain |
| F8 | randomness | wrong noise source, or shared where it must be fresh |
| F9 | api_misuse | right maths, wrong argument slot / interface assumption |
| F10 | claim_mismatch | a valid mechanism, but not for the claimed (eps, delta) |

Two traps worth re-reading before you start, both from the rules document:

- `max(0, noisy_count)` is **legitimate** post-processing — 0 is a public bound.
  Only a *data-dependent* clamp such as `min(noisy, max(data))` is F7.
- Reusing one noise draw is **not** automatically a violation. Across repeated
  releases of the *same* query it collapses them and is *more* private
  (`over_noising`); across *different* queries it makes a function of them
  deterministic and leaks. Read which case applies.

You are labelling what the program does wrong, **not** what the auditor caught.
Do not go looking for the counterexample: if labels were derived from
counterexamples, no family could ever be recorded as invisible to the auditor,
and that comparison is the point of the exercise.

---
"""


def load_falsified(pattern: str) -> list[dict]:
    rows, seen = [], set()
    for f in sorted(glob.glob(pattern)):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r["model"], r["task_id"], r["sample_idx"])
            if k in seen or r["outcome"] != "falsified":
                continue
            seen.add(k)
            rows.append(r)
    return rows


def make(n: int, seed: int, pattern: str):
    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml").read_text(encoding="utf-8"))["tasks"]}
    rows = load_falsified(pattern)
    sample = random.Random(seed).sample(rows, min(n, len(rows)))

    parts = [HEADER]
    for i, r in enumerate(sample, 1):
        t = tasks[r["task_id"]]
        delta = t.get("claimed_delta") or 0
        parts.append(f"""
## Program {i:02d}

<!-- key: {r['code_sha256']} -->

**Specification**

{t['spec'].strip()}

| | |
|---|---|
| function | `{t['function_name']}` |
| claimed epsilon | {t['claimed_epsilon']} |
| claimed delta | {delta} |
| adjacency | {t['adjacency']} |

```python
{r['code'].strip()}
```

**LABELS:**
**MARKERS:**
**NOTE:**

---
""")
    MD_PATH.write_text("".join(parts), encoding="utf-8")

    # Machine-readable twin, so the scorer never has to trust the Markdown.
    twin = ROOT / "results" / "processed" / "taxonomy_blind_worksheet.jsonl"
    with open(twin, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_header": True, "seed": seed, "n": len(sample),
                            "families": FAMILIES + ["ambiguous"],
                            "rules": "docs/TAXONOMY_RULES.md"}) + "\n")
        for i, r in enumerate(sample, 1):
            f.write(json.dumps({"blind_id": i, "_key": r["code_sha256"],
                                "labels": [], "markers": [], "note": ""}) + "\n")

    print(f"wrote {MD_PATH.relative_to(ROOT)}   ({len(sample)} programs, seed {seed})")
    print(f"wrote {twin.relative_to(ROOT)}")
    print("\nBlinded: no model name, no outcome, no counterexample, no pre-label.")
    print(f"Human-only families (the script does not attempt these): "
          f"{', '.join(f.split('_')[0] for f in NEEDS_HUMAN)}")


BLOCK_RE = re.compile(
    r"^## Program (\d+)\s*\n+<!-- key: ([0-9a-f]{64}) -->(.*?)(?=^## Program |\Z)",
    re.S | re.M)
FIELD_RE = {k: re.compile(rf"^\*\*{k}:\*\*(.*)$", re.M) for k in ("LABELS", "MARKERS", "NOTE")}


def normalise(tok: str) -> str | None:
    tok = tok.strip().strip(",").strip()
    if not tok:
        return None
    if tok.lower() in ("ambiguous", "over_noising"):
        return tok.lower()
    up = tok.upper().split("_")[0]
    if up in SHORT:
        return SHORT[up]
    if tok in FAMILIES:
        return tok
    raise ValueError(f"unrecognised label {tok!r}")


def parse():
    if not MD_PATH.exists():
        raise SystemExit(f"no worksheet at {MD_PATH} - run --make first")
    text = MD_PATH.read_text(encoding="utf-8")
    blocks = BLOCK_RE.findall(text)
    if not blocks:
        raise SystemExit("no program blocks found - was the worksheet edited structurally?")

    out, blank, errors = [], 0, []
    for num, key, body in blocks:
        vals = {}
        for k, rx in FIELD_RE.items():
            m = rx.search(body)
            vals[k] = (m.group(1).strip() if m else "")
        if not vals["LABELS"]:
            blank += 1
            continue
        try:
            labels = [x for x in (normalise(t) for t in vals["LABELS"].split(",")) if x]
            markers = [x for x in (normalise(t) for t in vals["MARKERS"].split(",")) if x]
        except ValueError as e:
            errors.append(f"Program {num}: {e}")
            continue
        out.append({"blind_id": int(num), "_key": key, "labels": labels,
                    "markers": markers, "note": vals["NOTE"]})

    if errors:
        print("could not parse:")
        for e in errors:
            print("  " + e)
        raise SystemExit(1)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(f"parsed {len(out)} labelled, {blank} left blank -> {OUT_PATH.relative_to(ROOT)}")
    if blank:
        print("  blank programs are excluded from agreement, not counted as disagreement")
    print("\nnow: python analysis/prelabel_taxonomy.py --score")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", type=int, metavar="N")
    ap.add_argument("--parse", action="store_true")
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--glob", default=str(ROOT / "results" / "processed" / "outcomes_*.jsonl"))
    a = ap.parse_args()
    if a.parse:
        parse()
    elif a.make:
        make(a.make, a.seed, a.glob)
    else:
        ap.error("use --make N or --parse")


if __name__ == "__main__":
    main()
