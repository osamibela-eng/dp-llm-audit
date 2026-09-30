"""E13 -- one blind annotator pass over the taxonomy worksheet, by a hosted model.

The annotator sees the full, pre-registered rules file and one worksheet item
(spec, reference, code). It never sees the model that wrote the code, the audit
outcome or the counterexample (rule 5 in TAXONOMY_RULES.md). The model used here
must not be one of the generators under study.

    python analysis/e13_annotate_llm.py --model gemini-3.8-flash
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generators.apikey import resolve  # noqa: E402
from generators.generation_runner import call_openai_compatible  # noqa: E402  (IPv4 fix, retries)

FAMILIES = ["F1_sensitivity", "F2_calibration", "F3_adjacency", "F4_composition",
            "F5_selection", "F6_data_dependent_flow", "F7_boundary", "F8_randomness",
            "F9_api_misuse", "F10_claim_mismatch", "ambiguous"]

PROMPT = """You are annotating defects in Python implementations of differentially private
mechanisms. Apply the decision rules below EXACTLY as written. Labelling is
multi-label: apply every family whose trigger holds. Use "ambiguous" only as the
rules describe. Mark over_noising true only when the defect makes the mechanism
MORE private than claimed.

===== DECISION RULES =====
{rules}
===== END RULES =====

Task specification:
{spec}

Claimed guarantee: epsilon = {eps}, delta = {delta}; adjacency = {adj}.

Correct reference implementation:
```python
{reference}
```

Program to annotate:
```python
{code}
```

Answer with ONLY a JSON object, no prose, of the form
{{"labels": [<one or more of {families}>], "over_noising": <true|false>, "note": "<one sentence>"}}"""


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    d = json.loads(m.group(0))
    bad = [l for l in d["labels"] if l not in FAMILIES]
    if bad or not d["labels"]:
        raise ValueError(f"invalid labels {d['labels']}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="https://generativelanguage.googleapis.com/v1beta/openai")
    ap.add_argument("--max-tokens", type=int, default=8000)
    a = ap.parse_args()

    rules = (ROOT / "docs" / "TAXONOMY_RULES.md").read_text(encoding="utf-8")
    key, _ = resolve("GEMINI_API_KEY")
    items = [json.loads(l) for l in open(ROOT / "results" / "processed" / "e13_worksheet.jsonl",
                                          encoding="utf-8")]
    out = ROOT / "results" / "processed" / f"e13_labels_{a.model.replace(':', '_')}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["item"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}

    with open(out, "a", encoding="utf-8") as fh:
        for it in items:
            if it["item"] in done:
                continue
            prompt = PROMPT.format(rules=rules, spec=it["spec"], eps=it["claimed_epsilon"],
                                   delta=it["claimed_delta"], adj=it["adjacency"],
                                   reference=it["reference"], code=it["code"],
                                   families=", ".join(FAMILIES))
            rec = None
            for attempt in range(3):
                try:
                    text = call_openai_compatible(a.base_url, key, a.model, prompt, 0.0, a.max_tokens)
                    d = parse(text)
                    rec = {"item": it["item"], "labels": sorted(set(d["labels"])),
                           "over_noising": bool(d.get("over_noising")), "note": d.get("note", ""),
                           "annotator": a.model}
                    break
                except Exception as e:                             # noqa: BLE001
                    print(f"  {it['item']} attempt {attempt + 1}: {type(e).__name__}: {e}", flush=True)
                    time.sleep(2)
            if rec is None:
                rec = {"item": it["item"], "labels": None, "error": "unparseable after 3 attempts",
                       "annotator": a.model}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{it['item']} {rec['labels']}", flush=True)


if __name__ == "__main__":
    main()
