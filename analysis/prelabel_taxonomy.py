"""Syntactic pre-labelling for the error taxonomy, plus the blinding workflow.

Reads `docs/TAXONOMY_RULES.md` as its specification. Every rule implemented here
corresponds to a canonical instance named in that document, and the mapping is
recorded in RULES below so a reader can check the implementation against the
published rule rather than against my intent.

What this script can and cannot do is the important part:

  detectable syntactically   F6 (data-dependent flow), F8 (randomness), F9 (API
                             misuse), F10 (claim mismatch), and two named
                             instances of F2
  partial                    F1 — one recognisable instance (clip bounds taken
                             from the data); the family as a whole is not
  NOT detectable             F3 (adjacency), F4 (composition), F5 (selection),
                             F7 (boundary), and most of F2

The second list is not a limitation to be worked around. Deciding whether a
sensitivity constant matches the query's true sensitivity requires reading the
spec against the code, which is the judgement the human annotator is there to
make. A script that guessed at F1 would produce labels that look authoritative
and are not, and the agreement statistic against a human would then be measuring
the quality of my guess rather than the clarity of the rules. Those families are
emitted as `needs_human` and left empty.

    python analysis/prelabel_taxonomy.py                    # pre-label
    python analysis/prelabel_taxonomy.py --blind-sample 40  # worksheet for a human
    python analysis/prelabel_taxonomy.py --score            # agreement, after filling it in
"""
from __future__ import annotations

import argparse
import ast
import collections
import glob
import json
import pathlib
import random

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]

FAMILIES = ["F1_sensitivity", "F2_calibration", "F3_adjacency", "F4_composition",
            "F5_selection", "F6_data_dependent_flow", "F7_boundary",
            "F8_randomness", "F9_api_misuse", "F10_claim_mismatch"]

# F7 is human-only for a reason worth stating rather than hiding. The rules
# document draws the line at whether the clamping bound is public: `max(0, noisy)`
# is legitimate post-processing, `min(noisy, max(data))` is not. Both are a call
# to min/max wrapping the release, and telling them apart needs the annotator to
# decide whether the bound is public — which is judgement, not syntax. A rule that
# guessed would flag correct code, and the taxonomy would inherit that error.
NEEDS_HUMAN = ["F1_sensitivity", "F3_adjacency", "F4_composition", "F5_selection",
               "F7_boundary"]

# rule id -> (families, the canonical instance in TAXONOMY_RULES.md it implements)
# Families is a LIST: the rules document is explicit that some defects belong to
# more than one family, and collapsing those to a primary label would lose the
# second one for every program that has it.
RULES = {
    "empty_early_return":  (["F6_data_dependent_flow"], "`if not data: return 0`"),
    "raise_on_data":       (["F6_data_dependent_flow"], "`if n < k: raise ValueError(...)`"),
    "data_dependent_clip": (["F6_data_dependent_flow", "F1_sensitivity"],
                            "clip bounds from min(data)/max(data) - also makes the "
                            "clip range data-dependent, hence F1"),
    "inverted_scale":      (["F2_calibration"], "`scale = epsilon / sensitivity`"),
    "gaussian_no_delta":   (["F2_calibration", "F10_claim_mismatch"],
                            "Gaussian released under a pure-eps claim - the rules "
                            "document names this as F2 AND F10"),
    "seeded_rng":          (["F8_randomness"], "seeded / deterministic RNG"),
    "noise_outside_loop":  (["F8_randomness"], "noise drawn once outside a per-iteration loop"),
    "uniform_noise":       (["F8_randomness"], "uniform substituted for Laplace/Gaussian"),
    "ignores_rng_arg":     (["F9_api_misuse"], "injected rng ignored for module-level randomness"),
    "hardcoded_epsilon":   (["F10_claim_mismatch"], "hard-coded epsilon shadowing the argument"),
}


# ---------------------------------------------------------------- detectors

def _calls(tree):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)]


def _call_name(node) -> str:
    f = node.func
    parts = []
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    return ".".join(reversed(parts))


def _returns_constant_on_empty(tree) -> bool:
    """`if not data: return <const>` / `if len(data) == 0: return <const>`."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        t = node.test
        empty = False
        if isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not):
            empty = True
        elif isinstance(t, ast.Compare):
            src = ast.dump(t)
            if "len" in src and ("Eq()" in src or "Lt()" in src or "LtE()" in src):
                empty = True
        if not empty:
            continue
        for s in node.body:
            if isinstance(s, ast.Return) and isinstance(s.value, ast.Constant):
                return True
    return False


def _raises_on_data(tree) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and any(isinstance(s, ast.Raise) for s in node.body):
            if "len" in ast.dump(node.test) or "data" in ast.dump(node.test):
                return True
    return False


def _data_dependent_clip(tree) -> bool:
    for c in _calls(tree):
        n = _call_name(c)
        if n.split(".")[-1] in ("min", "max", "amin", "amax", "percentile", "quantile"):
            if any("data" in ast.dump(arg) for arg in c.args):
                return True
    return False


def _inverted_scale(tree) -> bool:
    """`something / epsilon` is right; `epsilon / something` in a noise scale is not."""
    for c in _calls(tree):
        if _call_name(c).split(".")[-1] not in ("laplace", "normal", "gauss"):
            continue
        for arg in list(c.args) + [k.value for k in c.keywords]:
            for sub in ast.walk(arg):
                if (isinstance(sub, ast.BinOp) and isinstance(sub.op, ast.Div)
                        and isinstance(sub.left, ast.Name)
                        and sub.left.id in ("epsilon", "eps", "e")):
                    return True
    return False


def _gaussian_used(tree) -> bool:
    return any(_call_name(c).split(".")[-1] in ("normal", "gauss", "randn")
               for c in _calls(tree))


def _uniform_noise(tree) -> bool:
    return any(_call_name(c).split(".")[-1] in ("uniform", "random", "rand")
               for c in _calls(tree))


def _seeded_rng(tree) -> bool:
    for c in _calls(tree):
        n = _call_name(c)
        if n.endswith("seed") or n.endswith("default_rng") and c.args:
            return True
    return False


def _ignores_rng_arg(tree, fname: str) -> bool:
    """Signature takes `rng` but the body calls module-level np.random.*."""
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == fname), None)
    if fn is None or not any(a.arg == "rng" for a in fn.args.args):
        return False
    for c in _calls(fn):
        n = _call_name(c)
        if n.startswith("np.random.") or n.startswith("numpy.random.") or n.startswith("random."):
            return True
    return False


def _noise_outside_loop(tree) -> bool:
    """A draw assigned before a loop, and used inside it."""
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        drawn: dict[str, int] = {}
        for i, stmt in enumerate(fn.body):
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                if _call_name(stmt.value).split(".")[-1] in ("laplace", "normal", "gauss"):
                    for t in stmt.targets:
                        if isinstance(t, ast.Name):
                            drawn[t.id] = i
            if isinstance(stmt, (ast.For, ast.While)) and drawn:
                used = {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name)}
                if used & set(drawn):
                    return True
    return False


def _hardcoded_epsilon(tree, fname: str) -> bool:
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == fname), None)
    if fn is None or not any(a.arg in ("epsilon", "eps") for a in fn.args.args):
        return False
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Name) and t.id in ("epsilon", "eps")
                        and isinstance(node.value, ast.Constant)):
                    return True
    return False


def prelabel(code: str, task: dict) -> tuple[list[str], list[str], str | None]:
    """Returns (families, fired rule ids, parse error)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [], [], f"SyntaxError: {e}"

    fname = task["function_name"]
    claims_pure = not float(task.get("claimed_delta") or 0.0)

    fired = []
    if _returns_constant_on_empty(tree):
        fired.append("empty_early_return")
    if _raises_on_data(tree):
        fired.append("raise_on_data")
    if _data_dependent_clip(tree):
        fired.append("data_dependent_clip")
    if _inverted_scale(tree):
        fired.append("inverted_scale")
    if claims_pure and _gaussian_used(tree) and not _uniform_noise(tree):
        fired.append("gaussian_no_delta")
    if _seeded_rng(tree):
        fired.append("seeded_rng")
    if _noise_outside_loop(tree):
        fired.append("noise_outside_loop")
    if _uniform_noise(tree) and not _gaussian_used(tree):
        fired.append("uniform_noise")
    if _ignores_rng_arg(tree, fname):
        fired.append("ignores_rng_arg")
    if _hardcoded_epsilon(tree, fname):
        fired.append("hardcoded_epsilon")

    fams = sorted({f for r in fired for f in RULES[r][0]})
    return fams, fired, None


# Families any rule can emit. Overlaps NEEDS_HUMAN, and the overlap is the honest
# case: `data_dependent_clip` is one recognisable instance of F1, but recognising
# it does not make F1 script-detectable in general. Those families are reported as
# PARTIAL so the count is never mistaken for a complete F1 tally.
SCRIPT_FAMILIES = {f for fams, _ in RULES.values() for f in fams}
PARTIAL = sorted(SCRIPT_FAMILIES & set(NEEDS_HUMAN))


# ---------------------------------------------------------------- workflows

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


def cohens_kappa(a: list[bool], b: list[bool]) -> float | None:
    """Binary kappa. None when one rater never varies — kappa is undefined there
    and reporting 0.0 would read as disagreement rather than as no information."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(ROOT / "results" / "processed" / "outcomes_*.jsonl"))
    ap.add_argument("--blind-sample", type=int, default=None,
                    help="write a blinded worksheet of N programs for human labelling")
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--score", action="store_true",
                    help="compare the filled-in worksheet against the pre-labels")
    a = ap.parse_args()

    proc = ROOT / "results" / "processed"
    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml").read_text(encoding="utf-8"))["tasks"]}
    pre_path = proc / "taxonomy_prelabels.jsonl"
    work_path = proc / "taxonomy_blind_worksheet.jsonl"

    if a.score:
        return score(pre_path, work_path)

    rows = load_falsified(a.glob)
    print(f"{len(rows)} falsified programs\n")

    out = []
    for r in rows:
        fams, fired, err = prelabel(r["code"], tasks[r["task_id"]])
        out.append({"task_id": r["task_id"], "model": r["model"],
                    "sample_idx": r["sample_idx"], "code_sha256": r["code_sha256"],
                    "tier": tasks[r["task_id"]]["tier"],
                    "prelabels": fams, "rules_fired": fired,
                    "needs_human": NEEDS_HUMAN, "parse_error": err})
    with open(pre_path, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")

    c = collections.Counter(f for o in out for f in o["prelabels"])
    rc = collections.Counter(r for o in out for r in o["rules_fired"])
    none = sum(1 for o in out if not o["prelabels"])
    print("pre-labelled families (syntactic rules only):")
    for fam in FAMILIES:
        if fam in PARTIAL:
            print(f"  {fam:26s} {c[fam]:4d}   PARTIAL - one instance only, human must complete")
        elif fam in NEEDS_HUMAN:
            print(f"  {fam:26s}    -  human only, not attempted")
        else:
            print(f"  {fam:26s} {c[fam]:4d}")
    print(f"\n  no syntactic rule fired   {none:4d}   ({none/len(out):.0%})")
    print("  -> these are NOT unlabelled programs; they are programs whose defect")
    print("     lives in a family only a human can assign. Do not read the zero")
    print("     as 'no error found'.")
    print("\nrules fired:")
    for r, k in rc.most_common():
        print(f"  {r:22s} {k:4d}   ({RULES[r][1]})")
    print(f"\nwrote {pre_path.relative_to(ROOT)}")

    if a.blind_sample:
        rnd = random.Random(a.seed)
        sample = rnd.sample(out, min(a.blind_sample, len(out)))
        by_row = {(r["model"], r["task_id"], r["sample_idx"]): r for r in rows}
        with open(work_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "_header": True,
                "instructions": "Fill `labels` with a list from FAMILIES (multi-label; "
                                "use \"ambiguous\" if none fit). Optionally add `note`. "
                                "Do not consult prelabels, counterexamples, or model names.",
                "families": FAMILIES + ["ambiguous"],
                "marker": "add \"over_noising\" to `markers` if the defect makes the "
                          "mechanism MORE private than claimed",
                "rules": "docs/TAXONOMY_RULES.md",
                "seed": a.seed, "n": len(sample),
            }) + "\n")
            for i, o in enumerate(sample):
                src = by_row[(o["model"], o["task_id"], o["sample_idx"])]
                t = tasks[o["task_id"]]
                # Blinded: no model, no outcome, no prelabels, no counterexample.
                f.write(json.dumps({
                    "blind_id": i,
                    "_key": o["code_sha256"],
                    "spec": t["spec"].strip(),
                    "function_name": t["function_name"],
                    "claimed_epsilon": t["claimed_epsilon"],
                    "claimed_delta": t.get("claimed_delta"),
                    "adjacency": t["adjacency"],
                    "code": src["code"],
                    "labels": [], "markers": [], "note": "",
                }) + "\n")
        print(f"wrote {work_path.relative_to(ROOT)}  ({len(sample)} blinded programs)")
        print("  fill in `labels`, then: python analysis/prelabel_taxonomy.py --score")


def score(pre_path: pathlib.Path, work_path: pathlib.Path):
    """Compare human labels against the script's pre-labels.

    Human labels come from `taxonomy_human_labels.jsonl`, which is produced by
    `analysis/blind_worksheet.py --parse` from the Markdown worksheet the
    annotator actually edits. The JSONL worksheet is only the blinded stub list.
    """
    human_path = work_path.with_name("taxonomy_human_labels.jsonl")
    if not human_path.exists():
        raise SystemExit(
            f"no human labels at {human_path}\n"
            "  fill in docs/BLIND_WORKSHEET.md, then:\n"
            "    python analysis/blind_worksheet.py --parse")
    pre = {json.loads(l)["code_sha256"]: json.loads(l)
           for l in pre_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    hum = [json.loads(l) for l in human_path.read_text(encoding="utf-8").splitlines()
           if l.strip() and not json.loads(l).get("_header")]
    filled = [h for h in hum if h.get("labels") and h["_key"] in pre]

    stale = [h for h in hum if h.get("labels") and h["_key"] not in pre]
    if stale:
        print(f"  {len(stale)} labelled program(s) are not in the current pre-label set "
              f"- worksheet predates the latest audit; excluded")

    print(f"human labels: {len(hum)} rows, {len(filled)} usable")
    if not filled:
        raise SystemExit("nothing labelled yet")
    if len(filled) < len(hum):
        print(f"  agreement below covers the {len(filled)} that are labelled; blanks "
              f"are excluded rather than counted as disagreement")

    print("\n" + "=" * 70)
    print("PRE-LABEL vs HUMAN AGREEMENT   (per family, treated as binary)")
    print("=" * 70)
    print(f"  {'family':26s} {'both':>5s} {'human':>6s} {'script':>7s} {'agree':>7s} {'kappa':>7s}")
    for fam in FAMILIES:
        h = [fam in x["labels"] for x in filled]
        s = [fam in pre[x["_key"]]["prelabels"] for x in filled]
        both = sum(a and b for a, b in zip(h, s))
        agree = sum(a == b for a, b in zip(h, s)) / len(filled)
        k = cohens_kappa(h, s)
        ks = f"{k:7.2f}" if k is not None else "      —"
        tag = ("  (partial)" if fam in PARTIAL else "  (human only)") if fam in NEEDS_HUMAN else ""
        print(f"  {fam:26s} {both:5d} {sum(h):6d} {sum(s):7d} {agree:6.0%} {ks}{tag}")

    # ---- multi-label agreement ------------------------------------------
    # Assessment 6: report Jaccard rather than leaning on kappa, because the
    # labelling is multi-label and the classes are sparse. Under sparse
    # multi-label coding kappa is dominated by agreement on ABSENT families,
    # which inflates it; Jaccard only counts families at least one rater used.
    jac, exact, partial, none_ = [], 0, 0, 0
    for x in filled:
        a = set(x["labels"]) - {"ambiguous"}
        b = set(pre[x["_key"]]["prelabels"])
        u = a | b
        j = len(a & b) / len(u) if u else 1.0
        jac.append(j)
        if j == 1.0:
            exact += 1
        elif j > 0:
            partial += 1
        else:
            none_ += 1

    print("\n" + "=" * 70)
    print("MULTI-LABEL AGREEMENT   (Jaccard over label sets, per program)")
    print("=" * 70)
    print(f"  mean Jaccard          {sum(jac)/len(jac):.2f}")
    print(f"  identical label sets  {exact:3d} / {len(filled)}")
    print(f"  partial overlap       {partial:3d}")
    print(f"  disjoint              {none_:3d}")
    print("\n  Five families are human-only by design (the script does not attempt")
    print("  F3, F4, F5, F7, and attempts only one instance of F1), so Jaccard here")
    print("  has a ceiling well below 1.0 and is NOT a reliability estimate. It is")
    print("  a measure of how much of a human reading the syntactic rules recover.")

    amb = sum(1 for x in filled if "ambiguous" in x["labels"])
    ovn = sum(1 for x in filled if "over_noising" in x.get("markers", []))
    print(f"\n  ambiguous      {amb}")
    print(f"  over_noising   {ovn}")
    print("\nKappa is undefined (-) where a rater never varies; that is no information,")
    print("not disagreement. Under sparse multi-label coding kappa is also inflated by")
    print("agreement on absent families, so Jaccard above is the headline statistic.")


if __name__ == "__main__":
    main()
