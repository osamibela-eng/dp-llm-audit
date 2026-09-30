"""Independent cross-check of the paper's numbers against paper/macros.tex.

Deliberately does NOT import any analysis module. It re-derives the headline
figures from the outcome and confirmation files directly, then compares them to
the macros the paper actually typesets. If the generator and this file agree, two
independent implementations agree; if they disagree, one of them is wrong and the
paper cannot be trusted until it is resolved.

Assessment 6 asked to "verify all denominators and confidence intervals". This is
that check, in executable form rather than as a claim that it was done.

    python analysis/verify_paper_numbers.py
"""
from __future__ import annotations

import collections
import glob
import json
import math
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOT_EXECUTABLE = {"syntax_fail", "screen_fail", "runtime_fail", "timeout"}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def pct(k, n):
    return f"{k/n:.0%}".replace("%", r"\%")


def ci(k, n):
    lo, hi = wilson(k, n)
    return f"[{lo:.2f}, {hi:.2f}]"


def main():
    rows, seen, leaked = [], set(), 0
    for f in sorted(glob.glob(str(ROOT / "results" / "processed" / "outcomes_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            # Defence in depth. Repaired programs live in repair_outcomes.jsonl,
            # which no longer matches outcomes_*.jsonl - but if a future file is
            # named back into the glob, this catches it instead of silently
            # inflating every rate in the paper. That is not hypothetical: it
            # happened once, and this checker missed it because it shared the
            # generator's glob. A cross-check that reuses the suspect assumption
            # is not a cross-check.
            if r.get("repair_arm") or r.get("parent_sha256"):
                leaked += 1
                continue
            k = (r["model"], r["task_id"], r["sample_idx"])
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)

    if leaked:
        print(f"  !! {leaked} repair records matched the main-results glob and were "
              f"excluded. Fix the filename; do not rely on this filter.\n")

    # The corpus size is stated here as its three factors rather than as a bare
    # total, so that changing it forces you to say WHICH factor changed. E4
    # raises SAMPLES from 5 to 10; when that lands, edit SAMPLES and nothing
    # else. Editing the total to match whatever the data happens to contain
    # would turn this check into a no-op, which is the failure mode it exists
    # to prevent.
    # E4 raised samples per task from 5 to 10. Changed deliberately, on
    # 2026-08-24, after the generation finished; the corpus is now 480 programs.
    # The 240-program corpus that the earlier submission reports is preserved in
    # results/processed/*.bak240 and must not be mixed with this one.
    MODELS, TASKS, SAMPLES = 3, 16, 10
    EXPECTED = MODELS * TASKS * SAMPLES
    if len(rows) != EXPECTED:
        print(f"FATAL: main results hold {len(rows)} programs, design says "
              f"{EXPECTED} ({MODELS} models x {TASKS} tasks x {SAMPLES} samples).")
        print("Something is being counted that should not be, or generation is "
              "incomplete. Every rate below would be wrong.")
        if len(rows) % (MODELS * TASKS) == 0:
            print(f"Note: {len(rows)} is exactly {len(rows) // (MODELS * TASKS)} "
                  f"samples per model-task. If that is intended, set SAMPLES to "
                  f"it deliberately and regenerate paper/macros.tex, because "
                  f"every macro in the paper is keyed to this corpus.")
        sys.exit(1)

    c = collections.Counter(r["outcome"] for r in rows)
    gen = len(rows)
    ex = gen - sum(c[o] for o in NOT_EXECUTABLE)
    fn = ex - c["semantic_fail"]
    aud = c["falsified"] + c["not_falsified"]
    fals = c["falsified"]

    conf = [json.loads(l) for l in
            (ROOT / "results" / "processed" / "confirm_seeds.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]

    # Restrict to confirmation records whose program is still falsified in THIS
    # script's own reading of the corpus. Derived here from `rows` rather than
    # imported from the generator, so this stays an independent computation: if
    # the two disagree about which programs are live, the assertion below fires.
    live = {r["code_sha256"] for r in rows if r["outcome"] == "falsified"}
    conf = [r for r in conf if r["code_sha256"] in live]

    repro = sum(1 for r in conf if r["classification"] == "reproducible")
    oneoff = sum(1 for r in conf if r["classification"] == "one_off")

    src = (ROOT / "paper" / "macros.tex").read_text(encoding="utf-8")
    mac = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", src))

    checks = [
        ("Ngenerated", gen), ("Nexecutable", ex), ("Nfunctional", fn),
        ("Naudited", aud), ("Nfalsified", fals),
        ("Nreproducible", repro), ("Noneoff", oneoff),
        ("FalsRateGen", pct(fals, gen)), ("FalsCIGen", ci(fals, gen)),
        ("FalsRateExec", pct(fals, ex)), ("FalsCIExec", ci(fals, ex)),
        ("FalsRateFunc", pct(fals, fn)), ("FalsCIFunc", ci(fals, fn)),
        ("FalsRateAud", pct(fals, aud)), ("FalsCIAud", ci(fals, aud)),
        ("ReproFalsRateFunc", pct(repro, fn)), ("ReproFalsCIFunc", ci(repro, fn)),
        ("ReproFalsRateAud", pct(repro, aud)), ("ReproFalsCIAud", ci(repro, aud)),
    ]

    # per model, including the pair that produces the ordering inversion
    tex = {"qwen2.5-coder:7b": "Qwen", "deepseek-coder:6.7b": "Deepseek",
           "codellama:7b": "Codellama"}
    spread = {"Gen": [], "Exec": []}
    for model, tag in tex.items():
        sub = [r for r in rows if r["model"] == model]
        cm = collections.Counter(r["outcome"] for r in sub)
        mex = len(sub) - sum(cm[o] for o in NOT_EXECUTABLE)
        mf = cm["falsified"]
        checks += [(f"{tag}FalsGen", pct(mf, len(sub))),
                   (f"{tag}FalsExec", pct(mf, mex)),
                   (f"{tag}ExecRate", pct(mex, len(sub)))]
        spread["Gen"].append(mf / len(sub))
        spread["Exec"].append(mf / mex)
    for den, vals in spread.items():
        checks.append((f"Spread{den}", f"{(max(vals)-min(vals))*100:.0f}"))

    bad = []
    for name, val in checks:
        got = mac.get(name)
        ok = str(got) == str(val)
        if not ok:
            bad.append(name)
        print(f"  {'ok  ' if ok else 'BAD '} {name:22s} independent={val!s:16s} macro={got}")

    print()
    if bad:
        print(f"{len(bad)} MISMATCH(ES): {', '.join(bad)}")
        print("The paper typesets a number this file does not reproduce. Resolve")
        print("before the draft goes anywhere.")
        sys.exit(1)
    print(f"all {len(checks)} checked values agree between two independent computations")

    # Sanity assertions that do not depend on the macros at all.
    assert ex <= gen and fn <= ex and aud <= fn, "waterfall is not monotone"
    assert repro + oneoff <= fals, "more confirmations than falsifications"
    order_gen = sorted(tex.values(), key=lambda t: float(mac[f"{t}FalsGen"].rstrip("\\%")))
    order_exec = sorted(tex.values(), key=lambda t: float(mac[f"{t}FalsExec"].rstrip("\\%")))
    print(f"\n  ordering per generated  : {' < '.join(order_gen)}")
    print(f"  ordering per executable : {' < '.join(order_exec)}")
    print(f"  inverted for codellama  : "
          f"{order_gen[0] == 'Codellama' and order_exec[-1] == 'Codellama'}")


if __name__ == "__main__":
    main()
