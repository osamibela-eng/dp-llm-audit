"""Build every number, table and figure of the revised paper from the result files.

Nothing in the paper is typed by hand. This script reads results/processed/,
computes each reported value, and writes

    <paper>/numbers.tex     \\newcommand macros, one per reported value
    <paper>/tab_*.tex       tables
    <paper>/figures/*.pdf   figures

Experiments that have not produced output yet are skipped with a warning and
their macros are left undefined, so LaTeX fails loudly instead of printing a
stale value.

    python analysis/make_v3_assets.py --paper "C:/Users/Basic/Documents/Editing Paper/Revised Paper"
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import pathlib
import statistics

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "results" / "processed"

LOCAL = ["qwen2.5-coder:7b", "deepseek-coder:6.7b", "codellama:7b"]
PRETTY = {"qwen2.5-coder:7b": "Qwen2.5-Coder 7B", "deepseek-coder:6.7b": "DeepSeek-Coder 6.7B",
          "codellama:7b": "Code Llama 7B", "gemini-3.6-flash": "Gemini 3.6 Flash",
          "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite", "gemma-4-31b-it": "Gemma 4 31B",
          "gemma-4-26b-a4b-it": "Gemma 4 26B-A4B"}
HOSTED = ["gemini-3.1-flash-lite", "gemma-4-26b-a4b-it", "gemma-4-31b-it", "gemini-3.6-flash"]
EXEC_FAIL = {"syntax_fail", "runtime_fail", "timeout", "screen_fail"}
AUDITED = {"falsified", "not_falsified"}
FAMILIES = ["F1_sensitivity", "F2_calibration", "F3_adjacency", "F4_composition",
            "F5_selection", "F6_data_dependent_flow", "F7_boundary", "F8_randomness",
            "F9_api_misuse", "F10_claim_mismatch", "ambiguous"]
FAMILY_NAME = {"F1_sensitivity": "Sensitivity", "F2_calibration": "Noise calibration",
               "F3_adjacency": "Adjacency", "F4_composition": "Composition",
               "F5_selection": "Selection", "F6_data_dependent_flow": "Data-dependent control flow",
               "F7_boundary": "Boundary input", "F8_randomness": "Randomness misuse",
               "F9_api_misuse": "API misuse", "F10_claim_mismatch": "Claim mismatch",
               "ambiguous": "Ambiguous"}

MACROS: dict[str, str] = {}


def m(name: str, value) -> None:
    if not name.isalpha():
        raise SystemExit(f"macro name {name!r} is not letters-only; LaTeX would reject it")
    if name in MACROS:
        raise SystemExit(f"macro {name} defined twice")
    MACROS[name] = str(value)


def pct(k: int, n: int) -> str:
    return f"{round(100 * k / n)}\\%" if n else "--"


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def ci(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"[{lo:.2f}, {hi:.2f}]"


def jl(path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_corpus(pattern: str) -> list[dict]:
    rows, seen = [], set()
    for f in sorted(glob.glob(str(pattern))):
        for r in jl(f):
            k = (r["model"], r["task_id"], str(r["sample_idx"]))
            if k not in seen:
                seen.add(k)
                rows.append(r)
    return rows


def confirmations(path) -> dict:
    out = {}
    if pathlib.Path(path).exists():
        for c in jl(path):
            out[(c["model"], c["task_id"], str(c["sample_idx"]))] = c["classification"]
    return out


def stage_counts(rows, conf):
    g = len(rows)
    ex = sum(r["outcome"] not in EXEC_FAIL for r in rows)
    fn = sum(r["outcome"] not in EXEC_FAIL and r["outcome"] != "semantic_fail" for r in rows)
    au = sum(r["outcome"] in AUDITED for r in rows)
    fa = sum(r["outcome"] == "falsified" for r in rows)
    rp = sum(r["outcome"] == "falsified"
             and conf.get((r["model"], r["task_id"], str(r["sample_idx"]))) == "reproducible"
             for r in rows)
    return dict(gen=g, exe=ex, fun=fn, aud=au, fal=fa, rep=rp)


# ----------------------------------------------------------------------------- sections

def main_corpus(paper: pathlib.Path, tasks: dict):
    rows = load_corpus(P / "outcomes_*.jsonl")
    conf = confirmations(P / "confirm_seeds.jsonl")
    s = stage_counts(rows, conf)
    if (s["gen"], s["aud"], s["fal"], s["rep"]) != (480, 284, 184, 160):
        raise SystemExit(f"main corpus does not match the verified totals: {s}")
    m("NGen", s["gen"]); m("NExec", s["exe"]); m("NFunc", s["fun"]); m("NAud", s["aud"])
    m("NFals", s["fal"]); m("NRepro", s["rep"]); m("NOneOff", s["fal"] - s["rep"])
    m("NNotExec", s["gen"] - s["exe"])
    m("ReproRateAud", pct(s["rep"], s["aud"])); m("ReproCIAud", ci(s["rep"], s["aud"]))
    m("InitRateAud", pct(s["fal"], s["aud"])); m("InitCIAud", ci(s["fal"], s["aud"]))
    m("ReproRateGen", pct(s["rep"], s["gen"])); m("ReproCIGen", ci(s["rep"], s["gen"]))
    m("ExecRate", pct(s["exe"], s["gen"]))
    oc = collections.Counter(r["outcome"] for r in rows)
    m("NRuntimeFail", oc["runtime_fail"]); m("NSyntaxFail", oc["syntax_fail"])
    m("NAuditorError", oc["auditor_error"]); m("NAuditorUnsupported", oc["auditor_unsupported"])
    m("NSemanticFail", oc["semantic_fail"])

    gauss = [r for r in rows if r["task_id"] == "gaussian_bounded_count_v1"]
    g = stage_counts(gauss, conf)
    m("NGaussAud", g["aud"]); m("NGaussFals", g["fal"]); m("NGaussRepro", g["rep"])
    faithful_aud = s["aud"] - g["aud"]
    m("NFaithfulAud", faithful_aud)
    m("ReproRateFaithful", pct(s["rep"] - g["rep"], faithful_aud))
    m("ReproCIFaithful", ci(s["rep"] - g["rep"], faithful_aud))

    # per model
    lines = []
    for mod in LOCAL:
        c = stage_counts([r for r in rows if r["model"] == mod], conf)
        key = {"qwen2.5-coder:7b": "Qwen", "deepseek-coder:6.7b": "Deepseek", "codellama:7b": "Codellama"}[mod]
        m(f"N{key}Exec", c["exe"]); m(f"N{key}Aud", c["aud"]); m(f"N{key}Repro", c["rep"])
        m(f"{key}ReproAud", pct(c["rep"], c["aud"])); m(f"{key}ReproGen", pct(c["rep"], c["gen"]))
        lines.append(f"{PRETTY[mod]} & {c['gen']} & {c['exe']} & {c['aud']} & {c['fal']} & "
                     f"{c['rep']} & {pct(c['rep'], c['aud'])} {ci(c['rep'], c['aud'])} \\\\")
    lines.append("\\midrule")
    lines.append(f"All three & {s['gen']} & {s['exe']} & {s['aud']} & {s['fal']} & {s['rep']} & "
                 f"{pct(s['rep'], s['aud'])} {ci(s['rep'], s['aud'])} \\\\")
    (paper / "tab_models_local.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # per task
    tl = []
    tier_prev = None
    for tid, t in tasks.items():
        c = stage_counts([r for r in rows if r["task_id"] == tid], conf)
        if tier_prev is not None and t["tier"] != tier_prev:
            tl.append("\\addlinespace")
        tier_prev = t["tier"]
        eps = t["claimed_epsilon"]
        eps_s = "$\\ln 3$" if abs(eps - math.log(3)) < 1e-9 else f"{eps:g}"
        if t.get("claimed_delta"):
            eps_s += ", $\\delta{=}10^{-5}$"
        adj = "add/rem." if t["adjacency"] == "add_remove_one" else "replace"
        name = tid.replace("_v1", "").replace("_", "\\_")
        rate = pct(c["rep"], c["aud"]) if c["aud"] else "--"
        tl.append(f"\\texttt{{{name}}} & {t['tier']} & {eps_s} & {adj} & {c['exe']} & {c['aud']} & "
                  f"{c['rep']} & {rate} \\\\")
    (paper / "tab_tasks.tex").write_text("\n".join(tl) + "\n", encoding="utf-8")

    # violation strength
    rep = [r for r in rows if r["outcome"] == "falsified"
           and conf.get((r["model"], r["task_id"], str(r["sample_idx"]))) == "reproducible"]
    lbs = [r["counterexample"]["empirical_eps_lower_bound"] for r in rep]
    support = sum(r["counterexample"]["p2_upper_bound"] < 1e-3 for r in rep)
    twice = sum(r["counterexample"]["empirical_eps_lower_bound"] >= 2 * r["falsified_at_eps"] for r in rep)
    m("NSupport", support); m("SupportRate", pct(support, len(rep)))
    m("MedianEpsLB", f"{statistics.median(lbs):.1f}")
    m("NTwiceClaim", twice)
    m("NReproWide", sum(r["counterexample"]["empirical_eps_lower_bound"] - r["falsified_at_eps"] >= 0.5
                        for r in rep))
    m("MinEpsLB", f"{min(lbs):.2f}")
    one_off = collections.Counter(r["task_id"] for r in rows if r["outcome"] == "falsified"
                                  and conf.get((r["model"], r["task_id"], str(r["sample_idx"]))) == "one_off")
    m("NOneOffGauss", one_off["gaussian_bounded_count_v1"])
    return rows, conf, rep, lbs


def validation(paper, tasks):
    refs = jl(P / "e1_references.jsonl")
    m("NRefs", len(refs))
    m("NRefFalseOurs", sum(r["ours"]["status"] == "falsified" for r in refs))
    m("NRefFalseStatDP", sum(r["generic"]["status"] == "falsified" for r in refs))
    m("NRefUnsupStatDP", sum(r["generic"]["status"] == "unsupported" for r in refs))
    m("NRefErrOurs", sum(r["ours"]["status"] not in ("falsified", "not_falsified") for r in refs))

    pl = jl(P / "e1_probe_log.jsonl")
    d = {}
    for p in pl:
        d[(p["task"], json.dumps(p["probe"]))] = p["violations"]
    m("NProbes", len(d)); m("NProbesBad", sum(1 for v in d.values() if v))
    kinds = collections.Counter(v["kind"] for vs in d.values() for v in vs)
    m("NProbeType", kinds["record_type"]); m("NProbeValue", kinds["value_domain"]); m("NProbeSize", kinds["size"])

    bugs = [b for b in jl(P / "e1_bugs.jsonl") if b["kind"] == "bug"]
    det = [b for b in bugs if b["expected"] == "detectable"]
    lm = [b for b in bugs if b["expected"] == "likely_missed"]
    caught = sum(b["ours"]["status"] == "falsified" for b in det)
    err = sum(b["ours"]["status"] not in ("falsified", "not_falsified") for b in det)
    m("NBugs", len(bugs)); m("NBugsDet", len(det)); m("NBugsLM", len(lm))
    m("NBugsCaught", caught); m("NBugsErr", err); m("NBugsMissed", len(det) - caught - err)
    m("PowerDet", pct(caught, len(det))); m("PowerDetCI", ci(caught, len(det)))
    m("NBugsVerdict", len(det) - err)
    m("NBugsFailed", len(det) - caught)
    m("PowerVerdict", pct(caught, len(det) - err)); m("PowerVerdictCI", ci(caught, len(det) - err))
    m("NBugsLMCaught", sum(b["ours"]["status"] == "falsified" for b in lm))
    tier = collections.defaultdict(lambda: [0, 0])
    for b in det:
        t = tasks[b["task"]]["tier"]
        tier[t][0] += 1
        tier[t][1] += b["ours"]["status"] == "falsified"
    for t, name in ((1, "One"), (2, "Two"), (3, "Three")):
        m(f"NBugsTier{name}", tier[t][0]); m(f"NBugsTier{name}Caught", tier[t][1])
        m(f"NBugsTier{name}Missed", tier[t][0] - tier[t][1])

    e2 = jl(P / "e2_random_pairs.jsonl")
    conf = {c["code_sha256"]: c["classification"] for c in jl(P / "confirm_seeds.jsonl")}
    rep = [r for r in e2 if r["group"] == "falsified" and conf.get(r["code_sha256"]) == "reproducible"]
    one = [r for r in e2 if r["group"] == "falsified" and conf.get(r["code_sha256"]) == "one_off"]
    nf = [r for r in e2 if r["group"] == "not_falsified"]
    hit = sum(r["hits"] > 0 for r in rep)
    m("NEtwoRepro", len(rep)); m("NEtwoHit", hit); m("EtwoRate", pct(hit, len(rep)))
    m("EtwoCI", ci(hit, len(rep))); m("NEtwoAll", sum(r["hits"] == r["replicates"] for r in rep))
    m("NEtwoOneOff", len(one)); m("NEtwoOneOffHit", sum(r["hits"] > 0 for r in one))
    m("NEtwoNF", len(nf)); m("NEtwoNFHit", sum(r["hits"] > 0 for r in nf))
    m("NEtwoPrograms", len(e2))


def repair():
    e6 = json.loads((P / "e6_repair_arms.json").read_text())
    n = e6["paired_programs"]
    m("NRepair", n)
    for arm, word in (("R0", "Zero"), ("R2", "Two"), ("R3", "Three")):
        k = e6["repaired"][arm]
        m(f"Rep{word}", k); m(f"Rep{word}Rate", pct(k, n)); m(f"Rep{word}CI", ci(k, n))
    e7 = json.loads((P / "e7_paired.json").read_text())
    fb, ct = e7["arms"]["R2 (feedback)"], e7["arms"]["R0 (control)"]
    m("MultiFb", fb["repaired"]); m("MultiCt", ct["repaired"])
    m("MultiRate", pct(fb["repaired"], n)); m("MultiCI", ci(fb["repaired"], n))
    m("MultiOnlyFb", e7["mcnemar"]["only_feedback"]); m("MultiOnlyCt", e7["mcnemar"]["only_control"])
    for arm, tag in ((fb, "Fb"), (ct, "Ct")):
        for a in ("1", "2", "3"):
            tried, rep = arm["per_attempt"][a]
            m(f"Att{tag}{['','One','Two','Three'][int(a)]}", f"{rep}/{tried}")
    dead = ("syntax_fail", "runtime_fail", "unsupported", "error")   # as in make_paper_macros.py
    m("MultiDeadFb", sum(r["outcome"] in dead for r in jl(P / "e7_multi_round_local.jsonl")))
    m("MultiDeadCt", sum(r["outcome"] in dead for r in jl(P / "e7_multi_round_local_r0.jsonl")))
    e8b = P / "e8b_summary.json"
    if e8b.exists():
        s = json.loads(e8b.read_text())
        meas = s["measured"]
        n = sum(meas.values())
        m("NEeightbTotal", n)
        m("NEeightbCorrect", meas.get("correct", 0))
        m("NEeightbMemorised", meas.get("memorised", 0))
        m("NEeightbOver", meas.get("over_noised", 0))
        m("NEeightbUnder", meas.get("under_noised", 0))
        m("NEeightbUnmeasurable", meas.get("unmeasurable", 0))
        m("NEeightbOther", meas.get("other", 0))
        m("NEeightbMeasured", n - meas.get("unmeasurable", 0))
        a2 = s["by_arm"].get("A2", {})
        m("NEeightbFals", a2.get("falsified", 0))
        m("NEeightbAud", a2.get("falsified", 0) + a2.get("not_falsified", 0))
    else:
        print("  [skip] E8b (local-model perturbation) not finished")
    e8 = json.loads((P / "e8_summary.json").read_text())
    m("NEeightAOne", sum(e8["by_arm"]["A1"].values())); m("NEeightATwo", sum(e8["by_arm"]["A2"].values()))
    m("NEeightCorrect", e8["measured"].get("correct", 0))
    m("EeightCI", ci(e8["measured"].get("correct", 0), sum(e8["by_arm"]["A2"].values())))


def hosted(paper):
    """Hosted models: E5 (gemini-3.6-flash) plus E12 when its outcome files exist."""
    lines, avail = [], []
    for mod in HOSTED:
        if mod == "gemini-3.6-flash":
            rows = jl(P / "e5_frontier_outcomes.jsonl")
            conf = {}
        else:
            f = P / f"e12_outcomes_{mod}.jsonl"
            if not f.exists():
                print(f"  [skip] no outcomes for {mod}")
                continue
            rows = jl(f)
            conf = confirmations(P / f"e12_confirm_{mod}.jsonl")
        c = stage_counts(rows, conf)
        if c["fal"] and not conf:
            print(f"  [warn] {mod}: {c['fal']} detections without confirmation records")
        avail.append((mod, c))
        key = {"gemini-3.6-flash": "GemFlash", "gemini-3.1-flash-lite": "GemLite",
               "gemma-4-31b-it": "GemmaDense", "gemma-4-26b-a4b-it": "GemmaMoE"}[mod]
        m(f"N{key}Gen", c["gen"]); m(f"N{key}Aud", c["aud"]); m(f"N{key}Repro", c["rep"])
        m(f"N{key}Fals", c["fal"])
        lines.append(f"{PRETTY[mod]} & {c['gen']} & {c['exe']} & {c['aud']} & {c['fal']} & {c['rep']} & "
                     f"{pct(c['rep'], c['aud'])} {ci(c['rep'], c['aud'])} \\\\")
    (paper / "tab_models_hosted.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    m("NHostedModels", len(avail))
    m("NHostedAud", sum(c["aud"] for _, c in avail))
    m("NHostedRepro", sum(c["rep"] for _, c in avail))
    m("NHostedGen", sum(c["gen"] for _, c in avail))
    return avail


def prior_bugs(paper):
    f = P / "e9_prior_bugs.jsonl"
    if not f.exists():
        print("  [skip] E9 not run"); return None
    rows = jl(f)
    if len(rows) < 33:
        print(f"  [skip] E9 incomplete ({len(rows)}/33)"); return None
    viol = [r for r in rows if r["truly_violates"]]
    ok = [r for r in rows if not r["truly_violates"]]
    m("NPriorCases", len(rows)); m("NPriorViol", len(viol)); m("NPriorOk", len(ok))
    m("NPriorOursTP", sum(r["ours"]["reproducible"] for r in viol))
    m("NPriorOursFP", sum(r["ours"]["reproducible"] for r in ok))
    m("NPriorOursInitFP", sum(r["ours"]["campaign"] == "falsified" for r in ok))
    m("NPriorStatTP", sum(r["statdp"]["falsified"] for r in viol))
    m("NPriorStatFP", sum(r["statdp"]["falsified"] for r in ok))
    lines = []
    for name in dict.fromkeys(r["algorithm"] for r in rows):
        rs = [r for r in rows if r["algorithm"] == name]
        cells = []
        for r in rs:
            o = "\\ding{51}" if r["ours"]["reproducible"] else "\\ding{55}"
            s = "\\ding{51}" if r["statdp"]["falsified"] else "\\ding{55}"
            truth = "V" if r["truly_violates"] else "ok"
            cells.append(f"{truth} & {o} & {s}")
        defect = rs[0]["defect"] or "--- (correct)"
        lines.append(f"\\texttt{{{name.replace('_', chr(92) + '_')}}} & {defect} & " + " & ".join(cells) + " \\\\")
    (paper / "tab_prior.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def cross_auditor():
    f = P / "e10_cross_auditor.jsonl"
    if not f.exists():
        print("  [skip] E10 not run"); return
    rows = jl(f)
    # Compare against the label the paper reports (confirmed violation), not the
    # initial detection: one-off detections are not counted as violations anywhere.
    conf = confirmations(P / "confirm_seeds.jsonl")
    for r in rows:
        k = (r["model"], r["task_id"], str(r["sample_idx"]))
        r["ours"] = "falsified" if conf.get(k) == "reproducible" else "not_falsified"
    both = [r for r in rows if r["statdp"] in AUDITED]
    c = collections.Counter((r["ours"], r["statdp"]) for r in both)
    m("NCrossAll", len(rows)); m("NCrossBoth", len(both))
    m("NCrossUnsup", sum(r["statdp"] == "unsupported" for r in rows))
    m("NCrossFF", c[("falsified", "falsified")]); m("NCrossFN", c[("falsified", "not_falsified")])
    m("NCrossNF", c[("not_falsified", "falsified")]); m("NCrossNN", c[("not_falsified", "not_falsified")])
    agree = c[("falsified", "falsified")] + c[("not_falsified", "not_falsified")]
    m("CrossAgree", pct(agree, len(both)))
    n = len(both)
    po = agree / n if n else 0
    pa = ((c[("falsified", "falsified")] + c[("falsified", "not_falsified")]) / n) if n else 0
    pb = ((c[("falsified", "falsified")] + c[("not_falsified", "falsified")]) / n) if n else 0
    pe = pa * pb + (1 - pa) * (1 - pb)
    m("CrossKappa", f"{(po - pe) / (1 - pe):.2f}" if n and pe < 1 else "--")


def taxonomy(paper):
    parts = sorted(glob.glob(str(P / "e13_labels_claude_part*.jsonl")))
    gem = P / "e13_labels_gemini-3.8-flash.jsonl"
    if len(parts) < 4 or not gem.exists():
        print("  [skip] E13 incomplete"); return None
    A = {r["item"]: set(r["labels"] or []) for f in parts for r in jl(f)}
    B = {r["item"]: set(r["labels"] or []) for r in jl(gem) if r.get("labels")}
    items = sorted(set(A) & set(B))
    if len(items) < 160:
        print(f"  [skip] E13 incomplete ({len(items)}/160 labelled by both)"); return None
    m("NTaxItems", len(items))
    res = {}
    kappas = []
    for fam in FAMILIES:
        a = [fam in A[i] for i in items]
        b = [fam in B[i] for i in items]
        n = len(items)
        na, nb = sum(a), sum(b)
        both = sum(x and y for x, y in zip(a, b))
        po = sum(x == y for x, y in zip(a, b)) / n
        pe = (na / n) * (nb / n) + (1 - na / n) * (1 - nb / n)
        kap = (po - pe) / (1 - pe) if pe < 1 else float("nan")
        res[fam] = dict(a=na, b=nb, both=both, kappa=kap, po=po)
        if na + nb >= 10:
            kappas.append(kap)
    exact = sum(A[i] == B[i] for i in items)
    m("TaxExact", exact); m("TaxExactRate", pct(exact, len(items)))
    lines = []
    for fam in FAMILIES:
        r = res[fam]
        if r["a"] + r["b"] == 0:
            continue
        k = "--" if math.isnan(r["kappa"]) or r["a"] + r["b"] < 10 else f"{r['kappa']:.2f}"
        lines.append(f"{FAMILY_NAME[fam]} & {r['a']} & {r['b']} & {r['both']} & {k} \\\\")
    (paper / "tab_taxonomy.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for fam, tag in (("F2_calibration", "Calib"), ("F6_data_dependent_flow", "Flow"),
                     ("F1_sensitivity", "Sens"), ("F10_claim_mismatch", "Claim"),
                     ("F8_randomness", "Rand"), ("F4_composition", "Comp")):
        m(f"Tax{tag}Both", res[fam]["both"]); m(f"Tax{tag}Kappa", f"{res[fam]['kappa']:.2f}")
        m(f"Tax{tag}A", res[fam]["a"]); m(f"Tax{tag}B", res[fam]["b"])
    m("TaxMedianKappa", f"{statistics.median(kappas):.2f}")
    # disagreement sheet for the author's adjudication
    key = {r["item"]: r for r in jl(P / "e13_worksheet_key.jsonl")}
    with open(P / "e13_adjudication_sheet.jsonl", "w", encoding="utf-8") as fh:
        for i in items:
            if A[i] != B[i]:
                fh.write(json.dumps({"item": i, "task_id": key[i]["task_id"],
                                     "annotator_claude": sorted(A[i]),
                                     "annotator_gemini": sorted(B[i]),
                                     "adjudicated": None}) + "\n")
    return res, A, B, items


# ----------------------------------------------------------------------------- figures

INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
NEUTRAL, NEUTRAL2 = "#d6d5ce", "#ecebe6"


def style(plt):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5,
                         "axes.edgecolor": BASE, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": INK2, "axes.linewidth": 0.8,
                         "pdf.fonttype": 42})


def fig_waterfall(fig_dir, rows, conf, hosted_rows):
    import matplotlib.pyplot as plt
    style(plt)
    groups = [(PRETTY[mm], [r for r in rows if r["model"] == mm], conf) for mm in LOCAL]
    groups += hosted_rows
    labels, segs = [], []
    for name, rs, cf in groups:
        c = stage_counts(rs, cf)
        n = c["gen"]
        noexec = n - c["exe"]
        notaud = c["exe"] - c["aud"]
        notfal = c["aud"] - c["fal"]
        unstable = c["fal"] - c["rep"]
        segs.append([c["rep"] / n, (notfal + unstable) / n, notaud / n, noexec / n])
        labels.append(f"{name}  (n={n})")
    cols = [S2, S1, NEUTRAL, NEUTRAL2]
    names = ["Reproducibly falsified", "Audited, no violation found", "Ran, not auditable",
             "Did not run"]
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(labels) + 0.9))
    y = list(range(len(labels)))[::-1]
    for yi, sg in zip(y, segs):
        left = 0
        for j, (w, col) in enumerate(zip(sg, cols)):
            if w <= 0:
                continue
            ax.barh(yi, w - 0.004, left=left + 0.002, color=col, height=0.62, linewidth=0)
            if j == 0 and w >= 0.06:
                ax.text(left + w / 2, yi, f"{round(100 * w)}%", ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
            left += w
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of generated programs")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    if len(labels) > 3:
        ax.axhline(y[2] - 0.5, color=BASE, lw=0.8, ls=(0, (2, 2)))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cols]
    ax.legend(handles, names, ncol=4, frameon=False, fontsize=7.5, loc="lower center",
              bbox_to_anchor=(0.40, 1.0), handlelength=1.0, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_outcomes.pdf")
    plt.close(fig)


def fig_severity(fig_dir, lbs):
    import matplotlib.pyplot as plt
    style(plt)
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    bins = [x * 0.5 for x in range(0, 19)]
    ax.hist(lbs, bins=bins, color=S2, edgecolor="white", linewidth=1.0)
    ax.axvline(1.0, color=INK, lw=1.0, ls=(0, (3, 2)))
    ax.text(1.12, ax.get_ylim()[1] * 0.9, "claimed ε = 1", color=INK2, fontsize=7.5)
    ax.set_xlabel("certified lower bound on ε")
    ax.set_ylabel("programs")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_severity.pdf")
    plt.close(fig)


def fig_pipeline(fig_dir):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    style(plt)
    fig, ax = plt.subplots(figsize=(7.4, 1.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 22); ax.axis("off")
    boxes = [("Task +\ncontract", "domain, adjacency,\noutput, (ε, δ)"),
             ("Generator", "7 LLMs, one\nprompt, T = 0.8"),
             ("Functional\ngate", "runs and is\nnon-degenerate"),
             ("Audit", "contract-checked\npairs, CP bounds"),
             ("Confirm", "3 fresh-seed\nre-audits"),
             ("Verdict", "witness D, D′,\nevent, ε bound")]
    w, gap, x = 14.0, 3.0, 0.5
    for i, (t, sub) in enumerate(boxes):
        col = "#eef4fc" if i in (3, 4) else "#f4f3ef"
        ax.add_patch(FancyBboxPatch((x, 5), w, 14, boxstyle="round,pad=0.2,rounding_size=1.2",
                                    fc=col, ec=BASE, lw=0.8))
        ax.text(x + w / 2, 14.8, t, ha="center", va="center", fontsize=7.8, color=INK, fontweight="bold")
        ax.text(x + w / 2, 8.8, sub, ha="center", va="center", fontsize=6.4, color=INK2, linespacing=1.3)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + w + gap - 0.3, 12), xytext=(x + w + 0.3, 12),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))
        x += w + gap
    ax.text(50, 1.5, "Validated separately: correct references (false alarms) · planted and prior-work bugs (power) · "
                     "random contract pairs · second auditor", ha="center", fontsize=6.6, color=INK2, style="italic")
    fig.savefig(fig_dir / "fig_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_taxonomy(fig_dir, res):
    import matplotlib.pyplot as plt
    style(plt)
    fams = [f for f in FAMILIES if res[f]["a"] + res[f]["b"] > 0]
    fams.sort(key=lambda f: res[f]["a"] + res[f]["b"])
    fig, ax = plt.subplots(figsize=(4.3, 2.9))
    y = range(len(fams))
    h = 0.36
    ax.barh([i + h / 2 + 0.02 for i in y], [res[f]["a"] for f in fams], height=h, color=S1, label="Annotator A (Claude)")
    ax.barh([i - h / 2 - 0.02 for i in y], [res[f]["b"] for f in fams], height=h, color=S2, label="Annotator B (Gemini)")
    ax.set_yticks(list(y)); ax.set_yticklabels([FAMILY_NAME[f] for f in fams], fontsize=7.5)
    ax.set_xlabel("programs with this label (of 160)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_taxonomy.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True)
    a = ap.parse_args()
    paper = pathlib.Path(a.paper)
    (paper / "figures").mkdir(parents=True, exist_ok=True)
    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml").read_text(encoding="utf-8"))["tasks"]}
    m("NTasks", len(tasks))
    rows, conf, rep, lbs = main_corpus(paper, tasks)
    validation(paper, tasks)
    repair()
    avail = hosted(paper)
    prior_bugs(paper)
    cross_auditor()
    tax = taxonomy(paper)

    hosted_rows = []
    for mod, _ in avail:
        if mod == "gemini-3.6-flash":
            hosted_rows.append((PRETTY[mod], jl(P / "e5_frontier_outcomes.jsonl"), {}))
        else:
            hosted_rows.append((PRETTY[mod], jl(P / f"e12_outcomes_{mod}.jsonl"),
                                confirmations(P / f"e12_confirm_{mod}.jsonl")))
    fig_pipeline(paper / "figures")
    fig_waterfall(paper / "figures", rows, conf, hosted_rows)
    fig_severity(paper / "figures", lbs)
    if tax:
        fig_taxonomy(paper / "figures", tax[0])

    with open(paper / "numbers.tex", "w", encoding="utf-8") as fh:
        fh.write("% GENERATED by analysis/make_v3_assets.py from results/processed -- do not edit\n")
        for k, v in MACROS.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"{len(MACROS)} macros -> {paper / 'numbers.tex'}")


if __name__ == "__main__":
    main()
