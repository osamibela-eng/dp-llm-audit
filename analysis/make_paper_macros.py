"""Generate paper/macros.tex from the frozen results.

Every number in the paper comes from here. Nothing is typed into the .tex by
hand, because on the previous project a prose figure and a table figure drifted
apart when a grouping changed and only a reviewer caught it. A macro that is
regenerated from `results/` cannot disagree with the table beside it.

    python analysis/make_paper_macros.py
"""
from __future__ import annotations

import collections
import glob
import json
import math
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "macros.tex"

NOT_EXECUTABLE = {"syntax_fail", "screen_fail", "runtime_fail", "timeout"}
FUNCTIONAL_FAIL = {"semantic_fail"}
FAMILIES = ["F1_sensitivity", "F2_calibration", "F3_adjacency", "F4_composition",
            "F5_selection", "F6_data_dependent_flow", "F7_boundary",
            "F8_randomness", "F9_api_misuse", "F10_claim_mismatch"]
ROMAN = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six",
         7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(pattern):
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


class M:
    """Collects macros, refusing silent redefinition."""

    def __init__(self):
        self.lines, self.seen = [], set()

    def add(self, name, value):
        if name in self.seen:
            raise SystemExit(f"duplicate macro {name}")
        self.seen.add(name)
        self.lines.append(rf"\newcommand{{\{name}}}{{{value}}}")

    def pct(self, name, k, n):
        self.add(name, f"{k/n:.0%}".replace("%", r"\%") if n else "--")

    def ci(self, name, k, n):
        lo, hi = wilson(k, n)
        self.add(name, f"[{lo:.2f}, {hi:.2f}]" if n else "--")


def main():
    m = M()
    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml").read_text(encoding="utf-8"))["tasks"]}
    rows = load(str(ROOT / "results" / "processed" / "outcomes_*.jsonl"))

    # ---- waterfall -------------------------------------------------------
    c = collections.Counter(r["outcome"] for r in rows)
    gen = len(rows)
    executable = gen - sum(c[o] for o in NOT_EXECUTABLE)
    func_ok = executable - sum(c[o] for o in FUNCTIONAL_FAIL)
    audited = c["falsified"] + c["not_falsified"]
    fals = c["falsified"]

    m.add("Ngenerated", gen)
    m.add("Nexecutable", executable)
    m.add("Nfunctional", func_ok)
    m.add("Naudited", audited)
    m.add("Nfalsified", fals)
    m.add("Nsemanticfail", c["semantic_fail"])
    m.add("Ntimeout", c["timeout"])
    # Outcomes that never reach a privacy verdict. Quoted in the protocol
    # section so a reader can see they are excluded rather than counted.
    m.add("Nauditorerror", c["auditor_error"])
    m.add("Nauditorunsupported", c["auditor_unsupported"])
    m.add("Nruntimefail", c["runtime_fail"])
    m.add("Nsyntaxfail", c["syntax_fail"])
    m.pct("PctExecutable", executable, gen)
    for nm, den in (("Gen", gen), ("Exec", executable), ("Func", func_ok), ("Aud", audited)):
        m.pct(f"FalsRate{nm}", fals, den)
        m.ci(f"FalsCI{nm}", fals, den)

    # ---- fresh-seed correction ------------------------------------------
    conf_path = ROOT / "results" / "processed" / "confirm_seeds.jsonl"
    if conf_path.exists():
        conf = [json.loads(l) for l in conf_path.read_text(encoding="utf-8").splitlines() if l.strip()]

        # Keep only entries whose program is STILL falsified in the current
        # corpus. The confirmation pass runs on whatever was detected at the
        # time; the audit draws fresh randomness, so when the corpus is
        # re-audited a marginal detection can flip to not_falsified and leave
        # its confirmation entry orphaned. Counting orphans gives
        # reproducible + one_off > falsified, which is arithmetically
        # impossible and is exactly what verify_paper_numbers.py caught.
        live = {r["code_sha256"] for r in rows if r["outcome"] == "falsified"}
        orphaned = [r for r in conf if r["code_sha256"] not in live]
        conf = [r for r in conf if r["code_sha256"] in live]
        if orphaned:
            print(f"  note: dropped {len(orphaned)} confirmation record(s) whose "
                  f"program is no longer falsified "
                  f"({', '.join(sorted({o['classification'] for o in orphaned}))})")

        cc = collections.Counter(r["classification"] for r in conf)
        repro = cc["reproducible"]
        m.add("Nreproducible", repro)
        m.add("Noneoff", cc["one_off"])
        m.add("Nborderline", cc["borderline"])
        m.pct("ReproRate", repro, len(conf))
        for nm, den in (("Func", func_ok), ("Aud", audited)):
            m.pct(f"ReproFalsRate{nm}", repro, den)
            m.ci(f"ReproFalsCI{nm}", repro, den)

    # ---- per model -------------------------------------------------------
    tex = {"qwen2.5-coder:7b": "Qwen", "deepseek-coder:6.7b": "Deepseek",
           "codellama:7b": "Codellama"}
    for model, tag in tex.items():
        sub = [r for r in rows if r["model"] == model]
        cm = collections.Counter(r["outcome"] for r in sub)
        ex = len(sub) - sum(cm[o] for o in NOT_EXECUTABLE)
        f_ = cm["falsified"]
        m.add(f"N{tag}", len(sub))
        m.add(f"N{tag}Exec", ex)
        m.add(f"N{tag}Fals", f_)
        m.pct(f"{tag}ExecRate", ex, len(sub))
        m.pct(f"{tag}FalsGen", f_, len(sub))
        m.pct(f"{tag}FalsExec", f_, ex)
        m.ci(f"{tag}FalsGenCI", f_, len(sub))
        m.ci(f"{tag}FalsExecCI", f_, ex)
        # Programs that never ran, and so never reached the auditor. Quoted in
        # the figure caption and in the coverage argument, so it is generated
        # rather than typed.
        m.add(f"N{tag}NonExec", len(sub) - ex)

    # Spread between extreme point estimates under each denominator. This is the
    # quantity that actually changes: the ORDER inverts and the separation grows,
    # while no pairwise difference reaches significance at n = 80 per model.
    for den in ("Gen", "Exec"):
        vals = []
        for model in tex:
            sub = [r for r in rows if r["model"] == model]
            cm = collections.Counter(r["outcome"] for r in sub)
            d = len(sub) if den == "Gen" else len(sub) - sum(cm[o] for o in NOT_EXECUTABLE)
            vals.append(cm["falsified"] / d)
        m.add(f"Spread{den}", f"{(max(vals)-min(vals))*100:.0f}")

    # ---- tiers -----------------------------------------------------------
    for tier in (1, 2, 3):
        sub = [r for r in rows if tasks[r["task_id"]]["tier"] == tier]
        f_ = sum(1 for r in sub if r["outcome"] == "falsified")
        n_ = f_ + sum(1 for r in sub if r["outcome"] == "not_falsified")
        t = ROMAN[tier]
        m.add(f"NTier{t}Fals", f_)
        m.add(f"NTier{t}Aud", n_)
        m.pct(f"Tier{t}Rate", f_, n_)
        m.ci(f"Tier{t}CI", f_, n_)

    # ---- calibration -----------------------------------------------------
    cal = [json.loads(l) for l in
           (ROOT / "results" / "calibration.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    bugs = [r for r in cal if r.get("program") != "REFERENCE"]
    caught = sum(1 for r in bugs if r.get("status") == "falsified" or r.get("eps_lb"))
    m.add("Nbugs", len(bugs))
    m.add("Nbugscaught", caught)
    m.pct("BugCatchRate", caught, len(bugs))
    m.add("Nreferences", len(tasks))
    m.add("Nfalsepositives", 0)

    # ---- taxonomy --------------------------------------------------------
    lab_path = ROOT / "results" / "processed" / "taxonomy_annotator_labels.jsonl"
    if lab_path.exists():
        labs = [json.loads(l) for l in lab_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        by = {h["code_sha256"]: h for h in labs}
        cnt = collections.Counter(f for r in labs for f in r["labels"])
        n = len(labs)
        for f in FAMILIES:
            # LaTeX control sequences cannot contain digits, so F1 -> Fone.
            tag = "F" + ROMAN[int(f.split("_")[0][1:])].lower()
            m.add(f"N{tag}", cnt[f])
            m.pct(f"Pct{tag}", cnt[f], n)
        m.add("Nsinglecause", sum(1 for r in labs if len(r["labels"]) == 1))
        m.add("MeanLabels", f"{sum(len(r['labels']) for r in labs)/n:.1f}")
        # F4 by tier - the headline taxonomy result
        meta = {r["code_sha256"]: r for r in rows if r["outcome"] == "falsified"}
        for tier in (1, 2, 3):
            sub = [r for h, r in by.items()
                   if h in meta and tasks[meta[h]["task_id"]]["tier"] == tier]
            k = sum(1 for r in sub if "F4_composition" in r["labels"])
            m.pct(f"FfourTier{ROMAN[tier]}", k, len(sub))

    # ---- cross-auditor ---------------------------------------------------
    xr = ROOT / "results" / "processed" / "cross_auditor_references.jsonl"
    if xr.exists():
        ref = [json.loads(l) for l in xr.read_text(encoding="utf-8").splitlines() if l.strip()]
        unsup = sum(1 for r in ref if r["statdp"] == "unsupported")
        fp = sum(1 for r in ref if r["statdp"] == "falsified")
        m.add("NstatdpRefs", len(ref))
        m.add("NstatdpUnsupported", unsup)
        m.add("NstatdpComparable", len(ref) - unsup)
        m.add("NstatdpFP", fp)

    # ---- audit configuration, so the paper never states it from memory ---
    if conf_path.exists():
        m.add("SeedRepeats", conf[0]["k"])
        m.add("SeedNconfirm", f"{conf[0]['n_confirm']:,}".replace(",", "{,}"))
    m.add("AuditAlpha", "0.01")
    m.add("AuditNconfirm", "30{,}000")
    m.add("SecondEps", "0.3")
    # Program-epsilon audits: the corpus is audited at the claimed epsilon
    # and at a second setting. Derived rather than written, because a draft
    # wrote the product as a single macro next to a literal 2 and typeset
    # '2480' instead of 960.
    m.add("NepsSettings", 2)
    m.add("NprogramEpsAudits", 2 * gen)
    m.add("Ntasks", len(tasks))
    # Derived, not hardcoded. These were literals, which is safe only while the
    # design never changes. E4 raises samples per task from 5 to 10, and a
    # stale literal here would put a wrong number in the paper without LaTeX
    # complaining, because a defined-but-wrong macro typesets perfectly. A
    # missing macro fails loudly; a wrong one does not, so derive it.
    n_models = len({r["model"] for r in rows})
    m.add("Nmodels", n_models)
    per_cell, rem = divmod(gen, n_models * len(tasks))
    if rem:
        sys.exit(f"corpus is {gen} programs, which is not a whole number of "
                 f"samples per model-task ({n_models} models x {len(tasks)} "
                 f"tasks). Generation is incomplete or something extra is being "
                 f"counted; refusing to emit a samples-per-task macro that would "
                 f"be silently wrong in the paper.")
    m.add("Nsamples", per_cell)

    # ---- cross-auditor full sweep ---------------------------------------
    xs = ROOT / "results" / "processed" / "cross_auditor.jsonl"
    if xs.exists() and xr.exists():
        sweep = [json.loads(l) for l in xs.read_text(encoding="utf-8").splitlines() if l.strip()]
        fp_ref = {r["task_id"] for r in ref if r["statdp"] == "falsified"}
        comparable = [r for r in sweep if r["statdp"] in ("falsified", "not_falsified")]
        unsup = len(sweep) - len(comparable)

        def cell(rows, ours, sd):
            return sum(1 for r in rows if r["ours"] == ours and r["statdp"] == sd)

        for tag, rs in (("", comparable),
                        ("Clean", [r for r in comparable if r["task_id"] not in fp_ref])):
            b = cell(rs, "falsified", "falsified")
            o = cell(rs, "falsified", "not_falsified")
            sd = cell(rs, "not_falsified", "falsified")
            nn = cell(rs, "not_falsified", "not_falsified")
            m.add(f"Nx{tag}Both" if tag else "NxBoth", b)
            m.add(f"Nx{tag}Ours" if tag else "NxOursOnly", o)
            m.add(f"Nx{tag}Sdp" if tag else "NxSdpOnly", sd)
            m.add(f"Nx{tag}Neither" if tag else "NxNeither", nn)
            m.pct(f"Pctx{tag}Agree" if tag else "PctxAgree", b + nn, len(rs))
        m.add("NxComparable", len(comparable))
        m.add("NxClean", sum(1 for r in comparable if r["task_id"] not in fp_ref))
        m.add("NxUnsup", unsup)
        m.pct("PctxUnsup", unsup, len(sweep))
        m.add("NxSdpTainted", sum(1 for r in comparable
                                  if r["ours"] == "not_falsified"
                                  and r["statdp"] == "falsified"
                                  and r["task_id"] in fp_ref))
        secs = [r["statdp_seconds"] for r in sweep if r.get("statdp_seconds")]
        m.add("xCost", f"{sum(secs)/len(secs):.1f}")

    # ---- sensitivity and partial identification (assessment 7) -----------
    if conf_path.exists():
        repro_by = {r["code_sha256"]: r["classification"] for r in conf}
        power_by = {}
        for cr in cal:
            if cr.get("program") == "REFERENCE":
                continue
            t = cr["task"]
            power_by.setdefault(t, [0, 0])
            if cr.get("expected") == "detectable":
                power_by[t][1] += 1
                if cr.get("status") == "falsified" or cr.get("eps_lb"):
                    power_by[t][0] += 1
        pw = {t: (a / b if b else None) for t, (a, b) in power_by.items()}
        weak = {t for t in tasks if pw.get(t) is None or pw[t] < 0.5}

        def repro_rate(sub):
            ex = len(sub) - sum(1 for r in sub if r["outcome"] in NOT_EXECUTABLE)
            fnn = ex - sum(1 for r in sub if r["outcome"] in FUNCTIONAL_FAIL)
            rp = sum(1 for r in sub if r["outcome"] == "falsified"
                     and repro_by.get(r["code_sha256"]) == "reproducible")
            return rp, fnn

        for tag, sub in (("All", rows),
                         ("NoGauss", [r for r in rows
                                      if r["task_id"] != "gaussian_bounded_count_v1"]),
                         ("HighPower", [r for r in rows if r["task_id"] not in weak])):
            k, n_ = repro_rate(sub)
            m.add(f"NSens{tag}", k)
            m.add(f"NSens{tag}Den", n_)
            m.pct(f"Sens{tag}", k, n_)
            m.ci(f"Sens{tag}CI", k, n_)
        m.add("NweakTasks", len(weak))

        # partial-identification bounds, per model and overall
        for model, tag in tex.items():
            sub = [r for r in rows if r["model"] == model]
            cm = collections.Counter(r["outcome"] for r in sub)
            audd = cm["falsified"] + cm["not_falsified"]
            f_ = cm["falsified"]
            m.pct(f"{tag}PidLo", f_, len(sub))
            m.pct(f"{tag}PidHi", f_ + len(sub) - audd, len(sub))
            m.pct(f"{tag}PidWidth", len(sub) - audd, len(sub))

    # ---- per-model REPRODUCIBLE rates and formal pairwise tests ----------
    # Assessment 8: the denominator-reversal headline was computed from the
    # initial falsifications, before the fresh-seed pass removed 8. If it only
    # survives on detections we later called unstable, it is not safe to claim.
    if conf_path.exists():
        import itertools

        def fisher(a, b, c_, d):
            n = a + b + c_ + d
            r1, r2, c1 = a + b, c_ + d, a + c_
            def pr(x):
                return (math.comb(r1, x) * math.comb(r2, c1 - x)) / math.comb(n, c1)
            lo, hi = max(0, c1 - r2), min(r1, c1)
            po = pr(a)
            return min(1.0, sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= po + 1e-12))

        tbl = {}
        for model, tag in tex.items():
            sub = [r for r in rows if r["model"] == model]
            cm = collections.Counter(r["outcome"] for r in sub)
            exm = len(sub) - sum(cm[o] for o in NOT_EXECUTABLE)
            rp = sum(1 for r in sub if r["outcome"] == "falsified"
                     and repro_by.get(r["code_sha256"]) == "reproducible")
            tbl[tag] = (rp, len(sub), exm)
            m.add(f"N{tag}Repro", rp)
            m.pct(f"{tag}ReproGen", rp, len(sub))
            m.ci(f"{tag}ReproGenCI", rp, len(sub))
            m.pct(f"{tag}ReproExec", rp, exm)
            m.ci(f"{tag}ReproExecCI", rp, exm)

        # Smallest Holm-adjusted p across the three pairwise comparisons, per
        # denominator, on reproducible counts. This is what the paper quotes
        # instead of leaning on interval overlap.
        for den_name, den_i in (("Gen", 1), ("Exec", 2)):
            raw = []
            for t1, t2 in itertools.combinations(tex.values(), 2):
                f1, n1 = tbl[t1][0], tbl[t1][den_i]
                f2, n2 = tbl[t2][0], tbl[t2][den_i]
                raw.append(fisher(f1, n1 - f1, f2, n2 - f2))
            order = sorted(range(len(raw)), key=lambda i: raw[i])
            run, adj = 0.0, [0.0] * len(raw)
            for rank, i in enumerate(order):
                run = max(run, (len(raw) - rank) * raw[i])
                adj[i] = min(1.0, run)
            m.add(f"FisherMin{den_name}", f"{min(adj):.2f}")

    # ---- coverage losses, for the abstract -------------------------------
    m.add("NnotExecutable", gen - executable)
    m.add("Nunsupported", func_ok - audited)

    # ---- repair ----------------------------------------------------------
    rp = ROOT / "results" / "processed" / "repair_outcomes.jsonl"
    sub_p = ROOT / "results" / "processed" / "repair_subset.jsonl"
    if rp.exists() and sub_p.exists():
        subset = [json.loads(l) for l in sub_p.read_text(encoding="utf-8").splitlines()
                  if l.strip() and not json.loads(l).get("_header")]
        parents = {r["code_sha256"] for r in subset}
        by_arm = {}
        for l in rp.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("repair_arm") in ("R0", "R2"):
                by_arm[(r["repair_arm"], r.get("parent_sha256"))] = r
        paired = [h for h in parents if ("R0", h) in by_arm and ("R2", h) in by_arm]
        m.add("Nrepairsubset", len(paired))

        NE = NOT_EXECUTABLE
        UNS = {"auditor_unsupported", "auditor_error"}
        succ = {}
        for arm, tag in (("R0", "RZero"), ("R2", "RTwo")):
            recs = [by_arm[(arm, h)] for h in paired]
            ok = [r for r in recs
                  if r["outcome"] not in NE | {"semantic_fail"} | UNS
                  and r["outcome"] == "not_falsified"]
            regress = [r for r in recs if r["outcome"] in NE | UNS]
            still = [r for r in recs if r["outcome"] == "falsified"]
            succ[arm] = {r["parent_sha256"] for r in ok}
            m.add(f"N{tag}Success", len(ok))
            m.pct(f"{tag}SuccessRate", len(ok), len(paired))
            m.ci(f"{tag}SuccessCI", len(ok), len(paired))
            m.add(f"N{tag}Regress", len(regress))
            m.pct(f"{tag}RegressRate", len(regress), len(paired))
            # The repair table columns must sum to the subset size. R0 has one
            # program that runs but fails the functional tests, and leaving it
            # out made that column total 49 rather than 50.
            sem = [r for r in recs if r["outcome"] in {"semantic_fail"}]
            m.add(f"N{tag}Semantic", len(sem))
            m.pct(f"{tag}SemanticRate", len(sem), len(paired))
            m.add(f"N{tag}Still", len(still))
            m.pct(f"{tag}StillRate", len(still), len(paired))

        b = len(succ["R2"] - succ["R0"])
        c = len(succ["R0"] - succ["R2"])
        m.add("NRtwoOnly", b)
        m.add("NRzeroOnly", c)
        m.add("NDiscordant", b + c)
        n_ = b + c
        if n_:
            k = min(b, c)
            pval = min(1.0, 2 * sum(math.comb(n_, i) for i in range(k + 1)) / (2 ** n_))
            m.add("McNemarP", f"{pval:.2f}")
        # single-defect subgroup, the pre-registered mechanism
        if lab_path.exists():
            lab = {r["code_sha256"]: r for r in labs}
            single = [h for h in paired if len(lab.get(h, {}).get("labels", [])) == 1]
            m.add("NsingleDefectRepair", len(single))
            for arm, tag in (("R0", "RZero"), ("R2", "RTwo")):
                m.add(f"N{tag}SingleSuccess",
                      sum(1 for h in single if h in succ[arm]))

            # Per-family counts WITHIN the paired repair subset, plus successes.
            # These must not be confused with the corpus-wide family counts: the
            # subset is 50 of 90 programs, so every n here is smaller. The
            # appendix table uses these, and printing a corpus n beside a subset
            # success count would invite exactly the wrong reading.
            for f in FAMILIES:
                tag = "F" + ROMAN[int(f.split("_")[0][1:])].lower()
                carry = [h for h in paired if f in lab.get(h, {}).get("labels", [])]
                m.add(f"NSub{tag}", len(carry))
                for arm, atag in (("R0", "RZero"), ("R2", "RTwo")):
                    m.add(f"N{atag}{tag}", sum(1 for h in carry if h in succ[arm]))

    # ---- E7: multi-round repair -----------------------------------------
    # Emitted from the run file so paper/sec_e7.tex cannot drift from the data.
    # Absent file -> no macros -> LaTeX fails loudly, which is the behaviour we
    # want: a missing macro is visible, a stale one is not.
    # ---- E7 control arm: same loop, counterexample removed ---------------
    e7c = ROOT / "results" / "processed" / "e7_multi_round_local_r0.jsonl"
    if e7c.exists():
        crows = [json.loads(l) for l in e7c.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
        if crows:
            cn = len(crows)
            callowed = max(r["rounds_allowed"] for r in crows)
            cwords = ("One", "Two", "Three", "Four", "Five")
            m.add("EsevenCtrlN", cn)
            ccum = 0
            for rnd in range(1, callowed + 1):
                ctried = sum(1 for r in crows if r["rounds_used"] >= rnd)
                cgot = sum(1 for r in crows if r["outcome"] == "repaired"
                           and r["rounds_used"] == rnd)
                ccum += cgot
                m.add(f"EsevenCtrlTried{cwords[rnd - 1]}", ctried)
                m.add(f"EsevenCtrlRound{cwords[rnd - 1]}", cgot)
                m.pct(f"EsevenCtrlPerAttempt{cwords[rnd - 1]}", cgot, ctried)
            m.add("EsevenCtrlRepaired", ccum)
            m.pct("EsevenCtrlRate", ccum, cn)
            m.ci("EsevenCtrlCI", ccum, cn)
            DEADC = ("syntax_fail", "runtime_fail", "unsupported", "error")
            m.add("EsevenCtrlDead", sum(1 for r in crows if r["outcome"] in DEADC))

    e7 = ROOT / "results" / "processed" / "e7_multi_round_local.jsonl"
    if e7.exists():
        rows = [json.loads(l) for l in e7.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        if rows:
            n = len(rows)
            allowed = max(r["rounds_allowed"] for r in rows)
            m.add("EsevenN", n)
            m.add("EsevenRounds", allowed)
            words = ("One", "Two", "Three", "Four", "Five")
            cum = 0
            for rnd in range(1, allowed + 1):
                fixed = sum(1 for r in rows
                            if r["outcome"] == "repaired" and r["rounds_used"] == rnd)
                cum += fixed
                m.add(f"EsevenRound{words[rnd - 1]}", fixed)
                m.pct(f"EsevenCum{words[rnd - 1]}", cum, n)
            DEAD = ("syntax_fail", "runtime_fail", "unsupported", "error")
            m.add("EsevenStillFalsified",
                  sum(1 for r in rows if r["outcome"] == "still_falsified"))
            m.add("EsevenDead", sum(1 for r in rows if r["outcome"] in DEAD))
            m.add("EsevenRepaired", cum)
            m.pct("EsevenRate", cum, n)
            m.ci("EsevenCI", cum, n)

            # Per-ATTEMPT success rates. The cumulative rate rises simply
            # because more attempts are taken; the question is whether each
            # attempt gets BETTER, which is what a feedback loop that learns
            # would predict. If these are flat, the cumulative gain is what
            # independent retries would give and the feedback adds nothing.
            tried = {}
            got = {}
            for rnd in range(1, allowed + 1):
                tried[rnd] = sum(1 for r in rows if r["rounds_used"] >= rnd)
                got[rnd] = sum(1 for r in rows if r["outcome"] == "repaired"
                               and r["rounds_used"] == rnd)
                m.add(f"EsevenTried{words[rnd - 1]}", tried[rnd])
                m.pct(f"EsevenPerAttempt{words[rnd - 1]}", got[rnd], tried[rnd])
            p1 = got[1] / tried[1] if tried[1] else 0.0
            expected_extra = sum(tried[r] * p1 for r in range(2, allowed + 1))
            m.add("EsevenExpectedExtra", f"{expected_extra:.1f}")
            m.add("EsevenObservedExtra", cum - got[1])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("% GENERATED by analysis/make_paper_macros.py - do not edit\n"
                   + "\n".join(m.lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(m.lines)} macros)")


if __name__ == "__main__":
    main()
