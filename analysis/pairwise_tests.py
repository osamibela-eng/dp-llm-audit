"""Per-model rates on REPRODUCIBLE falsifications, plus formal pairwise tests.

Assessment 8 raised two things this file answers.

1. The denominator-reversal headline was computed from the 90 INITIAL
   falsifications, before the fresh-seed pass removed 8. If the reversal only
   survives on detections we later decided were unstable, the claim is not safe.
   So every per-model rate is recomputed counting only falsifications that
   reproduced in all three fresh-seed repeats, and we check whether the ordering
   still inverts.

2. Overlapping confidence intervals are not a test of equality. Fisher's exact
   test is used for every pairwise model comparison under every denominator,
   with Holm correction across the three comparisons within each denominator.
   Wilson intervals stay in the paper for transparency, but the inferential
   statement comes from the test.

    python analysis/pairwise_tests.py
"""
from __future__ import annotations

import collections
import glob
import itertools
import json
import math
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOT_EXECUTABLE = {"syntax_fail", "screen_fail", "runtime_fail", "timeout"}
FUNCTIONAL_FAIL = {"semantic_fail"}
SHORT = {"qwen2.5-coder:7b": "qwen", "deepseek-coder:6.7b": "deepseek",
         "codellama:7b": "codellama"}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fisher_exact_two_sided(a, b, c, d):
    """Two-sided Fisher exact test on [[a,b],[c,d]], by summing all tables at
    least as extreme (as probable or less) than the observed one."""
    n = a + b + c + d
    r1, r2 = a + b, c + d
    c1 = a + c

    def prob(x):
        return (math.comb(r1, x) * math.comb(r2, c1 - x)) / math.comb(n, c1)

    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    p_obs = prob(a)
    tol = 1e-12
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1)
                        if prob(x) <= p_obs + tol))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    for rank, i in enumerate(idx):
        val = (len(pvals) - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def load():
    rows, seen = [], set()
    for f in sorted(glob.glob(str(ROOT / "results" / "processed" / "outcomes_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("repair_arm") or r.get("parent_sha256"):
                continue
            k = (r["model"], r["task_id"], r["sample_idx"])
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
    return rows


def main():
    rows = load()
    conf = {r["code_sha256"]: r["classification"] for r in
            (json.loads(l) for l in
             (ROOT / "results" / "processed" / "confirm_seeds.jsonl")
             .read_text(encoding="utf-8").splitlines() if l.strip())}

    models = list(SHORT)

    def counts(model, reproducible_only):
        sub = [r for r in rows if r["model"] == model]
        c = collections.Counter(r["outcome"] for r in sub)
        gen = len(sub)
        ex = gen - sum(c[o] for o in NOT_EXECUTABLE)
        fn = ex - sum(c[o] for o in FUNCTIONAL_FAIL)
        aud = c["falsified"] + c["not_falsified"]
        if reproducible_only:
            f_ = sum(1 for r in sub if r["outcome"] == "falsified"
                     and conf.get(r["code_sha256"]) == "reproducible")
        else:
            f_ = c["falsified"]
        return gen, ex, fn, aud, f_

    for label, repro in (("INITIAL falsifications", False),
                         ("REPRODUCIBLE falsifications only", True)):
        print("=" * 74)
        print(f"PER-MODEL RATES, {label}")
        print("=" * 74)
        print(f"  {'model':11s} {'exec':>6s}   {'per generated':>22s}   {'per executable':>22s}")
        table = {}
        for m in models:
            gen, ex, fn, aud, f_ = counts(m, repro)
            lo1, hi1 = wilson(f_, gen)
            lo2, hi2 = wilson(f_, ex)
            table[m] = (f_, gen, ex)
            print(f"  {SHORT[m]:11s} {ex:3d}/{gen:<3d}  {f_:3d}/{gen:<3d} {f_/gen:5.0%} "
                  f"[{lo1:.2f},{hi1:.2f}]   {f_:3d}/{ex:<3d} {f_/ex:5.0%} [{lo2:.2f},{hi2:.2f}]")

        # Ties must be visible. Sorting silently breaks them, and on the
        # reproducible counts two models land on exactly the same rate, which
        # changes what can be claimed.
        def ordering(den_i):
            rates = {m: table[m][0] / table[m][den_i] for m in models}
            out, seen_r = [], []
            for m in sorted(models, key=lambda x: rates[x]):
                if seen_r and abs(rates[m] - seen_r[-1]) < 1e-9:
                    out[-1] += f" = {SHORT[m]}"
                else:
                    out.append(SHORT[m])
                seen_r.append(rates[m])
            return rates, out

        gen_rates, gen_order = ordering(1)
        exec_rates, exec_order = ordering(2)
        print(f"\n  ranking per generated  (lowest first): {' < '.join(gen_order)}")
        print(f"  ranking per executable (lowest first): {' < '.join(exec_order)}")

        cl = "codellama:7b"
        lowest_gen = abs(gen_rates[cl] - min(gen_rates.values())) < 1e-9
        strictly_lowest = lowest_gen and sum(
            1 for m in models if abs(gen_rates[m] - gen_rates[cl]) < 1e-9) == 1
        highest_exec = abs(exec_rates[cl] - max(exec_rates.values())) < 1e-9
        print(f"  codellama lowest per generated : {lowest_gen}"
              f"{'' if strictly_lowest else ' (tied)' if lowest_gen else ''}")
        print(f"  codellama highest per executable: {highest_exec}")

        # ---- formal pairwise tests ---------------------------------------
        for den_name, den_i in (("generated", 1), ("executable", 2)):
            raw, pairs = [], []
            for m1, m2 in itertools.combinations(models, 2):
                f1, n1 = table[m1][0], table[m1][den_i]
                f2, n2 = table[m2][0], table[m2][den_i]
                p = fisher_exact_two_sided(f1, n1 - f1, f2, n2 - f2)
                raw.append(p)
                pairs.append((SHORT[m1], SHORT[m2]))
            adj = holm(raw)
            print(f"\n  Fisher exact, per {den_name} (Holm-adjusted across 3 comparisons):")
            for (a, b), p, q in zip(pairs, raw, adj):
                sig = "significant" if q < 0.05 else "not significant"
                print(f"    {a:10s} vs {b:11s} p = {p:.3f}   adj p = {q:.3f}   {sig}")
        print()


if __name__ == "__main__":
    main()
