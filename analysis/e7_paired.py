"""E7 with its control: does the counterexample contribute anything?

E7 ran three attempts with counterexample feedback (R2) and found the cumulative
repair rate roughly doubling while the per-attempt rate stayed flat. That is
consistent with the gain coming from resampling rather than from the feedback,
but "consistent with" was an argument against an analytic expectation, not a
measured comparison.

This adds the matched control: the identical loop with the counterexample
removed (R0). Each round still sees the model's own previous code, so the only
difference between the arms is whether the audit's witness is shown.

Both arms run on the same pre-registered subset, so the comparison is paired and
the test is exact McNemar on the discordant programs.

Read the per-attempt rates, not just the cumulative ones. If R0 and R2 track each
other, the counterexample is not the active ingredient and the cumulative gain in
both arms is the arithmetic of taking more attempts.

    python analysis/e7_paired.py
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "processed"
DEAD = ("syntax_fail", "runtime_fail", "unsupported", "error")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(tag):
    p = OUT / f"e7_multi_round_{tag}.jsonl"
    if not p.exists():
        return {}
    return {r["parent_sha256"]: r
            for r in (json.loads(l) for l in
                      p.read_text(encoding="utf-8").splitlines() if l.strip())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feedback-tag", default="local")
    ap.add_argument("--control-tag", default="local_r0")
    a = ap.parse_args()

    fb, ct = load(a.feedback_tag), load(a.control_tag)
    if not ct:
        sys.exit(f"control arm not found: e7_multi_round_{a.control_tag}.jsonl")

    print("=" * 76)
    print("E7  DOES THE COUNTEREXAMPLE CONTRIBUTE, OR IS IT JUST MORE ATTEMPTS?")
    print("=" * 76)
    print(f"\n  feedback arm (R2, counterexample shown): {len(fb)} programs")
    print(f"  control arm  (R0, no counterexample)   : {len(ct)} programs")

    paired = sorted(set(fb) & set(ct))
    if not paired:
        sys.exit("no programs common to both arms")
    if len(paired) < min(len(fb), len(ct)):
        print(f"  restricted to the {len(paired)} programs both arms cover")

    def stats(arm, keys):
        rows = [arm[k] for k in keys]
        n = len(rows)
        allowed = max(r["rounds_allowed"] for r in rows)
        per = {}
        cum = 0
        for rnd in range(1, allowed + 1):
            tried = sum(1 for r in rows if r["rounds_used"] >= rnd)
            got = sum(1 for r in rows if r["outcome"] == "repaired"
                      and r["rounds_used"] == rnd)
            cum += got
            per[rnd] = (tried, got)
        return {"n": n, "allowed": allowed, "per": per, "repaired": cum,
                "dead": sum(1 for r in rows if r["outcome"] in DEAD),
                "still": sum(1 for r in rows
                             if r["outcome"] == "still_falsified")}

    S = {"R2 (feedback)": stats(fb, paired), "R0 (control)": stats(ct, paired)}

    print("\n  " + "-" * 72)
    print(f"  {'arm':16s} {'attempt':>8s} {'tried':>6s} {'repaired':>9s} "
          f"{'per-attempt':>12s} {'95% CI':>14s}")
    for name, s in S.items():
        for rnd in range(1, s["allowed"] + 1):
            tried, got = s["per"][rnd]
            lo, hi = wilson(got, tried)
            print(f"  {name if rnd == 1 else '':16s} {rnd:>8d} {tried:>6d} "
                  f"{got:>9d} {got / tried if tried else 0:>11.0%} "
                  f"  [{lo:.2f}, {hi:.2f}]")

    print("\n  " + "-" * 72)
    print(f"  {'arm':16s} {'cumulative':>11s} {'rate':>7s} {'95% CI':>14s} "
          f"{'still fals.':>12s} {'died':>6s}")
    for name, s in S.items():
        lo, hi = wilson(s["repaired"], s["n"])
        print(f"  {name:16s} {s['repaired']:>11d} "
              f"{s['repaired'] / s['n']:>7.0%}   [{lo:.2f}, {hi:.2f}] "
              f"{s['still']:>12d} {s['dead']:>6d}")

    # ---- the paired test -------------------------------------------------
    rep_fb = {k for k in paired if fb[k]["outcome"] == "repaired"}
    rep_ct = {k for k in paired if ct[k]["outcome"] == "repaired"}
    b = len(rep_fb - rep_ct)
    c = len(rep_ct - rep_fb)
    p = mcnemar_exact(b, c)

    print("\n  " + "-" * 72)
    print("  exact McNemar, paired over the same programs")
    print(f"    repaired only WITH the counterexample : {b}")
    print(f"    repaired only WITHOUT it              : {c}")
    print(f"    p = {p:.3f}")

    need = next((x for x in range(1, 40) if mcnemar_exact(x, 0) < 0.05), None)
    print(f"\n    For significance at 0.05 one arm would need to repair {need}")
    print("    programs the other missed while missing none of the other's.")

    print("\n" + "=" * 76)
    if p > 0.05:
        print("  The arms do not separate. Showing the model the audit's own")
        print("  witness did not measurably change how often it repaired the")
        print("  program, over three attempts, on the same 50 programs.")
        print("\n  Combined with the flat per-attempt rates in both arms, the")
        print("  cumulative gain from iterating is the arithmetic of taking more")
        print("  attempts, not evidence of a model learning from the correction.")
        print("\n  This is now a measured control rather than a comparison against")
        print("  an analytic expectation, which is what the earlier version of")
        print("  this claim rested on.")
    else:
        print(f"  The arms separate (p = {p:.3f}). Report which direction and")
        print("  revise the claim that the counterexample contributes nothing.")

    (OUT / "e7_paired.json").write_text(json.dumps({
        "paired": len(paired),
        "arms": {k: {"repaired": v["repaired"], "n": v["n"],
                     "per_attempt": {str(r): v["per"][r] for r in v["per"]}}
                 for k, v in S.items()},
        "mcnemar": {"only_feedback": b, "only_control": c, "p": p},
    }, indent=2), encoding="utf-8")
    print("\n  wrote results/processed/e7_paired.json")


if __name__ == "__main__":
    main()
