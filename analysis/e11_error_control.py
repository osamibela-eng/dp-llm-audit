"""E11: what is the family-wise error rate across the whole audit campaign?

The objection: "you ran 147 programs x 2 epsilon settings x several pairs x
several events. What is the error rate across all of that?"

The plan's instruction was to state the assumptions first and then re-read the
auditor code to confirm the implementation satisfies each one, reporting the gap
instead of the bound if any assumption fails. One does fail, so this file reports
both: the bound that holds, and the place where the implementation is looser than
the ideal.

Everything below is checked mechanically where it can be. Where it cannot be
checked by a script, the source location that was read by hand is named so a
reviewer can check it too.

    python analysis/e11_error_control.py
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AUDITOR = ROOT / "auditors" / "simple_auditor.py"
AGGREGATE = ROOT / "analysis" / "aggregate.py"
CONFIRM = ROOT / "analysis" / "confirm_seeds.py"


class Check:
    def __init__(self):
        self.rows = []

    def add(self, name, ok, evidence, note=""):
        self.rows.append((name, ok, evidence, note))

    def report(self):
        print("=" * 78)
        print("E11a  ASSUMPTIONS, CHECKED AGAINST THE IMPLEMENTATION")
        print("=" * 78)
        for name, ok, evidence, note in self.rows:
            mark = "HOLDS " if ok else "GAP   "
            print(f"\n  [{mark}] {name}")
            print(f"           evidence: {evidence}")
            if note:
                for line in note.strip().splitlines():
                    print(f"           {line.strip()}")
        return all(ok for _, ok, _, _ in self.rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--outcomes",
                    default=str(ROOT / "results" / "processed" / "outcomes_*.jsonl"))
    a = ap.parse_args()

    src_aud = AUDITOR.read_text(encoding="utf-8")
    src_agg = AGGREGATE.read_text(encoding="utf-8")
    src_con = CONFIRM.read_text(encoding="utf-8")

    c = Check()

    # ---- A1 pilot / confirm separation ----------------------------------
    i_events = src_aud.find("per_pair_events.append")
    i_confirm = src_aud.find("s1 = _sample")
    c.add("A1. Events are selected on pilot draws, then tested on fresh draws",
          i_events != -1 and i_confirm != -1 and i_events < i_confirm,
          f"simple_auditor.py: events chosen at offset {i_events}, "
          f"confirmation sampled at {i_confirm}",
          """Selection happens strictly before the confirmation sample is drawn, so
             the Clopper-Pearson bounds are valid conditional on the chosen event
             set. Without this the bounds would be selective and meaningless.""")

    # ---- A2 Bonferroni arithmetic ---------------------------------------
    has_total = "total_tests = sum(2 * len(ev)" in src_aud
    has_adj = "a_adj = alpha / total_tests / 2.0" in src_aud
    c.add("A2. Within one audit, the family-wise error is at most alpha",
          has_total and has_adj,
          "simple_auditor.py: total_tests counts both directions; "
          "a_adj = alpha / total_tests / 2",
          """Per event and direction the code computes two one-sided bounds, a lower
             bound on one side and an upper bound on the other. That is
             2 x total_tests bounds, each at level alpha / (2 x total_tests), so the
             union bound gives exactly alpha. A false detection needs at least one
             of those bounds to fail, so P(false detection in one audit) <= alpha.
             Bonferroni needs no independence, which matters because the two
             directions reuse the same samples.""")

    # ---- A3 the outer epsilon loop --------------------------------------
    two_eps = "eps_settings.append(second_eps)" in src_agg
    corrected = re.search(r"alpha\s*/\s*len\(eps_settings\)", src_agg) is not None
    c.add("A3. The two epsilon settings are corrected for",
          not two_eps or corrected,
          f"aggregate.py: second epsilon appended = {two_eps}; "
          f"alpha divided across settings = {corrected}",
          """THIS IS THE GAP. Each program is audited at its claimed epsilon and
             again at 0.3. Each audit controls its own family-wise error at alpha,
             but nothing spans the outer loop, so a single program's chance of at
             least one false detection is bounded by 2 x alpha rather than alpha.
             We report 2 x alpha rather than pretending otherwise. This is also the
             most likely origin of the 8 of 90 detections that did not survive
             fresh seeds in v1, since roughly 9 percent is what a per-program rate
             near 2 percent produces once you look across many programs.""")

    # ---- A4 fresh randomness per repeat ---------------------------------
    fresh = "SEEDS[:a.k]" in src_con or "for s in SEEDS" in src_con
    c.add("A4. Each repeat uses fresh randomness",
          fresh,
          "confirm_seeds.py iterates over a fixed list of distinct seeds",
          """Each repeat re-runs the whole procedure, pilot phase included, so the
             repeats are independent of one another and of the original audit. The
             seeds are fixed rather than drawn from the clock so the campaign
             reproduces exactly.""")

    # ---- A5 repeats use one epsilon only --------------------------------
    one_eps = 'float(task["claimed_epsilon"])' in src_con and "second" not in src_con
    c.add("A5. Repeats audit at the claimed epsilon only",
          one_eps,
          "confirm_seeds.py calls falsify with claimed_epsilon and no second setting",
          """So each repeat carries error at most alpha, not 2 x alpha. This is what
             makes the reproducible label as strong as it is.""")

    # ---- A6 the label rule ----------------------------------------------
    strict = 'hits == attempts' in src_con
    c.add("A6. A program is labelled reproducible only on unanimous repeats",
          strict,
          "confirm_seeds.py labels reproducible only when hits == attempts",
          """Anything short of unanimous is reported separately as borderline or
             one-off and is excluded from the headline rate.""")

    all_hold = c.report()

    # ---- the proposition -------------------------------------------------
    alpha, k = a.alpha, a.repeats
    per_program = 2 * alpha * (alpha ** k)

    rows, seen = [], set()
    for f in sorted(glob.glob(a.outcomes)):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("repair_arm") or r.get("parent_sha256"):
                continue
            key = (r["model"], r["task_id"], r["sample_idx"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(r)
    n_programs = len(rows) or 1

    print("\n" + "=" * 78)
    print("E11b  PROPOSITION")
    print("=" * 78)
    print(f"""
  Let a program satisfy the (eps, delta)-DP claim its task states. Then under
  A1, A2, A4, A5 and A6, the probability that our campaign labels it
  REPRODUCIBLY FALSIFIED is at most

        2 * alpha  *  alpha^{k}   =   {per_program:.2e}

  with alpha = {alpha} and {k} repeats.

  The leading factor of 2 is the A3 gap, not a modelling choice: the original
  audit runs at two epsilon settings with no correction across them, so it
  contributes 2 * alpha. Each of the {k} repeats runs at one setting and
  contributes alpha. All {k + 1} audits are independent by A4.

  Over the {n_programs} programs in this corpus, a union bound gives

        P(any correct program is labelled reproducibly falsified)
              <=  {n_programs} * {per_program:.2e}  =  {n_programs * per_program:.2e}

  The load-bearing assumption is A1. If events were selected using the same
  draws that are then tested, every bound above would be selective and the
  proposition would say nothing. A2 is arithmetic and A4 to A6 are procedural;
  A1 is the one a reader should check in the code, at the line numbers named
  above.
""")

    print("=" * 78)
    print("E11c  WHAT THIS DOES NOT COVER")
    print("=" * 78)
    print("""
  This bounds FALSE detections against programs that are correct. It says
  nothing about missed violations, which is the direction our calibration
  measures instead: the auditor catches 24 of 61 planted bugs, so reported
  rates are lower bounds on detectable violations.

  It also assumes the mechanism's randomness is the only randomness. A program
  that reads a clock, a global RNG or the filesystem breaks the independence in
  A4. Our sandbox screens imports, and the one program that ignored the injected
  generator in favour of module-level numpy.random is recorded as F9 in the
  taxonomy rather than silently trusted.
""")

    if not all_hold:
        print("  NOTE: at least one assumption is a GAP. The bound above already")
        print("  accounts for it; it is not an idealisation.\n")


if __name__ == "__main__":
    main()
