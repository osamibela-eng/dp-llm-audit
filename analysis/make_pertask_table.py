"""Generate paper/table_pertask.tex - the per-task appendix table.

Detection power sits beside every rate, because without it the zeros are
unreadable. Four tasks draw zero falsifications for three different reasons:

  randomized_response    power 1.00, 11 audited  -> the models get it right
  report_noisy_max       power 1.00,  2 audited  -> almost no evidence either way
  private_topk           power 0.33,  0 audited  -> no coverage at all
  exponential_mechanism  power 0.33,  4 audited  -> weak power AND thin coverage

Printing "0%" for all four without the power and the denominator beside it would
present four different situations as one finding.

    python analysis/make_pertask_table.py
"""
from __future__ import annotations

import collections
import glob
import json
import math
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "table_pertask.tex"

NOT_EXECUTABLE = {"syntax_fail", "screen_fail", "runtime_fail", "timeout"}


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def detection_power(cal):
    power = {}
    for c in cal:
        if c.get("program") == "REFERENCE":
            continue
        t = c["task"]
        power.setdefault(t, [0, 0])
        if c.get("expected") == "detectable":
            power[t][1] += 1
            if c.get("status") == "falsified" or c.get("eps_lb"):
                power[t][0] += 1
    return {t: (a / b if b else None) for t, (a, b) in power.items()}


def esc(s):
    return s.replace("_", r"\_")


def main():
    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml").read_text(encoding="utf-8"))["tasks"]}
    cal = [json.loads(l) for l in
           (ROOT / "results" / "calibration.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    power = detection_power(cal)

    rows, seen = [], set()
    for f in sorted(glob.glob(str(ROOT / "results" / "processed" / "outcomes_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r["model"], r["task_id"], r["sample_idx"])
            if k in seen or r.get("repair_arm"):
                continue
            seen.add(k)
            rows.append(r)

    lines = [r"\begin{tabular}{llrrrrl}", r"\toprule",
             r"Task & Tier & Gen. & Exec. & Aud. & Fals. & Rate (95\% CI) \quad power \\",
             r"\midrule"]
    last_tier = None
    for tid in sorted(tasks, key=lambda x: (tasks[x]["tier"], x)):
        sub = [r for r in rows if r["task_id"] == tid]
        if not sub:
            continue
        c = collections.Counter(r["outcome"] for r in sub)
        gen = len(sub)
        ex = gen - sum(c[o] for o in NOT_EXECUTABLE)
        aud = c["falsified"] + c["not_falsified"]
        f_ = c["falsified"]
        tier = tasks[tid]["tier"]
        if last_tier is not None and tier != last_tier:
            lines.append(r"\addlinespace")
        last_tier = tier

        pw = power.get(tid)
        pws = f"{pw:.2f}" if pw is not None else "--"
        if aud == 0:
            rate = r"\emph{no coverage}"
        else:
            lo, hi = wilson(f_, aud)
            rate = f"{f_/aud:.0%}".replace("%", r"\%") + f" [{lo:.2f}, {hi:.2f}]"
        lines.append(f"\\code{{{esc(tid.replace('_v1',''))}}} & {tier} & {gen} & {ex} & "
                     f"{aud} & {f_} & {rate} \\quad {pws} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(lines)-5} task rows)")


if __name__ == "__main__":
    main()
