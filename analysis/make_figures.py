"""The two figures assessment 6 asked for.

  fig_waterfall.pdf   coverage-aware outcome waterfall, per model
  fig_reversal.pdf    the model-ranking reversal under two denominators

Both are generated from the frozen results, like the macros, so a figure cannot
drift away from the table beside it.

    python analysis/make_figures.py
"""
from __future__ import annotations

import collections
import glob
import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "paper"

NOT_EXECUTABLE = {"syntax_fail", "screen_fail", "runtime_fail", "timeout"}
FUNCTIONAL_FAIL = {"semantic_fail"}
UNSUPPORTED = {"auditor_unsupported", "auditor_error"}

SHORT = {"qwen2.5-coder:7b": "qwen2.5-coder", "deepseek-coder:6.7b": "deepseek-coder",
         "codellama:7b": "codellama"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 200,
    # Matplotlib defaults to Type 3 fonts in PDF output. Several venues, PoPETs
    # among them, require Type 1 or TrueType and will desk-reject a submission
    # containing Type 3. 42 selects TrueType; 3 is the default and is wrong for
    # publication. This costs nothing and is invisible in the rendered figure,
    # which is exactly why it is easy to ship broken.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load():
    rows, seen = [], set()
    for f in sorted(glob.glob(str(ROOT / "results" / "processed" / "outcomes_*.jsonl"))):
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


def stages(sub):
    c = collections.Counter(r["outcome"] for r in sub)
    gen = len(sub)
    not_exec = sum(c[o] for o in NOT_EXECUTABLE)
    func_fail = sum(c[o] for o in FUNCTIONAL_FAIL)
    unsup = sum(c[o] for o in UNSUPPORTED)
    return {
        "generated": gen,
        "not_executable": not_exec,
        "functional_fail": func_fail,
        "unsupported": unsup,
        "not_falsified": c["not_falsified"],
        "falsified": c["falsified"],
    }


def fig_waterfall(rows):
    models = ["codellama:7b", "deepseek-coder:6.7b", "qwen2.5-coder:7b"]
    order = [("falsified", "#b03030", "falsified"),
             ("not_falsified", "#4a7ba7", "audited, not falsified"),
             ("unsupported", "#9aa6b2", "auditor unsupported"),
             ("functional_fail", "#d9a441", "fails functional tests"),
             ("not_executable", "#cfcfcf", "does not execute")]

    fig, ax = plt.subplots(figsize=(6.4, 2.5))
    ypos = range(len(models))
    lefts = [0] * len(models)
    for key, colour, label in order:
        vals = [stages([r for r in rows if r["model"] == m])[key] for m in models]
        ax.barh(list(ypos), vals, left=lefts, color=colour, label=label,
                height=0.62, edgecolor="white", linewidth=0.6)
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v >= 5:
                ax.text(l + v / 2, i, str(v), ha="center", va="center", fontsize=8,
                        color="white" if colour in ("#b03030", "#4a7ba7") else "#333")
        lefts = [l + v for l, v in zip(lefts, vals)]

    ax.set_yticks(list(ypos))
    ax.set_yticklabels([SHORT[m] for m in models])
    ax.set_xlabel("programs generated (80 per model)")
    ax.set_xlim(0, 80)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3,
              frameon=False, fontsize=8, handlelength=1.2)
    ax.set_title("Coverage-aware outcome waterfall", fontsize=10, loc="left", pad=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_waterfall.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote paper/fig_waterfall.pdf")


def fig_reversal(rows):
    """The ranking inverts between two denominators. Plot both with intervals."""
    models = ["qwen2.5-coder:7b", "deepseek-coder:6.7b", "codellama:7b"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)

    for ax, (den, title) in zip(axes, [("generated", "per generated program"),
                                       ("executable", "per executable program")]):
        pts, los, his, ns = [], [], [], []
        for m in models:
            s = stages([r for r in rows if r["model"] == m])
            n = s["generated"] if den == "generated" else s["generated"] - s["not_executable"]
            k = s["falsified"]
            lo, hi = wilson(k, n)
            pts.append(k / n)
            los.append(k / n - lo)
            his.append(hi - k / n)
            ns.append(n)
        y = range(len(models))
        ax.errorbar(pts, list(y), xerr=[los, his], fmt="o", color="#b03030",
                    capsize=3, markersize=5, linewidth=1.2)
        for i, (p, n) in enumerate(zip(pts, ns)):
            ax.text(p, i + 0.22, f"{p:.0%}  (n={n})", ha="center", fontsize=7.5,
                    color="#444")
        ax.set_xlim(0.18, 0.90)
        ax.set_ylim(-0.75, len(models) - 0.25)
        ax.set_yticks(list(y))
        ax.set_yticklabels([SHORT[m] for m in models])
        ax.set_xlabel("falsification rate")
        # Spread between the extreme point estimates - the quantity that grows,
        # and the honest summary: the ordering moves, the separation does not
        # become significant.
        spread = max(pts) - min(pts)
        ax.set_title(f"{title}\nspread across models: {spread*100:.0f} points",
                     fontsize=9, loc="left", color="#333")
        ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)

    # The model whose rank flips, marked below its own row in each panel.
    axes[0].text(0.355, 1.62, "lowest rate", ha="center", fontsize=7.5,
                 color="#1a7a3a")
    axes[1].text(0.683, 1.62, "highest rate", ha="center", fontsize=7.5,
                 color="#b03030")

    fig.suptitle("Point-estimate order inverts with the denominator; "
                 "no pairwise difference is significant", fontsize=9.5,
                 x=0.02, ha="left", y=1.06)
    fig.tight_layout()
    fig.savefig(OUT / "fig_reversal.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote paper/fig_reversal.pdf")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    rows = load()
    fig_waterfall(rows)
    fig_reversal(rows)
