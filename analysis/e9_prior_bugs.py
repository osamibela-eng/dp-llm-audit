"""E9 -- the auditor against bugs found by PRIOR auditing work.

Reviewer 64A asked why the method was not evaluated on existing buggy programs
found by previous auditing approaches. This is that evaluation.

The programs are the StatDP benchmark suite (Ding et al., CCS 2018), vendored
unchanged in third_party/statdp/statdp/algorithms.py. The same suite is the
shared test set of DP-Finder, CheckDP and DP-Sniper, and its incorrect SVT
variants are the ones analysed by Lyu, Su and Li (VLDB 2017). Nothing here is
written by us except the wrapper, so the ground truth is external.

Ground truth is per (algorithm, claimed epsilon), not per algorithm:
histogram_eps adds Laplace noise of scale epsilon instead of 1/epsilon, which
under-noises only when epsilon < 1. At epsilon = 1.5 its scale (1.5) exceeds the
required 1/1.5, so it genuinely satisfies the claim and must NOT be falsified.
Labelling it "buggy" everywhere would score a correct negative as a miss.

Both auditors see the same neighbouring inputs: the ones StatDP's own
generate_databases() builds (query vectors of length 5 and 10 whose entries
differ by at most one), under the sensitivity preset StatDP's benchmark uses for
that algorithm.

    ours    falsify() at n_confirm 30000, then 3 fresh-seed confirmations
            (same budget and seeds as the main campaign); label = reproducible
    StatDP  detect_counterexample() at the claimed epsilon, p < 0.05, defaults

    python analysis/e9_prior_bugs.py                 # both auditors
    python analysis/e9_prior_bugs.py --skip-statdp   # ours only (fast)
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
# analysis/coverage.py (our corpus checker) shadows the `coverage` package when
# this directory is on sys.path, and numba -- imported by StatDP -- then crashes
# on `coverage.types`. Drop the script directory before importing StatDP.
_HERE = pathlib.Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if pathlib.Path(p or ".").resolve() != _HERE]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "statdp"))

from auditors.simple_auditor import falsify  # noqa: E402
from statdp import algorithms as A  # noqa: E402
from statdp.generators import ALL_DIFFER, ONE_DIFFER, generate_databases  # noqa: E402

CONFIRM_SEEDS = [20260821, 777, 31337]          # analysis/confirm_seeds.py SEEDS[:3]
CLAIMED = (0.2, 0.7, 1.5)                       # StatDP benchmark's claimed levels

# (function, extra kwargs, sensitivity preset) exactly as in
# third_party/statdp/examples/benchmark.py
SUITE = [
    (A.noisy_max_v1a, {}, ALL_DIFFER),
    (A.noisy_max_v1b, {}, ALL_DIFFER),
    (A.noisy_max_v2a, {}, ALL_DIFFER),
    (A.noisy_max_v2b, {}, ALL_DIFFER),
    (A.histogram, {}, ONE_DIFFER),
    (A.histogram_eps, {}, ONE_DIFFER),
    (A.SVT, {"N": 1, "T": 0.5}, ALL_DIFFER),
    (A.iSVT1, {"T": 1, "N": 1}, ALL_DIFFER),
    (A.iSVT2, {"T": 1, "N": 1}, ALL_DIFFER),
    (A.iSVT3, {"T": 1, "N": 1}, ALL_DIFFER),
    (A.iSVT4, {"T": 1, "N": 1}, ALL_DIFFER),
]

# What each variant actually satisfies, from the source comments in
# algorithms.py and from Lyu et al. 2017 for the SVT family.
DEFECT = {
    "noisy_max_v1a": None,
    "noisy_max_v1b": "releases the noisy maximum value instead of its index",
    "noisy_max_v2a": None,
    "noisy_max_v2b": "releases the noisy maximum value instead of its index",
    "histogram": None,
    "histogram_eps": "Laplace scale eps instead of 1/eps",
    "SVT": None,
    "iSVT1": "no noise on the queries",
    "iSVT2": "query noise does not scale with N; no cap on positive answers",
    "iSVT3": "query noise does not scale with N (Lyu et al.: ((1+6N)/4)eps-DP)",
    "iSVT4": "releases the noisy query value instead of True",
}


def truly_violates(name: str, eps: float) -> bool:
    if DEFECT[name] is None:
        return False
    if name == "histogram_eps":
        return eps < 1.0 / eps          # under-noised iff scale eps < required 1/eps
    return True


class Wrapped:
    """Adapt StatDP's algo(prng, queries, epsilon, **kw) to our mech(data, eps, rng)."""

    def __init__(self, fn, kw):
        self.fn, self.kw = fn, kw

    def __call__(self, data, eps, rng):
        out = self.fn(rng, data, eps, **self.kw)
        if isinstance(out, tuple):      # iSVT4 returns (count, last); last may be False
            return [float(x) for x in out]
        return float(out)


def pairs_for(fn, kw, sens):
    pairs = []
    for n in (5, 10):
        for d1, d2, _ in generate_databases(fn, n, {**kw, "epsilon": 1.0}, sens):
            pairs.append((list(d1), list(d2)))
    return pairs


def audit_ours(fn, kw, sens, eps, n_confirm):
    mech, pairs = Wrapped(fn, kw), pairs_for(fn, kw, sens)
    t0 = time.time()
    first = falsify(mech, pairs, eps, 0.0, n_confirm=n_confirm)
    rec = {"campaign": first.status,
           "eps_lb": (first.counterexample.eps_lower_bound
                      if first.counterexample else None),
           "event": first.counterexample.event if first.counterexample else None}
    confirms = []
    if first.status == "falsified":
        for s in CONFIRM_SEEDS:
            confirms.append(falsify(mech, pairs, eps, 0.0, n_confirm=n_confirm,
                                    seed=s).status)
    rec["confirmations"] = confirms
    rec["reproducible"] = (first.status == "falsified"
                           and all(c == "falsified" for c in confirms))
    rec["seconds"] = round(time.time() - t0, 1)
    rec["n_pairs"] = len(pairs)
    return rec


def audit_statdp(fn, kw, sens, eps, cores):
    from statdp import detect_counterexample
    t0 = time.time()
    res = detect_counterexample(fn, (eps,), {**kw, "epsilon": eps},
                                sensitivity=sens, cores=cores, quiet=True)
    p = res[0][1]
    return {"p": p, "falsified": p < 0.05, "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-confirm", type=int, default=30_000)
    ap.add_argument("--skip-statdp", action="store_true")
    ap.add_argument("--cores", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "results" / "processed" / "e9_prior_bugs.jsonl"))
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    done = set()
    if out.exists():
        for l in out.read_text(encoding="utf-8").splitlines():
            if l.strip():
                d = json.loads(l)
                done.add((d["algorithm"], d["claimed_eps"]))

    with open(out, "a", encoding="utf-8") as fh:
        for fn, kw, sens in SUITE:
            for eps in CLAIMED:
                if (fn.__name__, eps) in done:
                    continue
                rec = {"algorithm": fn.__name__, "claimed_eps": eps,
                       "defect": DEFECT[fn.__name__],
                       "truly_violates": truly_violates(fn.__name__, eps),
                       "sensitivity": sens.name, "ours": audit_ours(fn, kw, sens, eps, a.n_confirm)}
                if not a.skip_statdp:
                    rec["statdp"] = audit_statdp(fn, kw, sens, eps, a.cores)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                o = rec["ours"]
                s = rec.get("statdp", {})
                print(f"{fn.__name__:14s} eps={eps:<4} truth={'VIOL' if rec['truly_violates'] else 'ok  '} "
                      f"ours={'REPRO' if o['reproducible'] else o['campaign']:14s} "
                      f"lb={o['eps_lb'] if o['eps_lb'] is None else round(o['eps_lb'], 2)} "
                      f"statdp_p={s.get('p')}", flush=True)


if __name__ == "__main__":
    main()
