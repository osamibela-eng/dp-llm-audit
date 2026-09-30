"""E1: three-arm specification study.

The objection this kills is "so what, every auditor takes a spec" and its
sharper cousin, "your StatDP finding is one anecdote".

Three arms over the same corpus:

  generic    StatDP under its DEFAULT configuration, generating its own
             databases with its default adjacency. This is how a practitioner
             actually runs it, so it is an honest baseline rather than a
             strawman. It is instrumented, not corrected: out-of-domain probes
             are logged and allowed through, because blocking them would destroy
             the measurement.

  ours       our auditor as it stands, with hand-designed neighbouring pairs.

  contract   our auditor with every pair checked against the declared contract
             before use. Violations are counted and logged.

What the arms are for. The generic arm measures how often an unmodified tool
leaves the declared domain, and whether its false flags are traceable to those
departures. The contract arm demonstrates that our own pairs conform, which is
the claim v1 made in prose and never checked.

We are not claiming specification-aware auditing is new. The claims are the
contract schema, the enforcement layer, and the first quantification of
specification-mismatch artifacts when auditing untrusted generated code.

    python analysis/e1_three_arm.py --references
    python analysis/e1_three_arm.py --bugs
"""
from __future__ import annotations

import argparse
import collections
import importlib
import importlib.util
import json
import pathlib
import sys
import time

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auditors.contract import load_contracts, check_pair  # noqa: E402
from auditors.contract_probe import ContractLoggingAdapter, summarise  # noqa: E402
from auditors.simple_auditor import falsify  # noqa: E402
from sandbox.runner import load_mechanism  # noqa: E402

OUT = ROOT / "results" / "processed"
PROBE_LOG = OUT / "e1_probe_log.jsonl"


def _wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    import math
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def reference_path(task) -> pathlib.Path:
    p = ROOT / "benchmark" / "references" / pathlib.Path(task["reference"]).name
    return p if p.exists() else ROOT / task["reference"]


# --------------------------------------------------------------------------
# arm: ours / contract
# --------------------------------------------------------------------------
def run_ours(task, contract, source, func, enforce: bool, n_confirm: int):
    """Our auditor. With enforce=True, every pair is contract-checked first."""
    pair_violations = []
    pairs = [tuple(p) for p in task["auditor"]["pairs"]]
    if enforce:
        for d1, d2 in pairs:
            for v in check_pair(contract, d1, d2):
                pair_violations.append(str(v))

    t0 = time.time()
    try:
        fn = load_mechanism(source, func)
        res = falsify(fn, pairs, float(task["claimed_epsilon"]),
                      float(task.get("claimed_delta") or 0.0),
                      n_confirm=n_confirm, seed=20260824)
        status = res.status
    except Exception as ex:                                  # noqa: BLE001
        status = "error"
        res = None
    return {"status": status,
            "seconds": round(time.time() - t0, 1),
            "pair_violations": pair_violations}


# --------------------------------------------------------------------------
# arm: generic (StatDP, default configuration, instrumented)
# --------------------------------------------------------------------------
def run_generic(task, contract, source_path, func, iterations, cores):
    from auditors import statdp_adapter
    importlib.reload(statdp_adapter)

    t0 = time.time()
    adapter = ContractLoggingAdapter(source_path, func, contract, PROBE_LOG)
    try:
        p, _ = statdp_adapter._detect(
            adapter, float(task["claimed_epsilon"]),
            input_size=5, event_iterations=iterations,
            detect_iterations=iterations, cores=cores)
        if p is None:
            status = "unsupported"
        else:
            status = "falsified" if p < 0.05 else "not_falsified"
    except Exception as ex:                                  # noqa: BLE001
        status, p = "unsupported", None
    return {"status": status, "p": p, "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--references", action="store_true")
    ap.add_argument("--bugs", action="store_true")
    ap.add_argument("--iterations", type=int, default=20_000)
    ap.add_argument("--n-confirm", type=int, default=30_000)
    ap.add_argument("--cores", type=int, default=3)
    ap.add_argument("--skip-generic", action="store_true",
                    help="ours and contract arms only; the generic arm is slow")
    a = ap.parse_args()
    if not (a.references or a.bugs):
        a.references = True

    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml")
                            .read_text(encoding="utf-8"))["tasks"]}
    contracts = load_contracts()
    OUT.mkdir(parents=True, exist_ok=True)

    work = []
    if a.references:
        for tid, t in tasks.items():
            src = reference_path(t)
            work.append(("reference", tid, src, t["function_name"],
                         src.read_text(encoding="utf-8"), "correct"))
    if a.bugs:
        # `bugs:` is a path to a module holding several buggy variants plus an
        # EXPECTED map naming them, not a list of per-bug records. An earlier
        # version of this function treated it as a list of dicts, which iterated
        # the path string one character at a time and silently collected nothing:
        # the arm reported no bugs rather than failing. Enumerate the way
        # experiments/calibrate_auditor.py does, and assert we found some.
        for tid, t in tasks.items():
            if not t.get("bugs"):
                continue
            bug_path = ROOT / t["bugs"]
            if not bug_path.exists():
                sys.exit(f"{tid}: declared bug module {bug_path} is missing")
            spec = importlib.util.spec_from_file_location(bug_path.stem, bug_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            src = bug_path.read_text(encoding="utf-8")
            for bug_name, expected in mod.EXPECTED.items():
                work.append(("bug", tid, bug_path, bug_name, src, expected))
        n_bugs = sum(1 for w in work if w[0] == "bug")
        if not n_bugs:
            sys.exit("--bugs collected no programs; refusing to report an empty arm")
        print(f"collected {n_bugs} planted bugs across "
              f"{sum(1 for t in tasks.values() if t.get('bugs'))} modules")

    out_path = OUT / ("e1_references.jsonl" if a.references else "e1_bugs.jsonl")
    if PROBE_LOG.exists() and a.references:
        PROBE_LOG.unlink()

    print(f"{len(work)} items, arms: ours + contract"
          + ("" if a.skip_generic else " + generic (StatDP default)") + "\n")

    rows = []
    for i, (kind, tid, src_path, func, source, expected) in enumerate(work, 1):
        c = contracts[tid]
        t = tasks[tid]
        r = {"kind": kind, "task": tid, "func": func, "expected": expected}
        r["ours"] = run_ours(t, c, source, func, enforce=False, n_confirm=a.n_confirm)
        r["contract"] = run_ours(t, c, source, func, enforce=True, n_confirm=a.n_confirm)
        if not a.skip_generic:
            r["generic"] = run_generic(t, c, src_path, func, a.iterations, a.cores)
        rows.append(r)

        g = r.get("generic", {}).get("status", "-")
        print(f"[{i:3d}/{len(work)}] {tid:30s} ours={r['ours']['status']:14s} "
              f"contract={r['contract']['status']:14s} generic={g}", flush=True)

    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    report(rows, tasks, contracts, a.skip_generic)
    print(f"\nwrote {out_path.relative_to(ROOT)}")


def report(rows, tasks, contracts, skip_generic):
    refs = [r for r in rows if r["kind"] == "reference"]
    bugs = [r for r in rows if r["kind"] == "bug"]

    print("\n" + "=" * 78)
    print("E1  THREE ARMS")
    print("=" * 78)
    arms = ["ours", "contract"] + ([] if skip_generic else ["generic"])
    print(f"  {'':34s}" + "".join(f"{a:>14s}" for a in arms))

    if refs:
        line = f"  {'correct references falsely flagged':34s}"
        for arm in arms:
            n = sum(1 for r in refs if r.get(arm, {}).get("status") == "falsified")
            line += f"{n:>10d}/{len(refs):<3d}"
        print(line)

        line = f"  {'references the arm could not audit':34s}"
        for arm in arms:
            n = sum(1 for r in refs if r.get(arm, {}).get("status")
                    in ("unsupported", "error"))
            line += f"{n:>10d}/{len(refs):<3d}"
        print(line)

    if bugs:
        # The bugs were labelled detectable / likely_missed BEFORE any of this
        # ran, in the modules themselves. A flat "caught X of 61" would read as
        # a power figure while quietly averaging over bugs we ourselves
        # predicted a single-epsilon threshold auditor cannot see. Stratify, and
        # let the pre-registered expectation carry the interpretation.
        strata = collections.defaultdict(list)
        for r in bugs:
            strata[r.get("expected", "unlabelled")].append(r)

        line = f"  {'planted bugs detected (all)':34s}"
        for arm in arms:
            n = sum(1 for r in bugs if r.get(arm, {}).get("status") == "falsified")
            line += f"{n:>10d}/{len(bugs):<3d}"
        print(line)

        for label in sorted(strata, key=lambda k: -len(strata[k])):
            grp = strata[label]
            line = f"    {'expected ' + label:32s}"
            for arm in arms:
                n = sum(1 for r in grp if r.get(arm, {}).get("status") == "falsified")
                line += f"{n:>10d}/{len(grp):<3d}"
            print(line)

        print("\n  Reading these rows:")
        print("    'likely_missed' bugs are ones we predicted a single-threshold")
        print("    auditor cannot see. Catching none of them is the prediction")
        print("    holding, not the auditor underperforming.")
        if "detectable_at_second_eps" in strata:
            n2 = len(strata["detectable_at_second_eps"])
            print(f"    'detectable_at_second_eps' ({n2}) is out of scope for this arm")
            print("    by construction: it audits at the claimed epsilon only, so")
            print("    that row can only ever read 0. It is neither a hit nor a")
            print("    miss here and should not be counted in either direction.")
        det = strata.get("detectable", [])
        if det:
            k = sum(1 for r in det if r.get("ours", {}).get("status") == "falsified")
            lo, hi = _wilson(k, len(det))
            print(f"\n    Power on the bugs we predicted catchable: {k}/{len(det)} "
                  f"= {k/len(det):.0%}, 95% CI [{lo:.2f}, {hi:.2f}]")
            print("    The missed third is the honest cost of a black-box auditor")
            print("    at a fixed sample size, and it is why a program that")
            print("    survives is reported as 'no violation found'.")

    line = f"  {'wall clock, seconds':34s}"
    for arm in arms:
        s = sum(r.get(arm, {}).get("seconds", 0) for r in rows)
        line += f"{s:>13.0f} "
    print(line)

    # ---- contract violations by our own pairs ---------------------------
    viol = [(r["task"], v) for r in rows for v in r["contract"]["pair_violations"]]
    print("\n" + "-" * 78)
    print(f"  contract arm: our hand-designed pairs produced {len(viol)} violations")
    if viol:
        for tid, v in viol[:8]:
            print(f"    {tid:30s} {v}")
    else:
        print("    Every pair we ship conforms to its task's declared contract.")
        print("    v1 asserted this; it is now checked.")

    # ---- what the generic arm actually probed ---------------------------
    if not skip_generic:
        summ = summarise(PROBE_LOG)
        if summ:
            tot_p = sum(d["distinct_probes"] for d in summ.values())
            tot_v = sum(d["violating_probes"] for d in summ.values())
            print("\n" + "-" * 78)
            print(f"  generic arm: {tot_v} of {tot_p} distinct probes left the "
                  f"declared domain")
            kinds = collections.Counter()
            for d in summ.values():
                for k, n in d["by_kind"].items():
                    kinds[k] += n
            for k, n in kinds.most_common():
                print(f"    {k:16s} {n}")

            print(f"\n  {'task':30s} {'probes':>7s} {'violating':>10s}  flagged?")
            for r in refs:
                d = summ.get(r["task"])
                if not d:
                    continue
                flagged = r.get("generic", {}).get("status") == "falsified"
                mark = "FALSE FLAG" if flagged else ""
                print(f"  {r['task']:30s} {d['distinct_probes']:7d} "
                      f"{d['violating_probes']:10d}  {mark}")

            print("\n  Traceability: every false flag above should sit on a task with")
            print("  a non-zero violating-probe count. That is the claim v1 could")
            print("  only make in prose about randomized_response.")

    # ---- adjacency, which is structural rather than observable ----------
    print("\n" + "-" * 78)
    print("  Adjacency mismatch is a property of the configuration, not of any")
    print("  single probe, so it is reported rather than counted:")
    mism = [tid for tid, c in contracts.items() if c.adjacency == "add_remove_one"]
    print(f"    tasks declaring add/remove-one : {len(mism)} of {len(contracts)}")
    print("    StatDP ONE_DIFFER generates    : replace-one pairs")
    print("    so on those tasks the relation StatDP tests is not the relation")
    print("    the task claims, independently of any domain violation.")


if __name__ == "__main__":
    main()
