"""Run the full evaluation over generated programs and aggregate outcomes.

Reads generation JSONL files from results/raw/, and for each program assigns
exactly ONE final outcome label:

  syntax_fail | screen_fail | runtime_fail | timeout | semantic_fail |
  auditor_unsupported | falsified | not_falsified | auditor_error

Then prints the per-model primary table (the paper's Table 1 skeleton) and
writes results/processed/outcomes.jsonl (+ counterexamples for repair).

Usage:
  python analysis/aggregate.py                       # evaluate all raw files
  python analysis/aggregate.py --n-audit 30000       # audit sample budget
  python analysis/aggregate.py --dedupe              # audit unique ASTs only

Deduplication: near-identical generations (same normalized AST) share one audit
result; the duplication rate itself is a reportable metric.
"""
from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import json
import pathlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auditors.simple_auditor import falsify              # noqa: E402
from sandbox.runner import (ScreenError, load_mechanism,      # noqa: E402
                            run_subprocess)


def normalized_ast_hash(code: str) -> str | None:
    try:
        tree = ast.parse(code)
        return hashlib.sha256(ast.dump(tree, annotate_fields=False).encode()).hexdigest()
    except SyntaxError:
        return None


#: Draws used by the non-degeneracy screen. See smoke_and_semantic.
SCREEN_DRAWS = 200


def smoke_and_semantic(fn, task) -> str | None:
    """Cheap semantic screening (not the full pytest suite): run the mechanism
    on each auditor pair input; must not raise; repeated calls must vary
    (mechanisms are randomized). Returns an outcome label or None if OK."""
    rng = np.random.default_rng()
    try:
        for d1, d2 in task["auditor"]["pairs"]:
            fn(list(d1), task["claimed_epsilon"], rng)
            fn(list(d2), task["claimed_epsilon"], rng)
    except Exception:
        return "runtime_fail"
    # Non-degeneracy: a randomized mechanism must not always return the same
    # value. The number of draws is not cosmetic. At 12 draws this screen fires
    # on CORRECT low-entropy mechanisms at a rate that is easy to compute and
    # far from negligible: a randomised response that reports truthfully with
    # probability 0.75 returns 12 identical answers about 3.2% of the time, and
    # measurement confirmed 3 of 8 exclusions in an earlier run were spurious.
    #
    # A false exclusion is not benign: it removes a working program from the
    # audited set and silently moves the denominator of every reported rate.
    # SCREEN_DRAWS is therefore set so that the false-positive rate is
    # negligible even for the most concentrated output in the benchmark:
    # 0.95^200 is about 3e-5. The cost is 200 cheap calls per program.
    try:
        outs = {repr(fn(list(task["auditor"]["pairs"][0][0]),
                        task["claimed_epsilon"], rng))
                for _ in range(SCREEN_DRAWS)}
    except Exception:
        return "runtime_fail"
    if len(outs) == 1:
        return "semantic_fail"  # genuinely deterministic release
    return None


def liveness_check(code: str, func_name: str, n_params: int, task: dict,
                   timeout_s: int = 10) -> str | None:
    """Does the program terminate at all? Returns "timeout" or None.

    Added after a codellama program hung the codellama audit for nearly two
    hours:

        while max(counts.values()) - min(counts.values()) < epsilon:

    The loop body never moves the counts, so it never exits. The auditor calls
    the mechanism ~10^5 times through the FAST path (`load_mechanism`, an
    in-process call with no timeout, because one subprocess per call would make
    auditing impossible), and a single non-terminating call blocks the whole run
    with no error and no partial output.

    The slow path already existed and is documented for exactly this — "one call
    in a separate resource-limited process" — it simply was not wired in. Every
    program now makes one supervised call per pair side before the fast path is
    used.

    Residual risk, stated rather than solved: a program that terminates on these
    inputs may still fail to terminate on some other draw. Bounding that would
    need a supervised call per audit sample, which costs more than the audit. A
    non-terminating program is far more likely to hang on every input than on a
    rare one, and the two pair sides are the inputs the auditor actually uses.
    """
    for d1, d2 in task["auditor"]["pairs"]:
        for d in (d1, d2):
            args = [list(d), task["claimed_epsilon"]]
            if n_params >= 4:
                args.append(task["claimed_delta"])
            try:
                r = run_subprocess(code, func_name, json.dumps(args),
                                   timeout_s=timeout_s)
            except Exception:
                # The gate must never be able to end the run. Anything that goes
                # wrong setting up the supervised call is not evidence about the
                # program, so fall through and let the normal path judge it. This
                # is defensive by design: the whole codellama audit was lost once
                # to an unhandled UnicodeEncodeError raised right here, on a
                # program that merely had a Greek epsilon in a comment.
                return None
            if not r.get("ok") and "timeout" in str(r.get("error", "")):
                return "timeout"
    return None


def _adapt_signature(fn, task):
    """Tasks with delta > 0 use signature (data, epsilon, delta, rng)."""
    import inspect
    try:
        n_params = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        n_params = 3
    if n_params >= 4:
        delta = task["claimed_delta"]
        return lambda data, eps, rng: fn(data, eps, delta, rng)
    return fn


def evaluate(rec: dict, task: dict, n_audit: int, seed, second_eps: float | None = 0.3) -> dict:
    code = rec.get("code")
    if not code:
        return {"outcome": "syntax_fail", "detail": rec.get("api_error", "no code")}
    if normalized_ast_hash(code) is None:
        return {"outcome": "syntax_fail"}
    try:
        fn = load_mechanism(code, rec["function_name"])
    except ScreenError as e:
        return {"outcome": "screen_fail", "detail": str(e)}
    except Exception as e:
        return {"outcome": "runtime_fail", "detail": f"{type(e).__name__}: {e}"}

    import inspect
    try:
        n_params = len(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        n_params = 3
    # Supervised first contact, BEFORE any unbounded in-process call.
    if liveness_check(code, rec["function_name"], n_params, task) == "timeout":
        return {"outcome": "timeout",
                "detail": "did not terminate within 10s on an auditor pair input"}

    fn = _adapt_signature(fn, task)

    label = smoke_and_semantic(fn, task)
    if label:
        # deterministic programs still go to the auditor: determinism IS a violation
        if label == "runtime_fail":
            return {"outcome": label}

    pairs = [tuple(p) for p in task["auditor"]["pairs"]]
    # Audit at the claimed epsilon AND at a second epsilon (default 0.3):
    # at eps=1.0, scale=epsilon vs scale=1/epsilon coincide, hiding inversion
    # bugs - the spec requires correctness for whatever epsilon is passed.
    eps_settings = [task["claimed_epsilon"]]
    if second_eps and abs(second_eps - task["claimed_epsilon"]) > 1e-9:
        eps_settings.append(second_eps)
    audit_time, unsupported_detail = 0.0, None
    for eps in eps_settings:
        r = falsify(fn, pairs, eps, task["claimed_delta"], n_confirm=n_audit, seed=seed)
        audit_time += r.elapsed_s
        if r.status == "falsified":
            return {"outcome": "falsified", "falsified_at_eps": eps,
                    "counterexample": r.counterexample.as_dict(),
                    "audit_elapsed_s": round(audit_time, 1)}
        if r.status == "unsupported":
            unsupported_detail = r.detail
        elif r.status == "error":
            return {"outcome": "auditor_error", "detail": r.detail}
    if unsupported_detail is not None:
        return {"outcome": "auditor_unsupported", "detail": unsupported_detail}
    out = "semantic_fail" if label == "semantic_fail" else "not_falsified"
    return {"outcome": out, "audit_elapsed_s": round(audit_time, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-glob", default=str(ROOT / "results" / "raw" / "gen_*.jsonl"))
    ap.add_argument("--tasks-file", default=None,
                    help="alternate benchmark YAML, e.g. the E8 perturbations")
    ap.add_argument("--n-audit", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dedupe", action="store_true")
    ap.add_argument("--out", default=None,
                    help="output path; defaults to results/processed/outcomes.jsonl. "
                         "Set this for repair re-audits so they do not overwrite the "
                         "main run's outcomes (raw JSONL is append-only, but this "
                         "processed file is not).")
    ap.add_argument("--second-eps", type=float, default=0.3,
                    help="also audit at this epsilon (0 disables); see AUDITOR_NOTES.md #6")
    args = ap.parse_args()

    tasks_path = (pathlib.Path(args.tasks_file) if args.tasks_file
                  else ROOT / "benchmark" / "tasks.yaml")
    tasks = {t["id"]: t for t in
             yaml.safe_load(tasks_path.read_text(encoding="utf-8"))["tasks"]}
    print(f"tasks: {tasks_path.name} ({len(tasks)})")

    records = []
    for path in sorted(glob.glob(args.raw_glob)):
        with open(path) as f:
            records += [json.loads(line) for line in f]
    # Record-level de-duplication, distinct from --dedupe (which collapses
    # identical ASTs to measure how often a model re-emits the same program).
    # A (model, task_id, sample_idx) triple should appear ONCE: it names one
    # slot in the experimental design. Duplicates arise when an interrupted run
    # is relaunched without --resume, or when a pilot run overlaps the main one,
    # and they silently inflate the denominator of every rate we report.
    # Keep the most recent record for each slot.
    before = len(records)
    slots = {}
    for rec in records:
        k = (rec.get("model"), rec.get("task_id"), rec.get("sample_idx"))
        prev = slots.get(k)
        if prev is None or str(rec.get("timestamp", "")) >= str(prev.get("timestamp", "")):
            slots[k] = rec
    records = list(slots.values())
    if before != len(records):
        print(f"  dropped {before - len(records)} duplicate (model, task, sample) "
              f"records; {len(records)} slots remain")

    if not records:
        sys.exit("no generation files found in results/raw/ - run the generator first")

    cache: dict[tuple, dict] = {}
    out_rows = []
    for rec in records:
        task = tasks[rec["task_id"]]
        key = (rec["task_id"], normalized_ast_hash(rec.get("code") or ""))
        if args.dedupe and key in cache and key[1] is not None:
            result = dict(cache[key], deduped=True)
        else:
            result = evaluate(rec, task, args.n_audit, args.seed,
                              second_eps=(args.second_eps or None))
            cache[key] = result
        # Repair records carry the arm and the hash of the program they were
        # asked to fix. Both must survive into the outcome file or the paired
        # R0-vs-R2 comparison cannot be reconstructed - the outcome would say
        # what happened without saying to which program, or under which arm.
        passthrough = {k: rec[k] for k in ("repair_arm", "parent_sha256")
                       if k in rec}
        out_rows.append({**{k: rec[k] for k in
                            ("task_id", "model", "sample_idx", "code_sha256")},
                         **passthrough,
                         "code": rec.get("code"), **result})
        print(f"{rec['model']:24s} {rec['task_id']:32s} #{rec['sample_idx']} "
              f"-> {result['outcome']}")

    out = (Path(args.out) if args.out
           else ROOT / "results" / "processed" / "outcomes.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")

    df = pd.DataFrame(out_rows)
    table = df.groupby(["model", "outcome"]).size().unstack(fill_value=0)
    print("\n=== primary table (counts) ===")
    print(table.to_string())
    dup = df.duplicated(subset=["task_id", "code_sha256"]).mean()
    print(f"\nexact-duplicate generation rate: {dup:.1%}")
    print(f"outcomes -> {out}")


if __name__ == "__main__":
    main()
