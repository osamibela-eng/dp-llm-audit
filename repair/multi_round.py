"""E7: does repair work if you let the model keep trying?

E6 gave each model one attempt per condition and found a floor: 4-6% repaired,
no condition separating from another. The obvious rebuttal is that one shot is
not a fair test. A developer would not give up after one try, they would feed
the new failure back and go again.

E7 runs that. Each round takes the program the model produced last round, audits
it, and if it is still falsified hands back the FRESH counterexample from that
audit. So the model is always being corrected on its current mistake rather than
its original one. Up to --rounds attempts per program.

Two things this design gets right that a naive loop would not:

  * The counterexample is regenerated every round. Replaying round 1's
    counterexample against round 3's program would be feeding back a failure
    that no longer exists, and any conclusion from it would be meaningless.
  * A program that stops running is a dead end, not a repair. Once a round
    produces code that does not execute or does not parse, that program is done:
    it cannot be audited, so there is no counterexample to feed back. It is
    recorded as failed at that round rather than silently dropped.

Reported as a cumulative repair rate by round, which is the quantity the
rebuttal is actually about: given k attempts, how many get fixed?

    python repair/multi_round.py --rounds 3 \
        --outcomes results/processed/repair_subset.jsonl \
        --base-url http://localhost:11434/v1 --tag local
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import pathlib
import sys
import time

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auditors.simple_auditor import falsify  # noqa: E402
import generators.generation_runner as _gen  # noqa: E402
from generators.generation_runner import (  # noqa: E402
    call_anthropic, call_openai_compatible, extract_code)
from repair.repair_runner import build_prompt  # noqa: E402
from sandbox.runner import load_mechanism  # noqa: E402

OUT = ROOT / "results" / "processed"


def audit(code: str, task: dict, n_confirm: int, seed: int):
    """Audit one program. Returns (status, counterexample_dict_or_None)."""
    try:
        fn = load_mechanism(code, task["function_name"])
    except Exception:                                        # noqa: BLE001
        return "syntax_fail", None
    try:
        res = falsify(fn, [tuple(p) for p in task["auditor"]["pairs"]],
                      float(task["claimed_epsilon"]),
                      float(task.get("claimed_delta") or 0.0),
                      n_confirm=n_confirm, seed=seed)
    except Exception:                                        # noqa: BLE001
        return "runtime_fail", None
    ce = res.counterexample.as_dict() if res.counterexample else None
    return res.status, ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--arm", default="R2", choices=["R0", "R2", "R3"],
                    help="feedback condition held constant across rounds")
    ap.add_argument("--outcomes", default=str(OUT / "repair_subset.jsonl"))
    ap.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--api-key-file", default=None)
    ap.add_argument("--tag", default="local")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--n-confirm", type=int, default=30_000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    key_env = a.api_key_env or ("ANTHROPIC_API_KEY" if a.provider == "anthropic"
                                else "OPENAI_API_KEY")
    from generators.apikey import resolve
    api_key, key_source = resolve(key_env, a.api_key_file)
    if api_key:
        print(f"api key: {key_source}")

    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml")
                            .read_text(encoding="utf-8"))["tasks"]}

    rows = [json.loads(l) for l in
            pathlib.Path(a.outcomes).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    rows = [r for r in rows if not r.get("_header")]
    seeds = [r for r in rows if r["outcome"] == "falsified"][: a.limit]
    if not seeds:
        sys.exit("no falsified programs in --outcomes")

    out_path = OUT / f"e7_multi_round_{a.tag}.jsonl"
    done = set()
    if a.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["parent_sha256"])
        print(f"resuming: {len(done)} programs already done")

    print(f"{len(seeds)} programs, up to {a.rounds} rounds each, arm {a.arm}, "
          f"n_confirm={a.n_confirm}\n", flush=True)

    t_start = time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        for i, seed_row in enumerate(seeds, 1):
            if seed_row["code_sha256"] in done:
                continue
            task = tasks[seed_row["task_id"]]
            cur = dict(seed_row)          # carries code + counterexample
            history = []
            outcome = "still_falsified"

            for rnd in range(1, a.rounds + 1):
                prompt = build_prompt(a.arm, cur, task)
                try:
                    if a.provider == "anthropic":
                        raw = call_anthropic(api_key, seed_row["model"], prompt,
                                             a.temperature, a.max_tokens)
                    else:
                        raw = call_openai_compatible(
                            a.base_url, api_key, seed_row["model"], prompt,
                            a.temperature, a.max_tokens)
                    code = extract_code(raw)
                except Exception as e:                       # noqa: BLE001
                    history.append({"round": rnd, "status": "api_error",
                                    "detail": f"{type(e).__name__}: {e}"})
                    outcome = "api_error"
                    break

                status, ce = audit(code, task, a.n_confirm, seed=7_000 + rnd)
                history.append({"round": rnd, "status": status,
                                "code_sha256": hashlib.sha256(
                                    code.encode()).hexdigest(),
                                "usage": dict(_gen.LAST_USAGE) or None})

                if status == "not_falsified":
                    outcome = "repaired"
                    break
                if status in ("syntax_fail", "runtime_fail", "unsupported", "error"):
                    # No audit means no counterexample, so there is nothing to
                    # feed back. Stop rather than re-sending stale feedback.
                    outcome = status
                    break
                # still falsified: carry the NEW program and its NEW counterexample
                cur = {**cur, "code": code, "counterexample": ce}

            rec = {"parent_sha256": seed_row["code_sha256"],
                   "task_id": seed_row["task_id"], "model": seed_row["model"],
                   "arm": a.arm, "rounds_allowed": a.rounds,
                   "rounds_used": len(history), "outcome": outcome,
                   "history": history,
                   "timestamp": dt.datetime.now().isoformat()}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{i:3d}/{len(seeds)}] {seed_row['task_id']:28s} "
                  f"{seed_row['model']:22s} {outcome:16s} "
                  f"after {len(history)} round(s)", flush=True)

    print(f"\n{(time.time() - t_start) / 60:.1f} min")
    report(out_path)


def report(out_path):
    rows = [json.loads(l) for l in
            pathlib.Path(out_path).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if not rows:
        return
    n = len(rows)
    allowed = max(r["rounds_allowed"] for r in rows)

    print("\n" + "=" * 74)
    print("E7  DOES REPAIR WORK IF YOU LET IT KEEP TRYING?")
    print("=" * 74)
    print(f"\n  programs: {n}   rounds allowed: {allowed}")

    print(f"\n  {'after round':14s} {'repaired':>9s} {'cumulative rate':>16s}")
    cum = 0
    for rnd in range(1, allowed + 1):
        fixed = sum(1 for r in rows
                    if r["outcome"] == "repaired" and r["rounds_used"] == rnd)
        cum += fixed
        print(f"  {rnd:<14d} {fixed:9d} {cum}/{n} = {cum / n:>7.0%}")

    # The auditor reports "error" when the mechanism raises during sampling,
    # which is the same condition aggregate.py calls runtime_fail. Group them:
    # from the loop's point of view all of these are the same dead end, a
    # program that cannot be audited and therefore yields no counterexample to
    # feed back into the next round.
    DEAD = ("syntax_fail", "runtime_fail", "unsupported", "error")
    dead = [r for r in rows if r["outcome"] in DEAD]
    still = [r for r in rows if r["outcome"] == "still_falsified"]
    errs = [r for r in rows if r["outcome"] == "api_error"]
    print(f"\n  still falsified after all rounds : {len(still):3d}")
    print(f"  died on unrunnable code          : {len(dead):3d}")
    if dead:
        kinds = collections.Counter(r["outcome"] for r in dead)
        print("    " + ", ".join(f"{k}={v}" for k, v in kinds.most_common()))
    if errs:
        print(f"  api errors (not the model's fault): {len(errs):3d}")

    # The comparison that matters against the single-shot arms in E6: how many
    # were repaired on ROUND ONE. Anything beyond that is what iteration buys.
    r1 = sum(1 for r in rows if r["outcome"] == "repaired" and r["rounds_used"] == 1)
    later = cum - r1
    print("\n  " + "-" * 70)
    print(f"  repaired on round 1 alone : {r1}/{n} = {r1 / n:.0%}")
    print(f"  added by rounds 2 and 3   : {later}/{n} = {later / n:.0%}")
    print("\n  The round-1 figure is the like-for-like comparison with the")
    print("  single-shot conditions in E6, which also gave one attempt and")
    print("  counted unrunnable output against the arm. Anything above it is")
    print("  what iteration buys, and it is the part E6 could not have seen.")

    print("\n  'died on unrunnable code' means a round produced something that")
    print("  does not parse or does not execute. That program cannot be audited,")
    print("  so there is no counterexample to feed back and the loop stops. It")
    print("  counts as a failure, not as missing data.")

    if cum == 0:
        print("\n  No program was repaired at any round. Extra attempts did not")
        print("  convert a single failure, so the one-shot result in E6 was not")
        print("  an artifact of stopping too early.")


if __name__ == "__main__":
    main()
