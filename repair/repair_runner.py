"""Repair runner: take falsified programs from results/processed/outcomes.jsonl,
build R0 (self-repair) or R2 (counterexample) prompts, query the SAME model
that generated the code, and store repaired code for re-evaluation.

Usage:
  python repair/repair_runner.py --arm R2 --provider openai --base-url ...
  python analysis/aggregate.py --raw-glob 'results/raw/repair_*.jsonl'   # re-audit

Primary comparison in the paper: R2 vs R0 (paired by program).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generators.generation_runner import (  # noqa: E402
    call_anthropic, call_openai_compatible, extract_code)

TEMPLATES = {
    "R0": (ROOT / "repair" / "prompts" / "repair_self.txt").read_text(),
    "R2": (ROOT / "repair" / "prompts" / "repair_counterexample.txt").read_text(),
    "R3": (ROOT / "repair" / "prompts" / "repair_invariant.txt").read_text(),
}


def build_prompt(arm: str, row: dict, task: dict) -> str:
    if arm == "R0":
        return TEMPLATES["R0"].format(spec=task["spec"].strip(), code=row["code"],
                                      function_name=task["function_name"])
    ce = row["counterexample"]
    if arm == "R3":
        # E6: identical to R2 except for a diagnosis derived mechanically from
        # the audit result. Nothing here is hand-written per task; see
        # repair/diagnose.py, which is the whole vocabulary.
        from auditors.contract import load_contracts
        from repair.diagnose import diagnose
        contract = load_contracts()[task["id"]]
        return TEMPLATES["R3"].format(
            spec=task["spec"].strip(), code=row["code"],
            function_name=task["function_name"],
            epsilon=task["claimed_epsilon"], delta=task["claimed_delta"],
            adjacency=task["adjacency"],
            d1=ce["D"], d2=ce["D_prime"], event=ce["event"],
            p1_lb=f"{ce['p1_lower_bound']:.4f}",
            p2_ub=f"{ce['p2_upper_bound']:.4f}",
            eps_lb=f"{ce['empirical_eps_lower_bound']:.2f}",
            diagnosis=diagnose(task, contract, ce))
    return TEMPLATES["R2"].format(
        spec=task["spec"].strip(), code=row["code"],
        function_name=task["function_name"],
        epsilon=task["claimed_epsilon"], delta=task["claimed_delta"],
        adjacency=task["adjacency"],
        d1=ce["D"], d2=ce["D_prime"], event=ce["event"],
        p1_lb=f"{ce['p1_lower_bound']:.4f}", p2_ub=f"{ce['p2_upper_bound']:.4f}",
        eps_lb=f"{ce['empirical_eps_lower_bound']:.2f}",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["R0", "R2", "R3"], required=True)
    ap.add_argument("--outcomes", default=str(ROOT / "results" / "processed" / "outcomes.jsonl"))
    ap.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--tag", default=None,
                    help="scope output filename and --resume glob, e.g. 'frontier'. Use whenever the generator differs from a previous run.")
    ap.add_argument("--api-key-file", default=None,
                    help="path to a file holding only the API key")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="skip programs already repaired in this arm, appending to the "
                         "newest matching file")
    args = ap.parse_args()

    key_env = args.api_key_env or ("ANTHROPIC_API_KEY" if args.provider == "anthropic"
                                   else "OPENAI_API_KEY")
    from generators.apikey import resolve
    api_key, key_source = resolve(key_env, getattr(args, "api_key_file", None))
    if api_key:
        print(f"api key: {key_source}")

    tasks = {t["id"]: t for t in
             yaml.safe_load((ROOT / "benchmark" / "tasks.yaml").read_text())["tasks"]}
    with open(args.outcomes, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    # A pre-specified subset file (analysis/select_repair_subset.py) carries its
    # selection rule as a leading header record. Skip it; do not treat it as a
    # program.
    rows = [r for r in rows if not r.get("_header")]
    falsified = [r for r in rows if r["outcome"] == "falsified"][: args.limit]
    if not falsified:
        sys.exit("no falsified programs found - run analysis/aggregate.py first")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_dir = ROOT / "results" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Resume against the newest file for this arm. Both arms must cover the same
    # programs for the paired comparison to hold, so re-running after an
    # interruption has to top up rather than start a fresh partial file.
    # --tag scopes both the filename and the resume glob. Without it, a repair
    # run against a different generator would append into the same
    # repair_R2_*.jsonl as the local-model arms. The dedup key includes the
    # model so nothing would be logically lost, but one file holding two
    # populations is precisely the shape that produced the earlier
    # outcomes-glob contamination. Keep the populations in separate files.
    tag = f"{args.tag}_" if args.tag else ""
    done, out_path = set(), None
    if args.resume:
        existing = sorted(raw_dir.glob(f"repair_{args.arm}_{tag}*.jsonl"),
                          key=lambda q: q.stat().st_mtime, reverse=True)
        for q in existing:
            for line in q.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    if d.get("code"):
                        done.add((d["model"], d["task_id"], d["parent_sha256"]))
        if existing:
            out_path = existing[0]
        if done:
            print(f"resuming {args.arm}: {len(done)} already repaired")
    if out_path is None:
        out_path = raw_dir / f"repair_{args.arm}_{tag}{stamp}.jsonl"

    with open(out_path, "a") as out:
        for row in falsified:
            if (row["model"], row["task_id"], row["code_sha256"]) in done:
                continue
            task = tasks[row["task_id"]]
            prompt = build_prompt(args.arm, row, task)
            try:
                if args.provider == "anthropic":
                    raw = call_anthropic(api_key, row["model"], prompt,
                                         args.temperature, args.max_tokens)
                else:
                    raw = call_openai_compatible(args.base_url, api_key, row["model"],
                                                 prompt, args.temperature, args.max_tokens)
                code, err = extract_code(raw), None
            except Exception as e:
                raw, code, err = None, None, f"{type(e).__name__}: {e}"
            out.write(json.dumps({
                "task_id": row["task_id"], "function_name": task["function_name"],
                "model": row["model"], "provider": args.provider,
                "sample_idx": f"{row['sample_idx']}:{args.arm}",
                "repair_arm": args.arm, "parent_sha256": row["code_sha256"],
                "timestamp": dt.datetime.now().isoformat(),
                "code": code, "raw_response": raw, "api_error": err,
                "code_sha256": hashlib.sha256((code or "").encode()).hexdigest(),
            }) + "\n")
            out.flush()
            print(f"{args.arm} {row['model']:20s} {row['task_id']:32s} "
                  f"{'ERR' if err else 'ok'}")
    print(f"\nsaved -> {out_path}\nre-evaluate with:\n"
          f"  python analysis/aggregate.py --raw-glob '{out_path}'")


if __name__ == "__main__":
    main()
