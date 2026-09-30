"""Generation runner: prompt models to implement benchmark tasks, store JSONL.

Supports any OpenAI-compatible chat endpoint (OpenAI, OpenRouter, ollama,
llama.cpp server, vLLM) and Anthropic's native API.

Examples:
  # OpenAI
  OPENAI_API_KEY=... python generators/generation_runner.py \
      --model gpt-4.1-mini --provider openai --samples 3

  # Local model via ollama (no key needed)
  python generators/generation_runner.py --model qwen2.5-coder:7b \
      --provider openai --base-url http://localhost:11434/v1 --samples 3

  # Anthropic
  ANTHROPIC_API_KEY=... python generators/generation_runner.py \
      --model claude-sonnet-4-5 --provider anthropic --samples 3

Freeze policy: model id, temperature, max_tokens, prompt template, and this
file's git hash are recorded per sample. Do not edit prompts mid-experiment.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import sys
import time

import requests
import urllib3.util.connection
import yaml

# The Google endpoint resolves to IPv6 first, and IPv6 connections from this
# machine hang until they time out (~40 s) before urllib3 falls back to IPv4.
# That made every hosted request take minutes (found 2026-09-29). Resolving IPv4
# only changes the transport, not the request, so it cannot affect any output.
urllib3.util.connection.HAS_IPV6 = False

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Run as a script, so the project root is not on sys.path by default and
# `from generators...` would fail.
sys.path.insert(0, str(ROOT))
PROMPT_TEMPLATE = (ROOT / "generators" / "prompts" / "minimal_spec.txt").read_text()

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
# Gemma 4 served through the OpenAI-compatible endpoint returns its reasoning
# inline as <thought>...</thought>, and that reasoning often contains a DRAFT
# code block. Taking the first fenced block would then record the draft rather
# than the answer. Stripping the thought first is a no-op for every model in the
# original corpus, none of which emits this tag (verified 2026-09-29).
THOUGHT_RE = re.compile(r"<thought>.*?</thought>", re.DOTALL)


def extract_code(text: str) -> str:
    text = THOUGHT_RE.sub("", text)
    m = CODE_BLOCK_RE.search(text)
    return (m.group(1) if m else text).strip()


class QuotaExhausted(RuntimeError):
    """A per-day quota was hit. Retrying will not help; the run must stop."""


#: Token usage reported by the last successful call. Recorded per row so a paid
#: run can be costed from what the provider actually billed rather than from a
#: character-count estimate. Reasoning tokens in particular are invisible to any
#: estimate made locally: they appear only as the gap between total_tokens and
#: prompt+completion.
LAST_USAGE: dict = {}


def _quota_detail(r) -> str:
    """Pull the limit and quota id out of a Google quota error, for the message."""
    try:
        err = r.json()
        if isinstance(err, list):
            err = err[0]
        err = err.get("error", {})
        bits = [err.get("message", "").split("\n")[0]]
        for d in err.get("details", []):
            for v in d.get("violations", []):
                bits.append(f"{v.get('quotaId')} = {v.get('quotaValue')} "
                            f"for {v.get('quotaDimensions', {}).get('model')}")
        return " | ".join(b for b in bits if b)
    except Exception:                                        # noqa: BLE001
        return r.text[:300]


def call_openai_compatible(base_url, api_key, model, prompt, temperature, max_tokens,
                           max_retries: int = 5):
    """POST one completion, retrying on rate limits and transient server errors.

    E5 runs against a free-tier hosted endpoint, where 429 is a normal part of
    operation rather than a failure. Without backoff the run would record a
    string of API errors and quietly under-sample that model, which would then
    look like a model that produced less code rather than a quota we hit. The
    server's Retry-After is honoured when present.
    """
    delay = 2.0
    for attempt in range(max_retries + 1):
        r = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key or 'none'}"},
            json={"model": model, "temperature": temperature, "max_tokens": max_tokens,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=300,
        )
        # A per-DAY quota is not a transient condition. Backing off and retrying
        # against it wastes about a minute per task and then records an error
        # row, so a run that hits it grinds through every remaining task
        # producing nothing. Worse, those error rows land in the corpus with
        # whatever task ordering the quota happened to die on, which is how you
        # end up with 10 samples of task 1 and none of tasks 4-16. Stop instead.
        if r.status_code == 429 and "PerDay" in r.text:
            raise QuotaExhausted(_quota_detail(r))

        if r.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
            wait = delay
            hdr = r.headers.get("Retry-After")
            if hdr:
                try:
                    wait = max(wait, float(hdr))
                except ValueError:
                    pass
            print(f"    HTTP {r.status_code}, retrying in {wait:.0f}s "
                  f"({attempt + 1}/{max_retries})", flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 120.0)
            continue
        r.raise_for_status()
        d = r.json()
        global LAST_USAGE
        LAST_USAGE = d.get("usage", {}) or {}
        choice = d.get("choices", [{}])[0]
        content = (choice.get("message") or {}).get("content")
        finish = choice.get("finish_reason")

        # Reasoning models spend part of max_tokens on internal reasoning that
        # never appears in the response, so a budget that is generous for a
        # non-reasoning model can return truncated code or no content at all.
        # Returning "" here would be recorded as an empty program and later
        # scored as a syntax failure, turning our own token budget into what
        # looks like a finding about the model. Fail loudly instead: the runner
        # records this as an api_error, which is excluded from rates rather than
        # counted against the model.
        if not content:
            usage = d.get("usage", {})
            raise RuntimeError(
                f"empty completion (finish_reason={finish!r}, "
                f"completion_tokens={usage.get('completion_tokens')}, "
                f"total_tokens={usage.get('total_tokens')}). If finish_reason is "
                f"'length', max_tokens={max_tokens} was consumed before any text "
                f"was emitted; raise it.")
        if finish == "length":
            print(f"    WARNING: response hit max_tokens={max_tokens} and is "
                  f"truncated; the recorded program is incomplete", flush=True)
        return content


def call_anthropic(api_key, model, prompt, temperature, max_tokens):
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        json={"model": model, "temperature": temperature, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=300,
    )
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--api-key-env", default=None,
                    help="env var holding the key (default: OPENAI_API_KEY / ANTHROPIC_API_KEY)")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--tasks", nargs="*", default=None, help="task ids (default: all)")
    ap.add_argument("--tasks-file", default=None,
                    help="alternate benchmark YAML, e.g. the E8 perturbations. Keeps a derived task set out of the main benchmark file.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sample-major", action="store_true",
                    help="sample 0 of every task, then sample 1, and so on. "
                         "Use under a request quota: stopping early then costs "
                         "samples per task instead of whole tasks.")
    ap.add_argument("--api-key-file", default=None,
                    help="path to a file holding only the API key")
    ap.add_argument("--resume", action="store_true",
                    help="skip (task, sample) pairs already generated for this model in "
                         "results/raw/, appending to the newest matching file. Long runs "
                         "here get interrupted, and regenerating costs model time for no "
                         "new data.")
    args = ap.parse_args()

    key_env = args.api_key_env or ("ANTHROPIC_API_KEY" if args.provider == "anthropic"
                                   else "OPENAI_API_KEY")
    # Falls back to a key file in the home directory, so a hosted run does not
    # require re-exporting the key in every shell. Local ollama needs no key at
    # all, which is why a missing key is only fatal for anthropic below.
    from generators.apikey import resolve
    api_key, key_source = resolve(key_env, getattr(args, "api_key_file", None))
    if api_key:
        print(f"api key: {key_source}")
    if args.provider == "anthropic" and not api_key:
        sys.exit(f"set {key_env}")

    tasks_path = (pathlib.Path(args.tasks_file) if args.tasks_file
                  else ROOT / "benchmark" / "tasks.yaml")
    tasks = yaml.safe_load(tasks_path.read_text(encoding="utf-8"))["tasks"]
    print(f"tasks: {tasks_path.name} ({len(tasks)})")
    if args.tasks:
        tasks = [t for t in tasks if t["id"] in set(args.tasks)]

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = args.model.replace('/', '_').replace(':', '_')
    raw_dir = ROOT / "results" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    done = set()
    out_path = None
    if args.resume:
        # Reuse the newest file for this model and skip what it already holds.
        #
        # An explicit --out must win. Without this, --resume discovered the
        # newest gen_<model>_*.jsonl and appended there, silently ignoring
        # --out. Running a DERIVED task set (E8) under the same model as the
        # main corpus (E5) therefore wrote the derived programs straight into
        # the main corpus file. Task ids made it recoverable; nothing in the
        # run said it had happened.
        if args.out:
            explicit = pathlib.Path(args.out)
            existing = [explicit] if explicit.exists() else []
        else:
            existing = sorted(raw_dir.glob(f"gen_{safe_model}_*.jsonl"),
                              key=lambda q: q.stat().st_mtime, reverse=True)
        for q in existing:
            for line in q.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("model") == args.model and r.get("code"):
                    done.add((r.get("task_id"), r.get("sample_idx")))
        if existing:
            out_path = existing[0]
        if done:
            print(f"  resuming: {len(done)} (task, sample) pairs already skipped",
                  flush=True)

    if out_path is None:
        out_path = pathlib.Path(args.out or raw_dir / f"gen_{safe_model}_{stamp}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Second guard, independent of the first. If the target file already holds
    # programs for tasks that are not in this run's task set, we are about to
    # mix two populations in one file. Refuse rather than produce a corpus whose
    # shape only shows up later as an odd task count.
    if out_path.exists():
        want = {t["id"] for t in tasks}
        have = set()
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    have.add(json.loads(line).get("task_id"))
                except json.JSONDecodeError:
                    continue
        foreign = {t for t in have if t and t not in want}
        if foreign:
            sys.exit(
                f"refusing to write: {out_path.name} already holds {len(foreign)} "
                f"task id(s) outside this run's task set "
                f"({', '.join(sorted(foreign)[:4])}...). Two populations in one "
                f"file is how a derived task set contaminates a main corpus. "
                f"Pass a different --out.")

    # Order the work sample-major when asked: sample 0 of every task, then
    # sample 1 of every task, and so on. Task-major order is fine with an
    # unlimited budget, but under a daily quota it fails catastrophically. It
    # spends the whole budget on the first few tasks and leaves the rest at
    # zero, so the partial corpus is not a smaller version of the full one, it
    # is a biased one. Sample-major degrades gracefully: stopping early costs
    # samples per task rather than whole tasks.
    plan = [(task, i) for task in tasks for i in range(args.samples)]
    if args.sample_major:
        plan = [(task, i) for i in range(args.samples) for task in tasks]

    stopped_early = None
    with open(out_path, "a") as out:
        for task, i in plan:
            prompt = PROMPT_TEMPLATE.format(spec=task["spec"].strip(),
                                            function_name=task["function_name"])
            if True:
                if (task["id"], i) in done:
                    continue
                t0 = time.time()
                try:
                    if args.provider == "anthropic":
                        raw = call_anthropic(api_key, args.model, prompt,
                                             args.temperature, args.max_tokens)
                    else:
                        raw = call_openai_compatible(args.base_url, api_key, args.model,
                                                     prompt, args.temperature,
                                                     args.max_tokens)
                    code = extract_code(raw)
                    err = None
                except QuotaExhausted as e:
                    # Do not record a row. An error row here would be
                    # indistinguishable from the model failing to produce code,
                    # and it would occupy the (task, sample) slot so --resume
                    # would not retry it tomorrow.
                    stopped_early = str(e)
                    break
                except Exception as e:
                    raw, code, err = None, None, f"{type(e).__name__}: {e}"
                rec = {
                    "task_id": task["id"], "function_name": task["function_name"],
                    "model": args.model, "provider": args.provider,
                    "sample_idx": i, "temperature": args.temperature,
                    "max_tokens": args.max_tokens, "timestamp": dt.datetime.now().isoformat(),
                    "elapsed_s": round(time.time() - t0, 1),
                    "usage": dict(LAST_USAGE) if not err else None,
                    "code": code, "raw_response": raw, "api_error": err,
                    "code_sha256": hashlib.sha256((code or "").encode()).hexdigest(),
                }
                out.write(json.dumps(rec) + "\n")
                out.flush()
                status = "ERR" if err else "ok"
                print(f"[{task['id']:32s}] sample {i} {status} ({rec['elapsed_s']}s)")

    print(f"\nsaved -> {out_path}")
    if stopped_early:
        print("\n" + "=" * 70)
        print("STOPPED: daily quota exhausted, not a model failure.")
        print(f"  {stopped_early}")
        print("\nNothing was recorded for the request that failed, so re-running")
        print("this exact command after the quota resets will pick up where it")
        print("stopped. Check coverage per task before using a partial corpus:")
        print("  python analysis/coverage.py --model " + args.model)
        print("=" * 70)
        sys.exit(2)


if __name__ == "__main__":
    main()
