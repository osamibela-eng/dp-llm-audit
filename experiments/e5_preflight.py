"""One cheap request against the frontier endpoint, to fail fast and cheaply.

E5 runs on a free-tier quota, so a misconfigured 160-request run is an hour lost
and a day of quota burned for nothing. This sends a single request and, when it
fails, says which of the three usual causes it is instead of printing a stack
trace.

    python experiments/e5_preflight.py
    python experiments/e5_preflight.py --model gemini-2.5-flash
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generators.apikey import resolve, describe_setup  # noqa: E402

DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--api-key-env", default="GEMINI_API_KEY")
    ap.add_argument("--api-key-file", default=None,
                    help="path to a file holding only the key")
    a = ap.parse_args()

    key, where = resolve(a.api_key_env, a.api_key_file)
    print(f"endpoint : {a.base_url}")
    print(f"model    : {a.model}")
    print(f"key env  : {a.api_key_env}")

    if not key:
        print()
        print(describe_setup(a.api_key_env))
        sys.exit(1)
    # Never print the key. First four characters are enough to tell a Google
    # key from something pasted by mistake, and short enough to be useless.
    print(f"key      : found in {where}, {len(key)} chars, starts {key[:4]}...")

    try:
        r = requests.post(
            f"{a.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": a.model, "max_tokens": 400,
                  "messages": [{"role": "user",
                                "content": "Reply with exactly: preflight ok"}]},
            timeout=60,
        )
    except requests.RequestException as e:
        print(f"\nFAIL: could not reach the endpoint.\n  {type(e).__name__}: {e}")
        sys.exit(1)

    if r.status_code == 200:
        try:
            d = r.json()
            choice = d.get("choices", [{}])[0]
        except ValueError:
            print(f"\nFAIL: 200 but the body was not JSON:\n{r.text[:400]}")
            sys.exit(1)

        msg = ((choice.get("message") or {}).get("content") or "").strip()
        usage = d.get("usage", {})
        prompt_t = usage.get("prompt_tokens") or 0
        visible_t = usage.get("completion_tokens") or 0
        total_t = usage.get("total_tokens") or 0
        hidden = total_t - prompt_t - visible_t

        if not msg:
            print(f"\nFAIL: HTTP 200 but no text came back "
                  f"(finish_reason={choice.get('finish_reason')!r}, "
                  f"completion_tokens={visible_t}, total_tokens={total_t}).")
            if choice.get("finish_reason") == "length":
                print("\nCause: this is a reasoning model. Its internal reasoning")
                print("is billed against max_tokens but never appears in the reply,")
                print("so the entire budget was spent before any text was emitted.")
                print("Raise --max-tokens. Generation needs several thousand.")
            sys.exit(1)

        print(f"\nOK: model replied {msg!r}")
        print(f"  tokens: prompt={prompt_t}  visible={visible_t}  "
              f"hidden reasoning={hidden}")
        if hidden > 0:
            print(f"\n  {hidden} tokens went to reasoning you never see, and they")
            print("  count against max_tokens. Budget generation accordingly: too")
            print("  small a budget returns truncated code, which would be scored")
            print("  as the model's mistake rather than as our configuration.")
        print("\nSafe to run the full generation now. Keep --resume on it.")
        return

    body = r.text[:500]
    print(f"\nFAIL: HTTP {r.status_code}\n{body}\n")
    if r.status_code in (401, 403):
        print("Cause: the key was rejected. Re-copy it from aistudio.google.com;")
        print("a truncated paste or a stray quote is the usual reason.")
    elif r.status_code == 404:
        print(f"Cause: no model named {a.model!r} at this endpoint. Model names")
        print("change. Check the current list in AI Studio and pass --model.")
    elif r.status_code == 429:
        print("Cause: quota already exhausted for now. This is not a")
        print("misconfiguration; wait for the window to reset and retry.")
    sys.exit(1)


if __name__ == "__main__":
    main()
