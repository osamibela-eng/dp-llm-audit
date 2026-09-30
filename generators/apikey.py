"""Resolve an API key without it ever touching the repository or a transcript.

Three ways a key gets leaked in a project like this, and what we do about each:

  pasted into a chat or an issue   -> read it from a file the user creates
                                      themselves, so it is never typed anywhere
                                      that gets logged
  committed with the code          -> the file lives in the user's home
                                      directory, outside the repo, so no
                                      gitignore rule has to be remembered
  set once and forgotten           -> the environment variable still wins, so
                                      CI and one-off shells work unchanged

Lookup order: the environment variable, then an explicit --api-key-file, then
the default file for that variable name in the home directory.

    from generators.apikey import resolve
    key = resolve("GEMINI_API_KEY")
"""
from __future__ import annotations

import os
import pathlib


def default_key_path(env_name: str) -> pathlib.Path:
    """~/.gemini_api_key for GEMINI_API_KEY, and so on."""
    return pathlib.Path.home() / ("." + env_name.lower())


def resolve(env_name: str, explicit_file: str | None = None) -> tuple[str, str]:
    """Return (key, where_it_came_from). Key is '' when nothing was found."""
    key = os.environ.get(env_name, "").strip()
    if key:
        return key, f"environment variable {env_name}"

    candidates = []
    if explicit_file:
        candidates.append(pathlib.Path(explicit_file))
    candidates.append(default_key_path(env_name))

    for p in candidates:
        try:
            if p.is_file():
                # A key pasted into a text editor picks up a trailing newline,
                # and sometimes quotes. Strip both rather than sending them.
                text = p.read_text(encoding="utf-8-sig").strip()
                text = text.strip('"').strip("'").strip()
                if text:
                    return text, f"file {p}"
        except OSError:
            continue

    return "", "nowhere"


def describe_setup(env_name: str) -> str:
    """The instructions to print when no key was found."""
    p = default_key_path(env_name)
    return (
        f"No key found for {env_name}.\n\n"
        f"Easiest fix, and it keeps the key out of your shell history and out\n"
        f"of any transcript:\n\n"
        f"  1. Open Notepad.\n"
        f"  2. Paste the key as the only thing in the file.\n"
        f"  3. Save it as exactly:  {p}\n"
        f"     (in the Save dialog set 'Save as type' to 'All Files', or\n"
        f"      Notepad will silently append .txt and this will not find it)\n\n"
        f"That path is in your home directory, not in the repository, so it\n"
        f"cannot be committed by accident.\n\n"
        f"Alternative, for this terminal only:\n"
        f'  $env:{env_name} = "your-key"'
    )
