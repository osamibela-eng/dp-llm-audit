"""Sandboxing for untrusted LLM-generated code.

Two layers, used together:

1. `screen_code`  - static AST screening: import allowlist, banned names/attributes.
   Cheap, runs on every generated program before anything executes.
2. `load_mechanism` - executes the screened code in a namespace with restricted
   builtins and returns the target function (FAST path, used for auditing,
   which needs ~10^5 calls and cannot afford one subprocess per call).
3. `run_subprocess`  - executes one call in a separate resource-limited process
   (SLOW path, used for first-contact smoke tests of each program).

Honesty note for the paper + README: AST screening + restricted builtins is a
research-grade sandbox, not a security boundary. For the full experiment runs,
execute the whole pipeline inside a container (docker run --network=none ...);
the harness itself then needs no further isolation. Keep API keys OUT of the
environment that executes generated code.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ALLOWED_IMPORTS = {"numpy", "math"}
BANNED_NAMES = {
    "eval", "exec", "compile", "open", "input", "__import__", "globals",
    "locals", "vars", "getattr", "setattr", "delattr", "exit", "quit",
    "breakpoint", "help", "memoryview",
}
BANNED_ATTR_SUBSTRINGS = ("__globals__", "__builtins__", "__subclasses__",
                          "__bases__", "__mro__", "__code__", "__reduce__")


class ScreenError(Exception):
    pass


def screen_code(code: str) -> ast.Module:
    """Raise ScreenError if the code violates the static policy; return the AST."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ScreenError(f"syntax error: {e}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise ScreenError(f"disallowed import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                raise ScreenError(f"disallowed import: from {node.module}")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise ScreenError(f"banned name: {node.id}")
        elif isinstance(node, ast.Attribute):
            if any(b in node.attr for b in BANNED_ATTR_SUBSTRINGS):
                raise ScreenError(f"banned attribute: {node.attr}")
    return tree


import builtins as _bi

_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float",
    "frozenset", "int", "isinstance", "issubclass", "iter", "len", "list", "map",
    "max", "min", "next", "pow", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "type", "zip", "print",
    "ValueError", "TypeError", "Exception", "ArithmeticError", "ZeroDivisionError",
    "KeyError", "IndexError", "StopIteration", "RuntimeError", "OverflowError",
)
_SAFE_BUILTINS = {n: getattr(_bi, n) for n in _SAFE_BUILTIN_NAMES}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Guarded __import__: only the allowlisted modules resolve inside the sandbox."""
    if name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"import of {name!r} is not allowed in the sandbox")
    return _bi.__import__(name, globals, locals, fromlist, level)


_SAFE_BUILTINS["__import__"] = _safe_import


def load_mechanism(code: str, func_name: str):
    """Screen then exec the code with restricted builtins; return the function."""
    screen_code(code)
    import math
    import numpy
    ns = {"__builtins__": dict(_SAFE_BUILTINS), "np": numpy, "numpy": numpy, "math": math}
    exec(compile(code, "<generated>", "exec"), ns)  # noqa: S102 - screened above
    if func_name not in ns or not callable(ns[func_name]):
        raise ScreenError(f"function {func_name!r} not defined")
    return ns[func_name]


def run_subprocess(code: str, func_name: str, args_json: str,
                   timeout_s: int = 10, mem_mb: int = 512) -> dict:
    """Run one call `func(*json.loads(args_json))` in an isolated subprocess.
    A numpy Generator is appended as the last argument automatically.
    Returns {"ok": bool, "result": ..., "error": ...}.
    """
    child = Path(__file__).with_name("exec_child.py")
    # encoding="utf-8" is required, not cosmetic: generated programs contain
    # literal Greek epsilon in comments and docstrings, and Windows defaults
    # this handle to cp1252, which cannot encode it.
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                     encoding="utf-8") as f:
        f.write(code)
        code_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, str(child), code_path, func_name, str(mem_mb)],
            input=args_json, capture_output=True, text=True, timeout=timeout_s,
            # Same reason as the temp file above, on the way back: a traceback
            # from generated code can carry any character, and decoding it under
            # the Windows locale would raise inside the harness rather than in
            # the program being judged.
            encoding="utf-8", errors="replace",
        )
        if proc.returncode != 0:
            return {"ok": False, "error": (proc.stderr or "nonzero exit")[-2000:]}
        return json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout_s}s"}
    finally:
        Path(code_path).unlink(missing_ok=True)
