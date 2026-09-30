"""Child process for sandbox.run_subprocess: applies rlimits, loads the
generated code (screened again defensively), runs ONE call, prints JSON."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sandbox.runner import load_mechanism  # noqa: E402


def _apply_rlimits(mem_mb: int):
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    except Exception:
        pass  # non-POSIX (Windows): rely on parent timeout only


def main():
    code_path, func_name, mem_mb = sys.argv[1], sys.argv[2], int(sys.argv[3])
    _apply_rlimits(mem_mb)
    import numpy as np
    code = Path(code_path).read_text(encoding="utf-8")
    args = json.loads(sys.stdin.read())          # e.g. [[1,0,1], 1.0]
    fn = load_mechanism(code, func_name)
    out = fn(*args, np.random.default_rng())     # fresh, unseeded rng
    if isinstance(out, (np.integer, np.floating)):
        out = float(out)
    elif isinstance(out, np.ndarray):
        out = out.tolist()
    elif isinstance(out, tuple):
        out = list(out)
    print(json.dumps({"ok": True, "result": out}))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
