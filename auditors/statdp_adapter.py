"""StatDP adapter - second, independent auditor (Ding et al., CCS 2018).

Setup:
    git clone https://github.com/cmla-psu/statdp.git third_party/statdp
    pip install numba tqdm
    # do NOT `pip install -e .` (its packaging is stale); import from the clone.

StatDP's mechanism contract differs from ours:
    algorithm(prng, queries, **kwargs) where `queries` is a numeric vector and
    neighboring inputs are generated internally from `num_input` and sensitivity
    presets. This adapter wraps our (data, epsilon, rng) mechanisms whose data
    is a NUMERIC list. Label-list tasks (histogram/argmax) need an encoding -
    start with the numeric tasks.

Interpretation: detect_counterexample returns [(test_eps, p_value, D, D', kwargs,
event)]. Small p-value => falsified at test_eps.

WINDOWS NOTE (found 2026-08-20, this machine)
---------------------------------------------
StatDP parallelises with multiprocessing.Pool. On Windows the start method is
`spawn`, so every object handed to the pool must be picklable and importable in
a fresh interpreter. The obvious wrapper -- a closure over the mechanism --
fails with:

    AttributeError: Can't get local object 'audit_numeric_task.<locals>.wrapped'

and a function loaded via importlib.spec_from_file_location fails too, because
the child cannot re-import a module that was never on sys.path under a real name.

Fix: `_MechAdapter` below stores only strings (a source path and a function
name) and re-imports the mechanism lazily inside whichever process calls it.
Strings pickle; closures do not. This also happens to be exactly what we need
for LLM-generated code, which arrives as source text rather than as an
importable module -- `audit_source` writes it to a temp file and audits that.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATDP_DIR = ROOT / "third_party" / "statdp"


def available() -> bool:
    return (STATDP_DIR / "statdp").exists()


def _signature_probe(prng, queries, epsilon):  # pragma: no cover - never called
    """Exists only to donate a __code__ object with StatDP's expected signature.

    statdp/generators.py:46 does
        algorithm.__code__.co_varnames[:algorithm.__code__.co_argcount]
    to discover which keyword arguments a mechanism takes. A callable class
    instance has no __code__, so we lend it this function's.
    """
    raise NotImplementedError


class _MechAdapter:
    """Picklable bridge between our mechanism contract and StatDP's.

    Holds a file path and a function name -- both strings, so this survives
    pickling to a spawned worker -- and imports the mechanism on first call in
    whatever process that turns out to be.
    """

    def __init__(self, source_path: str, func_name: str):
        self.source_path = str(source_path)
        self.func_name = func_name
        # statdp reads `algorithm.__name__` for logging and `algorithm.__code__`
        # for argument discovery; instances have neither by default.
        self.__name__ = func_name
        self.__code__ = _signature_probe.__code__
        self._fn = None

    def _load(self):
        spec = importlib.util.spec_from_file_location(
            f"_statdp_mech_{uuid.uuid4().hex[:8]}", self.source_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._fn = getattr(mod, self.func_name)

    def __call__(self, prng, queries, epsilon):
        # StatDP supplies `queries` (the dataset vector) and its own RandomState;
        # our mechanisms expect a Generator - both expose .laplace etc.
        if self._fn is None:
            self._load()
        return self._fn(list(queries), epsilon, prng)

    def __getstate__(self):
        # never ship the loaded function across the process boundary
        return {"source_path": self.source_path, "func_name": self.func_name,
                "__name__": self.__name__}

    def __setstate__(self, state):
        self.source_path = state["source_path"]
        self.func_name = state["func_name"]
        self.__name__ = state["__name__"]
        # code objects do not pickle; re-attach in the child process
        self.__code__ = _signature_probe.__code__
        self._fn = None


def _detect(adapter, claimed_eps, input_size, event_iterations, detect_iterations,
            cores):
    if not available():
        raise RuntimeError(
            "StatDP not found. Run:\n"
            "  git clone https://github.com/cmla-psu/statdp.git third_party/statdp\n"
            "  pip install numba tqdm")
    if str(STATDP_DIR) not in sys.path:
        sys.path.insert(0, str(STATDP_DIR))
    from statdp import detect_counterexample, ONE_DIFFER  # noqa: PLC0415

    # ADJACENCY (finding, 2026-08-20): StatDP defaults to sensitivity=ALL_DIFFER,
    # under which every entry of the input vector differs by one -- L1 distance
    # equal to the vector length, not 1. Our mechanisms are calibrated for a
    # single record changing. Auditing them under ALL_DIFFER falsifies CORRECT
    # references (measured p = 0.000 on the reference laplace_count), which is
    # not a bug in the mechanism but a mismatch between the auditor's neighbour
    # model and the task's declared adjacency. ONE_DIFFER matches tasks.yaml.
    results = detect_counterexample(
        adapter, test_epsilon=claimed_eps,
        default_kwargs={"epsilon": claimed_eps},
        num_input=(input_size,), event_iterations=event_iterations,
        detect_iterations=detect_iterations, cores=cores,
        sensitivity=ONE_DIFFER, quiet=True)
    test_eps, p_value, d1, d2, kwargs, event = results[0]
    return p_value, results[0]


def audit_file(source_path, func_name: str, claimed_eps: float,
               input_size: int = 5, event_iterations: int = 50_000,
               detect_iterations: int = 50_000, cores: int | None = None):
    """Audit a mechanism that already lives in a .py file."""
    return _detect(_MechAdapter(source_path, func_name), claimed_eps,
                   input_size, event_iterations, detect_iterations, cores)


def audit_source(source: str, func_name: str, claimed_eps: float,
                 input_size: int = 5, event_iterations: int = 50_000,
                 detect_iterations: int = 50_000, cores: int | None = None):
    """Audit mechanism SOURCE TEXT - the LLM-generated case.

    The source is written to a temp file so spawned workers can import it.
    """
    tmp = pathlib.Path(tempfile.gettempdir()) / f"statdp_mech_{uuid.uuid4().hex}.py"
    tmp.write_text(source, encoding="utf-8")
    try:
        return _detect(_MechAdapter(tmp, func_name), claimed_eps, input_size,
                       event_iterations, detect_iterations, cores)
    finally:
        tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    ref = ROOT / "benchmark" / "references" / "laplace_count.py"
    bugs = ROOT / "benchmark" / "known_bugs" / "laplace_count_bugs.py"

    p_ok, _ = audit_file(ref, "private_count", 1.0,
                         event_iterations=20_000, detect_iterations=20_000)
    print(f"correct laplace_count      : p={p_ok:.3f}   (expect > 0.05 -> not falsified)")

    p_bad, _ = audit_file(bugs, "private_count_wrong_scale", 1.0,
                          event_iterations=20_000, detect_iterations=20_000)
    print(f"buggy   private_count_wrong_scale: p={p_bad:.3f}   (expect ~0 -> falsified)")

    ok = p_ok > 0.05 and p_bad < 0.05
    print("\nSTATDP ADAPTER WORKING" if ok else "\nunexpected result - investigate")
