"""Record every distinct input an auditor actually probes a mechanism with.

E1 needs to answer a question v1 could only assert in prose: when a generic
auditor disagrees with ours, how often is it because it left the task's declared
input domain?

StatDP generates its own databases internally, so we cannot read them off its
configuration. But every database it generates is eventually passed to the
mechanism, and the mechanism is ours. Wrapping it therefore observes exactly the
inputs that mattered, without patching StatDP at all.

Two practical constraints shaped this:

* StatDP runs the mechanism in spawned worker processes, so an in-memory counter
  in the parent would see nothing. Each worker appends to a shared JSONL instead.
* The mechanism is called tens of thousands of times but on a handful of distinct
  databases. Each worker keeps a local set and writes a line only the first time
  it sees a database, which keeps the log small and the overhead invisible.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auditors.contract import Contract, check_dataset  # noqa: E402
from auditors.statdp_adapter import _MechAdapter  # noqa: E402


class ContractLoggingAdapter(_MechAdapter):
    """A StatDP mechanism adapter that logs contract violations as it is probed.

    Behaviour is otherwise identical to the plain adapter: it does NOT block
    out-of-domain probes. The whole point of the generic arm is to measure what
    an unmodified tool does, so intercepting and correcting here would destroy
    the measurement.
    """

    def __init__(self, source_path, func_name, contract: Contract, log_path):
        super().__init__(source_path, func_name)
        self._contract = contract
        self._log_path = str(log_path)
        self._seen: set | None = None

    def _record(self, data) -> None:
        if self._seen is None:
            self._seen = set()
        try:
            key = tuple(data)
        except TypeError:
            key = repr(data)
        if key in self._seen:
            return
        self._seen.add(key)

        violations = check_dataset(self._contract, data)
        row = {
            "task": self._contract.task_id,
            "pid": os.getpid(),
            "probe": list(data)[:12],
            "probe_len": len(list(data)),
            "violations": [{"kind": v.kind, "detail": v.detail} for v in violations],
        }
        # One short line per distinct probe; O_APPEND keeps concurrent workers
        # from interleaving mid-line at these sizes.
        with open(self._log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def __call__(self, prng, queries, epsilon):
        self._record(list(queries))
        return super().__call__(prng, queries, epsilon)

    # the parent class ships only strings across the process boundary; carry the
    # contract and log path too, or the child would silently stop recording
    def __getstate__(self):
        st = super().__getstate__()
        st["contract"] = {
            "task_id": self._contract.task_id,
            "record_type": self._contract.record_type,
            "domain": self._contract.domain,
            "min_size": self._contract.min_size,
            "max_size": self._contract.max_size,
            "adjacency": self._contract.adjacency,
            "output_type": self._contract.output_type,
            "ignored_ok": self._contract.ignored_ok,
        }
        st["log_path"] = self._log_path
        return st

    def __setstate__(self, state):
        super().__setstate__(state)
        c = state["contract"]
        self._contract = Contract(
            task_id=c["task_id"], record_type=c["record_type"], domain=c["domain"],
            min_size=c["min_size"], max_size=c["max_size"],
            adjacency=c["adjacency"], output_type=c["output_type"],
            ignored_ok=c["ignored_ok"], notes="")
        self._log_path = state["log_path"]
        self._seen = None


def summarise(log_path) -> dict:
    """Collapse a probe log into per-task counts."""
    p = pathlib.Path(log_path)
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    seen_global: dict[str, set] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn line from concurrent append; ignore rather than crash
        t = r["task"]
        d = out.setdefault(t, {"distinct_probes": 0, "violating_probes": 0,
                               "by_kind": {}})
        key = tuple(r["probe"]) + (r["probe_len"],)
        g = seen_global.setdefault(t, set())
        if key in g:
            continue           # same probe seen in several workers
        g.add(key)
        d["distinct_probes"] += 1
        if r["violations"]:
            d["violating_probes"] += 1
            for v in r["violations"]:
                d["by_kind"][v["kind"]] = d["by_kind"].get(v["kind"], 0) + 1
    return out
