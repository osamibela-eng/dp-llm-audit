"""VERTICAL SLICE: the whole research loop on one task, no API keys needed.

  spec -> (mock) LLM code -> screen -> smoke test -> DP audit
       -> counterexample -> R2 repair prompt -> (mock) repaired code -> re-audit

Run:  python experiments/vertical_slice.py
If this prints 'VERTICAL SLICE COMPLETE', your environment can run the entire
project. Swap the mock outputs for real API calls via generators/ and repair/.
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from auditors.simple_auditor import falsify           # noqa: E402
from sandbox.runner import load_mechanism             # noqa: E402

# --- a plausible "LLM generation" with a classic bug (wrong Laplace scale) ----
MOCK_LLM_OUTPUT = '''
import numpy as np

def private_count(data, epsilon, rng):
    """Differentially private count of 1s using the Laplace mechanism."""
    true_count = sum(1 for x in data if x == 1)
    noise = rng.laplace(loc=0.0, scale=1.0 / (4.0 * epsilon))  # BUG: 4x under-noised
    return float(true_count + noise)
'''

MOCK_REPAIRED_OUTPUT = '''
import numpy as np

def private_count(data, epsilon, rng):
    """Differentially private count of 1s using the Laplace mechanism."""
    true_count = sum(1 for x in data if x == 1)
    noise = rng.laplace(loc=0.0, scale=1.0 / epsilon)
    return float(true_count + noise)
'''


def main():
    tasks = yaml.safe_load((ROOT / "benchmark" / "tasks.yaml").read_text())["tasks"]
    task = next(t for t in tasks if t["id"] == "laplace_bounded_count_v1")
    pairs = [tuple(p) for p in task["auditor"]["pairs"]]
    eps = task["claimed_epsilon"]

    print("=" * 70)
    print("STEP 1  screen + load the (mock) LLM-generated code")
    fn = load_mechanism(MOCK_LLM_OUTPUT, task["function_name"])
    print("        screening passed, function loaded")

    print("STEP 2  smoke test")
    rng = np.random.default_rng()
    out = fn([1, 0, 1, 1], eps, rng)
    print(f"        private_count([1,0,1,1]) -> {out:.3f}  (runs, returns float)")

    print("STEP 3  DP audit (simple_auditor, claimed eps=1.0)")
    r = falsify(fn, pairs, eps, n_confirm=30_000, seed=0)
    print(f"        status = {r.status}   ({r.elapsed_s:.1f}s, {r.tests_run} tests)")
    assert r.status == "falsified", "expected the buggy mock to be falsified"
    ce = r.counterexample
    print(f"        counterexample: D={ce.d1}  D'={ce.d2}")
    print(f"                        event: {ce.event}")
    print(f"                        empirical eps lower bound: {ce.eps_lower_bound:.2f}"
          f"  (claimed {eps})")

    print("STEP 4  R2 repair prompt (what the model would receive)")
    tmpl = (ROOT / "repair" / "prompts" / "repair_counterexample.txt").read_text()
    prompt = tmpl.format(spec=task["spec"].strip(), code=MOCK_LLM_OUTPUT.strip(),
                         function_name=task["function_name"], epsilon=eps,
                         delta=task["claimed_delta"], adjacency=task["adjacency"],
                         d1=ce.d1, d2=ce.d2, event=ce.event,
                         p1_lb=f"{ce.p1_lb:.4f}", p2_ub=f"{ce.p2_ub:.4f}",
                         eps_lb=f"{ce.eps_lower_bound:.2f}")
    print("        " + "\n        ".join(prompt.splitlines()[:6]) + "\n        ...")

    print("STEP 5  re-audit the (mock) repaired code")
    fn2 = load_mechanism(MOCK_REPAIRED_OUTPUT, task["function_name"])
    r2 = falsify(fn2, pairs, eps, n_confirm=30_000, seed=0)
    print(f"        status = {r2.status}   ({r2.elapsed_s:.1f}s)")
    assert r2.status == "not_falsified"
    print("        NOTE: 'not_falsified' is NOT a proof of privacy - it means the")
    print("        program survived this audit budget over these pairs and events.")

    print("=" * 70)
    print("VERTICAL SLICE COMPLETE - the full pipeline runs on this machine.")


if __name__ == "__main__":
    main()
