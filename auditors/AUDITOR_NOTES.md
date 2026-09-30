# Auditor policy (pre-registered decisions)

These are the audit-policy decisions the paper must state. They were made
deliberately at project start — do not drift from them mid-experiment.

## 1. Two independent auditors

- **simple_auditor** (this repo): pilot/confirm threshold-and-exact-value event
  search, Clopper–Pearson bounds, Bonferroni at overall α = 0.01. Fast
  (~0.5–2 s per program at n = 30k on a laptop), supports pure and (ε, δ)
  claims, dicts/labels/tuples.
- **StatDP** (third_party clone, verified working on Python 3.11): independent
  hypothesis-test methodology; numeric-vector tasks first. Agreement between
  the two auditors on falsifications is itself a reported result
  ("detection complementarity").
- **DP-Sniper** (optional third): ETH SRI, conda-based; add only if it installs
  cleanly in week 1. Do not block the project on it.
- **Sequential auditing** (arXiv:2509.07055) is the modern sample-efficient
  option — try after the core pipeline works, never before.

## 2. Outcome vocabulary (never collapse these)

`syntax_fail | screen_fail | runtime_fail | semantic_fail |
auditor_unsupported | falsified | not_falsified | auditor_error | timeout`

**"not_falsified" is never reported as "correct" or "private".** The paper's
key sentence: a program that is not falsified has only survived a specified
audit budget over a specified family of neighboring inputs and output events.

## 3. Floating-point policy

Bit-level/floating-point DP violations (naive continuous Laplace sampling is
not exactly DP; Mironov 2012, arXiv:2112.05307) are **out of scope**: events
are coarse (thresholds, rounded exact values), references use standard numpy
samplers, and the paper states this explicitly. Calibration confirms the
references are not flagged under this event family (0 false positives).

## 4. Seeding policy

- Functional tests MAY seed rng (reproducibility of the test).
- Audit runs are NEVER seeded by default; `seed=` exists only to reproduce a
  specific audit end-to-end, and any reported falsification must also be
  reproduced with a fresh seed.
- Generated code must draw all randomness from the supplied Generator; use of
  `numpy.random.*` module functions or hardcoded seeds is a bug class
  (randomness error), detectable by AST scan.

## 5. Pair design is part of the audit (calibration finding)

During starter-kit calibration, deterministic-argmax bugs were MISSED until a
neighboring pair was added in which removing one record FLIPS the winner
(ties broke toward "A" on both sides otherwise). Lesson: pairs must be
designed per bug class, and the calibration table is what makes "missed"
interpretable. Keep `expected_auditability` predictions honest.

## 6. Epsilon-parameter trap (calibration finding)

At claimed ε = 1.0, `scale = epsilon` and `scale = 1/epsilon` coincide — the
classic inversion bug is INVISIBLE if you only audit at ε = 1. Policy: audit
every program at the task's claimed ε AND at a second setting (ε = 0.3),
since the spec requires correctness for the passed epsilon. (Implemented as a
flag in analysis/aggregate.py; include both in the results table.)

## 7. Known deliberate blind spot

`private_argmax_half_scale` (true 2ε-DP) is expected to survive the audit —
kept in the calibration set as the standing demonstration that silence ≠ safety.

## 8. StatDP on Windows: two fixes required (found 2026-08-20)

The starter kit's StatDP adapter was verified on Python 3.11; on this machine
(Windows, Python 3.13.9) it failed twice before working. Both are worth stating
in the paper's reproducibility section, because both would silently affect
anyone re-running the study on Windows.

**(a) Spawn cannot pickle a closure.** StatDP parallelises with
`multiprocessing.Pool`. Windows uses the `spawn` start method, so the mechanism
object must pickle *and* be importable in a fresh interpreter. The original
wrapper was a closure and failed with
`AttributeError: Can't get local object 'audit_numeric_task.<locals>.wrapped'`.
A function loaded by `importlib.spec_from_file_location` fails for the same
reason -- the child cannot re-import a module that never had a real name.
Fix: `_MechAdapter` stores only a source path and a function name (strings
pickle), re-imports lazily in the calling process, and lends StatDP a `__code__`
object from a probe function, because `statdp/generators.py:46` introspects
`algorithm.__code__.co_varnames` and a callable instance has none.

**(b) Adjacency mismatch produces FALSE POSITIVES.** StatDP defaults to
`sensitivity=ALL_DIFFER`, under which every entry of the input vector differs by
one -- L1 distance equal to the vector length, not 1. Our tasks declare
add/remove-one or replace-one adjacency. Auditing a *correct* reference under
ALL_DIFFER falsifies it:

| Setting | reference `private_count` | buggy `private_count_wrong_scale` |
|---|---|---|
| `ALL_DIFFER` (default) | p = 0.000 -- **false positive** | p = 0.000 |
| `ONE_DIFFER` | p = 0.840 -- correct | p = 0.000 -- correct |

This is not a bug in the mechanism; it is a mismatch between the auditor's
neighbour model and the task's declared adjacency. It reinforces §5: pair design
is part of the audit, and an auditor whose adjacency does not match the claim
will produce confident nonsense in either direction. Every auditor added to this
project must have its neighbour model checked against `tasks.yaml` adjacency
before any result from it is believed.

## 9. Task size bounds detectability (calibration finding, 2026-08-20)

Full-benchmark calibration caught 22 of 61 known bugs with 0 false positives
across all 16 references. The misses are not uniformly auditor weakness, and
separating the causes matters for how `not_falsified` is read:

**(a) Pair design.** `exponential_mechanism_v1` originally shipped two
neighbouring pairs, neither of which flipped the argmax. A *deterministic*
selection bug -- maximally non-private -- was therefore invisible: both sides
returned "A" every time. Adding the pair `(["B","C"], ["C"])`, where removing
one record changes the winner, takes the measured epsilon lower bound from
nothing to 9.21, while the reference stays at 0.43. This is the same failure
already recorded in section 5 for Report Noisy Max, now confirmed as a
recurring pattern rather than a one-off: **selection mechanisms need pairs
constructed to flip the selection, and no amount of sampling substitutes.**

**(b) The task instance caps the achievable loss.** For `above_threshold_v1`
the four buggy variants are genuine DP violations, but the harm each causes
GROWS WITH THE NUMBER OF QUERIES ANSWERED, and this task answers only three.
Searching neighbouring pairs for the maximum demonstrable epsilon gives:

| variant | best eps_lb | claim |
|---|---|---|
| no_query_noise | 1.04 | 1.0 |
| fresh_threshold_noise | 0.70 | 1.0 |
| missing_factor_two | 0.58 | 1.0 |
| returns_value | 0.56 | 1.0 |

Three of the four cannot be falsified at this task size *by any pair*, not
merely by the pairs we chose. Their `expected_auditability` labels were
corrected from `detectable` to `likely_missed` on the strength of the
measurement rather than the reasoning.

This is a benchmark-design limitation worth stating plainly in the paper: a
sparse-vector task with three queries cannot expose bugs whose cost scales with
query count. It also sharpens the headline caveat. A model that writes
`above_threshold` with per-query threshold noise -- the single most common
published error in this primitive -- will be reported `not_falsified` here, and
that verdict says something about our benchmark, not about the code.

**(c) Output-type coverage.** `private_topk_v1` returns an ordered list of
labels, which the auditor could not canonicalise and reported as
`unsupported`. Sequences of strings are now mapped to a single composite
category, order preserved, since for a top-k release ["A","B"] and ["B","A"]
are different outputs and pooling them would hide rank leakage.
