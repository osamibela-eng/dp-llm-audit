# Error taxonomy — decision rules

**Written before any program was labelled.** That ordering is the whole point: rules
authored after seeing the data can be tuned, consciously or not, to make the
distribution look tidy. These rules are committed first, applied second, and any
rule that turns out to be unworkable is amended *by appending a dated revision at
the bottom of this file*, never by editing the rule in place.

Ten families. **Labelling is multi-label**: one
program can carry several. A program that computes sensitivity as `1` for a sum
over `[0, 100]` *and* leaks through an early return on empty input has two
independent defects, and forcing a single label would erase one of them.

`ambiguous` is a real answer, not a failure to decide. Use it when the code is
wrong in a way none of the ten families describes, or when two families fit and
the code does not distinguish them.

---

## How to apply these rules

1. Read the task spec and the reference implementation.
2. Read the generated program.
3. Apply every rule whose trigger condition holds. Do not stop at the first match.
4. If no rule fires but the program was falsified, label `ambiguous` and write one
   sentence saying what is wrong.
5. Do **not** consult the auditor's counterexample while labelling. The taxonomy
   describes what the program does wrong, not what the auditor happened to catch;
   letting the counterexample drive the label would make the taxonomy a
   restatement of the auditor's blind spots.

Rule 5 is what makes the "which families are invisible to the auditor" analysis
(RQ2) meaningful. If labels were derived from counterexamples, no family could
ever be recorded as invisible.

---

## F1 — sensitivity

**Trigger.** The noise scale is derived from a sensitivity that does not match the
query's true sensitivity under the task's stated adjacency.

Canonical instances:
- Sum over values clipped to `[lo, hi]` treated as sensitivity `1` rather than
  `max(|lo|, |hi|)` (or `hi - lo` under replace-one).
- Mean treated as sensitivity `1/n` without accounting for `n` itself being
  data-dependent under add/remove-one.
- Histogram treated as sensitivity `2` under add/remove-one, or `1` under
  replace-one.

**Not F1.** Correct sensitivity, wrong arithmetic converting it to a scale — that
is F2.

**Boundary with F2.** Ask: *if the sensitivity constant were replaced by the
correct one and nothing else changed, would the program be correct?* If yes, F1.

---

## F2 — calibration

**Trigger.** Sensitivity is right, but the mapping from `(sensitivity, ε)` to a
noise parameter is wrong.

Canonical instances:
- `scale = epsilon / sensitivity` (inverted).
- Gaussian σ computed with the variance formula where the standard deviation is
  required, or `sqrt` applied to the wrong subexpression.
- `np.random.laplace(loc, scale)` given a *variance* rather than a scale.
- Gaussian noise released under a pure-ε claim with `δ = 0`.

The last one is F2 *and* F10, since it is simultaneously a miscalibration and a
claim the mechanism cannot support.

---

## F3 — adjacency

**Trigger.** The program is internally consistent but assumes a different
neighbouring relation than the task states.

Canonical instances:
- Task states add/remove-one; program computes sensitivity as if replace-one
  (factor of 2 on counts, and on histograms).
- Program assumes bounded `n` (i.e. `n` is public) when the task does not say so.

**Why this is separate from F1.** The arithmetic may be *correct for some
adjacency*. The error is in which privacy claim is being made, not in the algebra.
This distinction matters for the paper because F3 is the family most likely to
reflect a genuine ambiguity in DP literature rather than a coding mistake.

---

## F4 — composition

**Trigger.** Multiple releases, and the budget across them does not compose to the
claimed total.

Canonical instances:
- Two queries each given the full `ε` (total `2ε`).
- `k` releases each given `ε` with the claim that the total is `ε`.
- Advanced composition invoked without its `δ`.
- Budget split that does not sum to `ε` — note that splitting to *less* than `ε`
  is over-noising: still F4, but privacy-safe, and must be recorded as such (see
  "over-noising" below).

---

## F5 — selection

**Trigger.** A max / argmax / top-k / choice is made, and the selection step is
either unnoised or noised incorrectly.

Canonical instances:
- Exponential mechanism with the utility used directly instead of
  `exp(ε·u / (2·Δu))`, or with the factor of 2 omitted.
- Report Noisy Max returning the noisy *value* as well as the index.
- Top-k splitting `ε` over `k` selections incorrectly, or selecting
  deterministically after a single noised pass.
- Argmax taken over unnoised scores.

**Boundary with F4.** Top-k is both a selection and a composition. Label both.

---

## F6 — data-dependent control flow

**Trigger.** A branch, loop bound, exception, or early return depends on the
private data in a way that is visible in the output.

Canonical instances:
- `if not data: return 0` — the released value is deterministic on one side of a
  neighbouring pair.
- `if n < k: raise ValueError(...)` — the exception itself is a release.
- Loop that terminates when a noisy value crosses a threshold, where the number of
  iterations is then returned alongside.
- Clipping bounds computed from the data (`min(data)`, `max(data)`) rather than
  from public parameters. This one is also F1, since the clip range then has
  data-dependent sensitivity.

**Why this family is worth isolating.** These are *support violations*: an output
with probability exactly zero on one side of the pair. No choice of ε repairs
them, which makes them qualitatively different from every miscalibration family.

---

## F7 — boundary and degenerate input

**Trigger.** The program is correct on typical input but wrong at an edge of the
input domain, where "wrong" means the privacy guarantee fails rather than the code
raises.

Canonical instances:
- Empty dataset handled by a special case (also F6).
- `n = 0` division guarded by substituting a constant.
- Clipping applied *after* summation rather than per-record, so a single extreme
  record is unbounded.
- Result clamped to a valid range (`max(0, noisy_count)`) in a way that creates a
  point mass. Clamping to a *public* range is legitimate post-processing; clamping
  using a *data-dependent* bound is not. Only the latter is F7.

That last distinction is the one most likely to be got wrong by an annotator, and
`max(0, noisy_count)` specifically is **legitimate** — 0 is a public bound. It is
listed here because it looks suspicious and should be explicitly *not* labelled.

---

## F8 — randomness misuse

**Trigger.** The noise source itself is wrong, or noise is shared where it must be
fresh.

Canonical instances:
- One noise draw reused across releases that must be independent.
- A seeded / deterministic RNG, or `random.seed(0)`.
- Noise drawn once outside a loop that releases per-iteration.
- Uniform noise substituted for Laplace or Gaussian.
- The `rng` argument ignored in favour of the global `numpy.random` state. This is
  F8 only if it changes the distribution or the independence structure; using the
  global RNG with the correct distribution is an F9 API misuse, not F8.

**Critical sub-distinction, established empirically in `FINDINGS.md` §4.**
"Reuses one noise draw" is *not* a privacy violation on its own. It depends on
what the shared randomness couples:

- Repeated releases of the **same** query: sharing a draw *collapses* them to a
  single answer. This is **more** private than claimed. Label F8 and mark
  `over_noising`.
- **Different** queries: sharing a draw makes a function of them deterministic.
  This **leaks**. Label F8 without the marker.

The annotator must read which case applies. Labelling on the surface pattern alone
would put a privacy-safe bug and a total-leak bug in the same cell.

---

## F9 — API misuse

**Trigger.** A DP library or numpy API is called with arguments that do not mean
what the program assumes, where the error is in the interface rather than the
mathematics.

Canonical instances:
- `np.random.laplace(scale=...)` vs `b` vs variance confusion at the call site
  (F2 if the derivation is wrong; F9 if the derivation is right and the argument
  is passed in the wrong slot or position).
- Positional arguments in the wrong order.
- Ignoring the injected `rng` in favour of module-level randomness (see F8).
- A library's ε parameter interpreted as ε per record when it is per query, or
  vice versa.

**Boundary with F2.** F9 is a wiring error; F2 is a maths error. If the comments
or intermediate variables show the correct value being computed and then handed to
the wrong parameter, that is F9.

---

## F10 — claim mismatch

**Trigger.** The mechanism may be a valid DP mechanism, but not for the `(ε, δ)`
the task claims.

Canonical instances:
- Gaussian mechanism under `δ = 0`.
- A mechanism satisfying `(ε, δ)`-DP where the task asks for pure `ε`-DP.
- Correct implementation of a *different* ε than the one passed in (e.g. hard-coded
  `epsilon = 1.0` shadowing the argument).
- Claiming a total budget that the composition cannot deliver — F4 *and* F10.

---

## Cross-cutting marker: `over_noising`

Not a family. An orthogonal flag, applied alongside a family label whenever the
defect makes the mechanism **more** private than claimed.

This exists because the calibration set contains 11 deliberate over-noising probes
and a tool that flagged them would be useless (`FINDINGS.md` §2). The same logic
applies to the taxonomy: an over-noised program is a genuine implementation error
with a genuine utility cost, but it is not a privacy failure, and pooling it with
leaks would misstate what the falsification rate measures.

An `over_noising` program should be **not falsified** by a correct auditor. If one
is falsified, that is a false positive and must be reported as such.

---

## Blinding protocol

Assessment 5 asks for a blinded subset, and the design is:

1. The pre-labeller (`analysis/prelabel_taxonomy.py`) applies syntactic rules to
   every falsified program and writes its labels to
   `results/processed/taxonomy_prelabels.jsonl`.
2. `--blind-sample N` writes `results/processed/taxonomy_blind_worksheet.jsonl`
   containing, for each sampled program: the task spec, the code, and an empty
   `labels` field. **It does not contain the pre-label, the outcome, the model
   name, or the counterexample.** The order is shuffled under a fixed seed.
3. The human annotator fills in `labels` (a list) and optionally `note`.
4. `--score` compares the two and reports per-family agreement and Cohen's κ
   treating each family as a separate binary decision, since the labelling is
   multi-label and a single multi-class κ would not be defined.

Agreement is reported whatever it turns out to be. A low κ on a family is a
finding about that family's rules, and the honest response is to say so and
report the human labels as primary — not to revise the rules until they agree.

---

## Revisions

*(Append dated entries here. Do not edit rules above.)*

### 2026-08-21 — F7 applied strictly; two cases recorded that the rule does not reach

Applied during the single-annotator pass over all 90 falsified programs.

**Decision.** F7's trigger reads "correct on typical input but wrong at an edge of
the input domain". That is applied **literally**: a program already wrong on
typical input (bad scale, wrong noise family) that *also* special-cases the empty
dataset receives F6 alone, not F6 + F7.

Without this, F7 would be a synonym for F6 — every empty-input early return would
carry both — and the taxonomy would lose the distinction it exists to draw: a
mechanism broken everywhere versus one broken only at the boundary. The second
kind is the more interesting finding, since those programs are one line from
correct. Three such programs exist in `clipped_bounded_sum` alone
(`940a0f2d597b`, `801bd920b880`, `4e210ade6c91`), plus `51b84f29bdcb` in
`empty_dataset_count` and `259333efe758` in `laplace_bounded_count`.

**Cases the rule as written does not reach.** Two programs in
`empty_dataset_count` (`05d180abfdce`, `88b515b91e89`) fail on the empty dataset
with **no branch at all**. Their Gaussian scale is a function of `n` — e.g.
`sqrt(2*n*eps)` — so at `n = 0` the scale is 0, `rng.normal(0, 0)` returns exactly
0, and the empty dataset receives a deterministic release. That is the same
support violation an early return produces, arrived at by a different route.

Under the literal rule they get F1 (the scale should not depend on the data) and
not F7, because they are also wrong on non-empty input. Both were labelled that
way, and the observation is recorded in each note.

**This rule is not being edited**, per the append-only policy at the top of this
file. But it is flagged as a revision candidate: a future version should probably
read *"the program's boundary behaviour is a defect over and above whatever else
is wrong"*, which would capture both of these. Changing it now, after seeing which
programs it does and does not catch, is exactly the tuning the append-only policy
exists to prevent.
