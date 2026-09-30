"""Single-annotator taxonomy labels, applied against docs/TAXONOMY_RULES.md.

PROVENANCE, stated plainly because it changes how these labels may be used.

These are NOT an independent second annotation. They were produced by the same
author as `analysis/prelabel_taxonomy.py`, with knowledge of the outcome data.
Any agreement between the two is therefore a **consistency check between an
automatic pre-labeller and its author**, and must never be reported as
inter-annotator reliability or as a kappa between independent raters.

What they legitimately support is RQ2: which error families cause falsifications,
and which are invisible to the auditor. The study protocol allows exactly this -
solo annotation with published decision rules plus an `ambiguous` label, which is
acceptable at workshop level provided the rules are public - and the rules were
committed before any program was read.

A genuine inter-annotator number requires a second person labelling blind. The
worksheet machinery for that exists (`analysis/blind_worksheet.py`); until it is
filled in by someone else, the paper reports single-annotator labels and says so.

Keys are code_sha256 prefixes, resolved against the falsified set. Each entry is
(labels, markers, note).

    python analysis/annotator_pass.py          # write the label file
    python analysis/annotator_pass.py --check  # verify every key resolves uniquely
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "processed" / "taxonomy_annotator_labels.jsonl"

F1, F2, F3 = "F1_sensitivity", "F2_calibration", "F3_adjacency"
F4, F5, F6 = "F4_composition", "F5_selection", "F6_data_dependent_flow"
F7, F8 = "F7_boundary", "F8_randomness"
F9, F10 = "F9_api_misuse", "F10_claim_mismatch"
AMB, OVER = "ambiguous", "over_noising"

LABELS: dict[str, tuple[list[str], list[str], str]] = {}

# ---------------------------------------------------------------------------
# ANNOTATION CONVENTION, fixed after the first task and applied uniformly.
#
# F7's trigger in TAXONOMY_RULES.md is "correct on typical input but wrong at an
# edge of the input domain". Read literally, that means an empty-input early
# return earns F7 only when the program is OTHERWISE correct. A program that is
# already wrong on typical input (bad scale, wrong noise family) and also has an
# empty-input branch gets F6 alone.
#
# Without this rule F7 would be a synonym for F6 - every empty early return would
# carry both - and the taxonomy would lose the distinction it exists to draw:
# between a mechanism that is broken everywhere and one that is broken only at
# the boundary. The second kind is the more interesting finding, because those
# programs are one line away from correct.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# laplace_bounded_count_v1 - tier 1, eps=1.0, add/remove-one, sensitivity 1.
# Correct: count + Laplace(1/eps).
# ---------------------------------------------------------------------------
LABELS.update({
    "90cb998f1891": ([F2, F8], [],
                     "Cauchy noise, not Laplace; scale eps/2*sqrt(n) puts epsilon in "
                     "the numerator and makes the scale depend on n."),
    "09183331583a": ([F2, F6], [],
                     "Empty early return. Noise is a deterministic function of n "
                     "selected by a 3-way branch, so the release takes one of three "
                     "values given the data; not a calibrated mechanism."),
    "24cd48d804cb": ([F2, F6], [],
                     "Empty early return. Noise is multiplicative and the number of "
                     "draws equals the number of 1s, so both the scale and the draw "
                     "count are data-dependent."),
    "259333efe758": ([F6, F7], [],
                     "Laplace and scale are BOTH correct and max(0,.) is legitimate "
                     "post-processing. The sole defect is `if len(data)==0: return 0`, "
                     "a point mass of probability 1 on the empty side against 0.18 on "
                     "[1]. Cleanest single-cause program in the task."),
    "12e79ef7bea1": ([F2, F6, F8], [],
                     "No randomness at all - rng is never called, so the release is "
                     "deterministic. At eps=1 the noise term is identically zero."),
    "0b3d4a0c334a": ([F6, F7], [],
                     "Laplace and scale correct; max(0,.) fine. Two defects: empty "
                     "early return, and min(estimate, total) clamps with len(data), "
                     "which is not a public bound."),
    "1651a072c2f0": ([F2, F6, F9], [],
                     "np.random.geometric ignores the injected rng (module-level "
                     "state, correct-ish distribution -> F9 per the rules, not F8). "
                     "Geometric parameter 1/len(data) is not calibrated to epsilon. "
                     "F6 ADDED on review: this program also has `if len(data)==0: "
                     "return 0`, which the first pass missed. Caught by the "
                     "pre-labeller disagreement check, not by re-reading."),
    "d91610c5542e": ([F2, F6, F8], [],
                     "Empty early return. Noise is a sum of n Bernoulli draws, so its "
                     "magnitude grows with the dataset size instead of being fixed by "
                     "the sensitivity."),
    "81ac946cc7ee": ([F2, F6, F7, F8], [],
                     "Uniform noise on a bounded support cannot give pure eps-DP at "
                     "any scale - bounded support means the two distributions have "
                     "regions of disjoint support. Plus empty early return and a "
                     "min(n, .) clamp on a non-public bound."),
    "d5861981e52a": ([F2, F6, F10], [],
                     "Gaussian noise under a delta=0 claim (F2 and F10 per the rules). "
                     "Scale eps/n is inverted and data-dependent. Empty early return."),
})

# ---------------------------------------------------------------------------
# clipped_bounded_sum_v1 - tier 1, eps=1.0, add/remove-one, sensitivity B = 10.
# Correct: clip each value to [0,10], sum, + Laplace(10/eps).
# Falsified 12/12. The task the models get wrong most consistently.
# ---------------------------------------------------------------------------
LABELS.update({
    "d4d11dd687ce": ([F1, F2, F6, F10], [],
                     "Clips correctly, then never uses B in the scale: "
                     "sqrt(epsilon*len(data)) puts epsilon in the numerator and the "
                     "dataset size in the noise. Gaussian under a delta=0 claim."),
    "9bd5b4c8de38": ([F1, F2, F6, F10], [],
                     "Returns a single draw from N(mean, std) of the data rather than "
                     "a noisy sum, so the noise scale IS the data's standard "
                     "deviation. epsilon is never used anywhere in the function."),
    "a62c90ef3455": ([F1, F2, F6, F8], [],
                     "diff_privacy = abs(sum(x) - np.sum(x)) is identically zero, so "
                     "alpha is zero and rng.laplace(0.0, 0) returns the exact sum. "
                     "Sensitivity is also computed as max(data)-min(data)."),
    "58be14ebb4f1": ([F1, F2, F6, F10], [],
                     "Returns the noisy AVERAGE, not the sum, so the sensitivity B is "
                     "the wrong constant for the quantity released. Gaussian under a "
                     "delta=0 claim, with a sigma formula borrowed from the analytic "
                     "Gaussian mechanism."),
    "e0867d4940e0": ([F1, F2, F6, F10], [],
                     "Two data-dependent returns, not one: the empty check AND "
                     "`if total == 0.0: return 0.0`, which releases an exact zero "
                     "whenever the clipped sum is zero. Scale exp(eps/2) ignores B."),
    "8be8f418cdcc": ([F1, F6], [],
                     "Laplace family and epsilon placement both correct; scale is "
                     "1/epsilon where the clipped sum has sensitivity 10. Also "
                     "returns the average rather than the sum."),
    "537eb1eab4c6": ([F6, F8], [],
                     "The variable is named noisy_sum but no noise is ever drawn - "
                     "rng is unused and the release is the exact clipped sum, "
                     "deterministically rescaled."),
    "8eeef1d6a094": ([F1, F6], [],
                     "Textbook sensitivity error and the clearest instance of it in "
                     "the benchmark: correct Laplace, correct epsilon placement, "
                     "scale 1/epsilon instead of 10/epsilon. Noise is 10x too small."),
    "940a0f2d597b": ([F6, F7], [],
                     "Sensitivity, scale and noise family ALL correct. Sole defect is "
                     "`if not data: return 0.0`."),
    "801bd920b880": ([F6, F7], [],
                     "As above - fully correct mechanism, one-line boundary defect."),
    "4e210ade6c91": ([F6, F7], [],
                     "As above - fully correct mechanism, one-line boundary defect."),
    "0bb282c90f4d": ([F1, F2, F6], [],
                     "scale = epsilon/(2*sum_data) is inverted and data-dependent, and "
                     "len(data) independent draws are summed so the noise magnitude "
                     "grows with the dataset."),
})

# ---------------------------------------------------------------------------
# bounded_mean_v1 - tier 1, eps=1.0, REPLACE-one, sensitivity 1/n.
# Correct: mean + Laplace(1/(n*eps)).
# Note this is the only tier-1 task under replace-one, and no program was
# falsified for confusing the two adjacencies - F3 does not appear here.
# ---------------------------------------------------------------------------
LABELS.update({
    "4db42b0282e9": ([F8, F10], [],
                     "Branches on epsilon and returns the EXACT mean when "
                     "epsilon == 1.0 - which is the claimed epsilon. rng is never "
                     "called on that path. The mechanism is only non-trivial at "
                     "values of epsilon the task does not claim."),
    "9a33944fa387": ([F2, F8], [],
                     "Treats epsilon as a Bernoulli keep-probability. At epsilon = 1 "
                     "the mask is all-True almost surely, so the release is the exact "
                     "mean minus a deterministic function of the data."),
    "564815bbfa4b": ([F1, F2, F10], [],
                     "Noise is divided by the data's own root-mean-square and "
                     "multiplied by sqrt(2*epsilon), so the scale is data-dependent "
                     "and epsilon sits in the numerator. Gaussian under delta=0. "
                     "Also returns a length-n array rather than a scalar."),
    "180ec88090eb": ([F1, F2, F8], [],
                     "sqrt(2*variance*epsilon) is computed but rng is never called - "
                     "the 'noise' is a deterministic function of the data's variance."),
    "7056377dd15c": ([F2], [],
                     "Textbook F2 and the cleanest one in the benchmark: sensitivity "
                     "1/n is CORRECT, then the analytic-Gaussian sigma expression "
                     "sqrt(2*ln(1.25/eps)) is used as a Laplace scale and epsilon "
                     "never divides. At eps=1 the scale is 0.67/n instead of 1/n."),
    "4099ac1c07c3": ([F2], [],
                     "Sensitivity 1/n correct. Invents delta = 1/n^2 and divides by "
                     "log(1/delta)/epsilon, giving scale eps/(2n*ln n) - about 3x too "
                     "small at n=5. Right sensitivity, wrong map to a scale."),
    "6619178588b9": ([F1, F2, F6], [],
                     "Rejection-samples the noise in a while loop whose acceptance "
                     "test depends on a quantile of the data, so the ACCEPTED noise "
                     "distribution is data-dependent. The draw is then multiplied by "
                     "np.std(data). Two separate data-dependent channels."),
    "92850de81fab": ([F2, F10], [],
                     "Sensitivity correct, then Gaussian noise under a delta=0 claim "
                     "with epsilon in the numerator (sensitivity*eps/2)."),
    "4da5daa0b4c9": ([F2, F10], [],
                     "As above, with scale sensitivity*eps/sqrt(2n) - epsilon in the "
                     "numerator and an extra data-dependent factor."),
    "796bc952fc2e": ([F2], [],
                     "Scales the data by n, adds Laplace(1/(eps*n)), then divides the "
                     "whole thing by n - so the effective scale is 1/(eps*n^2), a "
                     "factor of n too small. The divide-by-n is applied twice."),
})

# ---------------------------------------------------------------------------
# empty_dataset_count_v1 - tier 3, eps=1.0, add/remove-one.
# The spec EXPLICITLY forbids a special-cased empty path. Correct: count +
# Laplace(1/eps), with no `if not data` anywhere.
# ---------------------------------------------------------------------------
LABELS.update({
    "818629e7aac1": ([F2, F6, F10], [],
                     "Empty early return plus per-element Gaussian noise summed, so "
                     "the noise magnitude grows with n. Gaussian under delta=0."),
    "849f74e96ea0": ([F2, F6, F10], [],
                     "Empty early return; Gaussian with scale sqrt(2*eps*n). The "
                     "clip to [0,1] is public and fine."),
    "df3904038d5e": ([F2, F6, F10], [],
                     "Empty early return; Gaussian under delta=0; returns an array "
                     "rather than a scalar."),
    "c6eaf62a7ad9": ([F2, F6], [],
                     "Empty early return. Laplace scale sqrt(2*ln(1.25/eps)/n) is "
                     "data-dependent and collapses as n grows; np.floor then "
                     "quantises the noise."),
    "51b84f29bdcb": ([F6, F7], [],
                     "Counts correctly and adds Laplace(1/eps) correctly. The ONLY "
                     "defect is `if not data: return 0` - the exact construct this "
                     "task exists to test. One line from correct."),
    "a822906c9b34": ([F2, F6], [],
                     "Empty branch releases a deterministic 0. Separately the scale "
                     "is multiplied by exp(eps/2), which OVER-noises; not marked "
                     "over_noising because the program as a whole leaks through the "
                     "empty branch and the marker would misrepresent it."),
    "f75a436d727d": ([F2, F6], [],
                     "Both branches add noise, so this is not a bare early return - "
                     "but they use DIFFERENT scales (1/eps empty, 2/eps otherwise), "
                     "so the release distribution still depends on whether the "
                     "dataset is empty. int() also truncates the noise."),
    "cda05768d0f3": ([F2, F9], [],
                     "No empty special case. scale = exp(epsilon) is over-noised at "
                     "eps=1 (2.72 vs 1.0) but UNDER-noised at eps=0.3 (1.35 vs 3.33), "
                     "so it is falsified only at the second audit epsilon. Exactly "
                     "the failure the two-epsilon design exists to catch. Also uses "
                     "module-level np.random instead of the injected rng."),
    "05d180abfdce": ([F1, F2, F10], [],
                     "Gaussian with scale sqrt(2*n*eps). Worth recording: at n=0 the "
                     "scale is 0, so rng.normal(0,0) returns exactly 0 and the empty "
                     "dataset gets a deterministic release - the same support "
                     "violation as an early return, produced with NO branch at all. "
                     "Not labelled F7 under the published rule (see revision note)."),
    "88b515b91e89": ([F1, F2, F10], [],
                     "Same shape as the previous: scale sqrt(2n*log(1.3/eps^2)) "
                     "collapses to 0 on the empty dataset with no branch present."),
    "37a929f73c39": ([F1, F2, F6, F10], [],
                     "Explicit empty branch that draws from a DIFFERENT distribution "
                     "rather than returning a constant - still data-dependent flow, "
                     "because which distribution you are sampled from reveals whether "
                     "the dataset was empty."),
    "6af9aa174c55": ([F1, F2, F8], [],
                     "Releases len(data) plus a single Bernoulli draw. It never counts "
                     "the 1s at all, and bounded {0,1} noise cannot satisfy pure "
                     "eps-DP at any scale."),
})

# ---------------------------------------------------------------------------
# two_queries_split_budget_v1 - tier 2, eps=1.0, add/remove-one.
# Two releases on the same data, so sequential composition applies: eps/2 each,
# Laplace scale 2/eps for BOTH. Falsified 11/11 despite detection power 0.33.
# ---------------------------------------------------------------------------
LABELS.update({
    "ba52f2e57b16": ([F2, F4, F6], [],
                     "Different scales for the two queries and neither composes: "
                     "1/(2*eps) implies spending 2*eps on the count, |eps| implies a "
                     "full eps on the sum. Total budget 3*eps against a claim of eps."),
    "2d6ffed8839c": ([F2, F4, F6, F10], [],
                     "Per-query budgets are log(1/eps)/n and log(1/eps)*n. At the "
                     "claimed eps = 1, log(1) = 0, so BOTH scales are zero and both "
                     "the count and the sum are released exactly."),
    "046ff82a5f99": ([F2, F4, F6], [],
                     "Splits the budget correctly to eps/2 and then passes it "
                     "straight into the scale slot, so scale = eps/2 = 0.5 where "
                     "2/eps = 2 is required. Each query spends 2*eps; total 4*eps."),
    "4db9182a9810": ([F2, F4, F6], [],
                     "scale = 1/exp(eps/2) = 0.61 against the required 2. No budget "
                     "accounting between the two releases."),
    "75ad790d9716": ([F6, F7], [],
                     "Sensitivity 2, scale 2/eps - CORRECT for both queries, and "
                     "floor()/max(0,.) are legitimate post-processing of the noisy "
                     "value. Sole defect is the empty-input early return."),
    "e42f73589e0b": ([F1, F2, F6], [],
                     "Centres each release on a deterministic shrunk value "
                     "floor(n/exp(eps)) and then subtracts Laplace noise whose scale "
                     "2n depends on the dataset size."),
    "d56d90e997b2": ([F6, F8], [],
                     "The count is correct (scale 2/eps). The sum instead adds "
                     "scale * rng.choice([-1,1]) - a Rademacher draw on two points. "
                     "Bounded two-point support cannot be DP at any scale."),
    "f67d05d01169": ([F2, F4, F6], [],
                     "noisy_count never reads n at all - it releases "
                     "0.5*(1 + Lap(1)), independent of the data. The sum is scaled by "
                     "eps/2 after noising, giving an effective scale of 0.5 against "
                     "the required 2."),
    "e1ae54acbccb": ([F6, F7], [],
                     "Scales are over-noised on the count and exact on the sum, so "
                     "neither leaks. The defect is min(n + laplace_count, n): the "
                     "noisy count is capped at the TRUE n, a data-dependent bound "
                     "that reveals n from above. min(sum, 1) is fine - 1 is public."),
    "2532e4c9d70c": ([F1, F2, F6], [],
                     "Splits the budget correctly then uses scale = alpha/len(data), "
                     "which is both inverted and data-dependent."),
    "135ad1a06042": ([F1, F2, F10], [],
                     "The only program in this task with NO empty branch, and it "
                     "still fails on empty input: scale sqrt(2*n*delta) with "
                     "delta = eps/(2(n+1)) is zero at n = 0. Gaussian under delta=0."),
})

# ---------------------------------------------------------------------------
# categorical_histogram_v1 - tier 1, eps=1.0, add/remove-one.
# Fixed public bins {A,B,C}; L1 sensitivity 1; per-bin Laplace(1/eps), and
# EVERY bin must be noised including empty ones.
# ---------------------------------------------------------------------------
LABELS.update({
    "6eb1fd2e48e8": ([F2, F10], [],
                     "rng.laplace(0, 1) hard-codes the scale, so epsilon never "
                     "reaches the noise. Correct by coincidence at the claimed "
                     "eps = 1 and under-noised at the second audit epsilon 0.3. The "
                     "nested loop also adds three draws to every bin, and the "
                     "max_noise loop mutates the loop variable and does nothing."),
    "62f31455578e": ([F1, F2, F8], [],
                     "Never counts the labels. Draws Binomial(n, 0.4) and "
                     "Binomial(n, 0.6), so the release is a function of len(data) "
                     "alone and concentrates around 0.4n and 0.6n. epsilon is unused "
                     "and binomial support is bounded."),
    "cb0034ba7a4f": ([F6, AMB], [],
                     "Laplace scale is correct and max(0,.) is fine, but after "
                     "noising it rescales every bin by total/total_noise, where "
                     "`total` is the TRUE count. Post-processing that reads private "
                     "state is not post-processing - the released bins now sum to the "
                     "exact true total. No family in the rules covers this; labelled "
                     "ambiguous deliberately (plus F6 for the total==0 early return). "
                     "See the taxonomy-gap note in FINDINGS."),
    "ec48cf299f03": ([F2, F6], [],
                     "Empty-input early return of the all-zero dict. scale "
                     "1/exp(eps/2) = 0.61 against the required 1.0; int(round(.)) is "
                     "legitimate post-processing of the noisy value."),
    "53384003d82a": ([F6, F7], [],
                     "Laplace scale 1/eps correct, every bin noised, max(0,.) fine. "
                     "Sole defect is `if not data: return counts`, releasing the "
                     "all-zero dict deterministically."),
    "015a7f861e96": ([F2, F10], [],
                     "No empty branch - handles zero bins correctly. Uses "
                     "rng.normal(), i.e. Gaussian noise, under a delta = 0 claim."),
    "1de5d82bfdba": ([F2], [],
                     "Clean single-cause inversion: scale = epsilon/3 where 1/epsilon "
                     "is required. Bins, counting and empty handling all correct."),
    "833ead78d227": ([F2], [],
                     "As above with scale = 2*epsilon/3. Epsilon in the numerator."),
})

# ---------------------------------------------------------------------------
# above_threshold_v1 - tier 2, eps=1.0, add/remove-one. Sparse Vector.
# Correct: rho ~ Lap(2/eps) ONCE on the threshold; nu_i ~ Lap(4/eps) fresh per
# query; halt at the first crossing; release the INDEX only.
# Calibrated detection power 0.00 at this task size (FINDINGS section 2b), so
# these seven are what the auditor caught DESPITE the task being too small to
# demonstrate the full harm.
# ---------------------------------------------------------------------------
LABELS.update({
    "d5d38e4865ab": ([F2, F4, F10], [],
                     "Structure is right - threshold noised once, index only - but "
                     "every scale is 1/sensitivity = 1, so epsilon never reaches any "
                     "noise draw. No budget split between threshold and queries."),
    "807e30a726f5": ([F2, F4, F10], [],
                     "The threshold is used RAW - no rho is ever drawn. This is the "
                     "single most common published SVT error. Query scale is a "
                     "hard-coded 1, so epsilon is unused."),
    "9bfbd03ed3fb": ([F2, F4], [],
                     "Threshold noised once and index-only, both correct. Scales are "
                     "epsilon/3 for threshold AND queries: inverted, and missing the "
                     "factor of 2 that pays for the unbounded below-threshold "
                     "answers."),
    "18735ec15b6c": ([F2, F4, F8], [],
                     "Draws ONE noise value per iteration and adds it to both sides: "
                     "`threshold = T + noise` then `if q + noise > threshold`. The "
                     "noise cancels algebraically and the test reduces to q > T, a "
                     "deterministic comparison. The FINDINGS section 4 case exactly - "
                     "shared randomness COUPLING two quantities rather than "
                     "collapsing repeats."),
    "646ccf828a48": ([F2, F4, F9], [],
                     "No threshold noise, query scale 1/eps instead of 4/eps, and "
                     "module-level np.random.laplace instead of the injected rng."),
    "e0b98e20b010": ([F1, F2, F4], [],
                     "Noise scale eps/(2*(q1+q2+q3+4)) is computed from the query "
                     "ANSWERS, so the amount of noise depends on the data it is "
                     "meant to hide. Threshold and query scales are also equal."),
    "c24be86750b2": ([F1, F2, F4], [],
                     "Threshold scale eps/(2(T+1)) is at least public. Query scale "
                     "eps/(2n) is data-dependent and inverted, and shrinks as the "
                     "dataset grows."),
})

# ---------------------------------------------------------------------------
# adaptive_two_step_v1 - tier 3, eps=1.0, add/remove-one.
# Correct: eps/2 per step (scale 2/eps), and the branch taken on the NOISY
# step-1 release, never the raw count.
# Notable: not one program branched on the raw count. The subtle failure the
# task was built to probe did not occur; the failures are ordinary budget and
# calibration errors.
# ---------------------------------------------------------------------------
LABELS.update({
    "37c488a4304b": ([F1, F2, F8], [],
                     "Never reads the data values. Draws len(data) uniforms and "
                     "counts how many exceed 0.5, so both releases are Binomial(n, "
                     "0.5) draws that reveal n. epsilon is unused."),
    "9eab532590bb": ([F1, F2, F8], [],
                     "Same shape: rng.binomial(n=len(data), p=0.5) in place of the "
                     "count, so the release is a function of n alone."),
    "177600641ef0": ([F2, F4], [],
                     "Branches on the noisy value - correct. Scale sqrt(2*eps) = 1.41 "
                     "against the required 2/eps = 2, so each step under-noises and "
                     "the two together exceed the claimed budget."),
    "b3508d598c5e": ([F1, F2, F10], [],
                     "Branches on the noisy value - correct. But the scales are "
                     "1/sqrt(n) and sqrt(n), which never mention epsilon and depend "
                     "on the data. Gaussian under a delta = 0 claim."),
    "d214c5bd497b": ([F4], [],
                     "Cleanest single-cause composition error in the benchmark. "
                     "Sensitivity, noise family, epsilon placement and the "
                     "branch-on-noisy-value are ALL correct; scale 1/eps is exactly "
                     "right for ONE release. Both steps get it, so the total is "
                     "2*eps against a claim of eps. The canonical F4 instance."),
    "e8c097986e7f": ([F4], [],
                     "Identical defect, more verbose: sensitivity/epsilon for both "
                     "steps, no split, total 2*eps."),
    "26fe876f579a": ([F1, F2, F4], [],
                     "Attempts a split (the /2) but writes eps/(2n) as the SCALE, so "
                     "it is inverted and data-dependent at once."),
})

# ---------------------------------------------------------------------------
# gaussian_bounded_count_v1 - tier 1, (eps=1.0, delta=1e-5). Classical Gaussian:
# sigma = sqrt(2*ln(1.25/delta))/eps = 4.85. Detection power 0.00 - the task is
# marked partially_auditable because a pure-eps auditor cannot properly falsify
# an (eps, delta) mechanism, so these three are weak evidence by construction.
# ---------------------------------------------------------------------------
LABELS.update({
    "c09e6cad10a0": ([F1, F2, F8], [],
                     "Concatenates the noise array onto the data and then counts "
                     "entries equal to 1. Continuous noise is almost never exactly "
                     "1.0, so the function returns the TRUE count with probability 1. "
                     "delta is never used."),
    "6fdf0cef3edc": ([F1, F2], [],
                     "sigma = sqrt(2*ln(1.25/delta)) omits the division by epsilon. "
                     "It also draws len(data) values and sums them, so the effective "
                     "sigma scales with sqrt(n) - and is exactly 0 on empty input."),
    "65ad3dd4597d": ([F2], [],
                     "Clean calibration error: sqrt(ln(2/delta)/(2*eps)) = 2.47 "
                     "against the classical 4.85. A zCDP-shaped formula used where "
                     "the classical Gaussian bound was claimed."),
})

# ---------------------------------------------------------------------------
# repeated_release_v1 - tier 2, eps=1.0. Four releases of the SAME query, so
# eps/4 each, scale 4/eps = 4.
# ---------------------------------------------------------------------------
LABELS.update({
    "024e93a0ad8d": ([F2, F4], [],
                     "Fresh noise per release - correct. scale sqrt(2*ln(1/(0.5 "
                     "eps))) = 1.18 against the required 4, so K = 4 never enters the "
                     "budget."),
    "bb390c764934": ([F2, F4, F10], [],
                     "scale = sensitivity/epsilon = 1: the correct scale for ONE "
                     "release, used for all four. Gaussian under a delta = 0 claim."),
    "b5a73c1d8774": ([F1, F2, F7, F10], [],
                     "Gaussian under delta = 0 with a hard-coded delta = 0.01; scale "
                     "multiplied by sqrt(n); and min(n, round(.)) clamps each release "
                     "to the true dataset size, a data-dependent bound."),
})

# ---------------------------------------------------------------------------
# report_noisy_max_v1 - tier 2, eps=1.0. Laplace(2/eps) on each of four counts,
# release the LABEL only. Only one program reached audit at all.
# ---------------------------------------------------------------------------
LABELS.update({
    "f5e191dd43a1": ([F1, F2, F5, F6], [],
                     "Three separate data-dependent early returns, each a distinct "
                     "string: 'No samples' on empty, 'Too many samples' when "
                     "len(data) > 4, and 'No tie' when the noisy max falls below a "
                     "threshold computed from the true total. Datasets of size 4 and "
                     "5 are perfectly distinguishable. Selection noise scale is "
                     "1/total_count - data-dependent, and epsilon is never used."),
})

# ---------------------------------------------------------------------------
# subsampled_count_v1 - tier 3, eps=0.5, q=0.5. Amplification runs BACKWARDS
# from the claim: eps0 = ln(1 + (e^eps - 1)/q) = 0.832, scale 1/eps0 = 1.20.
# The naive error is to run the base mechanism at eps0 = eps and then claim the
# amplified value, asserting a guarantee nothing paid for.
# ---------------------------------------------------------------------------
LABELS.update({
    "0e19910d98f8": ([F1, F2, F10], [],
                     "Subsamples from the supplied rng - correct. Then scale = "
                     "eps0/n, inverted and data-dependent (0.1 at n=5 against the "
                     "required 1.20). Sets eps0 = eps and claims the amplified value."),
    "23a8af2dc74a": ([F2, F10], [],
                     "scale = eps0/(2q) = 0.5 against 1.20. eps0 = eps = 0.5 with the "
                     "amplified guarantee claimed on top - the naive direction."),
    "7e6bae85a7c6": ([F2, F10], [],
                     "Picks eps0 = 0.3 and states the amplification relation in a "
                     "comment, but then writes scale = eps0/q = 0.6 where 1/eps0 = "
                     "3.33 is required. Epsilon inverted in the scale."),
})

# ---------------------------------------------------------------------------
# validated_clipped_count_v1 - tier 3, eps=1.0. Count valid ints in [0,100],
# + Laplace(1/eps), with validation that is TOTAL and SILENT.
# Notable: no program was falsified for raising on a malformed record - the
# leak-through-exception failure the task was built to probe did not occur.
# ---------------------------------------------------------------------------
LABELS.update({
    "73fd5fa086d3": ([F8], [],
                     "Returns the exact count. rng is never called and no noise of "
                     "any kind is added - a one-line function with no mechanism in "
                     "it at all."),
    "7062dc88f666": ([F1, F2, F10], [],
                     "Draws `valid_count` standard normals and returns their SUM, "
                     "discarding the count itself. The release is N(0, eps*count), so "
                     "its variance encodes exactly the quantity being protected."),
    "e53fcac5c488": ([F1, F2, F10], [],
                     "Subtracts a deterministic n*(1-exp(-eps)) offset from the count "
                     "and adds Gaussian noise with scale sqrt(eps*n) under a "
                     "delta = 0 claim. Both terms depend on the dataset size."),
})


# ---------------------------------------------------------------------------
# EXPLORATORY EXTENSION - deliberately kept OUT of the primary labels.
#
# Assessment 6: "do not silently add F11 to the primary analysis after seeing the
# output. That would look post hoc." Correct, and the fix is structural rather
# than a promise: the primary `labels` field is frozen as the pre-registered
# ten-family coding, and this second field carries the exploratory recoding. Both
# are written to the output, so any table can be produced under either coding and
# the difference is always visible.
#
# F11 was identified DURING annotation, not before. It covers post-processing
# that reads private state: a function applied after the noisy release that
# depends on data other than the release itself. The post-processing theorem does
# not cover it, because the theorem requires dependence only on the released
# value.
#
# Exactly one program in the corpus falls here, and under the pre-registered
# taxonomy it is coded `ambiguous`, which is the honest pre-registered answer.
F_ELEVEN = "F11_private_state_postprocessing"

EXPLORATORY: dict[str, list[str]] = {
    # categorical_histogram: noises every bin correctly at scale 1/eps, then
    # rescales all bins by total/total_noise where `total` is the TRUE count, so
    # the released bins sum to the exact private quantity.
    "cb0034ba7a4f": [F_ELEVEN],
}


def resolve() -> list[dict]:
    rows, seen = [], set()
    for f in sorted(glob.glob(str(ROOT / "results" / "processed" / "outcomes_*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r["model"], r["task_id"], r["sample_idx"])
            if k in seen or r["outcome"] != "falsified":
                continue
            seen.add(k)
            rows.append(r)

    by_prefix = collections.defaultdict(list)
    for r in rows:
        by_prefix[r["code_sha256"][:12]].append(r)

    out, problems = [], []
    for pref, (labels, markers, note) in LABELS.items():
        hits = by_prefix.get(pref, [])
        if len(hits) != 1:
            problems.append(f"{pref}: resolved to {len(hits)} programs")
            continue
        r = hits[0]
        # `labels` is the pre-registered ten-family coding and is never modified
        # by the exploratory extension. `exploratory_labels` replaces `ambiguous`
        # with F11 where applicable; everywhere else the two are identical.
        # Order is preserved rather than sorted, so a diff of the two fields
        # shows only genuine recodings and not alphabetisation artefacts.
        expl = EXPLORATORY.get(pref)
        exploratory = ([l for l in labels if l != AMB] + expl) if expl else list(labels)
        out.append({"code_sha256": r["code_sha256"], "task_id": r["task_id"],
                    "labels": labels, "markers": markers, "note": note,
                    "exploratory_labels": exploratory})
    if problems:
        raise SystemExit("key resolution failed:\n  " + "\n  ".join(problems))

    missing = [r for r in rows if r["code_sha256"][:12] not in LABELS]
    return out, missing, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    out, missing, total = resolve()
    print(f"labelled {len(out)} / {total} falsified programs")
    if missing:
        c = collections.Counter(r["task_id"] for r in missing)
        print(f"still unlabelled ({len(missing)}):")
        for t, n in sorted(c.items()):
            print(f"  {t:32s} {n}")
    if a.check:
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
