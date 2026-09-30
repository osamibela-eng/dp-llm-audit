"""Known-buggy variants of subsampled_count_v1, for auditor calibration.

Marked `partially_auditable` in tasks.yaml: subsampling flattens the output
distribution, so several of these are expected to survive the audit. That is a
coverage finding, not a defect -- it is exactly the kind of blind spot the
calibration table has to record for `not_falsified` to mean anything.
"""
import math

Q = 0.5

EXPECTED = {
    # runs the base mechanism at eps0 = eps and then CLAIMS the amplified
    # value. The true guarantee is ln(1 + q(e^eps - 1)) = 0.288 at eps = 0.5,
    # so the claim of 0.5 is... actually SATISFIED. This variant is wrong in
    # its reasoning but conservative in effect: it under-spends. Included
    # because it is the most likely model error and it must NOT be flagged.
    "private_subsampled_count_eps0_equals_eps": "likely_missed",
    # skips subsampling entirely but keeps the amplified eps0. The base
    # mechanism then runs at eps0 = 0.83 with no amplification to pay for it,
    # so the release is 0.83-DP against a claim of 0.5.
    "private_subsampled_count_no_subsampling": "detectable",
    # deterministic subsample (every other record) -> the "sample" is a fixed
    # function of position, so amplification does not apply at all
    "private_subsampled_count_deterministic_sample": "detectable",
    # samples with a module-level RNG instead of the supplied Generator. The
    # randomness is real, so the amplification argument still holds; the bug is
    # a seeding/reproducibility violation, detectable by AST scan rather than
    # by the auditor.
    "private_subsampled_count_wrong_rng": "likely_missed",
}


def private_subsampled_count_eps0_equals_eps(data, epsilon, rng):
    """BUG: base budget set to eps, then the amplified guarantee claimed.

    Over-noised relative to the claim, so not a violation -- but the reasoning
    is inverted, and a reviewer reading the code would accept a false argument.
    """
    keep = [x for x in data if rng.random() < Q]
    c = sum(1 for x in keep if x == 1)
    return float(c + rng.laplace(scale=1.0 / epsilon))


def private_subsampled_count_no_subsampling(data, epsilon, rng):
    """BUG: keeps the amplified base budget but never subsamples."""
    eps0 = math.log(1.0 + (math.exp(epsilon) - 1.0) / Q)
    c = sum(1 for x in data if x == 1)
    return float(c + rng.laplace(scale=1.0 / eps0))


def private_subsampled_count_deterministic_sample(data, epsilon, rng):
    """BUG: takes every other record instead of sampling."""
    eps0 = math.log(1.0 + (math.exp(epsilon) - 1.0) / Q)
    keep = list(data)[::2]                            # fixed, not random
    c = sum(1 for x in keep if x == 1)
    return float(c + rng.laplace(scale=1.0 / eps0))


def private_subsampled_count_wrong_rng(data, epsilon, rng):
    """BUG: subsamples with numpy's global RNG, not the supplied Generator."""
    import numpy as np
    eps0 = math.log(1.0 + (math.exp(epsilon) - 1.0) / Q)
    keep = [x for x in data if np.random.random() < Q]   # module-level RNG
    c = sum(1 for x in keep if x == 1)
    return float(c + rng.laplace(scale=1.0 / eps0))
