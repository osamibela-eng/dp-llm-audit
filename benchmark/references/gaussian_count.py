"""Reference implementation: task gaussian_bounded_count_v1 ((eps, delta)-DP)."""
import math


def private_count_gaussian(data, epsilon, delta, rng):
    """DP count of 1s via the CLASSICAL Gaussian mechanism.

    Calibration: classical Gaussian mechanism (Dwork & Roth, Thm A.1),
        sigma = sqrt(2 * ln(1.25 / delta)) * sensitivity / epsilon
    with sensitivity 1 under add/remove-one adjacency.

    This bound is only valid for epsilon <= 1; for larger epsilon the analytic
    Gaussian mechanism (Balle & Wang 2018) should be used instead. The task
    claims epsilon = 1.0, delta = 1e-5, so the classical form applies -- and we
    state it here because the task spec asks for the calibration to be named.

    Note for auditing: a pure-eps auditor cannot falsify an (eps, delta)
    mechanism without accounting for delta, which is why tasks.yaml marks this
    task `partially_auditable`. That expectation is part of the calibration
    story, not a defect.
    """
    sensitivity = 1.0
    true_count = sum(1 for x in data if x == 1)
    sigma = math.sqrt(2.0 * math.log(1.25 / delta)) * sensitivity / epsilon
    return float(true_count + rng.normal(loc=0.0, scale=sigma))
