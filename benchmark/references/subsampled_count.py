"""Reference implementation: task subsampled_count_v1 (correct, eps-DP)."""
import math

Q = 0.5        # inclusion probability, fixed by the task spec


def private_subsampled_count(data, epsilon, rng):
    """Poisson-subsampled Laplace count, claiming the AMPLIFIED epsilon.

    Amplification by subsampling (add/remove-one, Poisson sampling with
    inclusion probability q): a base mechanism that is eps0-DP becomes

        eps = ln(1 + q * (e^{eps0} - 1))

    DP overall. The task claims eps = 0.5, so we invert that relation to pick
    the base budget rather than reusing eps as if amplification were free:

        eps0 = ln(1 + (e^{eps} - 1) / q)

    At eps = 0.5, q = 0.5 this gives eps0 = 0.8318, i.e. the inner Laplace
    mechanism is run at a LOOSER budget than the claim, and subsampling pays
    the difference. Stating eps0 explicitly is required by the spec.

    Two things this gets right:

    * The subsample is drawn from the supplied `rng`. Sampling from a separate
      or fixed source breaks the guarantee, because amplification relies on the
      sample being genuinely random and unknown to the adversary.
    * The mechanism is honest about direction: the naive error is to run the
      base mechanism at eps0 = eps and then CLAIM the amplified value, which
      quietly asserts a guarantee stronger than anything that was paid for.

    Auditability note: tasks.yaml marks this `partially_auditable`. Amplified
    guarantees are hard to falsify empirically at these sample sizes because
    the sampling step flattens the output distribution, so a `not_falsified`
    verdict here carries even less weight than usual.
    """
    eps0 = math.log(1.0 + (math.exp(epsilon) - 1.0) / Q)

    keep = [x for x in data if rng.random() < Q]      # Poisson subsample
    true_count = sum(1 for x in keep if x == 1)
    return float(true_count + rng.laplace(loc=0.0, scale=1.0 / eps0))
