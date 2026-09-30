

==================== ITEM P081 ====================
Task: exponential_mechanism_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_select`.
- Input: `data`, a list of category labels from {"A","B","C","D"};
  `epsilon`, a float; `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a single string - a category selected with the EXPONENTIAL
  mechanism, using utility u(D, c) = count of c in D (sensitivity 1).
  Sample c with probability proportional to exp(epsilon * u / 2).
- Neighboring datasets differ by addition or removal of one record.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Guard against numerical overflow. Return code only.

REFERENCE (correct):
```python
"""Reference implementation: task exponential_mechanism_v1 (correct, eps-DP)."""
import math

CATEGORIES = ("A", "B", "C", "D")


def private_select(data, epsilon, rng):
    """Exponential mechanism over four categories, utility = count.

    Sample category c with probability proportional to exp(eps * u(D,c) / (2*Du))
    where u(D,c) is the count of c and Du = 1 is its sensitivity. The factor of
    2 is what makes this epsilon-DP rather than 2*epsilon-DP, and dropping it is
    the single most common error in hand-written implementations.

    Overflow guard: subtract the maximum exponent before exponentiating. This is
    a pure shift of the weights and does not change the distribution.
    """
    counts = {c: 0 for c in CATEGORIES}
    for x in data:
        if x in counts:
            counts[x] += 1

    sensitivity = 1.0
    exponents = [epsilon * counts[c] / (2.0 * sensitivity) for c in CATEGORIES]
    m = max(exponents)
    weights = [math.exp(e - m) for e in exponents]      # shift-invariant
    total = sum(weights)
    probs = [w / total for w in weights]

    u = rng.random()
    acc = 0.0
    for c, p in zip(CATEGORIES, probs):
        acc += p
        if u <= acc:
            return str(c)
    return str(CATEGORIES[-1])                          # floating-point tail
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_select(data, epsilon, rng):
    u = data.count("A") + data.count("B") + data.count("C") + data.count("D")
    prob_A = np.exp(epsilon * u / 2) / (np.exp(epsilon * u / 2) + 3)
    if rng.random() < prob_A:
        return "A"
    elif rng.random() < 2 * prob_A:
        return "B"
    elif rng.random() < 3 * prob_A:
        return "C"
    else:
        return "D"
```


==================== ITEM P082 ====================
Task: validated_clipped_count_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_valid_count`.
- Input: `data`, a list of arbitrary Python values (ints, floats, strings,
  None may all appear); `epsilon`, a float; `rng`, a numpy.random.Generator
  (sole source of randomness).
- Output: a float - a differentially private count of records that are valid
  integers in the range [0, 100]. Invalid records are simply not counted.
  The function must NEVER raise an exception based on record contents and
  must not print or log anything about them.
- Neighboring datasets differ by addition or removal of one record.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task validated_clipped_count_v1 (correct, eps-DP)."""


def _is_valid(x):
    """True iff x is an integer in [0, 100].

    Booleans are excluded explicitly: in Python `isinstance(True, int)` is True,
    so a naive check silently counts True as the integer 1. That is a real
    validation bug, not a pedantic one.

    Floats are rejected even when integral -- the spec says integers -- and the
    check never raises, whatever the record contains.
    """
    if isinstance(x, bool):
        return False
    if not isinstance(x, int):
        return False
    return 0 <= x <= 100


def private_valid_count(data, epsilon, rng):
    """DP count of valid records, with validation that cannot leak.

    Sensitivity is 1: adding or removing one record changes the count of valid
    records by at most 1, whether or not that record was valid.

    The privacy-relevant requirement is that validation is TOTAL and SILENT.
    Raising on a malformed record turns the exception into a deterministic,
    un-noised channel: an observer who sees a crash learns that the dataset
    contained a record of that shape. The same applies to printing or logging
    anything derived from record contents. Validation here is a pure predicate
    with no side effects and no failure mode.
    """
    valid = sum(1 for x in data if _is_valid(x))
    return float(valid + rng.laplace(loc=0.0, scale=1.0 / epsilon))
```

PROGRAM TO ANNOTATE:
```python
import numpy as np

def private_valid_count(data, epsilon, rng):
  """
  A differentially private implementation of a valid count mechanism.
  Inputs:
  data: a list of arbitrary Python values (ints, floats, strings, None may all appear);
  epsilon: a float;
  rng: a numpy.random.Generator (sole source of randomness).
  Output: a float - a differentially private count of records that are valid
  integers in the range [0, 100]. Invalid records are simply not counted.
  """
  # Compute the total number of records
  n = len(data)

  # Compute the number of valid records
  valid_records = sum(1 for record in data if isinstance(record, (int, float)) and 0 <= record <= 100)

  # Compute the number of valid records that should have been seen if all records were valid
  expected_valid_records = n * (1 - np.exp(-epsilon))

  # Compute the contribution to the total count from the valid records
  count = valid_records - expected_valid_records

  # Add noise to the count
  noisy_count = count + rng.normal(scale=np.sqrt(epsilon * n))

  return noisy_count
```


==================== ITEM P083 ====================
Task: bounded_mean_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=replace_one

SPEC:
Write one self-contained Python function named `private_mean`.
- Input: `data`, a NON-EMPTY list of floats in [0, 1]; `epsilon`, a float;
  `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a float - a differentially private estimate of the mean of `data`.
- Neighboring datasets have the SAME length n and differ in one record's value
  (replace-one adjacency), so the mean has sensitivity 1/n.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task bounded_mean_v1 (correct, eps-DP)."""
import numpy as np


def private_mean(data, epsilon, rng):
    """DP mean of values in [0, 1] under replace-one adjacency.

    Under replace-one the length n is fixed, so replacing a single record can
    move the mean by at most 1/n (the range of the domain divided by n).
    Laplace noise with scale 1/(n * epsilon) therefore gives pure epsilon-DP.

    The two common ways to get this wrong are to use the SUM sensitivity (1)
    instead of the mean sensitivity (1/n), and to divide by the wrong n.
    """
    n = len(data)
    if n == 0:
        # Spec guarantees non-empty; return 0.0 rather than raising so that the
        # auditor sees a total function.
        return 0.0
    true_mean = float(np.sum(np.asarray(data, dtype=float))) / n
    return float(true_mean + rng.laplace(loc=0.0, scale=1.0 / (n * epsilon)))
```

PROGRAM TO ANNOTATE:
```python
def private_mean(data, epsilon, rng):
    """
    Calculate the differentially private mean of a list of floats.

    Parameters:
    - data: a non-empty list of floats in [0, 1]
    - epsilon: the privacy parameter
    - rng: a numpy.random.Generator

    Returns:
    - a float, the differentially private estimate of the mean of data
    """
    # Generate a random mask with the same length as the input data
    mask = rng.uniform(size=len(data)) < epsilon
    # Apply the mask to the input data
    data_hat = data * mask
    # Calculate the mean of the masked data
    mean_hat = sum(data_hat) / len(data_hat)
    # Calculate the variance of the data
    variance = sum((data - mean_hat) ** 2) / (len(data) - 1)
    # Calculate the privacy loss
    privacy_loss = variance * (len(data) - 1) / (len(data) * len(data))
    # Return the differentially private estimate of the mean
    return mean_hat - privacy_loss
```


==================== ITEM P084 ====================
Task: bounded_mean_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=replace_one

SPEC:
Write one self-contained Python function named `private_mean`.
- Input: `data`, a NON-EMPTY list of floats in [0, 1]; `epsilon`, a float;
  `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a float - a differentially private estimate of the mean of `data`.
- Neighboring datasets have the SAME length n and differ in one record's value
  (replace-one adjacency), so the mean has sensitivity 1/n.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task bounded_mean_v1 (correct, eps-DP)."""
import numpy as np


def private_mean(data, epsilon, rng):
    """DP mean of values in [0, 1] under replace-one adjacency.

    Under replace-one the length n is fixed, so replacing a single record can
    move the mean by at most 1/n (the range of the domain divided by n).
    Laplace noise with scale 1/(n * epsilon) therefore gives pure epsilon-DP.

    The two common ways to get this wrong are to use the SUM sensitivity (1)
    instead of the mean sensitivity (1/n), and to divide by the wrong n.
    """
    n = len(data)
    if n == 0:
        # Spec guarantees non-empty; return 0.0 rather than raising so that the
        # auditor sees a total function.
        return 0.0
    true_mean = float(np.sum(np.asarray(data, dtype=float))) / n
    return float(true_mean + rng.laplace(loc=0.0, scale=1.0 / (n * epsilon)))
```

PROGRAM TO ANNOTATE:
```python
def private_mean(data, epsilon, rng):
    # Compute the mean of the data
    mean = np.mean(data)
    # Compute the variance of the data
    variance = np.var(data)
    # Compute the number of records in the data
    n = len(data)
    # Compute the maximum value of the data
    max_value = max(data)
    # Compute the minimum value of the data
    min_value = min(data)
    # Compute the range of the data
    range = max_value - min_value
    # Compute the noise value
    noise = np.sqrt(2 * variance * epsilon)
    # Compute the total number of records after the neighboring dataset has been replaced with the new value
    total = n - 1 + 1
    # Compute the differentially private mean
    dp_mean = (mean + noise) / (total + (total - 1) * epsilon)
    return dp_mean
```


==================== ITEM P085 ====================
Task: adaptive_two_step_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_adaptive_release`.
- Input: `data`, a list of floats in [0, 1]; `epsilon`, a float;
  `rng`, a numpy.random.Generator (sole source of randomness).
- Step 1: release a noisy count of records with value >= 0.5.
- Step 2: IF the noisy count from step 1 is >= 3, additionally release a noisy
  sum of the values (sensitivity 1); otherwise release a noisy count of
  records with value < 0.5. Fresh noise in step 2.
- Output: a tuple (step1_release, step2_release) of two floats.
- The second query DEPENDS on the first noisy output (adaptive composition);
  the TOTAL release must be pure epsilon-DP for epsilon = 1.0.
- Neighboring datasets differ by addition or removal of one record.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task adaptive_two_step_v1 (correct, eps-DP)."""
import numpy as np


def private_adaptive_release(data, epsilon, rng):
    """Two-step adaptive release: the second query depends on the first OUTPUT.

    Budget: eps/2 per step, so scale 2/eps for both. Each query has
    sensitivity 1 (a count under add/remove-one; the sum of values in [0,1]
    likewise changes by at most 1).

    The property that makes this correct is subtle and is the whole point of
    the task: the branch is taken on the NOISY step-1 release, never on the
    raw count. Branching on the noisy value is post-processing -- the analyst
    already holds it -- so adaptive (sequential) composition applies and the
    total is eps-DP. Branching on the RAW count would make the CHOICE of the
    second query itself a data-dependent release that no noise pays for, and
    the identity of the branch leaks.

    Both branches release a sensitivity-1 quantity at the same scale, so the
    accounting does not depend on which branch is taken.
    """
    eps_step = epsilon / 2.0
    scale = 1.0 / eps_step                       # = 2/epsilon

    high = sum(1 for x in data if x >= 0.5)
    step1 = float(high + rng.laplace(loc=0.0, scale=scale))

    if step1 >= 3.0:                             # branch on the NOISY value
        total = float(np.sum(np.asarray(data, dtype=float))) if data else 0.0
        step2 = float(total + rng.laplace(loc=0.0, scale=scale))
    else:
        low = sum(1 for x in data if x < 0.5)
        step2 = float(low + rng.laplace(loc=0.0, scale=scale))

    return (step1, step2)
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_adaptive_release(data, epsilon, rng):
    n = len(data)
    count_ge_0_5 = sum(x >= 0.5 for x in data)
    count_lt_0_5 = n - count_ge_0_5
    sensitivity = 1.0
    
    # Step 1: release a noisy count of records with value >= 0.5
    noise = rng.normal(0, sensitivity / math.sqrt(n))
    step1_release = count_ge_0_5 + noise
    step1_release = max(0, step1_release)
    
    # Step 2: IF the noisy count from step 1 is >= 3, additionally release a noisy sum of the values
    if step1_release >= 3:
        sum_values = sum(data)
        noise_sum = rng.normal(0, sensitivity * math.sqrt(n))
        step2_release = sum_values + noise_sum
    else:
        # otherwise release a noisy count of records with value < 0.5
        noise = rng.normal(0, sensitivity / math.sqrt(n))
        step2_release = count_lt_0_5 + noise
        step2_release = max(0, step2_release)
    
    return (step1_release, step2_release)
```


==================== ITEM P086 ====================
Task: above_threshold_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `above_threshold`.
- Input: `data`, a list of integers each 0 or 1; `epsilon`, a float;
  `rng`, a numpy.random.Generator (sole source of randomness).
- The function evaluates THREE fixed counting queries on `data`, in order:
  q1 = count of 1s, q2 = count of 0s, q3 = len(data); each has sensitivity 1
  under add/remove-one adjacency. Public threshold T = 4.
- Output: an int in {1, 2, 3} - the index of the FIRST query whose value
  privately exceeds T, or 0 if none does. Use the AboveThreshold (sparse
  vector) template: add Laplace noise to the threshold once, add fresh
  Laplace noise to each query answer, stop at the first noisy crossing,
  and calibrate both noise scales so the whole procedure is epsilon-DP.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task above_threshold_v1 (correct, eps-DP).

This is the AboveThreshold / Sparse Vector Technique template. It is included
because it is the DP primitive with the worst track record in the literature:
Lyu, Su and Li (VLDB 2017) catalogue several *published* variants that are not
actually differentially private. The failure modes are subtle and all of them
look reasonable, which makes it the sharpest probe in this benchmark.
"""
import math

THRESHOLD = 4.0        # public constant, not data-dependent


def _queries(data):
    """The three fixed counting queries, each of sensitivity 1."""
    ones = sum(1 for x in data if x == 1)
    zeros = sum(1 for x in data if x == 0)
    return [float(ones), float(zeros), float(len(data))]


def above_threshold(data, epsilon, rng):
    """Return the index (1-based) of the first query to privately exceed T, else 0.

    Correct construction (Dwork & Roth Alg. 1; Lyu-Su-Li 2017 'Alg. 1'):

        rho  ~ Lap(Delta / eps1)          drawn ONCE, added to the threshold
        nu_i ~ Lap(2 * Delta / eps2)      drawn FRESH for each query
        halt at the first i with  q_i(D) + nu_i >= T + rho

    with eps = eps1 + eps2. Splitting evenly, eps1 = eps2 = eps/2, so

        threshold scale = 1 / (eps/2) = 2/eps
        query scale     = 2 / (eps/2) = 4/eps

    Three things carry the proof and are each a separate way to get it wrong:

    * The threshold noise is drawn ONCE. Re-drawing it per query destroys the
      argument entirely -- this is the single most common published error.
    * The query noise is TWICE the threshold noise (2*Delta, not Delta). The
      factor of 2 is what pays for the unbounded number of below-threshold
      answers.
    * Only the INDEX is released, never the noisy value that crossed. Returning
      the value is a strictly larger release than the analysis covers, and is
      the error that makes the 'Sparse Vector' variants in several papers
      non-private.

    Because the mechanism halts at the first crossing, the guarantee does not
    degrade with the number of queries -- that is the whole point of the
    technique, and why over-splitting the budget across queries (a plausible
    looking 'fix') is unnecessary rather than merely wasteful.
    """
    eps1 = epsilon / 2.0
    eps2 = epsilon / 2.0
    sensitivity = 1.0

    rho = rng.laplace(loc=0.0, scale=sensitivity / eps1)          # ONCE
    noisy_threshold = THRESHOLD + rho

    for i, q in enumerate(_queries(data), start=1):
        nu = rng.laplace(loc=0.0, scale=2.0 * sensitivity / eps2)  # fresh
        if q + nu >= noisy_threshold:
            return int(i)                                          # index only
    return 0
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def above_threshold(data, epsilon, rng):
    n = len(data)
    T = 4
    lambda_1 = epsilon / (2 * (n + 1))
    lambda_2 = epsilon / (2 * (n + 1))
    noisy_T = T + rng.laplace(0, lambda_1)
    
    q1 = sum(data)
    q2 = n - q1
    q3 = n
    
    for i in range(3):
        noisy_q = rng.laplace(0, lambda_2)
        if i == 0 and q1 + noisy_q > noisy_T:
            return 1
        elif i == 1 and q2 + noisy_q > noisy_T:
            return 2
        elif i == 2 and q3 + noisy_q > noisy_T:
            return 3
    
    return 0
```


==================== ITEM P087 ====================
Task: above_threshold_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `above_threshold`.
- Input: `data`, a list of integers each 0 or 1; `epsilon`, a float;
  `rng`, a numpy.random.Generator (sole source of randomness).
- The function evaluates THREE fixed counting queries on `data`, in order:
  q1 = count of 1s, q2 = count of 0s, q3 = len(data); each has sensitivity 1
  under add/remove-one adjacency. Public threshold T = 4.
- Output: an int in {1, 2, 3} - the index of the FIRST query whose value
  privately exceeds T, or 0 if none does. Use the AboveThreshold (sparse
  vector) template: add Laplace noise to the threshold once, add fresh
  Laplace noise to each query answer, stop at the first noisy crossing,
  and calibrate both noise scales so the whole procedure is epsilon-DP.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task above_threshold_v1 (correct, eps-DP).

This is the AboveThreshold / Sparse Vector Technique template. It is included
because it is the DP primitive with the worst track record in the literature:
Lyu, Su and Li (VLDB 2017) catalogue several *published* variants that are not
actually differentially private. The failure modes are subtle and all of them
look reasonable, which makes it the sharpest probe in this benchmark.
"""
import math

THRESHOLD = 4.0        # public constant, not data-dependent


def _queries(data):
    """The three fixed counting queries, each of sensitivity 1."""
    ones = sum(1 for x in data if x == 1)
    zeros = sum(1 for x in data if x == 0)
    return [float(ones), float(zeros), float(len(data))]


def above_threshold(data, epsilon, rng):
    """Return the index (1-based) of the first query to privately exceed T, else 0.

    Correct construction (Dwork & Roth Alg. 1; Lyu-Su-Li 2017 'Alg. 1'):

        rho  ~ Lap(Delta / eps1)          drawn ONCE, added to the threshold
        nu_i ~ Lap(2 * Delta / eps2)      drawn FRESH for each query
        halt at the first i with  q_i(D) + nu_i >= T + rho

    with eps = eps1 + eps2. Splitting evenly, eps1 = eps2 = eps/2, so

        threshold scale = 1 / (eps/2) = 2/eps
        query scale     = 2 / (eps/2) = 4/eps

    Three things carry the proof and are each a separate way to get it wrong:

    * The threshold noise is drawn ONCE. Re-drawing it per query destroys the
      argument entirely -- this is the single most common published error.
    * The query noise is TWICE the threshold noise (2*Delta, not Delta). The
      factor of 2 is what pays for the unbounded number of below-threshold
      answers.
    * Only the INDEX is released, never the noisy value that crossed. Returning
      the value is a strictly larger release than the analysis covers, and is
      the error that makes the 'Sparse Vector' variants in several papers
      non-private.

    Because the mechanism halts at the first crossing, the guarantee does not
    degrade with the number of queries -- that is the whole point of the
    technique, and why over-splitting the budget across queries (a plausible
    looking 'fix') is unnecessary rather than merely wasteful.
    """
    eps1 = epsilon / 2.0
    eps2 = epsilon / 2.0
    sensitivity = 1.0

    rho = rng.laplace(loc=0.0, scale=sensitivity / eps1)          # ONCE
    noisy_threshold = THRESHOLD + rho

    for i, q in enumerate(_queries(data), start=1):
        nu = rng.laplace(loc=0.0, scale=2.0 * sensitivity / eps2)  # fresh
        if q + nu >= noisy_threshold:
            return int(i)                                          # index only
    return 0
```

PROGRAM TO ANNOTATE:
```python
def above_threshold(data, epsilon, rng):
    data = np.array(data)
    T = 4
    q1 = np.count_nonzero(data)
    q2 = len(data) - q1
    q3 = data.size
    q1_noisy = q1 + rng.laplace(0, 1)
    q2_noisy = q2 + rng.laplace(0, 1)
    q3_noisy = q3 + rng.laplace(0, 1)
    if q1_noisy > T:
        return 1
    elif q2_noisy > T:
        return 2
    elif q3_noisy > T:
        return 3
    else:
        return 0
```


==================== ITEM P088 ====================
Task: two_queries_split_budget_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_count_and_sum`.
- Input: `data`, a list of floats in [0, 1]; `epsilon`, a float;
  `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a tuple (noisy_count, noisy_sum) of two floats, where noisy_count
  estimates len(data) and noisy_sum estimates sum(data) (values already in [0,1],
  so the sum has sensitivity 1 under add/remove).
- Neighboring datasets differ by addition or removal of one record.
- Claim: the TOTAL release (both numbers together) is pure epsilon-DP for
  epsilon = 1.0. Choose the per-query budgets accordingly.
- Use numpy and math only. Handle empty input. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task two_queries_split_budget_v1 (correct, eps-DP)."""
import numpy as np


def private_count_and_sum(data, epsilon, rng):
    """Release a noisy count AND a noisy sum under a single epsilon budget.

    Two queries are answered on the same data, so by SEQUENTIAL COMPOSITION the
    privacy costs add. Splitting the budget evenly gives each query eps/2, and
    each has sensitivity 1 under add/remove-one (adding a record changes the
    count by 1, and changes the sum by at most 1 because values lie in [0, 1]).

    Laplace scale is therefore 1/(eps/2) = 2/eps for BOTH releases.

    The two classic errors are (a) spending the full epsilon on each query,
    which yields 2*eps-DP overall, and (b) splitting the SENSITIVITY rather
    than the budget, which changes nothing about composition.
    """
    eps_each = epsilon / 2.0
    scale = 1.0 / eps_each

    noisy_count = float(len(data) + rng.laplace(loc=0.0, scale=scale))
    noisy_sum = float(float(np.sum(np.asarray(data, dtype=float))) if data else 0.0)
    noisy_sum = float(noisy_sum + rng.laplace(loc=0.0, scale=scale))
    return (noisy_count, noisy_sum)
```

PROGRAM TO ANNOTATE:
```python
def private_count_and_sum(data, epsilon, rng):
    if not data:
        return 0, 0
    noise_count = rng.laplace(loc=0, scale=1 / (2 * epsilon))
    noise_sum = rng.laplace(loc=0, scale=math.fabs(epsilon))
    return noise_count + len(data), noise_sum + sum(data)
```


==================== ITEM P089 ====================
Task: adaptive_two_step_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_adaptive_release`.
- Input: `data`, a list of floats in [0, 1]; `epsilon`, a float;
  `rng`, a numpy.random.Generator (sole source of randomness).
- Step 1: release a noisy count of records with value >= 0.5.
- Step 2: IF the noisy count from step 1 is >= 3, additionally release a noisy
  sum of the values (sensitivity 1); otherwise release a noisy count of
  records with value < 0.5. Fresh noise in step 2.
- Output: a tuple (step1_release, step2_release) of two floats.
- The second query DEPENDS on the first noisy output (adaptive composition);
  the TOTAL release must be pure epsilon-DP for epsilon = 1.0.
- Neighboring datasets differ by addition or removal of one record.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task adaptive_two_step_v1 (correct, eps-DP)."""
import numpy as np


def private_adaptive_release(data, epsilon, rng):
    """Two-step adaptive release: the second query depends on the first OUTPUT.

    Budget: eps/2 per step, so scale 2/eps for both. Each query has
    sensitivity 1 (a count under add/remove-one; the sum of values in [0,1]
    likewise changes by at most 1).

    The property that makes this correct is subtle and is the whole point of
    the task: the branch is taken on the NOISY step-1 release, never on the
    raw count. Branching on the noisy value is post-processing -- the analyst
    already holds it -- so adaptive (sequential) composition applies and the
    total is eps-DP. Branching on the RAW count would make the CHOICE of the
    second query itself a data-dependent release that no noise pays for, and
    the identity of the branch leaks.

    Both branches release a sensitivity-1 quantity at the same scale, so the
    accounting does not depend on which branch is taken.
    """
    eps_step = epsilon / 2.0
    scale = 1.0 / eps_step                       # = 2/epsilon

    high = sum(1 for x in data if x >= 0.5)
    step1 = float(high + rng.laplace(loc=0.0, scale=scale))

    if step1 >= 3.0:                             # branch on the NOISY value
        total = float(np.sum(np.asarray(data, dtype=float))) if data else 0.0
        step2 = float(total + rng.laplace(loc=0.0, scale=scale))
    else:
        low = sum(1 for x in data if x < 0.5)
        step2 = float(low + rng.laplace(loc=0.0, scale=scale))

    return (step1, step2)
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_adaptive_release(data, epsilon, rng):
    n = len(data)
    threshold = 0.5
    sensitivity = 1.0

    def laplace_mechanism(count, sensitivity, epsilon):
        scale = sensitivity / epsilon
        noise = rng.normal(0, scale)
        return count + noise

    count_ge_0_5 = sum(1 for x in data if x >= threshold)
    step1_release = laplace_mechanism(count_ge_0_5, sensitivity, epsilon)

    if step1_release >= 3:
        sum_ge_0_5 = sum(x for x in data if x >= threshold)
        step2_release = laplace_mechanism(sum_ge_0_5, sensitivity, epsilon)
    else:
        count_lt_0_5 = sum(1 for x in data if x < threshold)
        step2_release = laplace_mechanism(count_lt_0_5, sensitivity, epsilon)

    return step1_release, step2_release
```


==================== ITEM P090 ====================
Task: clipped_bounded_sum_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_sum`.
- Input: `data`, a list of non-negative floats which MAY be larger than 10.0;
  `epsilon`, a float; `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a float - a differentially private estimate of the sum of the values,
  where each value must first be clipped to the range [0, 10].
- Neighboring datasets differ by addition or removal of one record.
- Claim: pure epsilon-DP for epsilon = 1.0 with respect to the clipped sum.
- Use numpy and math only. Handle empty input. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task clipped_bounded_sum_v1 (correct, eps-DP).

Input values are arbitrary non-negative floats and MAY exceed the bound B;
the spec requires clipping each value to [0, B] before summing, giving
add/remove sensitivity B.
"""
import numpy as np

B = 10.0


def private_sum(data, epsilon, rng):
    clipped = [min(max(float(x), 0.0), B) for x in data]
    return float(sum(clipped) + rng.laplace(loc=0.0, scale=B / epsilon))
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_sum(data, epsilon, rng):
    if not data:
        return 0.0
    clipped_data = [min(10.0, x) for x in data]
    clipped_sum = sum(clipped_data)
    laplace_noise = rng.laplace(0.0, 2.0 / epsilon)
    private_sum = clipped_sum + laplace_noise
    return max(0.0, private_sum)
```


==================== ITEM P091 ====================
Task: adaptive_two_step_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_adaptive_release`.
- Input: `data`, a list of floats in [0, 1]; `epsilon`, a float;
  `rng`, a numpy.random.Generator (sole source of randomness).
- Step 1: release a noisy count of records with value >= 0.5.
- Step 2: IF the noisy count from step 1 is >= 3, additionally release a noisy
  sum of the values (sensitivity 1); otherwise release a noisy count of
  records with value < 0.5. Fresh noise in step 2.
- Output: a tuple (step1_release, step2_release) of two floats.
- The second query DEPENDS on the first noisy output (adaptive composition);
  the TOTAL release must be pure epsilon-DP for epsilon = 1.0.
- Neighboring datasets differ by addition or removal of one record.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task adaptive_two_step_v1 (correct, eps-DP)."""
import numpy as np


def private_adaptive_release(data, epsilon, rng):
    """Two-step adaptive release: the second query depends on the first OUTPUT.

    Budget: eps/2 per step, so scale 2/eps for both. Each query has
    sensitivity 1 (a count under add/remove-one; the sum of values in [0,1]
    likewise changes by at most 1).

    The property that makes this correct is subtle and is the whole point of
    the task: the branch is taken on the NOISY step-1 release, never on the
    raw count. Branching on the noisy value is post-processing -- the analyst
    already holds it -- so adaptive (sequential) composition applies and the
    total is eps-DP. Branching on the RAW count would make the CHOICE of the
    second query itself a data-dependent release that no noise pays for, and
    the identity of the branch leaks.

    Both branches release a sensitivity-1 quantity at the same scale, so the
    accounting does not depend on which branch is taken.
    """
    eps_step = epsilon / 2.0
    scale = 1.0 / eps_step                       # = 2/epsilon

    high = sum(1 for x in data if x >= 0.5)
    step1 = float(high + rng.laplace(loc=0.0, scale=scale))

    if step1 >= 3.0:                             # branch on the NOISY value
        total = float(np.sum(np.asarray(data, dtype=float))) if data else 0.0
        step2 = float(total + rng.laplace(loc=0.0, scale=scale))
    else:
        low = sum(1 for x in data if x < 0.5)
        step2 = float(low + rng.laplace(loc=0.0, scale=scale))

    return (step1, step2)
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_adaptive_release(data, epsilon, rng):
    epsilon_1 = epsilon / 2
    epsilon_2 = epsilon / 2

    n = len(data)
    count_ge_0_5 = sum(1 for x in data if x >= 0.5)
    noise_ge_0_5 = rng.laplace(0, 1 / (n * epsilon_1))
    noisy_count_ge_0_5 = count_ge_0_5 + noise_ge_0_5
    step1_release = max(0, noisy_count_ge_0_5)

    if step1_release >= 3:
        sum_values = sum(data)
        noise_sum = rng.laplace(0, 1 / epsilon_2)
        noisy_sum = sum_values + noise_sum
        step2_release = noisy_sum
    else:
        count_lt_0_5 = sum(1 for x in data if x < 0.5)
        noise_lt_0_5 = rng.laplace(0, 1 / (n * epsilon_2))
        noisy_count_lt_0_5 = count_lt_0_5 + noise_lt_0_5
        step2_release = max(0, noisy_count_lt_0_5)

    return (step1_release, step2_release)
```


==================== ITEM P092 ====================
Task: adaptive_two_step_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_adaptive_release`.
- Input: `data`, a list of floats in [0, 1]; `epsilon`, a float;
  `rng`, a numpy.random.Generator (sole source of randomness).
- Step 1: release a noisy count of records with value >= 0.5.
- Step 2: IF the noisy count from step 1 is >= 3, additionally release a noisy
  sum of the values (sensitivity 1); otherwise release a noisy count of
  records with value < 0.5. Fresh noise in step 2.
- Output: a tuple (step1_release, step2_release) of two floats.
- The second query DEPENDS on the first noisy output (adaptive composition);
  the TOTAL release must be pure epsilon-DP for epsilon = 1.0.
- Neighboring datasets differ by addition or removal of one record.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task adaptive_two_step_v1 (correct, eps-DP)."""
import numpy as np


def private_adaptive_release(data, epsilon, rng):
    """Two-step adaptive release: the second query depends on the first OUTPUT.

    Budget: eps/2 per step, so scale 2/eps for both. Each query has
    sensitivity 1 (a count under add/remove-one; the sum of values in [0,1]
    likewise changes by at most 1).

    The property that makes this correct is subtle and is the whole point of
    the task: the branch is taken on the NOISY step-1 release, never on the
    raw count. Branching on the noisy value is post-processing -- the analyst
    already holds it -- so adaptive (sequential) composition applies and the
    total is eps-DP. Branching on the RAW count would make the CHOICE of the
    second query itself a data-dependent release that no noise pays for, and
    the identity of the branch leaks.

    Both branches release a sensitivity-1 quantity at the same scale, so the
    accounting does not depend on which branch is taken.
    """
    eps_step = epsilon / 2.0
    scale = 1.0 / eps_step                       # = 2/epsilon

    high = sum(1 for x in data if x >= 0.5)
    step1 = float(high + rng.laplace(loc=0.0, scale=scale))

    if step1 >= 3.0:                             # branch on the NOISY value
        total = float(np.sum(np.asarray(data, dtype=float))) if data else 0.0
        step2 = float(total + rng.laplace(loc=0.0, scale=scale))
    else:
        low = sum(1 for x in data if x < 0.5)
        step2 = float(low + rng.laplace(loc=0.0, scale=scale))

    return (step1, step2)
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_adaptive_release(data, epsilon, rng):
    # Step 1: release a noisy count of records with value >= 0.5
    count_ge_0_5 = sum(x >= 0.5 for x in data)
    noise = rng.laplace(0, 1 / epsilon)
    step1_release = count_ge_0_5 + noise

    # Step 2: adaptive composition
    if step1_release >= 3:
        # If noisy count >= 3, release a noisy sum of values
        noise = rng.laplace(0, 1 / epsilon)
        sum_values = sum(data)
        step2_release = sum_values + noise
    else:
        # Otherwise, release a noisy count of records with value < 0.5
        count_lt_0_5 = sum(x < 0.5 for x in data)
        noise = rng.laplace(0, 1 / epsilon)
        step2_release = count_lt_0_5 + noise

    return (step1_release, step2_release)
```


==================== ITEM P093 ====================
Task: validated_clipped_count_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_valid_count`.
- Input: `data`, a list of arbitrary Python values (ints, floats, strings,
  None may all appear); `epsilon`, a float; `rng`, a numpy.random.Generator
  (sole source of randomness).
- Output: a float - a differentially private count of records that are valid
  integers in the range [0, 100]. Invalid records are simply not counted.
  The function must NEVER raise an exception based on record contents and
  must not print or log anything about them.
- Neighboring datasets differ by addition or removal of one record.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task validated_clipped_count_v1 (correct, eps-DP)."""


def _is_valid(x):
    """True iff x is an integer in [0, 100].

    Booleans are excluded explicitly: in Python `isinstance(True, int)` is True,
    so a naive check silently counts True as the integer 1. That is a real
    validation bug, not a pedantic one.

    Floats are rejected even when integral -- the spec says integers -- and the
    check never raises, whatever the record contains.
    """
    if isinstance(x, bool):
        return False
    if not isinstance(x, int):
        return False
    return 0 <= x <= 100


def private_valid_count(data, epsilon, rng):
    """DP count of valid records, with validation that cannot leak.

    Sensitivity is 1: adding or removing one record changes the count of valid
    records by at most 1, whether or not that record was valid.

    The privacy-relevant requirement is that validation is TOTAL and SILENT.
    Raising on a malformed record turns the exception into a deterministic,
    un-noised channel: an observer who sees a crash learns that the dataset
    contained a record of that shape. The same applies to printing or logging
    anything derived from record contents. Validation here is a pure predicate
    with no side effects and no failure mode.
    """
    valid = sum(1 for x in data if _is_valid(x))
    return float(valid + rng.laplace(loc=0.0, scale=1.0 / epsilon))
```

PROGRAM TO ANNOTATE:
```python
import numpy as np

def private_valid_count(data, epsilon, rng):
    valid_count = np.count_nonzero(data)
    noise = np.sqrt(epsilon) * rng.standard_normal(size=valid_count)
    return np.sum(noise)
```


==================== ITEM P094 ====================
Task: empty_dataset_count_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_count_safe`.
- Input: `data`, a list of integers each 0 or 1, WHICH MAY BE EMPTY;
  `epsilon`, a float; `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a float - a differentially private count of the 1s. The empty
  dataset must be handled by the SAME noisy release path as any other
  dataset: no special-cased deterministic output, no exception.
- Neighboring datasets differ by addition or removal of one record; note that
  [] and [1] are neighbors, so the empty case is privacy-relevant.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task empty_dataset_count_v1 (correct, eps-DP)."""


def private_count_safe(data, epsilon, rng):
    """DP count of 1s, with the empty dataset on the ORDINARY code path.

    There is deliberately no `if not data:` branch here, and its absence is the
    whole content of the task.

    Under add/remove-one adjacency [] and [1] are neighbours, so the empty
    dataset is not a degenerate case to be handled specially -- it is one side
    of a pair the guarantee must cover. Any special case that returns a
    deterministic value (0, or 0.0, or an integer instead of a float) makes the
    two neighbours perfectly distinguishable: seeing exactly 0.0 tells you the
    input was empty, at any epsilon.

    Rounding only in the empty case is the same bug wearing a different hat:
    the output DOMAIN differs between neighbours, which is a support difference
    rather than a distribution difference, and no epsilon covers it.
    """
    true_count = sum(1 for x in data if x == 1)
    return float(true_count + rng.laplace(loc=0.0, scale=1.0 / epsilon))
```

PROGRAM TO ANNOTATE:
```python
import numpy as np

def private_count_safe(data, epsilon, rng):
    if not data:
        return 0
    else:
        n = len(data)
        noise = np.sqrt(2 * epsilon * n) * rng.standard_normal(n)
        return np.sum(np.clip(data + noise, 0, 1))
```


==================== ITEM P095 ====================
Task: laplace_bounded_count_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_count`.
- Input: `data`, a list of integers each equal to 0 or 1; `epsilon`, a float;
  `rng`, a numpy.random.Generator. All randomness must come from `rng`.
- Output: a float - a differentially private estimate of the number of 1s in `data`.
- Neighboring datasets differ by addition or removal of one record.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Handle empty input. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task laplace_bounded_count_v1 (correct, eps-DP)."""
import numpy as np


def private_count(data, epsilon, rng):
    """Count of records equal to 1 in a binary list, under add/remove-one adjacency.

    Sensitivity of the count is 1, so Laplace noise with scale 1/epsilon
    gives pure epsilon-DP.
    """
    true_count = sum(1 for x in data if x == 1)
    return float(true_count + rng.laplace(loc=0.0, scale=1.0 / epsilon))
```

PROGRAM TO ANNOTATE:
```python
def private_count(data, epsilon, rng):
    if not data:
        return 0
    m = len(data)
    r = rng.integers(0, m, size=1)[0]
    s = sum(data)
    z = math.sqrt(2 * epsilon ** 2 * m)
    y = math.exp(-math.log(2) * epsilon ** 2 * m)
    u = y / z
    if r == 0:
        return s + math.log(u)
    elif r == m:
        return s - math.log(u)
    else:
        return s + math.log(2 * u)
```


==================== ITEM P096 ====================
Task: empty_dataset_count_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_count_safe`.
- Input: `data`, a list of integers each 0 or 1, WHICH MAY BE EMPTY;
  `epsilon`, a float; `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a float - a differentially private count of the 1s. The empty
  dataset must be handled by the SAME noisy release path as any other
  dataset: no special-cased deterministic output, no exception.
- Neighboring datasets differ by addition or removal of one record; note that
  [] and [1] are neighbors, so the empty case is privacy-relevant.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task empty_dataset_count_v1 (correct, eps-DP)."""


def private_count_safe(data, epsilon, rng):
    """DP count of 1s, with the empty dataset on the ORDINARY code path.

    There is deliberately no `if not data:` branch here, and its absence is the
    whole content of the task.

    Under add/remove-one adjacency [] and [1] are neighbours, so the empty
    dataset is not a degenerate case to be handled specially -- it is one side
    of a pair the guarantee must cover. Any special case that returns a
    deterministic value (0, or 0.0, or an integer instead of a float) makes the
    two neighbours perfectly distinguishable: seeing exactly 0.0 tells you the
    input was empty, at any epsilon.

    Rounding only in the empty case is the same bug wearing a different hat:
    the output DOMAIN differs between neighbours, which is a support difference
    rather than a distribution difference, and no epsilon covers it.
    """
    true_count = sum(1 for x in data if x == 1)
    return float(true_count + rng.laplace(loc=0.0, scale=1.0 / epsilon))
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_count_safe(data, epsilon, rng):
    n = len(data)
    laplacian_noise = rng.laplace(0, 2 * n / epsilon)
    return (sum(data) + laplacian_noise) / (n + 1)
```


==================== ITEM P097 ====================
Task: laplace_bounded_count_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_count`.
- Input: `data`, a list of integers each equal to 0 or 1; `epsilon`, a float;
  `rng`, a numpy.random.Generator. All randomness must come from `rng`.
- Output: a float - a differentially private estimate of the number of 1s in `data`.
- Neighboring datasets differ by addition or removal of one record.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Handle empty input. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task laplace_bounded_count_v1 (correct, eps-DP)."""
import numpy as np


def private_count(data, epsilon, rng):
    """Count of records equal to 1 in a binary list, under add/remove-one adjacency.

    Sensitivity of the count is 1, so Laplace noise with scale 1/epsilon
    gives pure epsilon-DP.
    """
    true_count = sum(1 for x in data if x == 1)
    return float(true_count + rng.laplace(loc=0.0, scale=1.0 / epsilon))
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_count(data, epsilon, rng):
    n = len(data)
    if n == 0:
        return 0.0
    theta = rng.uniform(0, math.exp(epsilon))
    estimate = sum(data) + (theta - 1 / math.exp(epsilon)) / epsilon
    return max(0, min(n, estimate))
```


==================== ITEM P098 ====================
Task: bounded_mean_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=replace_one

SPEC:
Write one self-contained Python function named `private_mean`.
- Input: `data`, a NON-EMPTY list of floats in [0, 1]; `epsilon`, a float;
  `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a float - a differentially private estimate of the mean of `data`.
- Neighboring datasets have the SAME length n and differ in one record's value
  (replace-one adjacency), so the mean has sensitivity 1/n.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task bounded_mean_v1 (correct, eps-DP)."""
import numpy as np


def private_mean(data, epsilon, rng):
    """DP mean of values in [0, 1] under replace-one adjacency.

    Under replace-one the length n is fixed, so replacing a single record can
    move the mean by at most 1/n (the range of the domain divided by n).
    Laplace noise with scale 1/(n * epsilon) therefore gives pure epsilon-DP.

    The two common ways to get this wrong are to use the SUM sensitivity (1)
    instead of the mean sensitivity (1/n), and to divide by the wrong n.
    """
    n = len(data)
    if n == 0:
        # Spec guarantees non-empty; return 0.0 rather than raising so that the
        # auditor sees a total function.
        return 0.0
    true_mean = float(np.sum(np.asarray(data, dtype=float))) / n
    return float(true_mean + rng.laplace(loc=0.0, scale=1.0 / (n * epsilon)))
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_mean(data, epsilon, rng):
    n = len(data)
    sensitivity = 1 / n
    scale = sensitivity / epsilon
    noise = rng.normal(scale=scale)
    private_sum = sum(data) + noise
    private_mean = private_sum / n
    return private_mean
```


==================== ITEM P099 ====================
Task: empty_dataset_count_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_count_safe`.
- Input: `data`, a list of integers each 0 or 1, WHICH MAY BE EMPTY;
  `epsilon`, a float; `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a float - a differentially private count of the 1s. The empty
  dataset must be handled by the SAME noisy release path as any other
  dataset: no special-cased deterministic output, no exception.
- Neighboring datasets differ by addition or removal of one record; note that
  [] and [1] are neighbors, so the empty case is privacy-relevant.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task empty_dataset_count_v1 (correct, eps-DP)."""


def private_count_safe(data, epsilon, rng):
    """DP count of 1s, with the empty dataset on the ORDINARY code path.

    There is deliberately no `if not data:` branch here, and its absence is the
    whole content of the task.

    Under add/remove-one adjacency [] and [1] are neighbours, so the empty
    dataset is not a degenerate case to be handled specially -- it is one side
    of a pair the guarantee must cover. Any special case that returns a
    deterministic value (0, or 0.0, or an integer instead of a float) makes the
    two neighbours perfectly distinguishable: seeing exactly 0.0 tells you the
    input was empty, at any epsilon.

    Rounding only in the empty case is the same bug wearing a different hat:
    the output DOMAIN differs between neighbours, which is a support difference
    rather than a distribution difference, and no epsilon covers it.
    """
    true_count = sum(1 for x in data if x == 1)
    return float(true_count + rng.laplace(loc=0.0, scale=1.0 / epsilon))
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_count_safe(data, epsilon, rng):
    if len(data) == 0:
        return 0
    else:
        sensitivity = 1
        noise_level = math.sqrt((2 * math.log(1.25 / epsilon)) / len(data))
        return max(0, sum(data) + np.floor(rng.laplace(scale=noise_level) / sensitivity))
```


==================== ITEM P100 ====================
Task: subsampled_count_v1 | claimed epsilon=0.5 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_subsampled_count`.
- Input: `data`, a list of integers each 0 or 1; `epsilon`, a float;
  `rng`, a numpy.random.Generator (sole source of randomness).
- Procedure: first include each record independently with probability
  q = 0.5 (Poisson subsampling using `rng`), then release a noisy count of
  1s among the sampled records using the Laplace mechanism with a base
  budget eps0 of your choice.
- Output: a float.
- Claim: pure epsilon-DP for epsilon = 0.5 for the OVERALL procedure via
  privacy amplification by subsampling: eps = ln(1 + q*(e^{eps0} - 1)).
  Choose and state eps0 in a code comment so the claim holds.
- Neighboring datasets differ by addition or removal of one record.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
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
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_subsampled_count(data, epsilon, rng):
    q = 0.5
    eps0 = 0.4  # Chosen to ensure eps = 0.5
    sample = [x for x in data if rng.random() < q]
    count = sample.count(1)
    lap_scale = eps0 / (2 * q)
    noisy_count = count + rng.laplace(0, lap_scale)
    return noisy_count
```


==================== ITEM P101 ====================
Task: empty_dataset_count_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_count_safe`.
- Input: `data`, a list of integers each 0 or 1, WHICH MAY BE EMPTY;
  `epsilon`, a float; `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a float - a differentially private count of the 1s. The empty
  dataset must be handled by the SAME noisy release path as any other
  dataset: no special-cased deterministic output, no exception.
- Neighboring datasets differ by addition or removal of one record; note that
  [] and [1] are neighbors, so the empty case is privacy-relevant.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task empty_dataset_count_v1 (correct, eps-DP)."""


def private_count_safe(data, epsilon, rng):
    """DP count of 1s, with the empty dataset on the ORDINARY code path.

    There is deliberately no `if not data:` branch here, and its absence is the
    whole content of the task.

    Under add/remove-one adjacency [] and [1] are neighbours, so the empty
    dataset is not a degenerate case to be handled specially -- it is one side
    of a pair the guarantee must cover. Any special case that returns a
    deterministic value (0, or 0.0, or an integer instead of a float) makes the
    two neighbours perfectly distinguishable: seeing exactly 0.0 tells you the
    input was empty, at any epsilon.

    Rounding only in the empty case is the same bug wearing a different hat:
    the output DOMAIN differs between neighbours, which is a support difference
    rather than a distribution difference, and no epsilon covers it.
    """
    true_count = sum(1 for x in data if x == 1)
    return float(true_count + rng.laplace(loc=0.0, scale=1.0 / epsilon))
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_count_safe(data, epsilon, rng):
    n = len(data)
    noise = rng.normal(loc=0, scale=math.sqrt(2 * n * epsilon))
    return (np.sum(data) + noise) / (1 + epsilon)
```


==================== ITEM P102 ====================
Task: repeated_release_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_repeated_count`.
- Input: `data`, a list of integers each 0 or 1; `epsilon`, a float;
  `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a list of FOUR floats - four independent noisy releases of the
  count of 1s (fresh noise each time), e.g. for four different analysts.
- Neighboring datasets differ by addition or removal of one record.
- Claim: the TOTAL release of all four numbers is pure epsilon-DP for
  epsilon = 1.0 under sequential composition.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task repeated_release_v1 (correct, eps-DP)."""
import numpy as np

K = 4          # number of independent releases required by the spec


def private_repeated_count(data, epsilon, rng):
    """Four independent noisy counts whose TOTAL release is epsilon-DP.

    Four releases of the same query on the same data compose sequentially, so
    each release gets eps/4 and Laplace scale 4/eps. Sensitivity of the count
    is 1 under add/remove-one.

    Two failure modes this guards against:
      - spending the full epsilon per release (a 4x overspend, K*eps-DP), and
      - drawing the noise ONCE and returning four copies, which looks noisy but
        collapses to a single release and lets an attacker average nothing away
        while the spec promises four independent answers.
    """
    eps_each = epsilon / K
    scale = 1.0 / eps_each
    true_count = sum(1 for x in data if x == 1)
    return [float(true_count + rng.laplace(loc=0.0, scale=scale)) for _ in range(K)]
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_repeated_count(data, epsilon, rng):
    n = len(data)
    count = sum(data)
    sensitivity = 1.0
    
    delta = 0.01
    p = (math.exp(epsilon / 2) + 1) / (math.exp(epsilon / 2) + math.exp(-epsilon / 2))
    q = (math.exp(epsilon / 2) - 1) / (math.exp(epsilon / 2) + math.exp(-epsilon / 2))
    
    noisy_counts = []
    for _ in range(4):
        if rng.random() < p:
            noisy_count = rng.geometric(q) + count
        else:
            noisy_count = rng.geometric(p) + (n - count)
        noisy_counts.append(noisy_count)
    
    return noisy_counts
```


==================== ITEM P103 ====================
Task: laplace_bounded_count_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_count`.
- Input: `data`, a list of integers each equal to 0 or 1; `epsilon`, a float;
  `rng`, a numpy.random.Generator. All randomness must come from `rng`.
- Output: a float - a differentially private estimate of the number of 1s in `data`.
- Neighboring datasets differ by addition or removal of one record.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Handle empty input. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task laplace_bounded_count_v1 (correct, eps-DP)."""
import numpy as np


def private_count(data, epsilon, rng):
    """Count of records equal to 1 in a binary list, under add/remove-one adjacency.

    Sensitivity of the count is 1, so Laplace noise with scale 1/epsilon
    gives pure epsilon-DP.
    """
    true_count = sum(1 for x in data if x == 1)
    return float(true_count + rng.laplace(loc=0.0, scale=1.0 / epsilon))
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_count(data, epsilon, rng):
    if len(data) == 0:
        return 0
    else:
        n = len(data)
        count = np.sum(data)
        delta = count / n
        noise = np.sqrt(2 * math.log(1/epsilon)) * delta
        return max(0, count - noise + np.sqrt(2 * math.log(1/epsilon)) * delta - noise)
```


==================== ITEM P104 ====================
Task: randomized_response_v1 | claimed epsilon=1.0986122886681098 delta=0.0 | adjacency=replace_one

SPEC:
Write one self-contained Python function named `randomized_response`.
- Input: `data`, a list containing exactly ONE integer bit (0 or 1);
  `epsilon`, a float; `rng`, a numpy.random.Generator (sole source of randomness).
- Output: an int (0 or 1) - a randomized-response release of the bit:
  report the true bit with probability e^epsilon / (1 + e^epsilon),
  otherwise report the flipped bit.
- Neighboring datasets: the single record's value is replaced.
- Claim: pure epsilon-DP for epsilon = ln(3), i.e. truthful with probability 3/4.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task randomized_response_v1 (correct, eps-DP local)."""
import math


def randomized_response(data, epsilon, rng):
    """Randomized response on a single bit, pure epsilon-DP in the local model.

    Report the true bit with probability p = e^eps / (1 + e^eps), else the flip.
    The likelihood ratio between the two possible inputs is p / (1 - p) = e^eps
    for either output, which is exactly the epsilon-DP condition.

    At eps = ln(3), p = 3/4.
    """
    bit = int(data[0])
    p_truth = math.exp(epsilon) / (1.0 + math.exp(epsilon))
    if rng.random() < p_truth:
        return int(bit)
    return int(1 - bit)
```

PROGRAM TO ANNOTATE:
```python
def randomized_response(data, epsilon, rng):
    true_bit = data[0]
    flipped_bit = 1 - true_bit
    p = math.exp(epsilon) / (1 + math.exp(epsilon))
    return 1 if rng.uniform(0, 1) < p else flipped_bit
```


==================== ITEM P105 ====================
Task: repeated_release_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_repeated_count`.
- Input: `data`, a list of integers each 0 or 1; `epsilon`, a float;
  `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a list of FOUR floats - four independent noisy releases of the
  count of 1s (fresh noise each time), e.g. for four different analysts.
- Neighboring datasets differ by addition or removal of one record.
- Claim: the TOTAL release of all four numbers is pure epsilon-DP for
  epsilon = 1.0 under sequential composition.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task repeated_release_v1 (correct, eps-DP)."""
import numpy as np

K = 4          # number of independent releases required by the spec


def private_repeated_count(data, epsilon, rng):
    """Four independent noisy counts whose TOTAL release is epsilon-DP.

    Four releases of the same query on the same data compose sequentially, so
    each release gets eps/4 and Laplace scale 4/eps. Sensitivity of the count
    is 1 under add/remove-one.

    Two failure modes this guards against:
      - spending the full epsilon per release (a 4x overspend, K*eps-DP), and
      - drawing the noise ONCE and returning four copies, which looks noisy but
        collapses to a single release and lets an attacker average nothing away
        while the spec promises four independent answers.
    """
    eps_each = epsilon / K
    scale = 1.0 / eps_each
    true_count = sum(1 for x in data if x == 1)
    return [float(true_count + rng.laplace(loc=0.0, scale=scale)) for _ in range(K)]
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_repeated_count(data, epsilon, rng):
    count_ones = data.count(1)
    sensitivity = 1
    scale = sensitivity / epsilon
    noise_counts = [count_ones + rng.normal(0, scale) for _ in range(4)]
    return noise_counts
```


==================== ITEM P106 ====================
Task: clipped_bounded_sum_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_sum`.
- Input: `data`, a list of non-negative floats which MAY be larger than 10.0;
  `epsilon`, a float; `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a float - a differentially private estimate of the sum of the values,
  where each value must first be clipped to the range [0, 10].
- Neighboring datasets differ by addition or removal of one record.
- Claim: pure epsilon-DP for epsilon = 1.0 with respect to the clipped sum.
- Use numpy and math only. Handle empty input. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task clipped_bounded_sum_v1 (correct, eps-DP).

Input values are arbitrary non-negative floats and MAY exceed the bound B;
the spec requires clipping each value to [0, B] before summing, giving
add/remove sensitivity B.
"""
import numpy as np

B = 10.0


def private_sum(data, epsilon, rng):
    clipped = [min(max(float(x), 0.0), B) for x in data]
    return float(sum(clipped) + rng.laplace(loc=0.0, scale=B / epsilon))
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_sum(data, epsilon, rng):
    if not data:
        return 0.0
    
    clipped_data = [min(val, 10.0) for val in data]
    sum_clipped_data = sum(clipped_data)
    
    sensitivity = 10.0
    scale = sensitivity / epsilon
    
    noise = rng.laplace(0, scale)
    
    private_sum = sum_clipped_data + noise
    
    return private_sum
```


==================== ITEM P107 ====================
Task: laplace_bounded_count_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_count`.
- Input: `data`, a list of integers each equal to 0 or 1; `epsilon`, a float;
  `rng`, a numpy.random.Generator. All randomness must come from `rng`.
- Output: a float - a differentially private estimate of the number of 1s in `data`.
- Neighboring datasets differ by addition or removal of one record.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Handle empty input. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task laplace_bounded_count_v1 (correct, eps-DP)."""
import numpy as np


def private_count(data, epsilon, rng):
    """Count of records equal to 1 in a binary list, under add/remove-one adjacency.

    Sensitivity of the count is 1, so Laplace noise with scale 1/epsilon
    gives pure epsilon-DP.
    """
    true_count = sum(1 for x in data if x == 1)
    return float(true_count + rng.laplace(loc=0.0, scale=1.0 / epsilon))
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_count(data, epsilon, rng):
    if len(data) == 0:
        return 0

    num_ones = sum(data)
    num_zeroes = len(data) - num_ones
    total_count = num_ones + num_zeroes

    laplace_mechanism = (num_ones + num_zeroes * rng.laplace(scale=1/epsilon)) / total_count

    return laplace_mechanism
```


==================== ITEM P108 ====================
Task: subsampled_count_v1 | claimed epsilon=0.5 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_subsampled_count`.
- Input: `data`, a list of integers each 0 or 1; `epsilon`, a float;
  `rng`, a numpy.random.Generator (sole source of randomness).
- Procedure: first include each record independently with probability
  q = 0.5 (Poisson subsampling using `rng`), then release a noisy count of
  1s among the sampled records using the Laplace mechanism with a base
  budget eps0 of your choice.
- Output: a float.
- Claim: pure epsilon-DP for epsilon = 0.5 for the OVERALL procedure via
  privacy amplification by subsampling: eps = ln(1 + q*(e^{eps0} - 1)).
  Choose and state eps0 in a code comment so the claim holds.
- Neighboring datasets differ by addition or removal of one record.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
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
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_subsampled_count(data, epsilon, rng):
    q = 0.5
    eps0 = 1.2  # Choice to ensure eps = 0.5
    n = len(data)
    sampled_count = 0

    for x in data:
        if rng.binomial(1, q) == 1:
            sampled_count += x

    noisy_count = (sampled_count + rng.laplace(0, eps0 / n)) * (n / len(data))
    return noisy_count
```


==================== ITEM P109 ====================
Task: bounded_mean_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=replace_one

SPEC:
Write one self-contained Python function named `private_mean`.
- Input: `data`, a NON-EMPTY list of floats in [0, 1]; `epsilon`, a float;
  `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a float - a differentially private estimate of the mean of `data`.
- Neighboring datasets have the SAME length n and differ in one record's value
  (replace-one adjacency), so the mean has sensitivity 1/n.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task bounded_mean_v1 (correct, eps-DP)."""
import numpy as np


def private_mean(data, epsilon, rng):
    """DP mean of values in [0, 1] under replace-one adjacency.

    Under replace-one the length n is fixed, so replacing a single record can
    move the mean by at most 1/n (the range of the domain divided by n).
    Laplace noise with scale 1/(n * epsilon) therefore gives pure epsilon-DP.

    The two common ways to get this wrong are to use the SUM sensitivity (1)
    instead of the mean sensitivity (1/n), and to divide by the wrong n.
    """
    n = len(data)
    if n == 0:
        # Spec guarantees non-empty; return 0.0 rather than raising so that the
        # auditor sees a total function.
        return 0.0
    true_mean = float(np.sum(np.asarray(data, dtype=float))) / n
    return float(true_mean + rng.laplace(loc=0.0, scale=1.0 / (n * epsilon)))
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_mean(data, epsilon, rng):
    n = len(data)
    sensitivity = 1 / n
    scale = sensitivity / epsilon
    noise = rng.normal(0, scale, size=1)[0]
    estimate = np.mean(data) + noise
    return estimate
```


==================== ITEM P110 ====================
Task: categorical_histogram_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_histogram`.
- Input: `data`, a list of category labels, each one of "A", "B", or "C"
  (other labels may appear and must be ignored without error);
  `epsilon`, a float; `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a dict with keys "A", "B", "C" mapping to float noisy counts.
  Every bin must receive noise, including bins with zero count.
- Neighboring datasets differ by addition or removal of one record.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Handle empty input. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task categorical_histogram_v1 (correct, eps-DP).

Fixed public category set {A, B, C}; add/remove-one adjacency changes exactly
one bin by 1, so the histogram has L1 sensitivity 1 and per-bin Laplace noise
with scale 1/epsilon gives pure epsilon-DP. Noise must be added to EVERY bin,
including empty ones (a common bug is leaving zero bins deterministic).
Invalid labels are ignored (counted toward no bin) rather than raising,
so control flow does not depend on individual records.
"""
import numpy as np

CATEGORIES = ("A", "B", "C")


def private_histogram(data, epsilon, rng):
    counts = {c: 0 for c in CATEGORIES}
    for x in data:
        if x in counts:
            counts[x] += 1
    return {c: float(counts[c] + rng.laplace(loc=0.0, scale=1.0 / epsilon))
            for c in CATEGORIES}
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_histogram(data, epsilon, rng):
    # Define the possible categories
    categories = ["A", "B", "C"]

    # Create a dictionary to store the counts
    counts = {category: 0 for category in categories}

    # Count the occurrences of each category
    for category in data:
        if category in counts:
            counts[category] += 1

    # Calculate the total number of occurrences
    total = sum(counts.values())

    # If total is zero, return the counts directly
    if total == 0:
        return counts

    # Generate Laplace noise
    noise = {category: rng.laplace(loc=0, scale=1.0/epsilon) for category in categories}

    # Add noise to the counts
    for category in counts:
        counts[category] = max(0, counts[category] + noise[category])

    # Normalize the counts to sum up to the total
    total_noise = sum(counts.values())
    if total_noise > 0:
        for category in counts:
            counts[category] = counts[category] * total / total_noise

    return counts
```


==================== ITEM P111 ====================
Task: adaptive_two_step_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_adaptive_release`.
- Input: `data`, a list of floats in [0, 1]; `epsilon`, a float;
  `rng`, a numpy.random.Generator (sole source of randomness).
- Step 1: release a noisy count of records with value >= 0.5.
- Step 2: IF the noisy count from step 1 is >= 3, additionally release a noisy
  sum of the values (sensitivity 1); otherwise release a noisy count of
  records with value < 0.5. Fresh noise in step 2.
- Output: a tuple (step1_release, step2_release) of two floats.
- The second query DEPENDS on the first noisy output (adaptive composition);
  the TOTAL release must be pure epsilon-DP for epsilon = 1.0.
- Neighboring datasets differ by addition or removal of one record.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task adaptive_two_step_v1 (correct, eps-DP)."""
import numpy as np


def private_adaptive_release(data, epsilon, rng):
    """Two-step adaptive release: the second query depends on the first OUTPUT.

    Budget: eps/2 per step, so scale 2/eps for both. Each query has
    sensitivity 1 (a count under add/remove-one; the sum of values in [0,1]
    likewise changes by at most 1).

    The property that makes this correct is subtle and is the whole point of
    the task: the branch is taken on the NOISY step-1 release, never on the
    raw count. Branching on the noisy value is post-processing -- the analyst
    already holds it -- so adaptive (sequential) composition applies and the
    total is eps-DP. Branching on the RAW count would make the CHOICE of the
    second query itself a data-dependent release that no noise pays for, and
    the identity of the branch leaks.

    Both branches release a sensitivity-1 quantity at the same scale, so the
    accounting does not depend on which branch is taken.
    """
    eps_step = epsilon / 2.0
    scale = 1.0 / eps_step                       # = 2/epsilon

    high = sum(1 for x in data if x >= 0.5)
    step1 = float(high + rng.laplace(loc=0.0, scale=scale))

    if step1 >= 3.0:                             # branch on the NOISY value
        total = float(np.sum(np.asarray(data, dtype=float))) if data else 0.0
        step2 = float(total + rng.laplace(loc=0.0, scale=scale))
    else:
        low = sum(1 for x in data if x < 0.5)
        step2 = float(low + rng.laplace(loc=0.0, scale=scale))

    return (step1, step2)
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_adaptive_release(data, epsilon, rng):
    n = len(data)
    step1_noise = rng.laplace(0, 1 / n)
    step1_count = sum(1 for x in data if x >= 0.5) + step1_noise
    step1_release = max(0, step1_count)

    if step1_release >= 3:
        step2_noise = rng.laplace(0, 1 / n)
        step2_sum = sum(x for x in data if x >= 0.5) + step2_noise
        step2_release = max(0, step2_sum)
    else:
        step2_noise = rng.laplace(0, 1 / n)
        step2_count = sum(1 for x in data if x < 0.5) + step2_noise
        step2_release = max(0, step2_count)

    return (step1_release, step2_release)
```


==================== ITEM P112 ====================
Task: empty_dataset_count_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_count_safe`.
- Input: `data`, a list of integers each 0 or 1, WHICH MAY BE EMPTY;
  `epsilon`, a float; `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a float - a differentially private count of the 1s. The empty
  dataset must be handled by the SAME noisy release path as any other
  dataset: no special-cased deterministic output, no exception.
- Neighboring datasets differ by addition or removal of one record; note that
  [] and [1] are neighbors, so the empty case is privacy-relevant.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task empty_dataset_count_v1 (correct, eps-DP)."""


def private_count_safe(data, epsilon, rng):
    """DP count of 1s, with the empty dataset on the ORDINARY code path.

    There is deliberately no `if not data:` branch here, and its absence is the
    whole content of the task.

    Under add/remove-one adjacency [] and [1] are neighbours, so the empty
    dataset is not a degenerate case to be handled specially -- it is one side
    of a pair the guarantee must cover. Any special case that returns a
    deterministic value (0, or 0.0, or an integer instead of a float) makes the
    two neighbours perfectly distinguishable: seeing exactly 0.0 tells you the
    input was empty, at any epsilon.

    Rounding only in the empty case is the same bug wearing a different hat:
    the output DOMAIN differs between neighbours, which is a support difference
    rather than a distribution difference, and no epsilon covers it.
    """
    true_count = sum(1 for x in data if x == 1)
    return float(true_count + rng.laplace(loc=0.0, scale=1.0 / epsilon))
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_count_safe(data, epsilon, rng):
    n = len(data)
    count = sum(data)
    z = rng.normal(0, math.sqrt(2 * n * math.log(1.3 / epsilon / epsilon)))
    return count + z
```


==================== ITEM P113 ====================
Task: bounded_mean_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=replace_one

SPEC:
Write one self-contained Python function named `private_mean`.
- Input: `data`, a NON-EMPTY list of floats in [0, 1]; `epsilon`, a float;
  `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a float - a differentially private estimate of the mean of `data`.
- Neighboring datasets have the SAME length n and differ in one record's value
  (replace-one adjacency), so the mean has sensitivity 1/n.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task bounded_mean_v1 (correct, eps-DP)."""
import numpy as np


def private_mean(data, epsilon, rng):
    """DP mean of values in [0, 1] under replace-one adjacency.

    Under replace-one the length n is fixed, so replacing a single record can
    move the mean by at most 1/n (the range of the domain divided by n).
    Laplace noise with scale 1/(n * epsilon) therefore gives pure epsilon-DP.

    The two common ways to get this wrong are to use the SUM sensitivity (1)
    instead of the mean sensitivity (1/n), and to divide by the wrong n.
    """
    n = len(data)
    if n == 0:
        # Spec guarantees non-empty; return 0.0 rather than raising so that the
        # auditor sees a total function.
        return 0.0
    true_mean = float(np.sum(np.asarray(data, dtype=float))) / n
    return float(true_mean + rng.laplace(loc=0.0, scale=1.0 / (n * epsilon)))
```

PROGRAM TO ANNOTATE:
```python
import numpy as np

def private_mean(data, epsilon, rng):
    """
    Differentially private estimate of the mean.
    """
    n = len(data)
    data_bound = (2 * np.array(data) - 1) / epsilon
    data_noised = rng.laplace(scale=1.0/epsilon, size=n) + data_bound
    return np.mean(data_noised)
```


==================== ITEM P114 ====================
Task: clipped_bounded_sum_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_sum`.
- Input: `data`, a list of non-negative floats which MAY be larger than 10.0;
  `epsilon`, a float; `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a float - a differentially private estimate of the sum of the values,
  where each value must first be clipped to the range [0, 10].
- Neighboring datasets differ by addition or removal of one record.
- Claim: pure epsilon-DP for epsilon = 1.0 with respect to the clipped sum.
- Use numpy and math only. Handle empty input. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task clipped_bounded_sum_v1 (correct, eps-DP).

Input values are arbitrary non-negative floats and MAY exceed the bound B;
the spec requires clipping each value to [0, B] before summing, giving
add/remove sensitivity B.
"""
import numpy as np

B = 10.0


def private_sum(data, epsilon, rng):
    clipped = [min(max(float(x), 0.0), B) for x in data]
    return float(sum(clipped) + rng.laplace(loc=0.0, scale=B / epsilon))
```

PROGRAM TO ANNOTATE:
```python
def private_sum(data, epsilon, rng):
    if not data:
        return 0.0
    clipped_data = [min(10.0, x) for x in data]
    diff_privacy = np.abs(sum(clipped_data) - np.sum(clipped_data))
    sensitivity = np.max(clipped_data) - np.min(clipped_data)
    alpha = diff_privacy / (epsilon * sensitivity)
    if alpha > 1.0:
        return np.sum(clipped_data)
    else:
        return np.sum(clipped_data) + rng.laplace(0.0, alpha)
```


==================== ITEM P115 ====================
Task: private_topk_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_top2`.
- Input: `data`, a list of category labels from {"A","B","C","D"};
  `epsilon`, a float; `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a list of TWO DISTINCT strings - the two most frequent categories,
  selected privately by running Report Noisy Max twice ("peeling"): select a
  winner, remove it from the candidate set, select again. Split the budget
  correctly across the two rounds.
- Neighboring datasets differ by addition or removal of one record.
- Claim: the total release is pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task private_topk_v1 (correct, eps-DP)."""
import numpy as np

CATEGORIES = ("A", "B", "C", "D")


def private_top2(data, epsilon, rng):
    """Top-2 selection by PEELING: Report Noisy Max twice on a shrinking set.

    Two adaptive selections on the same data compose sequentially, so each
    round gets eps/2. Each round is a Report Noisy Max over the remaining
    candidates with Laplace(2*Delta/eps_round) noise, matching the convention
    in report_noisy_max_v1 (Delta = 1 for counts under add/remove-one).

        eps_round = eps / 2   ->   scale = 2 / (eps/2) = 4 / eps

    Three properties this gets right and the buggy variants do not:

    1. FRESH noise in round 2. Reusing round 1's draws makes the second
       selection a deterministic function of the first, which both breaks the
       composition argument and correlates the two outputs.
    2. The winner is REMOVED from the candidate set, not merely down-weighted,
       so the two outputs are always distinct as the spec requires.
    3. Only the labels are released -- never the noisy counts. Releasing the
       noisy scores would be a strictly larger release than the analysis covers.
    """
    eps_round = epsilon / 2.0
    scale = 2.0 / eps_round                     # = 4/epsilon

    remaining = list(CATEGORIES)
    winners = []
    for _ in range(2):
        counts = [sum(1 for x in data if x == c) for c in remaining]
        noisy = [c + rng.laplace(loc=0.0, scale=scale) for c in counts]
        idx = int(np.argmax(noisy))
        winners.append(str(remaining[idx]))
        remaining.pop(idx)                      # peel: never selectable again
    return winners
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_top2(data, epsilon, rng):
    categories, counts = np.unique(data, return_counts=True)
    noisy_counts = [math.floor(c + rng.laplace(scale=1/epsilon)) for c in counts]
    top2_categories = categories[np.argsort(-np.array(noisy_counts))[:2]]

    return list(top2_categories)
```


==================== ITEM P116 ====================
Task: empty_dataset_count_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_count_safe`.
- Input: `data`, a list of integers each 0 or 1, WHICH MAY BE EMPTY;
  `epsilon`, a float; `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a float - a differentially private count of the 1s. The empty
  dataset must be handled by the SAME noisy release path as any other
  dataset: no special-cased deterministic output, no exception.
- Neighboring datasets differ by addition or removal of one record; note that
  [] and [1] are neighbors, so the empty case is privacy-relevant.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task empty_dataset_count_v1 (correct, eps-DP)."""


def private_count_safe(data, epsilon, rng):
    """DP count of 1s, with the empty dataset on the ORDINARY code path.

    There is deliberately no `if not data:` branch here, and its absence is the
    whole content of the task.

    Under add/remove-one adjacency [] and [1] are neighbours, so the empty
    dataset is not a degenerate case to be handled specially -- it is one side
    of a pair the guarantee must cover. Any special case that returns a
    deterministic value (0, or 0.0, or an integer instead of a float) makes the
    two neighbours perfectly distinguishable: seeing exactly 0.0 tells you the
    input was empty, at any epsilon.

    Rounding only in the empty case is the same bug wearing a different hat:
    the output DOMAIN differs between neighbours, which is a support difference
    rather than a distribution difference, and no epsilon covers it.
    """
    true_count = sum(1 for x in data if x == 1)
    return float(true_count + rng.laplace(loc=0.0, scale=1.0 / epsilon))
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_count_safe(data, epsilon, rng):
    if len(data) == 0:
        return 0.0
    else:
        noise = rng.normal(0.0, math.sqrt(epsilon), size=len(data))
        return sum(data) + noise
```


==================== ITEM P117 ====================
Task: two_queries_split_budget_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_count_and_sum`.
- Input: `data`, a list of floats in [0, 1]; `epsilon`, a float;
  `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a tuple (noisy_count, noisy_sum) of two floats, where noisy_count
  estimates len(data) and noisy_sum estimates sum(data) (values already in [0,1],
  so the sum has sensitivity 1 under add/remove).
- Neighboring datasets differ by addition or removal of one record.
- Claim: the TOTAL release (both numbers together) is pure epsilon-DP for
  epsilon = 1.0. Choose the per-query budgets accordingly.
- Use numpy and math only. Handle empty input. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task two_queries_split_budget_v1 (correct, eps-DP)."""
import numpy as np


def private_count_and_sum(data, epsilon, rng):
    """Release a noisy count AND a noisy sum under a single epsilon budget.

    Two queries are answered on the same data, so by SEQUENTIAL COMPOSITION the
    privacy costs add. Splitting the budget evenly gives each query eps/2, and
    each has sensitivity 1 under add/remove-one (adding a record changes the
    count by 1, and changes the sum by at most 1 because values lie in [0, 1]).

    Laplace scale is therefore 1/(eps/2) = 2/eps for BOTH releases.

    The two classic errors are (a) spending the full epsilon on each query,
    which yields 2*eps-DP overall, and (b) splitting the SENSITIVITY rather
    than the budget, which changes nothing about composition.
    """
    eps_each = epsilon / 2.0
    scale = 1.0 / eps_each

    noisy_count = float(len(data) + rng.laplace(loc=0.0, scale=scale))
    noisy_sum = float(float(np.sum(np.asarray(data, dtype=float))) if data else 0.0)
    noisy_sum = float(noisy_sum + rng.laplace(loc=0.0, scale=scale))
    return (noisy_count, noisy_sum)
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_count_and_sum(data, epsilon, rng):
    if len(data) == 0:
        return 0.0, 0.0

    count = len(data)
    sum_data = sum(data)
    sensitivity = 1.0

    noise_factor = math.exp(epsilon / (2 * sensitivity))
    noisy_count = count + rng.laplace(0, 1/noise_factor)
    noisy_sum = sum_data + rng.laplace(0, 1/noise_factor)

    return noisy_count, noisy_sum
```


==================== ITEM P118 ====================
Task: categorical_histogram_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_histogram`.
- Input: `data`, a list of category labels, each one of "A", "B", or "C"
  (other labels may appear and must be ignored without error);
  `epsilon`, a float; `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a dict with keys "A", "B", "C" mapping to float noisy counts.
  Every bin must receive noise, including bins with zero count.
- Neighboring datasets differ by addition or removal of one record.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Handle empty input. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task categorical_histogram_v1 (correct, eps-DP).

Fixed public category set {A, B, C}; add/remove-one adjacency changes exactly
one bin by 1, so the histogram has L1 sensitivity 1 and per-bin Laplace noise
with scale 1/epsilon gives pure epsilon-DP. Noise must be added to EVERY bin,
including empty ones (a common bug is leaving zero bins deterministic).
Invalid labels are ignored (counted toward no bin) rather than raising,
so control flow does not depend on individual records.
"""
import numpy as np

CATEGORIES = ("A", "B", "C")


def private_histogram(data, epsilon, rng):
    counts = {c: 0 for c in CATEGORIES}
    for x in data:
        if x in counts:
            counts[x] += 1
    return {c: float(counts[c] + rng.laplace(loc=0.0, scale=1.0 / epsilon))
            for c in CATEGORIES}
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_histogram(data, epsilon, rng):
    counts = dict((k, 0) for k in "ABC")
    if not data:
        return counts
    for d in data:
        if d in counts:
            counts[d] += 1
            
    total = len(data)
    for key, count in counts.items():
        if total == 0:
            break
        noisy_count = count + rng.laplace(0, 1.0/epsilon)
        noisy_count = max(0, noisy_count)
        counts[key] = noisy_count
    return counts
```


==================== ITEM P119 ====================
Task: laplace_bounded_count_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_count`.
- Input: `data`, a list of integers each equal to 0 or 1; `epsilon`, a float;
  `rng`, a numpy.random.Generator. All randomness must come from `rng`.
- Output: a float - a differentially private estimate of the number of 1s in `data`.
- Neighboring datasets differ by addition or removal of one record.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Handle empty input. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task laplace_bounded_count_v1 (correct, eps-DP)."""
import numpy as np


def private_count(data, epsilon, rng):
    """Count of records equal to 1 in a binary list, under add/remove-one adjacency.

    Sensitivity of the count is 1, so Laplace noise with scale 1/epsilon
    gives pure epsilon-DP.
    """
    true_count = sum(1 for x in data if x == 1)
    return float(true_count + rng.laplace(loc=0.0, scale=1.0 / epsilon))
```

PROGRAM TO ANNOTATE:
```python
from numpy import random
from math import floor


def private_count(data, epsilon, rng):
    if not data:
        return 0.0
    noise = rng.normal(size=len(data)) * (epsilon / len(data))
    return floor(sum(data) + sum(noise)) / len(data)
```


==================== ITEM P120 ====================
Task: laplace_bounded_count_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_count`.
- Input: `data`, a list of integers each equal to 0 or 1; `epsilon`, a float;
  `rng`, a numpy.random.Generator. All randomness must come from `rng`.
- Output: a float - a differentially private estimate of the number of 1s in `data`.
- Neighboring datasets differ by addition or removal of one record.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Handle empty input. Return code only, no explanation.

REFERENCE (correct):
```python
"""Reference implementation: task laplace_bounded_count_v1 (correct, eps-DP)."""
import numpy as np


def private_count(data, epsilon, rng):
    """Count of records equal to 1 in a binary list, under add/remove-one adjacency.

    Sensitivity of the count is 1, so Laplace noise with scale 1/epsilon
    gives pure epsilon-DP.
    """
    true_count = sum(1 for x in data if x == 1)
    return float(true_count + rng.laplace(loc=0.0, scale=1.0 / epsilon))
```

PROGRAM TO ANNOTATE:
```python
import numpy as np
import math

def private_count(data, epsilon, rng):
    n = len(data)
    if n == 0:
        return 0.0
    count = sum(data)
    scale = math.exp(epsilon)
    noise = rng.normal(0, scale, 1)
    return (count + noise) / (n + 2)
```
