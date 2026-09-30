

==================== ITEM P121 ====================
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
        num_ones = np.count_nonzero(data)
        num_zeros = len(data) - num_ones
        sensitivity = 1
        ln_r = np.log(rng.random())
        if num_zeros == 0:
            return num_ones
        elif num_ones == 0:
            return 0
        else:
            return (num_ones - 0.5) + sensitivity * math.sqrt(2*math.log(1/epsilon)) * (ln_r if rng.random() < 0.5 else -ln_r)
```


==================== ITEM P122 ====================
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
    n = len(data)
    if n == 0:
        return 0.0
    
    clipped_data = [min(10.0, x) for x in data]
    sum_clipped_data = sum(clipped_data)
    
    laplace_scale = 2 * sum_clipped_data / epsilon
    noise = rng.laplace(0, laplace_scale, n)
    
    return sum_clipped_data + np.sum(noise)
```


==================== ITEM P123 ====================
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
    if len(set(data)) < 2:
        return [max(set(data), key=data.count)]*2
    counts = dict()
    for item in data:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1
    n = len(data)
    beta_distribution_parameter = (n * math.exp(epsilon) / 6) ** 2
    noisy_counts = dict()
    for item in counts:
        noisy_counts[item] = counts[item] + rng.laplace(0, 1 / beta_distribution_parameter)
    top2 = sorted(noisy_counts, key=noisy_counts.get, reverse=True)[:2]
    return top2
```


==================== ITEM P124 ====================
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
import math

def private_valid_count(data, epsilon, rng):
    # Initialize count of valid integers
    valid_count = 0
    
    # Iterate through each record in data
    for record in data:
        # Check if record is a valid integer in range [0, 100]
        if type(record) == int and 0 <= record <= 100:
            valid_count += 1
    
    # Calculate the Laplace noise
    noise = math.exp(epsilon * np.log(np.random.uniform(0, 1)))
    
    # Apply differential privacy
    priv_count = (1 - noise) * valid_count
    
    return priv_count
```


==================== ITEM P125 ====================
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
    threshold = 4
    sensitivity = 1

    # Laplace noise for threshold
    noisy_threshold = threshold + np.log(1 + epsilon) / epsilon * np.sign(rng.normal(0, 1))

    # Count of 1s, 0s, and length of data
    q1 = np.sum(data)
    q2 = len(data) - q1
    q3 = len(data)

    # Noisy count of 1s, 0s, and length of data
    noisy_q1 = q1 + np.log(1 + epsilon / (2 * len(data))) / (epsilon / (2 * len(data))) * np.sign(rng.normal(0, 1))
    noisy_q2 = q2 + np.log(1 + epsilon / (2 * len(data))) / (epsilon / (2 * len(data))) * np.sign(rng.normal(0, 1))
    noisy_q3 = len(data) + np.log(1 + epsilon / (2 * len(data))) / (epsilon / (2 * len(data))) * np.sign(rng.normal(0, 1))

    # Check which noisy count exceeds the threshold
    if noisy_q1 > noisy_threshold:
        return 1
    elif noisy_q2 > noisy_threshold:
        return 2
    elif noisy_q3 > noisy_threshold:
        return 3
    else:
        return 0
```


==================== ITEM P126 ====================
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
        return (0, 0)

    delta = 1.0 / (2 * len(data))
    sensitivity = 2.0

    noisy_count = np.floor(len(data) / np.exp(epsilon))
    noisy_sum = np.floor(sum(data) / np.exp(epsilon / 2))

    noisy_count = max(0, noisy_count - rng.laplace(loc=0, scale=sensitivity / (2 * delta)))
    noisy_sum = max(0, noisy_sum - rng.laplace(loc=0, scale=sensitivity / (2 * delta)))

    return (noisy_count, noisy_sum)
```


==================== ITEM P127 ====================
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
    scaled_data = [x * n for x in data]
    scaled_mean = np.mean(scaled_data)
    laplace_noise = rng.laplace(0, 1 / (epsilon * n))
    private_mean = scaled_mean + laplace_noise
    return private_mean / n
```


==================== ITEM P128 ====================
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
    base_noise = rng.normal(0, math.sqrt(2 * n * math.log(2 / epsilon)))
    noisy_counts = [count + base_noise + rng.normal(0, math.sqrt(2 * n * math.log(2 / epsilon))) for _ in range(4)]
    return noisy_counts
```


==================== ITEM P129 ====================
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
    if not data:
        return (0.0, 0.0)
    
    count = len(data)
    sum_data = sum(data)
    
    # Calculate the per-query budgets
    delta = 1e-10
    alpha = (math.sqrt(2) + 2 - math.sqrt(2 * epsilon)) / (2 * epsilon)
    beta = alpha * math.exp(epsilon) - 1
    
    # Add noise to count
    noisy_count = count + rng.normal(0, alpha)
    
    # Add noise to sum
    noisy_sum = sum_data + rng.normal(0, beta)
    
    # Ensure noisy_sum is in [0, 1]
    noisy_sum = max(0, min(noisy_sum, 1))
    
    return (noisy_count, noisy_sum)
```


==================== ITEM P130 ====================
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
    threshold = 4
    q1 = sum(data)
    q2 = n - q1
    q3 = n
    
    scale = (epsilon / (3 * (threshold + 1)))
    laplace_noise = rng.laplace(0, scale, 3)
    
    noisy_q1 = q1 + laplace_noise[0]
    noisy_q2 = q2 + laplace_noise[1]
    noisy_q3 = q3 + laplace_noise[2]
    
    if noisy_q1 > threshold:
        return 1
    if noisy_q2 > threshold:
        return 2
    if noisy_q3 > threshold:
        return 3
    
    return 0
```


==================== ITEM P131 ====================
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
import math

def private_valid_count(data, epsilon, rng):
    # Count the number of valid integers in the range [0, 100]
    valid_count = sum(1 for x in data if isinstance(x, int) and 0 <= x <= 100)
    
    # Calculate the probability of including each valid record
    p_include = math.exp(epsilon) / (math.exp(epsilon) + math.exp(-epsilon))
    
    # Use the generator to decide whether to include each valid record
    private_count = sum(rng.choice([0, 1], size=1, p=[1-p_include, p_include]) for _ in range(valid_count))
    
    return private_count
```


==================== ITEM P132 ====================
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
    eps0 = 0.3  # Choose eps0 such that the claim holds: eps = ln(1 + q*(e^{eps0} - 1)) <= 0.5
    sampled_data = [x for x in data if rng.random() < q]
    count = sum(sampled_data)
    noise = rng.laplace(0, eps0 / q, size=1)
    return count + noise
```


==================== ITEM P133 ====================
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
    if not data:
        return 0, 0
    
    # Calculate the sensitivity of the data
    sensitivity = 1
    
    # Calculate the scaling factor
    scale = sensitivity / epsilon
    
    # Noisy count is the length of the data plus a laplace noise
    noisy_count = len(data) + np.random.laplace(loc=0, scale=scale, size=1)[0]
    
    # Noisy sum is the sum of the data plus a laplace noise
    noisy_sum = sum(data) + np.random.laplace(loc=0, scale=scale, size=1)[0]
    
    return noisy_count, noisy_sum
```


==================== ITEM P134 ====================
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
    total = len(data)
    if total == 0:
        noise = rng.laplace(loc=0, scale=1.0/epsilon)
        return max(0, total + int(noise))
    else:
        count = data.count(1)
        noise = rng.laplace(loc=0, scale=2.0/epsilon)
        return max(0, count + int(noise))
```


==================== ITEM P135 ====================
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
    counts = {'A': 0, 'B': 0, 'C': 0}
    for label in data:
        if label in counts:
            counts[label] += 1
    
    noise_scale = (math.exp(epsilon) - 1) / epsilon
    noisy_counts = {label: counts[label] + rng.normal(loc=0, scale=noise_scale) for label in counts}
    
    return noisy_counts
```


==================== ITEM P136 ====================
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
def private_repeated_count(data, epsilon, rng):
    """Implement a differentially private mechanism to count the number of 1s in a list of integers.

    Parameters:
        data (list of ints): the list of integers to count
        epsilon (float): the target epsilon for differential privacy
        rng (numpy.random.Generator): the source of randomness

    Returns:
        a list of four floats, each representing the count of 1s in a different dataset
    """
    # Calculate the number of 1s in the dataset
    count = sum(data)

    # Generate four independent random variables with a Bernoulli distribution
    noise1, noise2, noise3, noise4 = rng.binomial(count, 0.5, size=4)

    # Calculate the noisy count for each analyst
    noisy_count1 = count + noise1
    noisy_count2 = count + noise2
    noisy_count3 = count + noise3
    noisy_count4 = count + noise4

    # Add the noise to the original count to obtain the total noisy count
    total_noisy_count = noisy_count1 + noisy_count2 + noisy_count3 + noisy_count4

    # Return the total noisy count as four floats
    return [noisy_count1, noisy_count2, noisy_count3, noisy_count4]
```


==================== ITEM P137 ====================
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
import math

def private_valid_count(data, epsilon, rng):
    valid_data = [x for x in data if isinstance(x, int) and 0 <= x <= 100]
    n = len(valid_data)
    delta = 2 / (1 + math.exp(-epsilon))
    noise = rng.laplace(0, 1 / epsilon)
    return max(0, np.clip(n + noise, 0, len(data)))
```


==================== ITEM P138 ====================
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
    n = len(data)
    q = 0.5
    eps0 = 0.5  # Choice to ensure epsilon = 0.5 via privacy amplification by subsampling
    sampled = [x for x in data if rng.binomial(1, q) == 1]
    count = sum(sampled)
    noise = rng.laplace(0, eps0 / n)
    return count + noise
```


==================== ITEM P139 ====================
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
    if len(data) == 0:
        return 0
    else:
        clipped_data = [min(10.0, value) for value in data]
        average_clipped_value = sum(clipped_data) / len(clipped_data)
        noise = rng.laplace(loc=0, scale=1 / epsilon) if epsilon > 0 else 0
        return average_clipped_value + noise
```


==================== ITEM P140 ====================
Task: report_noisy_max_v1 | claimed epsilon=1.0 delta=0.0 | adjacency=add_remove_one

SPEC:
Write one self-contained Python function named `private_argmax`.
- Input: `data`, a list of category labels from {"A","B","C","D"};
  `epsilon`, a float; `rng`, a numpy.random.Generator (sole source of randomness).
- Output: a single string - the label of the (privately selected) most
  frequent category, using the Report Noisy Max mechanism with correctly
  calibrated noise added to each of the four counts BEFORE selection.
  Release ONLY the label, never the counts.
- Neighboring datasets differ by addition or removal of one record.
- Claim: pure epsilon-DP for epsilon = 1.0.
- Use numpy and math only. Handle empty input and ties. Return code only.

REFERENCE (correct):
```python
"""Reference implementation: task report_noisy_max_v1 (correct, eps-DP).

Input is a list of category labels from {A, B, C, D}. The mechanism computes
the four counts, adds independent Laplace(2/epsilon) noise to each count
(counts have sensitivity 1 under add/remove-one; the classic Report Noisy Max
analysis gives pure epsilon-DP with scale 2*Delta/epsilon), and returns the
label of the noisy argmax. Only the winning LABEL is released - never the
noisy counts themselves.
"""
import numpy as np

CATEGORIES = ("A", "B", "C", "D")


def private_argmax(data, epsilon, rng):
    counts = [sum(1 for x in data if x == c) for c in CATEGORIES]
    noisy = [c + rng.laplace(loc=0.0, scale=2.0 / epsilon) for c in counts]
    return CATEGORIES[int(np.argmax(noisy))]
```

PROGRAM TO ANNOTATE:
```python
def private_argmax(data, epsilon, rng):
  # Ensure input is valid
  if len(data) == 0:
    return "No samples"
  if len(data) > 4:
    return "Too many samples"
  if epsilon <= 0:
    return "Invalid epsilon"

  # Calculate the total count for all categories
  total_count = sum(data.count(cat) for cat in ["A", "B", "C", "D"])

  # Calculate the threshold for the most frequent category
  threshold = total_count - epsilon

  # Calculate the noisy counts for each category
  noisy_counts = [
    math.floor(data.count(cat) + rng.laplace(0, 1 / total_count))
    for cat in ["A", "B", "C", "D"]
  ]

  # Select the category with the most noisy count
  most_noisy_count = max(noisy_counts)

  # Determine the category with the most noisy count is within the threshold
  if most_noisy_count <= threshold:
    return "No tie"

  # Return the category with the most noisy count
  return ["A", "B", "C", "D"][noisy_counts.index(most_noisy_count)]
```


==================== ITEM P141 ====================
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
    # get the count of 1s in the data
    count_1s = np.sum(data)
    
    # calculate the total sum of the noise for each analyst
    noise_total = math.sqrt(12 * epsilon * count_1s)
    
    # generate the noise for each analyst
    noise = rng.normal(0, noise_total, 4)
    
    # return the list of noisy counts
    return noise
```


==================== ITEM P142 ====================
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
    q1 = sum(data)
    q2 = n - q1
    q3 = n
    
    lambda_T = epsilon / (2 * (q1 + q2 + q3 + 4))
    lambda_q = epsilon / (2 * (q1 + q2 + q3 + 4))
    
    T_noisy = T + lambda_T * rng.laplace(0, 1)
    q1_noisy = q1 + lambda_q * rng.laplace(0, 1)
    q2_noisy = q2 + lambda_q * rng.laplace(0, 1)
    q3_noisy = q3 + lambda_q * rng.laplace(0, 1)
    
    if q1_noisy > T_noisy:
        return 1
    if q2_noisy > T_noisy:
        return 2
    if q3_noisy > T_noisy:
        return 3
    return 0
```


==================== ITEM P143 ====================
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
    n = len(data)
    candidates = list(set(data))
    alpha = epsilon / 2
    beta = epsilon / 2
    
    def noisy_max(categories, alpha, rng):
        max_count = 0
        max_category = None
        for category in categories:
            count = data.count(category)
            noise = rng.exponential(1 / (2 * alpha))
            noisy_count = count + noise
            if noisy_count > max_count:
                max_count = noisy_count
                max_category = category
        return max_category
    
    first_winner = noisy_max(candidates, alpha, rng)
    candidates.remove(first_winner)
    beta = epsilon - alpha
    second_winner = noisy_max(candidates, beta, rng)
    
    return [first_winner, second_winner]
```


==================== ITEM P144 ====================
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
    delta = 0.01
    q = np.exp(-epsilon) / (np.exp(epsilon) + 1)
    noise_multiplier = math.sqrt(2 * math.log(1.25 / delta)) / (epsilon * q)
    
    count = data.count(1)
    noisy_counts = []
    for _ in range(4):
        noisy_count = count + rng.normal(0, noise_multiplier * math.sqrt(n))
        noisy_counts.append(max(0, min(n, round(noisy_count))))
    
    return noisy_counts
```


==================== ITEM P145 ====================
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
    step1_release = rng.laplace(0, 1 / epsilon) + sum(x >= 0.5 for x in data)
    if step1_release >= 3:
        step2_release = rng.laplace(0, 1 / epsilon) + sum(x for x in data if x >= 0.5)
    else:
        step2_release = rng.laplace(0, 1 / epsilon) + sum(x < 0.5 for x in data)
    return (step1_release, step2_release)
```


==================== ITEM P146 ====================
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
    n = len(data)
    count_budget, sum_budget = epsilon / 2, epsilon / 2
    
    if n == 0:
        return (0, 0)
    
    noisy_count = count_budget * (1 + rng.laplace(0, 2 * count_budget))
    noisy_sum = sum_budget * (sum(data) + rng.laplace(0, 2 * sum_budget))
    
    return (noisy_count, noisy_sum)
```


==================== ITEM P147 ====================
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
    
    count_ones = sum(data)
    delta = math.exp(epsilon) / (math.exp(epsilon) + 1)
    noise = rng.choice([-1, 1], p=[delta, 1-delta])
    
    return count_ones + noise
```


==================== ITEM P148 ====================
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
    if not data:
        return 0, 0
    
    n = len(data)
    
    # Compute the sensitivity of count and sum
    sensitivity = [n, sum(data)]
    
    # Compute the noise contribution based on the given epsilon value
    noise_contribution = [math.exp(rng.laplace(0, scale=2/epsilon)) for _ in sensitivity]
    
    # Add the noise contribution to the actual count and sum
    noisy_count = min(n + noise_contribution[0], len(data))
    noisy_sum = min(sum(data) + noise_contribution[1], sum(data))
    
    return noisy_count, noisy_sum
```


==================== ITEM P149 ====================
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
def private_count(data,epsilon,rng):
    if len(data)==0:
        return 0
    else:
        count=0
        sigma = math.sqrt(2/epsilon/len(data))
        for i in range(len(data)):
            if data[i]==1:
                count+=1
                sigma*=max(0,1-abs(1-rng.normal(0,1))**2)
        return count*sigma
```


==================== ITEM P150 ====================
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
    
    count = np.sum(data)
    delta = 1.0 / len(data)
    
    sensitivity = 1.0
    
    random_value = rng.normal(0, sensitivity / epsilon)
    noisy_count = count + random_value
    
    return noisy_count
```


==================== ITEM P151 ====================
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
def private_adaptive_release(data, epsilon, rng):
    # release a noisy count of records with value >= 0.5
    noisy_count = rng.uniform(low=0.0, high=1.0, size=len(data))
    step1_release = sum(noisy_count >= 0.5)

    if step1_release >= 3:
        # release a noisy sum of the values (sensitivity 1)
        noisy_sum = rng.uniform(low=0.0, high=1.0, size=len(data))
        step2_release = sum(noisy_sum >= 0.5)
    else:
        # release a noisy count of records with value < 0.5
        noisy_count = rng.uniform(low=0.0, high=1.0, size=len(data))
        step2_release = sum(noisy_count < 0.5)

    return step1_release, step2_release
```


==================== ITEM P152 ====================
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
    p = (1 + math.exp(-epsilon)) / (1 + math.exp(epsilon))
    return n + rng.binomial(1, p, 1)[0]
```


==================== ITEM P153 ====================
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
    """
    Function to create a differentially private count of the 1s in a list.
    """
    # Count the number of 1s in the data
    count = np.sum(data)

    # If the data is empty, return a random float between 0 and 1
    if count == 0:
        return rng.random()

    # Compute the noise
    noise = np.sqrt(2 * math.log(1 / (0.5 * math.e ** (-epsilon))) * 2 * epsilon)

    # Generate a random float in the range [-noise, noise]
    noisy_count = count + rng.uniform(-noise, noise)

    # Ensure that the noisy count does not exceed the original count
    noisy_count = max(noisy_count, 0)

    return noisy_count
```


==================== ITEM P154 ====================
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
    T = 4
    sensitivity = 1
    c = 1.0 / epsilon
    q1 = sum(data)
    q2 = len(data) - q1
    queries = [q1, q2, len(data)]
    
    for i, q in enumerate(queries):
        noise = rng.laplace(loc=0, scale=1.0 / (c * sensitivity))
        threshold = T + noise
        if q + noise > threshold:
            return i + 1
    return 0
```


==================== ITEM P155 ====================
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
    n = len(data)
    delta = epsilon / (2 * (n + 1))
    noisy_count = n + rng.normal(0, math.sqrt(2 * n * delta))
    noisy_sum = sum(data) + rng.normal(0, math.sqrt(n * delta))
    return (noisy_count, noisy_sum)
```


==================== ITEM P156 ====================
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
from math import exp

def private_count_safe(data, epsilon, rng):
    n = len(data)
    if n == 0:
        return 0.0
    else:
        noise = rng.normal(size=n) * exp(epsilon)
        return np.sum(data) + np.sum(noise)
```


==================== ITEM P157 ====================
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
    sensitivity_count = 1
    sensitivity_sum = 1

    # Noise for the first step
    noise1 = rng.laplace(0, sensitivity_count / epsilon, 1)[0]

    # Count records with value >= 0.5
    count_ge_0_5 = sum(1 for x in data if x >= threshold)
    noisy_count_ge_0_5 = count_ge_0_5 + noise1

    if noisy_count_ge_0_5 >= 3:
        # Noise for the second step (sum)
        noise2 = rng.laplace(0, sensitivity_sum / epsilon, 1)[0]
        # Sum of values
        sum_values = sum(data)
        noisy_sum = sum_values + noise2
        step1_release = noisy_count_ge_0_5
        step2_release = noisy_sum
    else:
        # Noise for the second step (count)
        noise2 = rng.laplace(0, sensitivity_count / epsilon, 1)[0]
        # Count records with value < 0.5
        count_lt_0_5 = sum(1 for x in data if x < threshold)
        noisy_count_lt_0_5 = count_lt_0_5 + noise2
        step1_release = noisy_count_ge_0_5
        step2_release = noisy_count_lt_0_5

    return (step1_release, step2_release)
```


==================== ITEM P158 ====================
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
    sensitivity = 1/n
    noise = rng.laplace(loc=0, scale=sensitivity/(2*epsilon))
    return np.mean(data) + noise
```


==================== ITEM P159 ====================
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
    
    clipped_data = np.clip(data, 0, 10)
    data_sum = np.sum(clipped_data)
    
    sensitivity = 10.0
    
    noise = rng.laplace(0, sensitivity / epsilon)
    private_sum = data_sum + noise
    
    return private_sum
```


==================== ITEM P160 ====================
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
    
    # Calculate the probability of adding a 1 to the estimate
    q = (math.exp(epsilon) - 1) / (math.exp(epsilon) + 1)
    
    # Estimate the number of 1s
    estimate = np.sum(data)
    
    # Add noise
    noise = rng.choice([0, 1], size=n, p=[1-q, q]) - 0.5
    private_estimate = estimate + np.sum(noise)
    
    return private_estimate
```
