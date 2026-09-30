

==================== ITEM P001 ====================
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
    if n == 0:
        return (0.0, 0.0)
    
    alpha = epsilon / 2.0
    beta = epsilon / 2.0
    
    noisy_count = rng.laplace(0, 1 / alpha) + n
    noisy_sum = rng.laplace(0, 1 / beta) + sum(data)
    
    return (noisy_count, noisy_sum)
```


==================== ITEM P002 ====================
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
        return 0.0, 0.0
    noisy_count = len(data) + np.sqrt(2 * np.log(1/epsilon)) * rng.laplace(0, scale=1/epsilon)
    noisy_sum = sum(data) + np.sqrt(2 * np.log(1/epsilon)) * rng.laplace(0, scale=1/epsilon)
    return noisy_count, noisy_sum
```


==================== ITEM P003 ====================
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
        return 0.0, 0.0

    n = len(data)
    sensitivity = 2.0
    laplace_mechanism_scale = sensitivity / epsilon

    # Apply the Laplace mechanism to the count and sum
    noisy_count = n + rng.laplace(scale=laplace_mechanism_scale)
    noisy_count = max(0, noisy_count)  # Ensure the result is non-negative

    noisy_sum = sum(data) + laplace_mechanism_scale * rng.choice([-1, 1], p=[0.5, 0.5])

    return noisy_count, noisy_sum
```


==================== ITEM P004 ====================
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
    estimate = sum(data)
    scale = epsilon / n
    noise = rng.normal(0, scale=scale)
    return estimate + noise
```


==================== ITEM P005 ====================
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
def private_count_safe(data, epsilon, rng):
    if not data:  # special case: empty data set
        return 0
    mean = np.mean(data)
    std = np.std(data)
    z = rng.normal(size=len(data))
    noise = np.abs(z) * (std / math.sqrt(len(data)))
    return np.sum(noise) + mean
```


==================== ITEM P006 ====================
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
    total = sum(clipped_data)
    lap = rng.laplace(loc=0, scale=1/epsilon, size=1)[0]
    return total + lap
```


==================== ITEM P007 ====================
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
    noise = np.sqrt(2 * math.log(1.25 / epsilon)) * sensitivity
    return np.mean(data) + rng.laplace(0, noise)
```


==================== ITEM P008 ====================
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

def private_count(data, epsilon, rng):
    n = len(data)
    noise = epsilon / 2 * np.sqrt(n) * rng.standard_cauchy(n)
    return sum(data) + epsilon / 2 * np.mean(noise)
```


==================== ITEM P009 ====================
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
    
    # Initialize the counts for each category with Laplace noise
    counts = {category: max(0, rng.laplace(0.0, scale=1.0/epsilon)) for category in categories}

    # Increment the count for each category in the data
    for category in data:
        if category in counts:
            counts[category] += 1

    return counts
```


==================== ITEM P010 ====================
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
  # Step 1: release a noisy count of records with value >= 0.5.
  noisy_count = rng.binomial(n=len(data), p=0.5, size=1)[0]
  step1_release = epsilon * noisy_count

  # Step 2: IF the noisy count from step 1 is >= 3, additionally release a noisy
  # sum of the values (sensitivity 1). Otherwise release a noisy count of
  # records with value < 0.5. Fresh noise in step 2.
  if noisy_count >= 3:
    # release noisy sum of the values (sensitivity 1)
    noisy_sum = rng.normal(size=1, scale=math.sqrt(epsilon / 1))[0]
    step2_release = noisy_sum
  else:
    # release noisy count of records with value < 0.5
    noisy_count_below_0_5 = rng.binomial(n=len(data), p=0.5, size=1)[0]
    step2_release = epsilon * noisy_count_below_0_5

  return step1_release, step2_release
```


==================== ITEM P011 ====================
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
    noisy_counts = {}
    for label in 'ABC':
        laplace_noise = rng.laplace(0, 2 * epsilon / 3)
        noisy_counts[label] = max(0, counts[label] + laplace_noise)
    return noisy_counts
```


==================== ITEM P012 ====================
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
    clipped_data = [min(max(x, 0), 10) for x in data]
    n = len(clipped_data)
    scale = math.sqrt(2 * n * epsilon)
    noise = rng.normal(0, scale, n)
    return sum(clipped_data + list(noise))
```


==================== ITEM P013 ====================
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

    ones_count = sum(data)
    total_count = len(data)
    laplace_mechanism = math.exp(epsilon) * ones_count / total_count + \
                        rng.laplace(scale=1/epsilon)

    if laplace_mechanism < 0:
        return 0
    elif laplace_mechanism > total_count:
        return total_count
    else:
        return round(laplace_mechanism)
```


==================== ITEM P014 ====================
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
    n = len(data)
    if n == 0:
        return {"A": 0, "B": 0, "C": 0}
    
    delta = math.exp(epsilon) / (math.exp(epsilon) + 2)
    q = (math.exp(epsilon) - 1) / (math.exp(epsilon) + 2)
    
    count_A = sum(1 for x in data if x == "A")
    count_B = sum(1 for x in data if x == "B")
    count_C = sum(1 for x in data if x == "C")
    
    noisy_count_A = count_A + rng.poisson(q * n) - rng.poisson(q * (n - 1))
    noisy_count_B = count_B + rng.poisson(q * n) - rng.poisson(q * (n - 1))
    noisy_count_C = count_C + rng.poisson(q * n) - rng.poisson(q * (n - 1))
    
    return {"A": noisy_count_A, "B": noisy_count_B, "C": noisy_count_C}
```


==================== ITEM P015 ====================
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
    noise = rng.normal(0, sensitivity * epsilon / 2)
    mean_estimate = np.mean(data) + noise
    return mean_estimate
```


==================== ITEM P016 ====================
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

    # handle empty input
    if len(data) == 0:
        return 0.0

    # clip data to range [0, 10]
    data = [min(max(v, 0), 10) for v in data]

    # compute sum
    sum_ = sum(data)

    # compute noise
    noise = rng.normal(scale=math.sqrt(epsilon / len(data)))

    # add noise to sum
    sum_ += noise

    return sum_
```


==================== ITEM P017 ====================
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
    noise = rng.normal(size=len(data)) * math.sqrt(epsilon)
    return np.sum(data + noise)
```


==================== ITEM P018 ====================
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

def laplace_mechanism(value, sensitivity, epsilon, rng):
    l2_norm = math.sqrt(2 * sensitivity**2)
    noise = rng.laplace(scale=l2_norm / epsilon)
    return value + noise

def private_histogram(data, epsilon, rng):
    if not data:
        return {"A": 0.0, "B": 0.0, "C": 0.0}

    frequencies = {"A": 0, "B": 0, "C": 0}
    for label in data:
        if label in frequencies:
            frequencies[label] += 1

    total_count = sum(frequencies.values())
    sensitivity = max(frequencies.values())

    for label in frequencies:
        noisy_count = laplace_mechanism(frequencies[label], sensitivity, epsilon, rng)
        noisy_count = max(0, noisy_count) # Ensure the count is non-negative
        frequencies[label] = noisy_count

    return frequencies
```


==================== ITEM P019 ====================
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
    eps0 = 0.5  # Choice of eps0 to satisfy the privacy amplification claim
    q = 0.5
    sampled_data = [x for x in data if rng.random() < q]
    count = sum(sampled_data)
    noise = rng.laplace(scale=eps0 / (2 * q))
    return count + noise
```


==================== ITEM P020 ====================
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
    sensitivity = 1  # Since we are counting the number of 1s in the data
    if len(data) == 0:
        release = 0  # If the data is empty, the release is 0
    else:
        count = np.sum(data)  # Count of 1s in the data
        # Laplace Mechanism to differentially privatize the count
        release = (count + np.exp(epsilon / 2) * rng.laplace(0, sensitivity / epsilon))
    return max(0, release)  # Return 0 if the release is negative
```


==================== ITEM P021 ====================
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

def laplace_mechanism(x, sensitivity, epsilon):
    scale = math.log(2) / epsilon
    noise = np.random.laplace(loc=0, scale=scale)
    return x + noise

def private_repeated_count(data, epsilon, rng):
    ones_count = sum(data)
    sensitivity = ones_count
    noisy_count = laplace_mechanism(ones_count, sensitivity, epsilon)
    return [noisy_count for _ in range(4)]
```


==================== ITEM P022 ====================
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
    # Count of 1s, 0s, and total length of data
    counts = [sum(data), len(data) - sum(data), len(data)]
    # Public threshold T = 4
    T = 4
    # Laplace noise parameter for threshold
    delta = 1 / (len(data) * 2 ** epsilon)
    # Apply Laplace noise to threshold
    noisy_T = T + rng.laplace(scale=delta)
    # Define the first query to exceed noisy_T
    for i, count in enumerate(counts, start=1):
        # Laplace noise parameter for each query
        delta = 1 / (len(data) * 2 ** epsilon)
        # Apply Laplace noise to each query answer
        noisy_count = count + rng.laplace(scale=delta)
        if noisy_count > noisy_T:
            return i
    return 0
```


==================== ITEM P023 ====================
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
  noisy_count = np.sum(rng.binomial(n=1, p=0.5, size=len(data)))
  private_count = math.floor(noisy_count + epsilon)
  return private_count
```


==================== ITEM P024 ====================
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

    # Clip the values to the range [0, 10]
    clipped_data = [min(max(0, x), 10) for x in data]

    # Compute the sum of the clipped values
    sum_clipped = sum(clipped_data)

    # Compute the epsilon-DP sum
    sum_dp = sum_clipped + rng.laplace(loc=0, scale=1 / epsilon)

    return sum_dp
```


==================== ITEM P025 ====================
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
  """Private version of count and sum with epsilon-DP budget.
  Input:
    data: A list of floats in [0, 1].
    epsilon: A float representing the epsilon value of the privacy budget.
    rng: A numpy.random.Generator to generate randomness.
  Output:
    The tuple (noisy_count, noisy_sum) of two floats, where noisy_count
    estimates len(data) and noisy_sum estimates sum(data) (values already in [0,1],
    so the sum has sensitivity 1 under add/remove).
  """
  if len(data) == 0:
    return (0, 0)
  else:
    noisy_count = len(data) + rng.laplace(scale=(epsilon/2), size=1)
    noisy_sum = sum(data) + rng.laplace(scale=(epsilon/2), size=1)
  return (noisy_count, noisy_sum)
```


==================== ITEM P026 ====================
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
    q1 = sum(data)
    q2 = len(data) - q1
    q3 = len(data)
    n = len(data)
    
    lambda_threshold = epsilon / (2 * (T + 1))
    lambda_queries = epsilon / (2 * n)
    
    noisy_T = T + rng.laplace(0, lambda_threshold)
    
    noisy_q1 = q1 + rng.laplace(0, lambda_queries)
    noisy_q2 = q2 + rng.laplace(0, lambda_queries)
    noisy_q3 = q3 + rng.laplace(0, lambda_queries)
    
    if noisy_q1 > noisy_T:
        return 1
    if noisy_q2 > noisy_T:
        return 2
    if noisy_q3 > noisy_T:
        return 3
    return 0
```


==================== ITEM P027 ====================
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
    noise = rng.laplace(0, sensitivity * epsilon / 2)
    mean = sum(data) / n
    return mean + noise
```


==================== ITEM P028 ====================
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
def private_valid_count(data, epsilon, rng):
  l = len(data)
  n = rng.poisson(epsilon * l)
  valid = [d for d in data if type(d) == int and 0 <= d < 100]
  return math.log(1 + n) + len(valid)

# Code block ends
```


==================== ITEM P029 ====================
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
    
    min_val = 0.0
    max_val = 10.0
    
    # Clip the data to the range [0, 10]
    clipped_data = [min(max(x, min_val), max_val) for x in data]
    
    # Calculate the sum of the clipped data
    true_sum = sum(clipped_data)
    
    # Calculate the noise
    noise = rng.laplace(loc=0, scale=(2.0 / epsilon))
    
    # Add the noise to the true sum
    private_sum = true_sum + noise
    
    return private_sum
```


==================== ITEM P030 ====================
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
def private_histogram(data, epsilon, rng):
    bucket_A = rng.binomial(n=len(data), p=0.4, size=1)[0]
    bucket_B = rng.binomial(n=len(data), p=0.6, size=1)[0]
    bucket_C = len(data) - bucket_A - bucket_B
    buckets = {"A": bucket_A, "B": bucket_B, "C": bucket_C}
    return buckets
```


==================== ITEM P031 ====================
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
    if n == 0:
        return (0, 0)
    alpha = epsilon / 2
    beta = epsilon / 2
    laplace_count = rng.laplace(0, 2 / alpha)
    laplace_sum = rng.laplace(0, 1 / beta)
    noisy_count = max(0, min(n + laplace_count, n))
    noisy_sum = max(0, min(sum(data) + laplace_sum, 1))
    return (noisy_count, noisy_sum)
```


==================== ITEM P032 ====================
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
    delta = 1/(n*n)
    epsilon_diff = math.log(1/delta)
    lap_noise = rng.laplace(loc=0, scale=sensitivity/(epsilon_diff/epsilon))
    return np.mean(data) + lap_noise
```


==================== ITEM P033 ====================
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

    max_value = 10.0
    clipped_data = [min(value, max_value) for value in data]
    sum_clipped = sum(clipped_data)
    sensitivity = max_value

    noise = rng.laplace(0, sensitivity / epsilon)
    private_sum = sum_clipped + noise

    return private_sum
```


==================== ITEM P034 ====================
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
    noisy_count_1 = len([x for x in data if x >= 0.5]) + np.float64(0.0)
    noisy_count_1 += math.pow(10, rng.laplace(0)) / (2 * math.exp(1))

    if noisy_count_1 >= 3:
        noisy_sum_2 = sum(data) + math.pow(10, rng.laplace(0)) / (2 * math.exp(1))
        return (noisy_count_1, noisy_sum_2)
    else:
        noisy_count_2 = len([x for x in data if x < 0.5]) + math.pow(10, rng.laplace(0)) / (2 * math.exp(1))
        return (noisy_count_1, noisy_count_2)
```


==================== ITEM P035 ====================
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
    q1 = np.count_nonzero(data)
    q2 = len(data) - q1
    q3 = len(data)

    T = 4

    # Laplace mechanism
    q1_noise = np.random.laplace(loc=0, scale=1/epsilon, size=1)[0]
    q2_noise = np.random.laplace(loc=0, scale=1/epsilon, size=1)[0]
    q3_noise = np.random.laplace(loc=0, scale=1/epsilon, size=1)[0]

    q1_private = q1 + q1_noise
    q2_private = q2 + q2_noise
    q3_private = q3 + q3_noise

    # Check which query exceeds the threshold
    if q1_private > T:
        return 1
    elif q2_private > T:
        return 2
    elif q3_private > T:
        return 3
    else:
        return 0
```


==================== ITEM P036 ====================
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
    alpha = (1.0 / epsilon) * math.log((1.0 + math.exp(epsilon)) / 2.0)
    noise = rng.normal(0, alpha)
    return count + noise
```


==================== ITEM P037 ====================
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
        q = math.floor(math.log(1/epsilon)) + 1
        ans = sum(1 for _ in range(q)) * (len(data) - np.random.geometric(1/len(data), size=q))
        return ans + np.sum(data)
```


==================== ITEM P038 ====================
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
    if len(data) == 0:
        return 0.0
    data = np.clip(data, 0.0, 10.0)
    mean = np.mean(data)
    var = np.var(data)
    std = np.sqrt(var)
    t = rng.normal(mean, std)
    return min(max(t, 0.0), 10.0)
```


==================== ITEM P039 ====================
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
    
    clipped_data = [min(x, 10.0) for x in data]
    
    sum_data = sum(clipped_data)
    scale = epsilon / (2 * sum_data) if sum_data > 0 else epsilon
    
    laplace_noise = rng.laplace(scale=scale, size=len(clipped_data))
    noisy_sum = sum_data + np.sum(laplace_noise)
    
    return max(noisy_sum, 0.0)
```


==================== ITEM P040 ====================
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
    count = np.sum(data)
    noise = rng.laplace(0, 1/epsilon)
    return max(0, count + noise)
```
