

==================== ITEM P041 ====================
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
    counts = np.array([item == 'A' for item in data])
    laplace_scale = math.log(2) / epsilon
    noisy_counts = rng.laplace(scale=laplace_scale) * np.ones(len(counts)) + counts
    return {'A': noisy_counts.sum(), 'B': 0, 'C': 0}
```


==================== ITEM P042 ====================
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
from math import sqrt

def private_mean(data, epsilon, rng):
    n = len(data)
    mean = np.mean(data)
    std = sqrt(np.mean(np.square(data)))
    noise = rng.normal(0, 1, n)
    noise = noise / std * sqrt(2 * epsilon)
    return mean + noise
```


==================== ITEM P043 ====================
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
    if not data:
        return 0
    else:
        num_ones = len([x for x in data if x == 1])
        lap_noise = rng.laplace(0, 1/epsilon)
        noisy_count = num_ones + lap_noise
        return noisy_count
```


==================== ITEM P044 ====================
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
    
    clipped_data = np.minimum(data, 10.0)
    n = len(clipped_data)
    sensitivity = 10.0
    
    z = rng.laplace(0, sensitivity / epsilon)
    return np.sum(clipped_data) + z
```


==================== ITEM P045 ====================
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
    elif len(data) == 1:
        return min(data[0], 10.0)
    else:
        # clip the values to the range [0, 10]
        data = [min(x, 10.0) for x in data]
        # compute the sum of the clipped values
        sum_clipped = sum(data)
        # compute the difference between the sum of the original values and the sum of the clipped values
        diff = sum(data) - sum_clipped
        # compute the noise term using the difference and the specified epsilon
        noise = rng.laplace(loc=0.0, scale=1.0 / (2 * epsilon)) * diff
        # return the sum of the clipped values with noise added
        return sum_clipped + noise
```


==================== ITEM P046 ====================
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
    noise = rng.normal(0, sensitivity * epsilon / math.sqrt(2 * n))
    private_mean = np.mean(data) + noise
    return private_mean
```


==================== ITEM P047 ====================
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
    delta = math.exp(epsilon)
    dp_mean = sum(data) + rng.laplace(scale=sensitivity/(2*delta))
    return dp_mean
```


==================== ITEM P048 ====================
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
    """
    Returns a differentially private histogram of the input data.

    :param data: List of category labels, each one of "A", "B", or "C"
    :type data: list
    :param epsilon: Differential privacy parameter
    :type epsilon: float
    :param rng: Source of randomness for adding noise to the histogram
    :type rng: numpy.random.Generator
    :return: Differentially private histogram of the input data
    :rtype: dict
    """

    # Calculate the number of records in the dataset
    num_records = len(data)

    # Initialize the histogram with zero counts for all categories
    histogram = {"A": 0.0, "B": 0.0, "C": 0.0}

    # Add noise to the histogram
    for i in range(num_records):
        category = data[i]
        if category == "A":
            histogram["A"] += rng.laplace(loc=0, scale=1.0 / epsilon)
        elif category == "B":
            histogram["B"] += rng.laplace(loc=0, scale=1.0 / epsilon)
        else:
            histogram["C"] += rng.laplace(loc=0, scale=1.0 / epsilon)

    return histogram
```


==================== ITEM P049 ====================
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
    # Handle empty dataset
    if not data:
        return 0.0

    # Clip values to [0, 10]
    clipped_data = [min(10.0, max(0.0, x)) for x in data]

    # Calculate the sum of the clipped data
    clipped_sum = sum(clipped_data)

    # Calculate the number of values and the average
    N = len(clipped_data)
    avg = clipped_sum / N

    # Calculate the noise
    noise = math.sqrt(2 * math.log(1.25 / epsilon) / (N * epsilon))

    # Add noise to the average
    noisy_avg = avg + noise * rng.normal(0, 1, 1)[0]

    # Return the noisy average as the differentially private estimate
    return noisy_avg
```


==================== ITEM P050 ====================
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
    clipped_data = [min(10.0, max(0.0, x)) for x in data]
    total = sum(clipped_data)
    if total == 0.0:
        return 0.0
    scale = math.exp(epsilon / 2)
    noise = rng.normal(0, scale)
    return total + noise
```


==================== ITEM P051 ====================
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
        return 0, 0
    
    scale = math.sqrt(2) / epsilon
    noisy_count = scale * (len(data) - np.mean(data)) + len(data)
    noisy_count = max(0, noisy_count)
    
    noisy_sum = scale * (sum(data) - len(data) / 2) + sum(data)
    noisy_sum = max(0, noisy_sum)
    
    return noisy_count, noisy_sum
```


==================== ITEM P052 ====================
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
    true_count = sum(data)
    n = len(data)
    if n == 0:
        return rng.normal(0, math.sqrt(2 * epsilon))
    else:
        delta = math.exp(epsilon) - 1
        noisy_count = true_count + rng.normal(0, math.sqrt(2 * (delta * n + epsilon)))
        return noisy_count
```


==================== ITEM P053 ====================
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
    scale1 = math.exp(epsilon / 2) / (math.exp(epsilon / 2) + 1)
    scale2 = math.exp(epsilon / 2) / (math.exp(epsilon / 2) - 1)
    
    noisy_count_ge_0_5 = rng.poisson(n * scale1)
    noisy_count_lt_0_5 = rng.poisson(n * (1 - scale1))
    
    sum_ge_0_5 = 0
    if noisy_count_ge_0_5 >= 3:
        sum_ge_0_5 = rng.normal(sum(data[i] for i in range(n) if data[i] >= 0.5), math.sqrt(n * scale2))
    
    step1_release = noisy_count_ge_0_5
    step2_release = noisy_count_lt_0_5 + sum_ge_0_5
    
    return step1_release, step2_release
```


==================== ITEM P054 ====================
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
    if n == 0:
        return 0
    else:
        # Laplace Mechanism
        sensitivity = 1
        dp_count = sensitivity * math.log(n + 1) / math.log(math.exp(1) + 1)
        noise = rng.laplace(scale=1/epsilon)
        return max(0, dp_count + noise)
```


==================== ITEM P055 ====================
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
        return (0., 0.)

    sensitivity = 2.
    scale = sensitivity / epsilon
    noisy_count = math.floor(n + rng.laplace(scale=scale))
    noisy_count = max(0, noisy_count)  # to avoid negative counts

    sum_data = sum(data)
    diff = sum_data - noisy_count

    noisy_sum = sum_data + rng.laplace(scale=scale)
    noisy_sum = max(0, noisy_sum)  # to avoid negative sums

    return (noisy_count, noisy_sum)
```


==================== ITEM P056 ====================
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

def private_sum(data, epsilon, rng):
    if not data:
        return 0.0
    
    clipped_data = [min(10.0, x) for x in data]
    
    total = sum(clipped_data)
    noise = rng.laplace(loc=0, scale=1/epsilon)
    
    return total + noise
```


==================== ITEM P057 ====================
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

def private_count_and_sum(data, epsilon, rng):
    if not data:
        return 0, 0
    
    n = len(data)
    total = sum(data)
    sensitivity = 1
    delta = sensitivity / (2 * n)
    noise = np.sqrt(2 * np.log(1 / (epsilon / 2)))
    
    noisy_count = (n + rng.laplace(scale=sensitivity / (2 * delta), size=1))[0]
    noisy_count = max(0, noisy_count)
    noisy_sum = (total + rng.laplace(scale=sensitivity / (2 * delta), size=1))[0]
    
    return noisy_count, noisy_sum
```


==================== ITEM P058 ====================
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
        return 0.0
    else:
        # Clip the values to the range [0, 10]
        clipped_data = [min(10.0, val) for val in data]
        
        # Calculate the sum of the clipped values
        sum_clipped = sum(clipped_data)
        
        # Calculate the differentially private sum
        sum_private = sum_clipped + rng.normal(scale=math.sqrt(epsilon * len(data)))
        
        return sum_private
```


==================== ITEM P059 ====================
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
    laplace_scale = epsilon / (2 * n)
    noisy_count_ge_0_5 = rng.laplace(0, scale=laplace_scale) + sum(1 for x in data if x >= 0.5)
    
    if noisy_count_ge_0_5 >= 3:
        noisy_sum = rng.laplace(0, scale=laplace_scale) + sum(x for x in data if x >= 0.5)
        step1_release = noisy_count_ge_0_5
        step2_release = noisy_sum
    else:
        noisy_count_lt_0_5 = rng.laplace(0, scale=laplace_scale) + sum(1 for x in data if x < 0.5)
        step1_release = noisy_count_ge_0_5
        step2_release = noisy_count_lt_0_5
    
    return (step1_release, step2_release)
```


==================== ITEM P060 ====================
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
from numpy import mean
from math import fsum

def private_mean(data, epsilon, rng):
    if epsilon == 1.0:
        return (fsum(data) / len(data))
    else:
        delta = math.sqrt((2 * math.log(2 / epsilon)) ** 2 + 2)
        scaled_data = (data - mean(data)) / delta
        return mean(scaled_data)
```


==================== ITEM P061 ====================
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
    
    delta = epsilon / 2
    count_budget = math.exp(epsilon) / (math.exp(epsilon) + 1)
    sum_budget = epsilon / 2
    
    noisy_count = rng.poisson(count_budget) + rng.poisson(count_budget) - 1
    noisy_sum = rng.laplace(0, sum_budget / n, size=n).sum()
    
    return (noisy_count, noisy_sum)
```


==================== ITEM P062 ====================
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
    Return a differentially private count of records that are valid integers in the range [0, 100].
    """
    # Compute the number of valid records
    num_valid = len([x for x in data if isinstance(x, int) and 0 <= x <= 100])

    # Compute the total number of records
    num_total = len(data)

    # Compute the difference in the number of valid records between the two datasets
    diff = num_valid - num_total

    # Add noise to the difference to ensure differential privacy
    diff += rng.normal(scale=epsilon, size=1)

    return diff
```


==================== ITEM P063 ====================
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

def private_sum(data, epsilon, rng):
    if not data:
        return 0
    max_value = 10.0
    min_value = 0.0
    noisy_sum = sum(np.clip(x, min_value, max_value) for x in data)
    scaling_factor = max(1, noisy_sum * np.exp(epsilon) / max_value)
    scaled_sum = noisy_sum / scaling_factor
    return scaled_sum
```


==================== ITEM P064 ====================
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
    bins = {'A': 0, 'B': 0, 'C': 0}
    for label in data:
        if label in bins:
            bins[label] += 1

    counts = bins.copy()
    laplace_scale = epsilon / 3
    for label in counts:
        noise = rng.laplace(0, laplace_scale)
        counts[label] += noise

    # Ensure non-negative counts
    for label in counts:
        counts[label] = max(0, counts[label])

    return counts
```


==================== ITEM P065 ====================
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
    else:
        clipped_data = [min(10.0, x) for x in data]
        total = sum(clipped_data)
        delta = max(0, total - len(clipped_data) * 10.0) / (2 ** epsilon)
        noise = rng.laplace(0, 1/epsilon) * 2
        return max(0, total + delta + noise)
```


==================== ITEM P066 ====================
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
    count_ge_05 = rng.binomial(n=len(data), p=0.5)
    count_ge_05 = np.clip(count_ge_05, a_min=0, a_max=len(data))
    step1_release = count_ge_05

    # Step 2: IF the noisy count from step 1 is >= 3, additionally release a noisy
    # sum of the values (sensitivity 1); otherwise release a noisy count of
    # records with value < 0.5. Fresh noise in step 2.
    sum_values = rng.normal(size=len(data))
    sum_values_ge_05 = rng.binomial(n=len(sum_values), p=0.5)
    sum_values_ge_05 = np.clip(sum_values_ge_05, a_min=0, a_max=len(sum_values))
    step2_release = sum_values_ge_05 if count_ge_05 >= 3 else count_ge_05

    return step1_release, step2_release
```


==================== ITEM P067 ====================
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
    counts = {"A": 0, "B": 0, "C": 0}
    for item in data:
        if item in counts:
            counts[item] += 1
            
    total = len(data)
    if total == 0:
        return counts

    laplace_mech_param = math.exp(epsilon/2)
    for key in counts:
        # Noise addition as per the Laplace mechanism
        noise = rng.laplace(loc=0, scale=1/laplace_mech_param)
        counts[key] = int(round(counts[key] + noise))
        # Ensure count is a non-negative integer
        counts[key] = max(0, counts[key])

    return counts
```


==================== ITEM P068 ====================
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
    if len(data) == 0:
        return (0, 0)

    # Per-query budgets for count and sum, respectively
    count_budget = math.log(1 / epsilon) / len(data)
    sum_budget = math.log(1 / epsilon) * len(data)

    # Noise for count and sum
    noise_count = rng.normal(scale=math.sqrt(count_budget / len(data)))
    noise_sum = rng.normal(scale=math.sqrt(sum_budget / len(data)))

    return (len(data) + noise_count, sum(data) + noise_sum)
```


==================== ITEM P069 ====================
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
        return (0.0, 0.0)
    
    alpha = epsilon / 2
    beta = epsilon / 2
    
    noisy_count = len(data) + rng.laplace(0, alpha / len(data))
    noisy_sum = sum(data) + rng.laplace(0, beta / len(data))
    
    return (noisy_count, noisy_sum)
```


==================== ITEM P070 ====================
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
  return len([x for x in data if isinstance(x, int) and 0 <= x <= 100])
```


==================== ITEM P071 ====================
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
    if n == 0:
        return 0.0
    
    delta = 0.0001  # Smaller delta for higher probability of privacy
    q = 1 / (1 + math.exp(epsilon))
    
    if rng.binomial(1, q):
        return n + rng.poisson(epsilon * n * delta)
    else:
        return rng.poisson(epsilon * n * delta)
```


==================== ITEM P072 ====================
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
    # Handle empty input
    if not data:
        return 0.0
    
    # Clip values to range [0, 10]
    clipped_data = [min(10.0, max(0.0, x)) for x in data]
    
    # Differentially private estimate
    noise = rng.laplace(loc=0, scale=1/epsilon)
    return sum(clipped_data) + noise
```


==================== ITEM P073 ====================
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
    
    # Calculate the true count of 1s in the data
    true_count = np.sum(data)
    
    # Calculate the minimum and maximum possible counts
    min_count = max(0, true_count - 1)
    max_count = min(n, true_count + 1)
    
    # Compute the probability of each possible count
    probabilities = []
    for count in range(min_count, max_count + 1):
        prob = np.exp((count - true_count) * epsilon) / sum(np.exp((i - true_count) * epsilon) for i in range(min_count, max_count + 1))
        probabilities.append(prob)
    
    # Sample from the probabilities using the given rng
    sampled_count = rng.choice(range(min_count, max_count + 1), p=probabilities)
    
    # Return the sampled count
    return sampled_count
```


==================== ITEM P074 ====================
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
    n = len(data)
    sensitivity = 1 / n
    delta = sensitivity / (2 * n)
    normalized_data = [(x - np.mean(data)) / np.std(data) for x in data]
    quantile = np.quantile(normalized_data, 1 - (1 + sensitivity) / (2 * n))
    noise = rng.laplace(0, scale=2 / epsilon)
    while np.abs(quantile + noise) > 1:
        noise = rng.laplace(0, scale=2 / epsilon)
    return np.mean(data) + noise * np.std(data)
```


==================== ITEM P075 ====================
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
    # Check if the data is empty
    if len(data) == 0:
        return 0.0
    
    # Calculate the number of ones in the data
    num_ones = len([x for x in data if x == 1])
    
    # Calculate the number of pairs of neighboring records
    num_pairs = len(data) - 1
    
    # Calculate the differentially private noise
    noise = rng.normal(scale=math.sqrt(2 * epsilon * num_pairs))
    
    # Return the sum of the number of ones and the differentially private noise
    return num_ones + noise
```


==================== ITEM P076 ====================
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
    # threshold is the value of the private variable
    threshold = 4

    # calculate the sensitivity of each query
    sensitivity_1 = 1
    sensitivity_0 = 1
    sensitivity_len = 1

    # calculate the total sensitivity of the queries
    total_sensitivity = sensitivity_1 + sensitivity_0 + sensitivity_len

    # calculate the noise scale for the threshold
    threshold_noise_scale = epsilon / total_sensitivity

    # calculate the noise scale for each query
    query_noise_scale = epsilon / total_sensitivity

    # add Laplace noise to the threshold once
    threshold = threshold + rng.laplace(loc=0, scale=threshold_noise_scale)

    # add fresh Laplace noise to each query answer
    count_1 = len([x for x in data if x == 1]) + rng.laplace(loc=0, scale=query_noise_scale)
    count_0 = len([x for x in data if x == 0]) + rng.laplace(loc=0, scale=query_noise_scale)
    count_len = len(data) + rng.laplace(loc=0, scale=query_noise_scale)

    # check if the threshold has been crossed
    if count_1 >= threshold:
        return 1
    elif count_0 >= threshold:
        return 2
    elif count_len >= threshold:
        return 3
    else:
        return 0
```


==================== ITEM P077 ====================
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
    if not data:
        return 0.0

    n = len(data)
    k = sum(data)
    delta = 1 / (2 * n)

    p = 0.5 * (1 + math.tanh(n / (2 * math.sqrt(epsilon))))
    noise = rng.laplace(loc=0, scale=2 / epsilon)

    if p + delta <= noise <= 1 - delta:
        return k + noise
    elif noise < 1 - 2 * delta:
        return k - n
    else:
        return k + n
```


==================== ITEM P078 ====================
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
    counts = {"A": 0, "B": 0, "C": 0}
    for label in data:
        if label in counts:
            counts[label] += 1
    
    noise_scale = math.exp(epsilon) / (math.exp(epsilon) - 1)
    noisy_counts = {key: counts[key] + noise_scale * rng.normal() for key in counts}
    
    return noisy_counts
```


==================== ITEM P079 ====================
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
  This function takes in a list of floats in [0, 1], an epsilon value, and a numpy.random.Generator.
  """
  n = len(data)
  sum_x = float(sum(data))
  mean = sum_x/n
  delta = math.sqrt(2/n * math.log(1/epsilon))
  noise = rng.normal(scale=delta)
  return mean + noise
```


==================== ITEM P080 ====================
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
    
    ones = np.sum(data)
    total = len(data)
    
    # Laplace's mechanism
    noise = rng.laplace(loc=0, scale=1/epsilon)
    estimate = ones + noise
    
    # Clipping the estimate to be between 0 and total
    estimate = max(0, estimate)
    estimate = min(estimate, total)
    
    return estimate
```
