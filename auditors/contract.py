"""Audit contracts: what an auditor is allowed to probe a mechanism with.

An empirical DP auditor decides whether a program violates its claim by feeding
it inputs. Which inputs are admissible is part of the claim, not a property of
the auditor. When that is left implicit, a perfectly correct auditor can report a
perfectly real violation on an input the task never admitted, and the result is
uninterpretable.

That is not hypothetical. In v1, StatDP flagged our *correct* randomized-response
reference because it generated a record of 2 where the contract says {0,1}; the
mechanism's outputs then had disjoint support and any correct test reports a
violation. The finding was right and the input was inadmissible.

This module turns that anecdote into something measurable. It loads the declared
contracts, checks probe inputs and neighbouring pairs against them, and COUNTS
violations rather than silently allowing or silently blocking them. The counts
are the evidence: an auditor's disagreements are only interpretable once you know
how often it left the domain.

    from auditors.contract import load_contracts, check_dataset, check_pair

    C = load_contracts()
    v = check_dataset(C["randomized_response_v1"], [2])
    #  -> [Violation(kind='value_domain', detail='record 0 = 2 outside [0, 1]')]
"""
from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Iterable

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "benchmark" / "contracts.yaml"


@dataclasses.dataclass(frozen=True)
class Violation:
    kind: str          # value_domain | record_type | size | adjacency
    detail: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}"


@dataclasses.dataclass(frozen=True)
class Contract:
    task_id: str
    record_type: str
    domain: Any
    min_size: int
    max_size: int | None
    adjacency: str
    output_type: str
    ignored_ok: bool = False
    notes: str = ""

    @property
    def accepts_anything(self) -> bool:
        """Tasks whose specification admits arbitrary values by design."""
        return self.record_type == "any"


def load_contracts(path: pathlib.Path | None = None) -> dict[str, Contract]:
    raw = yaml.safe_load((path or CONTRACTS).read_text(encoding="utf-8"))
    out = {}
    for tid, c in raw["tasks"].items():
        out[tid] = Contract(
            task_id=tid,
            record_type=c["record_type"],
            domain=c.get("domain"),
            min_size=c.get("min_size", 0),
            max_size=c.get("max_size"),
            adjacency=c["adjacency"],
            output_type=c.get("output_type", "float"),
            ignored_ok=bool(c.get("ignored_ok", False)),
            notes=(c.get("notes") or "").strip(),
        )
    return out


def _bad_type(c: Contract, x: Any) -> bool:
    if c.record_type == "int":
        # bool is a subclass of int in Python; True is not the integer 1 here.
        return isinstance(x, bool) or not isinstance(x, int)
    if c.record_type == "float":
        return isinstance(x, bool) or not isinstance(x, (int, float))
    if c.record_type == "categorical":
        return not isinstance(x, str)
    return False


def check_dataset(c: Contract, data: Iterable[Any]) -> list[Violation]:
    """Every way a probe dataset can fall outside the declared contract."""
    data = list(data)
    v: list[Violation] = []

    if len(data) < c.min_size:
        v.append(Violation("size", f"length {len(data)} < min_size {c.min_size}"))
    if c.max_size is not None and len(data) > c.max_size:
        v.append(Violation("size", f"length {len(data)} > max_size {c.max_size}"))

    if c.accepts_anything:
        return v

    for i, x in enumerate(data):
        if _bad_type(c, x):
            v.append(Violation("record_type",
                               f"record {i} = {x!r} is not {c.record_type}"))
            continue
        if c.record_type == "categorical":
            # Tasks that promise to ignore unknown labels accept them as input.
            if c.domain and x not in c.domain and not c.ignored_ok:
                v.append(Violation("value_domain",
                                   f"record {i} = {x!r} outside {c.domain}"))
        elif c.domain is not None:
            lo, hi = c.domain
            if lo is not None and x < lo:
                v.append(Violation("value_domain",
                                   f"record {i} = {x} below {lo}"))
            if hi is not None and x > hi:
                v.append(Violation("value_domain",
                                   f"record {i} = {x} above {hi}"))
    return v


def check_pair(c: Contract, d1: Iterable[Any], d2: Iterable[Any]) -> list[Violation]:
    """Both sides admissible, AND actually neighbours under the declared relation.

    The adjacency check is the half that v1 had no way to express. StatDP's
    ONE_DIFFER generates replace-one pairs; our tasks mostly declare
    add/remove-one. Those are different relations, and a pair can satisfy one
    while violating the other.
    """
    d1, d2 = list(d1), list(d2)
    v = [Violation(x.kind, f"left {x.detail}") for x in check_dataset(c, d1)]
    v += [Violation(x.kind, f"right {x.detail}") for x in check_dataset(c, d2)]

    if c.adjacency == "add_remove_one":
        if abs(len(d1) - len(d2)) != 1:
            v.append(Violation(
                "adjacency",
                f"add_remove_one needs lengths differing by exactly 1, "
                f"got {len(d1)} and {len(d2)}"))
        else:
            big, small = (d1, d2) if len(d1) > len(d2) else (d2, d1)
            if not _is_subsequence_after_one_removal(big, small):
                v.append(Violation(
                    "adjacency",
                    "lengths differ by 1 but the shorter side is not the longer "
                    "one with a single record removed"))
    elif c.adjacency == "replace_one":
        if len(d1) != len(d2):
            v.append(Violation(
                "adjacency",
                f"replace_one needs equal lengths, got {len(d1)} and {len(d2)}"))
        else:
            diff = sum(1 for a, b in zip(d1, d2) if a != b)
            if diff != 1:
                v.append(Violation(
                    "adjacency",
                    f"replace_one needs exactly 1 differing record, got {diff}"))
    return v


def _is_subsequence_after_one_removal(big: list, small: list) -> bool:
    """True if `small` equals `big` with exactly one element deleted."""
    for i in range(len(big)):
        if big[:i] + big[i + 1:] == small:
            return True
    return False


# ---------------------------------------------------------------------------
# E3: predicting auditor capability from the contract alone.
#
# The claim being tested is that `auditor_unsupported` is not bookkeeping but a
# property derivable from the declared contract before any program is run. If a
# prediction here disagrees with an observed outcome, that is a finding about the
# contract, not a bug to be papered over.
# ---------------------------------------------------------------------------

#: Output types our auditor can canonicalise into events.
OURS_SUPPORTS = {"float", "int", "str", "dict", "list", "tuple"}
#: StatDP's event search assumes a numeric release.
STATDP_SUPPORTS = {"float", "int"}

#: The three things a contract can say about an auditor, which the first version
#: of this function wrongly collapsed into two.
#:
#:   auditable              the auditor can represent the release AND the test it
#:                          runs is the test the task claims
#:   unfaithful             the auditor will happily return a verdict, but the
#:                          condition it tests is not the claimed guarantee. Our
#:                          pure-eps auditor on an (eps, delta) task is the case:
#:                          it does not refuse, it answers the wrong question
#:   unsupported            the auditor cannot represent the declared release at
#:                          all and will decline
CAPABILITIES = ("auditable", "unfaithful", "unsupported")


def predict_capability(c: Contract, auditor: str, claimed_delta: float = 0.0) -> str:
    """contract -> capability, with no reference to any program.

    Note what this does NOT predict. It is a statement about a program that
    honours its own output contract. A generated program that returns None where
    the specification says "a single string" will be recorded as unsupported by
    the auditor, but that is the program breaking the contract, not the auditor
    failing to cover the task. E3 separates the two, and in our corpus every
    observed unsupported outcome falls in the second class.
    """
    if auditor == "ours":
        if c.output_type not in OURS_SUPPORTS:
            return "unsupported"
        if claimed_delta and claimed_delta > 0:
            return "unfaithful"
        return "auditable"
    if auditor == "statdp":
        if c.output_type not in STATDP_SUPPORTS or c.record_type == "categorical":
            return "unsupported"
        if claimed_delta and claimed_delta > 0:
            return "unfaithful"
        return "auditable"
    raise ValueError(f"unknown auditor {auditor!r}")


# ---------------------------------------------------------------------------
# E2: generating neighbouring pairs FROM the contract.
#
# The objection is that our hand-designed pairs drive the results. The answer is
# to generate pairs we did not choose, from the declared domain, and re-audit.
#
# Every pair produced here is passed through check_pair before being returned, so
# the generator cannot quietly emit something inadmissible and make the robustness
# result meaningless. If it ever does, that is a bug in this function and it
# raises rather than returning.
# ---------------------------------------------------------------------------

def _sample_record(c: Contract, rng) -> Any:
    if c.record_type == "int":
        lo, hi = (c.domain if c.domain else [0, 1])
        return int(rng.integers(lo, hi + 1))
    if c.record_type == "float":
        lo, hi = (c.domain if c.domain else [0.0, 1.0])
        lo = 0.0 if lo is None else float(lo)
        # An unbounded upper end means "may exceed the clip bound"; probe above it
        # deliberately, since exercising the clip is the point of such tasks.
        hi = (lo + 25.0) if hi is None else float(hi)
        return float(rng.uniform(lo, hi))
    if c.record_type == "categorical":
        return str(rng.choice(list(c.domain)))
    # record_type "any": the task promises to survive arbitrary values
    return rng.choice([0, 1, 7, -3, 2.5, "oops", "", None, True], size=1).tolist()[0]


def random_valid_pair(c: Contract, rng, size: int | None = None):
    """One neighbouring pair drawn from the declared domain, verified before use."""
    lo = max(c.min_size, 1 if c.adjacency == "add_remove_one" else c.min_size)
    hi = c.max_size if c.max_size is not None else max(lo, 8)
    if size is None:
        size = int(rng.integers(lo, hi + 1)) if hi > lo else lo

    if c.adjacency == "add_remove_one":
        # d1 has one more record than d2, and d2 is d1 with one record dropped.
        size = max(size, max(c.min_size + 1, 1))
        if c.max_size is not None:
            size = min(size, c.max_size)
        d1 = [_sample_record(c, rng) for _ in range(size)]
        drop = int(rng.integers(0, len(d1)))
        d2 = d1[:drop] + d1[drop + 1:]
    else:  # replace_one: equal lengths, exactly one record different
        size = max(size, max(c.min_size, 1))
        d1 = [_sample_record(c, rng) for _ in range(size)]
        d2 = list(d1)
        i = int(rng.integers(0, len(d1)))
        for _ in range(40):
            v = _sample_record(c, rng)
            if v != d1[i]:
                d2[i] = v
                break
        else:
            raise ValueError(f"{c.task_id}: could not vary a record to build a "
                             f"replace-one pair")

    bad = check_pair(c, d1, d2)
    if bad:
        raise ValueError(f"{c.task_id}: generator produced an invalid pair: "
                         + "; ".join(str(b) for b in bad))
    return d1, d2


def random_valid_pairs(c: Contract, rng, k: int = 20):
    """K distinct valid pairs, or as many distinct ones as the domain allows."""
    out, seen, attempts = [], set(), 0
    while len(out) < k and attempts < k * 40:
        attempts += 1
        d1, d2 = random_valid_pair(c, rng)
        key = (repr(d1), repr(d2))
        if key in seen:
            continue
        seen.add(key)
        out.append((d1, d2))
    return out


def conforms_to_output(c: Contract, value: Any) -> bool:
    """Does an actual release match the output type the task declared?"""
    t = c.output_type
    if t == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "str":
        return isinstance(value, str)
    if t == "dict":
        return isinstance(value, dict)
    if t == "tuple":
        return isinstance(value, (tuple, list)) and len(value) == 2
    if t == "list":
        return isinstance(value, (tuple, list))
    return True


if __name__ == "__main__":
    C = load_contracts()
    print(f"{len(C)} contracts loaded\n")

    print("the v1 StatDP episode, now machine-checkable:")
    for probe in ([1], [2], [1, 1]):
        vs = check_dataset(C["randomized_response_v1"], probe)
        print(f"  probe {probe!r:8s} -> "
              + ("admissible" if not vs else "; ".join(str(x) for x in vs)))

    print("\nadjacency, which v1 could not express:")
    rr = C["randomized_response_v1"]
    print(f"  ([1], [0]) replace_one -> "
          f"{[str(x) for x in check_pair(rr, [1], [0])] or 'admissible'}")
    lc = C["laplace_bounded_count_v1"]
    print(f"  ([1,1], [1]) add_remove -> "
          f"{[str(x) for x in check_pair(lc, [1, 1], [1])] or 'admissible'}")
    print(f"  ([1,1], [1,0]) add_remove -> "
          f"{[str(x) for x in check_pair(lc, [1, 1], [1, 0])] or 'admissible'}")
