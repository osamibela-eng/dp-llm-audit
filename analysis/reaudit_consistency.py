"""Did re-auditing the original programs reproduce their original outcomes?

The corpus was expanded from 240 to 480 programs and the whole thing re-audited,
not just the new half, so that every number in the paper comes from one pass with
one version of the auditor.

IMPORTANT, and corrected after this script first ran: the audit is **not**
deterministic across runs. `aggregate.py --seed` defaults to None, which seeds
numpy from OS entropy, so each campaign draws fresh randomness. This comparison
is therefore a RE-MEASUREMENT, not a reproduction, and near-total agreement is
the expected result rather than a guaranteed one.

That makes it more useful, not less. Programs whose outcome flips between two
independent draws are by definition marginal detections, and marginal detections
are exactly what the confirmation pass and the campaign error bound are about. A
flip here should show up as `one_off` in confirm_seeds; if it does not, the two
mechanisms disagree and that is worth knowing.

Reproducibility in this pipeline comes from the confirmation pass, not from
seed-fixing. Pinning a seed would make the numbers repeatable while concealing
the very instability the confirmation pass exists to measure.

    python analysis/reaudit_consistency.py
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "processed"


def load(path):
    d = {}
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            d[(r["model"], r["task_id"], r["sample_idx"])] = r
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["qwen", "deepseek", "codellama"])
    a = ap.parse_args()

    print("=" * 74)
    print("RE-AUDIT CONSISTENCY  (240-program corpus vs the same programs in 480)")
    print("=" * 74)

    total_same = total_diff = total_checked = 0
    changes = collections.Counter()
    flipped = []          # (code_sha256, key) for each changed outcome

    for tag in a.models:
        new_p = OUT / f"outcomes_{tag}.jsonl"
        old_p = OUT / f"outcomes_{tag}.jsonl.bak240"
        if not old_p.exists():
            print(f"\n  {tag}: no .bak240 baseline, skipping")
            continue
        if not new_p.exists():
            print(f"\n  {tag}: re-audit has not produced output yet")
            continue

        old, new = load(old_p), load(new_p)
        shared = set(old) & set(new)
        if not shared:
            print(f"\n  {tag}: no shared (model, task, sample) keys")
            continue

        # If the "new" file is still the same size as its own baseline, the
        # re-audit has not written yet and we are comparing a file to a copy of
        # itself. That returns a perfect score and means nothing. Refuse to
        # report it: a check that passes because it was never run is worse than
        # no check, and this project has already been bitten once by a
        # cross-check that shared the assumption it was meant to test.
        if len(new) == len(old):
            print(f"\n  {tag}: SKIPPED -- the outcomes file still has {len(new)} "
                  f"rows, the same as its baseline.")
            print("    The re-audit has not landed. Comparing now would compare "
                  "the\n    file against a copy of itself and report 100%.")
            continue

        same = sum(1 for k in shared if old[k]["outcome"] == new[k]["outcome"])
        diff = len(shared) - same
        total_same += same
        total_diff += diff
        total_checked += len(shared)

        print(f"\n  {tag}: {len(new)} programs now, {len(old)} before, "
              f"{len(shared)} comparable")
        print(f"    identical outcome: {same}/{len(shared)}")
        if diff:
            print(f"    CHANGED          : {diff}")
            for k in sorted(shared):
                if old[k]["outcome"] != new[k]["outcome"]:
                    changes[(old[k]["outcome"], new[k]["outcome"])] += 1
                    flipped.append((old[k]["code_sha256"],
                                    (old[k]["outcome"], new[k]["outcome"]), k))
                    if diff <= 12:
                        print(f"      {k[1]:28s} s{k[2]}  "
                              f"{old[k]['outcome']} -> {new[k]['outcome']}")

    print("\n" + "=" * 74)
    if total_checked == 0:
        print("  nothing to compare yet")
        return
    print(f"  {total_same}/{total_checked} outcomes reproduced exactly "
          f"({total_same / total_checked:.1%})")
    if changes:
        print("\n  transitions:")
        for (o, n), c in changes.most_common():
            print(f"    {o:22s} -> {n:22s} {c}")

        # A flip should be a marginal detection, and the confirmation pass
        # should already have said so. Check that rather than asserting it.
        conf_p = OUT / "confirm_seeds.jsonl"
        if conf_p.exists() and flipped:
            conf = {}
            for line in conf_p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    conf[d["code_sha256"]] = d
            print("\n  cross-check against the fresh-seed confirmation pass:")
            print("  (only flips OUT of `falsified` can be checked: the")
            print("   confirmation pass only ever ran on detected programs)")
            agree = checkable = 0
            for sha, (o_out, n_out), key in flipped:
                d = conf.get(sha)
                if o_out != "falsified":
                    # Never a detection in the first pass, so the confirmation
                    # pass had no reason to look at it. Saying it is "absent
                    # from confirm_seeds" as though that were informative would
                    # be misleading; it is simply out of scope.
                    print(f"    {key[1]:26s} {o_out} -> {n_out}: out of scope "
                          f"(was not a detection before)")
                    continue
                checkable += 1
                if d is None:
                    print(f"    {key[1]:26s} was falsified but is absent from "
                          f"confirm_seeds")
                else:
                    print(f"    {key[1]:26s} classified {d['classification']}, "
                          f"falsified in {d['falsified_in']}/{d['attempts']}")
                    if d["classification"] in ("one_off", "borderline"):
                        agree += 1
            if checkable:
                print(f"\n    {agree}/{checkable} lost detections were already "
                      f"flagged as non-reproducible.")
                print("    Two mechanisms sharing no randomness agreeing on which")
                print("    detections are unstable supports the campaign error")
                print("    analysis; a lost detection classified 'reproducible'")
                print("    would contradict it and needs investigating.")
            gained = [f for f in flipped if f[1][1] == "falsified"]
            if gained:
                print(f"\n    {len(gained)} program(s) became falsified that were "
                      f"not before.")
                print("    These are NEW detections and must go through the")
                print("    confirmation pass before they count. Re-run")
                print("    analysis/confirm_seeds.py, which resumes.")
    else:
        print("\n  No outcome changed, across independent random draws.")


if __name__ == "__main__":
    main()
