#!/usr/bin/env python3
"""Compare paired rollout accuracy within task groups."""
import argparse
import json
import math
import random
from pathlib import Path


def load_rows(path, group_field):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    indexed = {}
    for row in rows:
        question_id = row.get("id")
        if question_id is None or question_id in indexed:
            raise SystemExit(f"missing or duplicate question id in {path}")
        if row.get("correct") not in (0, 1, False, True):
            raise SystemExit(f"invalid correctness label for {question_id} in {path}")
        if group_field not in row or row[group_field] is None:
            raise SystemExit(f"missing group field {group_field!r} for {question_id} in {path}")
        indexed[question_id] = row
    if not indexed:
        raise SystemExit(f"no rollout rows in {path}")
    return indexed


def exact_mcnemar(candidate_only, baseline_only):
    discordant = candidate_only + baseline_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k)
               for k in range(min(candidate_only, baseline_only) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def compare_group(baseline, candidate, rng, bootstrap_samples):
    ids = sorted(baseline)
    differences = [int(candidate[qid]["correct"]) - int(baseline[qid]["correct"])
                   for qid in ids]
    baseline_only = sum(value == -1 for value in differences)
    candidate_only = sum(value == 1 for value in differences)
    bootstrap = [sum(differences[rng.randrange(len(ids))] for _ in ids) / len(ids)
                 for _ in range(bootstrap_samples)]
    return {
        "n": len(ids),
        "baseline_accuracy": sum(int(baseline[qid]["correct"]) for qid in ids) / len(ids),
        "candidate_accuracy": sum(int(candidate[qid]["correct"]) for qid in ids) / len(ids),
        "paired_accuracy_difference": sum(differences) / len(ids),
        "bootstrap_95_percentile_ci": [quantile(bootstrap, 0.025), quantile(bootstrap, 0.975)],
        "bootstrap_samples": bootstrap_samples,
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "exact_mcnemar_two_sided_p": exact_mcnemar(candidate_only, baseline_only),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--group-field", default="kind")
    parser.add_argument("--bootstrap-seed", type=int, default=5090)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("bootstrap-samples must be positive")

    baseline = load_rows(args.baseline, args.group_field)
    candidate = load_rows(args.candidate, args.group_field)
    if baseline.keys() != candidate.keys():
        raise SystemExit("paired comparison requires identical question id sets")
    for question_id in baseline:
        for field in ("prompt", "answer"):
            if baseline[question_id].get(field) != candidate[question_id].get(field):
                raise SystemExit(f"{field} differs for question {question_id}")
        if baseline[question_id][args.group_field] != candidate[question_id][args.group_field]:
            raise SystemExit(f"group assignment differs for question {question_id}")

    groups = sorted({str(row[args.group_field]) for row in baseline.values()})
    rng = random.Random(args.bootstrap_seed)
    by_group = {}
    for group in groups:
        group_ids = [qid for qid in sorted(baseline)
                     if str(baseline[qid][args.group_field]) == group]
        base_rows = {qid: baseline[qid] for qid in group_ids}
        candidate_rows = {qid: candidate[qid] for qid in group_ids}
        by_group[group] = compare_group(base_rows, candidate_rows, rng, args.bootstrap_samples)

    result = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "group_field": args.group_field,
        "n": len(baseline),
        "bootstrap_seed": args.bootstrap_seed,
        "by_group": by_group,
        "interpretation_scope": (
            "group-wise intervals estimate question-sampling uncertainty only; they do not "
            "estimate training-seed variation, and multiple group comparisons are exploratory."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
