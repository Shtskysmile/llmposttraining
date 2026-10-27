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


def paired_metric(baseline, candidate, ids, field, rng, bootstrap_samples):
    baseline_values = [float(baseline[qid][field]) for qid in ids]
    candidate_values = [float(candidate[qid][field]) for qid in ids]
    differences = [new - old for old, new in zip(baseline_values, candidate_values)]
    bootstrap = [sum(differences[rng.randrange(len(ids))] for _ in ids) / len(ids)
                 for _ in range(bootstrap_samples)]
    return {
        "baseline_mean_per_question": sum(baseline_values) / len(ids),
        "candidate_mean_per_question": sum(candidate_values) / len(ids),
        "paired_difference_candidate_minus_baseline": sum(differences) / len(ids),
        "question_bootstrap_95_percentile_ci": [
            quantile(bootstrap, 0.025), quantile(bootstrap, 0.975)],
        "bootstrap_samples": bootstrap_samples,
    }


def paired_tool_success_rate(baseline, candidate, ids, rng, bootstrap_samples):
    def rate(rows, sampled_ids):
        calls = sum(float(rows[qid]["tool_calls"]) for qid in sampled_ids)
        successes = sum(float(rows[qid]["successful_tool_calls"])
                        for qid in sampled_ids)
        return successes / calls if calls else None

    baseline_rate = rate(baseline, ids)
    candidate_rate = rate(candidate, ids)
    bootstrap = []
    for _ in range(bootstrap_samples):
        sampled_ids = [ids[rng.randrange(len(ids))] for _ in ids]
        old_rate, new_rate = rate(baseline, sampled_ids), rate(candidate, sampled_ids)
        if old_rate is not None and new_rate is not None:
            bootstrap.append(new_rate - old_rate)
    return {
        "baseline_rate": baseline_rate,
        "candidate_rate": candidate_rate,
        "paired_difference_candidate_minus_baseline": (
            candidate_rate - baseline_rate
            if baseline_rate is not None and candidate_rate is not None else None),
        "question_bootstrap_95_percentile_ci": (
            [quantile(bootstrap, 0.025), quantile(bootstrap, 0.975)] if bootstrap else None),
        "valid_bootstrap_samples": len(bootstrap),
        "bootstrap_samples_requested": bootstrap_samples,
        "denominator": "successful tool calls / all tool calls",
    }


def compare_group(baseline, candidate, rng, bootstrap_samples):
    ids = sorted(baseline)
    differences = [int(candidate[qid]["correct"]) - int(baseline[qid]["correct"])
                   for qid in ids]
    baseline_only = sum(value == -1 for value in differences)
    candidate_only = sum(value == 1 for value in differences)
    bootstrap = [sum(differences[rng.randrange(len(ids))] for _ in ids) / len(ids)
                 for _ in range(bootstrap_samples)]
    result = {
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
    optional_metrics = {}
    for field in ("generated_tokens", "tool_calls", "successful_tool_calls",
                  "truncated", "format_error"):
        baseline_presence = [field in baseline[qid] for qid in ids]
        candidate_presence = [field in candidate[qid] for qid in ids]
        if any(baseline_presence) and not all(baseline_presence):
            raise SystemExit(f"metric {field!r} is incomplete in the baseline group")
        if any(candidate_presence) and not all(candidate_presence):
            raise SystemExit(f"metric {field!r} is incomplete in the candidate group")
        baseline_has, candidate_has = all(baseline_presence), all(candidate_presence)
        if baseline_has != candidate_has:
            raise SystemExit(f"metric {field!r} is missing from one paired group")
        if baseline_has:
            metric_name = {"truncated": "truncated_rate",
                           "format_error": "format_error_rate"}.get(field, field)
            optional_metrics[metric_name] = paired_metric(
                baseline, candidate, ids, field, rng, bootstrap_samples)
    baseline_call_fields = [
        "tool_calls" in baseline[qid] and "successful_tool_calls" in baseline[qid]
        for qid in ids]
    candidate_call_fields = [
        "tool_calls" in candidate[qid] and "successful_tool_calls" in candidate[qid]
        for qid in ids]
    if any(baseline_call_fields) and not all(baseline_call_fields):
        raise SystemExit("tool success fields are incomplete in the baseline group")
    if any(candidate_call_fields) and not all(candidate_call_fields):
        raise SystemExit("tool success fields are incomplete in the candidate group")
    if all(baseline_call_fields) != all(candidate_call_fields):
        raise SystemExit("tool success fields are missing from one paired group")
    if all(baseline_call_fields):
        optional_metrics["tool_execution_success_rate"] = paired_tool_success_rate(
            baseline, candidate, ids, rng, bootstrap_samples)
    result["paired_auxiliary_metrics"] = optional_metrics
    return result


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
