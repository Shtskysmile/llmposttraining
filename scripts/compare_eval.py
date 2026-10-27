#!/usr/bin/env python3
"""Compare paired accuracy and inference-cost metrics with bootstrap confidence intervals."""
import argparse
import json
import math
import random
from pathlib import Path


def read_run(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise SystemExit(f"duplicate ids in {path}")
    if not indexed:
        raise SystemExit(f"evaluation run is empty: {path}")
    if any(row.get("correct") not in (0, 1, False, True) for row in rows):
        raise SystemExit(f"non-binary correctness value in {path}")
    return indexed


def exact_mcnemar(candidate_only, baseline_only):
    discordant = candidate_only + baseline_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(candidate_only, baseline_only) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def paired_metric(baseline, candidate, ids, field, rng, bootstrap_samples):
    baseline_values = [float(baseline[question_id][field]) for question_id in ids]
    candidate_values = [float(candidate[question_id][field]) for question_id in ids]
    differences = [new - old for old, new in zip(baseline_values, candidate_values)]
    bootstrap = [
        sum(differences[rng.randrange(len(ids))] for _ in ids) / len(ids)
        for _ in range(bootstrap_samples)
    ]
    return {
        "baseline_mean_per_question": sum(baseline_values) / len(ids),
        "candidate_mean_per_question": sum(candidate_values) / len(ids),
        "paired_difference_candidate_minus_baseline": sum(differences) / len(ids),
        "question_bootstrap_95_percentile_ci": [
            quantile(bootstrap, 0.025), quantile(bootstrap, 0.975)],
        "bootstrap_samples": bootstrap_samples,
    }


def tool_success_rate(rows, ids):
    calls = sum(float(rows[question_id]["tool_calls"]) for question_id in ids)
    successes = sum(float(rows[question_id]["successful_tool_calls"])
                    for question_id in ids)
    return successes / calls if calls else None


def paired_tool_success_rate(baseline, candidate, ids, rng, bootstrap_samples):
    baseline_rate = tool_success_rate(baseline, ids)
    candidate_rate = tool_success_rate(candidate, ids)
    bootstrap_differences = []
    for _ in range(bootstrap_samples):
        sampled_ids = [ids[rng.randrange(len(ids))] for _ in ids]
        old_rate = tool_success_rate(baseline, sampled_ids)
        new_rate = tool_success_rate(candidate, sampled_ids)
        if old_rate is not None and new_rate is not None:
            bootstrap_differences.append(new_rate - old_rate)
    difference = (candidate_rate - baseline_rate
                  if baseline_rate is not None and candidate_rate is not None else None)
    return {
        "baseline_rate": baseline_rate,
        "candidate_rate": candidate_rate,
        "paired_difference_candidate_minus_baseline": difference,
        "question_bootstrap_95_percentile_ci": (
            [quantile(bootstrap_differences, 0.025),
             quantile(bootstrap_differences, 0.975)] if bootstrap_differences else None
        ),
        "valid_bootstrap_samples": len(bootstrap_differences),
        "bootstrap_samples_requested": bootstrap_samples,
        "denominator": "successful tool calls / all tool calls",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, default=5090)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("bootstrap-samples must be positive")
    baseline = read_run(args.baseline)
    candidate = read_run(args.candidate)
    if baseline.keys() != candidate.keys():
        raise SystemExit("paired comparison requires identical question id sets")
    ids = sorted(baseline)
    differences = [int(candidate[key]["correct"]) - int(baseline[key]["correct"])
                  for key in ids]
    baseline_only = sum(value == -1 for value in differences)
    candidate_only = sum(value == 1 for value in differences)
    rng = random.Random(args.seed)
    boot = []
    for _ in range(args.bootstrap_samples):
        boot.append(sum(differences[rng.randrange(len(differences))] for _ in ids) / len(ids))
    result = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "n": len(ids),
        "baseline_accuracy": sum(int(baseline[key]["correct"]) for key in ids) / len(ids),
        "candidate_accuracy": sum(int(candidate[key]["correct"]) for key in ids) / len(ids),
        "paired_accuracy_difference": sum(differences) / len(ids),
        "bootstrap_95_percentile_ci": [quantile(boot, 0.025), quantile(boot, 0.975)],
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "exact_mcnemar_two_sided_p": exact_mcnemar(candidate_only, baseline_only),
        "interpretation_scope": "question-sampling uncertainty only; excludes training-seed variation",
    }
    optional_metrics = {}
    for field in ("generated_tokens", "tool_calls", "successful_tool_calls",
                  "truncated", "format_error"):
        baseline_presence = [field in baseline[key] for key in ids]
        candidate_presence = [field in candidate[key] for key in ids]
        if any(baseline_presence) and not all(baseline_presence):
            raise SystemExit(f"metric {field!r} is incomplete in the baseline run")
        if any(candidate_presence) and not all(candidate_presence):
            raise SystemExit(f"metric {field!r} is incomplete in the candidate run")
        baseline_has = all(baseline_presence)
        candidate_has = all(candidate_presence)
        if baseline_has != candidate_has:
            raise SystemExit(f"metric {field!r} is missing from one paired run")
        if baseline_has:
            metric_name = {"truncated": "truncated_rate",
                           "format_error": "format_error_rate"}.get(field, field)
            optional_metrics[metric_name] = paired_metric(
                baseline, candidate, ids, field, rng, args.bootstrap_samples)
    baseline_call_presence = [
        "tool_calls" in baseline[key] and "successful_tool_calls" in baseline[key]
        for key in ids
    ]
    candidate_call_presence = [
        "tool_calls" in candidate[key] and "successful_tool_calls" in candidate[key]
        for key in ids
    ]
    if any(baseline_call_presence) and not all(baseline_call_presence):
        raise SystemExit("tool-success fields are incomplete in the baseline run")
    if any(candidate_call_presence) and not all(candidate_call_presence):
        raise SystemExit("tool-success fields are incomplete in the candidate run")
    has_call_fields = all(baseline_call_presence)
    candidate_has_call_fields = all(candidate_call_presence)
    if has_call_fields != candidate_has_call_fields:
        raise SystemExit("tool-success fields are missing from one paired run")
    if has_call_fields:
        optional_metrics["tool_execution_success_rate"] = paired_tool_success_rate(
            baseline, candidate, ids, rng, args.bootstrap_samples)
    result["paired_auxiliary_metrics"] = optional_metrics
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
