#!/usr/bin/env python3
"""Paired question-level analysis for token-budgeted multi-sample runs."""
import argparse
from collections import Counter
import json
import math
import random
import statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.verifier import extract_answer, score


PAIRING_KEYS = (
    "data_sha256", "n", "model", "mode", "revision", "seed", "num_samples",
    "max_total_tokens", "max_new_tokens", "max_tool_calls", "temperature", "top_p",
    "top_k", "per_sample_token_budgets", "majority_vote_rule", "unparseable_answers_in_vote",
    "sampling", "seed_strategy", "gpu",
)


def load_run(path):
    path = Path(path)
    config_path = path.with_suffix(".run_config.json")
    if not config_path.is_file():
        raise SystemExit(f"missing run config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise SystemExit(f"duplicate question ids in {path}")
    if len(rows) != config.get("n"):
        raise SystemExit(f"run row count does not match config in {path}")
    return indexed, config


def metric_value(row, metric):
    if metric == "sample_accuracy":
        values = row.get("sample_correct")
        if not isinstance(values, list) or not values:
            raise SystemExit("sample_accuracy requires nonempty sample_correct arrays")
        if any(value not in (0, 1, False, True) for value in values):
            raise SystemExit("sample_correct entries must be binary")
        return statistics.mean(values)
    if metric == "majority_vote_coverage":
        return int(not row["majority_vote_no_consensus"])
    if metric in ("plurality_min2_accuracy", "plurality_min2_coverage"):
        counts = Counter(
            str(answer)
            for sample in row["samples"]
            if (answer := extract_answer(sample["completion"])) is not None
        )
        selected = None
        if counts:
            highest = max(counts.values())
            winners = [answer for answer, count in counts.items() if count == highest]
            if highest >= 2 and len(winners) == 1:
                selected = winners[0]
        if metric == "plurality_min2_coverage":
            return int(selected is not None)
        return int(selected is not None and
                   score(f"<answer>{selected}</answer>", row["answer"]))
    if metric in ("generated_tokens", "tool_calls", "successful_tool_calls"):
        value = row.get(metric)
        if not isinstance(value, (int, float)) or value < 0:
            raise SystemExit(f"{metric} requires a nonnegative numeric row field")
        return float(value)
    if metric == "truncated_rate":
        samples = row.get("samples", [])
        count = row.get("truncated_samples")
        if not samples or not isinstance(count, int) or not 0 <= count <= len(samples):
            raise SystemExit("truncated_rate requires a valid truncated_samples count")
        return count / len(samples)
    field = "pass_at_k" if metric == "pass_at_k" else "majority_vote_correct"
    value = row.get(field)
    if value not in (0, 1, False, True):
        raise SystemExit(f"{metric} requires a binary {field} field")
    return int(value)


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
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def compare_metric(baseline, candidate, ids, metric, rng, bootstrap_samples):
    baseline_values = [metric_value(baseline[key], metric) for key in ids]
    candidate_values = [metric_value(candidate[key], metric) for key in ids]
    differences = [new - old for old, new in zip(baseline_values, candidate_values)]
    boot = [statistics.mean(differences[rng.randrange(len(ids))] for _ in ids)
            for _ in range(bootstrap_samples)]
    result = {
        "n": len(ids),
        "baseline": statistics.mean(baseline_values),
        "candidate": statistics.mean(candidate_values),
        "paired_difference": statistics.mean(differences),
        "bootstrap_95_percentile_ci": [quantile(boot, 0.025), quantile(boot, 0.975)],
        "bootstrap_samples": bootstrap_samples,
    }
    if metric in ("pass_at_k", "majority_vote_accuracy", "majority_vote_coverage",
                  "plurality_min2_accuracy", "plurality_min2_coverage"):
        baseline_only = sum(value == -1 for value in differences)
        candidate_only = sum(value == 1 for value in differences)
        event_name = "covered" if metric.endswith("coverage") else "correct"
        result.update({
            f"baseline_only_{event_name}": baseline_only,
            f"candidate_only_{event_name}": candidate_only,
            "exact_mcnemar_two_sided_p": exact_mcnemar(candidate_only, baseline_only),
        })
    return result


def tool_success_rate(rows, ids):
    calls = sum(rows[question_id]["tool_calls"] for question_id in ids)
    successes = sum(rows[question_id]["successful_tool_calls"] for question_id in ids)
    return successes / calls if calls else None


def compare_tool_success_rate(baseline, candidate, ids, rng, bootstrap_samples):
    baseline_rate = tool_success_rate(baseline, ids)
    candidate_rate = tool_success_rate(candidate, ids)
    bootstrap_differences = []
    for _ in range(bootstrap_samples):
        sampled_ids = [ids[rng.randrange(len(ids))] for _ in ids]
        old_rate = tool_success_rate(baseline, sampled_ids)
        new_rate = tool_success_rate(candidate, sampled_ids)
        if old_rate is not None and new_rate is not None:
            bootstrap_differences.append(new_rate - old_rate)
    return {
        "baseline_rate": baseline_rate,
        "candidate_rate": candidate_rate,
        "paired_difference": (candidate_rate - baseline_rate
                              if baseline_rate is not None and candidate_rate is not None else None),
        "bootstrap_95_percentile_ci": (
            [quantile(bootstrap_differences, 0.025),
             quantile(bootstrap_differences, 0.975)] if bootstrap_differences else None
        ),
        "valid_bootstrap_samples": len(bootstrap_differences),
        "bootstrap_samples_requested": bootstrap_samples,
        "denominator": "successful tool calls / all tool calls",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, default=5090)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("bootstrap-samples must be positive")

    baseline, baseline_config = load_run(args.baseline)
    candidate, candidate_config = load_run(args.candidate)
    mismatches = [key for key in PAIRING_KEYS
                  if baseline_config.get(key) != candidate_config.get(key)]
    if mismatches:
        raise SystemExit("runs are not budget/data matched for: " + ", ".join(mismatches))
    if baseline.keys() != candidate.keys():
        raise SystemExit("paired comparison requires identical question id sets")
    ids = sorted(baseline)
    sample_counts = {
        len(row.get("sample_correct", []))
        for dataset in (baseline, candidate) for row in dataset.values()
    }
    if sample_counts != {baseline_config["num_samples"]}:
        raise SystemExit("sample_correct counts do not match the configured sample count")
    output_sample_counts = {
        len(row.get("samples", []))
        for dataset in (baseline, candidate) for row in dataset.values()
    }
    if output_sample_counts != {baseline_config["num_samples"]}:
        raise SystemExit("sample records do not match the configured sample count")

    rng = random.Random(args.seed)
    metrics = {}
    for metric in ("sample_accuracy", "pass_at_k", "majority_vote_accuracy",
                   "majority_vote_coverage", "plurality_min2_accuracy",
                   "plurality_min2_coverage", "generated_tokens", "tool_calls",
                   "successful_tool_calls", "truncated_rate"):
        metrics[metric] = compare_metric(
            baseline, candidate, ids, metric, rng, args.bootstrap_samples)
    metrics["tool_execution_success_rate"] = compare_tool_success_rate(
        baseline, candidate, ids, rng, args.bootstrap_samples)
    result = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "paired_configuration": {key: baseline_config[key] for key in PAIRING_KEYS},
        "bootstrap_seed": args.seed,
        "additional_voting_policy": {
            "name": "unique plurality with at least two votes",
            "minimum_votes": 2,
            "top_answer_ties": "abstain",
            "unparseable_samples": "do not vote",
        },
        "metrics": metrics,
        "interpretation_scope": (
            "question-level paired uncertainty only; one rollout seed and one training seed "
            "per model, so stochastic sampling and training-seed variation are not estimated"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
