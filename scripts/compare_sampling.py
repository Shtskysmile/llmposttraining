#!/usr/bin/env python3
"""Paired question-level analysis for token-budgeted multi-sample runs."""
import argparse
import json
import math
import random
import statistics
from pathlib import Path


PAIRING_KEYS = (
    "data_sha256", "n", "mode", "revision", "seed", "num_samples",
    "max_total_tokens", "max_new_tokens", "max_tool_calls", "temperature", "top_p",
    "per_sample_token_budgets", "majority_vote_rule", "unparseable_answers_in_vote",
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
    if metric in ("pass_at_k", "majority_vote_accuracy"):
        baseline_only = sum(value == -1 for value in differences)
        candidate_only = sum(value == 1 for value in differences)
        result.update({
            "baseline_only_correct": baseline_only,
            "candidate_only_correct": candidate_only,
            "exact_mcnemar_two_sided_p": exact_mcnemar(candidate_only, baseline_only),
        })
    return result


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

    rng = random.Random(args.seed)
    metrics = {}
    for metric in ("sample_accuracy", "pass_at_k", "majority_vote_accuracy"):
        metrics[metric] = compare_metric(
            baseline, candidate, ids, metric, rng, args.bootstrap_samples)
    result = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "paired_configuration": {key: baseline_config[key] for key in PAIRING_KEYS},
        "bootstrap_seed": args.seed,
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
