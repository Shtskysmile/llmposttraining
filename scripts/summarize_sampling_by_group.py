#!/usr/bin/env python3
"""Summarize multi-sample rollout metrics by task group."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def mean(values):
    return statistics.mean(values) if values else 0.0


def summarize(rows, num_samples):
    n = len(rows)
    resolved = [row for row in rows if not row["majority_vote_no_consensus"]]
    tool_calls = sum(row["tool_calls"] for row in rows)
    successful_calls = sum(row["successful_tool_calls"] for row in rows)
    return {
        "n": n,
        "single_sample_accuracy_with_budget_split": mean(
            [correct for row in rows for correct in row["sample_correct"]]),
        "pass_at_k": mean([row["pass_at_k"] for row in rows]),
        "strict_majority_accuracy_all_questions": mean(
            [row["majority_vote_correct"] for row in rows]),
        "strict_majority_coverage": len(resolved) / n if n else 0.0,
        "strict_majority_accuracy_when_resolved": mean(
            [row["majority_vote_correct"] for row in resolved]),
        "no_consensus_rate": mean([row["majority_vote_no_consensus"] for row in rows]),
        "no_parseable_vote_rate": mean(
            [row["parseable_answer_count"] == 0 for row in rows]),
        "mean_generated_tokens_per_question": mean([row["generated_tokens"] for row in rows]),
        "mean_tool_calls_per_question": mean([row["tool_calls"] for row in rows]),
        "tool_execution_success_rate": successful_calls / tool_calls if tool_calls else None,
        "truncated_sample_rate": sum(row["truncated_samples"] for row in rows) /
                                 (n * num_samples) if n else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--group-field", default="kind")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    config_path = input_path.with_suffix(".run_config.json")
    summary_path = input_path.with_suffix(".summary.json")
    if not config_path.is_file() or not summary_path.is_file():
        raise SystemExit("input requires its .run_config.json and .summary.json files")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    num_samples = config.get("num_samples")
    if not rows or len(rows) != config.get("n") or not isinstance(num_samples, int):
        raise SystemExit("run rows do not match the sampling configuration")
    if summary.get("n") != len(rows) or summary.get("num_samples") != num_samples:
        raise SystemExit("run summary does not match the sampling configuration")

    groups = defaultdict(list)
    for row in rows:
        if args.group_field not in row:
            raise SystemExit(f"missing group field {args.group_field!r} for {row.get('id')}")
        if len(row.get("sample_correct", [])) != num_samples or len(row.get("samples", [])) != num_samples:
            raise SystemExit(f"incomplete sample group for {row.get('id')}")
        groups[str(row[args.group_field])].append(row)

    result = {
        "input": str(input_path),
        "group_field": args.group_field,
        "data_sha256": config.get("data_sha256"),
        "num_samples": num_samples,
        "total_token_budget_per_question": config.get("max_total_tokens"),
        "by_group": {group: summarize(group_rows, num_samples)
                     for group, group_rows in sorted(groups.items())},
        "interpretation_scope": (
            "descriptive subgroup estimates for one run; they do not estimate sampling-seed or "
            "training-seed variation."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
