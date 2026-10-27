#!/usr/bin/env python3
"""Audit provenance, row integrity, and budgets for a multi-sample rollout run."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.verifier import extract_answer, score


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def vote_key(completion):
    answer = extract_answer(completion)
    return str(answer) if answer is not None else None


def expected_vote(completions):
    counts = Counter(key for completion in completions
                     if (key := vote_key(completion)) is not None)
    if not counts:
        return None, 0, True
    maximum = max(counts.values())
    winners = [key for key, count in counts.items() if count == maximum]
    if len(winners) != 1 or maximum <= len(completions) / 2:
        return None, maximum, True
    return winners[0], maximum, False


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_close(actual, expected, message):
    require(isinstance(actual, (int, float)) and abs(actual - expected) <= 1e-12,
            f"{message}: got {actual}, expected {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="evaluate_sampling.py JSONL output")
    parser.add_argument("--require-complete", action="store_true",
                        help="fail unless all configured dataset rows are present")
    args = parser.parse_args()

    output = Path(args.input)
    config_path = output.with_suffix(".run_config.json")
    require(output.is_file(), f"missing output: {output}")
    require(config_path.is_file(), f"missing run config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required_config = ("data", "data_sha256", "n", "seed", "num_samples",
                       "max_total_tokens", "max_tool_calls", "per_sample_token_budgets",
                       "seed_strategy", "majority_vote_rule")
    missing = [key for key in required_config if key not in config]
    require(not missing, f"run config is missing required fields: {missing}")

    data_path = Path(config["data"])
    require(data_path.is_file(), f"missing source data: {data_path}")
    raw_data = data_path.read_bytes()
    data_sha = hashlib.sha256(raw_data).hexdigest()
    require(data_sha == config["data_sha256"], "source data SHA-256 does not match run config")
    source = read_jsonl(data_path)
    if config.get("limit") is not None:
        source = source[:config["limit"]]
    require(len(source) == config["n"],
            f"configured row count {config['n']} does not match source rows {len(source)}")
    source_ids = [row["id"] for row in source]
    require(len(set(source_ids)) == len(source_ids), "source data has duplicate IDs")

    rows = read_jsonl(output)
    require(len(rows) <= len(source), "output has more rows than configured source data")
    output_ids = [row.get("id") for row in rows]
    require(output_ids == source_ids[:len(rows)],
            "output must contain a prefix of source IDs in original order")

    sample_count = config["num_samples"]
    sample_budgets = config["per_sample_token_budgets"]
    require(len(sample_budgets) == sample_count,
            "per-sample token-budget list length does not match num_samples")
    max_total = config["max_total_tokens"]
    max_calls = config["max_tool_calls"]
    for index, (row, reference) in enumerate(zip(rows, source)):
        for key, expected in reference.items():
            require(row.get(key) == expected,
                    f"row {index} ({row.get('id')}): source field {key!r} changed")
        expected_seed = (config["seed"] + index) % (2 ** 32)
        require(row.get("sampling_seed") == expected_seed,
                f"row {index}: sampling seed {row.get('sampling_seed')} != {expected_seed}")

        samples = row.get("samples", [])
        correct = row.get("sample_correct", [])
        require(len(samples) == sample_count and len(correct) == sample_count,
                f"row {index}: incomplete sample group")
        completions = [sample.get("completion", "") for sample in samples]
        recomputed = [int(score(completion, row["answer"])) for completion in completions]
        require(correct == recomputed, f"row {index}: sample correctness labels do not match verifier")
        require(row.get("pass_at_k") == int(any(recomputed)),
                f"row {index}: pass@k does not match sample correctness")

        for sample_index, sample in enumerate(samples):
            tokens = sample.get("generated_tokens")
            calls = sample.get("tool_calls")
            successes = sample.get("successful_tool_calls")
            require(isinstance(tokens, int) and 0 <= tokens <= sample_budgets[sample_index],
                    f"row {index}, sample {sample_index}: generated-token budget exceeded")
            require(isinstance(calls, int) and 0 <= calls <= max_calls,
                    f"row {index}, sample {sample_index}: tool-call budget exceeded")
            require(isinstance(successes, int) and 0 <= successes <= calls,
                    f"row {index}, sample {sample_index}: invalid successful tool-call count")
            require(isinstance(sample.get("truncated"), bool),
                    f"row {index}, sample {sample_index}: truncated must be boolean")

        require(row.get("generated_tokens") == sum(s["generated_tokens"] for s in samples),
                f"row {index}: aggregate generated-token count mismatch")
        require(row["generated_tokens"] <= max_total,
                f"row {index}: total generated-token budget exceeded")
        require(row.get("tool_calls") == sum(s["tool_calls"] for s in samples),
                f"row {index}: aggregate tool-call count mismatch")
        require(row.get("successful_tool_calls") ==
                sum(s["successful_tool_calls"] for s in samples),
                f"row {index}: aggregate successful-tool-call count mismatch")
        require(row.get("truncated_samples") == sum(s["truncated"] for s in samples),
                f"row {index}: aggregate truncated-sample count mismatch")

        winner, vote_count, no_consensus = expected_vote(completions)
        require(row.get("majority_answer") == winner and
                row.get("majority_vote_count") == vote_count and
                row.get("majority_vote_no_consensus") == no_consensus,
                f"row {index}: majority-vote decision mismatch")
        vote_correct = int(winner is not None and
                           score(f"<answer>{winner}</answer>", row["answer"]))
        require(row.get("majority_vote_correct") == vote_correct,
                f"row {index}: majority-vote correctness mismatch")
        parseable = sum(vote_key(completion) is not None for completion in completions)
        require(row.get("parseable_answer_count") == parseable,
                f"row {index}: parseable-answer count mismatch")

    complete = len(rows) == len(source)
    summary_path = output.with_suffix(".summary.json")
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        require(complete, "summary exists but the JSONL output is incomplete")
        require(summary.get("n") == len(rows), "summary row count does not match JSONL output")
        require(summary.get("data_sha256") == data_sha,
                "summary data SHA-256 does not match run config")
        require(summary.get("num_samples") == sample_count,
                "summary sample count does not match run config")
        require(summary.get("seed") == config["seed"],
                "summary seed does not match run config")
        require_close(
            summary.get("single_sample_accuracy_with_budget_split"),
            statistics.mean(value for row in rows for value in row["sample_correct"]),
            "summary single-sample accuracy mismatch",
        )
        require_close(summary.get("pass_at_k"),
                      statistics.mean(row["pass_at_k"] for row in rows),
                      "summary pass@k mismatch")
        require_close(summary.get("majority_vote_accuracy"),
                      statistics.mean(row["majority_vote_correct"] for row in rows),
                      "summary majority-vote accuracy mismatch")
        require_close(summary.get("mean_generated_tokens_per_question"),
                      statistics.mean(row["generated_tokens"] for row in rows),
                      "summary generated-token mean mismatch")
        require_close(summary.get("mean_tool_calls_per_question"),
                      statistics.mean(row["tool_calls"] for row in rows),
                      "summary tool-call mean mismatch")
        total_calls = sum(row["tool_calls"] for row in rows)
        total_successes = sum(row["successful_tool_calls"] for row in rows)
        require_close(summary.get("tool_execution_success_rate"),
                      total_successes / max(1, total_calls),
                      "summary tool-execution success rate mismatch")
        total_truncated = sum(row["truncated_samples"] for row in rows)
        require_close(summary.get("truncated_sample_rate"),
                      total_truncated / (len(rows) * sample_count),
                      "summary truncation rate mismatch")
    require(not args.require_complete or complete,
            f"run is incomplete: {len(rows)}/{len(source)} rows")
    report = {
        "input": str(output),
        "config": str(config_path),
        "data_sha256": data_sha,
        "rows_present": len(rows),
        "rows_expected": len(source),
        "complete": complete,
        "summary_present": summary_path.is_file(),
        "pass_at_k": (sum(row["pass_at_k"] for row in rows) / len(rows)
                      if rows else None),
        "majority_vote_accuracy": (statistics.mean(
            row["majority_vote_correct"] for row in rows) if rows else None),
        "rows_with_truncation": sum(row["truncated_samples"] > 0 for row in rows),
        "total_generated_tokens": sum(row["generated_tokens"] for row in rows),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"sampling-run audit failed: {exc}") from exc
