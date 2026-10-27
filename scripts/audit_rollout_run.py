#!/usr/bin/env python3
"""Audit provenance, per-question records, budgets, and summary for a rollout run."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.verifier import format_error, score


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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_close(actual, expected, message):
    require(isinstance(actual, (int, float)) and abs(actual - expected) <= 1e-12,
            f"{message}: got {actual}, expected {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="src/rollout.py JSONL output")
    parser.add_argument("--require-complete", action="store_true",
                        help="require all configured rows and a consistent summary")
    args = parser.parse_args()

    output = Path(args.input)
    config_path = output.with_suffix(".run_config.json")
    summary_path = output.with_suffix(".summary.json")
    require(output.is_file(), f"missing output: {output}")
    require(config_path.is_file(), f"missing run config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = ("data", "data_sha256", "n", "mode", "decoding",
                "max_total_tokens_per_question", "max_tool_calls")
    missing = [key for key in required if key not in config]
    require(not missing, f"run config is missing required fields: {missing}")
    require(config["n"] > 0, "run config has no questions")

    data_path = Path(config["data"])
    require(data_path.is_file(), f"missing source data: {data_path}")
    raw_data = data_path.read_bytes()
    data_sha = hashlib.sha256(raw_data).hexdigest()
    require(data_sha == config["data_sha256"], "source data hash does not match run config")
    source_rows = read_jsonl(data_path)
    require(config["n"] <= len(source_rows), "configured row count exceeds source data")
    source_rows = source_rows[:config["n"]]
    source_ids = [row["id"] for row in source_rows]
    require(len(set(source_ids)) == len(source_ids), "source data contains duplicate IDs")

    rows = read_jsonl(output)
    require(len(rows) <= config["n"], "output contains more rows than configured")
    output_ids = [row.get("id") for row in rows]
    require(output_ids == source_ids[:len(rows)],
            "output must contain a prefix of source IDs in original order")

    max_total = config["max_total_tokens_per_question"]
    max_calls = config["max_tool_calls"]
    for index, (row, source) in enumerate(zip(rows, source_rows)):
        for key, expected in source.items():
            require(row.get(key) == expected,
                    f"row {index} ({row.get('id')}): source field {key!r} changed")
        completion = row.get("completion")
        require(isinstance(completion, str), f"row {index}: missing completion text")
        require(row.get("correct") == score(completion, source["answer"]),
                f"row {index}: verifier score mismatch")
        require(row.get("format_error") == format_error(completion),
                f"row {index}: format-error label mismatch")

        tokens = row.get("generated_tokens")
        calls = row.get("tool_calls")
        successes = row.get("successful_tool_calls")
        require(isinstance(tokens, int) and 0 <= tokens <= max_total,
                f"row {index}: generated-token budget exceeded")
        require(isinstance(calls, int) and 0 <= calls <= max_calls,
                f"row {index}: tool-call budget exceeded")
        require(isinstance(successes, int) and 0 <= successes <= calls,
                f"row {index}: invalid successful-tool-call count")
        require(isinstance(row.get("truncated"), bool), f"row {index}: invalid truncation flag")
        events = row.get("tool_events")
        require(isinstance(events, list) and len(events) == calls,
                f"row {index}: tool-event count mismatch")
        event_successes = 0
        for event in events:
            require(isinstance(event, dict) and isinstance(event.get("result"), dict) and
                    isinstance(event["result"].get("ok"), bool),
                    f"row {index}: malformed tool event")
            event_successes += int(event["result"]["ok"])
        require(event_successes == successes, f"row {index}: tool-event success count mismatch")

    complete = len(rows) == config["n"]
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        require(complete, "summary exists but JSONL output is incomplete")
        for key in ("model", "base_model", "revision", "data", "data_sha256", "data_manifest",
                    "mode", "seed", "decoding", "max_new_tokens_per_turn",
                    "max_total_tokens_per_question", "max_tool_calls", "n"):
            require(summary.get(key) == config.get(key),
                    f"summary metadata mismatch for {key}")
        require_close(summary.get("accuracy"), statistics.mean(row["correct"] for row in rows),
                      "summary accuracy mismatch")
        require_close(summary.get("format_error_rate"),
                      statistics.mean(row["format_error"] for row in rows),
                      "summary format-error rate mismatch")
        require_close(summary.get("mean_generated_tokens"),
                      statistics.mean(row["generated_tokens"] for row in rows),
                      "summary generated-token mean mismatch")
        require_close(summary.get("mean_tool_calls"),
                      statistics.mean(row["tool_calls"] for row in rows),
                      "summary tool-call mean mismatch")
        total_calls = sum(row["tool_calls"] for row in rows)
        successes = sum(row["successful_tool_calls"] for row in rows)
        require_close(summary.get("tool_execution_success_rate"),
                      successes / max(1, total_calls),
                      "summary tool-success rate mismatch")
        require_close(summary.get("truncated_rate"),
                      statistics.mean(row["truncated"] for row in rows),
                      "summary truncation rate mismatch")

    require(not args.require_complete or complete,
            f"run is incomplete: {len(rows)}/{config['n']} rows")
    require(not args.require_complete or summary_path.is_file(),
            "complete audit requires a summary JSON")
    result = {
        "input": str(output),
        "run_config": str(config_path),
        "data_sha256": data_sha,
        "rows_present": len(rows),
        "rows_expected": config["n"],
        "complete": complete,
        "summary_present": summary_path.is_file(),
        "accuracy": statistics.mean(row["correct"] for row in rows) if rows else None,
        "total_generated_tokens": sum(row["generated_tokens"] for row in rows),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"rollout-run audit failed: {exc}") from exc
