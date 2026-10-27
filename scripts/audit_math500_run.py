#!/usr/bin/env python3
"""Audit provenance, answer labels, budgets, tool events, and summary for MATH-500 runs."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import evaluate_math500 as math500


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


def grouped_metrics(rows, field):
    groups = {}
    for row in rows:
        groups.setdefault(str(row[field]), []).append(row)
    return {
        key: {
            "n": len(group),
            "accuracy": sum(row["correct"] for row in group) / len(group),
            "prediction_parse_failure_rate": (
                sum(not row["prediction_parseable"] for row in group) / len(group)),
            "mean_generated_tokens": statistics.mean(row["generated_tokens"] for row in group),
        }
        for key, group in sorted(groups.items())
    }


def audit(path, require_complete):
    output = Path(path)
    config_path = output.with_suffix(".run_config.json")
    summary_path = output.with_suffix(".summary.json")
    require(output.is_file(), f"missing output: {output}")
    require(config_path.is_file(), f"missing run config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = ("model", "data", "data_sha256", "n", "mode", "max_turn_tokens",
                "max_total_tokens", "max_tool_calls", "decoding", "manifest")
    missing = [key for key in required if key not in config]
    require(not missing, f"run config is missing required fields: {missing}")
    require(config["n"] > 0, "run config has no questions")
    require(config["mode"] in ("direct", "tool"), "invalid evaluation mode")
    require(config["decoding"] == "greedy", "unexpected decoding configuration")

    data_path = Path(config["data"])
    require(data_path.is_file(), f"missing source data: {data_path}")
    raw_data = data_path.read_bytes()
    data_sha = hashlib.sha256(raw_data).hexdigest()
    require(data_sha == config["data_sha256"], "source data SHA-256 does not match run config")
    manifest_path = data_path.with_suffix(".manifest.json")
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file() else None)
    require(config["manifest"] == manifest, "run-config data manifest differs from source")
    source_rows = read_jsonl(data_path)
    source_rows = source_rows[:config["n"]]
    require(len(source_rows) == config["n"], "configured row count differs from source data")
    source_ids = [row["id"] for row in source_rows]
    require(len(set(source_ids)) == len(source_ids), "source data contains duplicate IDs")

    rows = read_jsonl(output)
    require(len(rows) <= config["n"], "output contains more rows than configured")
    require([row.get("id") for row in rows] == source_ids[:len(rows)],
            "output must be a prefix of source IDs in original order")
    max_total = config["max_total_tokens"]
    max_calls = config["max_tool_calls"]
    require(max_total > 0 and config["max_turn_tokens"] > 0 and max_calls >= 0,
            "invalid generation budget in run config")
    require(config["mode"] == "tool" or max_calls == 0,
            "direct evaluation must configure zero tool calls")

    gold_parse_failures = []
    gold_type_counts = {"math": 0, "string": 0}
    for index, (row, source) in enumerate(zip(rows, source_rows)):
        for key, expected in source.items():
            require(row.get(key) == expected,
                    f"row {index} ({row.get('id')}): source field {key!r} changed")
        completion = row.get("completion")
        require(isinstance(completion, str), f"row {index}: missing completion text")
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

        gold, gold_type = math500.parse_gold(source["answer"])
        require(row.get("gold_type") == gold_type,
                f"row {index}: gold answer type mismatch")
        gold_type_counts[gold_type] += 1
        if not gold:
            gold_parse_failures.append(row["id"])
        if "gold_parseable" in row:
            require(row["gold_parseable"] is bool(gold),
                    f"row {index}: gold parseability label mismatch")
        if not gold:
            predicted = []
        elif gold_type == "string":
            predicted = math500.parse_text_prediction(completion)
        else:
            predicted = math500.parse(
                completion, extraction_config=math500.PREDICTION_EXTRACTION, parsing_timeout=5)
        correct = bool(predicted) and math500.answers_match(gold, predicted)
        require(row.get("prediction_parseable") is bool(predicted),
                f"row {index}: prediction parseability label mismatch")
        require(row.get("correct") == int(correct), f"row {index}: correctness label mismatch")
        boxed = bool(re.search(r"\\boxed\s*\{", completion))
        require(row.get("boxed_answer_present") is boxed,
                f"row {index}: boxed-answer label mismatch")

    complete = len(rows) == config["n"]
    require(not require_complete or complete,
            f"run is incomplete: {len(rows)}/{config['n']} rows")
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        require(complete, "summary exists but JSONL output is incomplete")
        for key in ("model", "adapter", "revision", "data", "data_sha256", "mode",
                    "max_tool_calls", "n", "max_turn_tokens"):
            require(summary.get(key) == config.get(key),
                    f"summary metadata mismatch for {key}")
        require(summary.get("max_total_tokens_per_question") == config["max_total_tokens"],
                "summary total-token budget mismatch")
        require(summary.get("data_manifest") == config["manifest"],
                "summary data manifest mismatch")
        require(summary.get("gold_parse_failures") == gold_parse_failures,
                "summary gold-parse failures mismatch")
        require_close(summary.get("gold_math_answers"),
                      gold_type_counts["math"],
                      "summary math-answer count mismatch")
        require_close(summary.get("gold_string_answers"),
                      gold_type_counts["string"],
                      "summary string-answer count mismatch")
        require_close(summary.get("accuracy"), statistics.mean(row["correct"] for row in rows),
                      "summary accuracy mismatch")
        require(summary.get("accuracy_by_level") == grouped_metrics(rows, "level"),
                "summary level metrics mismatch")
        require(summary.get("accuracy_by_subject") == grouped_metrics(rows, "subject"),
                "summary subject metrics mismatch")
        total_calls = sum(row["tool_calls"] for row in rows)
        total_successes = sum(row["successful_tool_calls"] for row in rows)
        for key, expected in (
            ("prediction_parse_failure_rate", statistics.mean(
                not row["prediction_parseable"] for row in rows)),
            ("boxed_answer_rate", statistics.mean(row["boxed_answer_present"] for row in rows)),
            ("mean_generated_tokens", statistics.mean(row["generated_tokens"] for row in rows)),
            ("mean_tool_calls", statistics.mean(row["tool_calls"] for row in rows)),
            ("successful_tool_calls", total_successes),
            ("tool_execution_success_rate", total_successes / max(1, total_calls)),
            ("truncated_rate", statistics.mean(row["truncated"] for row in rows)),
        ):
            require_close(summary.get(key), expected, f"summary {key} mismatch")
        require(summary.get("math_verify_version") == version("math-verify"),
                "summary Math-Verify version differs from current environment")

    require(not require_complete or summary_path.is_file(),
            "complete audit requires a summary JSON")
    return {
        "input": str(output),
        "run_config": str(config_path),
        "data_sha256": data_sha,
        "rows_present": len(rows),
        "rows_expected": config["n"],
        "complete": complete,
        "summary_present": summary_path.is_file(),
        "accuracy": statistics.mean(row["correct"] for row in rows) if rows else None,
        "total_generated_tokens": sum(row["generated_tokens"] for row in rows),
        "total_tool_calls": sum(row["tool_calls"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="evaluate_math500.py JSONL output")
    parser.add_argument("--require-complete", action="store_true",
                        help="require all configured rows and a consistent summary")
    args = parser.parse_args()
    print(json.dumps(audit(args.input, args.require_complete), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"MATH-500 run audit failed: {exc}") from exc
