#!/usr/bin/env python3
"""Summarize descriptive accuracy and generation-cost subgroups in a MATH-500 run."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


Z_95 = 1.959963984540054


def wilson_interval(successes, n):
    if n == 0:
        return [None, None]
    p = successes / n
    z2 = Z_95 * Z_95
    denominator = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denominator
    radius = Z_95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return [center - radius, center + radius]


def summarize(rows):
    def group_by(key_fn):
        groups = defaultdict(list)
        for row in rows:
            groups[str(key_fn(row))].append(row)
        result = {}
        for key, group in sorted(groups.items()):
            correct = sum(int(row["correct"]) for row in group)
            n = len(group)
            result[key] = {
                "n": n,
                "correct": correct,
                "accuracy": correct / n,
                "wilson_95_ci": wilson_interval(correct, n),
                "mean_generated_tokens": sum(row["generated_tokens"] for row in group) / n,
                "mean_tool_calls": sum(row["tool_calls"] for row in group) / n,
            }
        return result

    return {
        "n": len(rows),
        "accuracy": sum(int(row["correct"]) for row in rows) / len(rows),
        "by_level": group_by(lambda row: row["level"]),
        "by_subject": group_by(lambda row: row["subject"]),
        "by_truncated": group_by(lambda row: bool(row["truncated"])),
        "by_level_and_truncated": group_by(
            lambda row: f"level={row['level']},truncated={str(bool(row['truncated'])).lower()}"
        ),
        "by_prediction_parseable": group_by(lambda row: bool(row["prediction_parseable"])),
        "by_any_successful_tool_call": group_by(lambda row: row["successful_tool_calls"] > 0),
        "interpretation_scope": (
            "Descriptive question subgroups only. Truncation, parseability, tool use, and correctness "
            "are associated outcomes; these summaries do not identify causal effects. Wilson intervals "
            "describe within-run question sampling and exclude training-seed variation."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no evaluation rows in {input_path}")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"duplicate question IDs in {input_path}")
    for row in rows:
        if int(row["correct"]) not in (0, 1):
            raise SystemExit(f"non-binary correctness for {row['id']}")
        for key in ("generated_tokens", "tool_calls", "successful_tool_calls"):
            if row[key] < 0:
                raise SystemExit(f"negative {key} for {row['id']}")

    result = {"input": str(input_path), **summarize(rows)}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
