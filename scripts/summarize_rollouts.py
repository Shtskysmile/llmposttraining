#!/usr/bin/env python3
"""Summarize paired synthetic tool-rollout results overall and by task family."""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


def summarize(rows):
    n = len(rows)
    calls = sum(row.get("tool_calls", 0) for row in rows)
    completions = [row.get("completion", "") for row in rows]
    return {
        "n": n,
        "accuracy": sum(row.get("correct", 0) for row in rows) / max(1, n),
        "format_error_rate": sum(row.get("format_error", 0) for row in rows) / max(1, n),
        "answer_tag_rate": sum(bool(re.search(r"<answer>\s*[^<>\n]+?</\s*answer\s*>", text, re.I))
                                for text in completions) / max(1, n),
        "strict_answer_tag_rate": sum(bool(re.search(r"<answer>\s*[^<>\n]+?</answer>", text, re.I))
                                      for text in completions) / max(1, n),
        "boxed_answer_rate": sum(bool(re.search(r"\\boxed\s*\{", text)) for text in completions) / max(1, n),
        "mean_generated_tokens": statistics.mean(row.get("generated_tokens", 0) for row in rows) if n else 0,
        "mean_tool_calls": statistics.mean(row.get("tool_calls", 0) for row in rows) if n else 0,
        "tool_execution_success_rate": sum(row.get("successful_tool_calls", 0) for row in rows) / max(1, calls),
        "truncated_rate": sum(row.get("truncated", False) for row in rows) / max(1, n),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        parser.error("input rollout file is empty")
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get("kind", "unknown"))].append(row)
    result = {
        "input": args.input,
        "n": len(rows),
        "overall": summarize(rows),
        "by_kind": {kind: summarize(group) for kind, group in sorted(groups.items())},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
