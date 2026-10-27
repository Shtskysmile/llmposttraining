#!/usr/bin/env python3
"""Summarize Python-tool execution failures in rollout JSONL."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def failure_label(output):
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "empty error output"
    if lines[0].startswith("disallowed ") or lines[0].startswith("only "):
        return lines[0]
    return lines[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--group-field", default="kind")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if not rows:
        parser.error("input rollout file is empty")

    groups = defaultdict(lambda: {"questions": set(), "calls": 0, "successes": 0,
                                  "failures": Counter(), "examples": defaultdict(list)})
    for row in rows:
        group = str(row.get(args.group_field, "unknown"))
        item = groups[group]
        for event in row.get("tool_events", []):
            result = event.get("result", {})
            item["questions"].add(row.get("id"))
            item["calls"] += 1
            if result.get("ok"):
                item["successes"] += 1
                continue
            label = failure_label(str(result.get("output", "")))
            item["failures"][label] += 1
            if len(item["examples"][label]) < 3:
                item["examples"][label].append({
                    "id": row.get("id"),
                    "code": str(event.get("code", ""))[:1000],
                })

    by_group = {}
    for group, item in sorted(groups.items()):
        failures = sum(item["failures"].values())
        by_group[group] = {
            "questions_with_tool_events": len(item["questions"]),
            "tool_calls": item["calls"],
            "successful_calls": item["successes"],
            "failed_calls": failures,
            "execution_success_rate": item["successes"] / item["calls"] if item["calls"] else None,
            "failure_counts": dict(sorted(item["failures"].items())),
            "failure_examples": dict(item["examples"]),
        }

    result = {"input": args.input, "group_field": args.group_field,
              "n_questions": len(rows), "by_group": by_group}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
