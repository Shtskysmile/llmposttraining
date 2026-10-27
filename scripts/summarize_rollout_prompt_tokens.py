#!/usr/bin/env python3
"""Measure prompt-token context for the direct and tool rollout protocols."""
import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tool_prompts import tool_messages


DIRECT_SYSTEM = (
    "Solve the math problem directly without tools. Keep the solution concise and put the exact final "
    "value in <answer>...</answer>."
)


def direct_messages(prompt):
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": prompt + "\nPut the exact final value in <answer>...</answer>."},
    ]


def token_count(tokenizer, messages):
    try:
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    return len(tokens)


def stats(values):
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data_path = Path(args.data)
    raw = data_path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not rows:
        parser.error("evaluation dataset is empty")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)

    counts = defaultdict(lambda: {"direct": [], "tool": []})
    for row in rows:
        kind = str(row.get("kind", "unknown"))
        counts[kind]["direct"].append(token_count(tokenizer, direct_messages(row["prompt"])))
        counts[kind]["tool"].append(token_count(tokenizer, tool_messages(row["prompt"])))

    by_kind = {}
    all_direct, all_tool = [], []
    for kind, values in sorted(counts.items()):
        all_direct.extend(values["direct"])
        all_tool.extend(values["tool"])
        direct_stats, tool_stats = stats(values["direct"]), stats(values["tool"])
        by_kind[kind] = {
            "n": len(values["direct"]),
            "direct_prompt_tokens": direct_stats,
            "tool_prompt_tokens": tool_stats,
            "tool_minus_direct_mean_prompt_tokens": tool_stats["mean"] - direct_stats["mean"],
        }

    direct_stats, tool_stats = stats(all_direct), stats(all_tool)
    result = {
        "model": args.model,
        "revision": args.revision,
        "data": str(data_path),
        "data_sha256": hashlib.sha256(raw).hexdigest(),
        "n": len(rows),
        "direct_prompt_tokens": direct_stats,
        "tool_prompt_tokens": tool_stats,
        "tool_minus_direct_mean_prompt_tokens": tool_stats["mean"] - direct_stats["mean"],
        "by_kind": by_kind,
        "scope": "prompt context only; generation tokens, tool returns, and runtime are excluded",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
