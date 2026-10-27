#!/usr/bin/env python3
"""Compare actual prompt-context token counts for matched-example direct/tool runs."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from summarize_rollout_prompt_tokens import direct_messages, stats, token_count
from tool_prompts import tool_messages


def load(path):
    raw = Path(path).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise SystemExit(f"duplicate IDs in {path}")
    return raw, indexed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--direct-data", required=True)
    parser.add_argument("--tool-data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    direct_raw, direct_rows = load(args.direct_data)
    tool_raw, tool_rows = load(args.tool_data)
    if direct_rows.keys() != tool_rows.keys():
        raise SystemExit("direct and tool prompt datasets have different question IDs")
    ids = sorted(direct_rows)
    if not ids:
        raise SystemExit("prompt datasets are empty")
    if any(direct_rows[qid].get("kind") != tool_rows[qid].get("kind") or
           direct_rows[qid].get("answer") != tool_rows[qid].get("answer")
           for qid in ids):
        raise SystemExit("direct and tool datasets differ in paired kind or answer")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    by_kind = defaultdict(lambda: {"direct": [], "tool": []})
    all_direct, all_tool = [], []
    for qid in ids:
        kind = str(direct_rows[qid].get("kind", "unknown"))
        direct_count = token_count(tokenizer, direct_messages(direct_rows[qid]["prompt"]))
        tool_count = token_count(tokenizer, tool_messages(tool_rows[qid]["prompt"]))
        by_kind[kind]["direct"].append(direct_count)
        by_kind[kind]["tool"].append(tool_count)
        all_direct.append(direct_count)
        all_tool.append(tool_count)

    def summarize(direct, tool):
        direct_stats, tool_stats = stats(direct), stats(tool)
        return {
            "n": len(direct),
            "direct_prompt_tokens": direct_stats,
            "tool_prompt_tokens": tool_stats,
            "tool_minus_direct_mean_prompt_tokens": tool_stats["mean"] - direct_stats["mean"],
        }

    result = {
        "model": args.model,
        "direct_data": args.direct_data,
        "direct_data_sha256": hashlib.sha256(direct_raw).hexdigest(),
        "tool_data": args.tool_data,
        "tool_data_sha256": hashlib.sha256(tool_raw).hexdigest(),
        "n": len(ids),
        "overall": summarize(all_direct, all_tool),
        "by_kind": {
            kind: summarize(values["direct"], values["tool"])
            for kind, values in sorted(by_kind.items())
        },
        "scope": "prompt context only; generated tokens, tool returns, and runtime are excluded",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
