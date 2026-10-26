#!/usr/bin/env python3
"""Turn generated problems into executable, replayable tool-interaction examples."""
import argparse
import json
from pathlib import Path

from tools import run_python
from verifier import score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/tool_math/train.jsonl")
    p.add_argument("--output", default="data/tool_math/train_trajectories.jsonl")
    args = p.parse_args()
    records = []
    for line in Path(args.data).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        result = run_python(row["tool_code"])
        if not result["ok"] or score(f"<answer>{result['output']}</answer>", row["answer"]) != 1:
            raise SystemExit(f"trajectory verification failed for {row['id']}: {result}")
        records.append({"id": row["id"], "kind": row["kind"], "messages": [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": f"I will compute this exactly.\n```python\n{row['tool_code']}\n```"},
            {"role": "user", "content": f"Tool result (success=True):\n{result['output']}\nContinue and give the final result in <answer>...</answer>."},
            {"role": "assistant", "content": f"<answer>{row['answer']}</answer>"}],
            "tool_code": row["tool_code"], "tool_output": result["output"], "answer": row["answer"]})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8")
    print(f"verified trajectories: {len(records)} -> {output}")


if __name__ == "__main__":
    main()
