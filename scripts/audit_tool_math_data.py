#!/usr/bin/env python3
"""Execute and audit generated tool-math examples and split separation."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tools import run_python
from verifier import score


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/tool_math_scale_seed5090")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.data_dir)
    rows_by_split = {name: read_rows(root / f"{name}.jsonl")
                     for name in ("train", "validation", "test")}
    report = {"data_dir": str(root), "splits": {}}
    prompt_sets = {}
    instance_sets = {}
    failures = []
    for split, rows in rows_by_split.items():
        prompts = [row["prompt"] for row in rows]
        prompt_sets[split] = set(prompts)
        instances = [row.get("instance_key", row["prompt"]) for row in rows]
        instance_sets[split] = set(instances)
        counts = Counter(row["kind"] for row in rows)
        verified = 0
        seen_ids = set()
        for row in rows:
            if row["id"] in seen_ids:
                failures.append({"split": split, "id": row["id"], "error": "duplicate id"})
            seen_ids.add(row["id"])
            result = run_python(row["tool_code"])
            if result["ok"] and score(f"<answer>{result['output']}</answer>", row["answer"]) == 1:
                verified += 1
            else:
                failures.append({"split": split, "id": row["id"], "result": result,
                                 "gold": row["answer"]})
        report["splits"][split] = {
            "rows": len(rows), "families": dict(sorted(counts.items())),
            "unique_prompts": len(prompt_sets[split]),
            "duplicate_prompt_rows": len(prompts) - len(prompt_sets[split]),
            "unique_instances": len(instance_sets[split]),
            "duplicate_instance_rows": len(instances) - len(instance_sets[split]),
            "tool_gold_verified": verified,
        }
    report["cross_split_prompt_overlap"] = {
        "train_validation": len(prompt_sets["train"] & prompt_sets["validation"]),
        "train_test": len(prompt_sets["train"] & prompt_sets["test"]),
        "validation_test": len(prompt_sets["validation"] & prompt_sets["test"]),
    }
    report["cross_split_instance_overlap"] = {
        "train_validation": len(instance_sets["train"] & instance_sets["validation"]),
        "train_test": len(instance_sets["train"] & instance_sets["test"]),
        "validation_test": len(instance_sets["validation"] & instance_sets["test"]),
    }
    report["failures"] = failures
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(f"data audit found {len(failures)} failure(s)")


if __name__ == "__main__":
    main()
