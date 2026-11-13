#!/usr/bin/env python3
"""Convert the project's JSONL arithmetic splits to verl Parquet format.

Every record carries the canonical integer expression in ``extra_info`` so the
process verifier never has to re-parse the prompt at reward time.  The script
fails loudly if a gold expression does not evaluate to the recorded answer.

Usage:
    python verl_impl/prepare_data.py \
        --input-dir data/expanded_seed2028 --output-dir verl_impl/data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets import Dataset  # noqa: E402

from verl_impl.verifier import canonical_derivation, expression_from_prompt  # noqa: E402

SPLITS = ("train", "validation", "test")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_records(rows: list[dict]) -> list[dict]:
    records = []
    for index, row in enumerate(rows):
        prompt = row["prompt"]
        answer = str(row["answer"])
        expression = expression_from_prompt(prompt)
        if expression is None:
            raise ValueError(f"Cannot extract an expression from prompt: {prompt!r}")
        derivation = canonical_derivation(expression)
        if derivation["value"] != int(answer):
            raise ValueError(
                f"Expression {expression!r} evaluates to {derivation['value']}, "
                f"but the recorded answer is {answer!r}"
            )
        records.append(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "data_source": "arithmetic",
                "ability": "math",
                "reward_model": {"ground_truth": answer, "style": "rule"},
                "extra_info": {
                    "index": index,
                    "kind": row.get("kind", "unknown"),
                    "expression": expression,
                    "prompt": prompt,
                    "answer": answer,
                    "intermediate_values": derivation["intermediate_values"],
                },
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/expanded_seed2028")
    parser.add_argument("--output-dir", default="verl_impl/data")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in SPLITS:
        source = input_dir / f"{split}.jsonl"
        if not source.exists():
            raise FileNotFoundError(source)
        records = build_records(read_jsonl(source))
        dataset = Dataset.from_list(records)
        target = output_dir / f"{split}.parquet"
        dataset.to_parquet(str(target))
        summary[split] = {
            "rows": len(records),
            "path": str(target),
            "kinds": sorted({record["extra_info"]["kind"] for record in records}),
        }
        print(f"{split}: {len(records)} rows -> {target}")

    (output_dir / "metadata.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("example:", json.dumps(records[0]["extra_info"], ensure_ascii=False))


if __name__ == "__main__":
    main()
