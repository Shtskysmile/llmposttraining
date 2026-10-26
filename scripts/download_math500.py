#!/usr/bin/env python3
"""Fetch a pinned MATH-500 snapshot and omit reference solutions from eval input."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


REPOSITORY = "HuggingFaceH4/MATH-500"
REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
SOURCE_SHA256 = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
SOURCE_URL = (f"https://huggingface.co/datasets/{REPOSITORY}/resolve/"
              f"{REVISION}/test.jsonl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/external/math500.jsonl")
    args = parser.parse_args()
    raw = urlopen(SOURCE_URL, timeout=60).read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise SystemExit(f"MATH-500 snapshot hash mismatch: {digest}")
    source_rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    if len(source_rows) != 500:
        raise SystemExit(f"Expected 500 records, received {len(source_rows)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = [{
        "id": row["unique_id"],
        "prompt": row["problem"],
        "answer": row["answer"],
        "subject": row["subject"],
        "level": row["level"],
    } for row in source_rows]
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
                      encoding="utf-8")
    manifest = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "source_file": "test.jsonl",
        "source_sha256": digest,
        "rows": len(records),
        "prompt_contains_reference_solution": False,
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                     encoding="utf-8")
    print(json.dumps({"output": str(output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
