#!/usr/bin/env python3
"""Create a reproducible, level-stratified subset of the pinned MATH-500 set."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/external/math500.jsonl")
    parser.add_argument("--per-level", type=int, default=20)
    parser.add_argument("--seed", type=int, default=5090)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.per_level < 1:
        parser.error("per-level must be positive")
    source = Path(args.data)
    source_bytes = source.read_bytes()
    rows = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines() if line.strip()]
    groups = {}
    for row in rows:
        groups.setdefault(str(row["level"]), []).append(row)
    rng = random.Random(args.seed)
    selected_ids = set()
    for level, group in sorted(groups.items()):
        if len(group) < args.per_level:
            parser.error(f"level {level} has only {len(group)} examples")
        selected_ids.update(row["id"] for row in rng.sample(group, args.per_level))
    selected = [row for row in rows if row["id"] in selected_ids]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_bytes = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected).encode("utf-8")
    output.write_bytes(output_bytes)
    manifest = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "seed": args.seed,
        "per_level": args.per_level,
        "n": len(selected),
        "selected_level_counts": dict(sorted(Counter(str(row["level"]) for row in selected).items())),
        "selected_ids": [row["id"] for row in selected],
        "output_sha256": hashlib.sha256(output_bytes).hexdigest(),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
