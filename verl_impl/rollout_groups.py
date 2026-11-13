#!/usr/bin/env python3
"""Extract per-prompt correctness lists from training rollout dumps.

The dumps for different variants use the same data order (``shuffle:false``),
so the resulting ``groups.json`` files can be paired by prompt with
``stats.py``.  Use ``--metric pass@k`` to compare pass@4 across variants.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    groups: dict[str, list[float]] = defaultdict(list)
    files = sorted(
        Path(args.dump_dir).glob("*.jsonl"),
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem,
    )
    for path in files:
        for line in path.open(encoding="utf-8"):
            row = json.loads(line)
            if not line.strip():
                continue
            groups[str(row.get("input", ""))].append(float(row.get("acc", 0.0)))
    Path(args.out).write_text(
        json.dumps({"groups": dict(groups)}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{len(groups)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
