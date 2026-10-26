#!/usr/bin/env python3
"""Collect checkpoint/split metrics from the legacy SFT target/LR diagnostic."""
import argparse
import json
import re
from pathlib import Path


NAME = re.compile(
    r"^sft_diagnostic_(final|trace|preserved)_lr([^_]+)_step(\d+)_(train|validation)\.summary\.json$"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    observed = set()
    for path in sorted(Path(args.runs_dir).glob("sft_diagnostic_*_step*_*.summary.json")):
        match = NAME.fullmatch(path.name)
        if not match:
            continue
        target, learning_rate, step, split = match.groups()
        key = (target, learning_rate, int(step), split)
        observed.add(key)
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "target_mode": target,
            "learning_rate": learning_rate,
            "checkpoint_step": key[2],
            "split": split,
            "n": summary.get("n"),
            "pass_at_1": summary.get("pass@1"),
            "format_error_rate": summary.get("format_error_rate"),
            "mean_generated_tokens": summary.get("mean_generated_tokens"),
            "source_summary": str(path),
        })

    expected = {
        (target, learning_rate, step, split)
        for target in ("final", "trace", "preserved")
        for learning_rate in ("1e-5", "3e-5", "1e-4")
        for step in (4, 8, 16, 32)
        for split in ("train", "validation")
    }
    missing = sorted(expected - observed)
    if missing:
        raise SystemExit(f"incomplete diagnostic: {len(missing)} expected summaries are missing: {missing[:8]}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
