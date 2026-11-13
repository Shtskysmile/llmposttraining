#!/usr/bin/env python3
"""Per-step training dynamics and length-drift report.

Reads the ``dynamics.json`` files produced by
``analyze.py --per-step`` and prints, for each variant, the per-step
accuracy / process coverage / zero-variance rates / mean output length.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", action="append", default=[], help="label=path/to/dynamics.json"
    )
    args = parser.parse_args()

    for item in args.input:
        label, path = item.split("=", 1)
        data = load(path)
        per_step = data.get("per_step") or {}
        if not per_step:
            continue
        keys = sorted(per_step, key=lambda k: int(k) if k.isdigit() else k)
        print(f"\n### {label}")
        print(
            "| step | mean acc | coverage | zero-var(acc) | zero-var(reward) | "
            "all-wrong | mean reward | group std | mean chars |"
        )
        print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        lengths = []
        for key in keys:
            s = per_step[key]
            lengths.append(s.get("mean_output_chars", 0.0))
            print(
                f"| {key} | {s.get('mean_acc_over_samples', 0):.3f} | "
                f"{s.get('mean_process_coverage', 0):.3f} | "
                f"{s.get('zero_variance_group_rate', 0):.3f} | "
                f"{s.get('reward_zero_variance_group_rate', 0):.3f} | "
                f"{s.get('all_wrong_group_rate', 0):.3f} | "
                f"{s.get('mean_reward', 0):.3f} | "
                f"{s.get('mean_group_reward_std', 0):.3f} | "
                f"{s.get('mean_output_chars', 0):.0f} |"
            )
        if len(lengths) >= 2:
            print(
                f"\nLength drift: first={lengths[0]:.0f} chars, "
                f"last={lengths[-1]:.0f} chars, "
                f"delta={lengths[-1] - lengths[0]:+.0f}."
            )


if __name__ == "__main__":
    main()
