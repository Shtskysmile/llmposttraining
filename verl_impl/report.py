#!/usr/bin/env python3
"""Merge analyze.py summaries into a Markdown comparison table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", default=[], help="label=path/to/summary.json")
    parser.add_argument("--dynamics", action="append", default=[], help="label=path/to/dynamics.json")
    args = parser.parse_args()

    rows = []
    for item in args.input:
        label, path = item.split("=", 1)
        data = json.loads(Path(path).read_text())
        summary = data.get("summary", data)
        passk_key = next(
            (key for key in summary if key.startswith("pass@") and key != "pass@1"),
            "pass@1",
        )
        rows.append(
            {
                "variant": label,
                "pass@1": summary.get("pass@1", 0.0),
                "pass@k": summary.get(passk_key, 0.0),
                "mean_acc": summary.get("mean_acc_over_samples", 0.0),
                "format": summary.get("format_ok_rate", 0.0),
                "coverage": summary.get("mean_process_coverage", 0.0),
                "loop": summary.get("mean_loop_ratio", 0.0),
                "zero_var": summary.get("zero_variance_group_rate", 0.0),
                "reward_zero_var": summary.get("reward_zero_variance_group_rate", 0.0),
                "all_wrong": summary.get("all_wrong_group_rate", 0.0),
                "reward": summary.get("mean_reward", 0.0),
                "group_std": summary.get("mean_group_reward_std", 0.0),
            }
        )

    if not rows:
        raise SystemExit("No summaries found.")

    header = (
        "| variant | pass@1 | pass@k | mean acc | format | coverage | loop | "
        "zero-var(acc) | zero-var(reward) | all-wrong | mean reward | group std |"
    )
    print(header)
    print("|" + "---|" * 12)
    for row in rows:
        print(
            f"| {row['variant']} | {row['pass@1']:.4f} | {row['pass@k']:.4f} | "
            f"{row['mean_acc']:.4f} | {row['format']:.4f} | {row['coverage']:.4f} | "
            f"{row['loop']:.4f} | {row['zero_var']:.4f} | {row['reward_zero_var']:.4f} | "
            f"{row['all_wrong']:.4f} | {row['reward']:.4f} | "
            f"{row['group_std']:.4f} |"
        )

    dynamics_rows = []
    for item in args.dynamics:
        label, path = item.split("=", 1)
        data = json.loads(Path(path).read_text())
        summary = data.get("summary")
        per_step = data.get("per_step") or {}
        if summary:
            dynamics_rows.append((label, "all", summary))
        elif per_step:
            last_key = sorted(per_step, key=lambda k: int(k) if k.isdigit() else k)[-1]
            dynamics_rows.append((label, last_key, per_step[last_key]))

    if dynamics_rows:
        print("\n### Training rollout dynamics (aggregate)")
        print(
            "| variant | step | mean acc | coverage | zero-var(acc) | "
            "zero-var(reward) | all-wrong | mean reward | group std | mean chars |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for label, step, stats in dynamics_rows:
            print(
                f"| {label} | {step} | {stats.get('mean_acc_over_samples', 0):.4f} | "
                f"{stats.get('mean_process_coverage', 0):.4f} | "
                f"{stats.get('zero_variance_group_rate', 0):.4f} | "
                f"{stats.get('reward_zero_variance_group_rate', 0):.4f} | "
                f"{stats.get('all_wrong_group_rate', 0):.4f} | "
                f"{stats.get('mean_reward', 0):.4f} | "
                f"{stats.get('mean_group_reward_std', 0):.4f} | "
                f"{stats.get('mean_output_chars', 0):.0f} |"
            )


if __name__ == "__main__":
    main()
