#!/usr/bin/env python3
"""Offline analysis of verl rollout / validation dumps.

verl writes one JSONL per step:

* ``trainer.validation_data_dir`` -> ``<step>.jsonl`` with n samples per prompt
* ``trainer.rollout_data_dir``    -> ``<step>.jsonl`` with n samples per prompt

This script groups samples by prompt (input text) and reports pass@1, pass@k,
zero-variance group rate, format compliance and process coverage.  Those are
the quantities used to compare answer-only, format-shaped and process rewards.

Usage:
    python verl_impl/analyze.py --dump-dir verl_impl/outputs/base_val --label base
    python verl_impl/analyze.py --dump-dir verl_impl/outputs/arithmetic_grpo_process/rollout --per-step
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _is_correct(row: dict) -> float:
    if "acc" in row:
        return float(row["acc"])
    return float(float(row.get("score", 0.0)) >= 0.5)


def _mean(values, default=0.0):
    values = [float(v) for v in values if v is not None]
    return statistics.fmean(values) if values else default


def group_accuracies(rows: list[dict]) -> dict[str, list[float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("input", ""))].append(_is_correct(row))
    return dict(groups)


def summarize(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("input", ""))].append(row)

    group_stats = []
    for samples in groups.values():
        accs = [_is_correct(sample) for sample in samples]
        scores = [float(sample.get("score", 0.0)) for sample in samples]
        group_stats.append(
            {
                "n": len(samples),
                "accs": accs,
                "scores": scores,
                "score_std": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            }
        )

    n_groups = len(group_stats)
    n_samples = sum(group["n"] for group in group_stats)
    pass1 = _mean([group["accs"][0] for group in group_stats])
    passk = _mean([float(any(acc >= 0.5 for acc in group["accs"])) for group in group_stats])
    mean_acc = _mean([acc for group in group_stats for acc in group["accs"]])
    zero_var = _mean(
        [float(len(set(group["accs"])) == 1) for group in group_stats]
    )
    all_wrong = _mean(
        [float(all(acc < 0.5 for acc in group["accs"])) for group in group_stats]
    )
    all_correct = _mean(
        [float(all(acc >= 0.5 for acc in group["accs"])) for group in group_stats]
    )
    mean_group_std = _mean([group["score_std"] for group in group_stats])
    reward_zero_var = _mean(
        [
            float(len({round(score, 6) for score in group["scores"]}) == 1)
            for group in group_stats
        ]
    )

    return {
        "groups": n_groups,
        "samples": n_samples,
        "samples_per_group": _mean([group["n"] for group in group_stats]),
        "pass@1": pass1,
        f"pass@{int(round(_mean([group['n'] for group in group_stats])))}": passk,
        "mean_acc_over_samples": mean_acc,
        "zero_variance_group_rate": zero_var,
        "reward_zero_variance_group_rate": reward_zero_var,
        "all_wrong_group_rate": all_wrong,
        "all_correct_group_rate": all_correct,
        "mean_group_reward_std": mean_group_std,
        "mean_reward": _mean([row.get("score", 0.0) for row in rows]),
        "format_ok_rate": _mean([row.get("format_ok", 0.0) for row in rows]),
        "mean_process_coverage": _mean([row.get("process_coverage", 0.0) for row in rows]),
        "mean_valid_steps": _mean([row.get("process_valid_steps", 0.0) for row in rows]),
        "mean_invalid_steps": _mean([row.get("process_invalid_steps", 0.0) for row in rows]),
        "mean_loop_ratio": _mean([row.get("loop_ratio", 0.0) for row in rows]),
        "mean_loop_penalty": _mean([row.get("loop_penalty", 0.0) for row in rows]),
        "mean_output_chars": _mean([len(str(row.get("output", ""))) for row in rows]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--per-step", action="store_true", help="also report each JSONL separately")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--save-groups", default=None, help="write per-prompt correctness for paired tests")
    args = parser.parse_args()

    dump_dir = Path(args.dump_dir)
    files = sorted(dump_dir.glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not files:
        raise SystemExit(f"No JSONL files under {dump_dir}")

    label = args.label or dump_dir.name
    all_rows = []
    per_step = {}
    for path in files:
        rows = _read_jsonl(path)
        all_rows.extend(rows)
        per_step[path.stem] = summarize(rows)

    result = {"label": label, "dump_dir": str(dump_dir), "summary": summarize(all_rows)}
    if args.per_step:
        result["per_step"] = per_step

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.save_groups:
        Path(args.save_groups).write_text(
            json.dumps(
                {"label": label, "groups": group_accuracies(all_rows)},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    s = result["summary"]
    print(f"\n== {label} ({dump_dir}) ==")
    print(f"groups={s['groups']} samples={s['samples']} n/group={s['samples_per_group']:.1f}")
    passk_keys = [k for k in s if k.startswith("pass@") and k != "pass@1"]
    passk_key = passk_keys[0] if passk_keys else "pass@1"
    print(
        f"pass@1={s['pass@1']:.4f} pass@k={s[passk_key]:.4f} "
        f"mean_acc={s['mean_acc_over_samples']:.4f} format_ok={s['format_ok_rate']:.4f}"
    )
    print(
        f"zero_var(acc)={s['zero_variance_group_rate']:.4f} "
        f"zero_var(reward)={s['reward_zero_variance_group_rate']:.4f} "
        f"all_wrong={s['all_wrong_group_rate']:.4f} "
        f"all_correct={s['all_correct_group_rate']:.4f} group_std={s['mean_group_reward_std']:.4f}"
    )
    print(
        f"process_coverage={s['mean_process_coverage']:.4f} valid_steps={s['mean_valid_steps']:.2f} "
        f"invalid_steps={s['mean_invalid_steps']:.2f} loop_ratio={s['mean_loop_ratio']:.4f} "
        f"mean_reward={s['mean_reward']:.4f}"
    )
    if args.per_step:
        print("\nstep  pass@1   mean_acc  zero_var  coverage  reward")
        for step, stats in per_step.items():
            print(
                f"{step:>4}  {stats['pass@1']:.4f}  {stats['mean_acc_over_samples']:.4f}  "
                f"{stats['zero_variance_group_rate']:.4f}  {stats['mean_process_coverage']:.4f}  "
                f"{stats['mean_reward']:.4f}"
            )


if __name__ == "__main__":
    main()
