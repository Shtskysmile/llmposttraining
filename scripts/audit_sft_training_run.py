#!/usr/bin/env python3
"""Audit dataset provenance, metrics, and adapter output for a completed SFT run."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def finite_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool) and
            math.isfinite(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="SFT output directory")
    parser.add_argument("--output", help="optional path for the audit JSON")
    args = parser.parse_args()

    run_dir = Path(args.input)
    config_path = run_dir / "run_config.json"
    metrics_path = run_dir / "metrics.jsonl"
    require(config_path.is_file(), f"missing run config: {config_path}")
    require(metrics_path.is_file(), f"missing SFT metrics: {metrics_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    dataset_path = Path(config["dataset"])
    require(dataset_path.is_file(), f"missing SFT dataset: {dataset_path}")
    dataset_bytes = dataset_path.read_bytes()
    dataset_sha = hashlib.sha256(dataset_bytes).hexdigest()
    require(dataset_sha == config.get("dataset_sha256"),
            "dataset hash disagrees with run config")
    dataset_rows = read_jsonl(dataset_path)
    require(len(dataset_rows) == config.get("dataset_rows"),
            "dataset row count disagrees with run config")

    batch_size = config.get("batch_size")
    epochs = config.get("epochs")
    max_steps = config.get("max_steps", -1)
    require(isinstance(batch_size, int) and batch_size > 0,
            "invalid batch_size in run config")
    require(finite_number(epochs) and epochs > 0,
            "invalid epochs in run config")
    require(isinstance(max_steps, int), "invalid max_steps in run config")
    steps_per_epoch = math.ceil(len(dataset_rows) / batch_size)
    expected_steps = (max_steps if max_steps > 0 else
                      math.ceil(steps_per_epoch * epochs))

    rows = read_jsonl(metrics_path)
    require(rows, "SFT metrics log is empty")
    summary_rows = [row for row in rows if "train_runtime" in row]
    require(len(summary_rows) == 1 and rows[-1] is summary_rows[0],
            "expected one final Trainer summary row at the end of metrics")
    training_rows = rows[:-1]
    require(len(training_rows) == expected_steps,
            f"expected {expected_steps} logged training steps, found {len(training_rows)}")
    require([row.get("step") for row in training_rows] ==
            list(range(1, expected_steps + 1)),
            "training metric steps are missing, duplicated, or out of order")

    for row in training_rows:
        step = row["step"]
        for key in ("loss", "grad_norm", "learning_rate", "epoch"):
            require(finite_number(row.get(key)), f"step {step}: invalid metric {key}")
        require(row["loss"] >= 0, f"step {step}: negative training loss")
        require(row["grad_norm"] >= 0, f"step {step}: negative gradient norm")
        require(row["learning_rate"] >= 0, f"step {step}: negative learning rate")
        require(row["epoch"] >= 0, f"step {step}: negative epoch")
        for key in ("gpu_allocated_gib", "gpu_peak_allocated_gib"):
            if key in row:
                require(finite_number(row[key]) and row[key] >= 0,
                        f"step {step}: invalid metric {key}")

    summary = summary_rows[0]
    require(summary.get("step") == expected_steps,
            "final Trainer summary step disagrees with expected steps")
    for key in ("train_runtime", "train_samples_per_second", "train_steps_per_second",
                "total_flos", "train_loss", "epoch"):
        require(finite_number(summary.get(key)), f"invalid final Trainer metric {key}")
    require(summary["train_runtime"] >= 0 and summary["train_loss"] >= 0,
            "negative runtime or train loss in final Trainer summary")

    adapter_path = run_dir / "adapter_model.safetensors"
    require(adapter_path.is_file() and adapter_path.stat().st_size > 0,
            f"missing or empty final adapter: {adapter_path}")
    result = {
        "run_dir": str(run_dir),
        "seed": config.get("seed"),
        "target_mode": config.get("target_mode"),
        "dataset_rows": len(dataset_rows),
        "dataset_sha256": dataset_sha,
        "expected_training_steps": expected_steps,
        "verified_training_steps": len(training_rows),
        "finite_step_metrics": len(training_rows),
        "reported_zero_loss_steps": sum(row["loss"] == 0 for row in training_rows),
        "median_reported_step_loss": statistics.median(
            row["loss"] for row in training_rows),
        "loss_logging_note": (
            "Transformers Trainer rounds logged step loss to four decimals; a logged zero "
            "is not an exact unrounded zero"
        ),
        "median_gradient_norm": statistics.median(row["grad_norm"] for row in training_rows),
        "final_train_loss": summary["train_loss"],
        "train_runtime_seconds": summary["train_runtime"],
        "final_adapter_bytes": adapter_path.stat().st_size,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"SFT training-run audit failed: {exc}") from exc
