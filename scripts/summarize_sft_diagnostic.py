#!/usr/bin/env python3
"""Validate and collect checkpoint/split metrics from the legacy SFT diagnostic."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from src.verifier import format_error, score


TARGETS = ("final", "trace", "preserved")
LEARNING_RATES = ("1e-5", "3e-5", "1e-4")
STEPS = (4, 8, 16, 32)
SPLITS = ("train", "validation")
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
NAME_PATTERN = "sft_diagnostic_*_step*_*.summary.json"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def require_close(actual, expected, message):
    require(isinstance(actual, (int, float)) and math.isfinite(actual) and
            abs(actual - expected) <= 1e-12,
            f"{message}: got {actual}, expected {expected}")


def paths_for(summary_path):
    suffix = ".summary.json"
    require(summary_path.name.endswith(suffix), f"unexpected summary filename: {summary_path}")
    output_path = summary_path.with_name(summary_path.name[:-len(suffix)] + ".jsonl")
    return output_path, output_path.with_suffix(".run_config.json")


def validate_result(summary_path, target, learning_rate, step, split):
    output_path, config_path = paths_for(summary_path)
    require(config_path.is_file(), f"missing run config for {summary_path}: {config_path}")
    require(output_path.is_file(), f"missing JSONL output for {summary_path}: {output_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    data_path = Path("data/seed2027/train_base_correct_common.jsonl" if split == "train"
                     else "data/validation.jsonl")
    require(data_path.is_file(), f"missing expected {split} data: {data_path}")
    raw_data = data_path.read_bytes()
    expected_sha = hashlib.sha256(raw_data).hexdigest()
    source_rows = read_jsonl(data_path)
    expected_model = f"runs/sft_diagnostic_{target}_lr{learning_rate}/checkpoint-{step}"

    expected_config = {
        "model": expected_model,
        "base_model": "Qwen/Qwen3-1.7B",
        "revision": REVISION,
        "data": str(data_path),
        "data_sha256": expected_sha,
        "n": len(source_rows),
        "samples_per_question": 1,
        "seed": 17,
        "max_new_tokens": 320,
        "batch_size": 4,
        "decoding": "greedy",
        "temperature": None,
    }
    mismatches = [key for key, expected in expected_config.items()
                  if config.get(key) != expected]
    require(not mismatches,
            f"run config mismatch for {summary_path}: {', '.join(mismatches)}")

    rows = read_jsonl(output_path)
    require(len(rows) == len(source_rows),
            f"row count mismatch for {output_path}: {len(rows)} != {len(source_rows)}")
    correct = []
    format_errors = []
    lengths = []
    for index, (row, source) in enumerate(zip(rows, source_rows)):
        require(row.get("prompt") == source.get("prompt") and
                str(row.get("gold")) == str(source.get("answer")),
                f"source alignment mismatch at {output_path}:{index + 1}")
        completions = row.get("completions", [])
        scores = row.get("scores", [])
        row_lengths = row.get("lengths", [])
        require(len(completions) == len(scores) == len(row_lengths) == 1,
                f"expected one completion per row at {output_path}:{index + 1}")
        expected_score = score(completions[0], source["answer"])
        require(scores[0] == expected_score,
                f"verifier score mismatch at {output_path}:{index + 1}")
        require(isinstance(row_lengths[0], int) and row_lengths[0] >= 0,
                f"invalid token length at {output_path}:{index + 1}")
        correct.append(expected_score)
        format_errors.append(int(format_error(completions[0])))
        lengths.append(row_lengths[0])

    expected_summary = {
        "model": expected_model,
        "revision": REVISION,
        "seed": 17,
        "n": len(source_rows),
        "data_sha256": expected_sha,
    }
    summary_mismatches = [key for key, expected in expected_summary.items()
                          if summary.get(key) != expected]
    require(not summary_mismatches,
            f"summary metadata mismatch for {summary_path}: {', '.join(summary_mismatches)}")
    require_close(summary.get("pass@1"), sum(correct) / len(correct),
                  f"pass@1 mismatch for {summary_path}")
    require_close(summary.get("format_error_rate"), sum(format_errors) / len(format_errors),
                  f"format-error rate mismatch for {summary_path}")
    require_close(summary.get("mean_generated_tokens"), sum(lengths) / len(lengths),
                  f"mean token length mismatch for {summary_path}")

    return {
        "target_mode": target,
        "learning_rate": learning_rate,
        "checkpoint_step": step,
        "split": split,
        "n": len(source_rows),
        "pass_at_1": sum(correct) / len(correct),
        "format_error_rate": sum(format_errors) / len(format_errors),
        "mean_generated_tokens": sum(lengths) / len(lengths),
        "data_sha256": expected_sha,
        "source_jsonl": str(output_path),
        "source_summary": str(summary_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    observed = set()
    for path in sorted(Path(args.runs_dir).glob(NAME_PATTERN)):
        parts = path.name.removesuffix(".summary.json").split("_")
        if len(parts) != 6 or parts[0:2] != ["sft", "diagnostic"]:
            continue
        target = parts[2]
        if target not in TARGETS or not parts[3].startswith("lr"):
            continue
        learning_rate = parts[3][2:]
        try:
            step = int(parts[4][4:]) if parts[4].startswith("step") else None
        except ValueError:
            continue
        split = parts[5]
        if learning_rate not in LEARNING_RATES or step not in STEPS or split not in SPLITS:
            continue
        key = (target, learning_rate, step, split)
        require(key not in observed, f"duplicate diagnostic summary for {key}")
        observed.add(key)
        rows.append(validate_result(path, *key))

    expected = {
        (target, learning_rate, step, split)
        for target in TARGETS
        for learning_rate in LEARNING_RATES
        for step in STEPS
        for split in SPLITS
    }
    missing = sorted(expected - observed)
    if missing:
        raise SystemExit(
            f"incomplete diagnostic: {len(missing)} expected summaries are missing: {missing[:8]}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(output),
                      "fully_audited": True}, indent=2))


if __name__ == "__main__":
    main()
