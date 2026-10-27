#!/usr/bin/env python3
"""Summarize pre-clip gradient norms and clipping frequency from GRPO metrics."""
import argparse
import json
import math
from pathlib import Path
import statistics


def read_metrics(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    if not rows:
        raise ValueError(f"no metrics rows in {path}")
    return rows


def quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (ordered[lower] * (upper - position) +
            ordered[upper] * (position - lower))


def summarize(path, clip_norm):
    metrics_path = Path(path)
    rows = read_metrics(metrics_path)
    steps = [row.get("step") for row in rows]
    if any(not isinstance(step, int) for step in steps) or len(set(steps)) != len(steps):
        raise ValueError(f"step indices are missing, invalid, or duplicated in {metrics_path}")
    if steps != sorted(steps):
        raise ValueError(f"step indices are not ordered in {metrics_path}")

    updates = []
    skipped = 0
    for index, row in enumerate(rows, 1):
        applied = row.get("optimizer_step_applied")
        if not isinstance(applied, bool):
            raise ValueError(f"row {index}: optimizer_step_applied must be boolean")
        norm = row.get("gradient_norm_before_clip")
        if applied:
            if (not isinstance(norm, (int, float)) or isinstance(norm, bool) or
                    not math.isfinite(norm) or norm < 0):
                raise ValueError(f"row {index}: applied update has an invalid gradient norm")
            updates.append((row, float(norm)))
        else:
            if norm is not None:
                raise ValueError(f"row {index}: skipped update unexpectedly has a gradient norm")
            skipped += 1

    if not updates:
        raise ValueError(f"no optimizer updates recorded in {metrics_path}")
    norms = [norm for _, norm in updates]
    clipped = [norm for norm in norms if norm > clip_norm]
    config_path = metrics_path.parent / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    modes = {row.get("mode") for row in rows}
    if len(modes) != 1:
        raise ValueError(f"inconsistent modes in {metrics_path}")
    mode = next(iter(modes))
    if config.get("mode") is not None and config["mode"] != mode:
        raise ValueError(f"metrics mode disagrees with run config in {metrics_path}")

    return {
        "input": str(metrics_path),
        "mode": mode,
        "training_seed": config.get("seed"),
        "metric_rows": len(rows),
        "optimizer_updates": len(updates),
        "skipped_optimizer_steps": skipped,
        "attempted_groups": sum(row.get("attempted_groups", 0) for row in rows),
        "accepted_groups": sum(row.get("accepted_groups", 0) for row in rows),
        "discarded_zero_variance_groups": sum(
            row.get("discarded_zero_variance_groups", 0) for row in rows),
        "gradient_clip_norm_threshold": clip_norm,
        "updates_exceeding_clip_threshold": len(clipped),
        "fraction_updates_exceeding_clip_threshold": len(clipped) / len(updates),
        "pre_clip_gradient_norm": {
            "minimum": min(norms),
            "median": statistics.median(norms),
            "p90_linear": quantile(norms, 0.90),
            "p95_linear": quantile(norms, 0.95),
            "maximum": max(norms),
        },
        "clip_coefficient_for_clipped_updates": {
            "median": statistics.median(clip_norm / (norm + 1e-6) for norm in clipped)
            if clipped else None,
            "minimum": min((clip_norm / (norm + 1e-6) for norm in clipped), default=None),
        },
        "per_update": [
            {
                "step": row["step"],
                "pre_clip_gradient_norm": norm,
                "clipped": norm > clip_norm,
                "gradient_rescale_coefficient": min(1.0, clip_norm / (norm + 1e-6)),
            }
            for row, norm in updates
        ],
        "interpretation": (
            "The recorded norm is the global gradient norm before clipping. The reported "
            "clip coefficient describes gradient rescaling only; it is not the parameter "
            "update scale for AdamW."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True,
                        help="GRPO metrics JSONL; repeat for multiple runs")
    parser.add_argument("--clip-norm", type=float, default=1.0,
                        help="clip_grad_norm_ max_norm used by the training code")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not math.isfinite(args.clip_norm) or args.clip_norm <= 0:
        parser.error("clip-norm must be finite and positive")
    result = {
        "clip_norm_threshold": args.clip_norm,
        "runs": [summarize(path, args.clip_norm) for path in args.input],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"GRPO gradient-clipping summary failed: {exc}") from exc
