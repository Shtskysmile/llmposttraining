#!/usr/bin/env python3
"""Paired significance tests for two evaluation dumps.

Usage:
    python verl_impl/stats.py \
      --a base=verl_impl/outputs/eval_base/groups.json \
      --b process=verl_impl/outputs/eval_process/groups.json

Reports, for each metric, the paired mean difference, a 95% paired-bootstrap
confidence interval, and an exact McNemar test for pass@1.  Prompts are aligned
by their input text; only the intersection is used.

This is deliberately conservative: 64 validation prompts and one training seed
cannot support a broad claim, but the paired design removes prompt-difficulty
noise and makes the uncertainty explicit.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

METRICS = {
    "pass@1": lambda accs: accs[0],
    "pass@k": lambda accs: float(any(a >= 0.5 for a in accs)),
    "mean_acc": lambda accs: sum(accs) / len(accs),
}


def load_groups(path: str) -> dict[str, list[float]]:
    data = json.loads(Path(path).read_text())
    return data["groups"] if "groups" in data else data


def mcnemar_exact(b01: int, b10: int) -> float:
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def paired_bootstrap(diffs: np.ndarray, n_boot: int = 10000, seed: int = 17):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diffs[idx].mean(axis=1)
    return float(diffs.mean()), tuple(float(x) for x in np.percentile(means, [2.5, 97.5]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="label=groups.json (reference)")
    parser.add_argument("--b", required=True, help="label=groups.json (treatment)")
    parser.add_argument("--metric", choices=sorted(METRICS), default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    label_a, path_a = args.a.split("=", 1)
    label_b, path_b = args.b.split("=", 1)
    groups_a = load_groups(path_a)
    groups_b = load_groups(path_b)
    prompts = sorted(set(groups_a) & set(groups_b))
    if not prompts:
        raise SystemExit("No overlapping prompts between the two dumps.")

    metrics = [args.metric] if args.metric else ["pass@1", "pass@k", "mean_acc"]
    results = {}
    print(f"\nPaired comparison: {label_b} - {label_a}  (n={len(prompts)} prompts)")
    print("| metric | A | B | diff | 95% CI | p (McNemar for pass@1) |")
    print("|---|---:|---:|---:|---|---:|")
    for metric in metrics:
        fn = METRICS[metric]
        values_a = np.array([fn(groups_a[p]) for p in prompts])
        values_b = np.array([fn(groups_b[p]) for p in prompts])
        diffs = values_b - values_a
        mean_diff, ci = paired_bootstrap(diffs)
        if metric == "pass@1":
            b01 = int(((values_a < 0.5) & (values_b >= 0.5)).sum())
            b10 = int(((values_a >= 0.5) & (values_b < 0.5)).sum())
            p_value = mcnemar_exact(b01, b10)
            p_text = f"{p_value:.4f} ({b01}+ / {b10}-)"
        else:
            p_value = None
            p_text = "—"
        results[metric] = {
            "a": float(values_a.mean()),
            "b": float(values_b.mean()),
            "diff": mean_diff,
            "ci95": ci,
            "p_value": p_value,
        }
        print(
            f"| {metric} | {values_a.mean():.4f} | {values_b.mean():.4f} | "
            f"{mean_diff:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {p_text} |"
        )

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {"a": label_a, "b": label_b, "n": len(prompts), "results": results},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
