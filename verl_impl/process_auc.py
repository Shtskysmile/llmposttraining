#!/usr/bin/env python3
"""Does the dense PCPR signal predict the final outcome?

For each greedy screen dump, compute the AUC of ``process_coverage`` (the
fraction of gold intermediate values the model actually proved) as a predictor
of final-answer correctness, plus the mean coverage split by correctness.

An AUC of 0.5 means the dense reward carries no outcome information; 1.0 means
it perfectly ranks correct above incorrect samples.  This is the calibration
check that a dense process reward should pass before it is used as training
signal: it must at least agree with the outcome it is meant to densify.

Usage:
    python verl_impl/process_auc.py verl_impl/outputs/screen_*/[0-9]*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def auc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney AUC with ties counted as 0.5."""
    if not pos or not neg:
        return float("nan")
    # O(n log n) rank-based implementation.
    combined = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    combined.sort(key=lambda t: t[0])
    rank_sum_pos = 0.0
    i = 0
    n = len(combined)
    ranks = [0.0] * n
    while i < n:
        j = i
        while j + 1 < n and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    for rank, (_, label) in zip(ranks, combined):
        if label == 1:
            rank_sum_pos += rank
    n_pos, n_neg = len(pos), len(neg)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dumps", nargs="+", help="jsonl dump paths or globs")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    paths: list[str] = []
    for pattern in args.dumps:
        matched = sorted(glob.glob(pattern))
        paths.extend(matched if matched else [pattern])

    out = []
    print(f"{'variant':<14} {'n':>4} {'correct':>8} {'cov|correct':>12} "
          f"{'cov|wrong':>10} {'AUC':>6}")
    for path in paths:
        rows = load_jsonl(path)
        if not rows:
            continue
        label = Path(path).parent.name.replace("screen_", "")
        pos = [float(r.get("process_coverage", 0.0)) for r in rows if r.get("correct", 0) > 0]
        neg = [float(r.get("process_coverage", 0.0)) for r in rows if r.get("correct", 0) <= 0]
        a = auc(pos, neg)
        row = {
            "variant": label,
            "path": path,
            "n": len(rows),
            "n_correct": len(pos),
            "n_wrong": len(neg),
            "coverage_correct": mean(pos),
            "coverage_wrong": mean(neg),
            "auc": a,
        }
        out.append(row)
        print(f"{label:<14} {len(rows):>4} {len(pos):>8} "
              f"{mean(pos):>12.3f} {mean(neg):>10.3f} {a:>6.3f}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
