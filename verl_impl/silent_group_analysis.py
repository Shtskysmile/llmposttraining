#!/usr/bin/env python3
"""Compare the theoretical silent-group rate with the empirical rate.

For each group of ``n`` samples, the outcome-only silent probability is
``p^n + (1-p)^n`` where ``p`` is the group's empirical accuracy.  Averaging
that over groups gives an upper-biased estimate (p is estimated from the same
samples); the global-p estimate is a lower bound.  The empirical silent rate is
``P(all acc equal)``.  This script reports all three per step.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def analyze_step(path: Path) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        groups[str(row.get("input", ""))].append(row)

    formula = []
    empirical = []
    reward_silent = []
    for samples in groups.values():
        n = len(samples)
        accs = [float(s.get("acc", 0.0)) for s in samples]
        p = sum(accs) / n
        formula.append(p**n + (1 - p) ** n)
        empirical.append(float(len(set(accs)) == 1))
        scores = [float(s.get("score", 0.0)) for s in samples]
        reward_silent.append(float(len({round(x, 6) for x in scores}) == 1))
    return {
        "groups": len(groups),
        "formula_per_group": sum(formula) / len(formula),
        "empirical_acc": sum(empirical) / len(empirical),
        "empirical_reward": sum(reward_silent) / len(reward_silent),
        "global_p": sum(
            float(s.get("acc", 0.0)) for g in groups.values() for s in g
        )
        / sum(len(g) for g in groups.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True)
    args = parser.parse_args()
    dump_dir = Path(args.dump_dir)
    files = sorted(dump_dir.glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    print(f"{'step':>4} {'groups':>6} {'global_p':>8} {'formula':>8} {'emp_acc':>8} {'emp_reward':>10}")
    for path in files:
        s = analyze_step(path)
        print(
            f"{path.stem:>4} {s['groups']:>6} {s['global_p']:>8.3f} "
            f"{s['formula_per_group']:>8.3f} {s['empirical_acc']:>8.3f} "
            f"{s['empirical_reward']:>10.3f}"
        )


if __name__ == "__main__":
    main()
