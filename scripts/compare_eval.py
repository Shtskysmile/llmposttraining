#!/usr/bin/env python3
"""Compare two paired deterministic evaluation JSONL runs with bootstrap CI."""
import argparse
import json
import math
import random
from pathlib import Path


def read_run(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    indexed = {row["id"]: int(row["correct"]) for row in rows}
    if len(indexed) != len(rows):
        raise SystemExit(f"duplicate ids in {path}")
    return indexed


def exact_mcnemar(candidate_only, baseline_only):
    discordant = candidate_only + baseline_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(candidate_only, baseline_only) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, default=5090)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline = read_run(args.baseline)
    candidate = read_run(args.candidate)
    if baseline.keys() != candidate.keys():
        raise SystemExit("paired comparison requires identical question id sets")
    ids = sorted(baseline)
    differences = [candidate[key] - baseline[key] for key in ids]
    baseline_only = sum(value == -1 for value in differences)
    candidate_only = sum(value == 1 for value in differences)
    rng = random.Random(args.seed)
    boot = []
    for _ in range(args.bootstrap_samples):
        boot.append(sum(differences[rng.randrange(len(differences))] for _ in ids) / len(ids))
    result = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "n": len(ids),
        "baseline_accuracy": sum(baseline.values()) / len(ids),
        "candidate_accuracy": sum(candidate.values()) / len(ids),
        "paired_accuracy_difference": sum(differences) / len(ids),
        "bootstrap_95_percentile_ci": [quantile(boot, 0.025), quantile(boot, 0.975)],
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "baseline_only_correct": baseline_only,
        "candidate_only_correct": candidate_only,
        "exact_mcnemar_two_sided_p": exact_mcnemar(candidate_only, baseline_only),
        "interpretation_scope": "question-sampling uncertainty only; excludes training-seed variation",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
