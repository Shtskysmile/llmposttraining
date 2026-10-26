#!/usr/bin/env python3
"""Paired analysis across multiple sampling seeds for multi-sample evaluations."""
import argparse
import json
import math
import random
import statistics
from pathlib import Path


PAIRING_KEYS = (
    "data_sha256", "n", "mode", "revision", "num_samples", "max_total_tokens",
    "max_new_tokens", "max_tool_calls", "temperature", "top_k", "top_p",
    "per_sample_token_budgets", "majority_vote_rule", "unparseable_answers_in_vote",
    "sampling", "seed_strategy", "gpu",
)


def load_run(path):
    path = Path(path)
    config_path = path.with_suffix(".run_config.json")
    if not config_path.is_file():
        raise SystemExit(f"missing run config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise SystemExit(f"duplicate question ids in {path}")
    if len(rows) != config.get("n"):
        raise SystemExit(f"run row count does not match config in {path}")
    if any(len(row.get("sample_correct", [])) != config.get("num_samples")
           for row in rows):
        raise SystemExit(f"sample count mismatch in {path}")
    return {"path": str(path), "config": config, "rows": indexed}


def value(row, metric):
    if metric == "sample_accuracy":
        return statistics.mean(row["sample_correct"])
    key = "pass_at_k" if metric == "pass_at_k" else "majority_vote_correct"
    return int(row[key])


def exact_mcnemar(candidate_only, baseline_only):
    discordant = candidate_only + baseline_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k)
               for k in range(min(candidate_only, baseline_only) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=5090)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.baseline) != len(args.candidate) or len(args.baseline) < 2:
        parser.error("provide at least two baseline and candidate runs")
    if args.bootstrap_samples < 1:
        parser.error("bootstrap-samples must be positive")

    def collect(paths, label):
        runs = {}
        for path in paths:
            run = load_run(path)
            seed = run["config"].get("seed")
            if seed is None or seed in runs:
                raise SystemExit(f"{label} runs need unique recorded sampling seeds")
            runs[seed] = run
        return runs

    baseline = collect(args.baseline, "baseline")
    candidate = collect(args.candidate, "candidate")
    if baseline.keys() != candidate.keys():
        raise SystemExit("baseline and candidate need the same sampling seed set")
    seeds = sorted(baseline)
    ids = sorted(baseline[seeds[0]]["rows"])
    if not ids:
        raise SystemExit("evaluation runs are empty")
    ref_config = baseline[seeds[0]]["config"]
    metrics = {}
    for seed in seeds:
        base_run, candidate_run = baseline[seed], candidate[seed]
        if sorted(base_run["rows"]) != ids or candidate_run["rows"].keys() != base_run["rows"].keys():
            raise SystemExit(f"question ids differ at sampling seed {seed}")
        for key in PAIRING_KEYS:
            if base_run["config"].get(key) != candidate_run["config"].get(key):
                raise SystemExit(f"paired setting {key} differs at sampling seed {seed}")
            if ref_config.get(key) != base_run["config"].get(key):
                raise SystemExit(f"sampling setting {key} differs across seeds")

    rng = random.Random(args.bootstrap_seed)
    id_to_index = {question_id: index for index, question_id in enumerate(ids)}
    for metric in ("sample_accuracy", "pass_at_k", "majority_vote_accuracy"):
        differences = {}
        per_seed = {}
        for seed in seeds:
            base_rows, candidate_rows = baseline[seed]["rows"], candidate[seed]["rows"]
            diff = [value(candidate_rows[qid], metric) - value(base_rows[qid], metric)
                    for qid in ids]
            differences[seed] = diff
            base_mean = statistics.mean(value(base_rows[qid], metric) for qid in ids)
            candidate_mean = statistics.mean(value(candidate_rows[qid], metric) for qid in ids)
            row = {"baseline": base_mean, "candidate": candidate_mean,
                   "paired_difference": statistics.mean(diff)}
            if metric != "sample_accuracy":
                baseline_only = sum(x == -1 for x in diff)
                candidate_only = sum(x == 1 for x in diff)
                row.update({
                    "baseline_only_correct": baseline_only,
                    "candidate_only_correct": candidate_only,
                    "exact_mcnemar_two_sided_p": exact_mcnemar(candidate_only, baseline_only),
                })
            per_seed[str(seed)] = row

        question_boot, seed_boot, hierarchical_boot = [], [], []
        for _ in range(args.bootstrap_samples):
            sampled_ids = [ids[rng.randrange(len(ids))] for _ in ids]
            sampled_seeds = [seeds[rng.randrange(len(seeds))] for _ in seeds]
            question_boot.append(statistics.mean(
                differences[seed][id_to_index[qid]]
                for seed in seeds for qid in sampled_ids))
            seed_boot.append(statistics.mean(
                differences[seed][index]
                for seed in sampled_seeds for index in range(len(ids))))
            hierarchical_boot.append(statistics.mean(
                differences[seed][id_to_index[qid]]
                for seed in sampled_seeds for qid in sampled_ids))

        seed_differences = [per_seed[str(seed)]["paired_difference"] for seed in seeds]
        metrics[metric] = {
            "mean_baseline": statistics.mean(per_seed[str(seed)]["baseline"] for seed in seeds),
            "mean_candidate": statistics.mean(per_seed[str(seed)]["candidate"] for seed in seeds),
            "mean_paired_difference": statistics.mean(seed_differences),
            "paired_difference_sd_across_sampling_seeds": statistics.stdev(seed_differences),
            "question_bootstrap_95_percentile_ci": [
                quantile(question_boot, 0.025), quantile(question_boot, 0.975)],
            "sampling_seed_bootstrap_95_percentile_ci": [
                quantile(seed_boot, 0.025), quantile(seed_boot, 0.975)],
            "hierarchical_bootstrap_95_percentile_ci": [
                quantile(hierarchical_boot, 0.025), quantile(hierarchical_boot, 0.975)],
            "per_seed": per_seed,
        }

    result = {
        "baseline_runs": args.baseline,
        "candidate_runs": args.candidate,
        "n_questions": len(ids),
        "n_sampling_seeds": len(seeds),
        "sampling_seeds": seeds,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "paired_configuration": {key: ref_config[key] for key in PAIRING_KEYS},
        "metrics": metrics,
        "interpretation_scope": (
            "intervals estimate question and rollout-sampling variation. Each model uses a "
            "single training seed, so these intervals do not estimate training-seed variation. "
            "With three sampling seeds, seed-level intervals remain exploratory."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
