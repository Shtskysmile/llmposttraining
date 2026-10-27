#!/usr/bin/env python3
"""Paired comparison across matched evaluation runs from multiple training seeds."""
import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path


PAIRING_KEYS = (
    "data_sha256", "n", "base_model", "mode", "revision", "decoding",
    "max_new_tokens_per_turn", "max_total_tokens_per_question", "max_tool_calls", "gpu",
)
PAIRING_KEY_ALIASES = {
    "max_new_tokens_per_turn": ("max_new_tokens_per_turn", "max_turn_tokens"),
    "max_total_tokens_per_question": ("max_total_tokens_per_question", "max_total_tokens"),
}


def pairing_value(config, key):
    if key == "base_model":
        return config.get("base_model") or config.get("model")
    for alias in PAIRING_KEY_ALIASES.get(key, (key,)):
        if alias in config:
            return config[alias]
    return None


def read_run(path):
    path = Path(path)
    config_path = path.with_suffix(".run_config.json")
    if not config_path.is_file():
        raise SystemExit(f"missing run config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "max_total_tokens_per_question" in config:
        audit_script = Path(__file__).with_name("audit_rollout_run.py")
        subprocess.run(
            [sys.executable, str(audit_script), "--input", str(path), "--require-complete"],
            check=True,
        )
    elif {"max_turn_tokens", "max_total_tokens", "manifest"} <= config.keys():
        audit_script = Path(__file__).with_name("audit_math500_run.py")
        subprocess.run(
            [sys.executable, str(audit_script), "--input", str(path), "--require-complete"],
            check=True,
        )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    indexed = {row["id"]: int(row["correct"]) for row in rows}
    if len(indexed) != len(rows):
        raise SystemExit(f"duplicate question ids in {path}")
    if len(rows) != config.get("n"):
        raise SystemExit(f"run row count does not match config in {path}")
    if any(value not in (0, 1) for value in indexed.values()):
        raise SystemExit(f"non-binary correctness value in {path}")
    if "max_total_tokens_per_question" not in config:
        data_path = Path(config["data"])
        raw_data = data_path.read_bytes()
        data_sha = hashlib.sha256(raw_data).hexdigest()
        if data_sha != config.get("data_sha256"):
            raise SystemExit(f"source data SHA-256 differs from run config in {path}")
        source_rows = [json.loads(line) for line in raw_data.decode("utf-8").splitlines()
                       if line.strip()]
        if config.get("limit") is not None:
            source_rows = source_rows[:config["limit"]]
        source_ids = [row["id"] for row in source_rows]
        if len(source_rows) != config.get("n") or [row["id"] for row in rows] != source_ids:
            raise SystemExit(f"source IDs or order differ from run output in {path}")
        for index, (row, source) in enumerate(zip(rows, source_rows)):
            for key, expected in source.items():
                if row.get(key) != expected:
                    raise SystemExit(
                        f"source field {key!r} changed at row {index} in {path}"
                    )
    return indexed, config


def training_seed(config):
    if config.get("training_seed") is not None:
        return config["training_seed"]
    for model_path in (config.get("adapter"), config.get("model")):
        if not model_path:
            continue
        run_config_path = Path(model_path) / "run_config.json"
        if run_config_path.is_file():
            seed = json.loads(run_config_path.read_text(encoding="utf-8")).get("seed")
            if seed is not None:
                return seed
    return None


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
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="append", required=True,
                        help="repeat once per seed, in any order")
    parser.add_argument("--candidate", action="append", required=True,
                        help="repeat once per seed, in any order")
    parser.add_argument("--seed", type=int, default=5090,
                        help="bootstrap RNG seed, separate from training seeds")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.candidate) < 2 or len(args.baseline) not in (1, len(args.candidate)):
        parser.error("provide at least two candidate runs and either one fixed baseline or one per seed")
    if args.bootstrap_samples < 1:
        parser.error("bootstrap-samples must be positive")

    def collect_runs(role, paths):
        destination = {}
        for path in paths:
            rows, config = read_run(path)
            # Seed sweeps compare separately trained checkpoints. Prefer that
            # training seed when the run config points to an adapter/checkpoint;
            # rollout seed may intentionally be the same for every model.
            seed = training_seed(config)
            if seed is None:
                seed = config.get("seed")
            if seed is None and role == "baseline" and len(paths) == 1:
                seed = "__fixed_baseline__"
            if seed is None or seed in destination:
                raise SystemExit(
                    f"{role} runs need unique evaluation seeds or adapter training seeds"
                )
            destination[seed] = {"path": path, "rows": rows, "config": config}
        return destination

    candidate_runs = collect_runs("candidate", args.candidate)
    supplied_baselines = collect_runs("baseline", args.baseline)
    if len(supplied_baselines) == 1:
        baseline_run = next(iter(supplied_baselines.values()))
        if baseline_run["config"].get("decoding") != "greedy":
            raise SystemExit("a single reused baseline requires deterministic greedy decoding")
        baseline_runs = {seed: baseline_run for seed in candidate_runs}
    else:
        baseline_runs = supplied_baselines
        if baseline_runs.keys() != candidate_runs.keys():
            raise SystemExit("baseline and candidate need the same evaluation seed set")

    seeds = sorted(baseline_runs)
    ids = sorted(next(iter(baseline_runs.values()))["rows"])
    if not ids:
        raise SystemExit("evaluation runs are empty")
    baseline_reference = baseline_runs[seeds[0]]["config"]
    candidate_reference = candidate_runs[seeds[0]]["config"]
    mismatches = [key for key in PAIRING_KEYS
                  if pairing_value(baseline_reference, key) !=
                  pairing_value(candidate_reference, key)]
    if mismatches:
        raise SystemExit("paired settings differ for: " + ", ".join(mismatches))
    for seed in seeds:
        base = baseline_runs[seed]
        candidate = candidate_runs[seed]
        baseline_training_seed = training_seed(base["config"])
        if baseline_training_seed is not None and baseline_training_seed != seed:
            raise SystemExit(
                f"evaluation seed {seed} does not match baseline adapter training seed "
                f"{baseline_training_seed} in {base['path']}"
            )
        candidate_training_seed = training_seed(candidate["config"])
        if candidate_training_seed is None or candidate_training_seed != seed:
            raise SystemExit(
                f"candidate run {candidate['path']} must have an adapter run_config "
                f"with training seed {seed}"
            )
        if base["rows"].keys() != candidate["rows"].keys():
            raise SystemExit(f"question ids differ at evaluation seed {seed}")
        if sorted(base["rows"]) != ids:
            raise SystemExit(f"question ids differ across runs at evaluation seed {seed}")
        for key in PAIRING_KEYS:
            if pairing_value(base["config"], key) != pairing_value(candidate["config"], key):
                raise SystemExit(f"paired setting {key} differs at evaluation seed {seed}")

    per_seed = {}
    differences = {}
    for seed in seeds:
        base = baseline_runs[seed]["rows"]
        candidate = candidate_runs[seed]["rows"]
        diff = [candidate[key] - base[key] for key in ids]
        differences[seed] = diff
        baseline_only = sum(value == -1 for value in diff)
        candidate_only = sum(value == 1 for value in diff)
        per_seed[str(seed)] = {
            "baseline": statistics.mean(base[key] for key in ids),
            "candidate": statistics.mean(candidate[key] for key in ids),
            "paired_difference": statistics.mean(diff),
            "baseline_only_correct": baseline_only,
            "candidate_only_correct": candidate_only,
            "exact_mcnemar_two_sided_p": exact_mcnemar(candidate_only, baseline_only),
            "baseline_run": baseline_runs[seed]["path"],
            "candidate_run": candidate_runs[seed]["path"],
            "candidate_training_seed": training_seed(candidate_runs[seed]["config"]),
        }

    rng = random.Random(args.seed)
    id_to_index = {question_id: index for index, question_id in enumerate(ids)}
    question_boot = []
    training_seed_boot = []
    hierarchical_boot = []
    for _ in range(args.bootstrap_samples):
        sampled_ids = [ids[rng.randrange(len(ids))] for _ in ids]
        question_boot.append(statistics.mean(
            differences[seed][id_to_index[question_id]]
            for seed in seeds for question_id in sampled_ids))
        sampled_seeds = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        training_seed_boot.append(statistics.mean(
            differences[seed][index] for seed in sampled_seeds for index in range(len(ids))))
        hierarchical_boot.append(statistics.mean(
            differences[seed][id_to_index[question_id]]
            for seed in sampled_seeds for question_id in sampled_ids))

    seed_differences = [per_seed[str(seed)]["paired_difference"] for seed in seeds]
    has_evaluation_seeds = all(
        candidate_runs[seed]["config"].get("seed") is not None for seed in seeds
    )
    result = {
        "baseline_runs": args.baseline,
        "candidate_runs": args.candidate,
        "n_questions": len(ids),
        "n_training_seed_pairs": len(seeds),
        "pairing_seeds": seeds,
        "pairing_seed_source": "evaluation" if has_evaluation_seeds else "training",
        "evaluation_seeds": seeds if has_evaluation_seeds else None,
        "candidate_training_seeds": [per_seed[str(seed)]["candidate_training_seed"]
                                     for seed in seeds],
        "mean_baseline_accuracy": statistics.mean(
            per_seed[str(seed)]["baseline"] for seed in seeds),
        "mean_candidate_accuracy": statistics.mean(
            per_seed[str(seed)]["candidate"] for seed in seeds),
        "mean_paired_accuracy_difference": statistics.mean(seed_differences),
        "paired_difference_sd_across_training_seeds": (
            statistics.stdev(seed_differences) if len(seed_differences) > 1 else None),
        "question_bootstrap_95_percentile_ci": [
            quantile(question_boot, 0.025), quantile(question_boot, 0.975)],
        "training_seed_bootstrap_95_percentile_ci": [
            quantile(training_seed_boot, 0.025), quantile(training_seed_boot, 0.975)],
        "hierarchical_bootstrap_95_percentile_ci": [
            quantile(hierarchical_boot, 0.025), quantile(hierarchical_boot, 0.975)],
        "bootstrap_seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "per_seed": per_seed,
        "interpretation_scope": (
            "question bootstrap resamples paired questions; training-seed bootstrap resamples "
            "paired seeds; hierarchical bootstrap resamples both. With only three training "
            "seeds, seed-level intervals are exploratory and unstable."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
