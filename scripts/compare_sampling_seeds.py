#!/usr/bin/env python3
"""Paired analysis across multiple sampling seeds for multi-sample evaluations."""
import argparse
from collections import Counter
import json
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.verifier import extract_answer, score


PAIRING_KEYS = (
    "data_sha256", "n", "model", "mode", "revision", "num_samples", "max_total_tokens",
    "max_new_tokens", "max_tool_calls", "temperature", "top_k", "top_p",
    "per_sample_token_budgets", "majority_vote_rule", "unparseable_answers_in_vote",
    "sampling", "seed_strategy", "gpu",
)


def load_run(path):
    path = Path(path)
    audit_script = Path(__file__).with_name("audit_sampling_run.py")
    subprocess.run(
        [sys.executable, str(audit_script), "--input", str(path), "--require-complete"],
        check=True,
    )
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
    if any(len(row.get("sample_correct", [])) != config.get("num_samples") or
           len(row.get("samples", [])) != config.get("num_samples")
           for row in rows):
        raise SystemExit(f"sample count mismatch in {path}")
    return {"path": str(path), "config": config, "rows": indexed}


def value(row, metric):
    if metric == "sample_accuracy":
        return statistics.mean(row["sample_correct"])
    if metric in ("generated_tokens", "tool_calls", "successful_tool_calls"):
        return float(row[metric])
    if metric == "truncated_rate":
        return row["truncated_samples"] / len(row["samples"])
    if metric == "pass_at_k":
        return int(row["pass_at_k"])
    if metric == "majority_vote_coverage":
        return int(not row["majority_vote_no_consensus"])
    if metric in ("plurality_min2_accuracy", "plurality_min2_coverage"):
        counts = Counter(
            str(answer)
            for sample in row["samples"]
            if (answer := extract_answer(sample["completion"])) is not None
        )
        answer = None
        if counts:
            highest = max(counts.values())
            winners = [candidate for candidate, count in counts.items()
                       if count == highest]
            if highest >= 2 and len(winners) == 1:
                answer = winners[0]
        if metric == "plurality_min2_coverage":
            return int(answer is not None)
        return int(answer is not None and score(f"<answer>{answer}</answer>", row["answer"]))
    return int(row["majority_vote_correct"])


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


METRICS = (
    "sample_accuracy", "pass_at_k", "majority_vote_accuracy",
    "majority_vote_coverage", "plurality_min2_accuracy", "plurality_min2_coverage",
    "generated_tokens", "tool_calls", "successful_tool_calls", "truncated_rate",
)
BINARY_METRICS = {
    "pass_at_k", "majority_vote_accuracy", "majority_vote_coverage",
    "plurality_min2_accuracy", "plurality_min2_coverage",
}


def summarize_metric(baseline, candidate, seeds, ids, metric, rng, bootstrap_samples):
    id_to_index = {question_id: index for index, question_id in enumerate(ids)}
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
        if metric in BINARY_METRICS:
            baseline_only = sum(x == -1 for x in diff)
            candidate_only = sum(x == 1 for x in diff)
            event_name = "covered" if metric.endswith("_coverage") else "correct"
            row.update({
                f"baseline_only_{event_name}": baseline_only,
                f"candidate_only_{event_name}": candidate_only,
                "exact_mcnemar_two_sided_p": exact_mcnemar(candidate_only, baseline_only),
            })
        per_seed[str(seed)] = row

    question_boot, seed_boot, hierarchical_boot = [], [], []
    for _ in range(bootstrap_samples):
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
    return {
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


def call_success_ratio(runs, seeds, ids):
    calls = sum(runs[seed]["rows"][question_id]["tool_calls"]
                for seed in seeds for question_id in ids)
    successes = sum(runs[seed]["rows"][question_id]["successful_tool_calls"]
                    for seed in seeds for question_id in ids)
    return successes / calls if calls else None


def summarize_tool_success(baseline, candidate, seeds, ids, rng, bootstrap_samples):
    per_seed = {}
    for seed in seeds:
        base_rate = call_success_ratio(baseline, [seed], ids)
        candidate_rate = call_success_ratio(candidate, [seed], ids)
        per_seed[str(seed)] = {
            "baseline_rate": base_rate,
            "candidate_rate": candidate_rate,
            "paired_difference": (candidate_rate - base_rate
                                  if base_rate is not None and candidate_rate is not None else None),
        }

    baseline_rate = call_success_ratio(baseline, seeds, ids)
    candidate_rate = call_success_ratio(candidate, seeds, ids)
    question_boot, seed_boot, hierarchical_boot = [], [], []
    for _ in range(bootstrap_samples):
        sampled_ids = [ids[rng.randrange(len(ids))] for _ in ids]
        sampled_seeds = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        for destination, selected_seeds, selected_ids in (
            (question_boot, seeds, sampled_ids),
            (seed_boot, sampled_seeds, ids),
            (hierarchical_boot, sampled_seeds, sampled_ids),
        ):
            base = call_success_ratio(baseline, selected_seeds, selected_ids)
            new = call_success_ratio(candidate, selected_seeds, selected_ids)
            if base is not None and new is not None:
                destination.append(new - base)

    paired_difference = (candidate_rate - baseline_rate
                         if baseline_rate is not None and candidate_rate is not None else None)
    seed_differences = [row["paired_difference"] for row in per_seed.values()
                        if row["paired_difference"] is not None]

    def ci(values):
        return [quantile(values, 0.025), quantile(values, 0.975)] if values else None

    return {
        "baseline_pooled_call_success_rate": baseline_rate,
        "candidate_pooled_call_success_rate": candidate_rate,
        "pooled_paired_difference": paired_difference,
        "mean_paired_difference_across_sampling_seeds": (
            statistics.mean(seed_differences) if seed_differences else None
        ),
        "question_bootstrap_95_percentile_ci": ci(question_boot),
        "sampling_seed_bootstrap_95_percentile_ci": ci(seed_boot),
        "hierarchical_bootstrap_95_percentile_ci": ci(hierarchical_boot),
        "valid_bootstrap_samples": {
            "question": len(question_boot),
            "sampling_seed": len(seed_boot),
            "hierarchical": len(hierarchical_boot),
        },
        "bootstrap_samples_requested": bootstrap_samples,
        "denominator": "successful tool calls / all tool calls",
        "per_seed": per_seed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="append", required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=5090)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--group-field", default=None,
                        help="optional row field for paired subgroup summaries; defaults to kind when present")
    parser.add_argument("--difficulty-calibration-baseline", action="append", default=None,
                        help="independent baseline sampling run used to assign empirical difficulty bins")
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
        adapters = {run["config"].get("adapter") for run in runs.values()}
        if len(adapters) != 1:
            raise SystemExit(
                f"{label} runs must use the same adapter across sampling seeds"
            )
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
    metrics = {
        metric: summarize_metric(baseline, candidate, seeds, ids, metric, rng,
                                 args.bootstrap_samples)
        for metric in METRICS
    }
    metrics["tool_execution_success_rate"] = summarize_tool_success(
        baseline, candidate, seeds, ids, rng, args.bootstrap_samples)

    group_field = args.group_field
    if group_field is None and all(
        "kind" in baseline[seeds[0]]["rows"][question_id] for question_id in ids
    ):
        group_field = "kind"
    groups = {}
    if group_field:
        for question_id in ids:
            expected_group = None
            for seed in seeds:
                base_group = baseline[seed]["rows"][question_id].get(group_field)
                candidate_group = candidate[seed]["rows"][question_id].get(group_field)
                if base_group is None or candidate_group is None:
                    raise SystemExit(f"missing group field {group_field!r} for {question_id}")
                if str(base_group) != str(candidate_group):
                    raise SystemExit(f"paired group assignment differs at seed {seed} for {question_id}")
                if expected_group is None:
                    expected_group = str(base_group)
                elif str(base_group) != expected_group:
                    raise SystemExit(f"group assignment differs across sampling seeds for {question_id}")
            groups.setdefault(expected_group, []).append(question_id)
    by_group = {
        group: {
            "n_questions": len(group_ids),
            "metrics": {
                metric: summarize_metric(baseline, candidate, seeds, group_ids, metric,
                                         rng, args.bootstrap_samples)
                for metric in METRICS
            } | {
                "tool_execution_success_rate": summarize_tool_success(
                    baseline, candidate, seeds, group_ids, rng, args.bootstrap_samples)
            },
        }
        for group, group_ids in sorted(groups.items())
    }

    difficulty_calibration = None
    by_calibrated_difficulty = {}
    if args.difficulty_calibration_baseline:
        calibration = collect(args.difficulty_calibration_baseline,
                              "difficulty-calibration baseline")
        calibration_seeds = sorted(calibration)
        if set(calibration_seeds) & set(seeds):
            raise SystemExit("difficulty-calibration seeds must be independent of evaluation seeds")
        calibration_sample_count = 0
        for seed in calibration_seeds:
            run = calibration[seed]
            if sorted(run["rows"]) != ids:
                raise SystemExit(f"difficulty-calibration question ids differ at seed {seed}")
            if run["config"].get("adapter") != ref_config.get("adapter"):
                raise SystemExit("difficulty calibration must use the paired baseline adapter")
            for key in PAIRING_KEYS:
                if run["config"].get(key) != ref_config.get(key):
                    raise SystemExit(f"difficulty-calibration setting {key} differs from baseline")
            calibration_sample_count += run["config"]["num_samples"]
        if not 4 <= calibration_sample_count <= 8:
            raise SystemExit("difficulty calibration must use 4 to 8 independent samples per question")

        difficulty_labels = {}
        for question_id in ids:
            correct = sum(
                sum(calibration[seed]["rows"][question_id]["sample_correct"])
                for seed in calibration_seeds
            )
            rate = correct / calibration_sample_count
            if rate <= 0.25:
                label = "hard"
            elif rate >= 0.75:
                label = "easy"
            else:
                label = "medium"
            difficulty_labels.setdefault(label, []).append(question_id)
        by_calibrated_difficulty = {
            label: {
                "n_questions": len(group_ids),
                "metrics": {
                    metric: summarize_metric(baseline, candidate, seeds, group_ids,
                                             metric, rng, args.bootstrap_samples)
                    for metric in METRICS
                } | {
                    "tool_execution_success_rate": summarize_tool_success(
                        baseline, candidate, seeds, group_ids, rng, args.bootstrap_samples)
                },
            }
            for label, group_ids in sorted(difficulty_labels.items())
        }
        difficulty_calibration = {
            "runs": args.difficulty_calibration_baseline,
            "sampling_seeds": calibration_seeds,
            "samples_per_question": calibration_sample_count,
            "baseline_success_rate_bins": {
                "hard": "0% to 25% correct",
                "medium": "more than 25% and less than 75% correct",
                "easy": "75% to 100% correct",
            },
            "questions_per_bin": {
                label: len(group_ids) for label, group_ids in sorted(difficulty_labels.items())
            },
            "interpretation": (
                "Difficulty labels use independent baseline sampling seeds; comparison metrics "
                "use the separate evaluation seeds. Bins are empirical and exploratory."
            ),
        }

    result = {
        "baseline_runs": args.baseline,
        "candidate_runs": args.candidate,
        "n_questions": len(ids),
        "n_sampling_seeds": len(seeds),
        "sampling_seeds": seeds,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "group_field": group_field,
        "by_group": by_group,
        "difficulty_calibration": difficulty_calibration,
        "by_calibrated_difficulty": by_calibrated_difficulty,
        "additional_voting_policy": {
            "name": "unique plurality with at least two votes",
            "minimum_votes": 2,
            "top_answer_ties": "abstain",
            "unparseable_samples": "do not vote",
        },
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
