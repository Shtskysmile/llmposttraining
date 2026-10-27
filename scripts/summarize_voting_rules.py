#!/usr/bin/env python3
"""Compare abstaining answer-vote thresholds on a completed multi-sample run."""
import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.verifier import extract_answer, normalize_gold


def quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def percentile_interval(values):
    if not values:
        return None
    return [quantile(values, 0.025), quantile(values, 0.975)]


def summarize(records, thresholds, strict_threshold, bootstrap_samples, rng):
    n = len(records)
    parsed = []
    for row in records:
        votes = Counter(
            str(answer)
            for sample in row["samples"]
            if (answer := extract_answer(sample["completion"])) is not None
        )
        parsed.append((row, votes, normalize_gold(str(row["answer"]))))

    results = {}
    policy_rows_by_threshold = {}
    for threshold in thresholds:
        policy_rows = []
        for _, votes, gold in parsed:
            selected = None
            selected_votes = 0
            if votes:
                highest = max(votes.values())
                winners = [answer for answer, count in votes.items() if count == highest]
                if highest >= threshold and len(winners) == 1:
                    selected = winners[0]
                    selected_votes = highest
            covered = selected is not None
            correct = covered and normalize_gold(selected) == gold
            policy_rows.append({"covered": int(covered), "correct": int(correct),
                                "selected_votes": selected_votes})
        policy_rows_by_threshold[threshold] = policy_rows

        coverage = statistics.mean(row["covered"] for row in policy_rows)
        accuracy = statistics.mean(row["correct"] for row in policy_rows)
        answered = sum(row["covered"] for row in policy_rows)
        correct_answered = sum(row["correct"] for row in policy_rows)
        answered_accuracy = correct_answered / answered if answered else None

        boot_coverage, boot_accuracy, boot_answered_accuracy = [], [], []
        for _ in range(bootstrap_samples):
            indices = [rng.randrange(n) for _ in range(n)]
            covered_n = sum(policy_rows[index]["covered"] for index in indices)
            correct_n = sum(policy_rows[index]["correct"] for index in indices)
            boot_coverage.append(covered_n / n)
            boot_accuracy.append(correct_n / n)
            if covered_n:
                boot_answered_accuracy.append(correct_n / covered_n)

        results[str(threshold)] = {
            "min_votes_for_unique_top": threshold,
            "n": n,
            "answered": answered,
            "correct_answers": correct_answered,
            "coverage": coverage,
            "coverage_question_bootstrap_95_percentile_ci": [
                quantile(boot_coverage, 0.025), quantile(boot_coverage, 0.975)],
            "accuracy_all_questions": accuracy,
            "accuracy_all_questions_question_bootstrap_95_percentile_ci": [
                quantile(boot_accuracy, 0.025), quantile(boot_accuracy, 0.975)],
            "accuracy_when_answered": answered_accuracy,
            "accuracy_when_answered_question_bootstrap_95_percentile_ci":
                percentile_interval(boot_answered_accuracy),
            "accuracy_when_answered_question_bootstrap_valid_replicates":
                len(boot_answered_accuracy),
            "mean_votes_for_answered_questions": (
                statistics.mean(row["selected_votes"] for row in policy_rows
                                if row["covered"]) if answered else None
            ),
        }

    strict_rows = policy_rows_by_threshold[strict_threshold]
    for threshold in thresholds:
        policy_rows = policy_rows_by_threshold[threshold]
        delta_coverage, delta_accuracy, delta_answered_accuracy = [], [], []
        for _ in range(bootstrap_samples):
            indices = [rng.randrange(n) for _ in range(n)]
            strict_covered = sum(strict_rows[index]["covered"] for index in indices)
            strict_correct = sum(strict_rows[index]["correct"] for index in indices)
            policy_covered = sum(policy_rows[index]["covered"] for index in indices)
            policy_correct = sum(policy_rows[index]["correct"] for index in indices)
            delta_coverage.append((policy_covered - strict_covered) / n)
            delta_accuracy.append((policy_correct - strict_correct) / n)
            if strict_covered and policy_covered:
                strict_conditional = strict_correct / strict_covered
                policy_conditional = policy_correct / policy_covered
                delta_answered_accuracy.append(policy_conditional - strict_conditional)
        point_coverage = statistics.mean(row["covered"] for row in policy_rows) - statistics.mean(
            row["covered"] for row in strict_rows)
        point_accuracy = statistics.mean(row["correct"] for row in policy_rows) - statistics.mean(
            row["correct"] for row in strict_rows)
        point_strict_covered = sum(row["covered"] for row in strict_rows)
        point_policy_covered = sum(row["covered"] for row in policy_rows)
        point_conditional = None
        if point_policy_covered and point_strict_covered:
            point_conditional = (
                sum(row["correct"] for row in policy_rows) / point_policy_covered
                - sum(row["correct"] for row in strict_rows) / point_strict_covered
            )
        results[str(threshold)]["paired_difference_vs_strict_majority"] = {
            "coverage": point_coverage,
            "coverage_question_bootstrap_95_percentile_ci": [
                quantile(delta_coverage, 0.025), quantile(delta_coverage, 0.975)],
            "accuracy_all_questions": point_accuracy,
            "accuracy_all_questions_question_bootstrap_95_percentile_ci": [
                quantile(delta_accuracy, 0.025), quantile(delta_accuracy, 0.975)],
            "accuracy_when_answered": point_conditional,
            "accuracy_when_answered_question_bootstrap_95_percentile_ci":
                percentile_interval(delta_answered_accuracy),
            "accuracy_when_answered_question_bootstrap_valid_replicates":
                len(delta_answered_accuracy),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--group-field", default="kind")
    parser.add_argument("--bootstrap-seed", type=int, default=5090)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        parser.error("bootstrap-samples must be positive")

    input_path = Path(args.input)
    config_path = input_path.with_suffix(".run_config.json")
    if not config_path.is_file():
        raise SystemExit(f"missing run config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    num_samples = config.get("num_samples")
    if (not rows or len(rows) != config.get("n") or
            not isinstance(num_samples, int) or num_samples < 1):
        raise SystemExit("run rows do not match the sampling configuration")
    ids = [row.get("id") for row in rows]
    if any(question_id is None for question_id in ids) or len(set(ids)) != len(ids):
        raise SystemExit("question ids are missing or duplicated")
    if any(len(row.get("samples", [])) != num_samples or
           len(row.get("sample_correct", [])) != num_samples for row in rows):
        raise SystemExit("incomplete sample group in input")
    if args.group_field and any(args.group_field not in row for row in rows):
        raise SystemExit(f"missing group field {args.group_field!r}")

    strict_threshold = num_samples // 2 + 1
    thresholds = sorted({1, 2, strict_threshold})
    rng = random.Random(args.bootstrap_seed)
    by_group = defaultdict(list)
    for row in rows:
        by_group[str(row[args.group_field])].append(row)
    result = {
        "input": str(input_path),
        "data_sha256": config.get("data_sha256"),
        "num_samples": num_samples,
        "vote_rule": (
            "count exact normalized answers among parseable samples; select a unique top answer "
            "only when its vote count reaches the policy minimum; otherwise abstain"
        ),
        "strict_majority_min_votes": strict_threshold,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "overall": summarize(rows, thresholds, strict_threshold,
                             args.bootstrap_samples, rng),
        "by_group": {
            group: summarize(group_rows, thresholds, strict_threshold,
                             args.bootstrap_samples, rng)
            for group, group_rows in sorted(by_group.items())
        },
        "interpretation_scope": (
            "question bootstrap for one trained model and one rollout-sampling seed; it does not "
            "estimate training-seed or rollout-seed variation. Thresholds are exploratory."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
