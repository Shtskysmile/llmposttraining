#!/usr/bin/env python3
"""Audit provenance, rewards, metrics, and summary for a GRPO training run."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.verifier import score


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def close(actual, expected, message, tolerance=1e-9):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool) and
            math.isfinite(actual) and abs(actual - expected) <= tolerance,
            f"{message}: got {actual!r}, expected {expected!r}")


def check_count(summary, key, expected):
    if key in summary:
        require(summary[key] == expected,
                f"summary {key} mismatch: got {summary[key]!r}, expected {expected!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="GRPO training output directory")
    parser.add_argument("--output", help="optional path for the audit JSON")
    args = parser.parse_args()

    run_dir = Path(args.input)
    config_path = run_dir / "run_config.json"
    metrics_path = run_dir / "metrics.jsonl"
    rollouts_path = run_dir / "rollouts.jsonl"
    summary_path = run_dir / "summary.json"
    for path in (config_path, metrics_path, rollouts_path, summary_path):
        require(path.is_file(), f"missing training artifact: {path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mode = config.get("mode")
    require(mode in ("fixed", "dynamic"), f"unsupported training mode: {mode!r}")
    require(summary.get("mode") == mode, "summary mode disagrees with run config")
    generations = config.get("num_generations")
    max_tokens = config.get("max_total_tokens")
    max_calls = config.get("max_tool_calls")
    require(isinstance(generations, int) and generations > 0,
            "invalid num_generations in run config")
    require(isinstance(max_tokens, int) and max_tokens > 0,
            "invalid max_total_tokens in run config")
    require(isinstance(max_calls, int) and max_calls > 0,
            "invalid max_tool_calls in run config")
    token_weight = config.get("token_cost_weight", 0.0)
    call_weight = config.get("tool_call_cost_weight", 0.0)
    for name, weight in (("token_cost_weight", token_weight),
                         ("tool_call_cost_weight", call_weight)):
        require(isinstance(weight, (int, float)) and not isinstance(weight, bool) and
                math.isfinite(weight) and weight >= 0,
                f"invalid {name} in run config")
    require(token_weight + call_weight < 1,
            "cost weights must sum to less than one")

    data_path = Path(config["data"])
    require(data_path.is_file(), f"missing training data: {data_path}")
    data_bytes = data_path.read_bytes()
    data_sha = hashlib.sha256(data_bytes).hexdigest()
    recorded_data_sha = config.get("dataset_sha256")
    data_hash_verified = recorded_data_sha is not None
    if data_hash_verified:
        require(data_sha == recorded_data_sha,
                "training data hash disagrees with run config")
    source_rows = read_jsonl(data_path)
    require(len(source_rows) == config.get("dataset_rows"),
            "training data row count disagrees with run config")
    source_by_id = {row.get("id"): row for row in source_rows}
    require(None not in source_by_id and len(source_by_id) == len(source_rows),
            "training data has missing or duplicate IDs")

    rollouts = read_jsonl(rollouts_path)
    metrics = read_jsonl(metrics_path)
    require(rollouts and metrics, "training rollout or metrics log is empty")
    require([row.get("step") for row in metrics] == list(range(1, len(metrics) + 1)),
            "metric steps are missing, duplicated, or out of order")
    require(len(metrics) <= config.get("steps", 0),
            "metrics contain more steps than configured")
    groups_by_step = defaultdict(list)
    reward_variance_groups = 0
    total_correct = total_trajectories = 0
    total_tokens = total_calls = total_discarded_tokens = 0
    total_successful_calls = total_truncated = 0

    for group_index, group in enumerate(rollouts, 1):
        step = group.get("step")
        require(isinstance(step, int) and 1 <= step <= len(metrics),
                f"rollout group {group_index}: invalid step {step!r}")
        require(group.get("attempt") == len(groups_by_step[step]) + 1,
                f"step {step}: attempt numbers are missing or out of order")
        prompt_id = group.get("prompt_id")
        source = source_by_id.get(prompt_id)
        require(source is not None, f"step {step}: unknown training prompt {prompt_id!r}")
        for key in ("prompt", "answer", "kind"):
            if key in source:
                require(group.get(key) == source[key],
                        f"step {step}, prompt {prompt_id}: {key} differs from training data")

        episodes = group.get("episodes")
        correctness = group.get("correctness")
        rewards = group.get("rewards")
        require(isinstance(episodes, list) and len(episodes) == generations,
                f"step {step}: expected {generations} episodes")
        if correctness is not None:
            require(isinstance(correctness, list) and len(correctness) == generations,
                    f"step {step}: correctness count mismatch")
        require(isinstance(rewards, list) and len(rewards) == generations,
                f"step {step}: reward count mismatch")
        expected_correctness = []
        expected_rewards = []
        for episode_index, episode in enumerate(episodes):
            correct = int(score(episode.get("completion", ""), source["answer"]))
            if "correct" in episode:
                require(episode["correct"] == correct,
                        f"step {step}, trajectory {episode_index}: verifier score mismatch")
            generated = episode.get("generated_tokens")
            calls = episode.get("tool_calls")
            successes = episode.get("successful_tool_calls")
            require(isinstance(generated, int) and 0 <= generated <= max_tokens,
                    f"step {step}, trajectory {episode_index}: invalid token count")
            require(isinstance(calls, int) and 0 <= calls <= max_calls,
                    f"step {step}, trajectory {episode_index}: invalid tool-call count")
            require(isinstance(successes, int) and 0 <= successes <= calls,
                    f"step {step}, trajectory {episode_index}: invalid tool success count")
            require(isinstance(episode.get("truncated"), bool),
                    f"step {step}, trajectory {episode_index}: invalid truncation flag")
            events = episode.get("tool_events")
            require(isinstance(events, list) and len(events) == calls,
                    f"step {step}, trajectory {episode_index}: tool-event count mismatch")
            event_successes = 0
            for event in events:
                require(isinstance(event, dict) and isinstance(event.get("result"), dict) and
                        isinstance(event["result"].get("ok"), bool),
                        f"step {step}, trajectory {episode_index}: malformed tool event")
                event_successes += int(event["result"]["ok"])
            require(event_successes == successes,
                    f"step {step}, trajectory {episode_index}: tool success mismatch")

            expected_reward = correct * (
                1.0 - token_weight * min(generated / max_tokens, 1.0)
                - call_weight * min(calls / max_calls, 1.0))
            if "reward" in episode:
                close(episode["reward"], expected_reward,
                      f"step {step}, trajectory {episode_index}: reward formula mismatch")
            expected_correctness.append(correct)
            expected_rewards.append(expected_reward)
            total_correct += correct
            total_trajectories += 1
            total_tokens += generated
            total_calls += calls
            total_successful_calls += successes
            total_truncated += int(episode["truncated"])

        if correctness is not None:
            require(correctness == expected_correctness,
                    f"step {step}: group correctness disagrees with verifier")
        group["_verified_correctness"] = expected_correctness
        require(len(rewards) == len(expected_rewards), f"step {step}: reward count mismatch")
        for episode_index, (actual, expected) in enumerate(zip(rewards, expected_rewards)):
            close(actual, expected, f"step {step}, trajectory {episode_index}: group reward mismatch")
        for episode_index, (episode, expected) in enumerate(zip(episodes, expected_rewards)):
            if "reward" in episode:
                close(episode["reward"], expected,
                      f"step {step}, trajectory {episode_index}: episode reward mismatch")
        for correct, reward in zip(expected_correctness, expected_rewards):
            if not correct:
                require(reward == 0.0,
                        f"step {step}: incorrect trajectory received nonzero reward")
            else:
                require(1.0 - token_weight - call_weight - 1e-12 <= reward <= 1.0,
                        f"step {step}: correct trajectory reward violates configured bounds")

        varying = len(set(expected_rewards)) > 1
        accepted = group.get("accepted")
        require(isinstance(accepted, bool), f"step {step}: accepted must be boolean")
        expected_accepted = True if mode == "fixed" else varying
        require(accepted == expected_accepted,
                f"step {step}: group acceptance disagrees with mode and reward variance")
        reward_variance_groups += int(varying)
        if not accepted:
            total_discarded_tokens += sum(ep["generated_tokens"] for ep in episodes)
        groups_by_step[step].append(group)

    require(set(groups_by_step) == set(range(1, len(metrics) + 1)),
            "rollout groups do not cover all metric steps")
    optimizer_updates = attempted_groups = accepted_groups = discarded_groups = 0
    for row in metrics:
        step = row["step"]
        groups = groups_by_step[step]
        accepted = [group for group in groups if group["accepted"]]
        discarded = [group for group in groups if not group["accepted"]]
        episodes = [episode for group in groups for episode in group["episodes"]]
        correct_count = sum(correct for group in groups
                            for correct in group["_verified_correctness"])
        token_count = sum(episode["generated_tokens"] for episode in episodes)
        call_count = sum(episode["tool_calls"] for episode in episodes)
        success_count = sum(episode["successful_tool_calls"] for episode in episodes)
        truncated_count = sum(int(episode["truncated"]) for episode in episodes)
        require(row.get("mode") == mode, f"step {step}: metrics mode mismatch")
        require(row.get("attempted_groups") == len(groups),
                f"step {step}: attempted group count mismatch")
        require(row.get("accepted_groups") == len(accepted),
                f"step {step}: accepted group count mismatch")
        require(row.get("discarded_zero_variance_groups") == len(discarded),
                f"step {step}: discarded group count mismatch")
        require(row.get("attempted_prompt_ids") == [g["prompt_id"] for g in groups],
                f"step {step}: attempted prompt IDs mismatch")
        require(row.get("accepted_prompt_ids") == [g["prompt_id"] for g in accepted],
                f"step {step}: accepted prompt IDs mismatch")
        require(row.get("accepted_task_kinds") == [g.get("kind") for g in accepted],
                f"step {step}: accepted task kinds mismatch")
        require(row.get("accepted_group_rewards") == [g["rewards"] for g in accepted],
                f"step {step}: accepted rewards mismatch")
        check_count(row, "generated_tokens_including_discarded", token_count)
        check_count(row, "discarded_generated_tokens",
                    sum(ep["generated_tokens"] for group in discarded
                        for ep in group["episodes"]))
        check_count(row, "tool_calls", call_count)
        check_count(row, "truncated_trajectories", truncated_count)
        if "tool_call_success_rate" in row:
            close(row["tool_call_success_rate"], success_count / max(1, call_count),
                  f"step {step}: tool-call success rate mismatch")
        if "attempted_answer_accuracy" in row:
            close(row["attempted_answer_accuracy"], correct_count / max(1, len(episodes)),
                  f"step {step}: attempted answer accuracy mismatch")
        accepted_rewards = [reward for group in accepted for reward in group["rewards"]]
        accepted_correct = [correct for group in accepted
                            for correct in group["_verified_correctness"]]
        if "accepted_mean_reward" in row:
            expected = statistics.mean(accepted_rewards) if accepted_rewards else None
            if expected is None:
                require(row["accepted_mean_reward"] is None,
                        f"step {step}: expected null accepted_mean_reward")
            else:
                close(row["accepted_mean_reward"], expected,
                      f"step {step}: accepted mean reward mismatch")
        if "accepted_mean_correctness" in row:
            expected = statistics.mean(accepted_correct) if accepted_correct else None
            if expected is None:
                require(row["accepted_mean_correctness"] is None,
                        f"step {step}: expected null accepted_mean_correctness")
            else:
                close(row["accepted_mean_correctness"], expected,
                      f"step {step}: accepted mean correctness mismatch")

        applied = row.get("optimizer_step_applied")
        require(isinstance(applied, bool) and applied == bool(accepted),
                f"step {step}: optimizer update status disagrees with accepted groups")
        if applied:
            optimizer_updates += 1
            norm = row.get("gradient_norm_before_clip")
            require(isinstance(norm, (int, float)) and not isinstance(norm, bool) and
                    math.isfinite(norm) and norm >= 0,
                    f"step {step}: invalid pre-clip gradient norm")
        attempted_groups += len(groups)
        accepted_groups += len(accepted)
        discarded_groups += len(discarded)

        for key in ("loss", "policy_loss", "sampled_kl", "step_elapsed_seconds",
                    "elapsed_seconds", "peak_vram_gib"):
            value = row.get(key)
            require(isinstance(value, (int, float)) and not isinstance(value, bool) and
                    math.isfinite(value), f"step {step}: invalid metric {key}")

    require(summary.get("steps") == len(metrics), "summary step count mismatch")
    require(summary.get("configured_max_steps") == config.get("steps"),
            "summary configured steps disagree with run config")
    require(summary.get("optimizer_updates") == optimizer_updates,
            "summary optimizer update count mismatch")
    check_count(summary, "attempted_groups", attempted_groups)
    check_count(summary, "accepted_groups", accepted_groups)
    check_count(summary, "discarded_zero_variance_groups", discarded_groups)
    check_count(summary, "sampled_trajectories", total_trajectories)
    check_count(summary, "generated_tokens_including_discarded", total_tokens)
    check_count(summary, "discarded_generated_tokens", total_discarded_tokens)
    check_count(summary, "tool_calls", total_calls)
    if "sampled_answer_accuracy" in summary:
        close(summary["sampled_answer_accuracy"],
              total_correct / max(1, total_trajectories),
              "summary sampled answer accuracy mismatch")
    budget = config.get("rollout_token_budget", 0) or 0
    check_count(summary, "rollout_token_budget", budget or None)
    check_count(summary, "rollout_budget_overshoot_tokens", max(0, total_tokens - budget))
    check_count(summary, "token_cost_weight", token_weight)
    check_count(summary, "tool_call_cost_weight", call_weight)
    adapter_path = run_dir / "adapter_model.safetensors"
    require(adapter_path.is_file() and adapter_path.stat().st_size > 0,
            f"missing or empty final adapter: {adapter_path}")

    result = {
        "run_dir": str(run_dir),
        "mode": mode,
        "seed": config.get("seed"),
        "dataset_sha256": data_sha,
        "dataset_hash_verified_against_run_config": data_hash_verified,
        "metric_steps": len(metrics),
        "attempted_groups": attempted_groups,
        "accepted_groups": accepted_groups,
        "discarded_groups": discarded_groups,
        "optimizer_updates": optimizer_updates,
        "sampled_trajectories": total_trajectories,
        "sampled_answer_accuracy": total_correct / max(1, total_trajectories),
        "generated_tokens": total_tokens,
        "rollout_budget_overshoot_tokens": max(0, total_tokens - budget),
        "tool_calls": total_calls,
        "reward_variance_groups": reward_variance_groups,
        "verified_reward_formula": True,
        "verified_verifier_scores": total_trajectories,
        "final_adapter_bytes": adapter_path.stat().st_size,
    }
    if not data_hash_verified:
        result["provenance_warning"] = (
            "run config did not record dataset_sha256; current data hash and row count "
            "cannot prove which historical file contents were used"
        )
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"GRPO training-run audit failed: {exc}") from exc
