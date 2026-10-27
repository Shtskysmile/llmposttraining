#!/usr/bin/env python3
"""On-policy group-relative updates for multi-turn Python-tool trajectories.

This is a small, single-GPU research implementation rather than a general RL
trainer. Only generated assistant tokens receive policy gradients; calculator
outputs are fed back as context and never receive a loss.
"""
import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import torch
import peft
from peft import PeftModel, prepare_model_for_kbit_training
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tool_prompts import tool_messages
from tools import extract_python_action, run_python
from verifier import score


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def encode_messages(tokenizer, messages, device):
    kwargs = {"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt", "return_dict": True}
    try:
        encoded = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        encoded = tokenizer.apply_chat_template(messages, **kwargs)
    return encoded["input_ids"].to(device)


def sample_episode(model, tokenizer, row, device, max_new_tokens, max_total_tokens,
                   max_tool_calls, temperature, top_k, top_p, few_shot):
    messages = tool_messages(row["prompt"], few_shot=few_shot)
    turns = []
    total_tokens = 0
    calls = successful_calls = 0
    tool_events = []
    truncated = False
    model.config.use_cache = True
    model.eval()
    for _ in range(max_tool_calls + 1):
        remaining = max_total_tokens - total_tokens
        if remaining <= 0:
            break
        context_ids = encode_messages(tokenizer, messages, device)
        limit = min(max_new_tokens, remaining)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=context_ids, attention_mask=torch.ones_like(context_ids),
                do_sample=True, temperature=temperature, top_p=top_p,
                top_k=top_k,
                max_new_tokens=limit, pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id, use_cache=True,
            )[0, context_ids.shape[1]:]
        token_ids = generated.detach().cpu().tolist()
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        if not token_ids:
            break
        turns.append({"context_ids": context_ids[0].detach().cpu().tolist(), "completion_ids": token_ids})
        total_tokens += len(token_ids)
        truncated |= len(token_ids) == limit and token_ids[-1] != tokenizer.eos_token_id
        action = extract_python_action(text) if calls < max_tool_calls else None
        if action is None:
            break
        result = run_python(action)
        calls += 1
        successful_calls += int(result["ok"])
        tool_events.append({"code": action, "result": result})
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": (
            f"Tool result (success={result['ok']}):\n{result['output']}\n"
            "Continue and give the final result in <answer>...</answer>.")})
    return {"turns": turns, "completion": "\n".join(
        tokenizer.decode(turn["completion_ids"], skip_special_tokens=True) for turn in turns),
        "generated_tokens": total_tokens, "tool_calls": calls,
        "successful_tool_calls": successful_calls, "tool_events": tool_events,
        "truncated": truncated}


def transform_sampling_logits(logits, temperature, top_k, top_p):
    """Apply the same Hugging Face warpers used by generate()."""
    scores = logits / temperature
    if top_k:
        k = min(top_k, scores.shape[-1])
        threshold = torch.topk(scores, k, dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < threshold, -float("inf"))
    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=False, dim=-1)
        cumulative = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
        remove_sorted = cumulative <= (1.0 - top_p)
        remove_sorted[..., -1] = False
        remove = remove_sorted.scatter(1, sorted_indices, remove_sorted)
        scores = scores.masked_fill(remove, -float("inf"))
    return scores


def token_logprobs(model, context_ids, completion_ids, device, adapter_name,
                   temperature, top_k, top_p):
    """Recompute sampled-token log probabilities for one assistant turn."""
    model.set_adapter(adapter_name)
    prefix = torch.tensor(context_ids, dtype=torch.long, device=device)
    completion = torch.tensor(completion_ids, dtype=torch.long, device=device)
    full = torch.cat((prefix, completion)).unsqueeze(0)
    model.config.use_cache = False
    output = model(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False)
    start = prefix.numel() - 1
    logits = output.logits[0, start:start + completion.numel()]
    scores = transform_sampling_logits(logits, temperature, top_k, top_p).float()
    if torch.isnan(scores).any() or torch.isposinf(scores).any():
        raise FloatingPointError(f"non-finite {adapter_name} sampling logits")
    log_probs = scores.log_softmax(dim=-1)
    if torch.isnan(log_probs).any():
        raise FloatingPointError(f"non-finite {adapter_name} token log probabilities")
    # A sampled token can leave top-k/top-p support after an update. Keep its
    # training log-probability finite instead of passing -inf into the KL term.
    return log_probs.gather(1, completion.unsqueeze(1)).squeeze(1).clamp_min(-30.0)


def trajectory_objective(model, trajectory, device, beta, temperature, top_k, top_p,
                         kl_log_ratio_clip):
    policy_logps, kl_values = [], []
    for turn in trajectory["turns"]:
        policy = token_logprobs(model, turn["context_ids"], turn["completion_ids"], device,
                                "default", temperature, top_k, top_p)
        policy_logps.append(policy)
        if beta:
            with torch.no_grad():
                reference = token_logprobs(model, turn["context_ids"], turn["completion_ids"], device,
                                           "reference", temperature, top_k, top_p)
            # Nonnegative sampled estimate of KL(pi || pi_ref), as used in GRPO.
            log_ratio = (reference - policy).clamp(-kl_log_ratio_clip, kl_log_ratio_clip)
            kl_values.append(torch.exp(log_ratio) - log_ratio - 1.0)
    # PEFT's set_adapter toggles requires_grad on adapter weights. Restore the
    # policy before backward so both its log-probability and KL gradients flow.
    model.set_adapter("default")
    if not policy_logps:
        return None, 0.0
    mean_logp = torch.cat(policy_logps).mean()
    mean_kl = torch.cat(kl_values).mean() if kl_values else mean_logp.detach().new_zeros(())
    return mean_logp, mean_kl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="base model directory or model id")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--init-adapter", required=True, help="verified tool-trace SFT adapter")
    parser.add_argument("--data", default="data/tool_math/train.jsonl")
    parser.add_argument("--kind", choices=("constrained_counting", "conditional_probability", "modular_system"),
                        default=None, help="optional task-family filter for controlled pilots")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("fixed", "dynamic"), default="fixed")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--groups-per-step", type=int, default=1)
    parser.add_argument("--max-attempt-groups", type=int, default=8,
                        help="dynamic sampler's total group attempts per optimizer update")
    parser.add_argument("--rollout-token-budget", type=int, default=0,
                        help="stop after this many sampled assistant tokens (group overshoot is at most one group)")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-total-tokens", type=int, default=768)
    parser.add_argument("--max-tool-calls", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.02, help="KL penalty against the initial SFT adapter")
    parser.add_argument("--token-cost-weight", type=float, default=0.0,
                        help="penalty per generated-token budget fraction")
    parser.add_argument("--tool-call-cost-weight", type=float, default=0.0,
                        help="penalty per tool-call budget fraction")
    parser.add_argument("--kl-log-ratio-clip", type=float, default=5.0,
                        help="symmetric clip for sampled KL log-ratios to avoid exponential overflow")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=5090)
    parser.add_argument("--no-few-shot", action="store_true")
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="sdpa")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("Tool GRPO requires exactly one visible CUDA GPU.")
    if args.num_generations < 2 or args.groups_per_step < 1 or args.steps < 1:
        raise SystemExit("num-generations must be >= 2; groups-per-step and steps must be >= 1")
    if args.rollout_token_budget < 0:
        raise SystemExit("rollout-token-budget must be nonnegative")
    if args.max_new_tokens < 1 or args.max_total_tokens < 1 or args.max_tool_calls < 0:
        raise SystemExit("generation limits must be positive and tool-call limit nonnegative")
    if args.mode == "dynamic" and args.max_attempt_groups < args.groups_per_step:
        raise SystemExit("max-attempt-groups must be >= groups-per-step")
    if args.temperature <= 0 or args.top_k < 0 or not 0 < args.top_p <= 1:
        raise SystemExit("temperature must be positive, top-k nonnegative, and top-p in (0, 1].")
    if args.kl_log_ratio_clip <= 0:
        raise SystemExit("kl-log-ratio-clip must be positive")
    if (args.token_cost_weight < 0 or args.tool_call_cost_weight < 0 or
            args.token_cost_weight + args.tool_call_cost_weight >= 1):
        raise SystemExit("cost weights must be nonnegative and sum to less than 1")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    data_path = Path(args.data)
    raw_data = data_path.read_bytes()
    rows = read_jsonl(args.data)
    if args.kind:
        rows = [row for row in rows if row.get("kind") == args.kind]
    if not rows:
        raise SystemExit("The selected dataset contains no rows after filtering.")
    manifest_path = data_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        manifest_path = data_path.parent / "manifest.json"
    data_manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                     if manifest_path.is_file() else None)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16,
        quantization_config=quantization, device_map={"": 0},
        attn_implementation=args.attn_implementation)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base, args.init_adapter, is_trainable=True)
    model.load_adapter(args.init_adapter, adapter_name="reference", is_trainable=False)
    model.set_adapter("default")
    model.config.use_cache = False
    device = torch.device("cuda:0")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise SystemExit("No trainable adapter parameters were found.")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "dataset_rows": len(rows),
        "dataset_sha256": hashlib.sha256(raw_data).hexdigest(),
        "dataset_manifest": data_manifest,
        "precision": "nf4 base / bf16 compute",
        "gpu": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "peft_version": peft.__version__,
        "policy_objective": "group-relative REINFORCE with sampled KL penalty; no PPO clipping",
        "reference_policy": "frozen copy of the initial tool-trace SFT adapter",
        "sampling_distribution": "temperature/top-k/top-p transformed categorical distribution; logprobs use same warpers",
        "dynamic_acceptance": "retain only groups with nonzero shaped-reward variance",
        "reward_definition": (
            "correctness * (1 - token_cost_weight * generated_tokens/max_total_tokens "
            "- tool_call_cost_weight * tool_calls/max_tool_calls); ratios capped at 1; "
            "cost penalties apply only to correct trajectories"
            if args.token_cost_weight or args.tool_call_cost_weight else "exact final-answer correctness"
        ),
        "cost_penalty_scope": (
            "correct trajectories only; incorrect trajectories receive zero reward"
            if args.token_cost_weight or args.tool_call_cost_weight else "disabled"
        ),
        "token_cost_weight": args.token_cost_weight,
        "tool_call_cost_weight": args.tool_call_cost_weight,
        "correctness_reward_margin_guarantee": (
            "correct rewards are at least 1 - sum(cost weights) > 0; incorrect rewards are zero"
        ),
        "tool_output_has_policy_loss": False,
        "sampled_kl_stabilization": f"clip log(reference/policy) to +/- {args.kl_log_ratio_clip}; token log-probabilities floor at -30",
    }
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    metrics_path = output_dir / "metrics.jsonl"
    rollouts_path = output_dir / "rollouts.jsonl"
    accepted_total = attempted_total = discarded_total = optimizer_updates = 0
    sampled_trajectory_total = correct_trajectory_total = 0
    generated_tokens_total = discarded_tokens_total = tool_calls_total = 0
    started = time.perf_counter()

    steps_completed = 0
    for step in range(1, args.steps + 1):
        if args.rollout_token_budget and generated_tokens_total >= args.rollout_token_budget:
            break
        step_started = time.perf_counter()
        accepted = []
        attempted_prompt_ids = []
        step_attempts = step_discarded = 0
        step_discarded_tokens = 0
        step_trajectories = step_correct_trajectories = 0
        target_groups = args.groups_per_step
        attempt_limit = target_groups if args.mode == "fixed" else args.max_attempt_groups
        step_tokens = step_calls = step_successful_calls = step_truncated = 0
        while (len(accepted) < target_groups and step_attempts < attempt_limit and
               (not args.rollout_token_budget or
                generated_tokens_total + step_tokens < args.rollout_token_budget)):
            row = random.choice(rows)
            attempted_prompt_ids.append(row.get("id"))
            group = [sample_episode(
                model, tokenizer, row, device, args.max_new_tokens, args.max_total_tokens,
                args.max_tool_calls, args.temperature, args.top_k, args.top_p, not args.no_few_shot)
                for _ in range(args.num_generations)]
            correctness = [int(score(episode["completion"], row["answer"])) for episode in group]
            rewards = []
            for episode, correct in zip(group, correctness):
                token_cost = args.token_cost_weight * min(
                    episode["generated_tokens"] / args.max_total_tokens, 1.0)
                call_cost = args.tool_call_cost_weight * min(
                    episode["tool_calls"] / max(1, args.max_tool_calls), 1.0)
                rewards.append(correct * (1.0 - token_cost - call_cost))
            step_attempts += 1
            step_trajectories += len(group)
            step_correct_trajectories += sum(correctness)
            group_tokens = sum(episode["generated_tokens"] for episode in group)
            step_tokens += group_tokens
            step_calls += sum(episode["tool_calls"] for episode in group)
            step_successful_calls += sum(episode["successful_tool_calls"] for episode in group)
            step_truncated += sum(episode["truncated"] for episode in group)
            varying = len(set(rewards)) > 1
            group_accepted = args.mode != "dynamic" or varying
            with rollouts_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "step": step, "attempt": step_attempts, "accepted": group_accepted,
                    "prompt_id": row.get("id"), "kind": row.get("kind"), "prompt": row["prompt"],
                    "answer": row["answer"], "correctness": correctness, "rewards": rewards,
                    "episodes": [{**{key: episode[key] for key in (
                        "completion", "generated_tokens", "tool_calls", "successful_tool_calls",
                        "tool_events", "truncated")}, "correct": correct, "reward": reward}
                        for episode, correct, reward in zip(group, correctness, rewards)],
                }, ensure_ascii=False) + "\n")
            if not group_accepted:
                step_discarded += 1
                step_discarded_tokens += group_tokens
                continue
            mean_reward = statistics.mean(rewards)
            scale = (sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)) ** 0.5
            advantages = [(reward - mean_reward) / max(scale, 1e-6) for reward in rewards]
            accepted.append((row, group, correctness, rewards, advantages))

        attempted_total += step_attempts
        sampled_trajectory_total += step_trajectories
        correct_trajectory_total += step_correct_trajectories
        discarded_total += step_discarded
        accepted_total += len(accepted)
        generated_tokens_total += step_tokens
        discarded_tokens_total += step_discarded_tokens
        tool_calls_total += step_calls
        optimizer.zero_grad(set_to_none=True)
        trajectory_count = sum(len(group) for _, group, _, _, _ in accepted)
        step_loss = step_pg = step_kl = 0.0
        step_grad_norm = None
        optimizer_step_applied = False
        if trajectory_count:
            for _, group, _, rewards, advantages in accepted:
                for episode, reward, advantage in zip(group, rewards, advantages):
                    mean_logp, mean_kl = trajectory_objective(
                        model, episode, device, args.beta, args.temperature, args.top_k, args.top_p,
                        args.kl_log_ratio_clip)
                    if mean_logp is None:
                        continue
                    if not torch.isfinite(mean_logp).all() or not torch.isfinite(mean_kl).all():
                        raise FloatingPointError(f"non-finite policy objective before backward at step {step}")
                    pg_loss = -float(advantage) * mean_logp
                    loss = (pg_loss + args.beta * mean_kl) / trajectory_count
                    if not torch.isfinite(loss).all():
                        raise FloatingPointError(f"non-finite loss before backward at step {step}")
                    loss.backward()
                    step_loss += float(loss.detach())
                    step_pg += float(pg_loss.detach()) / trajectory_count
                    step_kl += float(mean_kl.detach()) / trajectory_count
            step_grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True))
            optimizer.step()
            optimizer_updates += 1
            optimizer_step_applied = True
        model.set_adapter("default")
        row = {
            "step": step, "mode": args.mode, "accepted_groups": len(accepted),
            "attempted_groups": step_attempts, "discarded_zero_variance_groups": step_discarded,
            "accepted_mean_reward": statistics.mean(
                [reward for _, _, _, rewards, _ in accepted for reward in rewards]) if accepted else None,
            "accepted_mean_correctness": statistics.mean(
                [correct for _, _, correctness, _, _ in accepted for correct in correctness]
            ) if accepted else None,
            "attempted_answer_accuracy": step_correct_trajectories / max(1, step_trajectories),
            "accepted_group_rewards": [rewards for _, _, _, rewards, _ in accepted],
            "accepted_prompt_ids": [prompt.get("id") for prompt, _, _, _, _ in accepted],
            "accepted_task_kinds": [prompt.get("kind") for prompt, _, _, _, _ in accepted],
            "attempted_prompt_ids": attempted_prompt_ids,
            "optimizer_step_applied": optimizer_step_applied,
            "loss": step_loss, "policy_loss": step_pg, "sampled_kl": step_kl,
            "gradient_norm_before_clip": step_grad_norm,
            "generated_tokens_including_discarded": step_tokens, "tool_calls": step_calls,
            "discarded_generated_tokens": step_discarded_tokens,
            "tool_call_success_rate": step_successful_calls / max(1, step_calls),
            "truncated_trajectories": step_truncated,
            "step_elapsed_seconds": time.perf_counter() - step_started,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        steps_completed += 1
        if args.rollout_token_budget and generated_tokens_total >= args.rollout_token_budget:
            break

    model.set_adapter("default")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    summary = {
        "steps": steps_completed, "configured_max_steps": args.steps,
        "mode": args.mode, "accepted_groups": accepted_total,
        "token_cost_weight": args.token_cost_weight,
        "tool_call_cost_weight": args.tool_call_cost_weight,
        "optimizer_updates": optimizer_updates,
        "attempted_groups": attempted_total, "discarded_zero_variance_groups": discarded_total,
        "sampled_trajectories": sampled_trajectory_total,
        "sampled_answer_accuracy": correct_trajectory_total / max(1, sampled_trajectory_total),
        "discard_rate": discarded_total / max(1, attempted_total),
        "generated_tokens_including_discarded": generated_tokens_total,
        "rollout_token_budget": args.rollout_token_budget or None,
        "rollout_budget_overshoot_tokens": (max(0, generated_tokens_total - args.rollout_token_budget)
                                             if args.rollout_token_budget else 0),
        "discarded_generated_tokens": discarded_tokens_total,
        "tool_calls": tool_calls_total,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
