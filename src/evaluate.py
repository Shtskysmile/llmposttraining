#!/usr/bin/env python3
"""Greedy/pass@k evaluation with answer-format and token-length reporting."""
import argparse
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verifier import format_error, score
from train import read_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="base id for baseline, adapter directory after training")
    p.add_argument("--base-model", default=None, help="required when --model is an adapter")
    p.add_argument("--revision", default=None, help="model revision; defaults to pinned config.json commit")
    p.add_argument("--data", default="data/test.jsonl")
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1,
                   help="number of prompts generated together; use the same value for comparisons")
    p.add_argument("--output", default="runs/eval.jsonl")
    args = p.parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; GPU model evaluation cannot run here.")
    config = json.loads(Path("config.json").read_text())
    seed = config["seed"] if args.seed is None else args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_id = args.base_model or args.model
    revision = args.revision or config["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16,
                                                  device_map="cuda", attn_implementation="sdpa",
                                                  revision=revision)
    if args.base_model:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model)
    rows = read_jsonl(args.data)
    records, total_tokens, errs, correct1, correctk = [], 0, 0, 0, 0
    group_stds, zero_var_groups, reward_sum = [], 0, 0
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start:start + args.batch_size]
        messages = [[{"role": "user", "content": row["prompt"]}] for row in batch_rows]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            padding=True, return_tensors="pt", return_dict=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_width = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inputs, do_sample=args.samples > 1, temperature=0.7 if args.samples > 1 else None,
                                 num_return_sequences=args.samples, max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tokenizer.eos_token_id)
        for index, row in enumerate(batch_rows):
            sequences = out[index * args.samples:(index + 1) * args.samples]
            completions = [tokenizer.decode(seq[prompt_width:], skip_special_tokens=True)
                           for seq in sequences]
            scores = [score(c, row["answer"]) for c in completions]
            group_std = statistics.pstdev(scores)
            group_stds.append(group_std)
            zero_var_groups += int(len(set(scores)) == 1)
            reward_sum += sum(scores)
            lengths = [len(tokenizer.encode(c, add_special_tokens=False)) for c in completions]
            total_tokens += sum(lengths)
            errs += sum(format_error(c) for c in completions)
            correct1 += scores[0]
            correctk += int(any(scores))
            records.append({"prompt": row["prompt"], "gold": row["answer"], "completions": completions,
                            "scores": scores, "group_reward_std": group_std, "lengths": lengths})
    n = len(rows)
    summary = {"n": n, "seed": seed, "model": args.model, "revision": revision,
               "pass@1": correct1 / n, f"pass@{args.samples}": correctk / n,
               "mean_reward": reward_sum / (n * args.samples),
               "mean_group_reward_std": sum(group_stds) / n,
               "zero_variance_group_rate": zero_var_groups / n,
               "mean_generated_tokens": total_tokens / (n * args.samples),
               "format_error_rate": errs / (n * args.samples)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
