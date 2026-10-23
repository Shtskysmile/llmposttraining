#!/usr/bin/env python3
"""Single-GPU QLoRA GRPO runner. Fails early if CUDA is unavailable."""
import argparse
import json
import math
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback

from verifier import score


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def reward_func(completions, answer, **kwargs):
    return [score(c[0]["content"] if isinstance(c, list) else c, a) for c, a in zip(completions, answer)]


class JsonlMetricsCallback(TrainerCallback):
    """Persist TRL metrics plus derived diagnostics and GPU memory."""
    def __init__(self, output_dir):
        self.path = Path(output_dir) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        row = {"step": state.global_step, **(logs or {})}
        # Keep outcome entropy distinct from the sampled-token entropy below.
        p = row.get("rewards/reward_func/mean")
        if p is not None:
            p = min(1.0, max(0.0, float(p)))
            row["reward_outcome_entropy_bits"] = 0.0 if p in (0.0, 1.0) else -p * math.log2(p) - (1-p) * math.log2(1-p)
        if torch.cuda.is_available():
            row["gpu_allocated_gib"] = torch.cuda.memory_allocated() / 2**30
            row["gpu_reserved_gib"] = torch.cuda.memory_reserved() / 2**30
            row["gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return control


class EntropyGRPOTrainer(GRPOTrainer):
    """Add empirical within-group token entropy (exploration/diversity proxy)."""
    def _generate_and_score_completions(self, inputs):
        result = super()._generate_and_score_completions(inputs)
        token_ids, mask = result["completion_ids"], result["completion_mask"].bool()
        group = self.num_generations
        entropies = []
        for start in range(0, token_ids.shape[0], group):
            ids, valid = token_ids[start:start + group], mask[start:start + group]
            # Compute per-position empirical entropy in one device operation.
            # The previous position-by-position loop repeatedly synchronized
            # CPU and GPU even though this value is only a diagnostic metric.
            same_token = ids[:, None, :] == ids[None, :, :]
            same_token &= valid[None, :, :]
            counts = same_token.sum(dim=1).float()
            valid_counts = valid.sum(dim=0).clamp_min(1).float()
            log_probs = (counts / valid_counts[None, :]).clamp_min(1e-12).log2()
            per_position = -(log_probs * valid).sum(dim=0) / valid_counts
            has_tokens = valid.any(dim=0)
            if has_tokens.any():
                entropies.append(per_position[has_tokens].mean())
        if entropies:
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["group_token_entropy_bits"].append(torch.stack(entropies).mean().item())
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--data", default="data/train.jsonl")
    p.add_argument("--output", default="runs/grpo")
    p.add_argument("--model", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--completion-length", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="use 0.6B, 3 steps, 2 samples and 96 tokens")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable (torch.cuda.is_available() is false); no training was started.")
    if torch.cuda.device_count() != 1:
        raise SystemExit(f"Expected exactly one visible GPU, found {torch.cuda.device_count()}.")
    seed = cfg["seed"] if args.seed is None else args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_name = args.model or (cfg["smoke_model"] if args.smoke else cfg["model"])
    steps = args.steps or (3 if args.smoke else cfg["max_steps"])
    generations = args.generations or (2 if args.smoke else cfg["num_generations"])
    completion_length = args.completion_length or (96 if args.smoke else cfg["max_completion_length"])
    if cfg["per_device_train_batch_size"] < generations or cfg["per_device_train_batch_size"] % generations:
        raise SystemExit("per_device_train_batch_size must be a multiple of --generations for TRL GRPO.")

    # Render the model's chat template once, with Qwen3 thinking mode disabled,
    # then pass plain rendered prompts to the version-pinned TRL trainer.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=cfg["revision"])
    rows = read_jsonl(args.data)
    for row in rows:
        row["prompt"] = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
    ds = Dataset.from_list(rows)
    lora = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    train_args = GRPOConfig(
        output_dir=args.output, max_steps=steps,
        per_device_train_batch_size=cfg["per_device_train_batch_size"], gradient_accumulation_steps=1,
        num_generations=generations, max_prompt_length=cfg["max_prompt_length"],
        max_completion_length=completion_length, learning_rate=cfg["learning_rate"],
        logging_steps=1, save_strategy="no", report_to="none", seed=seed,
        bf16=True, gradient_checkpointing=False, disable_dropout=True, scale_rewards=False,
        model_init_kwargs={"revision": cfg["revision"], "torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"},
        log_completions=True, num_completions_to_print=generations,
    )
    trainer = EntropyGRPOTrainer(model=model_name, reward_funcs=reward_func, args=train_args,
                                 train_dataset=ds, peft_config=lora, processing_class=tokenizer)
    trainer.add_callback(JsonlMetricsCallback(args.output))
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.output, "run_config.json").write_text(json.dumps({
        "model": model_name, "revision": cfg["revision"], "steps": steps, "generations": generations,
        "max_completion_length": completion_length, "seed": seed,
        "dataset": args.data, "dataset_rows": len(ds), "precision": "bf16", "adapter": "LoRA",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": model_name, "steps": steps, "generations": generations,
                      "max_completion_length": completion_length, "seed": seed,
                      "gpu": torch.cuda.get_device_name(0),
                      "vram_free_gib": round(torch.cuda.mem_get_info()[0] / 2**30, 2),
                      "vram_total_gib": round(torch.cuda.mem_get_info()[1] / 2**30, 2)}, indent=2))
    trainer.train()
    trainer.save_model(args.output)
    print(f"peak_vram_gib={torch.cuda.max_memory_allocated() / 2**30:.2f}")


if __name__ == "__main__":
    main()
