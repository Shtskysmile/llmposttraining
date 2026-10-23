#!/usr/bin/env python3
"""Small response-only LoRA SFT warmup for the answer-tag format."""
import argparse
import ast
import json
import re
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainerCallback, TrainingArguments)

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def arithmetic_trace(prompt, expected):
    match = re.search(r"Compute exactly:\s*(.*?)\. Return the final integer", prompt)
    if not match:
        raise ValueError(f"Cannot extract expression from prompt: {prompt}")
    tree = ast.parse(match.group(1), mode="eval").body
    steps = []

    def visit(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            raise ValueError("Unsupported operation in generated arithmetic expression.")
        left, right = visit(node.left), visit(node.right)
        if isinstance(node.op, ast.Add):
            symbol, value = "+", left + right
        elif isinstance(node.op, ast.Sub):
            symbol, value = "-", left - right
        else:
            symbol, value = "×", left * right
        steps.append(f"{left} {symbol} {right} = {value}")
        return value

    value = visit(tree)
    if value != int(expected):
        raise ValueError(f"Generated trace answer {value} does not match gold {expected}.")
    explanation = "Let's calculate one operation at a time.\n" + "\n".join(steps)
    return explanation + f"\n<answer>{value}</answer>"


class JsonlMetricsCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.path = Path(output_dir) / "metrics.jsonl"

    def on_log(self, args, state, control, logs=None, **kwargs):
        row = {"step": state.global_step, **(logs or {})}
        if torch.cuda.is_available():
            row["gpu_allocated_gib"] = torch.cuda.memory_allocated() / 2**30
            row["gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return control


class DynamicPaddingCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        width = max(len(row["input_ids"]) for row in features)
        input_ids, attention_mask, labels = [], [], []
        for row in features:
            n = len(row["input_ids"])
            pad = width - n
            input_ids.append(row["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append([1] * n + [0] * pad)
            labels.append(row["labels"] + [-100] * pad)
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.json")
    p.add_argument("--data", default="data/train.jsonl")
    p.add_argument("--output", default="runs/sft_format_seed17")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--target-mode", choices=("final", "trace"), default="final")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    seed = cfg["seed"] if args.seed is None else args.seed
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("SFT expects exactly one visible CUDA GPU.")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_id, revision = cfg["model"], cfg["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    rows = read_jsonl(args.data)
    examples = []
    for row in rows:
        user = {"role": "user", "content": row["prompt"]}
        target = (f"<answer>{row['answer']}</answer>" if args.target_mode == "final"
                  else arithmetic_trace(row["prompt"], row["answer"]))
        assistant = {"role": "assistant", "content": target}
        prompt_ids = tokenizer.apply_chat_template(
            [user], tokenize=True, add_generation_prompt=True,
            enable_thinking=False)
        full_ids = tokenizer.apply_chat_template(
            [user, assistant], tokenize=True, add_generation_prompt=False,
            enable_thinking=False)
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise RuntimeError("Qwen chat template did not preserve the prompt prefix.")
        if len(full_ids) > 384:
            raise RuntimeError(f"SFT example exceeds 384 tokens ({len(full_ids)}).")
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        examples.append({"input_ids": full_ids, "labels": labels})

    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa")
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1, learning_rate=args.learning_rate,
        lr_scheduler_type="linear", warmup_ratio=0.05,
        logging_steps=1, save_strategy="no", report_to="none",
        seed=seed, bf16=True, gradient_checkpointing=False,
        max_grad_norm=1.0, weight_decay=0.0,
        dataloader_num_workers=0, remove_unused_columns=False)
    trainer = Trainer(
        model=model, args=train_args, train_dataset=examples,
        data_collator=DynamicPaddingCollator(tokenizer.pad_token_id))
    trainer.add_callback(JsonlMetricsCallback(output))
    (output / "run_config.json").write_text(json.dumps({
        "model": model_id, "revision": revision, "seed": seed,
        "dataset": args.data, "dataset_rows": len(examples),
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "max_sequence_length": 384,
        "target_mode": args.target_mode,
        "objective": "response-only SFT on verified arithmetic traces and final answer tag"
        if args.target_mode == "trace" else "response-only SFT on <answer>integer</answer>",
        "precision": "bf16", "adapter": "LoRA",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": model_id, "rows": len(examples),
                      "epochs": args.epochs, "batch_size": args.batch_size,
                      "gpu": torch.cuda.get_device_name(0),
                      "vram_free_gib": round(torch.cuda.mem_get_info()[0] / 2**30, 2)},
                     indent=2))
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    print(f"peak_vram_gib={torch.cuda.max_memory_allocated() / 2**30:.2f}")


if __name__ == "__main__":
    main()
