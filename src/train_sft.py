#!/usr/bin/env python3
"""Small response-only LoRA SFT warmup for the answer-tag format."""
import argparse
import ast
import hashlib
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
    p.add_argument("--target-mode", choices=("final", "trace", "tool_trace"), default="final")
    p.add_argument("--model", default=None, help="override the model identifier from config.json")
    p.add_argument("--revision", default=None, help="pin a model revision")
    p.add_argument("--load-in-4bit", action="store_true", help="QLoRA with NF4 and BF16 compute")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    seed = cfg["seed"] if args.seed is None else args.seed
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("SFT expects exactly one visible CUDA GPU.")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_id = args.model or cfg["model"]
    revision = args.revision or (None if Path(model_id).exists() else cfg["revision"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    rows = read_jsonl(args.data)
    def encode_chat(messages):
        # Qwen3's template accepts enable_thinking; Qwen2.5-Math's does not.
        try:
            return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                                 enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)

    def encode_prefix(messages):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    examples = []
    max_length = 2048 if args.target_mode == "tool_trace" else 384
    for row in rows:
        if args.target_mode == "tool_trace":
            messages = row["messages"]
            full_ids = encode_chat(messages)
            labels = [-100] * len(full_ids)
            for message_idx, message in enumerate(messages):
                if message["role"] == "assistant":
                    start_ids = encode_prefix(messages[:message_idx])
                    end_ids = encode_chat(messages[:message_idx + 1])
                    if end_ids[:len(start_ids)] != start_ids or full_ids[:len(end_ids)] != end_ids:
                        raise RuntimeError("Chat template does not preserve assistant-turn token boundaries.")
                    labels[len(start_ids):len(end_ids)] = full_ids[len(start_ids):len(end_ids)]
        else:
            user = {"role": "user", "content": row["prompt"]}
            target = (f"<answer>{row['answer']}</answer>" if args.target_mode == "final"
                      else arithmetic_trace(row["prompt"], row["answer"]))
            messages = [user, {"role": "assistant", "content": target}]
            full_ids = encode_chat(messages)
            prompt_ids = encode_prefix([user])
            if full_ids[:len(prompt_ids)] != prompt_ids:
                raise RuntimeError("Qwen chat template did not preserve the prompt prefix.")
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        if len(full_ids) > max_length:
            raise RuntimeError(f"SFT example exceeds {max_length} tokens ({len(full_ids)}).")
        if all(label == -100 for label in labels):
            raise RuntimeError("SFT example has no assistant tokens to supervise.")
        examples.append({"input_ids": full_ids, "labels": labels})

    load_kwargs = {"revision": revision, "torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
        load_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.config.use_cache = False
    if args.load_in_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    peft_config = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    supervised_tokens = sum(sum(label != -100 for label in row["labels"]) for row in examples)
    train_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1, learning_rate=args.learning_rate,
        lr_scheduler_type="linear", warmup_ratio=0.05,
        logging_steps=1, save_strategy="no", report_to="none",
        seed=seed, bf16=True, gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=1.0, weight_decay=0.0,
        dataloader_num_workers=0, remove_unused_columns=False)
    trainer = Trainer(
        model=model, args=train_args, train_dataset=examples,
        data_collator=DynamicPaddingCollator(tokenizer.pad_token_id))
    trainer.add_callback(JsonlMetricsCallback(output))
    (output / "run_config.json").write_text(json.dumps({
        "model": model_id, "revision": revision, "seed": seed,
        "dataset": args.data, "dataset_rows": len(examples),
        "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
        "supervised_assistant_tokens": supervised_tokens,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.learning_rate, "max_sequence_length": max_length,
        "target_mode": args.target_mode,
        "assistant_end_of_turn_supervised": args.target_mode == "tool_trace",
        "quantization": "4bit_nf4_double_quant_bf16_compute" if args.load_in_4bit else "bf16",
        "gradient_checkpointing": args.gradient_checkpointing,
        "objective": ("assistant-turn-only SFT on verified multi-turn Python tool trajectories"
                      if args.target_mode == "tool_trace" else
                      "response-only SFT on verified arithmetic traces and final answer tag"
                      if args.target_mode == "trace" else
                      "response-only SFT on <answer>integer</answer>"),
        "precision": "bf16", "adapter": "LoRA",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": model_id, "rows": len(examples),
                      "epochs": args.epochs, "batch_size": args.batch_size,
                      "target_mode": args.target_mode,
                      "gpu": torch.cuda.get_device_name(0),
                      "vram_free_gib": round(torch.cuda.mem_get_info()[0] / 2**30, 2)},
                     indent=2))
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(output)
    print(f"peak_vram_gib={torch.cuda.max_memory_allocated() / 2**30:.2f}")


if __name__ == "__main__":
    main()
