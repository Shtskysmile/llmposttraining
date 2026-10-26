#!/usr/bin/env python3
"""Direct and Python-tool-assisted rollout evaluation for tool-math JSONL."""
import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tools import extract_python_action, run_python
from tool_prompts import tool_messages
from verifier import format_error, score


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def generate_one(model, tokenizer, prompt, mode, max_new_tokens, max_tool_calls, max_total_tokens,
                 do_sample=False, temperature=0.7, top_p=0.95, top_k=50):
    prompt = prompt + "\nPut the exact final value in <answer>...</answer>."
    if mode == "tool":
        messages = tool_messages(prompt[:-len("\nPut the exact final value in <answer>...</answer>.")])
    else:
        messages = [{"role": "system", "content":
            "Solve the math problem directly without tools. Keep the solution concise and put the exact final value in <answer>...</answer>."},
            {"role": "user", "content": prompt}]
    whole = []
    tool_events = []
    calls = successful_calls = tokens = 0
    truncated = False
    for turn in range(max_tool_calls + 1):
        remaining = max_total_tokens - tokens
        if remaining <= 0:
            break
        try:
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True, enable_thinking=False)
        except TypeError:
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        input_len = encoded["input_ids"].shape[-1]
        turn_budget = min(max_new_tokens, remaining)
        with torch.inference_mode():
            generation = {"do_sample": do_sample, "max_new_tokens": turn_budget,
                          "pad_token_id": tokenizer.eos_token_id}
            if do_sample:
                generation.update(temperature=temperature, top_p=top_p, top_k=top_k)
            out = model.generate(**encoded, **generation)
        generated = out[0, input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        tokens += int(generated.numel())
        truncated |= generated.numel() == turn_budget and (not generated.numel() or
            int(generated[-1]) != tokenizer.eos_token_id)
        whole.append(text)
        action = extract_python_action(text) if mode == "tool" and calls < max_tool_calls else None
        if action is None:
            break
        result = run_python(action)
        calls += 1
        successful_calls += int(result["ok"])
        tool_events.append({"code": action, "result": result})
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": f"Tool result (success={result['ok']}):\n{result['output']}\nContinue and give the final result in <answer>...</answer> and \\boxed{{}}."})
    return {"completion": "\n".join(whole), "tool_calls": calls,
            "successful_tool_calls": successful_calls, "tool_events": tool_events,
            "generated_tokens": tokens, "truncated": truncated}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Math-7B-Instruct")
    p.add_argument("--base-model", default=None, help="base model when --model points to a LoRA adapter")
    p.add_argument("--revision", default=None)
    p.add_argument("--data", default="data/tool_math/test.jsonl")
    p.add_argument("--mode", choices=("direct", "tool"), default="direct")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-total-tokens", type=int, default=1024,
                   help="total generated-token budget per question across tool turns")
    p.add_argument("--max-tool-calls", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=5090)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; rollout evaluation requires a GPU.")
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model_id = args.base_model or args.model
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=args.revision,
        torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    if args.base_model:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.model)
    rows = load_rows(args.data)
    if args.limit:
        rows = rows[:args.limit]
    output = Path(args.output or f"runs/tool_math_{args.mode}.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)
    raw_data = data_path.read_bytes()
    manifest_path = data_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        manifest_path = data_path.parent / "manifest.json"
    data_manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                     if manifest_path.is_file() else None)
    run_config = {
        "model": args.model, "base_model": args.base_model, "revision": args.revision,
        "data": str(data_path), "data_sha256": hashlib.sha256(raw_data).hexdigest(),
        "data_manifest": data_manifest, "n": len(rows), "mode": args.mode,
        "decoding": "greedy", "seed": args.seed,
        "max_new_tokens_per_turn": args.max_new_tokens,
        "max_total_tokens_per_question": args.max_total_tokens,
        "max_tool_calls": args.max_tool_calls if args.mode == "tool" else 0,
        "gpu": torch.cuda.get_device_name(0),
    }
    output.with_suffix(".run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    records = []
    started = time.perf_counter()
    with output.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(rows):
            pred = generate_one(model, tokenizer, row["prompt"], args.mode, args.max_new_tokens,
                                args.max_tool_calls, args.max_total_tokens)
            record = {**row, **pred, "correct": score(pred["completion"], row["answer"]),
                      "format_error": format_error(pred["completion"])}
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if (idx + 1) % 10 == 0:
                print(f"evaluated {idx + 1}/{len(rows)}", flush=True)
    elapsed = time.perf_counter() - started
    n = max(1, len(records))
    summary = {"model": args.model, "base_model": args.base_model,
        "revision": args.revision, "data": args.data, "mode": args.mode,
        "data_sha256": hashlib.sha256(raw_data).hexdigest(),
        "data_manifest": data_manifest, "seed": args.seed,
        "decoding": "greedy", "max_new_tokens_per_turn": args.max_new_tokens,
        "max_total_tokens_per_question": args.max_total_tokens,
        "max_tool_calls": args.max_tool_calls if args.mode == "tool" else 0,
        "n": len(records), "accuracy": sum(r["correct"] for r in records) / n,
        "format_error_rate": sum(r["format_error"] for r in records) / n,
        "mean_generated_tokens": statistics.mean(r["generated_tokens"] for r in records) if records else 0,
        "mean_tool_calls": statistics.mean(r["tool_calls"] for r in records) if records else 0,
        "tool_execution_success_rate": (sum(r["successful_tool_calls"] for r in records) /
            max(1, sum(r["tool_calls"] for r in records))),
        "truncated_rate": sum(r["truncated"] for r in records) / n,
        "elapsed_seconds": elapsed, "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30}
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
