#!/usr/bin/env python3
"""Evaluate direct or tool-assisted answers on a pinned MATH-500 JSONL file."""
import argparse
import hashlib
from importlib.metadata import version
import json
import re
import statistics
import time
from pathlib import Path

import torch
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from tools import extract_python_action, run_python
from tool_prompts import TOOL_SYSTEM


DIRECT_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
PREDICTION_EXTRACTION = [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()]


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_gold(answer):
    parsed = parse(answer, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
                   parsing_timeout=5)
    if parsed:
        return parsed, "math"
    return [normalize_text_answer(answer)], "string"


def normalize_text_answer(answer):
    answer = re.sub(r"\\(?:text|textrm|mathrm)\s*\{([^{}]*)\}", r"\1", answer)
    answer = answer.strip().strip("$").strip().strip(". ")
    return re.sub(r"\s+", " ", answer).casefold()


def balanced_boxed_contents(text):
    contents = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        start = match.end()
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    contents.append(text[start:index].strip())
                    break
    return contents


def parse_text_prediction(text):
    boxed = balanced_boxed_contents(text)
    if not boxed:
        return []
    candidate = normalize_text_answer(boxed[-1])
    return [candidate] if candidate else []


def answers_match(gold_values, predicted_values):
    for gold_value in gold_values:
        for predicted_value in predicted_values:
            if isinstance(gold_value, str) and isinstance(predicted_value, str):
                if gold_value == predicted_value:
                    return True
                continue
            try:
                if verify(gold_value, predicted_value):
                    return True
            except Exception:
                continue
    return False


def grouped_metrics(records, field):
    groups = {}
    for record in records:
        key = str(record[field])
        groups.setdefault(key, []).append(record)
    return {
        key: {
            "n": len(group),
            "accuracy": sum(row["correct"] for row in group) / len(group),
            "prediction_parse_failure_rate": sum(not row["prediction_parseable"] for row in group) / len(group),
            "mean_generated_tokens": statistics.mean(row["generated_tokens"] for row in group),
        }
        for key, group in sorted(groups.items())
    }


def generate(model, tokenizer, record, mode, max_turn_tokens, max_total_tokens, max_tool_calls):
    messages = [{"role": "system", "content": DIRECT_SYSTEM if mode == "direct" else TOOL_SYSTEM},
                {"role": "user", "content": record["prompt"]}]
    outputs = []
    tool_events = []
    generated_tokens = calls = successes = 0
    truncated = False
    for _ in range(max_tool_calls + 1):
        remaining = max_total_tokens - generated_tokens
        if remaining <= 0:
            break
        try:
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True, enable_thinking=False)
        except TypeError:
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[-1]
        budget = max_total_tokens if mode == "direct" else min(max_turn_tokens, remaining)
        with torch.inference_mode():
            result = model.generate(**encoded, do_sample=False, max_new_tokens=budget,
                                    pad_token_id=tokenizer.eos_token_id)
        token_ids = result[0, prompt_length:]
        completion = tokenizer.decode(token_ids, skip_special_tokens=True)
        count = int(token_ids.numel())
        generated_tokens += count
        outputs.append(completion)
        truncated |= count == budget and (not count or int(token_ids[-1]) != tokenizer.eos_token_id)
        action = extract_python_action(completion) if mode == "tool" and calls < max_tool_calls else None
        if not action:
            break
        result = run_python(action)
        calls += 1
        successes += int(result["ok"])
        tool_events.append({"code": action, "result": result})
        messages.extend([
            {"role": "assistant", "content": completion},
            {"role": "user", "content": (
                f"Tool result (success={result['ok']}):\n{result['output']}\n"
                "Continue and give the exact final value in <answer>...</answer> and \\boxed{}.")},
        ])
    return {
        "completion": "\n".join(outputs),
        "generated_tokens": generated_tokens,
        "tool_calls": calls,
        "successful_tool_calls": successes,
        "tool_events": tool_events,
        "truncated": truncated,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="base checkpoint path or model id")
    parser.add_argument("--adapter", default=None, help="optional LoRA adapter directory")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--data", default="data/external/math500.jsonl")
    parser.add_argument("--mode", choices=("direct", "tool"), default="direct")
    parser.add_argument("--max-turn-tokens", type=int, default=1024)
    parser.add_argument("--max-total-tokens", type=int, default=2048)
    parser.add_argument("--max-tool-calls", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.max_turn_tokens < 1 or args.max_total_tokens < 1 or args.max_tool_calls < 0:
        parser.error("token limits must be positive and tool-call limit nonnegative")

    data_path = Path(args.data)
    raw_data = data_path.read_bytes()
    manifest_path = data_path.with_suffix(".manifest.json")
    data_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    rows = [json.loads(line) for line in raw_data.decode("utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        parser.error("evaluation dataset is empty")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="sdpa")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.perf_counter()
    gold_parse_failures = []
    pred_parse_failures = 0
    with output.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            prediction = generate(model, tokenizer, row, args.mode, args.max_turn_tokens,
                                  args.max_total_tokens, args.max_tool_calls)
            gold, gold_type = parse_gold(row["answer"])
            if not gold:
                gold_parse_failures.append(row["id"])
                correct = False
                pred_values = []
            else:
                if gold_type == "string":
                    pred_values = parse_text_prediction(prediction["completion"])
                else:
                    pred_values = parse(prediction["completion"],
                                        extraction_config=PREDICTION_EXTRACTION, parsing_timeout=5)
                correct = bool(pred_values) and answers_match(gold, pred_values)
            pred_parse_failures += int(not pred_values)
            boxed = bool(re.search(r"\\boxed\s*\{", prediction["completion"]))
            record = {
                **row,
                **prediction,
                "correct": int(correct),
                "gold_type": gold_type,
                "prediction_parseable": bool(pred_values),
                "boxed_answer_present": boxed,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if (index + 1) % 25 == 0:
                print(f"evaluated {index + 1}/{len(rows)}", flush=True)

    elapsed = time.perf_counter() - started
    tool_calls = sum(row["tool_calls"] for row in records)
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "revision": args.revision,
        "data": str(data_path),
        "data_sha256": hashlib.sha256(raw_data).hexdigest(),
        "data_manifest": data_manifest,
        "mode": args.mode,
        "max_tool_calls": args.max_tool_calls if args.mode == "tool" else 0,
        "n": len(records),
        "gold_parse_failures": gold_parse_failures,
        "gold_math_answers": sum(row["gold_type"] == "math" for row in records),
        "gold_string_answers": sum(row["gold_type"] == "string" for row in records),
        "accuracy": sum(row["correct"] for row in records) / max(1, len(records)),
        "accuracy_by_level": grouped_metrics(records, "level") if "level" in records[0] else {},
        "accuracy_by_subject": grouped_metrics(records, "subject") if "subject" in records[0] else {},
        "prediction_parse_failure_rate": pred_parse_failures / max(1, len(records)),
        "boxed_answer_rate": sum(row["boxed_answer_present"] for row in records) / max(1, len(records)),
        "mean_generated_tokens": statistics.mean(row["generated_tokens"] for row in records),
        "mean_tool_calls": statistics.mean(row["tool_calls"] for row in records),
        "successful_tool_calls": sum(row["successful_tool_calls"] for row in records),
        "tool_execution_success_rate": (sum(row["successful_tool_calls"] for row in records) /
                                        max(1, tool_calls)),
        "truncated_rate": sum(row["truncated"] for row in records) / max(1, len(records)),
        "max_turn_tokens": args.max_turn_tokens,
        "elapsed_seconds": elapsed,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "math_verify_version": version("math-verify"),
        "prompt_protocol": "Qwen2.5-Math CoT/TIR boxed-answer prompt; TIR receives the restricted Python tool",
        "max_total_tokens_per_question": args.max_total_tokens,
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                   encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if gold_parse_failures:
        raise SystemExit(f"Gold answer parsing failed for {len(gold_parse_failures)} records")


if __name__ == "__main__":
    main()
