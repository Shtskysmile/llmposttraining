#!/usr/bin/env python3
"""Token-budget-matched multi-sample and majority-vote evaluation."""
import argparse
from collections import Counter
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from rollout import generate_one, load_rows
from verifier import extract_answer, score


def vote_key(completion):
    """Return an exact rational answer key, or None when no answer parses."""
    answer = extract_answer(completion)
    return str(answer) if answer is not None else None


def majority_vote(completions):
    keys = [vote_key(completion) for completion in completions]
    counts = Counter(key for key in keys if key is not None)
    if not counts:
        return None, 0, True
    largest = max(counts.values())
    tied = {key for key, count in counts.items() if count == largest}
    # Require a strict majority of all samples. Invalid/unparseable samples do
    # not create a vote, but still count in the denominator. Ties and pluralities
    # without a majority produce no selected answer.
    if len(tied) != 1 or largest <= len(completions) / 2:
        return None, largest, True
    return next(iter(tied)), largest, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--data", default="data/tool_math/test.jsonl")
    parser.add_argument("--mode", choices=("direct", "tool"), default="tool")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-total-tokens", type=int, default=1024,
                        help="total generated-token budget per question across all samples")
    parser.add_argument("--max-tool-calls", type=int, default=2,
                        help="per sampled answer")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=5090)
    parser.add_argument("--resume", action="store_true",
                        help="resume a prefix-complete run with a matching run config")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.num_samples < 2:
        parser.error("num-samples must be at least 2")
    if args.max_total_tokens < args.num_samples:
        parser.error("max-total-tokens must be at least num-samples")
    if args.max_new_tokens < 1 or args.max_tool_calls < 0:
        parser.error("token limit must be positive and tool-call limit nonnegative")
    if args.temperature <= 0 or args.top_k < 0 or not 0 < args.top_p <= 1:
        parser.error("temperature must be positive, top-k nonnegative, and top-p in (0, 1]")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; sampling evaluation requires a GPU.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    data_path = Path(args.data)
    raw_data = data_path.read_bytes()
    rows = load_rows(args.data)
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        parser.error("evaluation dataset is empty")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config_path = output.with_suffix(".run_config.json")
    manifest_path = data_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        manifest_path = data_path.parent / "manifest.json"
    data_manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                     if manifest_path.is_file() else None)
    per_sample_budgets = [args.max_total_tokens // args.num_samples] * args.num_samples
    for index in range(args.max_total_tokens % args.num_samples):
        per_sample_budgets[index] += 1
    run_config = vars(args).copy()
    run_config.pop("resume")
    run_config |= {
        "n": len(rows), "data_sha256": hashlib.sha256(raw_data).hexdigest(),
        "data_manifest": data_manifest,
        "sampling": "temperature/top-k/top-p categorical sampling",
        "per_sample_token_budgets": per_sample_budgets,
        "seed_strategy": "base seed plus source row index, reset before each question",
        "majority_vote_rule": "require one exact answer with strictly more than half of all samples",
        "unparseable_answers_in_vote": "no vote, but remain in the denominator",
        "gpu": torch.cuda.get_device_name(0),
    }

    row_ids = [row["id"] for row in rows]
    if len(set(row_ids)) != len(row_ids):
        parser.error("evaluation dataset contains duplicate ids")
    rows_by_id = {row["id"]: row for row in rows}
    records = []
    completed_ids = set()
    if args.resume:
        if not output.is_file() or not config_path.is_file():
            parser.error("resume requires an existing JSONL output and .run_config.json")
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        mismatches = [key for key, value in run_config.items()
                      if saved_config.get(key) != value]
        if mismatches:
            parser.error("resume configuration mismatch for: " + ", ".join(mismatches))
        raw_output = output.read_bytes()
        if raw_output and not raw_output.endswith(b"\n"):
            last_newline = raw_output.rfind(b"\n")
            trailing = raw_output[last_newline + 1:]
            try:
                json.loads(trailing)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raw_output = raw_output[:last_newline + 1] if last_newline >= 0 else b""
            else:
                raw_output += b"\n"
            output.write_bytes(raw_output)
        records = [json.loads(line) for line in raw_output.decode("utf-8").splitlines()
                   if line.strip()]
        completed = [record.get("id") for record in records]
        if len(set(completed)) != len(completed):
            parser.error("resume output contains duplicate ids")
        if completed != row_ids[:len(completed)]:
            parser.error("resume output must contain a prefix of dataset ids in original order")
        for record in records:
            source = rows_by_id[record["id"]]
            if record.get("prompt") != source.get("prompt") or record.get("answer") != source.get("answer"):
                parser.error("resume output contains a row that differs from source data")
            if len(record.get("samples", [])) != args.num_samples:
                parser.error("resume output has an incomplete sample group")
            if len(record.get("sample_correct", [])) != args.num_samples:
                parser.error("resume output has incomplete correctness labels")
        completed_ids = set(completed)
    else:
        config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
        output.touch()

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="sdpa")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    started = time.perf_counter()
    with output.open("a" if args.resume else "w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            if row["id"] in completed_ids:
                continue
            sample_seed = (args.seed + index) % (2 ** 32)
            random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            torch.cuda.manual_seed_all(sample_seed)
            question_started = time.perf_counter()
            samples = []
            for budget in per_sample_budgets:
                samples.append(generate_one(
                    model, tokenizer, row["prompt"], args.mode,
                    min(args.max_new_tokens, budget), args.max_tool_calls, budget,
                    do_sample=True, temperature=args.temperature, top_p=args.top_p,
                    top_k=args.top_k))
            completions = [sample["completion"] for sample in samples]
            correct = [int(score(completion, row["answer"])) for completion in completions]
            parseable_count = sum(vote_key(completion) is not None for completion in completions)
            winner, vote_count, vote_tie = majority_vote(completions)
            vote_correct = int(winner is not None and score(f"<answer>{winner}</answer>", row["answer"]))
            record = {
                **row,
                "samples": samples,
                "sample_correct": correct,
                "pass_at_k": int(any(correct)),
                "majority_answer": winner,
                "majority_vote_count": vote_count,
                "parseable_answer_count": parseable_count,
                "majority_vote_no_consensus": vote_tie,
                "majority_vote_correct": vote_correct,
                "generated_tokens": sum(sample["generated_tokens"] for sample in samples),
                "tool_calls": sum(sample["tool_calls"] for sample in samples),
                "successful_tool_calls": sum(sample["successful_tool_calls"] for sample in samples),
                "truncated_samples": sum(sample["truncated"] for sample in samples),
                "sampling_seed": sample_seed,
                "elapsed_seconds": time.perf_counter() - question_started,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if (index + 1) % 10 == 0:
                print(f"evaluated {index + 1}/{len(rows)}", flush=True)

    elapsed = time.perf_counter() - started
    total_calls = sum(row["tool_calls"] for row in records)
    summary = {
        "model": args.model, "adapter": args.adapter, "revision": args.revision,
        "seed": args.seed, "temperature": args.temperature,
        "top_k": args.top_k, "top_p": args.top_p,
        "data": args.data, "data_sha256": hashlib.sha256(raw_data).hexdigest(),
        "data_manifest": data_manifest,
        "mode": args.mode, "n": len(records), "num_samples": args.num_samples,
        "total_token_budget_per_question": args.max_total_tokens,
        "single_sample_accuracy_with_budget_split": statistics.mean(
            correct for row in records for correct in row["sample_correct"]),
        "pass_at_k": statistics.mean(row["pass_at_k"] for row in records),
        "majority_vote_accuracy": statistics.mean(row["majority_vote_correct"] for row in records),
        "majority_vote_no_consensus_rate": statistics.mean(
            row["majority_vote_no_consensus"] for row in records),
        "no_parseable_vote_rate": statistics.mean(row["parseable_answer_count"] == 0 for row in records),
        "mean_generated_tokens_per_question": statistics.mean(row["generated_tokens"] for row in records),
        "mean_tool_calls_per_question": statistics.mean(row["tool_calls"] for row in records),
        "mean_question_elapsed_seconds": statistics.mean(row["elapsed_seconds"] for row in records),
        "tool_execution_success_rate": sum(row["successful_tool_calls"] for row in records) /
                                        max(1, total_calls),
        "truncated_sample_rate": sum(row["truncated_samples"] for row in records) /
                                 max(1, len(records) * args.num_samples),
        "elapsed_seconds": elapsed,
        "elapsed_seconds_this_invocation": elapsed,
        "elapsed_seconds_scope": "current invocation only; prior interrupted attempts are excluded",
        "resumed_from_existing_rows": len(completed_ids),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "interpretation": (
            "pass_at_k is an oracle upper bound using gold after sampling; majority voting "
            "requires a strict majority among all samples, with unparseable outputs counted "
            "as non-votes that remain in the denominator."
        ),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
