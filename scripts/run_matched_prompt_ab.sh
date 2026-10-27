#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

data=runs/tool_math_scale_seed5090_test_direct_fewshot.jsonl
output=runs/tool_math_scale_seed5090_base_direct_fewshot_test.jsonl

env-grpo/bin/python scripts/build_direct_fewshot_data.py \
  --input data/tool_math_scale_seed5090/test.jsonl --output "$data"

env-grpo/bin/python scripts/summarize_matched_prompt_context.py \
  --model /root/.cache/modelscope/Qwen2.5-Math-7B-Instruct \
  --direct-data "$data" --tool-data data/tool_math_scale_seed5090/test.jsonl \
  --output runs/tool_math_scale_seed5090_base_tool_vs_direct_fewshot_prompt_tokens.json

env-grpo/bin/python src/rollout.py \
  --model /root/.cache/modelscope/Qwen2.5-Math-7B-Instruct \
  --data "$data" --mode direct --max-new-tokens 1024 \
  --max-total-tokens 1024 --max-tool-calls 0 --seed 5090 --output "$output"

env-grpo/bin/python scripts/compare_eval.py \
  --baseline runs/tool_math_scale_seed5090_base_tool_test_repro.jsonl \
  --candidate "$output" \
  --output runs/tool_math_scale_seed5090_base_tool_repro_vs_direct_fewshot_paired.json

env-grpo/bin/python scripts/compare_rollouts_by_group.py \
  --baseline runs/tool_math_scale_seed5090_base_tool_test_repro.jsonl \
  --candidate "$output" --group-field kind --bootstrap-seed 5090 \
  --bootstrap-samples 10000 \
  --output runs/tool_math_scale_seed5090_base_tool_repro_vs_direct_fewshot_by_kind.json
