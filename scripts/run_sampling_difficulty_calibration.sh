#!/usr/bin/env bash
set -euo pipefail

for seed in 5080 5081; do
  output="runs/tool_math_scale_seed5090_base_tool_pass4_calibration_seed${seed}.jsonl"
  args=(
    --model /root/.cache/modelscope/Qwen2.5-Math-7B-Instruct
    --data data/tool_math_scale_seed5090/test.jsonl
    --mode tool
    --num-samples 4
    --max-new-tokens 384
    --max-total-tokens 1024
    --max-tool-calls 2
    --temperature 0.7
    --top-k 50
    --top-p 0.95
    --seed "$seed"
  )
  if [ -f "$output" ]; then
    args+=(--resume)
  fi
  env-grpo/bin/python src/evaluate_sampling.py "${args[@]}" --output "$output"
done

env-grpo/bin/python scripts/compare_sampling_seeds.py \
  --baseline runs/tool_math_scale_seed5090_base_tool_pass4_seed5090.jsonl \
  --baseline runs/tool_math_scale_seed5090_base_tool_pass4_seed5091.jsonl \
  --baseline runs/tool_math_scale_seed5090_base_tool_pass4_seed5092.jsonl \
  --candidate runs/tool_math_scale_seed5090_sft_tool_pass4_seed5090.jsonl \
  --candidate runs/tool_math_scale_seed5090_sft_tool_pass4_seed5091.jsonl \
  --candidate runs/tool_math_scale_seed5090_sft_tool_pass4_seed5092.jsonl \
  --difficulty-calibration-baseline runs/tool_math_scale_seed5090_base_tool_pass4_calibration_seed5080.jsonl \
  --difficulty-calibration-baseline runs/tool_math_scale_seed5090_base_tool_pass4_calibration_seed5081.jsonl \
  --output runs/tool_math_scale_seed5090_base_vs_sft_pass4_multiseed_calibrated_difficulty.json
