#!/usr/bin/env bash
# Held-out word-problem test set: 96 prompts, greedy, one sample per prompt.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p verl_impl/outputs/logs

eval_one() {
  local name="$1" config="$2" device="$3" ckpt="${4:-}"
  local args=(DEVICE="$device" CONFIG="$config" MODE=eval
              VAL_FILES="$ROOT/verl_impl/data/test.parquet"
              VAL_N=1 VAL_DO_SAMPLE=false VAL_BATCH=32 ROLLOUT_MB=16
              EVAL_DIR="$ROOT/verl_impl/outputs/test_$name")
  if [ -n "$ckpt" ]; then
    args+=("CKPT=$ckpt")
  fi
  echo "[test] $name ($config) on logical $device -> test_$name"
  env "${args[@]}" bash verl_impl/run.sh > "verl_impl/outputs/logs/test_$name.log" 2>&1 &
}

eval_one base arithmetic_grpo_process 0 ""
eval_one answer6 arithmetic_grpo_answer 1 "$ROOT/verl_impl/outputs/answer6/global_step_6"
eval_one drgrpo6 arithmetic_drgrpo_process 2 "$ROOT/verl_impl/outputs/drgrpo6/global_step_6"
eval_one process6 arithmetic_grpo_process 3 "$ROOT/verl_impl/outputs/process6/global_step_6"
wait
echo "[test] all finished"
