#!/usr/bin/env bash
# Full pre-registered evaluation: 64 validation prompts × 4 stochastic samples.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p verl_impl/outputs/logs

eval_one() {
  local name="$1" config="$2" device="$3" ckpt="${4:-}"
  local args=(DEVICE="$device" CONFIG="$config" MODE=eval
              VAL_N=4 VAL_DO_SAMPLE=true VAL_BATCH=16 ROLLOUT_MB=16
              EVAL_DIR="$ROOT/verl_impl/outputs/full_$name")
  if [ -n "$ckpt" ]; then
    args+=("CKPT=$ckpt")
  fi
  echo "[full] $name ($config) on logical $device -> full_$name"
  env "${args[@]}" bash verl_impl/run.sh > "verl_impl/outputs/logs/full_$name.log" 2>&1 &
}

eval_one base arithmetic_grpo_process 0 ""
eval_one drgrpo6 arithmetic_drgrpo_process 1 "$ROOT/verl_impl/outputs/drgrpo6/global_step_6"
eval_one answer6 arithmetic_grpo_answer 2 "$ROOT/verl_impl/outputs/answer6/global_step_6"
eval_one process6 arithmetic_grpo_process 3 "$ROOT/verl_impl/outputs/process6/global_step_6"
eval_one dense6 arithmetic_process_dense 4 "$ROOT/verl_impl/outputs/dense6/global_step_6"
wait
echo "[full] all finished"
