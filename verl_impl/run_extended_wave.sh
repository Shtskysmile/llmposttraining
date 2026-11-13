#!/usr/bin/env bash
# Extended 6-step comparison: RLOO, DAPO-lite, decomposed-GRPO,
# difficulty-GRPO, and n=8 group size.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p verl_impl/outputs/logs

train_one() {
  local name="$1" config="$2" device="$3"
  echo "[ext train] $name ($config) on logical $device"
  DEVICE="$device" CONFIG="$config" STEPS=6 SAVE_FREQ=2 ROLLOUT_MB=16 \
    bash verl_impl/run.sh > "verl_impl/outputs/logs/train_${name}.log" 2>&1 &
}

screen_one() {
  local name="$1" config="$2" device="$3"
  echo "[ext screen] $name on logical $device"
  DEVICE="$device" CONFIG="$config" MODE=eval \
    CKPT="$ROOT/verl_impl/outputs/${config}/global_step_6" \
    VAL_FILES="$ROOT/verl_impl/data/validation32.parquet" \
    VAL_N=1 VAL_DO_SAMPLE=false VAL_BATCH=32 ROLLOUT_MB=16 \
    EVAL_DIR="$ROOT/verl_impl/outputs/screen_${name}" \
    bash verl_impl/run.sh > "verl_impl/outputs/logs/screen_${name}.log" 2>&1 &
}

# Training (5 devices).
train_one rloo arithmetic_rloo_process 0
train_one dapo arithmetic_dapo_process 1
train_one decomposed arithmetic_process_decomposed 2
train_one difficulty arithmetic_difficulty_process 3
train_one n8 arithmetic_grpo_process_n8 4
wait

# Greedy screens (5 devices).
screen_one rloo arithmetic_rloo_process 0
screen_one dapo arithmetic_dapo_process 1
screen_one decomposed arithmetic_process_decomposed 2
screen_one difficulty arithmetic_difficulty_process 3
screen_one n8 arithmetic_grpo_process_n8 4
wait
echo "[ext] all finished"
