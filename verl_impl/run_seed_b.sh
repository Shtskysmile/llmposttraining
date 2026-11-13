#!/usr/bin/env bash
# Second-seed replication of the key answer-vs-Dr.GRPO comparison.
# verl 0.5.0 does not expose a global rollout seed; a fresh run uses a new RNG
# stream, so this is an independent seed in practice.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p verl_impl/outputs/logs

train_one() {
  local name="$1" device="$2" config="$3"
  echo "[seed-b train] $name on logical $device"
  DEVICE="$device" CONFIG="$config" STEPS=6 SAVE_FREQ=2 ROLLOUT_MB=16 \
    bash verl_impl/run.sh \
      trainer.default_local_dir="verl_impl/outputs/${name}" \
      trainer.rollout_data_dir="verl_impl/outputs/${name}/rollout" \
      trainer.validation_data_dir="verl_impl/outputs/${name}/val" \
      demo.output_dir="verl_impl/outputs/${name}/hooks" \
      trainer.resume_mode=disable \
    > "verl_impl/outputs/logs/train_${name}.log" 2>&1 &
}

screen_one() {
  local name="$1" device="$2" config="$3"
  echo "[seed-b screen] $name on logical $device"
  DEVICE="$device" CONFIG="$config" MODE=eval \
    CKPT="$ROOT/verl_impl/outputs/${name}/global_step_6" \
    VAL_FILES="$ROOT/verl_impl/data/validation32.parquet" \
    VAL_N=1 VAL_DO_SAMPLE=false VAL_BATCH=32 ROLLOUT_MB=16 \
    EVAL_DIR="$ROOT/verl_impl/outputs/screen_${name}" \
    bash verl_impl/run.sh > "verl_impl/outputs/logs/screen_${name}.log" 2>&1 &
}

train_one answer6b 0 arithmetic_grpo_answer
train_one drgrpo6b 1 arithmetic_drgrpo_process
wait

screen_one answer6b 0 arithmetic_grpo_answer
screen_one drgrpo6b 1 arithmetic_drgrpo_process
wait
echo "[seed-b] all finished"
