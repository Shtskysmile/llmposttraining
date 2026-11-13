#!/usr/bin/env bash
# Greedy 32-prompt screening for a list of trained checkpoints.
#
#   bash verl_impl/screen_wave.sh "answer format process drgrpo" 8
#
# One process per logical NPU (0..5), one checkpoint per process.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="/root/work/tiankuan/shtskysmile/verllearning/.conda/envs/shtsky/bin/python"
NAMES="${1:-answer format process drgrpo}"
STEPS="${2:-8}"
ROLLOUT_MB="${ROLLOUT_MB:-16}"
mkdir -p verl_impl/outputs/logs

declare -A CONFIGS=(
  [answer]=arithmetic_grpo_answer
  [format]=arithmetic_grpo_format
  [process]=arithmetic_grpo_process
  [drgrpo]=arithmetic_drgrpo_process
  [rloo]=arithmetic_rloo_process
  [dense]=arithmetic_process_dense
  [dense_sparse]=arithmetic_process_dense_sparse
  [n8]=arithmetic_grpo_process_n8
  [dapo]=arithmetic_dapo_process
  [decomposed]=arithmetic_process_decomposed
  [difficulty]=arithmetic_difficulty_process
  [loop]=arithmetic_process_loop
)
DEVICES=(0 1 2 3 4 5)
index=0
for name in $NAMES; do
  config="${CONFIGS[$name]}"
  ckpt="$ROOT/verl_impl/outputs/${config}/global_step_${STEPS}"
  if [ ! -d "$ckpt" ]; then
    echo "[skip] $name: $ckpt not found"
    continue
  fi
  device="${DEVICES[$index]}"
  index=$((index + 1))
  echo "[screen] $name on logical NPU $device"
  DEVICE="$device" CONFIG="$config" MODE=eval \
    VAL_FILES="$ROOT/verl_impl/data/validation32.parquet" \
    VAL_N=1 VAL_DO_SAMPLE=false VAL_BATCH=32 ROLLOUT_MB="$ROLLOUT_MB" \
    CKPT="$ckpt" EVAL_DIR="$ROOT/verl_impl/outputs/screen_$name" \
    nohup bash verl_impl/run.sh > "verl_impl/outputs/logs/screen_$name.log" 2>&1 < /dev/null &
done
wait
echo "[screen] all finished"
