#!/usr/bin/env bash
# Second-seed replication of the pre-registered 64-prompt x 4-sample evaluation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p verl_impl/outputs/logs

eval_one() {
  local name="$1" config="$2" device="$3" ckpt="$4"
  echo "[seed-b full] $name on logical $device"
  DEVICE="$device" CONFIG="$config" MODE=eval CKPT="$ckpt" \
    VAL_N=4 VAL_DO_SAMPLE=true VAL_BATCH=16 ROLLOUT_MB=16 \
    EVAL_DIR="$ROOT/verl_impl/outputs/full_${name}" \
    bash verl_impl/run.sh > "verl_impl/outputs/logs/full_${name}.log" 2>&1 &
}

eval_one answer6b arithmetic_grpo_answer 0 "$ROOT/verl_impl/outputs/answer6b/global_step_6"
eval_one drgrpo6b arithmetic_drgrpo_process 1 "$ROOT/verl_impl/outputs/drgrpo6b/global_step_6"
wait
echo "[seed-b full] all finished"
