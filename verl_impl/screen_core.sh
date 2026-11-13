#!/usr/bin/env bash
# Greedy 32-prompt screen of the six 6-step checkpoints.
set -euo pipefail
cd /root/work/tiankuan/shtskysmile/llmposttraining
mkdir -p verl_impl/outputs/logs

screen_one() {
  local name="$1" config="$2" device="$3"
  echo "[screen6] $name on logical $device"
  DEVICE="$device" CONFIG="$config" MODE=eval \
    CKPT="$PWD/verl_impl/outputs/${name}/global_step_6" \
    VAL_FILES="$PWD/verl_impl/data/validation32.parquet" \
    VAL_N=1 VAL_DO_SAMPLE=false VAL_BATCH=32 ROLLOUT_MB=16 \
    EVAL_DIR="$PWD/verl_impl/outputs/screen_${name}" \
    bash verl_impl/run.sh > "verl_impl/outputs/logs/screen_${name}.log" 2>&1 &
}

screen_one process6 arithmetic_grpo_process 0
screen_one answer6 arithmetic_grpo_answer 1
screen_one dense6 arithmetic_process_dense 2
screen_one drgrpo6 arithmetic_drgrpo_process 3
screen_one format6 arithmetic_grpo_format 4
screen_one loop6 arithmetic_process_loop 5
wait
echo "[screen6] all finished"
