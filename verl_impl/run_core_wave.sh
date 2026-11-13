#!/usr/bin/env bash
# 6-step comparison wave (after the NPU window frees).
set -euo pipefail
cd /root/work/tiankuan/shtskysmile/llmposttraining
mkdir -p verl_impl/outputs/logs

run_one() {
  local name="$1" config="$2" device="$3"
  echo "[train6] $name ($config) on logical $device"
  DEVICE="$device" CONFIG="$config" STEPS=6 SAVE_FREQ=2 ROLLOUT_MB=16 \
    bash verl_impl/run.sh \
      trainer.default_local_dir="verl_impl/outputs/${name}" \
      trainer.rollout_data_dir="verl_impl/outputs/${name}/rollout" \
      trainer.validation_data_dir="verl_impl/outputs/${name}/val" \
      demo.output_dir="verl_impl/outputs/${name}/hooks" \
      trainer.resume_mode=disable \
    > "verl_impl/outputs/logs/train_${name}.log" 2>&1 &
}

run_one answer6 arithmetic_grpo_answer 1
run_one dense6 arithmetic_process_dense 2
run_one drgrpo6 arithmetic_drgrpo_process 3
run_one format6 arithmetic_grpo_format 4
run_one loop6 arithmetic_process_loop 5
wait
echo "[train6] all finished"
