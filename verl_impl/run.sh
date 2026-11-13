#!/usr/bin/env bash
# Entry point for the verl arithmetic-RLVR experiments.
#
#   DEVICE=0 CONFIG=arithmetic_grpo_process STEPS=10 bash verl_impl/run.sh
#   DEVICE=0 CONFIG=arithmetic_grpo_process MODE=eval \
#       CKPT=verl_impl/outputs/arithmetic_grpo_process/global_step_10 \
#       bash verl_impl/run.sh
#
#   MODE=eval EVAL_DIR=verl_impl/outputs/base_val bash verl_impl/run.sh
#
# Environment:
#   DEVICE      logical Ascend device id (0-5)
#   CONFIG      Hydra config name (default arithmetic_grpo_process)
#   MODE        train | eval
#   STEPS       optimizer steps (default 10)
#   SAVE_FREQ   checkpoint frequency (default = STEPS)
#   CKPT        checkpoint directory for MODE=eval (omit for base model)
#   EVAL_DIR    where validation generations are dumped
#   ROLLOUT_MB  HF rollout micro batch size (default 8)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# shtsky's Ascend environment: torch 2.4 + torch_npu + verl 0.5.0 source.
VERL_LEARNING="${VERL_LEARNING:-/root/work/tiankuan/shtskysmile/verllearning}"
export PYTHONNOUSERSITE=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export PYTHONPATH="$ROOT:$VERL_LEARNING:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="${DEVICE:-0}"

PY="$VERL_LEARNING/.conda/envs/shtsky/bin/python"
CONFIG="${CONFIG:-arithmetic_grpo_process}"
MODE="${MODE:-train}"
STEPS="${STEPS:-10}"
SAVE_FREQ="${SAVE_FREQ:-$STEPS}"
ROLLOUT_MB="${ROLLOUT_MB:-8}"
VAL_N="${VAL_N:-4}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-true}"
VAL_BATCH="${VAL_BATCH:-16}"
VAL_FILES="${VAL_FILES:-}"

args=(
  "+actor_rollout_ref.rollout.micro_batch_size=${ROLLOUT_MB}"
  "custom_reward_function.path=${ROOT}/verl_impl/rewards.py"
  "data.custom_cls.path=${ROOT}/verl_impl/dataset.py"
)

if [ "${MODE}" = "eval" ]; then
  EVAL_DIR="${EVAL_DIR:-verl_impl/outputs/eval_${CONFIG}}"
  args+=(
    "trainer.val_only=true"
    "trainer.val_before_train=true"
    "trainer.save_freq=-1"
    "trainer.validation_data_dir=${EVAL_DIR}"
    "trainer.rollout_data_dir=null"
    # Evaluation only needs the actor; skip the reference worker.
    "actor_rollout_ref.actor.use_kl_loss=false"
    "actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}"
    "actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}"
    "data.val_batch_size=${VAL_BATCH}"
  )
  if [ -n "${VAL_FILES}" ]; then
    args+=("data.val_files=${VAL_FILES}")
  fi
  if [ -n "${CKPT:-}" ]; then
    args+=(
      "trainer.resume_mode=resume_path"
      "trainer.resume_from_path=${CKPT}"
    )
  else
    # A true base-model evaluation: no randomly initialised LoRA adapter.
    args+=(
      "trainer.resume_mode=disable"
      "actor_rollout_ref.model.lora_rank=0"
    )
  fi
else
  args+=(
    "trainer.total_training_steps=${STEPS}"
    "trainer.save_freq=${SAVE_FREQ}"
    "trainer.resume_mode=disable"
  )
fi

echo "[verl_impl] mode=${MODE} config=${CONFIG} device=${ASCEND_RT_VISIBLE_DEVICES} steps=${STEPS}"
exec "$PY" "${ROOT}/verl_impl/main.py" --config-name "${CONFIG}" "${args[@]}" "$@"
