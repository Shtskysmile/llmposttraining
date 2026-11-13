#!/usr/bin/env bash
# Batch launcher for the verl arithmetic-RLVR comparison.
#
#   bash verl_impl/run_experiments.sh base          # evaluate the base model
#   bash verl_impl/run_experiments.sh wave1         # 4 parallel trainings
#   bash verl_impl/run_experiments.sh wave2         # remaining trainings
#   bash verl_impl/run_experiments.sh eval_all      # evaluate every checkpoint
#   bash verl_impl/run_experiments.sh report        # Markdown summary
#
# Every run writes a log under verl_impl/outputs/logs/ and a checkpoint under
# verl_impl/outputs/<variant>/global_step_<steps>/actor.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="/root/work/tiankuan/shtskysmile/verllearning/.conda/envs/shtsky/bin/python"
LOG_DIR="$ROOT/verl_impl/outputs/logs"
mkdir -p "$LOG_DIR"

STEPS="${STEPS:-8}"
SAVE_FREQ="${SAVE_FREQ:-2}"
ROLLOUT_MB="${ROLLOUT_MB:-16}"
VAL_FILES32="$ROOT/verl_impl/data/validation32.parquet"

# name:config pairs
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

train_one() {
  local name="$1" device="$2"
  local config="${CONFIGS[$name]}"
  echo "[train] $name ($config) on logical NPU $device -> $LOG_DIR/train_$name.log"
  DEVICE="$device" CONFIG="$config" STEPS="$STEPS" SAVE_FREQ="$SAVE_FREQ" \
    ROLLOUT_MB="$ROLLOUT_MB" \
    nohup bash verl_impl/run.sh > "$LOG_DIR/train_$name.log" 2>&1 < /dev/null &
}

eval_one() {
  local name="$1" device="$2" config="$3" ckpt="${4:-}"
  local eval_dir="$ROOT/verl_impl/outputs/eval_$name"
  local args=(DEVICE="$device" CONFIG="$config" MODE=eval EVAL_DIR="$eval_dir"
              ROLLOUT_MB="$ROLLOUT_MB" VAL_BATCH=16)
  if [ -n "$ckpt" ]; then
    args+=("CKPT=$ckpt")
  fi
  echo "[eval] $name on logical NPU $device -> $eval_dir"
  env "${args[@]}" nohup bash verl_impl/run.sh > "$LOG_DIR/eval_$name.log" 2>&1 < /dev/null &
}

screen_one() {
  # Cheap greedy screen: 32 validation prompts, one sample each.
  local name="$1" device="$2" config="$3" ckpt="${4:-}"
  local eval_dir="$ROOT/verl_impl/outputs/screen_$name"
  local args=(DEVICE="$device" CONFIG="$config" MODE=eval EVAL_DIR="$eval_dir"
              ROLLOUT_MB="$ROLLOUT_MB" VAL_FILES="$VAL_FILES32"
              VAL_N=1 VAL_DO_SAMPLE=false VAL_BATCH=32)
  if [ -n "$ckpt" ]; then
    args+=("CKPT=$ckpt")
  fi
  echo "[screen] $name on logical NPU $device -> $eval_dir"
  env "${args[@]}" nohup bash verl_impl/run.sh > "$LOG_DIR/screen_$name.log" 2>&1 < /dev/null &
}

wait_all() {
  wait
  echo "[done] all background jobs finished"
}

cmd="${1:-}"
case "$cmd" in
  base)
    eval_one base 0 arithmetic_grpo_process ""
    wait_all
    ;;
  wave1)
    train_one answer 0
    train_one format 1
    train_one process 2
    train_one drgrpo 3
    wait_all
    ;;
  wave2)
    train_one rloo 3
    train_one dense 4
    train_one dapo 5
    train_one decomposed 0
    wait_all
    ;;
  wave3)
    train_one dense_sparse 1
    train_one difficulty 2
    train_one n8 3
    train_one loop 4
    wait_all
    ;;
  eval_all)
    running=0
    for name in answer format process drgrpo rloo dense dense_sparse n8 dapo decomposed difficulty loop; do
      ckpt="$ROOT/verl_impl/outputs/${CONFIGS[$name]}/global_step_${STEPS}"
      if [ ! -d "$ckpt" ]; then
        echo "[skip] $name: $ckpt not found"
        continue
      fi
      case "$name" in
        answer) device=0 ;;
        format) device=1 ;;
        process) device=2 ;;
        drgrpo) device=3 ;;
        rloo) device=4 ;;
        dense) device=5 ;;
        dense_sparse) device=0 ;;
        n8) device=1 ;;
        dapo) device=2 ;;
        decomposed) device=3 ;;
        difficulty) device=4 ;;
        loop) device=5 ;;
      esac
      eval_one "$name" "$device" "${CONFIGS[$name]}" "$ckpt"
      running=$((running + 1))
      if [ "$running" -ge 5 ]; then
        wait
        running=0
      fi
    done
    wait_all
    ;;
  analyze)
    for name in base answer format process drgrpo rloo dense dense_sparse n8 dapo decomposed difficulty loop; do
      dir="$ROOT/verl_impl/outputs/eval_$name"
      if [ -d "$dir" ]; then
        "$PY" verl_impl/analyze.py --dump-dir "$dir" --label "$name" \
          --json-out "$dir/summary.json" --save-groups "$dir/groups.json"
      fi
    done
    for name in answer format process drgrpo rloo dense dense_sparse n8 dapo decomposed difficulty loop; do
      dir="$ROOT/verl_impl/outputs/${CONFIGS[$name]}/rollout"
      if [ -d "$dir" ]; then
        "$PY" verl_impl/analyze.py --dump-dir "$dir" --label "$name" --per-step \
          --json-out "$dir/dynamics.json"
      fi
    done
    ;;
  report)
    inputs=()
    dynamics=()
    for name in base answer format process drgrpo rloo dense dense_sparse n8 dapo decomposed difficulty loop; do
      f="$ROOT/verl_impl/outputs/eval_$name/summary.json"
      [ -f "$f" ] && inputs+=(--input "$name=$f")
      if [ "$name" != "base" ]; then
        d="$ROOT/verl_impl/outputs/${CONFIGS[$name]}/rollout/dynamics.json"
        [ -f "$d" ] && dynamics+=(--dynamics "$name=$d")
      fi
    done
    "$PY" verl_impl/report.py "${inputs[@]}" "${dynamics[@]}"
    ;;
  screen)
    declare -A SCREEN_DEVICES=( [answer]=1 [format]=2 [process]=3 [drgrpo]=4
                                [rloo]=5 [dense]=0 [dapo]=1 [decomposed]=2
                                [dense_sparse]=3 [difficulty]=4 [n8]=5 [loop]=0 )
    screen_one base 0 arithmetic_grpo_process ""
    running=1
    for name in answer format process drgrpo rloo dense dapo decomposed dense_sparse difficulty n8 loop; do
      ckpt="$ROOT/verl_impl/outputs/${CONFIGS[$name]}/global_step_${STEPS}"
      [ -d "$ckpt" ] || continue
      screen_one "$name" "${SCREEN_DEVICES[$name]}" "${CONFIGS[$name]}" "$ckpt"
      running=$((running + 1))
      if [ "$running" -ge 5 ]; then
        wait
        running=0
      fi
    done
    wait_all
    ;;
  *)
    echo "usage: $0 {base|wave1|wave2|wave3|eval_all|screen|analyze|report}" >&2
    exit 2
    ;;
esac
