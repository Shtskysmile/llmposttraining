# Runbook

All commands run from the project root with `shtsky`'s environment.  Logical
NPU ids 0–5 map to the six available physical devices (0,2,3,4,6,7); use one
id per process.

**The machine is shared.**  Another session may run six jobs covering all
logical ids.  Before launching, check `npu-smi info` and the env of other
`main.py` processes (`tr '\0' '\n' < /proc/PID/environ | grep ASCEND`).  On
2026-11-06 four parallel jobs plus a base eval collided with another session's
six MedQA jobs and the Ray workers died at step 2 (`ActorDiedError`, native
worker death).  Running at most one or two jobs at a time, on devices that are
verifiably free, is much safer.  A crashed run does not save a checkpoint, so
it must be restarted from scratch.

## 0. One-time setup

```bash
# data: 256/64/96 parquet
PY=/root/work/tiankuan/shtskysmile/verllearning/.conda/envs/shtsky/bin/python
$PY verl_impl/prepare_data.py --input-dir data/expanded_seed2028 --output-dir verl_impl/data

# unit tests (verifier, rewards, advantage invariants)
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD:/root/work/tiankuan/shtskysmile/verllearning:$PYTHONPATH" \
  $PY -m pytest -q verl_impl/tests

# controlled reward-hacking stress test
$PY verl_impl/stress_test.py --rows 16 --json-out verl_impl/outputs/stress_test.json
```

## 1. Smoke

> **Launcher gotcha.** `env VAR=value ... command` treats `trainer.*=...` as
> environment variables, not Hydra overrides.  Put the script name before the
> overrides: `DEVICE=0 CONFIG=... bash verl_impl/run.sh trainer.default_local_dir=...`.
> `run.sh` itself handles this correctly; only hand-written `env` wrappers can
> get it wrong.

```bash
DEVICE=0 CONFIG=arithmetic_grpo_process STEPS=1 SAVE_FREQ=-1 \
  trainer.default_local_dir=verl_impl/outputs/smoke3 \
  trainer.rollout_data_dir=verl_impl/outputs/smoke3/rollout \
  demo.output_dir=verl_impl/outputs/smoke3/hooks \
  bash verl_impl/run.sh
```

Check the log for `timing_s/gen`, `response_length/clip_ratio` (should be well
below 1.0), and inspect `verl_impl/outputs/smoke3/rollout/1.jsonl`: outputs
must contain `a op b = c` lines, not digit loops.

## 2. Base-model evaluation

Cheap greedy screen (32 prompts, one sample):

```bash
DEVICE=0 CONFIG=arithmetic_grpo_process MODE=eval \
  VAL_FILES="$PWD/verl_impl/data/validation32.parquet" \
  VAL_N=1 VAL_DO_SAMPLE=false VAL_BATCH=32 ROLLOUT_MB=16 \
  EVAL_DIR="$PWD/verl_impl/outputs/screen_base" \
  bash verl_impl/run.sh
```

Full base evaluation (64 prompts, 4 samples):

```bash
DEVICE=0 CONFIG=arithmetic_grpo_process MODE=eval ROLLOUT_MB=16 VAL_BATCH=16 \
  EVAL_DIR="$PWD/verl_impl/outputs/eval_base" bash verl_impl/run.sh
```

Analyze / audit:

```bash
$PY verl_impl/analyze.py --dump-dir verl_impl/outputs/screen_base --label base \
  --json-out verl_impl/outputs/screen_base/summary.json \
  --save-groups verl_impl/outputs/screen_base/groups.json
$PY verl_impl/audit.py --dump-dir verl_impl/outputs/screen_base --samples 3
```

## 3. Training waves (8 steps each)

```bash
# wave 1: reward ablation
DEVICE=0 CONFIG=arithmetic_grpo_answer       STEPS=8 ROLLOUT_MB=16 bash verl_impl/run.sh &
DEVICE=1 CONFIG=arithmetic_grpo_format       STEPS=8 ROLLOUT_MB=16 bash verl_impl/run.sh &
DEVICE=2 CONFIG=arithmetic_grpo_process      STEPS=8 ROLLOUT_MB=16 bash verl_impl/run.sh &
DEVICE=3 CONFIG=arithmetic_drgrpo_process    STEPS=8 ROLLOUT_MB=16 bash verl_impl/run.sh &
wait

# wave 2: advantage / token-level
DEVICE=0 CONFIG=arithmetic_rloo_process         STEPS=8 ROLLOUT_MB=16 bash verl_impl/run.sh &
DEVICE=1 CONFIG=arithmetic_process_dense        STEPS=8 ROLLOUT_MB=16 bash verl_impl/run.sh &
DEVICE=2 CONFIG=arithmetic_dapo_process         STEPS=8 ROLLOUT_MB=16 bash verl_impl/run.sh &
DEVICE=3 CONFIG=arithmetic_process_decomposed   STEPS=8 ROLLOUT_MB=16 bash verl_impl/run.sh &
wait

# wave 3: ablations
DEVICE=0 CONFIG=arithmetic_process_dense_sparse STEPS=8 ROLLOUT_MB=16 bash verl_impl/run.sh &
DEVICE=1 CONFIG=arithmetic_difficulty_process   STEPS=8 ROLLOUT_MB=16 bash verl_impl/run.sh &
DEVICE=2 CONFIG=arithmetic_grpo_process_n8      STEPS=8 ROLLOUT_MB=16 bash verl_impl/run.sh &
DEVICE=3 CONFIG=arithmetic_process_loop         STEPS=8 ROLLOUT_MB=16 bash verl_impl/run.sh &
wait
```

Or use `bash verl_impl/run_experiments.sh wave1` etc. (it logs under
`verl_impl/outputs/logs/`).

### 3.1 6-step core comparison (custom output directories)

The shared-NPU window used a compact six-variant comparison with explicit
output directories so runs could be isolated from the earlier wave:

```bash
# answer6, dense6, drgrpo6, format6, loop6 on logical 1–5 (process6 on 0)
bash verl_impl/run_core_wave.sh

# 32-prompt greedy screen of all six (including process6 on logical 0)
bash verl_impl/screen_core.sh

# summaries, per-prompt groups, and training dynamics
bash verl_impl/analyze_core.sh
```

Each training run writes `verl_impl/outputs/<name>/global_step_6/` and
`verl_impl/outputs/<name>/rollout/1..6.jsonl`.  The screen dumps are under
`verl_impl/outputs/screen_<name>/`.

## 4. Screening evaluation of every checkpoint

```bash
# one at a time to avoid two processes on one NPU
for spec in "answer 0" "format 1" "process 2" "drgrpo 3" "rloo 4" "dense 5"; do
  set -- $spec
  DEVICE=$2 CONFIG=$(case $1 in answer) echo arithmetic_grpo_answer;; format) echo arithmetic_grpo_format;; process) echo arithmetic_grpo_process;; drgrpo) echo arithmetic_drgrpo_process;; rloo) echo arithmetic_rloo_process;; dense) echo arithmetic_process_dense;; esac) \
  MODE=eval VAL_FILES="$PWD/verl_impl/data/validation32.parquet" VAL_N=1 VAL_DO_SAMPLE=false VAL_BATCH=32 ROLLOUT_MB=16 \
  CKPT="$PWD/verl_impl/outputs/<config>/global_step_8" \
  EVAL_DIR="$PWD/verl_impl/outputs/screen_$1" bash verl_impl/run.sh
done
```

(The shell snippet above is illustrative; `run_experiments.sh screen` automates
it but launches several jobs at once, so check device conflicts.)

## 5. Analysis

```bash
$PY verl_impl/analyze.py --dump-dir verl_impl/outputs/screen_process --label process \
  --json-out verl_impl/outputs/screen_process/summary.json \
  --save-groups verl_impl/outputs/screen_process/groups.json

# paired comparison, e.g. process vs base
$PY verl_impl/stats.py \
  --a base=verl_impl/outputs/screen_base/groups.json \
  --b process=verl_impl/outputs/screen_process/groups.json

# training dynamics from the free rollout dumps
$PY verl_impl/analyze.py --dump-dir verl_impl/outputs/arithmetic_grpo_process/rollout \
  --label process --per-step --json-out verl_impl/outputs/arithmetic_grpo_process/rollout/dynamics.json

# markdown table over all screening summaries
$PY verl_impl/report.py --input base=verl_impl/outputs/screen_base/summary.json \
  --input process=verl_impl/outputs/screen_process/summary.json ...
```

## 6. Expected durations (measured)

- One 32-completion training step: ~4.5 min generation + ~0.5 min update.
- 8-step run: ~40 min.
- 16-prompt greedy eval: ~3 min.
- 32-prompt greedy screen: ~6 min.
- 64-prompt × 4-sample full eval: ~35 min (run in parallel for key variants).

These are HF-rollout numbers on Ascend; the environment has no vLLM.
