# verl implementation: proof-carrying process rewards for arithmetic RLVR

This directory ports the project's TRL/GRPO arithmetic experiment to **verl
0.5.0** and extends it with an original dense-reward method.  It runs on the
machine's **Ascend 910B2 NPUs** using `shtsky`'s Conda environment
(`.conda/envs/shtsky`, torch 2.4 + torch_npu 2.4 + the locally patched verl
0.5.0 source).

The original TRL implementation under `src/` and the results in `RESULTS.md`
are kept as the reference record.  This directory is the new primary
implementation.

## 1. What is new

The TRL experiments found three problems with answer-only RLVR on this task:

1. **No signal in all-wrong groups.** Validation groups were zero-variance
   about 75% of the time, so GRPO's group-relative advantage was often 0.
2. **Format is ignored.** The model almost never emitted `<answer>` tags
   (format-error rate ~100%) because the reward only looked at the number.
3. **A correct final number says nothing about the reasoning.** The verifier
   cannot distinguish a lucky answer from a correct derivation.

This implementation addresses all three with **proof-carrying process rewards
(PCPR)**:

* **Deterministic process verifier.**  The gold integer expression is parsed
  into an AST, and its canonical intermediate values are enumerated.  The model
  may write a simple `a op b = c` step or a multi-equality chain such as
  `552 × 866 = 552 × (800+60+6) = … = 478032`; every part is evaluated exactly,
  and the chain is accepted only if (a) all parts are equal and (b) the leaves
  of the leftmost part are grounded in prompt literals or the result of an
  earlier accepted step.  Coverage is the fraction of *gold* intermediate
  values actually proved.  LaTeX (`\times`, `\cdot`, `\div`), unicode
  operators, thousands separators, step labels and negative intermediates are
  normalised while preserving character offsets for token-level rewards.
* **Hack resistance.**  Trivial true equations (`1 + 1 = 2`) are ungrounded and
  contribute no coverage; repeated equations are de-duplicated; wrong
  arithmetic is penalised.  Rewarding a correct intermediate while missing the
  final answer still gives a positive dense signal, which is exactly what
  all-wrong groups need.
* **Token-level reward placement.**  `verl_impl/reward_manager.py` places each
  verified step's reward on the token where the step completes, and the
  final-answer/format reward on the last token.  The per-sample total is
  unchanged, but credit is assigned to the tokens that produced the step.
* **Dense Process-GRPO.**  `verl_impl/algorithms.py::process_grpo` converts
  sparse step rewards into reverse-cumulative returns and normalises them per
  group *and per token position*.  With answer-only rewards and
  `process_return_mode=outcome` it reduces to standard GRPO.
* **Degeneration penalty.**  The 0.6B model sometimes falls into digit loops
  (`982 × 451 = 441, 982 × 451 = 441, ...`).  An optional reward term detects a
  40-character chunk repeated three or more times and subtracts a fraction of
  its coverage.  This gives even all-wrong, zero-coverage groups a within-group
  signal (loop vs. non-loop) without penalising legitimate arithmetic.
* **Diagnostics used by the analysis.**  Every reward returns `acc`,
  `format_ok`, `process_coverage`, valid/invalid step counts, `loop_ratio`, and
  the all-wrong / all-correct / zero-variance group rates are recomputed
  offline from dumped rollouts.

## 2. Environment

```bash
export PYTHONNOUSERSITE=1
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1   # Ray + torch_npu 2.4
export PYTHONPATH="$PWD:/root/work/tiankuan/shtskysmile/verllearning:$PYTHONPATH"
PY=/root/work/tiankuan/shtskysmile/verllearning/.conda/envs/shtsky/bin/python
```

The harness (`demo/task_runner.py`, `demo/trainer.py`, `demo/hooks.py`) is
reused from `verllearning`; it wraps verl 0.5.0's `RayPPOTrainer` with hooks and
handles the custom-advantage whitelist.  `verl_impl/main.py` adds both project
roots to `sys.path`, so no manual `PYTHONPATH` is required when using
`run.sh`.

Hardware: 6× Ascend 910B2 (64 GiB HBM each), logical device ids 0–5.
Model: `/root/work/models/Qwen3-0.6B` (the machine has no 1.7B checkpoint).

### Framework fix: batched generation on torch_npu

The local verl 0.5.0 rollout passed explicit `position_ids` to
`transformers.generate`.  On this torch_npu 2.4 / transformers 4.53 stack that
makes batched generation produce degenerate digit loops for Qwen3-0.6B (the
single-prompt path is fine).  The first smoke run showed
`response_length/clip_ratio=1.0` and `actor/entropy≈0` because every sample was
garbage.  `verl_impl/patches/0001-hf-rollout-drop-position-ids.patch` removes
the argument; the model derives positions from `attention_mask`.  After the
fix, rollouts contain the clean `a op b = c` traces used by the verifier.

This bug is not specific to this project: `verllearning`'s MedQA demo uses the
same rollout path, and its base accuracy of 2.35% (below the 25% random
baseline) is consistent with the same corruption.  Re-running that demo after
the patch would be worthwhile.

## 3. Data

```bash
$PY verl_impl/prepare_data.py \
  --input-dir data/expanded_seed2028 --output-dir verl_impl/data
```

This converts the 256/64/96 expanded split to verl Parquet.  Each record's
`extra_info` contains the canonical `expression` and its `intermediate_values`,
so the reward function does not re-parse the prompt.  The script fails if a
gold expression does not evaluate to the recorded answer.

The held-out test prompts are word problems; `expression_from_prompt` maps the
factory template to `start + per_batch * batches - discarded`, so the same
process verifier applies to both symbolic and word-problem splits.

## 4. Reward modes

`custom_reward_function.reward_kwargs.mode` selects the training objective:

| mode | score |
| --- | --- |
| `answer` | `1.0 * final_correct` (answer-only baseline) |
| `format` | `1.0 * final_correct + 0.1 * answer_tag_ok` |
| `process` | `1.0 * final_correct + 0.5 * coverage + 0.1 * answer_tag_ok - 0.05 * invalid_rate` |

All modes return the same diagnostic fields, so every run is evaluated with
the same accuracy/format/process metrics.

## 5. Configurations

| config | reward | advantage | purpose |
| --- | --- | --- | --- |
| `arithmetic_grpo_answer` | answer | GRPO | ported baseline |
| `arithmetic_grpo_format` | format | GRPO | format-shaping ablation |
| `arithmetic_grpo_process` | process | GRPO | dense scalar process reward |
| `arithmetic_drgrpo_process` | process | Dr.GRPO | no group-std normalisation |
| `arithmetic_rloo_process` | process | RLOO | leave-one-out baseline |
| `arithmetic_process_dense` | process + token-level | Process-GRPO (cumsum) | proposed method |
| `arithmetic_process_dense_sparse` | process + token-level | Process-GRPO (no cumsum) | credit-propagation ablation |
| `arithmetic_grpo_process_n8` | process | GRPO, n=8 | group-size ablation |
| `arithmetic_dapo_process` | process + overlong | GRPO | DAPO-lite: clip-higher, token-mean, no KL |
| `arithmetic_process_decomposed` | process + token-level | Decomposed-GRPO | separate final/process advantage channels |
| `arithmetic_difficulty_process` | process | Difficulty-GRPO | variance-weighted Dr.GRPO (soft dynamic sampling) |
| `arithmetic_process_loop` | process + loop penalty | GRPO | penalise digit-repetition degeneration |

The literature mapping, hypotheses and experimental protocol are in
[`RELATED_WORK.md`](RELATED_WORK.md).  A controlled verifier stress test
(`stress_test.py`), paired significance tests (`stats.py`), and a
coverage→correctness calibration analysis (`process_auc.py`, AUC of the dense
signal against final correctness) accompany the training comparison.

Common settings: Qwen3-0.6B + LoRA r=16/α=32, batch 8 prompts × 4 samples
(32 completions/step), 6–8 steps (the shared-NPU window used 6), lr 1e-5,
KL loss 0.001, prompt ≤256, response ≤320, rollout temperature 0.7 /
top_p 0.95 / top_k 20, FSDP `use_orig_params=true`.

## 6. Running

Smoke (1 step, no checkpoint):

```bash
DEVICE=0 CONFIG=arithmetic_grpo_process STEPS=1 SAVE_FREQ=-1 bash verl_impl/run.sh
```

Train one variant:

```bash
DEVICE=0 CONFIG=arithmetic_grpo_process STEPS=10 bash verl_impl/run.sh
```

Evaluate the base model or a checkpoint (4 stochastic samples per validation
prompt, dumped to JSONL):

```bash
# base model
DEVICE=0 CONFIG=arithmetic_grpo_process MODE=eval \
  EVAL_DIR=verl_impl/outputs/base_val bash verl_impl/run.sh

# checkpoint
DEVICE=0 CONFIG=arithmetic_grpo_process MODE=eval \
  CKPT=verl_impl/outputs/arithmetic_grpo_process/global_step_10 \
  EVAL_DIR=verl_impl/outputs/arithmetic_grpo_process/val_eval \
  bash verl_impl/run.sh
```

Batch scripts (each logs under `verl_impl/outputs/logs/`):

```bash
bash verl_impl/run_core_wave.sh      # 6-step: answer, dense, Dr.GRPO, format, loop
bash verl_impl/screen_core.sh        # 32-prompt greedy screens
bash verl_impl/analyze_core.sh       # summaries, groups, dynamics
bash verl_impl/full_eval_core.sh     # 64-prompt × 4-sample full evaluation
bash verl_impl/eval_test.sh          # 96-prompt held-out word-problem test
bash verl_impl/run_extended_wave.sh  # RLOO, DAPO-lite, decomposed, difficulty, n=8
bash verl_impl/run_seed_b.sh         # second-seed answer vs Dr.GRPO
```

Analyse a dump:

```bash
$PY verl_impl/analyze.py --dump-dir verl_impl/outputs/base_val --label base
$PY verl_impl/analyze.py --dump-dir verl_impl/outputs/arithmetic_grpo_process/rollout --per-step
```

The analysis reports pass@1, pass@k, mean accuracy, format rate, process
coverage, and the zero-variance / all-wrong / all-correct group rates.

## 7. Tests

```bash
$PY -m pytest -q verl_impl/tests/test_process_verifier.py
```

The tests cover strict answer extraction, symbolic and word-problem expression
parsing, full traces, correct-intermediate/wrong-final traces, ungrounded
equation rejection, arithmetic-error penalties, unicode operators, duplicate
steps, and the claim that trivial-equation spam earns no process reward.

## 8. Relationship to the TRL implementation

| | TRL (`src/`) | verl (`verl_impl/`) |
| --- | --- | --- |
| trainer | HF `GRPOTrainer` | verl 0.5.0 `RayPPOTrainer` + FSDP |
| rollout | HF generate on CUDA | HF rollout on Ascend NPU |
| reward | answer-only / trace SFT | answer / format / proof-carrying process |
| advantage | GRPO | GRPO / Dr.GRPO / RLOO / Process-GRPO |
| checkpoint | PEFT adapter | FSDP LoRA checkpoint |
| diagnostics | in-process metrics | dumped rollouts + offline analysis |

The final-answer extractor is behaviour-compatible with `src/verifier.py`, so
`acc` and `pass@k` are directly comparable across implementations.

## 8. Results

* [`RESULTS.md`](RESULTS.md): 32-prompt greedy comparison, training dynamics
  (zero-variance 34–50% reduction), paired tests, and the per-prompt audit.
* [`RESULTS_FULL.md`](RESULTS_FULL.md): 64-prompt × 4-sample full evaluation.
* [`RESULTS_EXTENDED.md`](RESULTS_EXTENDED.md): 96-prompt held-out test set,
  RLOO/DAPO-lite/decomposed/difficulty/n=8, and the second seed.
* [`PAPER_DRAFT.md`](PAPER_DRAFT.md): paper-style abstract, method, results and
  discussion.
* [`RELATED_WORK.md`](RELATED_WORK.md): literature map, hypotheses, theory and
  pre-registered protocol.

## 9. Limitations

* Qwen3-0.6B is much weaker than the original 1.7B model; absolute accuracy is
  not comparable to `RESULTS.md`.
* The executed comparison uses 6 optimizer steps, 32 greedy validation prompts
  and one seed.  Results are descriptive; multiple seeds and the prepared
  64-prompt × 4-sample evaluation are needed for a statistical claim.
* The process verifier only covers `+`, `-`, `*` and exact `/` on integers.  It
  targets this synthetic arithmetic domain, not general mathematical reasoning.
* The dense return is a heuristic.  `process_return_mode=cumsum` can credit
  tokens that precede a verified step; the sparse ablation isolates this.
