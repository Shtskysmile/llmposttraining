# Executive summary

## What was built

A verl 0.5.0 implementation of the arithmetic RLVR experiment, running on the
machine's Ascend 910B2 NPUs with `shtsky`'s Conda environment
(torch_npu 2.4, FSDP, HF rollout).  The TRL implementation under `src/` is kept
as the reference record.

## Original contributions

1. **Proof-carrying process reward (PCPR).**  A deterministic verifier parses
   the gold integer expression into an AST, accepts model-written arithmetic
   equality chains only when every part is exact and the leftmost operands are
   grounded, and rewards coverage of the gold intermediate values.  No learned
   PRM, no Monte-Carlo rollouts.
2. **Arithmetic chain parsing.**  Handles `a op b = c`, place-value chains
   (`552×866 = 552×(800+60+6) = … = 478032`), LaTeX/Unicode operators, step
   labels, thousands separators and negative intermediates while preserving
   character offsets for token-level rewards.
3. **Token-level reward + Process-GRPO**, **Decomposed-GRPO**,
   **Difficulty-GRPO**, and a **degeneration (digit-loop) penalty**.
4. **Framework fix.**  `HFRollout` passed `position_ids` to
   `transformers.generate`; on torch_npu 2.4 / transformers 4.53 that corrupts
   batched Qwen3-0.6B generation into digit loops.  Removing the argument fixed
   the rollouts; the patch is `patches/0001-hf-rollout-drop-position-ids.patch`.
5. **Research tooling.**  28 unit tests, a 300+ expression fuzz test, a
   controlled reward-hacking stress test, per-sample audit, paired bootstrap
   and exact McNemar, per-step dynamics/length-drift reports, and a
   coverage→correctness calibration (AUC) analysis (`process_auc.py`).

## Key results (6 steps, Qwen3-0.6B + LoRA)

### 64-prompt × 4-sample pre-registered evaluation

| variant | pass@1 | pass@4 | mean acc | coverage | loop |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.531 | 0.797 | 0.523 | 0.666 | 0.022 |
| answer6 | 0.531 | **0.875** | 0.578 | 0.717 | 0.017 |
| process6 | 0.547 | 0.797 | 0.547 | 0.689 | 0.022 |
| dense6 | 0.563 | 0.797 | 0.578 | 0.699 | 0.023 |
| **drgrpo6** | **0.672** | 0.859 | **0.660** | **0.775** | 0.012 |
| loop6 | 0.609 | 0.844 | 0.594 | 0.727 | **0.004** |

* Dr.GRPO+process beats answer-only on pass@1 (+0.141, CI [+0.031, +0.250],
  McNemar p=0.035) and mean accuracy (+0.082, CI [+0.020, +0.145]).
* It beats base on mean accuracy (+0.137, CI [+0.078, +0.195]); pass@1
  +0.141 (p=0.078).
* Answer-only is the best best-of-4 variant (pass@4 0.875, all-wrong 0.125).
* The loop-penalty variant has the lowest loop rate (0.4%) and a significant
  mean-accuracy gain over base (+0.070, CI [+0.004, +0.137]).
* Process reward with standard GRPO is not better than answer-only; the
  algorithm–reward interaction is the main story.

### 96-prompt held-out word-problem test (greedy)

| variant | accuracy | coverage | format | loop |
| --- | ---: | ---: | ---: | ---: |
| base | 0.594 | 0.609 | 0.104 | 0.042 |
| **answer6** | **0.635** | 0.667 | 0.031 | 0.042 |
| drgrpo6 | 0.615 | 0.656 | 0.052 | 0.042 |
| process6 | 0.542 | 0.589 | 0.094 | 0.010 |

Paired vs base: answer6 +0.042 [0.000, +0.094]; drgrpo6 +0.021 [−0.031, +0.073];
process6 **−0.052 [−0.104, −0.010]** (0 improvements / 5 regressions).
**The in-distribution winner is not the OOD winner.**  Answer-only transfers
best; Dr.GRPO+process is positive but smaller; standard GRPO+process hurts OOD.
The process reward's gains appear to be partly symbolic-template-specific.

### Extended algorithm wave (32-prompt greedy screen, 6 steps)

| variant | accuracy | coverage | loop |
| --- | ---: | ---: | ---: |
| RLOO | 0.656 | **0.750** | 0.028 |
| DAPO-lite | 0.656 | 0.734 | 0.063 |
| Decomposed-GRPO | 0.625 | 0.703 | 0.091 |
| Difficulty-GRPO | 0.656 | 0.734 | 0.060 |
| GRPO, n=8 | 0.656 | 0.719 | 0.091 |
| _ref_ GRPO+process (n=4) | 0.562 | 0.688 | 0.020 |
| _ref_ Dr.GRPO+process | **0.750** | **0.850** | 0.016 |

**Negative result:** no additional advantage estimator or group-size change
reproduces the Dr.GRPO gain; all five land at 0.625–0.656 (one prompt = 3.1
pp).  **Positive robustness:** in training rollouts all five hold the reward
zero-variance rate at 0.33–0.38 versus 0.46 for answer-only, so the dense-signal
mechanism is estimator-independent.  n=8 has the best training pass@k (0.875)
and lowest all-wrong rate (0.125) from more samples per prompt.

### Dense-signal calibration

`process_coverage` predicts final correctness with **AUC 0.93–0.96** on the 96
held-out word problems (95% CI lower bounds 0.88–0.91) and 0.91–0.96 across all
thirteen symbolic-screen checkpoints, and 0.91–0.94 on the 256-sample
stochastic 64×4 evaluation.  Correct answers carry ~0.95 coverage, wrong ones
~0.13 under greedy decoding (~0.35 under sampling).  The dense signal is
therefore aligned with the outcome it
is meant to densify — even in the base model that never saw the dense reward.
The flip side: PCPR is well-calibrated OOD while standard GRPO training on it
*hurts* OOD accuracy.  A dense reward can be a good predictor and a bad
learning signal; the advantage estimator decides which.

### Silent-group rescue

On the untrained base model's 64×4 evaluation, 13 of 64 groups are all-wrong
(outcome-silent).  **11/13 get a nonzero process score and 10/13 have
within-group variance**; 37/52 samples (71%) get a nonzero score, with mean
coverage 0.356, even though every final answer is wrong.  The process reward's
dense property is present without any training.

### Reward-hacking stress test

PCPR pays 0.583 for a correct derivation with a wrong final answer, but only
1.100 for trivial spam, ungrounded gold values, and grounded-but-gold-
irrelevant steps (a naive step-count reward pays 1.333–1.500).

## Reproduce

```bash
PY=/root/work/tiankuan/shtskysmile/verllearning/.conda/envs/shtsky/bin/python
$PY verl_impl/prepare_data.py --input-dir data/expanded_seed2028 --output-dir verl_impl/data
$PY -m pytest -q verl_impl/tests
bash verl_impl/run_core_wave.sh      # 6-step comparison
bash verl_impl/screen_core.sh        # 32-prompt greedy screens
bash verl_impl/analyze_core.sh       # summaries + dynamics
bash verl_impl/full_eval_core.sh     # 64-prompt × 4-sample full evaluation
bash verl_impl/analyze_full.sh       # full-eval summaries + paired tests
```

Detailed results: `RESULTS.md` (screen + dynamics), `RESULTS_FULL.md`
(64×4), `RESULTS_EXTENDED.md` (held-out test + extended algorithms),
`PAPER_DRAFT.md` (paper-style write-up), `RELATED_WORK.md` (literature and
pre-registered protocol).

## Limitations

* Six optimizer steps, one seed, one 0.6B model.  The 64×4 evaluation is
  statistically informative but the training budget is short.
* PCPR checks arithmetic equalities and gold intermediate values, not the
  semantic decomposition of a word problem.
* The extended wave (RLOO, DAPO-lite, Decomposed-GRPO, Difficulty-GRPO, n=8)
  is trained and screened on the 32-prompt greedy set; none beats Dr.GRPO at
  six steps, but the budget is short for these estimators.
* The shared machine required explicit NPU coordination; one collision killed
  the first wave at step 2.
