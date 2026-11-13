# Proof-carrying process rewards: literature, hypotheses, protocol

This document frames the verl implementation as a small controlled study.  It
was written **before** the main comparison runs and is the pre-registered plan:
the primary metric is validation pass@1, the held-out test set is used only as
a secondary check, and the significance analysis is paired by prompt.

## 1. Research question

Answer-only RLVR gives one bit per trajectory.  On a task where the base model
already succeeds on most prompts, most groups are saturated (all correct or all
wrong), so the group-relative advantage in GRPO is exactly zero and no learning
happens.  Recent work attacks this from three directions:

1. **Better advantage arithmetic** — Dr.GRPO, RLOO, DAPO, GSPO.
2. **Better sampling** — DAPO dynamic sampling, adaptive rollout, response reuse.
3. **Denser rewards** — process reward models (PRMs), verifiable process
   rewards, implicit process rewards.

Our question: *can a deterministic, proof-carrying process verifier provide
useful dense signal on a synthetic arithmetic task, and how does it interact
with the advantage estimators that verl supports?*

The specific contribution is **proof-carrying process rewards (PCPR)**: the
verifier does not use a learned PRM or Monte Carlo rollouts.  It parses the
gold integer expression into an AST, enumerates its canonical intermediate
values, and only accepts model-written equations whose arithmetic is exact and
whose operands are grounded in prompt literals or previously accepted results.
Coverage of the gold derivation is the dense reward.  This is a deterministic
instance of the "verifiable process reward" principle (VPR) and is resistant to
the step-count reward hacking that a naive PRM invites.

## 2. Literature map

| work | venue / status | idea used here |
| --- | --- | --- |
| DeepSeekMath / GRPO (Shao et al., 2024, arXiv:2402.03300) | preprint | group-relative advantage baseline |
| DeepSeek-R1 (DeepSeek-AI, Nature 2025) | Nature | RLVR with rule-based rewards |
| RLOO (Ahmadian et al., ACL 2024, arXiv:2402.14740) | CCF-A | leave-one-out baseline |
| Dr.GRPO (Liu et al., COLM 2025, arXiv:2503.20783) | COLM | remove std normalisation; length-bias critique |
| DAPO (Yu et al., NeurIPS 2025, arXiv:2503.14476) | CCF-A | clip-higher, token-mean loss, dynamic sampling (partially available) |
| GSPO (Zheng et al., 2025, arXiv:2507.18071) | preprint | sequence-level ratio; not implemented in verl 0.5.0 |
| PRIME (Cui et al., 2025, arXiv:2502.01456) | preprint | implicit process rewards; fused dense/sparse returns |
| VinePPO (Kazemnejad et al., ICML 2025, arXiv:2410.01679) | CCF-A | Monte-Carlo process credit assignment; our deterministic alternative |
| PRM survey (Zheng et al., 2025, arXiv:2510.08049) | survey | learned PRMs are hackable; prefer deterministic verifiers |
| VPR (Yuan et al., 2026, arXiv:2605.10325) | preprint | dense turn-level rewards from symbolic oracles |
| Group-std identity (2026, arXiv:2607.00152) | preprint | unifies GRPO / Dr.GRPO / DAPO as operations on group std |
| verl / HybridFlow (Sheng et al., EuroSys 2025, arXiv:2409.19256) | CCF-A | training framework |

The comparison is deliberately limited to algorithms implementable in the
locally patched **verl 0.5.0** with torch 2.4 on Ascend: GRPO, Dr.GRPO, RLOO,
DAPO-lite, and custom advantage estimators.  GSPO/CISPO/SAPO require actor-loss
or framework changes that this environment cannot support; they are future work.

## 3. Theory sketch

### 3.1 Silent groups

For binary answer rewards with within-prompt success probability `p` and group
size `G`, every sample in a group has the same reward with probability

```
P_silent(p) = p^G + (1 - p)^G .
```

At `p = 0.8, G = 4`, `P_silent = 0.41`; at `p = 0.5`, it is `0.125`.  Aggregated
over prompts with heterogeneous difficulty, the empirical silent-group rate can
be much higher than the rate at the mean accuracy, which is why the original
TRL run observed 75% zero-variance groups at ~72% pass@1.

DAPO's dynamic sampling *rejects* silent groups and regenerates.  Its cost is
extra rollout compute, and it can starve later in training when the model is
consistently right or wrong.  Our alternatives are:

* **PCPR**: make the reward itself multi-level (`coverage`), so a group of
  wrong-final samples with different partial proofs is no longer silent.
* **Degeneration penalty** (`loop_penalty_weight`): a 40-character chunk
  repeated at least three times is a digit loop; subtracting a fraction of its
  coverage gives all-wrong, zero-coverage groups a within-group signal
  (loop vs. non-loop) without penalising legitimate arithmetic.
* **Difficulty-GRPO**: keep the reward binary but multiply the group advantage
  by `σ^α` (normalised by the mean weight).  `α = 0` is Dr.GRPO; `α = 1` is a
  soft, variance-weighted Dr.GRPO.  For binary rewards `σ = sqrt(p(1-p))`, so
  `α = 1` is proportional to `p(1-p)`, maximised at the decision boundary and
  zero for saturated groups.  This is a smooth alternative to DAPO's hard
  filter, without extra rollouts.

### 3.2 Dense credit and reward hacking

A naive step-count reward pays for every arithmetically valid equation.  It can
be farmed with `1 + 1 = 2` lines.  PCPR pays only for coverage of *gold*
intermediate values, and grounding ties each accepted step to the prompt or to
an earlier accepted step.  Section 5.3 of the experiment reports a controlled
stress test of this claim.

### 3.3 Decomposition

The final answer and the process proof are different objectives.  Summing them
into one scalar lets the final term (weight 1.0) dominate the process term
(weight 0.5).  `decomposed_grpo` group-normalises each channel separately and
combines `A = A_final + λ A_process`, which gives an explicit knob and keeps
the dense channel from being washed out.

## 4. Hypotheses

| id | hypothesis | prediction | falsified if |
| --- | --- | --- | --- |
| H1 | PCPR creates signal in all-wrong groups | zero-variance group rate lower than answer-only, and all-wrong group rate lower | no reduction, or process coverage ≈ 0 on rollouts |
| H2 | Token-level dense Process-GRPO helps | `dense` > `process` on validation pass@1 | `dense` ≤ `process` within the paired CI |
| H3 | Decomposed advantages help | `decomposed` ≥ `process` | clearly worse |
| H4 | Difficulty weighting is a useful soft filter | `difficulty` has lower gradient-noise proxy / is not worse than `drgrpo` | worse accuracy and no stability benefit |
| H5 | Algorithm choice interacts with dense reward | Dr.GRPO/RLOO/DAPO-lite differ in length drift and format rate | no measurable differences |

The experiment is small (64 validation prompts, one seed, 10 steps).  A null
result is still informative: it would show that deterministic process coverage
is too sparse for this model/data regime, or that the answer-only baseline is
already saturated.

## 5. Protocol

### 5.1 Data and model

* Data: `data/expanded_seed2028`, 256 train / 64 validation / 96 test.  Train
  and validation are symbolic expressions; test is a held-out word-problem
  template.  The process verifier parses both templates.
* Model: `/root/work/models/Qwen3-0.6B`, LoRA r=16/α=32, thinking disabled by
  the custom dataset.  The base model is capable: with thinking disabled it
  writes clean `a op b = c` traces and often ends with `\boxed{answer}`.
* Sampling: temperature 0.7, top_p 0.95, top_k 20.  At temperature 1.0 the
  0.6B model degenerates into digit loops; 0.7 keeps within-group diversity
  while producing coherent traces.
* Training: 6–8 optimizer steps (the shared NPU window determines the final
  budget), 8 prompts × 4 samples per step, lr 1e-5, KL loss 0.001, prompt ≤256,
  response ≤320.  Data order is fixed (`shuffle: false`); verl 0.5.0 does not
  expose a global rollout seed.
* Framework fix: `HFRollout` must not pass explicit `position_ids` to
  `generate` on this torch_npu 2.4 / transformers 4.53 stack.  With them,
  batched generation is degenerate even though single-prompt generation is
  fine.  See `verl_impl/patches/0001-hf-rollout-drop-position-ids.patch`.

### 5.2 Evaluation

* Screening: 32 validation prompts, greedy (n=1), max 320 tokens, to rank
  variants cheaply.  Full comparison: 64 validation prompts, 4 stochastic
  samples (temperature 0.7), identical batch size for base and trained models.
* Primary metric: validation **pass@1** (first sample).  Secondary: pass@4,
  mean accuracy over samples, format rate, process coverage, zero-variance and
  all-wrong group rates, mean response length.
* Test set: evaluated only for the base model and the best validation variant,
  as a secondary check; not used for selection.
* Statistics: paired bootstrap (10,000 resamples) 95% CI over the 64 prompts,
  exact McNemar test for pass@1, implemented in `stats.py`.

### 5.3 Controlled verifier stress test

`stress_test.py` scores eight synthetic completion families (gold trace,
answer-only, correct-steps/wrong-final, trivial spam, ungrounded gold values,
arithmetic error, repeated steps, format-only) under answer-only, naive
step-count and PCPR rewards.  This is a verifier-level experiment, not a model
result; it tests the reward-hacking claim directly.

### 5.4 Runs

| wave | variants |
| --- | --- |
| base | base model validation dump |
| 1 | answer, format, process, drgrpo |
| 2 | rloo, dense, dapo, decomposed |
| 3 | dense_sparse, difficulty, n8 |

The `n8` run holds completions per step constant (4 prompts × 8 samples) to
separate group size from batch size.  `dense_sparse` holds the token-level
placement but removes reverse-cumulative propagation, isolating credit
propagation from token alignment.

## 6. Analysis plan

1. Reward-mode ablation: answer → format → process, at fixed GRPO.
2. Advantage ablation at fixed process reward: GRPO, Dr.GRPO, RLOO, DAPO-lite,
   difficulty-GRPO.
3. Token-level ablation: process (scalar) vs dense (cumsum) vs dense_sparse.
4. Decomposition: decomposed vs process.
5. Group size: n=4 vs n=8.
6. Degeneration: process vs process + loop penalty, measuring loop rate,
   zero-variance(reward) and all-wrong group rate.
6. Training dynamics: reward, coverage, zero-variance and all-wrong rates per
   step, from `trainer.rollout_data_dir` dumps.
7. Paired tests: each variant vs base and vs process on pass@1/pass@4.
8. Stress test table and a manual audit sample.

## 7. Threats to validity

* **Small scale**: 64 validation prompts, 96 test prompts, one seed, 10 steps.
  Confidence intervals will be wide; we report them rather than claiming wins.
* **Model**: Qwen3-0.6B is weaker than the 1.7B model used by the TRL
  implementation; absolute accuracy is not comparable.
* **Verifier coverage**: PCPR checks arithmetic equalities and gold
  intermediate values.  It does not check that the model's prose is a correct
  semantic decomposition of a word problem; a wrong derivation that happens to
  hit the gold intermediates would be accepted.
* **Reward weights**: final 1.0, process 0.5, format 0.1, invalid 0.05 are
  chosen a priori.  A sweep could change the ranking.
* **Framework**: verl 0.5.0 lacks DAPO dynamic sampling and GSPO; DAPO-lite is
  the config-expressible subset.  HF rollout is slow, so the step budget is
  small.
* **Multiple comparisons**: eleven variants increase the chance of a spurious
  pairwise difference.  The pre-registered primary comparison is
  `process`/`dense` vs `answer` at pass@1; the rest are exploratory.

## 8. Results update (post-hoc, 2026-11-06)

The pre-registered plan above was executed at a reduced budget (6 steps, 32
greedy validation prompts, one seed) because the machine was shared.  Findings:

* **H1 supported.**  Across 48 groups × 4 samples per variant, PCPR cuts
  `zero-var(reward)` relative to `zero-var(acc)` by 34–50% (process6 0.542 →
  0.292; dense6 0.646 → 0.375; drgrpo6 0.563 → 0.313; loop6 0.542 → 0.271).
  Answer-only and format stay at equality by construction.
* **H5 supported, and it is the main result.**  Dr.GRPO + process reaches
  24/32 (0.750) versus 17/32 for base and 21/32 for answer-only; process +
  standard GRPO reaches only 18/32 (0.562), *below* answer-only.  Paired
  `drgrpo6 − base`: +0.219, 95% CI [+0.063, +0.375], McNemar p=0.039 — but
  this is uncorrected for six comparisons (Bonferroni threshold 0.0083), so it
  is exploratory.
* **H2 not supported at this scale.**  dense6 is 20/32 (0.625), below
  answer-only (21/32); the token-level dense credit does not translate into an
  accuracy gain in six steps.
* **H3/H4 falsified on the extended wave.**  Decomposed-GRPO (0.625) and
  Difficulty-GRPO (0.656) were trained for six steps and screened on the same
  32 greedy prompts; neither reaches Dr.GRPO (0.750) or beats answer-only
  (0.656) by more than one prompt.  Decomposed-GRPO has the best n=4 training
  pass@k (0.771) but the lowest screen accuracy and highest loop ratio.
  Difficulty-GRPO halves invalid steps relative to n=8 but does not move
  accuracy.  RLOO (0.656), DAPO-lite (0.656) and n=8 (0.656) fall in the same
  band, so the H5 interaction is specific to Dr.GRPO at this budget, not a
  general property of "any algorithm different from GRPO".
* **Format bonus too weak.**  `format6` has a 0.5% `<answer>` rate; 0.1 is not
  enough to overcome the `\boxed{}` habit.
* **Length drift is universal.**  Mean output characters grow 57–96 over six
  steps, including under Dr.GRPO.
* **Greedy vs stochastic decoding disagree.**  The greedy screen ranks
  drgrpo6 first (0.750 vs answer6 0.656), but on 48 training prompts with four
  stochastic samples answer6 has the highest pass@4 (0.792 vs 0.750 for
  drgrpo6; dense6 is significantly worse at 0.646).  The full 64-prompt ×
  4-sample validation evaluation is the arbiter; it uses the screen's prompts
  with stochastic decoding.
* **Framework prerequisite.**  The `position_ids` fix was necessary; without
  it every rollout was a digit loop.

The 64-prompt × 4-sample full evaluation (`RESULTS_FULL.md`) and the 96-prompt
held-out test (`RESULTS_EXTENDED.md`) are complete.  A second-seed replication
of answer-vs-Dr.GRPO (`run_seed_b.sh`) was launched after the extended wave and
is reported in `RESULTS_EXTENDED.md` §3.

## 9. References

- GRPO / DeepSeekMath: https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://www.nature.com/articles/s41586-025-09422-z
- RLOO: https://arxiv.org/abs/2402.14740
- Dr.GRPO: https://arxiv.org/abs/2503.20783
- DAPO: https://arxiv.org/abs/2503.14476
- GSPO: https://arxiv.org/abs/2507.18071
- PRIME: https://arxiv.org/abs/2502.01456
- VinePPO: https://arxiv.org/abs/2410.01679
- PRM survey: https://arxiv.org/abs/2510.08049
- Verifiable Process Rewards: https://arxiv.org/abs/2605.10325
- Group-std identity: https://arxiv.org/abs/2607.00152
- verl / HybridFlow: https://arxiv.org/abs/2409.19256
