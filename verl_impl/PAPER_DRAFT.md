# Proof-Carrying Process Rewards for Arithmetic RLVR on Ascend NPUs

*Draft — 2026-11-06. All numbers are produced by `verl_impl/` scripts; see
`RESULTS.md` for the raw record and `RELATED_WORK.md` for the pre-registered
protocol.*

## Abstract

Outcome-only RLVR gives one bit per trajectory.  On a task where the base model
already solves roughly half the prompts, most GRPO groups are saturated (all
correct or all wrong), so the group-relative advantage is exactly zero and no
learning happens.  We introduce **proof-carrying process rewards (PCPR)**: a
deterministic verifier parses the gold integer expression into an AST, accepts
model-written arithmetic equality chains only when every part is exact and the
leftmost operands are grounded in prompt literals or earlier accepted steps,
and rewards coverage of the gold intermediate values.  PCPR needs no learned
process reward model and no Monte-Carlo rollouts.  We implement it in verl
0.5.0 on Ascend 910B2 with Qwen3-0.6B + LoRA, together with token-level reward
placement, a dense Process-GRPO estimator, decomposed final/process advantages,
variance-weighted Dr.GRPO, and a degeneration penalty for digit loops.  In a
six-step comparison, PCPR cuts the zero-variance group rate by 28–50% across
four advantage variants.  On the pre-registered 64-prompt × 4-sample
evaluation, Dr.GRPO + process reaches **pass@1 0.672 and mean accuracy 0.660**,
versus 0.531/0.523 for the base model and 0.531/0.578 for answer-only GRPO; it
significantly beats answer-only on pass@1 (McNemar p=0.035, 12 improvements /
3 regressions) and on mean accuracy (paired CI excludes zero).  Answer-only is
the best best-of-4 variant (pass@4 0.875, all-wrong 0.125) but does not move
pass@1.  The degeneration-penalty variant has the lowest loop rate (0.4%) and
a significant mean-accuracy gain over base.  Process reward under standard
GRPO is not better than answer-only, so the algorithm–reward interaction is
more important than the reward alone.  On the 96 held-out word problems, the
in-distribution winner is **not** the transfer winner: answer-only scores
0.635 vs base 0.594, Dr.GRPO+process 0.615, and standard GRPO+process 0.542
(paired CI excludes zero on the negative side; 0 improvements / 5
regressions).  Dense process rewards must be evaluated for OOD transfer, not
only in-distribution accuracy.

## 1. Introduction

RLVR with a rule-based final-answer verifier is the standard recipe for
mathematical reasoning.  Its weaknesses are well known:

1. **Silent groups.**  For binary rewards with within-prompt success rate `p`
   and group size `G`, a group is silent with probability `p^G + (1-p)^G`.
   With `G=4`, `p=0.8` gives 0.41 and `p=0.9` gives 0.66.  DAPO's dynamic
   sampling discards and regenerates silent groups; the cost is extra rollout
   compute and starvation late in training.
2. **Format is ignored.**  Our base model emits `\boxed{}` but almost never
   the requested `<answer>` tag, and an answer-only reward does not care.
3. **A correct number is not a proof.**  A final-answer verifier cannot
   distinguish a lucky answer from a correct derivation.

PCPR addresses all three with a deterministic dense reward.  The key design
choice is to reward **coverage of the gold derivation**, not the number of
arithmetically valid equations: the latter is trivially farmed with
`1 + 1 = 2`, while the former requires the model to actually compute.

## 2. Method

### 2.1 Proof-carrying process verifier

Given a prompt, `expression_from_prompt` extracts the canonical integer
expression for both templates (symbolic `Compute exactly: ...` and the factory
word problem).  `canonical_derivation` evaluates the AST and enumerates the
intermediate values.

The model output is normalised (LaTeX `\times`/`\cdot`/`\div`, Unicode
operators, thousands separators, step labels, negative intermediates) while
preserving character offsets.  Maximal arithmetic runs are split on `=` and
each part is evaluated exactly.  A chain is accepted iff

* all parts are equal, and
* every leaf of the leftmost part is in the grounded set (prompt literals plus
  results of earlier accepted chains).

Accepted results are added to the grounded set.  `coverage` is the fraction of
gold intermediate values that appear as accepted results.  This handles flat
steps and place-value decompositions such as
`552 × 866 = 552 × (800+60+6) = … = 478032`.

### 2.2 Rewards and advantages

The default reward is

```
score = 1.0 * correct + 0.5 * coverage + 0.1 * <answer>_ok
        - 0.05 * invalid_rate - loop_penalty
```

`loop_penalty` detects a 40-character chunk repeated at least three times, a
strong signature of the 0.6B model's digit-loop degeneration.  All modes
(`answer`, `format`, `process`) return the same diagnostics, so every run is
evaluated identically.

We implement five advantage estimators:

* **GRPO** and **Dr.GRPO** (group mean, with/without std normalisation).
* **Process-GRPO**: reverse-cumulative returns from token-level rewards,
  normalised per group and per token position.
* **Decomposed-GRPO**: group-normalise the final-answer and process channels
  separately, then `A = A_final + λ A_process`.
* **Difficulty-GRPO**: multiply the Dr.GRPO advantage by `σ^α` (normalised by
  the mean weight), a soft alternative to DAPO's hard silent-group filter.

A custom reward manager places each verified step's reward on the token where
the step completes and the final/format reward on the last token; the
per-sample total is unchanged.

### 2.3 Framework fix

`HFRollout` passed explicit `position_ids` to `transformers.generate`.  On
torch_npu 2.4 / transformers 4.53 this corrupts batched Qwen3-0.6B generation
into digit loops even without padding; single-prompt generation is fine.  The
first smoke had `clip_ratio=1.0` and `actor/entropy≈0`.  Removing the argument
(letting the model derive positions from `attention_mask`) fixed the rollouts.
The patch is `patches/0001-hf-rollout-drop-position-ids.patch`.  The same
rollout path is used by the machine's MedQA demo, whose base accuracy of 2.35%
(below 25% random) may share this cause.

## 3. Experimental setup

* Model: Qwen3-0.6B, LoRA r=16/α=32, thinking disabled.
* Data: 256 symbolic train / 64 validation / 96 held-out word-problem test.
* Training: 6 steps, 8 prompts × 4 samples per step, lr 1e-5, KL 0.001,
  prompt ≤256, response ≤320, temperature 0.7, top_p 0.95, top_k 20.
* Evaluation: 32-prompt greedy screen (primary, single seed); a 64-prompt ×
  4-sample full evaluation and the 96-prompt test set are prepared and run
  separately.
* Statistics: paired bootstrap (10,000 resamples) 95% CI and exact McNemar.

## 4. Results

### 4.1 Reward-hacking stress test

16 validation prompts, eight synthetic completion families, averaged:

| family | answer-only | naive step-count | PCPR | coverage |
| --- | ---: | ---: | ---: | ---: |
| gold_trace | 1.000 | 1.500 | **1.600** | 1.000 |
| wrong_final_correct_steps | 0.000 | 0.333 | **0.583** | 1.000 |
| trivial_spam | 1.000 | 1.500 | **1.100** | 0.000 |
| ungrounded_gold_steps | 1.000 | 1.333 | **1.100** | 0.000 |
| grounded_irrelevant_steps | 1.000 | 1.500 | **1.100** | 0.000 |
| arithmetic_error | 1.000 | 1.333 | 1.083 | 0.000 |
| repeated_steps | 1.000 | 1.167 | 1.350 | 0.500 |
| format_only | 0.000 | 0.000 | 0.100 | 0.000 |

PCPR is the only design that pays for a correct derivation with a wrong final
answer while refusing trivial and ungrounded spam.  It also refuses
`grounded_irrelevant_steps`: arithmetically valid steps built from prompt
literals that do not contribute any gold intermediate value earn no coverage
(naive step-count pays 1.5, PCPR 1.1).

### 4.2 Six-step greedy screen (32 prompts)

| variant | reward | advantage | acc | coverage | loop |
| --- | --- | --- | ---: | ---: | ---: |
| base | — | — | 0.531 | 0.562 | 0.125 |
| process6 | process | GRPO | 0.562 | 0.672 | 0.094 |
| answer6 | answer | GRPO | 0.656 | 0.719 | 0.083 |
| dense6 | process + token | Process-GRPO | 0.625 | 0.703 | 0.091 |
| **drgrpo6** | process | **Dr.GRPO** | **0.750** | **0.828** | 0.060 |
| format6 | format | GRPO | 0.594 | 0.641 | 0.062 |
| loop6 | process + loop | GRPO | 0.625 | 0.703 | 0.091 |

Paired vs base: only `drgrpo6` has a CI excluding zero (+0.219,
[+0.063, +0.375], McNemar p=0.039, 8 improvements / 1 regression).  With six
comparisons a Bonferroni threshold is 0.0083, so this is exploratory, not
confirmed.  `answer6 → process6` is negative (−0.094), which is the other side
of the interaction.

Per-prompt audit of the 9 changed prompts: four improvements replace a base
digit loop with a correct trace, four fix arithmetic errors, and the single
regression is itself a new digit loop.  The degeneration mode is a major and
stochastic contributor to the accuracy gap; reward alone does not eliminate it.

The 24 correct prompts of `drgrpo6` are a superset of `answer6`'s 21
(Jaccard 0.88); dense6/loop6 add no complementary prompts.  The union of all
seven variants is 25/32, so `drgrpo6` is one prompt short of the all-variant
oracle (the missing one is its new loop); seven prompts are unsolved by every
variant, the hard ceiling of the six-step setup.

### 4.2b 64-prompt × 4-sample full evaluation

On the pre-registered stochastic evaluation (all six variants):

| variant | pass@1 | pass@4 | mean acc | coverage | loop | all-wrong |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.531 | 0.797 | 0.523 | 0.666 | 0.022 | 0.203 |
| answer6 | 0.531 | **0.875** | 0.578 | 0.717 | 0.017 | **0.125** |
| process6 | 0.547 | 0.797 | 0.547 | 0.689 | 0.022 | 0.203 |
| dense6 | 0.563 | 0.797 | 0.578 | 0.699 | 0.023 | 0.203 |
| **drgrpo6** | **0.672** | 0.859 | **0.660** | **0.775** | 0.012 | 0.141 |
| loop6 | 0.609 | 0.844 | 0.594 | 0.727 | **0.004** | 0.156 |

Dr.GRPO+process is first on pass@1 and mean accuracy.  Paired vs base:
pass@1 +0.141 [0.000, +0.281], McNemar p=0.078 (15+/6−); mean accuracy +0.137
[+0.078, +0.195].  Paired vs answer-only: pass@1 +0.141 [+0.031, +0.250],
p=0.035 (12+/3−); mean accuracy +0.082 [+0.020, +0.145].  Answer-only is the
best best-of-4 variant (pass@4 0.875, all-wrong 0.125) but its pass@1 is
unchanged.  The loop-penalty variant has the lowest loop rate (0.4%) and a
significant mean-accuracy gain vs base (+0.070, CI [+0.004, +0.137]).

Best-of-4 gain (`pass@4 - pass@1`): answer6 +0.344, base +0.266, process6
+0.250, loop6 +0.234, dense6 +0.234, drgrpo6 +0.188.  Answer-only gains the
most from sampling diversity; Dr.GRPO+process gains the least but starts from
the highest pass@1.  The variants improve different parts of the distribution.

### 4.2c Held-out word-problem transfer

The 96 held-out word problems use a different prompt template from the symbolic
training data (greedy, one sample per prompt):

| variant | accuracy | coverage | format | loop |
| --- | ---: | ---: | ---: | ---: |
| base | 0.594 | 0.609 | 0.104 | 0.042 |
| **answer6** | **0.635** | 0.667 | 0.031 | 0.042 |
| drgrpo6 | 0.615 | 0.656 | 0.052 | 0.042 |
| process6 | 0.542 | 0.589 | 0.094 | 0.010 |

Paired vs base: answer6 +0.042 [0.000, +0.094] (p=0.219, 5+/1−);
drgrpo6 +0.021 [−0.031, +0.073] (p=0.688, 4+/2−); process6 **−0.052**
[−0.104, −0.010] (p=0.063, **0+/5−**).

**The in-distribution winner is not the OOD winner.**  Answer-only transfers
best (57→61 correct); Dr.GRPO+process is positive but smaller (57→59); standard
GRPO+process hurts, with all five discordant pairs regressing and a CI
excluding zero.  The process reward's in-distribution gains appear to be
partly template-specific; answer-only learns a more general "solve and answer"
behavior.  An honest evaluation of dense process rewards must report OOD
transfer, not only in-distribution pass@1.

### 4.2d Extended algorithm wave (negative result)

We trained five further PCPR variants in the same six-step budget: RLOO, a
DAPO-lite recipe (clip-higher 0.28, no KL, overlong penalty), Decomposed-GRPO,
Difficulty-GRPO, and a group-size ablation at `G=8` with four prompts per step
(same 32 completions per step).  On the 32-prompt greedy screen:

| variant | advantage | accuracy | coverage | loop | mean reward |
| --- | --- | ---: | ---: | ---: | ---: |
| rloo | RLOO (G=4) | 0.656 | **0.750** | 0.028 | 1.028 |
| dapo | GRPO (DAPO-lite) | 0.656 | 0.734 | 0.063 | 1.014 |
| decomposed | Decomposed-GRPO | 0.625 | 0.703 | 0.091 | 0.974 |
| difficulty | Difficulty-GRPO | 0.656 | 0.734 | 0.060 | 1.021 |
| n8 | GRPO, G=8 | 0.656 | 0.719 | 0.091 | 1.012 |
| _ref_ process6 | GRPO (G=4) | 0.562 | 0.672 | 0.094 | 0.895 |
| _ref_ answer6 | answer-only | 0.656 | 0.719 | 0.083 | 0.656 |
| _ref_ drgrpo6 | Dr.GRPO (G=4) | **0.750** | **0.828** | 0.060 | 1.160 |

With 32 prompts one sample is 3.1 pp, so only Dr.GRPO's 0.750 separates from
the pack; all five new variants sit at 0.625–0.656.  **No advantage estimator
or group-size change reproduces the Dr.GRPO+process gain at this budget.**  In
the training rollouts, however, all five keep the reward zero-variance rate at
0.33–0.38 versus 0.46 for answer-only, so the dense-signal mechanism is
estimator-independent.  `G=8` has the best training pass@k (0.875 at k=8) and
the lowest all-wrong-group rate (0.125), consistent with more samples per
prompt, but this does not transfer to the greedy screen.  Decomposed-GRPO has
the best training pass@k among the extended n=4 variants (0.771, below
answer-only's 0.792) yet the lowest screen accuracy — at six steps the
final/process decomposition adds variance without adding signal.

One secondary signal is worth flagging: length drift over six steps is +97
(RLOO), +95 (DAPO-lite), +95 (Difficulty-GRPO), +68 (Decomposed-GRPO) and only
**+27 for G=8**, versus +96 for Dr.GRPO and +79 for process-GRPO at G=4.  The
G=8 run is the only one that roughly holds output length.  With eight samples
per prompt both the correct and the incorrect mode appear in every group, so
the comparison is available within the group rather than across prompts.  This
is suggestive, not established: one seed, six steps, four prompts per step.
The drift itself is not reward-driven: length correlates negatively with
reward in every variant (r = −0.10 to −0.30), so longer traces score worse,
yet outputs still grow.  It is a policy-distribution drift, not verbosity
exploitation.

### 4.2e Calibration of the dense signal

A dense reward is useful only if it agrees with the outcome.  We score
`process_coverage` as a predictor of final correctness (`process_auc.py`).
On the 96 held-out word problems the AUC is 0.958 base, 0.947 answer6, 0.933
drgrpo6, 0.947 process6 — bootstrap 95% CIs all have lower bounds ≥0.875.
Correct answers carry mean coverage 0.947–0.975; wrong answers 0.115–0.162.
Across the thirteen symbolic screens the AUC range is 0.914–0.963.  On the
256-sample stochastic 64×4 evaluation it is 0.910–0.942 (CIs ≥0.871), lower
because wrong samples then include partial traces with nonzero coverage.  The
verifier is measuring genuine arithmetic progress on both prompt templates,
including in the base model that was never trained with this reward.

The separation is partly driven by degenerate outputs with zero coverage, but
17–25% of wrong OOD samples contain at least one verified step with mean
coverage 0.64–0.75 ("right steps, wrong final answer").  The calibration
result and the OOD-accuracy result point in opposite directions: PCPR is
well-calibrated out of distribution while standard GRPO training on it
degrades OOD accuracy.  A dense reward can be an accurate predictor and a poor
learning signal; the advantage estimator determines how its variance is used.

### 4.3 Training rollout dynamics

Each variant has 48 groups × 4 samples (192 rollouts).  `zero-var(reward)`
versus `zero-var(acc)`:

| variant | zero-var(acc) | zero-var(reward) | reduction |
| --- | ---: | ---: | ---: |
| answer6 | 0.458 | 0.458 | 0% |
| format6 | 0.563 | 0.563 | 0% |
| process6 | 0.542 | 0.292 | **46%** |
| dense6 | 0.646 | 0.375 | **42%** |
| drgrpo6 | 0.563 | 0.313 | **44%** |
| loop6 | 0.542 | 0.271 | **50%** |

On the hard step 4 (mean accuracy 0.22–0.375), the process variants reach
`zero-var(reward)=0.000` while answer/format stay at `zero-var(reward)=zero-var(acc)`.
This is the mechanism the paper hypothesises.  Length drift is universal:
mean output characters grow by 57–96 over six steps, and Dr.GRPO does not
eliminate it in this short run.

We also checked the silent-group formula directly.  With the global success
rate `p`, `p^4+(1-p)^4` is a lower bound because it ignores between-prompt
difficulty heterogeneity (≈0.11–0.14 here); using each group's empirical `p`
is an upper bound because `p` is estimated from the same four samples (≈0.58
for answer6).  The empirical accuracy-based silent rate is 0.458, between the
bounds.  PCPR moves the reward-based silent rate to 0.292 (process6) and 0.313
(drgrpo6), below both the accuracy-based rate and the upper bound — a direct
signature of dense signal in groups that outcome-only GRPO leaves silent.

The untrained base model on the 64-prompt × 4-sample evaluation has pass@1
0.531 and pass@4 0.797.  Its accuracy-based silent rate is 0.547, but applying
the process-shaped reward to the *same untrained policy* lowers the
reward-based silent rate to 0.375.  The dense property is therefore a property
of the reward, not of training.  More directly: of the 13 all-wrong
(outcome-silent) groups, **11 receive a nonzero process score and 10 have
within-group score variance** — the process reward rescues 77% of the silent
groups without any training.  Across the 52 samples in those groups, 37 (71%)
get a nonzero score (mean 0.167, mean coverage 0.356) even though every final
answer is wrong.

| base group category | groups | mean acc | mean coverage | mean process score |
| --- | ---: | ---: | ---: | ---: |
| all-correct | 22 | 1.000 | 0.994 | 1.497 |
| mixed | 29 | 0.397 | 0.556 | 0.670 |
| all-wrong | 13 | 0.000 | **0.356** | **0.167** |

Under answer-only reward the all-wrong row is exactly 0; PCPR turns those 13
groups into a mean score of 0.167 with 35.6% of the gold derivation proved.

## 5. Discussion

The main finding is an **algorithm–reward interaction**.  PCPR reliably
increases within-group reward variance and halves the silent-group rate, but
that signal only translates into accuracy when the advantage estimator does
not over-amplify it.  Dr.GRPO's removal of std normalisation is the right
partner: std normalisation divides by a small group std exactly in the
near-saturated groups where PCPR is trying to create signal, which can make
the update noisy or harmful.  Process reward with standard GRPO is worse than
answer-only on this task.  The 64×4 pre-registered evaluation confirms the
interaction with statistical support: Dr.GRPO+process reaches pass@1 0.672 /
mean accuracy 0.660, significantly above answer-only on pass@1 (p=0.035) and
on mean accuracy (paired CI excludes zero).  Answer-only is the best best-of-4
variant (pass@4 0.875), and the degeneration-penalty variant has the lowest
loop rate (0.4%) with a significant mean-accuracy gain over base.  The three
outcomes are complementary rather than contradictory: process+Dr.GRPO improves
the first sample, answer-only improves diversity, and the loop penalty
suppresses the degeneration mode.

A crucial caveat: the greedy screen and the stochastic rollouts disagree.  On
the 32-prompt greedy validation screen `drgrpo6` is first (0.750 vs answer6
0.656), but on 48 training prompts with four stochastic samples the paired
pass@4 is 0.750 for drgrpo6 versus 0.792 for answer6 (diff −0.042, CI
[−0.167, +0.063]); dense6 is significantly worse (0.646, CI [−0.271, −0.021]).
The two comparisons differ in both prompt set and decoding.  PCPR may sharpen
the greedy mode while reducing sample diversity.  A direct decoding check on
the 32 shared prompts confirms the effect: base greedy accuracy is 0.531, but
base stochastic first-sample pass@1 is 0.594.  The 64-prompt × 4-sample
validation evaluation, which uses the screen's prompts with stochastic
decoding, is the arbiter and is reported in §4.2b: Dr.GRPO+process wins
there.

The format bonus is too weak: 0.1 is not enough to overcome the model's
`\boxed{}` habit in six steps (format rate 0.5%).  The degeneration penalty
targets a real failure mode but does not show an accuracy gain in six steps.

## 6. Limitations

* 32 greedy validation prompts, six steps, one seed for the main comparison;
  the 64×4 full evaluation and 96-prompt test set are separate.
* Qwen3-0.6B is much weaker than the 1.7B model used by the original TRL
  implementation; absolute accuracy is not comparable.
* PCPR verifies arithmetic equalities and gold intermediate values, not the
  semantic decomposition of a word problem.
* The extended wave (RLOO, DAPO-lite, Decomposed-GRPO, Difficulty-GRPO, G=8)
  is trained but screened only on 32 greedy prompts; none beats Dr.GRPO+process
  there, and a larger budget may be needed for those estimators to separate.
* Reward weights are chosen a priori and not swept.
* The shared machine required explicit NPU coordination; one collision killed
  the first wave at step 2.

## 7. Related work

GRPO (Shao et al., 2024), Dr.GRPO (Liu et al., COLM 2025), DAPO (Yu et al.,
NeurIPS 2025), RLOO (Ahmadian et al., ACL 2024), GSPO (Zheng et al., 2025),
PRIME (Cui et al., 2025), VinePPO (Kazemnejad et al., ICML 2025), the PRM
survey (Zheng et al., 2025), Verifiable Process Rewards (Yuan et al., 2026),
the group-std identity (2026), and verl/HybridFlow (Sheng et al., EuroSys
2025).  PCPR is a deterministic, arithmetic-domain instance of the VPR
principle; unlike a learned PRM or VinePPO's Monte-Carlo estimates, its reward
cannot be farmed with trivially true equations.
