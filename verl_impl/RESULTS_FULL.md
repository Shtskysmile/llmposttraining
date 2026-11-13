# Full 64-prompt × 4-sample evaluation

Produced by `verl_impl/full_eval_core.sh` (64 validation prompts, 4 stochastic
samples each, temperature 0.7) and analyzed by `verl_impl/analyze_full.sh`.
This is the pre-registered evaluation; the 32-prompt greedy screen in
`RESULTS.md` is the cheaper ranking.  All six variants completed.

## 1. Main table

| variant | pass@1 | pass@4 | mean acc | coverage | format | loop | mean reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.531 | 0.797 | 0.523 | 0.666 | 0.000 | 0.022 | 0.852 |
| answer6 | 0.531 | **0.875** | 0.578 | 0.717 | 0.004 | 0.017 | 0.578 |
| process6 | 0.547 | 0.797 | 0.547 | 0.689 | 0.000 | 0.022 | 0.887 |
| dense6 | 0.563 | 0.797 | 0.578 | 0.699 | 0.000 | 0.023 | 0.924 |
| **drgrpo6** | **0.672** | 0.859 | **0.660** | **0.775** | 0.000 | 0.012 | 1.045 |
| loop6 | 0.609 | 0.844 | 0.594 | 0.727 | 0.000 | **0.004** | 0.952 |

**Dr.GRPO + process is the best variant on first-sample accuracy and mean
accuracy**; answer-only is best on pass@4.  This confirms the 32-prompt greedy
screen's top ranking under the pre-registered stochastic evaluation.  The
degeneration-penalty variant has the lowest loop rate (0.4%) and the second
best pass@1.

Best-of-4 gain (`pass@4 - pass@1`): answer6 +0.344, base +0.266, process6
+0.250, loop6 +0.234, dense6 +0.234, drgrpo6 +0.188.  Answer-only gains the
most from sampling diversity; Dr.GRPO+process gains the least but starts from
the highest pass@1.  The variants improve different parts of the distribution.

## 2. Group variance

| variant | zero-var(acc) | zero-var(reward) | reduction | all-wrong | all-correct | group std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.547 | 0.375* | — | 0.203 | 0.344 | 0.283 |
| answer6 | 0.453 | 0.453 | 0% | **0.125** | 0.328 | 0.246 |
| process6 | 0.531 | 0.359 | 32% | 0.203 | 0.328 | 0.303 |
| dense6 | 0.594 | 0.406 | 32% | 0.203 | 0.391 | 0.260 |
| drgrpo6 | 0.563 | 0.406 | 28% | 0.141 | 0.422 | 0.277 |
| loop6 | 0.531 | 0.344 | **35%** | 0.156 | 0.375 | 0.302 |

\* The base eval uses `arithmetic_grpo_process`, so its `score` is already
process-shaped; `zero-var(reward)=0.375` is not an answer-only baseline.  The
policy-level comparison is `zero-var(acc)=0.547`.  That the process-shaped
reward lowers the *untrained* base model's silent-group rate from 0.547 to
0.375 is a direct demonstration that the dense property comes from the reward,
not from training.

Every process-reward variant reduces the reward zero-variance rate by 28–35%
relative to its own accuracy zero-variance; answer-only cannot by construction.

## 3. Paired tests vs base (pass@1, 64 prompts)

| comparison | base | variant | diff | 95% CI | McNemar |
| --- | ---: | ---: | ---: | --- | --- |
| base → answer6 | 0.531 | 0.531 | 0.000 | [−0.125, +0.125] | p=1.000 (9+ / 9−) |
| base → process6 | 0.531 | 0.547 | +0.016 | [−0.109, +0.141] | p=1.000 (8+ / 7−) |
| base → dense6 | 0.531 | 0.563 | +0.031 | [−0.109, +0.172] | p=0.824 (11+ / 9−) |
| **base → drgrpo6** | 0.531 | **0.672** | **+0.141** | **[0.000, +0.281]** | p=0.078 (15+ / 6−) |
| base → loop6 | 0.531 | 0.609 | +0.078 | [−0.063, +0.219] | p=0.383 (13+ / 8−) |
| **answer6 → drgrpo6** | 0.531 | **0.672** | **+0.141** | **[+0.031, +0.250]** | **p=0.035 (12+ / 3−)** |

Mean accuracy (paired bootstrap):
`base → drgrpo6` +0.137 [+0.078, +0.195], `base → loop6` +0.070
[+0.004, +0.137], `answer6 → drgrpo6` +0.082 [+0.020, +0.145].  pass@4:
`base → answer6` +0.078 [−0.016, +0.172]; `answer6 → drgrpo6` −0.016
[−0.078, +0.047].  **Dr.GRPO+process significantly beats answer-only on
pass@1 and mean accuracy, but answer-only is best on pass@4.**

## 4. Base difficulty and silent-group rescue

Base difficulty distribution (64 prompts × 4): **22 all-correct, 13 all-wrong,
29 mixed**.  The 13 all-wrong groups are the outcome-silent groups PCPR
targets; the 29 mixed groups are where it adds variance; the 22 all-correct
groups are saturated for any outcome reward.

**Silent-group rescue.**  Of the 13 all-wrong groups, **11 receive a nonzero
process score and 10 have within-group score variance** (77% of the outcome-
silent groups).  The 13 groups contain 52 samples: **37/52 (71%) get a nonzero
process score**, with mean score 0.167 and mean coverage 0.356 even though
every final answer is wrong.  Examples: `[0.00, 0.00, 0.000, 0.24]` (one sample
has coverage 0.5) and `[0.00, 0.00, 0.237, 0.25]` (two samples have coverage
0.5).  This is the direct mechanism for the zero-variance reduction, and it is
present in the *untrained* base model.

| base group category | groups | mean acc | mean coverage | mean process score |
| --- | ---: | ---: | ---: | ---: |
| all-correct | 22 | 1.000 | 0.994 | 1.497 |
| mixed | 29 | 0.397 | 0.556 | 0.670 |
| all-wrong | 13 | 0.000 | **0.356** | **0.167** |

Under answer-only reward the all-wrong row would be exactly 0; the process
reward turns those 13 groups into a mean score of 0.167 with 35.6% of the gold
derivation proved.

## 5. Interpretation

1. **The core finding holds under the pre-registered evaluation.**
   Dr.GRPO+process is first on pass@1 (0.672 vs base 0.531, p=0.078;
   vs answer-only 0.531, p=0.035) and mean accuracy (0.660 vs base 0.523,
   +0.137 with a CI excluding zero; vs answer-only +0.082 with a CI excluding
   zero).
2. **Answer-only is the best-of-4 variant.**  Its pass@4 is 0.875 and its
   all-wrong rate is the lowest (0.125), but its pass@1 is unchanged.  It
   improves diversity/best-of-n, not the first sample.
3. **The degeneration penalty works.**  loop6 has the lowest loop rate
   (0.004 vs base 0.022) and a significant mean-accuracy gain vs base
   (+0.070, CI [+0.004, +0.137]).  Its pass@1 is 0.609, second to drgrpo6.
4. **Standard GRPO + process is not enough.**  process6 and dense6 are close
   to base on pass@1 (0.547/0.563) and pass@4 (0.797).  The algorithm–reward
   interaction remains the main story.
5. **PCPR's dense property is real.**  Even the untrained base model's
   13 all-wrong groups get nonzero process scores in 71% of samples and
   35.6% mean coverage; the process reward is not learning a value function,
   it is exposing the partial derivation that outcome-only reward discards.

## 6. Power projection

The 32-prompt screen had 8 improvements and 1 regression for `drgrpo6` vs base
(fix rate 8/15 among base-wrong prompts, break rate 1/17 among base-correct
prompts).  The 64-prompt full evaluation observed 15 improvements / 6
regressions vs base, with a pass@1 diff of +0.141 and a paired bootstrap CI of
[0.000, +0.281]; the exact McNemar p is 0.078.  The larger evaluation is the
more informative one and confirms the direction.
