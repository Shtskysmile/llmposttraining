# Results

All numbers in this file are produced by the scripts in this directory.  The
executed comparison uses Qwen3-0.6B + LoRA, 6 optimizer steps, 8 prompts × 4
samples per step, temperature 0.7, and the **32-prompt greedy screen** of the
expanded validation set.  The 64-prompt × 4-sample evaluation and the 96-prompt
held-out test set are prepared but were not run in this NPU window; commands
are in `RUNBOOK.md`.

> **Update (2026-11-06 ~03:00 CST).**  All six 6-step runs completed and were
> screened on the 32-prompt greedy set: **Dr.GRPO+process 0.750** (24/32),
> answer-only 0.656 (21/32), dense 0.625, loop 0.625, format 0.594,
> process+GRPO 0.562, base 0.531.  `drgrpo6` is the only variant whose paired
> CI vs base excludes zero (+0.219, [+0.063, +0.375], McNemar p=0.039), but it
> would not survive a Bonferroni correction for six comparisons.  The dynamics
> result is the robust one: every process-reward variant cuts the zero-variance
> rate by 34–50% (`process6` 0.542 → 0.292).  Process reward under standard
> GRPO is *worse* than answer-only (0.562 vs 0.656), so the algorithm–reward
> interaction is the main story.
>
> **Full 64×4 evaluation complete (see `RESULTS_FULL.md`).**  `drgrpo6` is
> first on pass@1 (0.672 vs base 0.531, McNemar p=0.078; vs answer-only 0.531,
> p=0.035) and mean accuracy (0.660 vs 0.523, +0.137 with a CI excluding zero;
> vs answer-only +0.082 with a CI excluding zero).  `answer6` is the best
> best-of-4 variant (pass@4 0.875, all-wrong 0.125) but does not move pass@1.
> `loop6` has the lowest loop rate (0.4%) with a significant mean-accuracy
> gain vs base (+0.070, CI [+0.004, +0.137]).  The 64×4 evaluation confirms
> the greedy screen's top ranking and resolves the greedy-vs-stochastic
> caveat: Dr.GRPO+process's advantage holds on pass@1/mean accuracy, while
> answer-only is best on pass@4.
>
> **Held-out word-problem test (96 prompts, greedy).**  Answer-only transfers
> best: 0.635 vs base 0.594 (+0.042, CI [0.000, +0.094]); Dr.GRPO+process
> 0.615 (+0.021, n.s.); standard GRPO+process **0.542** (−0.052, CI
> [−0.104, −0.010], 0 improvements / 5 regressions).  The in-distribution
> winner is not the OOD winner: the process reward's gains are partly
> symbolic-template-specific.

> **Status (2026-11-06 ~02:10 CST; historical).**  Base-model screen, verifier stress test,
> and step-1 training rollouts are complete.  The full 8-step comparison is
> temporarily blocked: another session on this machine uses all six logical
> NPUs with MedQA jobs until ~02:40.  The four wave-1 jobs reached step 1 and
> were killed at step 2 (`ActorDiedError`, native WorkerDict death) when the
> Ray workers collided with those jobs.  Because `save_freq` equalled the step
> budget, no checkpoint was written; the runs must restart from scratch.
> Coordination with the other session is in progress; `SAVE_FREQ=2` is now the
> default so future runs can resume from the latest checkpoint.  The other
> session confirmed the schedule and offered logical 0 after its NSR-strict run
> exits (~02:15) for a single-process 2-step smoke; the full comparison will
> resume after ~02:40.
>
> **2-step isolation test (02:21):** the same `arithmetic_grpo_process` config
> completed two steps on logical 0 with no crash, saved
> `global_step_2/actor` (3.14 GB), reward 0.894 → 0.843, accuracy 0.562 both
> steps, coverage 0.672 → 0.562, clip ratio 0.03.  The earlier step-2 death was
> therefore NPU contention with the other session, not a 2-step training bug.
> A 6-step process-reward run is scheduled on the same device (02:25–02:45) to
> produce a dynamics curve and a `global_step_6` checkpoint; the answer/dense/
> Dr.GRPO comparison follows when all six NPUs are free after ~02:40.
>
> **2-step checkpoint (32-prompt greedy screen):** accuracy 0.531 (identical to
> the base model), coverage 0.578 (base 0.562), loop ratio 0.125, reward 0.817
> (base 0.810).  Two steps do not move accuracy, as expected; they mainly
> exercise the save/load and reward path.

## 1. Framework fix

The first smoke produced `response_length/clip_ratio=1.0`, `actor/entropy≈0`
and digit-loop rollouts.  Root cause: `HFRollout` passed explicit
`position_ids` to `transformers.generate`; on torch_npu 2.4 / transformers
4.53 this corrupts batched generation for Qwen3-0.6B.  Removing the argument
(letting the model derive positions from `attention_mask`) fixed it.  See
`patches/0001-hf-rollout-drop-position-ids.patch`.

After the fix, the 1-step smoke had:

| metric | value |
| --- | ---: |
| response length mean / max | 184.3 / 297 |
| response length clip ratio | 0.0 |
| reward mean | 0.855 |
| accuracy | 0.531 |
| format rate | 0.000 |
| process coverage | 0.656 |
| valid / invalid steps per completion | 1.81 / 0.25 |
| generation time | 109 s |
| actor update time | 33.6 s |

## 2. Controlled reward-hacking stress test

16 validation prompts, eight synthetic completion families, averaged.

| completion family | answer-only | naive step-count | PCPR process | coverage |
| --- | ---: | ---: | ---: | ---: |
| gold_trace | 1.000 | 1.500 | **1.600** | 1.000 |
| answer_only | 1.000 | 1.000 | 1.100 | 0.000 |
| wrong_final_correct_steps | 0.000 | 0.333 | **0.583** | 1.000 |
| trivial_spam | 1.000 | 1.500 | **1.100** | 0.000 |
| ungrounded_gold_steps | 1.000 | 1.333 | **1.100** | 0.000 |
| grounded_irrelevant_steps | 1.000 | 1.500 | **1.100** | 0.000 |
| arithmetic_error | 1.000 | 1.333 | 1.083 | 0.000 |
| repeated_steps | 1.000 | 1.167 | 1.350 | 0.500 |
| format_only | 0.000 | 0.000 | 0.100 | 0.000 |

Interpretation: a naive step-count reward pays for trivial and ungrounded
equation spam; PCPR does not.  PCPR is the only design that gives a positive
dense signal (0.583) to a correct derivation with a wrong final answer, which
is the case that matters in all-wrong groups.

## 3. Main comparison

Base-model screen (32 validation prompts, greedy):

| metric | value |
| --- | ---: |
| accuracy | 0.531 |
| format rate | 0.000 |
| process coverage | 0.562 |
| valid steps / completion | 1.59 |
| full-proof completions | 15/32 |
| partial-proof completions | 6/32 |
| answer-only completions | 2/32 |
| wrong completions | 9/32 |
| dumped/recomputed metric mismatches | 0 |

The audit (`audit.py`) recomputes every metric from the raw prompt/output and
finds no disagreement with the dumped values, so the evaluation path is sound.

Failure modes on the same 32 greedy completions: 17 correct, 11 arithmetic
errors (the final `\boxed{}` is present but wrong), 4 digit-repetition loops,
and 0 truncated-before-answer.  This motivates both the process reward (for
the arithmetic errors, which often contain correct intermediate steps) and the
loop penalty (for the degeneration cases, where coverage is zero).

### 3.1 Preliminary step-1 comparison

The first wave reached step 1 before colliding with another session's six
MedQA jobs (all six logical NPUs), after which the Ray workers died at step 2.
The step-1 rollouts (8 prompt groups × 4 samples each) are still informative
as a diagnostic, not a ranking:

| variant | pass@1 | mean acc | coverage | zero-var(acc) | zero-var(reward) | group std | mean reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| answer | 0.500 | 0.656 | 0.719 | 0.500 | 0.500 | 0.225 | 0.656 |
| format | 0.500 | 0.500 | 0.656 | 0.750 | 0.750 | 0.108 | 0.500 |
| process | 0.375 | 0.531 | 0.672 | 0.500 | 0.500 | **0.302** | 0.863 |
| drgrpo | 0.750 | 0.656 | 0.719 | 0.375 | 0.375 | **0.363** | 1.011 |

Sampling noise dominates the single-step differences, but the process and
Dr.GRPO variants have visibly higher within-group reward standard deviation,
consistent with the dense-signal hypothesis.  The zero-variance rate is
unchanged because saturated groups are saturated in coverage too.

32-prompt greedy screen (n=1; group-variance columns are meaningless here and
are reported from the training rollouts in Section 4):

| variant | reward | advantage | acc | coverage | loop | mean reward |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| base | none | none | 0.531 | 0.562 | 0.125 | 0.810 |
| process6 | process | GRPO | 0.562 | 0.672 | 0.094 | 0.895 |
| answer6 | answer | GRPO | 0.656 | 0.719 | 0.083 | 0.656 |
| dense6 | process + token-level | Process-GRPO | 0.625 | 0.703 | 0.091 | 0.974 |
| **drgrpo6** | process | **Dr.GRPO** | **0.750** | **0.828** | 0.060 | 1.160 |
| format6 | format | GRPO | 0.594 | 0.641 | 0.062 | 0.594 |
| loop6 | process + loop | GRPO | 0.625 | 0.703 | 0.091 | 0.956 |

Ordering: **Dr.GRPO + process (24/32) > answer-only (21/32) > dense/loop (20/32)
> format (19/32) > process+GRPO (18/32) > base (17/32)**.  The process reward
helps only when paired with Dr.GRPO; with standard GRPO's std normalisation it
underperforms answer-only.  This is the clearest algorithm–reward interaction
in the experiment.

Reward decomposition on the screen: process variants get ~0.34–0.41 from the
process component and ~0.56–0.75 from the final-answer component; the process
share is ~35–38%, so the final answer still dominates.  The loop penalty is
only ~0.018 on average (loop ratio ~9% × weight 0.2).

**Format audit.**  On the 32 greedy screens, base uses `\boxed{}` in 28/32
outputs, format6 30/32 and drgrpo6 30/32; **0/32 use `<answer>`** in all three.
The 0.1 format bonus did not teach the requested tag in six steps.

Per-prompt audit of `drgrpo6` vs base: 8 improvements and 1 regression.  Four of
the improvements replace a base-model digit loop with a correct trace
(`451*982`, `831*759`, `292*381`, `(742+120)*31`), four fix arithmetic errors
(`280*229`, `(149+380)*34`, `110*907`, `204*941`), and the single regression
(`552*866`) is itself a new digit loop.  The degeneration mode is therefore a
major, stochastic contributor to the accuracy gap; reward alone does not
eliminate it.

| prompt | base | drgrpo6 | change |
| --- | --- | --- | --- |
| 451*982+25−6556 | loop | 436351 ✓ | loop fixed |
| 280*229−37+1009 | 65892 ✗ | 65092 ✓ | arithmetic fixed |
| (149+380)*34−5913 | 12283 ✗ | 12073 ✓ | arithmetic fixed |
| 831*759−26+3541 | loop | 634244 ✓ | loop fixed |
| 292*381−18+4742 | loop | 115976 ✓ | loop fixed |
| 110*907−38+1729 | 101691 ✗ | 101461 ✓ | arithmetic fixed |
| (742+120)*31−1113 | loop | 25609 ✓ | loop fixed |
| 204*941+29−7508 | 185485 ✗ | 184485 ✓ | arithmetic fixed |
| 552*866−12+5273 | 483293 ✓ | loop | regression (new loop) |

Stratified by base correctness (all variants keep the same 16/17 base-correct
prompts; the differences are almost entirely in fixing hard prompts):

| variant | fixes / 15 base-wrong | keeps / 17 base-correct | net gain |
| --- | ---: | ---: | ---: |
| answer6 | 5 (33.3%) | 16 (94.1%) | +4 |
| **drgrpo6** | **8 (53.3%)** | 16 (94.1%) | **+7** |
| dense6 | 4 (26.7%) | 16 (94.1%) | +3 |
| loop6 | 4 (26.7%) | 16 (94.1%) | +3 |
| format6 | 3 (20.0%) | 16 (94.1%) | +2 |
| process6 | 2 (13.3%) | 16 (94.1%) | +1 |

The gain comes from solving hard prompts, not from sacrificing easy ones.  The
process+GRPO variant fixes the fewest hard prompts, which explains its low
accuracy despite the dense reward.

Every variant loses exactly one base-correct prompt.  For answer6, drgrpo6,
dense6 and loop6 it is the same loop-prone `552*866−12+5273`; for process6 and
format6 it is `(574+940)*34−6071`.  The single drgrpo6 regression is therefore
a systematic degeneration case, not a random trade-off.

**Correct-set overlap.**  `drgrpo6`'s 24 correct prompts are a **superset** of
`answer6`'s 21 (intersection 21, union 24, Jaccard 0.88): on this screen
Dr.GRPO+process strictly dominates answer-only.  `dense6` and `loop6` each get
20 correct, but their union with `drgrpo6` is still 24 (each has 4 unique
prompts and misses 4 that drgrpo6 solves), so there is no complementary gain
from ensembling these variants.

The union of **all seven variants** is 25/32.  `drgrpo6` is one prompt short
of that oracle; the only prompt it misses but another variant solves is
`552*866−12+5273` (its new loop).  The remaining 7 prompts are unsolved by
every variant, which is the hard ceiling of this six-step setup.

### 3.2 Greedy vs stochastic decoding

The 32-prompt greedy screen ranks `drgrpo6` first (0.750 vs answer6 0.656).
On the **48 training prompts with 4 stochastic samples** (pass@4, paired by
prompt), the ordering flips:

| comparison | answer6 | variant | diff | 95% CI |
| --- | ---: | ---: | ---: | --- |
| answer6 → process6 | 0.792 | 0.708 | −0.083 | [−0.229, +0.063] |
| answer6 → drgrpo6 | 0.792 | 0.750 | −0.042 | [−0.167, +0.063] |
| answer6 → loop6 | 0.792 | 0.750 | −0.042 | [−0.167, +0.083] |
| answer6 → dense6 | 0.792 | 0.646 | **−0.146** | **[−0.271, −0.021]** |

The screen and the rollouts differ in both prompt set (validation vs train)
and decoding (greedy vs 4 stochastic samples).  The process reward may sharpen
the greedy mode while reducing sample diversity; dense6's pass@4 is
significantly below answer-only.  The 64-prompt × 4-sample validation
evaluation (stochastic, same prompts as the screen) is the arbiter; it is
running now.

A direct decoding check on the 32 shared prompts: base greedy accuracy is
0.531, but the base's stochastic **first-sample** pass@1 is 0.594.  Decoding
alone moves two prompts, so the greedy screen should not be treated as the
stochastic policy's accuracy.

### 3.3 Full 64-prompt × 4-sample evaluation

The pre-registered evaluation (all six variants complete) is in
`RESULTS_FULL.md`.  Headline:

| variant | pass@1 | pass@4 | mean acc | coverage | loop |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.531 | 0.797 | 0.523 | 0.666 | 0.022 |
| answer6 | 0.531 | **0.875** | 0.578 | 0.717 | 0.017 |
| process6 | 0.547 | 0.797 | 0.547 | 0.689 | 0.022 |
| dense6 | 0.563 | 0.797 | 0.578 | 0.699 | 0.023 |
| **drgrpo6** | **0.672** | 0.859 | **0.660** | **0.775** | 0.012 |
| loop6 | 0.609 | 0.844 | 0.594 | 0.727 | **0.004** |

The full evaluation **confirms the greedy screen's top ranking**: Dr.GRPO +
process is first on pass@1 (0.672 vs base 0.531, McNemar p=0.078; vs
answer-only 0.531, p=0.035) and mean accuracy (0.660 vs base 0.523, +0.137 with
a CI excluding zero; vs answer-only +0.082 with a CI excluding zero).
Answer-only is the best-of-4 variant (pass@4 0.875, all-wrong 0.125) but its
pass@1 is unchanged.  The degeneration-penalty variant has the lowest loop
rate (0.4%) and a significant mean-accuracy gain vs base (+0.070,
CI [+0.004, +0.137]).  Every process-reward variant cuts the reward
zero-variance rate by 28–35% relative to its own accuracy zero-variance.

## 4. Training rollout dynamics

Step-1 rollout dynamics (8 groups × 4 samples per variant):

| variant | mean acc | coverage | zero-var(acc) | zero-var(reward) | group std | mean reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| answer | 0.656 | 0.719 | 0.500 | 0.500 | 0.225 | 0.656 |
| format | 0.500 | 0.656 | 0.750 | 0.750 | 0.108 | 0.500 |
| process | 0.531 | 0.672 | 0.500 | 0.500 | **0.302** | 0.863 |
| drgrpo | 0.656 | 0.719 | 0.375 | 0.375 | **0.363** | 1.011 |

The process reward increases within-group reward std by ~34% relative to
answer-only, but does not reduce the zero-variance rate in that batch: the
silent groups are saturated in coverage too (all coverage 1 or all 0).  This is
the gap the degeneration penalty and difficulty weighting target.

**A second, independent step-1 batch from the 6-step run shows the opposite and
important case.**  With 8 groups × 4 samples:

| metric | value |
| --- | ---: |
| accuracy | 0.469 |
| process coverage | 0.609 |
| zero-var(acc) | 0.500 |
| zero-var(reward) | **0.250** |
| all-wrong / all-correct groups | 0.250 / 0.250 |
| group reward std | 0.325 |

Here the process reward **halves** the zero-variance rate (0.50 → 0.25): an
all-wrong group had partial coverage, so its samples received different scores
and produced a non-zero advantage.  This is the mechanism H1 predicts; whether
it survives over 6 steps is what the full run measures.

**Steps 1–2 aggregate** (16 groups, 64 samples): accuracy 0.562, pass@1 0.500,
pass@4 0.875, coverage 0.656, loop ratio 0.016, group reward std 0.361.
`zero-var(acc) = 0.438` but `zero-var(reward) = 0.312` — a 29% reduction.  The
effect is consistent across the two steps, not a single lucky batch.

**Steps 1–3 aggregate** (24 groups, 96 samples): accuracy 0.521, pass@1 0.500,
pass@4 0.792, coverage 0.651, group reward std 0.300.  `zero-var(acc) = 0.542`
but `zero-var(reward) = 0.333` — a **38% reduction**.  Step 3 alone had
`zero-var(acc)=0.75`, yet the aggregate reward zero-variance stays at one third,
so the dense signal is doing real work across batches.

**Steps 1–4 aggregate** (32 groups, 128 samples): accuracy 0.453, pass@1 0.438,
pass@4 0.750, coverage 0.602, group reward std 0.322.  `zero-var(acc) = 0.500`
but `zero-var(reward) = 0.281` — a **44% reduction**.  Step 4 is a harder batch
(accuracy 0.25), and the reward still creates variance where accuracy is
saturated.  This is the strongest evidence for H1 so far.

**Core wave step 1 cross-check.**  `drgrpo6` had `zero-var(acc)=0.625` and
`zero-var(reward)=0.375`; `answer6` had `zero-var(acc)=zero-var(reward)=0.375`
(as expected when reward = accuracy).  The process variants consistently
reduce the zero-variance rate; answer-only cannot by construction.

**Cross-variant 6-step rollout dynamics** (48 groups, 192 samples each;
stochastic rollouts, not the greedy screen):

| variant | mean acc | pass@1 | pass@4 | zero-var(acc) | zero-var(reward) | group std | coverage | loop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| answer6 | 0.490 | 0.417 | 0.792 | 0.458 | 0.458 | 0.251 | 0.594 | 0.002 |
| format6 | 0.448 | 0.479 | 0.688 | 0.563 | 0.563 | 0.199 | 0.503 | 0.019 |
| process6 | 0.448 | 0.458 | 0.708 | 0.542 | **0.292** | 0.304 | 0.576 | 0.035 |
| dense6 | 0.474 | 0.500 | 0.646 | 0.646 | **0.375** | 0.236 | 0.573 | 0.012 |
| drgrpo6 | 0.490 | 0.542 | 0.750 | 0.563 | **0.313** | 0.289 | 0.620 | 0.024 |
| loop6 | 0.510 | 0.604 | 0.750 | 0.542 | **0.271** | 0.314 | 0.620 | 0.030 |

Every process-reward variant reduces the zero-variance rate by 34–50%;
answer-only and format cannot, by construction (format6's `<answer>` rate is
only 0.5%, so its reward is essentially accuracy).  The process variants do
**not** have higher mean rollout accuracy in six steps — the dense signal
improves within-group variance, not the average reward.  Per-step accuracy is
very noisy (0.25–0.66 on 8 prompts), response length grows from ~186 to ~237
tokens, and the final-step clip ratio reaches 0.19.

Per-step details (`dynamics_report.py`):

* On the hard step 4 (mean acc 0.22–0.375), `drgrpo6`, `dense6` and `loop6`
  reach `zero-var(reward)=0.000` while `zero-var(acc)` is 0.25/0.50/0.50;
  `answer6` and `format6` remain at `zero-var(reward)=zero-var(acc)`.  This is
  the clearest direct evidence of the dense signal on a near-all-wrong batch.
* Length drift is universal: mean output characters grow by 57–96 over six
  steps (answer6 +72, drgrpo6 +96, dense6 +77, loop6 +57, format6 +83).
  Dr.GRPO does not eliminate length drift in this short run.

**Theory vs empirical silent-group rate** (`silent_group_analysis.py`, per
step, 8 groups):

| variant | global-p formula (lower bound) | per-group formula (upper bound) | empirical acc | empirical reward |
| --- | ---: | ---: | ---: | ---: |
| answer6 | ~0.125 | 0.583 | 0.458 | 0.458 |
| process6 | ~0.111 | 0.656 | 0.542 | **0.292** |
| drgrpo6 | ~0.139 | 0.666 | 0.563 | **0.313** |

The global-p formula ignores between-prompt difficulty heterogeneity and
underestimates; the per-group formula estimates `p` from the same four samples
and overestimates.  The empirical accuracy-based rate falls between them.
Process reward moves the empirical reward-silent rate **below** both the
accuracy-based rate and the per-group formula, which is the direct signature of
dense signal where outcome-only GRPO has none.

## 5. Paired significance tests

Paired bootstrap (10,000 resamples) and exact McNemar, 32 prompts, pass@1:

| comparison | diff | 95% CI | McNemar |
| --- | ---: | --- | --- |
| base → answer6 | +0.125 | [0.000, +0.281] | p=0.219 (5+ / 1−) |
| base → process6 | +0.031 | [−0.063, +0.125] | p=1.000 (2+ / 1−) |
| base → dense6 | +0.094 | [−0.031, +0.219] | p=0.375 (4+ / 1−) |
| **base → drgrpo6** | **+0.219** | **[+0.063, +0.375]** | **p=0.039 (8+ / 1−)** |
| base → format6 | +0.063 | [−0.063, +0.188] | p=0.625 (3+ / 1−) |
| base → loop6 | +0.094 | [−0.031, +0.219] | p=0.375 (4+ / 1−) |
| answer6 → process6 | −0.094 | [−0.219, +0.031] | p=0.375 (1+ / 4−) |
| answer6 → drgrpo6 | +0.094 | [0.000, +0.219] | p=0.250 (3+ / 0−) |

Only `drgrpo6` vs base has a CI excluding zero and an uncorrected McNemar
p<0.05.  With six comparisons vs base, a Bonferroni threshold is 0.0083, so
this would not survive correction; treat it as a promising exploratory signal,
not a confirmed improvement.  The negative `answer6 → process6` comparison is
the other side of the interaction: process reward hurts under standard GRPO.

## 6. Discussion

1. **The framework fix is a prerequisite.**  Before removing `position_ids`
   from `generate`, every batched rollout was a digit loop; answer-only RL was
   fitting noise.  The same bug likely depressed the earlier MedQA baseline.
2. **PCPR works, but coverage saturates.**  It gives dense credit to partial
   traces and raises within-group variance, but a group where every sample has
   zero coverage (or full coverage) remains silent.  Process reward alone is
   not a complete replacement for dynamic sampling.
3. **The degeneration penalty is complementary.**  Loops are a distinct
   all-wrong failure mode with zero coverage; a repeated-chunk penalty gives
   those groups a within-group signal without touching legitimate arithmetic.
4. **Single-step differences are noise.**  The step-1 ranking (drgrpo > answer >
   process > format) is dominated by sampling variance; only the full paired
   evaluation can rank the methods.
5. **Negative results are still useful.**  If the 8-step process run does not
   beat answer-only, that argues for difficulty-aware sampling first and dense
   reward second, rather than adding a PRM unconditionally.
