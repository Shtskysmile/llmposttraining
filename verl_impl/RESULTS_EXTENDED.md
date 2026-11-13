# Extended results

This file holds the 96-prompt held-out test evaluation and the extended
algorithm wave.  It is filled by `eval_test.sh` and `run_extended_wave.sh`
(logs under `verl_impl/outputs/logs/`).

## 1. Held-out word-problem test (96 prompts, greedy)

| variant | accuracy | coverage | format | loop |
| --- | ---: | ---: | ---: | ---: |
| base | 0.594 | 0.609 | 0.104 | 0.042 |
| **answer6** | **0.635** | 0.667 | 0.031 | 0.042 |
| drgrpo6 | 0.615 | 0.656 | 0.052 | 0.042 |
| process6 | 0.542 | 0.589 | 0.094 | 0.010 |

The base model scores **59.4%** (57/96) on the held-out word problems, higher
than its 53.1% on the symbolic validation screen.  Its `<answer>`-tag rate is
10.4% on the word problems versus 0% on the symbolic prompts, so the prompt
template materially changes format compliance.  Failure modes: 57 correct,
**35 arithmetic errors**, 4 digit loops, 0 truncated — the OOD failures are
mostly arithmetic slips, not degeneration.

**Transfer:** the algorithm–reward interaction extends to the OOD test, but the
best OOD variant is **answer-only**, not the in-distribution winner.

| comparison | base | variant | diff | 95% CI | McNemar |
| --- | ---: | ---: | ---: | --- | --- |
| base → answer6 | 0.594 | **0.635** | **+0.042** | **[0.000, +0.094]** | p=0.219 (5+ / 1−) |
| base → drgrpo6 | 0.594 | 0.615 | +0.021 | [−0.031, +0.073] | p=0.688 (4+ / 2−) |
| base → process6 | 0.594 | 0.542 | **−0.052** | **[−0.104, −0.010]** | p=0.063 (0+ / 5−) |

Answer-only transfers best (57→61 correct); Dr.GRPO+process is positive but
smaller (57→59); standard process+GRPO **hurts** OOD: all five discordant
pairs regress, and the paired CI excludes zero.  The in-distribution winner is
not the OOD winner.  A plausible interpretation is that the process reward
encourages a symbolic-trace style that helps the symbolic validation mode but
does not transfer as well as the answer-only "solve and answer" behavior; with
standard GRPO the extra dense signal is actively harmful out of distribution.

**Correct-set union:** 63/96 across all four variants.  `answer6` has 61
correct (2 unique), `drgrpo6` 59 (1 unique), base 57 (0 unique), `process6` 52
(0 unique).  An answer6+drgrpo6 oracle ensemble would reach 63/96 (65.6%),
only +2 over answer6 alone; the complementarity is small, and answer-only is
the best single transfer variant.

## 2. Extended algorithm wave (32-prompt greedy screen)

> The first n=8 launch hit verl's
> `assert train_batch_size >= ppo_mini_batch_size` check (n=8 uses
> `train_batch_size=4`).  The config now sets `ppo_mini_batch_size=4` and the
> run was relaunched; RLOO and DAPO-lite were trained separately on two free
> devices.

| variant | reward | advantage | accuracy | coverage | loop | mean reward |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| rloo | process | RLOO (n=4) | 0.656 | **0.750** | 0.028 | 1.028 |
| dapo | process + overlong | GRPO (DAPO-lite) | 0.656 | 0.734 | 0.063 | 1.014 |
| decomposed | process + token | Decomposed-GRPO | 0.625 | 0.703 | 0.091 | 0.974 |
| difficulty | process | Difficulty-GRPO | 0.656 | 0.734 | 0.060 | 1.021 |
| n8 | process | GRPO, n=8 | 0.656 | 0.719 | 0.091 | 1.012 |
| _reference_ process6 | process | GRPO (n=4) | 0.562 | 0.672 | 0.094 | 0.895 |
| _reference_ answer6 | answer | GRPO (n=4) | 0.656 | 0.719 | 0.083 | 0.656 |
| _reference_ drgrpo6 | process | Dr.GRPO (n=4) | **0.750** | **0.828** | 0.060 | 1.160 |

The screen is a single greedy sample over 32 prompts, so one prompt = 3.1
percentage points: 0.625 and 0.656 differ by one prompt, and only drgrpo6's
0.750 (24/32) is cleanly separated from the rest.  **Every extended variant
lands at 0.625–0.656 — at or above answer-only but none reproduces the
Dr.GRPO gain.**  RLOO has the highest coverage (0.750) and the lowest loop
ratio (0.028) of the wave; Decomposed-GRPO and n=8 have the highest loop ratio
(0.091, ~3/32 samples with a digit loop); Difficulty-GRPO halves invalid steps
relative to n=8 (0.12 vs 0.22).  Compared with the earlier core screens,
Dr.GRPO keeps its lead (0.750 / coverage 0.828), and plain GRPO+process remains
the weakest process variant (0.562).

Paired against drgrpo6 on the same 32 prompts (`stats.py --metric pass@1`):

| comparison | drgrpo6 | variant | diff | 95% CI | McNemar |
| --- | ---: | ---: | ---: | --- | --- |
| drgrpo6 → rloo | 0.750 | 0.656 | −0.094 | [−0.219, 0.000] | p=0.250 (0+ / 3−) |
| drgrpo6 → dapo | 0.750 | 0.656 | −0.094 | [−0.219, +0.031] | p=0.375 (1+ / 4−) |
| drgrpo6 → decomposed | 0.750 | 0.625 | −0.125 | **[−0.250, −0.031]** | p=0.125 (0+ / 4−) |
| drgrpo6 → difficulty | 0.750 | 0.656 | −0.094 | [−0.219, 0.000] | p=0.250 (0+ / 3−) |
| drgrpo6 → n8 | 0.750 | 0.656 | −0.094 | [−0.219, 0.000] | p=0.250 (0+ / 3−) |

Only Decomposed-GRPO's paired bootstrap CI excludes zero, and there the exact
McNemar has just four discordant pairs (p=0.125); with five uncorrected
comparisons these are exploratory.  The honest reading is that the extended
variants are *not significantly worse* than Dr.GRPO except possibly
Decomposed-GRPO, and none is better — the effect that separated Dr.GRPO from
answer-only (12 improvements / 3 regressions on 64×4) is not reproduced.

Training-rollout dynamics (48 groups × 4 = 192 samples; n=8 is 24 groups × 8):

| variant | mean acc | pass@1 | pass@k | zero-var(acc) | zero-var(reward) | coverage | loop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rloo | 0.516 | 0.583 | 0.729 | 0.542 | **0.333** | 0.609 | 0.031 |
| dapo | 0.469 | 0.542 | 0.688 | 0.583 | **0.333** | 0.586 | 0.019 |
| decomposed | 0.448 | 0.458 | **0.771** | 0.479 | **0.333** | 0.594 | 0.017 |
| difficulty | 0.479 | 0.417 | 0.708 | 0.583 | 0.375 | 0.591 | **0.013** |
| n8 | **0.552** | **0.625** | **0.875** | **0.458** | **0.333** | 0.625 | 0.024 |
| _reference_ process6 | 0.448 | 0.458 | 0.708 | 0.542 | 0.292 | 0.576 | 0.035 |
| _reference_ answer6 | 0.490 | 0.417 | 0.792 | 0.458 | 0.458 | 0.594 | 0.002 |
| _reference_ drgrpo6 | 0.490 | 0.542 | 0.750 | 0.562 | 0.313 | 0.620 | 0.024 |

All five extended variants hold the reward zero-variance rate at 0.33–0.38,
versus 0.46 for answer-only — the dense PCPR signal keeps within-group reward
variance alive even when every final answer in a group is wrong, and this is
independent of the advantage estimator.  Three secondary observations:

* **n=8 over n=4.**  Same 32 completions per optimizer step, but 4 prompts ×
  8 samples instead of 8 × 4.  It has the best training-rollout pass@k (0.875
  at k=8) and mean accuracy (0.552), and its all-wrong group rate falls to
  0.125 (vs 0.21–0.31 for the n=4 variants) purely from more samples per
  prompt.  Its greedy screen accuracy (0.656) still does not beat Dr.GRPO.
* **Decomposed-GRPO** separates the final-answer channel from the process
  channel and normalizes each before combining.  It has the best training
  pass@k among the extended n=4 variants (0.771, though still below answer-only
  at 0.792) but the *lowest* greedy screen accuracy (0.625) and the highest
  loop ratio — at six steps the decomposition adds variance without adding
  signal.
* **Difficulty-GRPO** weights Dr.GRPO advantages by group reward std.  It
  does not move the screen accuracy (0.656 vs 0.562 for plain process-GRPO)
  but gives the lowest invalid-step rate and lowest loop ratio of the wave.

A secondary observation concerns length drift.  Mean output characters from
step 1 to step 6 grow by +97 (RLOO), +95 (DAPO-lite), +95 (Difficulty-GRPO),
+68 (Decomposed-GRPO), and **+27 (n=8)**, versus +96 for Dr.GRPO and +79 for
plain process-GRPO at n=4.  The n=8 run is the only variant that keeps output
length roughly stable.  With four prompts × eight samples, each prompt's
correct and incorrect modes are both represented in every group, so the
policy receives a direct within-group comparison that does not depend on
length.  This is consistent with Dr.GRPO's length-bias argument but suggests
that group size, not just std normalization, is a lever on drift; a single
six-step run is only suggestive.

The drift is **not** a verbosity-reward artifact.  Across all 192 training
rollouts of every variant, response length correlates *negatively* with reward
(Pearson r = −0.10 RLOO, −0.12 Dr.GRPO, −0.12 answer6, −0.16 Decomposed, −0.19
Difficulty, −0.22 process6, −0.30 n=8) and with correctness.  Longer traces are
systematically worse, yet outputs still lengthen over training.  The growth is
therefore a policy-distribution drift rather than optimization toward longer
answers — and n=8, the variant with the smallest drift, also has the most
negative length–reward correlation, consistent with its optimization being
best aligned with shorter correct traces.

The honest summary of Section 2 is a **negative result with one positive
robustness finding**: none of RLOO, DAPO-lite, Decomposed-GRPO,
Difficulty-GRPO, or a larger group size beats Dr.GRPO+process on the
32-prompt greedy screen at this budget.  The reproducible positive is that the
PCPR process reward reduces reward-level zero variance by ~27–38% under every
advantage estimator tested, including RLOO and DAPO-style clipping.

## 3. Calibration: does the dense signal predict the outcome?

A dense process reward is only useful if it agrees with final correctness.  We
compute the AUC of `process_coverage` (fraction of gold intermediate values the
model proved) as a predictor of final-answer correctness, on the greedy dumps
(`verl_impl/process_auc.py`).

### 96 held-out word problems (OOD)

| variant | n | correct | coverage \| correct | coverage \| wrong | AUC | 95% CI (bootstrap) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| base | 96 | 57 | 0.947 | 0.115 | 0.958 | [0.910, 0.995] |
| answer6 | 96 | 61 | 0.975 | 0.129 | 0.947 | [0.891, 0.991] |
| drgrpo6 | 96 | 59 | 0.966 | 0.162 | 0.933 | [0.875, 0.983] |
| process6 | 96 | 52 | 0.952 | 0.159 | 0.947 | [0.898, 0.986] |

### 64-prompt × 4-sample stochastic validation (256 samples)

| variant | n | correct | coverage \| correct | coverage \| wrong | AUC | 95% CI (bootstrap) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| base | 256 | 134 | 0.970 | 0.332 | 0.942 | [0.909, 0.971] |
| answer6 | 256 | 148 | 0.956 | 0.338 | 0.922 | [0.885, 0.956] |
| process6 | 256 | 140 | 0.954 | 0.371 | 0.910 | [0.871, 0.946] |
| dense6 | 256 | 148 | 0.959 | 0.343 | 0.937 | [0.902, 0.967] |
| drgrpo6 | 256 | 169 | 0.973 | 0.391 | 0.922 | [0.884, 0.958] |
| loop6 | 256 | 152 | 0.961 | 0.385 | 0.930 | [0.896, 0.963] |

Under stochastic decoding the AUC is lower than under greedy decoding (0.91–0.94
vs 0.93–0.96) because wrong samples now include many partial traces with
nonzero coverage (mean 0.33–0.39).  The signal is still a strong predictor.

### 32-prompt symbolic validation screen

AUCs across all thirteen screened checkpoints: 0.914–0.963 (base 0.918;
drgrpo6 0.961; rloo 0.959; dapo 0.961; decomposed 0.960; difficulty 0.961;
n8 0.963).  Coverage for correct samples is 0.88–0.96, for incorrect samples
0.20–0.44.

**Interpretation.**  The proof-carrying coverage is a strong, stable predictor
of the outcome on both prompt distributions and for every trained variant.  On
the OOD word problems a correct answer comes with ~0.95 coverage and a wrong
one with ~0.13 — the verifier is measuring genuine arithmetic progress, not
format artefacts.  This is the property that makes PCPR a usable dense reward:
it is not independent noise; it is aligned with the outcome it densifies.  The
AUC is highest for `n8` and `dapo`/`difficulty`, but the differences between
variants are small and not tested.

The caveat runs the other way from the accuracy result: PCPR is well-calibrated
OOD even though training on it *hurts* OOD accuracy under standard GRPO (§1).
A dense reward can be a good predictor and still be a bad learning signal — the
advantage estimator decides how the extra variance is used.  That separation is
the central lesson of this report.

One caveat on the AUC: most wrong samples are degenerate/looping outputs with
exactly zero coverage, which inflates the separation.  It is not purely a
"garbage vs text" detector, though: 17–25% of wrong OOD samples (7/39 base,
6/35 answer6, 8/37 drgrpo6, 11/44 process6) do contain at least one verified
step, and their mean coverage is 0.64–0.75.  These are genuine "right steps,
wrong final answer" cases, and they are exactly the samples a dense reward is
supposed to help — at the cost of also rewarding a model that stops before the
final arithmetic.

## 4. Second-seed replication (answer vs Dr.GRPO)

`run_seed_b.sh` retrains answer-only and Dr.GRPO+process with a fresh RNG
stream (verl 0.5.0 has no global rollout seed, so a new run is an independent
seed in practice).  Six steps, same configs, same 32-prompt greedy screen.

### 4.1 Screen

| variant | seed | accuracy | coverage | loop |
| --- | --- | ---: | ---: | ---: |
| answer6 | A | 0.656 | 0.719 | 0.083 |
| drgrpo6 | A | **0.750** | 0.828 | 0.060 |
| answer6b | B | **0.719** | 0.781 | 0.060 |
| drgrpo6b | B | 0.688 | 0.781 | 0.060 |

The ordering **flips between seeds**: Dr.GRPO leads by three prompts in seed A
(24 vs 21) and trails by one in seed B (22 vs 23).  Paired within seed B:
drgrpo6b − answer6b = −0.031, 95% CI [−0.125, +0.063], McNemar p=1.000
(1+ / 2−).  Pooling the two seeds as 64 paired (prompt, seed) observations:

| comparison | answer | drgrpo | diff | 95% CI | McNemar |
| --- | ---: | ---: | ---: | --- | --- |
| seed A | 0.656 | 0.750 | +0.094 | [0.000, +0.219] | p=0.250 (3+ / 0−) |
| seed B | 0.719 | 0.688 | −0.031 | [−0.125, +0.063] | p=1.000 (1+ / 2−) |
| **pooled** | **0.688** | **0.719** | **+0.031** | **[−0.047, +0.109]** | **p=0.688 (4+ / 2−)** |

**The greedy-screen accuracy advantage does not replicate.**  The pooled
effect is +3.1 pp with a CI containing zero.  The seed-A screen difference was
always exploratory — its paired CI touched zero and the exact McNemar on three
discordant pairs was p=0.250 — and seed B confirms it is within run-to-run
noise.  The pre-registered endpoint was the 64×4 stochastic evaluation, whose
second-seed replication is reported in §4.3.

### 4.2 Training dynamics

| variant | mean acc | pass@1 | pass@k | zero-var(acc) | zero-var(reward) | coverage | loop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| answer6 (A) | 0.490 | 0.417 | 0.792 | 0.458 | 0.458 | 0.594 | 0.002 |
| drgrpo6 (A) | 0.490 | 0.542 | 0.750 | 0.562 | **0.313** | 0.620 | 0.024 |
| answer6b (B) | 0.479 | 0.500 | 0.729 | 0.562 | 0.562 | 0.583 | 0.032 |
| drgrpo6b (B) | 0.521 | 0.458 | 0.812 | 0.479 | **0.313** | 0.594 | 0.030 |

**The mechanism replicates exactly.**  Dr.GRPO+process reaches
`zero-var(reward) = 0.3125` in both seeds, against 0.458 (seed A) and 0.5625
(seed B) for answer-only — a 32% and 44% reduction.  The accuracy endpoints do
not replicate consistently: mean accuracy difference (drgrpo − answer) is
0.000 in seed A and +0.042 in seed B; training pass@k difference is −0.042 in
A and +0.083 in B.  Neither is significant.

### 4.3 Pre-registered 64×4 evaluation

Second-seed replication of the pre-registered endpoint (`run_seed_b_full.sh`):

| variant | pass@1 | pass@4 | mean acc |
| --- | ---: | ---: | ---: |
| _pending_ | _pending_ | _pending_ | _pending_ |


