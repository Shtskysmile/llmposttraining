# Experiment results

All evaluation uses the fixed 48-question held-out production-story test set, five stochastic samples per prompt, a 256-token cap, and seed 17. Pass@1 is the first sampled completion; pass@5 is whether any of the five completions is correct. Since test size is small, these are descriptive results, not evidence of a statistically reliable improvement.

## Test comparison

| Run | Steps | GRPO group | Pass@1 | Pass@5 | Mean generated tokens | Format error | Peak training VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-1.7B | — | — | 81.3% | 100.0% | 205.9 | 100.0% | — |
| LoRA GRPO | 20 | 4 | 83.3% | 97.9% | 207.6 | 99.6% | 17.00 GiB |
| LoRA GRPO ablation | 20 | 2 | 81.3% | 97.9% | 206.3 | 100.0% | 17.05 GiB |

The group-4 run changes pass@1 by +2.1 percentage points versus baseline, while group 2 is unchanged. Both trained runs have one fewer pass@5 success than baseline. With 48 questions and stochastic decoding, these small differences should not be interpreted as a robust gain. Pass@5 is saturated near 100% and is less informative here than pass@1.

## Expanded-data rerun

A second deterministic split was generated with seed 2028: 256 train, 64 validation, and 96 held-out word problems. All prompts are distinct across its splits; its train prompts also do not overlap with the original seed-2026 training set. Evaluation used five samples per test question, 256 max new tokens, and prompt batch size 4 for both baseline and post-GRPO models.

| Run | Train steps | Test size | Pass@1 | Pass@5 | Mean reward | Mean tokens | Format error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-1.7B | — | 96 | 82.3% | 94.8% | 0.842 | 203.6 | 100.0% |
| Expanded-data GRPO, group 4 | 40 | 96 | 79.2% | 93.8% | 0.781 | 204.3 | 100.0% |

On the 64-question validation set, pass@1 increased from 57.8% to 60.9%, and pass@4 from 75.0% to 78.1%. The larger held-out test did not confirm that small validation gain: test pass@1 fell from 79/96 to 76/96, and pass@5 from 91/96 to 90/96. Mean correct completions fell from 404/480 to 375/480. Answer-format error stayed at 100% and mean output length barely changed. The result does not support a claim that this scaled run improved the held-out word problems.

A deterministic random audit of 20 first-sample test responses found 17 exact final answers. The three first-sample failures included one arithmetic slip (`13043` gold, `12057` predicted; four of its other five-sample group were correct), one item where all five samples missed by a small arithmetic amount (`14549` gold, `14559` predicted), and one first answer cut off at the 256-token limit (two other samples for that question were correct). No incorrect final number among these audited samples earned verifier reward. The audited negative-result case illustrates the semantic limitation: the arithmetic result was negative and exact, while the model's prose described having no units left.

The expanded run used 256 new symbolic training prompts, group size 4, completion batch 8, 40 steps, and the same learning rate/LoRA settings as the original direct GRPO. It took 667.7 seconds and peaked at 17.05 GiB allocated (about 20.9 GiB reported as total GPU use). Mean train reward was 0.756, mean group reward std 0.293, zero-variance group rate 66.7%, mean completion length 230.4 tokens, cap rate 12.5%, and mean KL 0.0004. The larger data and doubled step count changed together; this is a scaled replicate, not a controlled ablation of data volume alone. The train/validation prompts use symbolic expressions while the test uses word problems, so the validation/test divergence is consistent with limited transfer or sampling noise, but this run cannot identify the cause.

The format error metric means the response did not contain exactly one valid `<answer>...</answer>` tag. The verifier still gives numeric reward to an unambiguous boxed or bare final answer, so this metric diagnoses instruction-format compliance separately. The model almost never adopted the requested tag format.

## Validation and training diagnostics

| Run | Validation pass@1 | Validation pass@4 | Mean train reward | Mean group reward std | Zero-variance groups | Mean completion tokens | At generation cap | Mean KL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | 71.9% | 81.3% | — | — | 75.0% | 200.3 | — | — |
| GRPO group 4 | — | — | 0.730 | 0.322 | 68.4% | 235.8 | 9.9% | 0.0002 |
| GRPO group 2 | — | — | 0.743 | 0.360 | 75.0% | 230.5 | 10.5% | 0.0003 |

Train means use the 20 optimizer steps, deduplicating the final repeated step-20 log record. The completion cap was not the dominant failure mode: roughly one in ten training completions hit 320 tokens, while test mean lengths stayed close to baseline. Group-2 and group-4 entropy proxies are not directly comparable because group size changes the estimator's sample bias.

Training took 348.2 seconds for group 4 and 348.8 seconds for group 2. Peak allocated memory was about 17 GiB on the host's A100-SXM4-40GB. The run used BF16 base weights, LoRA rank 16, one GPU, batch size 8, seed 17, and max completion 320. Group 4 sampled two prompts × four responses per optimizer step; group 2 sampled four prompts × two responses, holding total completions per step constant.

## Manual verifier audit

A deterministic random sample of the first completion for 20 test questions from the group-4 output was read manually. Seventeen received reward 1 and matched the expected final integer. Three received reward 0:

- One had an arithmetic error and a wrong final integer.
- One reached the 256-token evaluation limit before giving a final answer.
- One gave the correct numeric answer twice in separate `\boxed{}` expressions. The strict extractor rejected the repeated boxed answer as ambiguous.

No incorrect final numeric answer among these 20 samples received reward 1. This check does not validate every intermediate reasoning step. The verifier can reward a correct final value even if the explanation contains an arithmetic mistake, and it cannot tell whether the answer was guessed. Thus the current reward is a final-answer signal, not a proof-of-reasoning signal.

## Throughput note

The diagnostic group-token-entropy calculation was changed from a token-position Python loop with repeated device synchronization to a vectorized tensor calculation. It matched the old calculation for synthetic group sizes 2, 4, and 8. A separate 3-step 1.7B smoke run took 47.9 seconds and peaked at 17.00 GiB; an earlier same-shape smoke took 50.4 seconds. This roughly 5% difference is too noisy and based on too few steps to claim a stable speedup. The change affects diagnostics only, not prompts, rewards, or gradients. More memory alone is not a speed target: autoregressive generation dominated wall time, while the existing run already fits a 24 GiB GPU with some margin.

## SFT and SFT -> GRPO follow-up

The SFT follow-up used the same 128 original symbolic training prompts and one LoRA epoch (batch 4, learning rate `1e-4`, seed 17). The first target contained only `<answer>integer</answer>`. It learned the format but failed on arithmetic: on the fixed test it scored pass@1 = 0%, pass@5 = 0%, and format-error rate = 0%. This is a negative result and shows that answer-only SFT taught the output shape without transferring calculation ability.

A second SFT target contained deterministic intermediate arithmetic steps generated from each expression's AST, followed by the tagged answer. Every generated trace was checked against the gold answer before training. On the 32-question validation set it scored pass@1 = 68.8%, pass@4 = 71.9%, mean reward = 0.641, mean generated length = 69.4 tokens, and format-error rate = 0%. This retained most single-sample accuracy but did not beat the base validation result (71.9% / 81.3%).

To test whether GRPO could refine that SFT policy, a fresh 128-row symbolic training set with seed 2027 was generated; its prompts have no overlap with the original SFT train prompts. The GRPO run started from the trace-SFT adapter, used group 4, batch 8, 20 steps, a 320-token cap, and seed 17. Its KL reference was a frozen copy of the initial SFT adapter, so KL measures change from the SFT policy rather than the original base. Training took 98.8 seconds and peaked at 11.4 GiB. Across steps, mean reward was 0.513, mean group reward standard deviation was 0.370, 78.9% of logged groups had zero reward variance, mean completion length was 73.7 tokens, and mean KL was 0.0006. On validation, SFT -> GRPO scored pass@1 = 68.8%, pass@4 = 68.8%, mean reward = 0.609, mean generated length = 69.5 tokens, and format-error rate = 0%. GRPO kept the format but did not improve accuracy; pass@4 fell slightly from the SFT checkpoint. Since validation did not improve, this follow-up was not evaluated on the fixed test set.

The pass@4 drop is one question out of 32 (23 solved by SFT versus 22 after GRPO); pass@1 stayed 22/32 for both. The changed outcome was the validation expression `(669 + 809) * 18 - 8930 = 17674`: SFT had one correct answer among samples 2–4, while all four post-GRPO samples missed it. Across all 128 sampled completions, correct answers decreased from 82 to 78, with the first-sample total unchanged. Several other questions still had at least one correct sample but acquired one or two wrong alternatives. This is consistent with a small change in the stochastic candidate distribution and sparse, noisy policy updates; it is not evidence that pass@1 regressed.

This demonstrates that SFT -> GRPO is mechanically viable and gets nonzero exact-match rewards on fresh training prompts, but this short run did not produce a validation gain. Most sampled groups still had zero reward variance, so reward sparsity remains the limiting factor. A longer or more complex pipeline is not justified by this result alone.

## Artifacts

- Base test completions: `runs/base_test.jsonl`
- Group-4 test completions: `runs/grpo_g4_test.jsonl`
- Group-2 test completions: `runs/grpo_g2_test.jsonl`
- Group-4 step metrics and run configuration: `runs/grpo_g4_seed17/metrics.jsonl`, `runs/grpo_g4_seed17/run_config.json`
- Group-2 step metrics and run configuration: `runs/grpo_g2_seed17/metrics.jsonl`, `runs/grpo_g2_seed17/run_config.json`
- Vectorized-metric smoke: `runs/speed_smoke_vec/metrics.jsonl`, `runs/speed_smoke_vec/run_config.json`
- Answer-only SFT adapter and test output: `runs/sft_format_seed17/`, `runs/sft_test.jsonl`
- Verified-trace SFT adapter and validation output: `runs/sft_trace_seed17/`, `runs/sft_trace_validation.jsonl`
- SFT -> GRPO adapter, logs and validation output: `runs/sft_grpo_trace_g4_seed17/`, `runs/sft_grpo_trace_g4_seed17/metrics.jsonl`, `runs/sft_grpo_trace_validation.jsonl`
- Fresh SFT -> GRPO training split: `data/seed2027/train.jsonl` (generated with seed 2027; no prompt overlap with `data/train.jsonl`)
- Expanded split and scaled GRPO run: `data/expanded_seed2028/{train,validation,test}.jsonl`, `runs/expanded_grpo_g4_seed17/`
- Expanded evaluation outputs: `runs/expanded_base_validation.jsonl`, `runs/expanded_base_test.jsonl`, `runs/expanded_grpo_validation.jsonl`, `runs/expanded_grpo_test.jsonl`

