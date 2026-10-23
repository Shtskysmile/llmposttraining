# Experiment results

All evaluation uses the fixed 48-question held-out production-story test set, five stochastic samples per prompt, a 256-token cap, and seed 17. Pass@1 is the first sampled completion; pass@5 is whether any of the five completions is correct. Since test size is small, these are descriptive results, not evidence of a statistically reliable improvement.

## Test comparison

| Run | Steps | GRPO group | Pass@1 | Pass@5 | Mean generated tokens | Format error | Peak training VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-1.7B | — | — | 81.3% | 100.0% | 205.9 | 100.0% | — |
| LoRA GRPO | 20 | 4 | 83.3% | 97.9% | 207.6 | 99.6% | 17.00 GiB |
| LoRA GRPO ablation | 20 | 2 | 81.3% | 97.9% | 206.3 | 100.0% | 17.05 GiB |

The group-4 run changes pass@1 by +2.1 percentage points versus baseline, while group 2 is unchanged. Both trained runs have one fewer pass@5 success than baseline. With 48 questions and stochastic decoding, these small differences should not be interpreted as a robust gain. Pass@5 is saturated near 100% and is less informative here than pass@1.

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
- One gave the correct numeric answer twice in separate `\\boxed{}` expressions. The strict extractor rejected the repeated boxed answer as ambiguous.

No incorrect final numeric answer among these 20 samples received reward 1. This check does not validate every intermediate reasoning step. The verifier can reward a correct final value even if the explanation contains an arithmetic mistake, and it cannot tell whether the answer was guessed. Thus the current reward is a final-answer signal, not a proof-of-reasoning signal.

## Throughput note

The diagnostic group-token-entropy calculation was changed from a token-position Python loop with repeated device synchronization to a vectorized tensor calculation. It matched the old calculation for synthetic group sizes 2, 4, and 8. A separate 3-step 1.7B smoke run took 47.9 seconds and peaked at 17.00 GiB; an earlier same-shape smoke took 50.4 seconds. This roughly 5% difference is too noisy and based on too few steps to claim a stable speedup. The change affects diagnostics only, not prompts, rewards, or gradients. More memory alone is not a speed target: autoregressive generation dominated wall time, while the existing run already fits a 24 GiB GPU with some margin.

## Artifacts

- Base test completions: `runs/base_test.jsonl`
- Group-4 test completions: `runs/grpo_g4_test.jsonl`
- Group-2 test completions: `runs/grpo_g2_test.jsonl`
- Group-4 step metrics and run configuration: `runs/grpo_g4_seed17/metrics.jsonl`, `runs/grpo_g4_seed17/run_config.json`
- Group-2 step metrics and run configuration: `runs/grpo_g2_seed17/metrics.jsonl`, `runs/grpo_g2_seed17/run_config.json`
- Vectorized-metric smoke: `runs/speed_smoke_vec/metrics.jsonl`, `runs/speed_smoke_vec/run_config.json`

