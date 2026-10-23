# Tiny GRPO / RLVR arithmetic experiment

Small, reproducible interview project for post-training: train a Qwen3 policy with TRL GRPO and LoRA using only exact, programmatically checked integer answers. It deliberately targets a short single-GPU experiment, not a benchmark claim.

## Host snapshot and constraints

Checked 2026-10-23 as root: NVIDIA A100-SXM4-40GB, driver 595.84 (reports CUDA 13.2), 16 CPU cores / about 47 GiB RAM, 196 GiB filesystem with about 157 GiB free. The host Conda Python is 3.14.7 and has none of the ML packages installed. The default sandbox hides GPU devices; run training from the host shell or in an approved host-permission context. The user supplied A30 specs, but the live host reports A100; scripts use one visible GPU and do not hard-code a model-specific device.

## Data and verifier

`data/generate.py` deterministically produces 128 train and 32 validation three-operation integer expressions, plus 48 test word problems. Training/validation use symbolic expression templates; the test uses a held-out production-story template and fresh generated values. Validation is for diagnostic selection only; report final results on test. The final validation baseline found `pass@1=71.9%`, `pass@4=81.3%`, mean group reward standard deviation `0.115`, and zero-variance group rate `75.0%` on 32 questions. The 320-token training limit was chosen after the 256-token smoke showed 50–62.5% clipped completions. The synthetic domain is intentionally narrow and cannot establish general mathematical reasoning.

`src/verifier.py` awards 1 only when a strict final answer is extractable and numerically equals the gold value: one valid `<answer>...</answer>` tag, one explicit `\boxed{...}` final, a bare numeric final line, or a standalone arithmetic equality whose arithmetic is itself checked. Multiple/malformed answer tags, equations that do not balance, and explanation/keyword-only text score 0. Missing the requested `<answer>` tag is counted separately as a format error even when a bare/boxed numeric answer is correct. It does not execute model-generated code. Exact numeric equivalence does not assess reasoning correctness, and an answer-only reward can reward guessing.

## Setup

Use Python 3.12 in a project-local Conda environment (the host's existing base is Python 3.14). Install the pinned CUDA 12.8 PyTorch wheel first; the installed NVIDIA driver is backward compatible with that runtime. Remaining dependencies are pinned in `requirements.txt`, and the complete resolved environment is in `requirements-lock.txt`.

```bash
cd /root/llmposttraining
conda create -y -p ./env-grpo --override-channels -c conda-forge python=3.12 pip
conda activate ./env-grpo
python -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.7.1'
python -m pip install -r requirements.txt
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
python data/generate.py --seed 2026
python -m unittest discover -s tests -v
```

The pins are `torch 2.7.1`, `transformers 4.51.3`, `trl 0.18.1`, `peft 0.15.2`, `accelerate 1.6.0`, `datasets 3.6.0`, and `bitsandbytes 0.46.1`. This implementation uses LoRA (BF16 base weights), not 4-bit QLoRA: on a 40GB A100, this avoids quantization-kernel variables and leaves enough room for short GRPO generations. No full fine-tuning, multi-GPU, or vLLM.

## Baseline, smoke, and run

The requested identifier `Qwen/Qwen3-1.7B-Instruct` returns 401 / repository-not-found on the model hub. The official Qwen3 1.7B chat-capable checkpoint is `Qwen/Qwen3-1.7B`; its chat template supports disabling thinking mode. This project uses that official repo ID and pins revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. On the model page choose **Files and versions**, then download all original Transformers `safetensors` files (both shards, their index, config and tokenizer assets); the Deploy menu is for hosted inference and GGUF/GPTQ/FP8 artifacts are not compatible with this training path. To download to the host through the mirror and reuse the standard Hub cache, run `HF_ENDPOINT=https://hf-mirror.com env-grpo/bin/hf download Qwen/Qwen3-1.7B --revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. First check model load/generation and determine whether initial rewards are all-zero or all-one:

```bash
python src/evaluate.py --model Qwen/Qwen3-1.7B --data data/validation.jsonl --samples 4 --output runs/baseline_validation.jsonl
python src/evaluate.py --model Qwen/Qwen3-1.7B --data data/test.jsonl --samples 5 --output runs/baseline_test.jsonl
```

The short smoke uses 3 steps. On this A100 we use the requested 1.7B checkpoint, group size 4, 8 completions per optimizer batch (two prompt groups), and a 320-token cap; if peak memory is too high, drop to batch 4 or run the 0.6B fallback:

```bash
python src/train.py --smoke --model Qwen/Qwen3-1.7B --generations 4 --completion-length 320 --data data/train.jsonl --output runs/smoke
# low-memory fallback: python src/train.py --smoke --data data/train.jsonl --output runs/smoke_06b
```

If the smoke is healthy, train the requested model for 20 steps, group size 4, batch 8, max completion 320. Gradient checkpointing stays off to save recomputation on the 40GB A100; monitor the measured peak VRAM:

```bash
python src/train.py --data data/train.jsonl --output runs/grpo_seed17
python src/evaluate.py --model Qwen/Qwen3-1.7B --data data/test.jsonl --samples 5 --output runs/base_test.jsonl
python src/evaluate.py --model runs/grpo_seed17 --base-model Qwen/Qwen3-1.7B --data data/test.jsonl --samples 5 --output runs/grpo_test.jsonl
```

For the group-size ablation, change only group size from 4 to 2 while keeping the per-device completion batch at 8. GRPO then samples four prompts with two completions each instead of two prompts with four completions, holding completions per optimizer step constant:

```bash
python src/train.py --generations 2 --data data/train.jsonl --output runs/grpo_g2_seed17
```

All runs print seed, model, step count, generation count, completion limit, GPU and VRAM. Per-step logs are saved to `metrics.jsonl`: reward mean and group reward standard deviation, KL, completion length/clipping, group token entropy, reward-outcome entropy, and live/peak allocated GPU memory. `group_token_entropy_bits` is empirical token diversity across samples in the same prompt group, averaged by generation position; it is an exploration proxy, not the model's full next-token distribution entropy. Reward-outcome entropy is binary success/failure entropy and equals zero when rewards saturate. Use separate output directories and preserve the generated JSONL files as the experiment record. For stability claims, repeat with another seed; this first project run is a controlled demonstration, not a statistically powered comparison.

## Metrics and review

`src/evaluate.py` reports pass@1, pass@k, average generated tokens and format-error rate, and saves completions for manual inspection. With `--samples 5`, the first sample is stochastic as well; pass@1 is therefore sample 1, not greedy. For a deterministic greedy pass@1, run with `--samples 1`. Review at least 20 random test completions, every incorrect completion that received reward 1, and cases with long or malformed answers. Compare initial validation reward distribution before training: all 0 means no within-group advantage signal; all 1 means reward saturation. The verifier's unit tests are not evidence that model answers are correct.

Report a compact table with model, seed, steps, group size, completion limit, test pass@1/pass@5, mean tokens, format-error rate and peak VRAM. Analyze reward saturation, completion length drift, invalid formats, arithmetic slips, lucky guesses, and whether gains transfer from symbolic expressions to held-out word problems. Keep a sample of reward-positive completions for hand audit.

## Results and speed decision

The baseline, 20-step group-4 run, group-size-2 ablation, fixed test evaluations, and a post-change speed smoke have completed. Results, training diagnostics, manual verifier audit, and limits are in [RESULTS.md](RESULTS.md). On this 48-question test, pass@1 changed from 81.3% at baseline to 83.3% for group 4 and 81.3% for group 2; this is too small a sample to support a general performance claim. Pass@5 is near saturation, and answer-format errors remain about 100%.

The live host is an A100 with 40 GiB, while the originally supplied target was an A30 with 24 GiB. The measured training peak was about 17 GiB, so the current batch 8 configuration also fits the stated 24 GiB target with some margin. The entropy-diagnostic calculation is vectorized to reduce CPU/GPU synchronization without changing the training objective. A 3-step timing comparison was noisy, so the training batch and core GRPO settings were left unchanged. Increasing them may improve sample throughput on a 40 GiB card, but increases activation/KV memory and changes the experiment's effective batch; measure it in a separate run before adopting it on a 24 GiB card.

The completed outputs are stored under `runs/`; do not reuse those directories for new experiments. Use a new `--output` path and record any changed seed, group size, batch size, or generation limit.
