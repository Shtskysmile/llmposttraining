"""Token-level process reward manager for verl 0.5.0.

The default ``naive`` manager puts the whole scalar reward on the final
response token.  For dense process supervision we instead place the *process*
part of the reward on the token where each verified arithmetic step completes,
and the final-answer/format part on the last response token.  The total reward
of every sample is unchanged, so outcome-based estimators remain valid while
``process_grpo`` can exploit finer credit assignment.

Registered as ``reward_model.reward_manager: arithmetic_process``.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from verl.workers.reward_manager import register  # noqa: E402
from verl.workers.reward_manager.naive import NaiveRewardManager  # noqa: E402

from verl_impl.verifier import char_ends_to_token_indices  # noqa: E402


@register("arithmetic_process")
class ArithmeticProcessRewardManager(NaiveRewardManager):
    """Naive manager + per-step token-level reward placement.

    ``reward_model.reward_kwargs`` may contain ``overlong_len``,
    ``max_resp_len`` and ``overlong_penalty_factor`` to enable DAPO-style
    overlong reward shaping.
    """

    def __init__(
        self,
        *args,
        overlong_len=None,
        max_resp_len=None,
        overlong_penalty_factor=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.overlong_len = overlong_len
        self.max_resp_len = max_resp_len
        self.overlong_penalty_factor = float(overlong_penalty_factor)

    def _overlong_penalty(self, response_length: int) -> float:
        if not self.overlong_len or response_length <= self.overlong_len:
            return 0.0
        if not self.max_resp_len or self.max_resp_len <= self.overlong_len:
            return -self.overlong_penalty_factor
        buffer = self.max_resp_len - self.overlong_len
        exceed = min(response_length - self.overlong_len, buffer)
        return -self.overlong_penalty_factor * exceed / buffer

    def __call__(self, data, return_dict=False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            if valid_response_length <= 0:
                continue
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            if extra_info is None:
                extra_info = {}
            extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__", None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                include_spans=True,
            )

            overlong_penalty = self._overlong_penalty(valid_response_length)

            if isinstance(score, dict):
                step_char_ends = score.pop("verified_char_ends", [])
                process_component = float(score.pop("process_component", 0.0))
                final_component = float(score.pop("final_component", score.get("score", 0.0)))
                final_component += overlong_penalty
                score["score"] = float(
                    score.get("score", final_component + process_component)
                ) + overlong_penalty
                score["overlong_penalty"] = overlong_penalty
                score["overlong"] = float(overlong_penalty < 0.0)
                for key, value in score.items():
                    if isinstance(value, (int, float, bool)) or hasattr(value, "item"):
                        try:
                            value = float(value)
                        except (TypeError, ValueError):
                            continue
                    else:
                        # Keep reward_extra_info numeric; it is averaged by the
                        # demo RewardRecorder and by verl's validation metrics.
                        continue
                    reward_extra_info[key].append(value)
                reward = float(score["score"])
            else:
                step_char_ends = []
                process_component = 0.0
                final_component = float(score) + overlong_penalty
                reward = final_component
                reward_extra_info["overlong_penalty"].append(overlong_penalty)
                reward_extra_info["overlong"].append(float(overlong_penalty < 0.0))

            last = valid_response_length - 1
            placed_process = 0.0
            if process_component > 0.0 and step_char_ends:
                token_indices = char_ends_to_token_indices(
                    self.tokenizer, valid_response_ids.tolist(), step_char_ends
                )
                token_indices = [idx for idx in token_indices if 0 <= idx < valid_response_length]
                if token_indices:
                    per_step = process_component / len(token_indices)
                    for idx in token_indices:
                        reward_tensor[i, idx] += per_step
                    placed_process = per_step * len(token_indices)
            reward_tensor[i, last] += final_component + (process_component - placed_process)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", reward)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
