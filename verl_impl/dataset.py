"""RLHFDataset with Qwen3 thinking mode disabled.

verl 0.5.0's ``RLHFDataset`` calls ``apply_chat_template`` without the
Qwen3-specific ``enable_thinking`` flag.  On the base Qwen3 checkpoints this
sometimes lets the model fall into an unterminated ``<think>`` loop, which
wastes the rollout budget before any arithmetic is produced.

Overriding ``__getitem__`` is the supported extension point
(``data.custom_cls.path/name``); the text-only path mirrors verl 0.5.0.
"""
from __future__ import annotations

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


class ArithmeticDataset(RLHFDataset):
    """Text-only RLHFDataset that renders prompts with thinking disabled."""

    def __getitem__(self, item):
        row_dict = self.dataframe[item]
        messages = self._build_messages(row_dict)

        raw_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs.pop("input_ids")
        attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            else:
                raise RuntimeError(
                    f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}."
                )
        row_dict["raw_prompt_ids"] = raw_prompt_ids

        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt

        extra_info = row_dict.get("extra_info", {})
        row_dict["index"] = extra_info.get("index", 0)
        row_dict["tools_kwargs"] = extra_info.get("tools_kwargs", {})
        row_dict["interaction_kwargs"] = extra_info.get("interaction_kwargs", {})
        return row_dict
