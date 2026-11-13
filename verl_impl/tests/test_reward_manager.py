"""Integration test: token-level process reward placement in verl DataProto."""
from __future__ import annotations

import sys
import unittest
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL = Path("/root/work/models/Qwen3-0.6B")
EXPR = "(221 + 427) * 28 - 8485"
GOLD = "9659"
TRACE = "221 + 427 = 648\n648 * 28 = 18144\n18144 - 8485 = 9659\n<answer>9659</answer>"


@unittest.skipUnless(MODEL.exists(), "local Qwen3-0.6B checkpoint is required")
class TestTokenLevelRewardManager(unittest.TestCase):
    def test_reward_is_placed_on_verified_steps(self):
        import torch
        from transformers import AutoTokenizer
        from verl import DataProto

        from verl_impl.reward_manager import ArithmeticProcessRewardManager
        from verl_impl.rewards import compute_score

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
        prompt_text = (
            f"Compute exactly: {EXPR}. Return the final integer in <answer>...</answer>."
        )
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(TRACE, add_special_tokens=False)["input_ids"]

        prompts = torch.tensor([prompt_ids], dtype=torch.long)
        responses = torch.tensor([response_ids], dtype=torch.long)
        attention = torch.ones(1, len(prompt_ids) + len(response_ids), dtype=torch.long)
        data = DataProto.from_dict(
            tensors={"prompts": prompts, "responses": responses, "attention_mask": attention},
            non_tensors={
                "reward_model": np.array([{"ground_truth": GOLD}], dtype=object),
                "data_source": np.array(["arithmetic"], dtype=object),
                "extra_info": np.array(
                    [
                        {
                            "expression": EXPR,
                            "prompt": prompt_text,
                        }
                    ],
                    dtype=object,
                ),
            },
        )

        manager = ArithmeticProcessRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=partial(compute_score, mode="process"),
        )
        result = manager(data, return_dict=True)
        reward = result["reward_tensor"][0]
        # Total reward is final (1.1) + process (0.5).
        self.assertAlmostEqual(float(reward.sum()), 1.6, places=5)
        nonzero = torch.nonzero(reward, as_tuple=False).flatten().tolist()
        # Verified step tokens + the final response token.
        self.assertGreaterEqual(len(nonzero), 3)
        self.assertAlmostEqual(float(reward[reward.shape[0] - 1]), 1.1, places=5)
        self.assertAlmostEqual(float(result["reward_extra_info"]["acc"][0]), 1.0)
        self.assertAlmostEqual(float(result["reward_extra_info"]["process_coverage"][0]), 1.0)

    def test_overlong_penalty_is_applied_to_the_tensor(self):
        import torch
        from transformers import AutoTokenizer
        from verl import DataProto

        from verl_impl.reward_manager import ArithmeticProcessRewardManager
        from verl_impl.rewards import compute_score

        tokenizer = AutoTokenizer.from_pretrained(str(MODEL))
        prompt_text = f"Compute exactly: {EXPR}. Return the final integer in <answer>...</answer>."
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(TRACE, add_special_tokens=False)["input_ids"]
        data = DataProto.from_dict(
            tensors={
                "prompts": torch.tensor([prompt_ids], dtype=torch.long),
                "responses": torch.tensor([response_ids], dtype=torch.long),
                "attention_mask": torch.ones(
                    1, len(prompt_ids) + len(response_ids), dtype=torch.long
                ),
            },
            non_tensors={
                "reward_model": np.array([{"ground_truth": GOLD}], dtype=object),
                "data_source": np.array(["arithmetic"], dtype=object),
                "extra_info": np.array([{"expression": EXPR, "prompt": prompt_text}], dtype=object),
            },
        )
        manager = ArithmeticProcessRewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=partial(compute_score, mode="process"),
            overlong_len=8,
            max_resp_len=16,
            overlong_penalty_factor=0.1,
        )
        result = manager(data, return_dict=True)
        # 1.6 base reward minus the capped 0.1 overlong penalty.
        self.assertAlmostEqual(float(result["reward_tensor"].sum()), 1.5, places=5)
        self.assertAlmostEqual(
            float(result["reward_extra_info"]["overlong_penalty"][0]), -0.1, places=5
        )

    def test_overlong_penalty_schedule(self):
        from verl_impl.reward_manager import ArithmeticProcessRewardManager

        manager = ArithmeticProcessRewardManager.__new__(ArithmeticProcessRewardManager)
        manager.overlong_len = 288
        manager.max_resp_len = 320
        manager.overlong_penalty_factor = 0.1
        self.assertEqual(manager._overlong_penalty(288), 0.0)
        self.assertAlmostEqual(manager._overlong_penalty(304), -0.05)
        self.assertAlmostEqual(manager._overlong_penalty(320), -0.1)
        self.assertAlmostEqual(manager._overlong_penalty(400), -0.1)


if __name__ == "__main__":
    unittest.main()
