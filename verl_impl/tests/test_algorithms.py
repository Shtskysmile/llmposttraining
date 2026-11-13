"""Invariant tests for the custom advantage estimators."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verl_impl.algorithms import (  # noqa: E402
    compute_decomposed_grpo_advantage,
    compute_difficulty_grpo_advantage,
    compute_process_grpo_advantage,
)


def make_batch(rewards: list[float], group_size: int = 4, width: int = 6):
    """Scalar outcome rewards placed on the last token of each response."""
    batch = torch.zeros(len(rewards), width)
    for i, reward in enumerate(rewards):
        batch[i, width - 1] = reward
    mask = torch.ones(len(rewards), width)
    index = np.repeat(np.arange(len(rewards) // group_size), group_size)
    return batch, mask, index


class TestProcessGRPO(unittest.TestCase):
    def test_outcome_mode_matches_grpo(self):
        rewards = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        batch, mask, index = make_batch(rewards)
        adv, _ = compute_process_grpo_advantage(
            batch, mask, index=index, config={"process_return_mode": "outcome"}
        )
        # Saturated group -> zero; mixed group of two 0s and two 1s -> +-1.
        expected_first = torch.zeros(4)
        expected_second = torch.tensor([-1.0, 1.0, -1.0, 1.0])
        self.assertTrue(torch.allclose(adv[:4, -1], expected_first, atol=1e-5))
        self.assertTrue(torch.allclose(adv[4:, -1], expected_second, atol=1e-5))
        # Outcome mode broadcasts the same advantage to every response token.
        self.assertTrue(torch.allclose(adv[:4], torch.zeros(4, 6), atol=1e-6))
        self.assertTrue(
            torch.allclose(adv[4:], expected_second.unsqueeze(-1).expand(4, 6), atol=1e-5)
        )

    def test_dense_cumsum_gives_earlier_tokens_credit(self):
        rewards = torch.zeros(4, 6)
        rewards[:, 2] = torch.tensor([0.1, 0.3, 0.0, 0.2])
        rewards[:, 5] = torch.tensor([1.0, 1.0, 0.0, 0.0])
        mask = torch.ones(4, 6)
        adv, returns = compute_process_grpo_advantage(
            rewards, mask, index=np.zeros(4, dtype=int),
            config={"process_return_mode": "cumsum"},
        )
        # Position 2 returns include the later outcome reward, so the dense
        # advantage at position 2 is nonzero even though only the last token
        # carries the outcome.
        self.assertGreater(float(adv[:, 2].abs().sum()), 0.0)
        self.assertTrue(torch.allclose(adv * (1 - mask), torch.zeros_like(adv)))


class TestDecomposedGRPO(unittest.TestCase):
    def test_without_process_channel_reduces_to_grpo(self):
        rewards = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        batch, mask, index = make_batch(rewards)
        adv, _ = compute_decomposed_grpo_advantage(
            batch, mask, index=index, config={"process_adv_weight": 0.5}
        )
        self.assertTrue(torch.allclose(adv[:4], torch.zeros(4, 6), atol=1e-5))
        self.assertTrue(
            torch.allclose(adv[4:, -1], torch.tensor([-1.0, 1.0, -1.0, 1.0]), atol=1e-5)
        )

    def test_process_channel_is_active(self):
        batch = torch.zeros(4, 6)
        # verified step at token 2, final answer at token 5
        batch[:, 2] = torch.tensor([0.5, 0.0, 0.5, 0.0])
        batch[:, 5] = torch.tensor([1.0, 1.0, 0.0, 0.0])
        mask = torch.ones(4, 6)
        adv, _ = compute_decomposed_grpo_advantage(
            batch, mask, index=np.zeros(4, dtype=int),
            config={"process_adv_weight": 0.5, "process_return_mode": "none"},
        )
        # With no reverse cumsum, token 2 carries process advantage.
        self.assertGreater(float(adv[:, 2].abs().sum()), 0.0)
        # Token 5 carries only the final-answer advantage: [1, 1, 0, 0] is
        # normalised to [1, 1, -1, -1] within the group.
        self.assertTrue(
            torch.allclose(adv[:, 5], torch.tensor([1.0, 1.0, -1.0, -1.0]), atol=1e-5)
        )


class TestDifficultyGRPO(unittest.TestCase):
    def test_alpha_zero_matches_drgrpo(self):
        rewards = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        batch, mask, index = make_batch(rewards)
        adv, _ = compute_difficulty_grpo_advantage(
            batch, mask, index=index,
            config={"difficulty_alpha": 0.0, "difficulty_norm_by_std": False},
        )
        # Group 0 mean 0.75 -> [0.25, 0.25, -0.75, 0.25] relative differences.
        expected = torch.tensor([0.25, 0.25, -0.75, 0.25])
        self.assertTrue(torch.allclose(adv[:4, -1], expected, atol=1e-5))

    def test_saturated_groups_are_downweighted(self):
        rewards = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        batch, mask, index = make_batch(rewards)
        adv, _ = compute_difficulty_grpo_advantage(
            batch, mask, index=index, config={"difficulty_alpha": 1.0}
        )
        self.assertTrue(torch.allclose(adv[:4], torch.zeros(4, 6), atol=1e-6))
        self.assertGreater(float(adv[4:, -1].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
