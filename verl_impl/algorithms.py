"""Custom advantage estimators for dense process rewards.

``process_grpo`` generalises GRPO's outcome advantage to token-level rewards:

1. optionally convert sparse step rewards into dense returns with a reverse
   cumulative sum (each token is credited for future verified steps);
2. normalise the returns *per group and per token position* instead of using
   a single scalar per response.

With a naive answer-only reward (one non-zero token per response) and
``process_return_mode=outcome`` this reduces exactly to GRPO.  The dense
variant is what allows partially-correct, all-wrong groups to produce a
gradient instead of collapsing to zero variance.

Registered through verl's ``ADV_ESTIMATOR_REGISTRY``; the demo harness imports
this module before constructing the trainer, so ``adv_estimator=process_grpo``
passes the trainer's whitelist.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from verl.trainer.ppo.core_algos import register_adv_est


def _reverse_cumsum(values: torch.Tensor) -> torch.Tensor:
    return torch.flip(torch.cumsum(torch.flip(values, dims=[-1]), dim=-1), dims=[-1])


def _groups(index: np.ndarray | None, batch_size: int) -> dict[object, list[int]]:
    if index is None:
        index = np.arange(batch_size)
    groups: dict[object, list[int]] = defaultdict(list)
    for i, uid in enumerate(index):
        groups[uid].append(i)
    return groups


def _group_normalize(
    values: torch.Tensor,
    positions: list[int],
    epsilon: float = 1e-6,
    norm_by_std: bool = True,
    dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-relative normalisation along ``dim`` for one prompt group."""
    idx = torch.tensor(positions, device=values.device, dtype=torch.long)
    group_values = values[idx]
    mean = group_values.mean(dim=dim, keepdim=True)
    if norm_by_std and len(positions) > 1:
        std = group_values.std(dim=dim, keepdim=True, unbiased=False)
    else:
        std = torch.zeros_like(mean)
    result = (group_values - mean) / (std + epsilon)
    return idx, result


@register_adv_est("process_grpo")
def compute_process_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray | None = None,
    epsilon: float = 1e-6,
    config=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-relative advantage on dense (or sparse) token-level rewards."""
    return_mode = "cumsum"
    norm_by_std = True
    if config is not None:
        return_mode = str(config.get("process_return_mode", "cumsum"))
        norm_by_std = bool(config.get("process_norm_by_std", True))

    if return_mode == "cumsum":
        returns = _reverse_cumsum(token_level_rewards)
    elif return_mode == "none":
        returns = token_level_rewards
    elif return_mode == "outcome":
        returns = token_level_rewards.sum(dim=-1, keepdim=True).expand_as(token_level_rewards)
    else:
        raise ValueError(f"Unknown process_return_mode: {return_mode!r}")

    returns = returns * response_mask

    if index is None:
        index = np.arange(returns.shape[0])

    groups: dict[object, list[int]] = defaultdict(list)
    for i, uid in enumerate(index):
        groups[uid].append(i)

    advantages = torch.zeros_like(returns)
    for positions in groups.values():
        idx = torch.tensor(positions, device=returns.device, dtype=torch.long)
        group_returns = returns[idx]
        mean = group_returns.mean(dim=0, keepdim=True)
        if norm_by_std and len(positions) > 1:
            std = group_returns.std(dim=0, keepdim=True, unbiased=False)
        else:
            std = torch.zeros_like(mean)
        advantages[idx] = (group_returns - mean) / (std + epsilon)

    advantages = advantages * response_mask
    return advantages, returns


@register_adv_est("process_rloo")
def compute_process_rloo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray | None = None,
    epsilon: float = 1e-6,
    config=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Leave-one-out baseline applied per token position (dense RLOO)."""
    return_mode = "cumsum"
    if config is not None:
        return_mode = str(config.get("process_return_mode", "cumsum"))

    if return_mode == "cumsum":
        returns = _reverse_cumsum(token_level_rewards)
    elif return_mode == "none":
        returns = token_level_rewards
    elif return_mode == "outcome":
        returns = token_level_rewards.sum(dim=-1, keepdim=True).expand_as(token_level_rewards)
    else:
        raise ValueError(f"Unknown process_return_mode: {return_mode!r}")
    returns = returns * response_mask

    if index is None:
        index = np.arange(returns.shape[0])

    groups: dict[object, list[int]] = defaultdict(list)
    for i, uid in enumerate(index):
        groups[uid].append(i)

    advantages = torch.zeros_like(returns)
    for positions in groups.values():
        if len(positions) < 2:
            continue
        idx = torch.tensor(positions, device=returns.device, dtype=torch.long)
        group_returns = returns[idx]
        # RLOO baseline: mean of the *other* samples at each position.
        total = group_returns.sum(dim=0, keepdim=True)
        baseline = (total - group_returns) / (len(positions) - 1)
        advantages[idx] = group_returns - baseline

    advantages = advantages * response_mask
    return advantages, returns


@register_adv_est("decomposed_grpo")
def compute_decomposed_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray | None = None,
    epsilon: float = 1e-6,
    config=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-channel GRPO: separate advantages for the final answer and process.

    The token-level reward manager writes the final-answer/format reward on the
    last valid token and verified-step rewards on earlier tokens.  This
    estimator normalises the two channels independently and combines them as

        A = A_final + lambda_process * A_process

    which prevents the (often larger) outcome term from swamping the dense
    process term and gives an explicit knob for the trade-off.  With a naive
    reward manager the process channel is empty and this reduces to GRPO.
    """
    process_weight = 0.5
    return_mode = "cumsum"
    final_norm_by_std = True
    process_norm_by_std = True
    if config is not None:
        process_weight = float(config.get("process_adv_weight", 0.5))
        return_mode = str(config.get("process_return_mode", "cumsum"))
        final_norm_by_std = bool(config.get("final_norm_by_std", True))
        process_norm_by_std = bool(config.get("process_norm_by_std", True))

    bsz, width = token_level_rewards.shape
    lengths = response_mask.sum(dim=-1).long().clamp_min(1)
    last_idx = (lengths - 1).clamp_max(width - 1)
    rows = torch.arange(bsz, device=token_level_rewards.device)

    final_scores = token_level_rewards[rows, last_idx]
    process_rewards = token_level_rewards.clone()
    process_rewards[rows, last_idx] = 0.0

    if return_mode == "cumsum":
        process_returns = _reverse_cumsum(process_rewards)
    elif return_mode == "none":
        process_returns = process_rewards
    elif return_mode == "outcome":
        process_returns = process_rewards.sum(dim=-1, keepdim=True).expand_as(process_rewards)
    else:
        raise ValueError(f"Unknown process_return_mode: {return_mode!r}")
    process_returns = process_returns * response_mask

    groups = _groups(index, bsz)
    final_adv = torch.zeros_like(final_scores)
    process_adv = torch.zeros_like(process_returns)
    for positions in groups.values():
        idx, normalised = _group_normalize(
            final_scores, positions, epsilon, final_norm_by_std, dim=0
        )
        final_adv[idx] = normalised
        idx, normalised = _group_normalize(
            process_returns, positions, epsilon, process_norm_by_std, dim=0
        )
        process_adv[idx] = normalised

    final_adv = final_adv.unsqueeze(-1) * response_mask
    process_adv = process_adv * response_mask
    advantages = final_adv + process_weight * process_adv
    returns = final_adv + process_returns
    return advantages, returns


@register_adv_est("difficulty_grpo")
def compute_difficulty_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray | None = None,
    epsilon: float = 1e-6,
    config=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Variance-weighted Dr.GRPO.

    Groups whose rewards are nearly saturated (all correct or all wrong) carry
    little learning signal.  We weight each group's advantage by its reward
    standard deviation raised to ``difficulty_alpha`` (alpha=0 recovers
    Dr.GRPO).  Unlike GRPO's division by std, this *downweights* saturated
    groups instead of amplifying tiny differences in them, and it avoids the
    length-bias mechanism criticised by Dr.GRPO.  Weights are normalised by
    their mean so the overall advantage scale stays comparable across runs.

    With binary rewards, std = sqrt(p(1-p)); alpha=1 is therefore a difficulty
    weight proportional to p(1-p), maximised at p=0.5.
    """
    alpha = 1.0
    norm_by_std = False
    if config is not None:
        alpha = float(config.get("difficulty_alpha", 1.0))
        norm_by_std = bool(config.get("difficulty_norm_by_std", False))

    scores = token_level_rewards.sum(dim=-1)
    groups = _groups(index, scores.shape[0])

    weights: dict[object, float] = {}
    for uid, positions in groups.items():
        idx = torch.tensor(positions, device=scores.device, dtype=torch.long)
        group_scores = scores[idx]
        std = (
            float(group_scores.std(unbiased=False).item())
            if len(positions) > 1
            else 0.0
        )
        weights[uid] = std**alpha
    mean_weight = sum(weights.values()) / max(1, len(weights))
    scale = mean_weight if mean_weight > epsilon else 1.0

    advantages = torch.zeros_like(scores)
    for uid, positions in groups.items():
        idx = torch.tensor(positions, device=scores.device, dtype=torch.long)
        group_scores = scores[idx]
        mean = group_scores.mean()
        if norm_by_std and len(positions) > 1:
            denominator = group_scores.std(unbiased=False) + epsilon
        else:
            denominator = 1.0
        advantages[idx] = (group_scores - mean) / denominator * (weights[uid] / scale)

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages
