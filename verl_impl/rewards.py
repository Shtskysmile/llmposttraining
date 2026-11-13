"""verl custom reward function for the arithmetic RLVR task.

Configured through ``custom_reward_function`` in the Hydra config:

    custom_reward_function:
      path: verl_impl/rewards.py
      name: compute_score
      reward_kwargs:
        mode: process

``mode`` selects the training reward:

* ``answer``  – exact final answer only (answer-only RLVR baseline)
* ``format``  – exact final answer + a small bonus for the requested tag
* ``process`` – exact final answer + proof-carrying process coverage + format

All modes still return the same diagnostic fields (``acc``, ``format_ok``,
``process_coverage`` ...) so every run is evaluated with identical metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from verl_impl.verifier import (  # noqa: E402
    answer_score,
    expression_from_prompt,
    has_answer_tag,
    loop_repetition_ratio,
    process_reward,
    verify_process,
)


def _extra(extra_info) -> dict:
    if isinstance(extra_info, dict):
        return extra_info
    return {}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    mode: str = "process",
    final_weight: float = 1.0,
    process_weight: float = 0.5,
    format_weight: float = 0.1,
    invalid_penalty: float = 0.05,
    loop_penalty_weight: float = 0.0,
    include_spans: bool = False,
    **kwargs,
) -> dict:
    """Return a reward dict consumed by verl's reward manager."""
    info = _extra(extra_info)
    prompt = info.get("prompt")
    expression = info.get("expression")
    if not expression and prompt:
        expression = expression_from_prompt(prompt)

    if mode == "process":
        result = process_reward(
            prompt,
            solution_str,
            ground_truth,
            expression=expression,
            final_weight=final_weight,
            process_weight=process_weight,
            format_weight=format_weight,
            invalid_penalty=invalid_penalty,
            loop_penalty_weight=loop_penalty_weight,
        )
    else:
        final_correct = answer_score(solution_str, ground_truth)
        format_ok = float(has_answer_tag(solution_str))
        loop_ratio = loop_repetition_ratio(solution_str)
        loop_penalty = loop_penalty_weight * loop_ratio
        if mode == "answer":
            score_value = final_weight * final_correct
        elif mode == "format":
            score_value = final_weight * final_correct + format_weight * format_ok
        else:
            raise ValueError(f"Unknown reward mode: {mode!r}")
        score_value -= loop_penalty

        proc = verify_process(expression, solution_str, ground_truth)
        result = {
            "score": score_value,
            "acc": final_correct,
            "correct": final_correct,
            "format_ok": format_ok,
            "process_component": 0.0,
            "final_component": score_value,
            "loop_ratio": loop_ratio,
            "loop_penalty": loop_penalty,
            **{k: v for k, v in proc.items() if k != "verified_char_ends"},
            "verified_char_ends": proc["verified_char_ends"],
        }

    if not include_spans:
        result = {k: v for k, v in result.items() if k != "verified_char_ends"}
    return result
