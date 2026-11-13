#!/usr/bin/env python3
"""Hydra entry point for the verl arithmetic-RLVR experiments.

The actual training loop is verl 0.5.0.  We reuse the small, machine-tested
harness from ``verllearning/demo`` (Ray + FSDP + HF rollout + hook injection)
and plug in this project's reward function, reward manager and advantage
estimators through verl's public extension points.

Run with the ``shtsky`` environment; ``verl_impl/run.sh`` does the setup.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERL_LEARNING = Path("/root/work/tiankuan/shtskysmile/verllearning")
for path in (str(PROJECT_ROOT), str(VERL_LEARNING)):
    if path not in sys.path:
        sys.path.insert(0, path)

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from demo.task_runner import run_demo  # noqa: E402


@hydra.main(config_path="config", config_name="arithmetic_grpo_process", version_base=None)
def main(config: DictConfig) -> None:
    print(
        "\n===== verl arithmetic RLVR =====\n"
        "generate -> reward (answer/format/process) -> old_log_prob -> "
        "ref_log_prob -> advantage -> update_actor -> checkpoint\n"
        "reward: verl_impl/rewards.py + verl_impl/reward_manager.py\n"
        "advantage: verl_impl/algorithms.py (grpo / rloo / process_grpo)\n"
        "================================\n"
    )
    run_demo(config)


if __name__ == "__main__":
    main()
