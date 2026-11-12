from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_fpo.rl_cfg import FpoRslRlOnPolicyRunnerCfg


def add_fpo_args(parser: argparse.ArgumentParser):
    """Add iMFPO training arguments."""
    arg_group = parser.add_argument_group("imfpo", description="Arguments for the iMFPO agent.")
    arg_group.add_argument("--experiment_name", type=str, default=None)
    arg_group.add_argument("--run_name", type=str, default=None)
    arg_group.add_argument("--resume", action="store_true", default=False)
    arg_group.add_argument("--load_run", type=str, default=None)
    arg_group.add_argument("--checkpoint", type=str, default=None)
    arg_group.add_argument("--teacher_checkpoint", type=str, default=None)
    arg_group.add_argument("--teacher_kd_coef", type=float, default=None)
    arg_group.add_argument("--teacher_aux_zero_x0_prob", type=float, default=None)
    arg_group.add_argument("--teacher_action_mix_prob", type=float, default=None)
    arg_group.add_argument("--teacher_kd_steps", type=int, nargs="+", default=None)
    arg_group.add_argument("--learning_rate", type=float, default=None)
    arg_group.add_argument("--save_interval", type=int, default=None)
    arg_group.add_argument("--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"})
    arg_group.add_argument("--fpo_variant", type=str, default="imfpo", choices={"imfpo"})
    arg_group.add_argument("--log_project_name", type=str, default=None)
    arg_group.add_argument("--policy_init_checkpoint", type=str, default=None)


def _variants_for_task(task_name: str):
    from isaaclab_fpo.task_cfgs import (
        GO2_FPO_VARIANTS,
        UNITREE_G1_FPO_VARIANTS,
        UNITREE_H1_FPO_VARIANTS,
        UNITREE_SPOT_FPO_VARIANTS,
    )

    if task_name.startswith("Unitree-Spot") or "Spot" in task_name:
        return dict(UNITREE_SPOT_FPO_VARIANTS)
    if "Unitree-Go2" in task_name or "Go2" in task_name:
        return dict(GO2_FPO_VARIANTS)
    if task_name.startswith("Unitree-G1") or ("G1" in task_name and "Tracking" not in task_name):
        return dict(UNITREE_G1_FPO_VARIANTS)
    if task_name.startswith("Unitree-H1") or "H1" in task_name:
        return dict(UNITREE_H1_FPO_VARIANTS)
    raise KeyError(f"No iMFPO variant registry for task '{task_name}'.")


def parse_fpo_cfg(task_name: str, args_cli: argparse.Namespace) -> FpoRslRlOnPolicyRunnerCfg:
    """Parse configuration for the iMFPO agent."""
    variant = getattr(args_cli, "fpo_variant", None) or "imfpo"
    variant_registry = _variants_for_task(task_name)
    if variant not in variant_registry:
        raise KeyError(
            f"Unknown fpo_variant '{variant}' for task '{task_name}'. "
            f"Available: {sorted(variant_registry.keys())}"
        )
    agent_cfg = variant_registry[variant]()
    return update_fpo_cfg(agent_cfg, args_cli)


def update_fpo_cfg(agent_cfg: FpoRslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    elif getattr(agent_cfg, "seed", None) == -1:
        agent_cfg.seed = random.randint(0, 10000)

    agent_cfg.resume = bool(getattr(args_cli, "resume", False))
    if getattr(args_cli, "load_run", None) is not None:
        agent_cfg.load_run = args_cli.load_run
    if getattr(args_cli, "checkpoint", None) is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if getattr(args_cli, "teacher_checkpoint", None):
        agent_cfg.algorithm.reflow_teacher_checkpoint = args_cli.teacher_checkpoint
        agent_cfg.algorithm.teacher_kind = "ppo"
        agent_cfg.algorithm.teacher_kd_enabled = True
    if getattr(args_cli, "teacher_kd_coef", None) is not None:
        agent_cfg.algorithm.teacher_kd_coef = float(args_cli.teacher_kd_coef)
    if getattr(args_cli, "teacher_aux_zero_x0_prob", None) is not None:
        agent_cfg.algorithm.teacher_aux_zero_x0_prob = float(args_cli.teacher_aux_zero_x0_prob)
    if getattr(args_cli, "teacher_action_mix_prob", None) is not None:
        agent_cfg.algorithm.teacher_action_mix_prob = float(args_cli.teacher_action_mix_prob)
    if getattr(args_cli, "teacher_kd_steps", None):
        agent_cfg.algorithm.teacher_kd_steps = [int(s) for s in args_cli.teacher_kd_steps]
    if getattr(args_cli, "learning_rate", None) is not None:
        agent_cfg.algorithm.learning_rate = float(args_cli.learning_rate)
    if getattr(args_cli, "save_interval", None) is not None:
        agent_cfg.save_interval = int(args_cli.save_interval)
    if getattr(args_cli, "run_name", None) is not None:
        agent_cfg.run_name = args_cli.run_name
    if getattr(args_cli, "experiment_name", None) is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if getattr(args_cli, "logger", None) is not None:
        agent_cfg.logger = args_cli.logger
    if agent_cfg.logger in {"wandb", "neptune"} and getattr(args_cli, "log_project_name", None):
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name
    if getattr(args_cli, "device", None):
        agent_cfg.device = args_cli.device
    return agent_cfg
