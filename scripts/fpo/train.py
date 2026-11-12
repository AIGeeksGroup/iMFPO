#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Train iMFPO on official unitree_rl_lab velocity tasks."""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "source" / "isaaclab_fpo"))
sys.path.insert(0, str(_ROOT / "source" / "unitree_rl_lab"))

from isaaclab.app import AppLauncher
from isaaclab_fpo import cli_args  # isort: skip

import argparse

parser = argparse.ArgumentParser(description="Train iMFPO on a Unitree RL Lab task.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Unitree-Go2-Velocity")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--disable_post_training_eval", action="store_true", default=False)
cli_args.add_fpo_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_fpo.patches import apply_isaaclab_patches

apply_isaaclab_patches()

import gymnasium as gym
import inspect
import os
import shutil
import torch
from datetime import datetime

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

from isaaclab_fpo import FpoRslRlVecEnvWrapper
from isaaclab_fpo.runners import OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = cli_args.parse_fpo_cfg(args_cli.task, args_cli)
    if args_cli.disable_post_training_eval:
        agent_cfg.enable_post_training_eval = False
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.abspath(os.path.join("logs", "fpo", agent_cfg.experiment_name))
    print(f"[INFO] Logging iMFPO experiment in {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    resume_path = None
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        ckpt = getattr(agent_cfg, "load_checkpoint", None)
        if ckpt and os.path.isfile(ckpt):
            resume_path = os.path.abspath(ckpt)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    env = FpoRslRlVecEnvWrapper(
        env,
        clip_actions=agent_cfg.clip_actions,
        actor_obs_group=agent_cfg.actor_obs_group,
    )
    runner = OnPolicyRunner(env, agent_cfg, log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if args_cli.policy_init_checkpoint:
        if agent_cfg.resume:
            raise ValueError("--policy_init_checkpoint cannot be combined with --resume")
        init_ckpt = torch.load(args_cli.policy_init_checkpoint, map_location="cpu", weights_only=False)
        source_state = init_ckpt.get("model_state_dict", init_ckpt)
        target_state = runner.alg.policy.state_dict()
        copied = 0
        for key, source_value in source_state.items():
            if key in target_state and target_state[key].shape == source_value.shape:
                target_state[key] = source_value
                copied += 1
        runner.alg.policy.load_state_dict(target_state)
        if runner.alg.ema is not None:
            runner.alg.ema.reset_to_current()
        print(f"[INFO] Warm-started policy from {args_cli.policy_init_checkpoint} ({copied} tensors)")

    if resume_path is not None:
        print(f"[INFO] Loading checkpoint: {resume_path}")
        runner.load(resume_path)

    teacher = getattr(agent_cfg.algorithm, "reflow_teacher_checkpoint", "") or ""
    print(f"[INFO] iMFPO teacher checkpoint: {teacher or '(none)'}")
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
