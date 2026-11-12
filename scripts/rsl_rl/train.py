#!/usr/bin/env python3
"""Train a native RSL-RL PPO teacher on a Unitree velocity task."""

from __future__ import annotations

import argparse
import datetime
import importlib
import inspect
import os
import pathlib
import shutil
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "source" / "unitree_rl_lab"))
sys.path.insert(0, str(_ROOT / "scripts" / "rsl_rl"))

import cli_args as rsl_cli_args
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Unitree-Go2-Velocity")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
rsl_cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401
importlib.import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof")
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    agent_cfg = rsl_cli_args.parse_rsl_rl_cfg(args.task, args)
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args.device if args.device is not None else env_cfg.sim.device

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root, log_dir)
    print(f"[INFO] Logging PPO teacher to {log_dir}")

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if args.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args.video_interval == 0,
            "video_length": args.video_length,
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

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if agent_cfg.resume:
        if not args.checkpoint:
            raise ValueError("--resume requires --checkpoint /path/to/model_*.pt")
        print(f"[INFO] Resuming PPO teacher from {args.checkpoint}")
        runner.load(args.checkpoint)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
