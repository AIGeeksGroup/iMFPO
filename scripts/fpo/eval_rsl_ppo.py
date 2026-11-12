#!/usr/bin/env python3
"""Evaluate an RSL-RL PPO checkpoint in a Unitree velocity task.

This mirrors the FPO evaluation environment but loads a native RSL-RL PPO policy,
so frozen teachers can be scored directly under a selected reward configuration.
"""

import argparse
import json
import os
import pathlib
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
_RSL_RL_PATH = os.environ.get("RSL_RL_PATH", "")
if _RSL_RL_PATH:
    sys.path.insert(0, _RSL_RL_PATH)
sys.path.insert(0, str(_ROOT / "source" / "unitree_rl_lab"))
sys.path.insert(0, str(_ROOT / "source" / "isaaclab_fpo"))
import cli_args as rsl_cli_args

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Unitree-G1-29dof-Velocity")
parser.add_argument("--model_checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--eval_episodes", type=int, default=256)
parser.add_argument("--output_json", default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--clip_actions", type=float, default=None,
                    help="Optional action clamp applied by the vector wrapper.")
parser.add_argument("--latency_warmup", type=int, default=30)
parser.add_argument("--latency_iters", type=int, default=100)
rsl_cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        entry_point_key="play_env_cfg_entry_point",
    )
    env_cfg.seed = args.seed
    agent_cfg = rsl_cli_args.parse_rsl_rl_cfg(args.task, args)
    agent_cfg.device = args.device
    # Historical G1 checkpoints use the scalar `std` parameter; newer task
    # presets default to `log_std`. Preserve the checkpoint's parameterization.
    if hasattr(agent_cfg.policy, "noise_std_type"):
        agent_cfg.policy.noise_std_type = "scalar"
    if args.clip_actions is not None:
        agent_cfg.clip_actions = args.clip_actions
    env = gym.make(args.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    checkpoint = retrieve_file_path(args.model_checkpoint)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=args.device)
    obs, _ = env.get_observations()
    obs = obs.to(args.device)

    # Benchmark one deployed PPO action on the same single-sample, GPU path
    # used by eval_imfpo.py. This keeps the latency panel comparable.
    probe = torch.zeros((1, obs.shape[-1]), device=args.device)
    if torch.device(args.device).type == "cuda":
        torch.cuda.synchronize(args.device)
    with torch.inference_mode():
        for _ in range(args.latency_warmup):
            policy(probe)
        if torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(args.device)
        latency_samples = []
        for _ in range(args.latency_iters):
            if torch.device(args.device).type == "cuda":
                torch.cuda.synchronize(args.device)
            start = time.perf_counter()
            policy(probe)
            if torch.device(args.device).type == "cuda":
                torch.cuda.synchronize(args.device)
            latency_samples.append((time.perf_counter() - start) * 1e3)
    latency_batch1_gpu = {
        "mean_ms": float(np.mean(latency_samples)),
        "p50_ms": float(np.percentile(latency_samples, 50)),
        "p99_ms": float(np.percentile(latency_samples, 99)),
    }

    target = max(1, args.eval_episodes // env.num_envs)
    per_env_count = torch.zeros(env.num_envs, dtype=torch.long, device=args.device)
    ep_return = torch.zeros(env.num_envs, device=args.device)
    ep_length = torch.zeros(env.num_envs, device=args.device)
    returns, lengths = [], []
    max_length = getattr(env, "max_episode_length", 1000)
    with torch.inference_mode():
        while (per_env_count < target).any():
            actions = policy(obs)
            obs, rewards, dones, _ = env.step(actions)
            obs = obs.to(args.device)
            ep_return += rewards.to(args.device)
            ep_length += 1
            done_mask = (dones.to(args.device) > 0) | (ep_length >= max_length)
            for i in done_mask.nonzero(as_tuple=False).flatten().tolist():
                if per_env_count[i] < target:
                    returns.append(ep_return[i].item())
                    lengths.append(ep_length[i].item())
                    per_env_count[i] += 1
            ep_return[done_mask] = 0
            ep_length[done_mask] = 0

    result = {
        "task": args.task,
        "checkpoint": checkpoint,
        "num_envs": args.num_envs,
        "num_episodes": len(returns),
        "mean_reward": float(np.mean(returns)),
        "std_reward": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "latency_batch1_gpu": latency_batch1_gpu,
        "clip_actions": agent_cfg.clip_actions,
        "reward_env_overrides": {"G1_ARM_DEV_WEIGHT": os.getenv("G1_ARM_DEV_WEIGHT")},
    }
    print(json.dumps(result, indent=2))
    if args.output_json:
        out = pathlib.Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
