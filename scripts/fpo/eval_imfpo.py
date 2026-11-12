#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Deployment evaluation for the iMF-RFPO study: return, NFE and latency.

Only ``act_inference`` is executed (no JVP, no V, no ODE for MeanFlow, no training
terms), so the measured actor latency is the deployed policy alone.

Per checkpoint and Euler budget (flow) / 1 NFE (MeanFlow) it reports:
  - mean / std episodic return and episode length (zero and random x0)
  - NFE per action (counted with a forward hook on the actor MLP)
  - actor forward latency during rollout (batch = num_envs, CUDA-synchronized)
  - env step latency
  - micro-benchmark latency at batch 1 on GPU and single-thread CPU (30 warm-up, 100 timed)

Usage (one variant per call; the actor layout depends on the variant):
  python scripts/fpo/eval_imfpo.py --task Unitree-Go2-Velocity --fpo_variant baseline \
      --model fpo=/path/model_1499.pt --sampling_steps 64 1 --headless
  python scripts/fpo/eval_imfpo.py --task Unitree-Go2-Velocity --fpo_variant imf \
      --model imf=/path/model_1499.pt --headless
"""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "source" / "isaaclab_fpo"))

from isaaclab.app import AppLauncher
from isaaclab_fpo import cli_args  # isort: skip

import argparse

parser = argparse.ArgumentParser(description="Return / NFE / latency evaluation of flow and MeanFlow policies.")
parser.add_argument("--task", type=str, default="Unitree-Go2-Velocity")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--model", type=str, action="append", required=True, help="label=path (repeatable).")
parser.add_argument("--sampling_steps", type=int, nargs="+", default=[64, 1], help="Flow Euler budgets.")
parser.add_argument("--eval_episodes", type=int, default=256)
parser.add_argument("--seed", type=int, default=42, help="Fixed environment seed for repeatable checkpoint comparisons.")
parser.add_argument("--eval_modes", type=str, nargs="+", default=["zero", "random"])
parser.add_argument("--latency_warmup", type=int, default=30)
parser.add_argument("--latency_iters", type=int, default=100)
parser.add_argument("--output_json", type=str, default=None)
parser.add_argument("--disable_fabric", action="store_true", default=False)
cli_args.add_fpo_args(parser)
if "--help" in sys.argv or "-h" in sys.argv:
    parser.print_help()
    raise SystemExit(0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_fpo.patches import apply_isaaclab_patches

apply_isaaclab_patches()

import copy
import json
import os
import time

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

from isaaclab_fpo import FpoRslRlVecEnvWrapper
from isaaclab_fpo.runners import OnPolicyRunner


def parse_checkpoint_arg(spec: str) -> tuple[str, str]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), path.strip()
    return os.path.basename(spec).replace(".pt", ""), spec


def _sync(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _seed_eval_rng(eval_mode: str) -> int | None:
    """Reset stochastic action/physics RNGs so checkpoint comparisons share noise.

    The environment seed alone does not seed ``torch.randn`` used by the
    ``random`` initial-flow-state mode.  Reset after each environment reset so
    every checkpoint sees the same rollout noise stream for a given mode.
    """
    if eval_mode == "random":
        seed = int(args_cli.seed) + 1
    elif eval_mode == "zero":
        seed = int(args_cli.seed)
    else:
        # ``fixed_seed`` is seeded inside act_inference; still fix any other
        # stochastic torch operations in the environment rollout.
        seed = int(args_cli.seed) + 2
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2**32))
    return seed


def _stats_ms(samples: list[float]) -> dict:
    arr = np.asarray(samples) * 1e3
    return {"mean_ms": float(arr.mean()), "p50_ms": float(np.percentile(arr, 50)), "p99_ms": float(np.percentile(arr, 99))}


@torch.no_grad()
def benchmark_batch1(policy, num_obs: int, device: str) -> dict:
    """Latency of one deployed action at batch size 1 (x0 = 0, as on the robot)."""
    bench = copy.deepcopy(policy).to(device).eval()
    bench._compiled_integrate_flow = bench._integrate_flow
    obs = torch.zeros(1, num_obs, device=device)
    for _ in range(args_cli.latency_warmup):
        bench.act_inference(obs, eval_mode="zero")
    samples = []
    for _ in range(args_cli.latency_iters):
        _sync(device)
        start = time.perf_counter()
        bench.act_inference(obs, eval_mode="zero")
        _sync(device)
        samples.append(time.perf_counter() - start)
    return _stats_ms(samples)


@torch.no_grad()
def evaluate(runner: OnPolicyRunner, eval_mode: str) -> dict:
    policy = runner.alg.policy
    nfe_counter = [0]
    hook = policy.actor.register_forward_hook(lambda *_: nfe_counter.__setitem__(0, nfe_counter[0] + 1))

    obs, _ = runner.env.reset()
    obs = obs.to(runner.device)
    eval_rng_seed = _seed_eval_rng(eval_mode)
    num_envs = runner.env.num_envs
    rewards_done, lengths_done = [], []
    episode_reward = torch.zeros(num_envs, device=runner.device)
    episode_length = torch.zeros(num_envs, device=runner.device)
    episodes_per_env = torch.zeros(num_envs, dtype=torch.long, device=runner.device)
    target_per_env = max(1, args_cli.eval_episodes // num_envs)
    max_episode_length = getattr(runner.env, "max_episode_length", 1000)
    actor_times, env_times, num_calls = [], [], 0

    while (episodes_per_env < target_per_env).any():
        norm_obs = runner.obs_normalizer(obs) if runner.cfg.empirical_normalization else obs
        _sync(runner.device)
        start = time.perf_counter()
        actions = policy.act_inference(norm_obs, eval_mode=eval_mode, eval_fixed_seed=runner.cfg.flow_eval_fixed_seed)
        _sync(runner.device)
        actor_times.append(time.perf_counter() - start)
        num_calls += 1

        start = time.perf_counter()
        obs, rewards, dones, _ = runner.env.step(actions.to(runner.env.device))
        _sync(runner.device)
        env_times.append(time.perf_counter() - start)

        obs = obs.to(runner.device)
        episode_reward += rewards.to(runner.device)
        episode_length += 1
        done_mask = (dones.to(runner.device) > 0) | (episode_length >= max_episode_length)
        for idx in done_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
            if episodes_per_env[idx] < target_per_env:
                rewards_done.append(episode_reward[idx].item())
                lengths_done.append(episode_length[idx].item())
                episodes_per_env[idx] += 1
        episode_reward[done_mask] = 0
        episode_length[done_mask] = 0

    hook.remove()
    return {
        "mean_reward": float(np.mean(rewards_done)),
        "std_reward": float(np.std(rewards_done)),
        "mean_length": float(np.mean(lengths_done)),
        "num_episodes": len(rewards_done),
        "eval_rng_seed": eval_rng_seed,
        "nfe": nfe_counter[0] / max(num_calls, 1),
        "actor_latency_rollout": _stats_ms(actor_times),
        "env_step_latency": _stats_ms(env_times),
    }


def main():
    try:
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
            entry_point_key="play_env_cfg_entry_point",
        )
    except ValueError as exc:
        if "play_env_cfg_entry_point" not in str(exc):
            raise
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
            entry_point_key="env_cfg_entry_point",
        )
    # Keep episode randomization identical across separately launched model
    # evaluations (for example, iMFPO versus RFPO).
    env_cfg.seed = args_cli.seed
    agent_cfg = cli_args.parse_fpo_cfg(args_cli.task, args_cli)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = FpoRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg.device)
    runner.eval_mode()
    policy = runner.alg.policy
    is_meanflow = getattr(policy, "is_meanflow", False)
    budgets = [1] if is_meanflow else list(args_cli.sampling_steps)

    results = {}
    for label, ckpt_path in (parse_checkpoint_arg(spec) for spec in args_cli.model):
        ckpt_path = retrieve_file_path(ckpt_path)
        print(f"\n>>> {label} ({args_cli.fpo_variant}, {'meanflow' if is_meanflow else 'flow'}): {ckpt_path}")
        runner.load(ckpt_path, load_optimizer=False)
        runner.eval_mode()
        policy._compiled_integrate_flow = policy._integrate_flow
        results[label] = {"variant": args_cli.fpo_variant, "checkpoint": ckpt_path, "budgets": {}}
        default_steps = policy.sampling_steps
        for steps in budgets:
            policy.sampling_steps = steps
            entry = {
                "latency_batch1_gpu": benchmark_batch1(policy, policy.num_actor_obs, runner.device),
                "latency_batch1_cpu_1thread": None,
            }
            num_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            entry["latency_batch1_cpu_1thread"] = benchmark_batch1(policy, policy.num_actor_obs, "cpu")
            torch.set_num_threads(num_threads)
            for mode in args_cli.eval_modes:
                entry[mode] = evaluate(runner, mode)
                res = entry[mode]
                print(
                    f"  budget={steps:3d} {mode:6s} reward={res['mean_reward']:8.2f}±{res['std_reward']:6.2f} "
                    f"len={res['mean_length']:6.1f} NFE={res['nfe']:.0f} "
                    f"actor={res['actor_latency_rollout']['mean_ms']:.3f}ms env={res['env_step_latency']['mean_ms']:.2f}ms"
                )
            print(
                f"  budget={steps:3d} batch-1 latency: gpu={entry['latency_batch1_gpu']['mean_ms']:.3f}ms "
                f"cpu(1 thread)={entry['latency_batch1_cpu_1thread']['mean_ms']:.3f}ms "
                f"(p99 {entry['latency_batch1_cpu_1thread']['p99_ms']:.3f}ms)"
            )
            results[label]["budgets"][steps] = entry
        policy.sampling_steps = default_steps

    print("\n" + "=" * 110)
    print(f"{'model':<20}{'budget':>7}{'NFE':>5}" + "".join(f"{m + ' return':>16}" for m in args_cli.eval_modes)
          + f"{'cpu1 ms':>10}{'gpu1 ms':>10}{'env ms':>9}")
    for label, res in results.items():
        for steps, entry in res["budgets"].items():
            row = f"{label:<20}{steps:>7}{entry[args_cli.eval_modes[0]]['nfe']:>5.0f}"
            row += "".join(f"{entry[m]['mean_reward']:>16.2f}" for m in args_cli.eval_modes)
            row += f"{entry['latency_batch1_cpu_1thread']['mean_ms']:>10.3f}{entry['latency_batch1_gpu']['mean_ms']:>10.3f}"
            row += f"{entry[args_cli.eval_modes[0]]['env_step_latency']['mean_ms']:>9.2f}"
            print(row)
    print("=" * 110)

    if args_cli.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.output_json)), exist_ok=True)
        with open(args_cli.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[INFO] wrote {args_cli.output_json}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
