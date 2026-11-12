"""End-to-end smoke test of OnPolicyRunner + FPO with a tiny mock locomotion env.

Exercises rollout (1-NFE / multi-step sampling), storage of (eps, t, r), FPO update with
the iMF / naive MeanFlow regression, diagnostics, checkpointing, post-training eval and
ONNX export, for every variant of the controlled study. No Isaac Lab needed.

Run:  python tests/smoke_runner_mock_env.py [--device cuda:0] [--iters 3]
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import tempfile

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _stubs  # noqa: E402

_stubs.install()

from isaaclab_fpo.env import VecEnv  # noqa: E402
from isaaclab_fpo.runners import OnPolicyRunner  # noqa: E402
from isaaclab_fpo.task_cfgs import GO2_FPO_VARIANTS  # noqa: E402

VARIANTS = ("baseline", "meanflow_naive", "imf", "imf_full")
REPORT_KEYS = (
    "surrogate_loss",
    "metrics/policy_nfe",
    "metrics/actor_grad/norm_mean",
    "metrics/actor_grad/nonfinite_total",
    "metrics/action/saturation_ratio",
    "metrics/meanflow/jvp_norm_mean",
    "metrics/meanflow/jvp_norm_p99",
    "metrics/meanflow/V_minus_vc_sq_mean",
    "metrics/meanflow/u_minus_vc_sq_mean",
    "metrics/meanflow/u_vhat_gap_mean",
    "metrics/drift/target",
    "metrics/drift/action",
)


class MockLocomotionEnv(VecEnv):
    """Linear-Gaussian dynamics; reward for tracking a fixed nonlinear action map."""

    def __init__(self, num_envs=64, num_obs=12, num_actions=4, max_episode_length=40, device="cpu"):
        self.num_envs, self.num_obs, self.num_actions = num_envs, num_obs, num_actions
        self.max_episode_length = max_episode_length
        self.device = torch.device(device)
        self.cfg = {}
        self.unwrapped = self
        g = torch.Generator().manual_seed(0)
        self._w = (torch.randn(num_obs, num_actions, generator=g) / math.sqrt(num_obs)).to(self.device)
        self._b = torch.randn(num_actions, num_obs, generator=g).to(self.device) * 0.1
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._obs = torch.randn(num_envs, num_obs, device=self.device)

    def get_observations(self):
        return self._obs, {"observations": {}}

    def reset(self):
        self._obs = torch.randn(self.num_envs, self.num_obs, device=self.device)
        self.episode_length_buf.zero_()
        return self._obs, {"observations": {}}

    def step(self, actions):
        actions = actions.clamp(-2.0, 2.0)
        target = torch.tanh(self._obs @ self._w)
        rewards = 1.0 - (actions - target).pow(2).mean(dim=-1)
        self._obs = 0.9 * self._obs + 0.1 * actions @ self._b + 0.1 * torch.randn_like(self._obs)
        self.episode_length_buf += 1
        time_outs = self.episode_length_buf >= self.max_episode_length
        falls = (actions - target).abs().max(dim=-1).values > 1.9
        dones = time_outs | falls
        if dones.any():
            self._obs[dones] = torch.randn(int(dones.sum()), self.num_obs, device=self.device)
            self.episode_length_buf[dones] = 0
        infos = {"observations": {}, "time_outs": time_outs.float(), "log": {"/mock/target_err": (actions - target).abs().mean()}}
        return self._obs, rewards, dones.float(), infos


def shrink(cfg):
    cfg.num_steps_per_env = 8
    cfg.save_interval = 1
    cfg.eval_episodes = 64
    cfg.logger = "tensorboard"
    cfg.policy.sampling_steps = 8
    cfg.algorithm.n_samples_per_action = 4
    cfg.algorithm.num_learning_epochs = 2
    cfg.algorithm.num_mini_batches = 2
    cfg.algorithm.target_drift_num_samples = 128
    return cfg


def run_variant(variant: str, device: str, iters: int, workdir: pathlib.Path) -> dict:
    torch.manual_seed(0)
    cfg = shrink(GO2_FPO_VARIANTS[variant]())
    cfg.device = device
    env = MockLocomotionEnv(device=device)
    log_dir = workdir / variant
    runner = OnPolicyRunner(env, cfg, log_dir=str(log_dir), device=device)
    runner.alg.policy._compiled_integrate_flow = runner.alg.policy._integrate_flow

    history = []
    original_update = runner.alg.update

    def recording_update(*args, **kwargs):
        out = original_update(*args, **kwargs)
        flat = {k: v for k, v in out.items() if k != "metrics"}
        flat.update({f"metrics/{k}": v for k, v in out["metrics"].items()})
        history.append(flat)
        return out

    runner.alg.update = recording_update
    runner.learn(num_learning_iterations=iters)

    ckpts = sorted(log_dir.glob("model_*.pt"))
    assert ckpts, "no checkpoint saved"
    runner.load(str(ckpts[-1]))
    policy = runner.alg.policy
    policy.eval()
    calls = []
    handle = policy.actor.register_forward_hook(lambda *_: calls.append(1))
    actions = policy.act_inference(env.get_observations()[0], eval_mode="zero")
    handle.remove()
    assert torch.isfinite(actions).all()

    onnx_status = "skipped"
    try:
        from isaaclab_fpo.exporter import export_policy_as_onnx

        export_policy_as_onnx(policy, str(log_dir / "exported"))
        onnx_status = "ok"
    except Exception as exc:  # onnx may be missing in the test interpreter
        onnx_status = f"failed ({type(exc).__name__}: {str(exc)[:80]})"
    policy.to(device)

    for record in history:
        for key, value in record.items():
            assert math.isfinite(float(value)), f"{variant}: non-finite {key}={value}"
    return {"history": history, "nfe": len(calls), "onnx": onnx_status}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = parser.parse_args()

    workdir = pathlib.Path(tempfile.mkdtemp(prefix="imfpo_smoke_"))
    results = {v: run_variant(v, args.device, args.iters, workdir) for v in args.variants}

    print("\n" + "=" * 100)
    print(f"SMOKE TEST SUMMARY  (device={args.device}, iters={args.iters}, logs={workdir})")
    for variant, res in results.items():
        last = res["history"][-1]
        print(f"\n[{variant}] deploy NFE={res['nfe']}  onnx={res['onnx']}")
        for key in REPORT_KEYS:
            if key in last:
                print(f"    {key:<42s} {float(last[key]): .5f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
