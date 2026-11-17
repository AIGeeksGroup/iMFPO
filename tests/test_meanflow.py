"""CPU regression tests for MeanFlow / iMFPO policy modules."""

from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "source" / "isaaclab_fpo"))

import _stubs

_stubs.install()

import torch
from isaaclab_fpo.modules.actor_critic import ActorCritic
from isaaclab_fpo.rl_cfg import FpoRslRlPpoActorCriticCfg
from isaaclab_fpo.task_cfgs import (
    GO2_FPO_VARIANTS,
    TASK_CONFIGS,
    UNITREE_G1_FPO_VARIANTS,
    UNITREE_H1_FPO_VARIANTS,
    UNITREE_SPOT_FPO_VARIANTS,
)


class TestMeanFlowActor(unittest.TestCase):
    def test_one_step_action_shape(self):
        cfg = FpoRslRlPpoActorCriticCfg(
            actor_hidden_dims=[64, 64],
            critic_hidden_dims=[64, 64],
            activation="elu",
            policy_parameterization="meanflow",
            meanflow_loss_mode="imf",
        )
        policy = ActorCritic(45, 45, 12, cfg)
        obs = torch.randn(8, 45)
        with torch.no_grad():
            actions = policy.act_inference(obs)
        self.assertEqual(tuple(actions.shape), (8, 12))

    def test_imfpo_variant_registry(self):
        self.assertEqual(sorted(GO2_FPO_VARIANTS.keys()), ["imfpo"])
        cfg = GO2_FPO_VARIANTS["imfpo"]()
        self.assertEqual(cfg.policy.policy_parameterization, "meanflow")
        self.assertEqual(cfg.policy.meanflow_loss_mode, "imf")
        self.assertTrue(cfg.algorithm.teacher_kd_enabled)
        self.assertFalse(cfg.algorithm.reflow_enabled)
        self.assertIn("Unitree-Go2-Velocity", TASK_CONFIGS)

    def test_all_robots_use_rhoF_recipe(self):
        """Paper protocol: all embodiments share MeanFlow-iMF + ρF + PPO-KD (no Gaussian ρG)."""
        registries = (
            GO2_FPO_VARIANTS,
            UNITREE_G1_FPO_VARIANTS,
            UNITREE_H1_FPO_VARIANTS,
            UNITREE_SPOT_FPO_VARIANTS,
        )
        for registry in registries:
            cfg = registry["imfpo"]()
            self.assertEqual(cfg.policy.policy_parameterization, "meanflow")
            self.assertEqual(cfg.policy.meanflow_loss_mode, "imf")
            self.assertEqual(float(getattr(cfg.policy, "meanflow_gaussian_std_init", 0.0) or 0.0), 0.0)
            self.assertTrue(cfg.algorithm.teacher_kd_enabled)
            self.assertFalse(cfg.algorithm.reflow_enabled)
            self.assertEqual(float(getattr(cfg.algorithm, "gaussian_entropy_coef", 0.0) or 0.0), 0.0)
            self.assertEqual(float(getattr(cfg.algorithm, "meanflow_aux_imf_coef", 0.0) or 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
