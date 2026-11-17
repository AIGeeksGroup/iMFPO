"""iMFPO runner configs and task registry.

Only the paper iMFPO recipe is exposed: MeanFlow-iMF + velocity-based ratio ρF
+ frozen PPO action distillation (no reflow / Gaussian ρG path).
Train a frozen RSL-RL PPO teacher first, then distill with ``--fpo_variant imfpo``
and ``--teacher_checkpoint``.
"""

from __future__ import annotations

import os

from isaaclab.utils import configclass

from isaaclab_fpo.rl_cfg import (
    FpoRslRlOnPolicyRunnerCfg,
    FpoRslRlPpoActorCriticCfg,
    FpoRslRlPpoAlgorithmCfg,
)

_UNITREE_GO2_PPO_TEACHER = os.environ.get("UNITREE_GO2_PPO_TEACHER", "")


def _meanflow_policy(
    loss_mode: str = "imf", interval_mode: str = "random", **overrides
) -> FpoRslRlPpoActorCriticCfg:
    kwargs = dict(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[768, 768, 768],
        activation="elu",
        policy_parameterization="meanflow",
        meanflow_loss_mode=loss_mode,
        meanflow_interval_mode=interval_mode,
    )
    kwargs.update(overrides)
    return FpoRslRlPpoActorCriticCfg(**kwargs)


def _imfpo_algorithm(**overrides) -> FpoRslRlPpoAlgorithmCfg:
    """Shared iMFPO + PPO-teacher KD settings (ρF surrogate; no reflow / adaptive)."""
    kwargs = dict(
        reflow_enabled=False,
        reflow_teacher_checkpoint=_UNITREE_GO2_PPO_TEACHER,
        teacher_kind="ppo",
        teacher_kd_enabled=True,
        teacher_kd_coef=0.1,
        teacher_kd_steps=[64, 8, 4],
        teacher_aux_zero_x0_prob=0.25,
        theory_metrics_enabled=False,
        adaptive_compute_enabled=False,
    )
    kwargs.update(overrides)
    return FpoRslRlPpoAlgorithmCfg(**kwargs)


@configclass
class UnitreeGo2ImfpoRunnerCfg(FpoRslRlOnPolicyRunnerCfg):
    """Go2 paper iMFPO: 1-NFE MeanFlow + ρF + PPO action KD."""

    experiment_name = "unitree_go2_imfpo"
    empirical_normalization = False
    clip_actions = None
    max_iterations = 2500
    policy = _meanflow_policy("imf")
    algorithm = _imfpo_algorithm()


@configclass
class UnitreeG1ImfpoRunnerCfg(FpoRslRlOnPolicyRunnerCfg):
    """G1 paper iMFPO: same ρF + PPO-KD recipe as Go2 (no Gaussian ρG path)."""

    experiment_name = "unitree_g1_imfpo"
    empirical_normalization = False
    clip_actions = 2.0
    max_iterations = 2500
    enable_post_training_eval = False
    policy = _meanflow_policy("imf")
    algorithm = _imfpo_algorithm(reflow_teacher_checkpoint="")


@configclass
class UnitreeH1ImfpoRunnerCfg(FpoRslRlOnPolicyRunnerCfg):
    """H1 paper iMFPO: same ρF + PPO-KD recipe as Go2 (no Gaussian ρG path)."""

    experiment_name = "unitree_h1_imfpo"
    empirical_normalization = False
    clip_actions = 2.0
    max_iterations = 2500
    enable_post_training_eval = False
    policy = _meanflow_policy("imf")
    algorithm = _imfpo_algorithm(reflow_teacher_checkpoint="")


@configclass
class UnitreeSpotImfpoRunnerCfg(FpoRslRlOnPolicyRunnerCfg):
    """Spot paper iMFPO: same ρF + PPO-KD recipe as Go2 (no Gaussian ρG path)."""

    experiment_name = "unitree_spot_imfpo"
    empirical_normalization = False
    clip_actions = None
    max_iterations = 2500
    enable_post_training_eval = False
    policy = _meanflow_policy("imf")
    algorithm = _imfpo_algorithm(reflow_teacher_checkpoint="")


GO2_FPO_VARIANTS = {"imfpo": UnitreeGo2ImfpoRunnerCfg}
UNITREE_G1_FPO_VARIANTS = {"imfpo": UnitreeG1ImfpoRunnerCfg}
UNITREE_H1_FPO_VARIANTS = {"imfpo": UnitreeH1ImfpoRunnerCfg}
UNITREE_SPOT_FPO_VARIANTS = {"imfpo": UnitreeSpotImfpoRunnerCfg}
G1_FPO_VARIANTS = UNITREE_G1_FPO_VARIANTS
SPOT_FPO_VARIANTS = UNITREE_SPOT_FPO_VARIANTS

TASK_CONFIGS = {
    "Unitree-Go2-Velocity": UnitreeGo2ImfpoRunnerCfg,
    "Unitree-G1-29dof-Velocity": UnitreeG1ImfpoRunnerCfg,
    "Unitree-H1-Velocity": UnitreeH1ImfpoRunnerCfg,
    "Unitree-Spot-Velocity": UnitreeSpotImfpoRunnerCfg,
}
