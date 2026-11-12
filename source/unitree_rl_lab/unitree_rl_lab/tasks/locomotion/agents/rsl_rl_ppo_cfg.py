# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "unitree_go2_velocity"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1PPORunnerCfg(BasePPORunnerCfg):
    """Official Unitree-G1-29dof-Velocity PPO teacher."""

    experiment_name = "unitree_g1_29dof_velocity"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )


@configclass
class SpotPPORunnerCfg(BasePPORunnerCfg):
    """Spot velocity MDP PPO teacher."""

    experiment_name = "spot_unitree_mdp"
    max_iterations = 3000


@configclass
class H1PPORunnerCfg(BasePPORunnerCfg):
    """H1 velocity MDP PPO teacher."""

    experiment_name = "unitree_h1_velocity"
    max_iterations = 3000
RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1PPORunnerCfg(BasePPORunnerCfg):
    """Official Unitree-G1-29dof-Velocity PPO. Same algorithm as Go2; separate log dir.

    Only change vs GitHub default: ``noise_std_type="log"`` so std cannot go
    negative/NaN (scalar std blew up near iter 22k twice).
    """

    experiment_name = "unitree_g1_29dof_velocity"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )


@configclass
class SpotPPORunnerCfg(BasePPORunnerCfg):
    """Spot on Unitree-lab velocity MDP (not Isaac Flat Spot). Same PPO as Go2."""

    experiment_name = "spot_unitree_mdp"
    max_iterations = 3000

