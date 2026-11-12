from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_effort(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint efforts applied to the simulation. Isaac Lab 2.3+ has this in isaaclab.envs.mdp; 2.1 does not."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]


def gait_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    return phase


def waypoint_velocity(
    env: ManagerBasedRLEnv,
    command_name: str = "waypoint",
    linear_gain: float = 0.6,
    lateral_gain: float = 0.5,
    yaw_gain: float = 1.2,
    stop_radius: float = 0.55,
) -> torch.Tensor:
    """Convert a privileged waypoint into the velocity command used by the Go2 teacher.

    The waypoint command is expressed in the robot base frame.  This helper is
    deliberately only used by the privileged ``policy`` observation group of the visual
    task; the RGB student never receives this tensor.
    """
    waypoint_b = env.command_manager.get_command(command_name)
    target_xy = waypoint_b[:, :2]
    heading_error = torch.atan2(target_xy[:, 1], target_xy[:, 0])
    command = torch.empty(env.num_envs, 3, device=env.device, dtype=target_xy.dtype)
    command[:, 0] = torch.clamp(linear_gain * target_xy[:, 0], min=0.0, max=1.0)
    command[:, 1] = torch.clamp(lateral_gain * target_xy[:, 1], min=-0.4, max=0.4)
    command[:, 2] = torch.clamp(yaw_gain * heading_error, min=-1.0, max=1.0)
    # A navigation instruction needs a terminal action, not infinitesimal
    # tracking commands forever.  The same radius is used by closed-loop
    # evaluation, so the RGB model receives an unambiguous ``stop`` label
    # once it is close enough to the visible landmark.
    command[target_xy.norm(dim=-1) <= stop_radius] = 0.0
    return command
