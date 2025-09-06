# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for a Unitree Go2 robust walking task.

This environment is designed as a curriculum step to teach the robot
to walk and run at a commanded forward velocity on flat ground while
resisting random pushes and mass changes (Domain Randomization).

The resulting policy is an ideal checkpoint for more complex downstream tasks.
"""

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

# Import the base configuration classes we will inherit from
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg as Go2BaseActionsCfg,
    ObservationsCfg as Go2BaseObservationsCfg,
    RewardsCfg as Go2BaseRewardsCfg,
    TerminationsCfg as Go2BaseTerminationsCfg,
)


# -- Domain Randomization & Helper Functions --

def push_robot(env: ManagerBasedRLEnvCfg, params: dict):
    """Applies a random push to the robot's base to improve robustness."""
    min_push_interval = params["min_push_interval"]
    push_vel_xy_range = params["push_vel_xy_range"]

    last_push_time = env.scene.data.get("last_push_time", torch.zeros(env.num_envs, device=env.device))
    env.scene.data["last_push_time"] = last_push_time
    
    sim_time = env.sim.time
    time_since_last_push = sim_time - last_push_time
    can_push = (time_since_last_push >= min_push_interval)
    
    vel_xy = torch.rand(env.num_envs, 2, device=env.device)
    vel_xy[:, 0] = (push_vel_xy_range["x"][1] - push_vel_xy_range["x"][0]) * vel_xy[:, 0] + push_vel_xy_range["x"][0]
    vel_xy[:, 1] = (push_vel_xy_range["y"][1] - push_vel_xy_range["y"][0]) * vel_xy[:, 1] + push_vel_xy_range["y"][0]
    
    robot = env.scene["robot"]
    robot.data.root_lin_vel_w[can_push, :2] += vel_xy[can_push]
    
    last_push_time[can_push] = sim_time

def add_body_mass_randomization(env: ManagerBasedRLEnvCfg, env_ids: torch.Tensor, mass_range: tuple):
    """Randomizes the mass of the robot's base on reset."""
    if len(env_ids) == 0:
        return
    robot = env.scene["robot"]
    
    base_body_name = "base"
    # CORRECTED: Access the body_props through the rigid_props attribute
    default_base_mass = UNITREE_GO2_CFG.rigid_props.body_props[base_body_name].mass

    rand_mass_ratio = (mass_range[1] - mass_range[0]) * torch.rand(len(env_ids), device=env.device) + mass_range[0]
    rand_mass = default_base_mass * rand_mass_ratio
    
    base_body_idx = robot.body_names.index(base_body_name)
    robot.set_body_masses(rand_mass, body_ids=base_body_idx, env_ids=env_ids)

def penalize_turning(env: ManagerBasedRLEnvCfg) -> torch.Tensor:
    """Penalizes the robot for turning when not commanded."""
    return torch.square(env.scene["robot"].data.root_ang_vel_b[:, 2])


# -- Configuration Classes for the Walking Task --

@configclass
class UnitreeGo2WalkSceneCfg(InteractiveSceneCfg):
    """Scene with only a robot and a flat ground plane."""
    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    base_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/base")


@configclass
class UnitreeGo2WalkObservationsCfg(Go2BaseObservationsCfg):
    """Observations for the walking task (disables height scanner)."""
    def __post_init__(self):
        super().__post_init__()
        self.policy.height_scan = None


@configclass
class UnitreeGo2WalkRewardsCfg(Go2BaseRewardsCfg):
    """Rewards for learning to walk on flat ground."""
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=15.0,
        params={"std": 1.0, "command_name": "base_velocity"},
    )
    is_alive = RewTerm(func=mdp.is_alive, weight=1.0)
    penalize_turning = RewTerm(func=penalize_turning, weight=-2.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6)
    feet_air_time = None
    undesired_contacts = None


@configclass
class UnitreeGo2WalkTerminationsCfg(Go2BaseTerminationsCfg):
    """Terminations for learning to walk on flat ground."""
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("base_contact"), "threshold": 1.0},
    )


@configclass
class UnitreeGo2WalkCommandsCfg:
    """Commands for the walking task."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(1.5, 2.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


@configclass
class UnitreeGo2WalkEventsCfg:
    """Event terms for the walking task."""
    reset_robot_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot")},
    )
    push_robot_interval = EventTerm(
        func=push_robot,
        mode="interval",
        interval_range_s=(0.8, 2.0),
        params={"min_push_interval": 4.0, "push_vel_xy_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
    randomize_base_mass = EventTerm(
        func=add_body_mass_randomization,
        mode="reset",
        params={"mass_range": (0.8, 1.2)}
    )


@configclass
class UnitreeGo2WalkEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Unitree Go2 walking environment."""
    scene: UnitreeGo2WalkSceneCfg = UnitreeGo2WalkSceneCfg(num_envs=4096, env_spacing=4.0)
    rewards: UnitreeGo2WalkRewardsCfg = UnitreeGo2WalkRewardsCfg()
    terminations: UnitreeGo2WalkTerminationsCfg = UnitreeGo2WalkTerminationsCfg()
    commands: UnitreeGo2WalkCommandsCfg = UnitreeGo2WalkCommandsCfg()
    events: UnitreeGo2WalkEventsCfg = UnitreeGo2WalkEventsCfg()
    actions: Go2BaseActionsCfg = Go2BaseActionsCfg()
    observations: UnitreeGo2WalkObservationsCfg = UnitreeGo2WalkObservationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005

# Registration is now handled in the __init__.py file for this directory.