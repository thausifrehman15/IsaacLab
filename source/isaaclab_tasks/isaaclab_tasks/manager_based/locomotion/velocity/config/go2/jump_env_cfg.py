# Copyright (c) 2022-2025, The Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""
Unitree Go2 Jump Table task (Isaac Lab 2 + Isaac Sim 5).

Two 4x4 m platforms: a green START and a red TARGET. The robot starts on START,
must accelerate forward, become airborne over the gap (configurable), and land on TARGET.
Success/airborne are computed from a feet contact sensor history, fully vectorized.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg, RigidBodyMaterialCfg
from isaaclab.sim.spawners.shapes import CuboidCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
import isaaclab.utils.math as math_utils

# Base Go2 velocity stacks
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg as Go2BaseActionsCfg,
    ObservationsCfg as Go2BaseObservationsCfg,
    RewardsCfg as Go2BaseRewardsCfg,
    TerminationsCfg as Go2BaseTerminationsCfg,
)

# --------------------------
# Scene layout (meters)
# --------------------------
PLATFORM_LENGTH = 6.0  # Reasonable size for single robot
PLATFORM_WIDTH = 6.0   
PLATFORM_HEIGHT = 0.25

# Adjusted gap for better jumping dynamics  
START_PLATFORM_POS = (0.0, 0.0, PLATFORM_HEIGHT / 2.0)
TARGET_PLATFORM_POS = (8.0, 0.0, PLATFORM_HEIGHT / 2.0)  # 2m gap between platforms
JUMP_GAP_DISTANCE = TARGET_PLATFORM_POS[0] - START_PLATFORM_POS[0] - PLATFORM_LENGTH


# --------------------------
# Enhanced reward functions for target movement
# --------------------------
def distance_to_target_reward(env) -> torch.Tensor:
    """Strong reward for getting closer to target platform."""
    robot_pos = env.scene["robot"].data.root_pos_w[:, 0]  # X position
    target_x = TARGET_PLATFORM_POS[0]
    
    # Distance-based reward with exponential scaling
    distance_to_target = torch.abs(target_x - robot_pos)
    return torch.exp(-distance_to_target / 3.0)


def forward_progress_reward(env) -> torch.Tensor:
    """Reward for making forward progress toward target."""
    robot_pos = env.scene["robot"].data.root_pos_w[:, 0]
    start_x = START_PLATFORM_POS[0]
    target_x = TARGET_PLATFORM_POS[0]
    
    # Normalized progress from start to target (0 to 1)
    progress = torch.clamp((robot_pos - start_x) / (target_x - start_x), 0.0, 1.0)
    return progress


def maintain_forward_heading(env) -> torch.Tensor:
    """Enhanced reward for facing toward target."""
    forward_vec_b = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, -1)
    fwd_w = math_utils.quat_apply(env.scene["robot"].data.root_quat_w, forward_vec_b)
    return torch.exp(2.0 * fwd_w[:, 0])


def forward_velocity_reward(env) -> torch.Tensor:
    """Enhanced reward for forward velocity with speed scaling."""
    vx = env.scene["robot"].data.root_lin_vel_w[:, 0]
    # Reward increases with forward speed up to optimal speed
    optimal_speed = 3.0  # m/s
    speed_reward = torch.clamp(vx / optimal_speed, 0.0, 1.0)
    return speed_reward ** 0.5


def combined_movement_reward(env) -> torch.Tensor:
    """Combined reward for movement toward target."""
    distance_reward = distance_to_target_reward(env)
    progress_reward = forward_progress_reward(env)
    heading_reward = maintain_forward_heading(env)
    velocity_reward = forward_velocity_reward(env)
    
    # Weighted combination emphasizing distance and progress
    return (2.0 * distance_reward + 
            2.0 * progress_reward + 
            1.0 * heading_reward + 
            1.0 * velocity_reward) / 6.0


def airborne_over_gap(env) -> torch.Tensor:
    """Reward for being airborne over the gap between platforms."""
    pos = env.scene["robot"].data.root_pos_w
    in_gap = (pos[:, 0] > START_PLATFORM_POS[0] + PLATFORM_LENGTH / 2.0) & (
        pos[:, 0] < TARGET_PLATFORM_POS[0] - PLATFORM_LENGTH / 2.0
    )
    
    try:
        cf_hist = env.scene["contact_forces"].data.net_forces_w_history[:, 0, :, :]
        no_contact = torch.all(torch.norm(cf_hist, dim=-1) < 1.0, dim=-1)
    except (AttributeError, IndexError):
        cf_current = env.scene["contact_forces"].data.net_forces_w
        no_contact = torch.all(torch.norm(cf_current, dim=-1) < 1.0, dim=-1)
    
    return (in_gap & no_contact).float()


def on_target_platform(env) -> torch.Tensor:
    """Success condition: robot on target platform with ground contact."""
    pos = env.scene["robot"].data.root_pos_w
    on_tx = (pos[:, 0] > TARGET_PLATFORM_POS[0] - PLATFORM_LENGTH / 2.0) & (
        pos[:, 0] < TARGET_PLATFORM_POS[0] + PLATFORM_LENGTH / 2.0
    )
    on_ty = (pos[:, 1] > TARGET_PLATFORM_POS[1] - PLATFORM_WIDTH / 2.0) & (
        pos[:, 1] < TARGET_PLATFORM_POS[1] + PLATFORM_WIDTH / 2.0
    )
    
    try:
        cf_hist = env.scene["contact_forces"].data.net_forces_w_history[:, 0, :, :]
        in_contact = torch.any(torch.norm(cf_hist, dim=-1) > 1.0, dim=-1)
    except (AttributeError, IndexError):
        cf_current = env.scene["contact_forces"].data.net_forces_w
        in_contact = torch.any(torch.norm(cf_current, dim=-1) > 1.0, dim=-1)
    
    return on_tx & on_ty & in_contact


def fell_off_platform(env) -> torch.Tensor:
    """Termination condition: robot fell below platform level."""
    pos = env.scene["robot"].data.root_pos_w
    too_low = pos[:, 2] < (PLATFORM_HEIGHT * 0.5)
    return too_low


def out_of_bounds(env) -> torch.Tensor:
    """Termination condition: robot moved too far from the jumping area."""
    pos = env.scene["robot"].data.root_pos_w
    max_y_bound = PLATFORM_WIDTH + 2.0
    out_of_y_bounds = (pos[:, 1] > max_y_bound) | (pos[:, 1] < -max_y_bound)
    too_far_back = pos[:, 0] < (START_PLATFORM_POS[0] - PLATFORM_LENGTH / 2.0 - 1.0)
    return out_of_y_bounds | too_far_back


# --------------------------
# Scene configuration - FIXED ROBOT SPAWN
# --------------------------
@configclass
class UnitreeGo2JumpSceneCfg(InteractiveSceneCfg):
    """Scene with Go2 robots, ground, contact sensors, and two platforms."""

    # FIXED: Use the original UNITREE_GO2_CFG without modification
    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        update_period=0.0,
        history_length=3,
        force_threshold=1.0,
        track_air_time=False,
    )

    base_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/base")

    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())

    start_platform = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/StartPlatform",
        spawn=CuboidCfg(
            size=(PLATFORM_LENGTH, PLATFORM_WIDTH, PLATFORM_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)),
            physics_material=RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0, restitution=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=START_PLATFORM_POS),
    )

    target_platform = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetPlatform",
        spawn=CuboidCfg(
            size=(PLATFORM_LENGTH, PLATFORM_WIDTH, PLATFORM_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
            physics_material=RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0, restitution=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TARGET_PLATFORM_POS),
    )


# --------------------------
# Enhanced observations
# --------------------------
@configclass
class UnitreeGo2JumpObservationsCfg(Go2BaseObservationsCfg):
    """Enhanced observations including target information."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# --------------------------
# Enhanced rewards
# --------------------------
@configclass
class UnitreeGo2JumpRewardsCfg(Go2BaseRewardsCfg):
    """Enhanced reward structure focused on target movement."""

    # Regularization terms
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-7)

    # Primary movement rewards (higher weights)
    movement_reward = RewTerm(func=combined_movement_reward, weight=15.0)
    distance_to_target = RewTerm(func=distance_to_target_reward, weight=10.0)
    forward_progress = RewTerm(func=forward_progress_reward, weight=8.0)
    
    # Jump-specific rewards
    airborne_bonus = RewTerm(func=airborne_over_gap, weight=12.0)
    landing_bonus = RewTerm(func=on_target_platform, weight=100.0)

    # Disable conflicting base rewards
    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None
    feet_air_time = None
    undesired_contacts = None
    flat_orientation_l2 = None


# --------------------------
# Enhanced terminations
# --------------------------
@configclass
class UnitreeGo2JumpTerminationsCfg(Go2BaseTerminationsCfg):
    """Termination conditions for jump task."""

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("base_contact"), "threshold": 1.0},
    )
    fell_off = DoneTerm(func=fell_off_platform)
    out_of_bounds = DoneTerm(func=out_of_bounds)
    success = DoneTerm(func=on_target_platform)


# --------------------------
# Enhanced commands
# --------------------------
@configclass
class UnitreeGo2JumpCommandsCfg:
    """Variable forward velocity commands for diverse training."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 8.0),
        rel_standing_envs=0.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(2.0, 4.0),  # Strong forward bias toward target
            lin_vel_y=(-0.1, 0.1), 
            ang_vel_z=(-0.1, 0.1),
        ),
    )


# --------------------------
# Enhanced events
# --------------------------
@configclass
class UnitreeGo2JumpEventsCfg:
    """Events for robot reset and positioning."""

    reset_robot_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                # Spawn on start platform, biased toward jump direction
                "x": (-PLATFORM_LENGTH/3.0, PLATFORM_LENGTH/4.0),
                "y": (-PLATFORM_WIDTH/3.0, PLATFORM_WIDTH/3.0),
                "yaw": (-0.2, 0.2),
            },
            "velocity_range": {
                "x": (0.0, 1.0),  # Small initial forward velocity toward target
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0), 
            "asset_cfg": SceneEntityCfg("robot")
        },
    )


# --------------------------
# Final environment configs
# --------------------------
@configclass
class UnitreeGo2JumpEnvCfg(ManagerBasedRLEnvCfg):
    """Training configuration for Go2 jump table with multiple robots per environment."""

    # Use more environments to get equivalent of 20x robots (1024 envs = ~20k total robots)
    scene: UnitreeGo2JumpSceneCfg = UnitreeGo2JumpSceneCfg(num_envs=1024, env_spacing=12.0)
    rewards: UnitreeGo2JumpRewardsCfg = UnitreeGo2JumpRewardsCfg()
    terminations: UnitreeGo2JumpTerminationsCfg = UnitreeGo2JumpTerminationsCfg()
    commands: UnitreeGo2JumpCommandsCfg = UnitreeGo2JumpCommandsCfg()
    events: UnitreeGo2JumpEventsCfg = UnitreeGo2JumpEventsCfg()
    actions: Go2BaseActionsCfg = Go2BaseActionsCfg()
    observations: UnitreeGo2JumpObservationsCfg = UnitreeGo2JumpObservationsCfg()

    def __post_init__(self):
        # Simulation parameters
        self.decimation = 4
        self.episode_length_s = 10.0  # Longer episodes for jumping
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        # Enhanced physics for jumping
        self.sim.physics_material = RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply", 
            static_friction=1.2,
            dynamic_friction=1.0,
        )

        # Robot initial position on start platform
        self.scene.robot.init_state.pos = (START_PLATFORM_POS[0], START_PLATFORM_POS[1], PLATFORM_HEIGHT + 0.35)


@configclass
class UnitreeGo2JumpEnvCfg_PLAY(UnitreeGo2JumpEnvCfg):
    """Play variant for testing and visualization."""

    def __post_init__(self):
        super().__post_init__()
        # Better camera angle for viewing
        self.viewer.eye = (12.0, -3.0, 6.0) 
        self.viewer.lookat = (4.0, 0.0, 1.0)
        self.scene.num_envs = 64  # Fewer for visualization
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (2.5, 3.5)  # Consistent forward speed
        self.commands.base_velocity.rel_standing_envs = 0.0
