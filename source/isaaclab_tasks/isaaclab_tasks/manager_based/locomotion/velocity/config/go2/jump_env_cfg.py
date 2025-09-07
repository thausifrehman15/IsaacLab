# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for a Unitree Go2 jumping task.

The robot starts on a green platform and must learn to jump over a gap
to land on a red platform.
"""

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.materials import PreviewSurfaceCfg, RigidBodyMaterialCfg
from isaaclab.sim.spawners.shapes import CuboidCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
import isaaclab.utils.math as math_utils

# Import the base configuration classes we will inherit from
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg as Go2BaseActionsCfg,
    ObservationsCfg as Go2BaseObservationsCfg,
    RewardsCfg as Go2BaseRewardsCfg,
    TerminationsCfg as Go2BaseTerminationsCfg,
)

# -- Hardcoded Scene Layout --
PLATFORM_LENGTH = 4.0
PLATFORM_WIDTH = 4.0
PLATFORM_HEIGHT = 0.25
START_PLATFORM_POS = (0.0, 0.0, PLATFORM_HEIGHT / 2.0)
TARGET_PLATFORM_POS = (4.0, 0.0, PLATFORM_HEIGHT / 2.0)
JUMP_GAP_DISTANCE = TARGET_PLATFORM_POS[0] - START_PLATFORM_POS[0] - PLATFORM_LENGTH

# -- Custom Reward & Termination Functions (Vectorized) --

def approach_target(env: ManagerBasedRLEnvCfg, std: float = 2.0) -> torch.Tensor:
    """Reward for approaching the jump-off edge of the platform."""
    target_x = START_PLATFORM_POS[0] + PLATFORM_LENGTH / 2.0
    distance_to_edge = torch.abs(target_x - env.scene["robot"].data.root_pos_w[:, 0])
    return torch.exp(-distance_to_edge / std)

def maintain_forward_direction(env: ManagerBasedRLEnvCfg) -> torch.Tensor:
    """Reward for facing the target platform (positive x-direction)."""
    forward_vec_world = math_utils.quat_apply(
        env.scene["robot"].data.root_quat_w, env.scene["robot"].data.FORWARD_VEC_B
    )
    return torch.exp(forward_vec_world[:, 0])

def forward_world_velocity(env: ManagerBasedRLEnvCfg) -> torch.Tensor:
    """Reward for moving forward in the world's positive x-direction."""
    # Use world frame velocity (_w) instead of body frame (_b)
    return torch.clamp(env.scene["robot"].data.root_lin_vel_w[:, 0], min=0.0)

def forward_velocity(env: ManagerBasedRLEnvCfg) -> torch.Tensor:
    """Reward for moving forward."""
    return torch.clamp(env.scene["robot"].data.root_lin_vel_b[:, 0], min=0.0)

def approach_and_align_reward(env: ManagerBasedRLEnvCfg) -> torch.Tensor:
    """Combined reward for approaching, aligning, and moving forward."""
    return approach_target(env) * maintain_forward_direction(env) * forward_world_velocity(env)

def is_airborne_over_gap(env: ManagerBasedRLEnvCfg) -> torch.Tensor:
    """Reward for being airborne over the gap between platforms."""
    robot_pos = env.scene["robot"].data.root_pos_w
    in_gap_x = (robot_pos[:, 0] > START_PLATFORM_POS[0] + PLATFORM_LENGTH / 2.0) & (
        robot_pos[:, 0] < TARGET_PLATFORM_POS[0] - PLATFORM_LENGTH / 2.0
    )
    latest_forces = env.scene["contact_forces"].data.net_forces_w_history[:, 0, :, :]
    in_air = torch.all(torch.sum(latest_forces, dim=-1).abs() < 1.0, dim=-1)
    return (in_air & in_gap_x).float()

def penalize_turning(env: ManagerBasedRLEnvCfg) -> torch.Tensor:
    """Penalizes the robot for turning by squaring its angular z-velocity."""
    # Extract the angular velocity around the z-axis (in the robot's body frame)
    ang_vel_z = env.scene["robot"].data.root_ang_vel_b[:, 2]
    # Square the velocity to create a penalty that is always positive
    # and grows quadratically with the turning speed.
    return torch.square(ang_vel_z)

def landed_on_target(env: ManagerBasedRLEnvCfg) -> torch.Tensor:
    """Boolean check for successfully landing on the target platform."""
    robot_pos = env.scene["robot"].data.root_pos_w
    on_target_x = (robot_pos[:, 0] > 3.0) & (robot_pos[:, 0] < 5.0) 
    on_target_y = (robot_pos[:, 1] > -1.5) & (robot_pos[:, 1] < 1.5) 
    latest_forces = env.scene["contact_forces"].data.net_forces_w_history[:, 0, :, :]
    in_contact = torch.any(torch.sum(latest_forces, dim=-1).abs() > 1.0, dim=-1)
    # -- Add printing logic --
    should_terminate = on_target_x & on_target_y & in_contact
    terminating_env_ids = torch.where(should_terminate)[0]
    print_robot_position(env, terminating_env_ids, event_type="SUCCESS-TERMINATION")
    return on_target_x & on_target_y & in_contact

def fell_off_start_platform(env: ManagerBasedRLEnvCfg) -> torch.Tensor:
    """Terminate if the robot falls off the starting platform."""
    robot_pos = env.scene["robot"].data.root_pos_w
    on_start_side = robot_pos[:, 0] < START_PLATFORM_POS[0] + PLATFORM_LENGTH / 2.0 + 0.5
    is_low = robot_pos[:, 2] < PLATFORM_HEIGHT * 0.8
    # -- Add printing logic --
    should_terminate = on_start_side & is_low
    terminating_env_ids = torch.where(should_terminate)[0]
    print_robot_position(env, terminating_env_ids, event_type="FELL-OFF-TERMINATION")
    return on_start_side & is_low

def reset_robot_to_start(env: ManagerBasedRLEnvCfg, env_ids: torch.Tensor):
    """Explicitly resets the robot to a random position on its own starting platform."""
    if len(env_ids) == 0:
        return
    # Get the origins of the environments that are resetting
    env_origins = env.scene.env_origins[env_ids]
    # Define the reset boundaries on the starting platform (local coordinates)
    x_min, x_max = -PLATFORM_LENGTH / 4, -PLATFORM_LENGTH / 4 + 1.0
    y_min, y_max = -PLATFORM_WIDTH / 4, PLATFORM_WIDTH / 4
    num_resets = len(env_ids)
    # Get the root state tensor for the robots that are resetting
    root_states = env.scene["robot"].data.root_state_w[env_ids]
    # -- Manually set the new state in the WORLD frame --
    # Generate random local positions
    local_x = x_min + (x_max - x_min) * torch.rand(num_resets, device=root_states.device)
    local_y = y_min + (y_max - y_min) * torch.rand(num_resets, device=root_states.device)
    # Add the local positions to the environment origins to get the correct world position
    root_states[:, 0] = env_origins[:, 0] + local_x
    root_states[:, 1] = env_origins[:, 1] + local_y
    # Set a fixed, stable height above the platform (also relative to the origin)
    root_states[:, 2] = env_origins[:, 2] + PLATFORM_HEIGHT + 0.32
    # Reset orientation to default (no rotation)
    # Quaternion (w, x, y, z) -> [1, 0, 0, 0]
    root_states[:, 3] = 1.0
    root_states[:, 4:7] = 0.0
    # Reset linear and angular velocities to zero
    root_states[:, 7:13] = 0.0
    # Write the new state back to the simulation for the correct robots
    env.scene["robot"].write_root_state_to_sim(root_states, env_ids)

def penalize_falling(env: ManagerBasedRLEnvCfg) -> torch.Tensor:
    """Returns a penalty when the robot is in a state that would terminate for falling."""
    return fell_off_start_platform(env).float()

###########logging

def print_robot_position(env: ManagerBasedRLEnvCfg, env_ids: torch.Tensor, event_type: str):
    """Prints the position of specified robots."""
    if len(env_ids) == 0:
        return
    positions = env.scene["robot"].data.root_pos_w[env_ids].cpu().numpy()
    for i, env_id in enumerate(env_ids):
        pos = positions[i]
        print(f"[Env {env_id.item()}] {event_type} at position: x={pos[0]:.2f}, y={pos[1]:.2f}, z={pos[2]:.2f}")

def log_reset_location(env: ManagerBasedRLEnvCfg, env_ids: torch.Tensor):
    """Event function to log the robot's position upon reset."""
    print_robot_position(env, env_ids, event_type="RESET")

# -- Configuration Classes --

@configclass
class UnitreeGo2JumpSceneCfg(InteractiveSceneCfg):
    """Scene with two platforms for the jump task."""
    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_foot", update_period=0.0, history_length=6)
    base_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/base")
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    start_platform = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/StartPlatform",
        spawn=CuboidCfg(
            size=(PLATFORM_LENGTH, PLATFORM_WIDTH, PLATFORM_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.3, 0.8, 0.3)),
            visual_material_path="/World/Looks/green_material",
            physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=START_PLATFORM_POS),
    )
    target_platform = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetPlatform",
        spawn=CuboidCfg(
            size=(PLATFORM_LENGTH, PLATFORM_WIDTH, PLATFORM_HEIGHT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.3)),
            visual_material_path="/World/Looks/red_material",
            physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TARGET_PLATFORM_POS),
    )

@configclass
class UnitreeGo2JumpObservationsCfg(Go2BaseObservationsCfg):
    """A minimal observation space for the jumping task."""
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        actions = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class UnitreeGo2JumpRewardsCfg(Go2BaseRewardsCfg):
    """Reward terms for the jumping task."""
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6)
    penalize_turning = RewTerm(func=penalize_turning, weight=-1.5)
    approach_and_align = RewTerm(func=approach_and_align_reward, weight=10.0)
    airborne_bonus = RewTerm(func=is_airborne_over_gap, weight=5.0)
    landing_bonus = RewTerm(func=landed_on_target, weight=25.0)
    fell_off_penalty = RewTerm(func=penalize_falling, weight=-20.0)
    track_lin_vel_xy_exp = None
    feet_air_time = None
    undesired_contacts = None

@configclass
class UnitreeGo2JumpTerminationsCfg(Go2BaseTerminationsCfg):
    """Termination conditions for the jumping task."""
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("base_contact"), "threshold": 1.0},
    )
    fell_off = DoneTerm(func=fell_off_start_platform)
    success = DoneTerm(func=landed_on_target)

@configclass
class UnitreeGo2JumpCommandsCfg:
    """Command terms for the jumping task."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(1.5, 1.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )

@configclass
class UnitreeGo2JumpEventsCfg:
    """Event terms for the jumping task."""
    reset_robot_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.5, 0.5),
                "y": (-1.5, 1.5),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # reset_robot_position = EventTerm(
    #   func=reset_robot_to_start,
    #   mode="reset",
    # )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot")},
    )
    log_on_reset = EventTerm(
        func=log_reset_location,
        mode="reset",
    )

@configclass
class UnitreeGo2JumpEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Unitree Go2 jumping environment."""
    scene: UnitreeGo2JumpSceneCfg = UnitreeGo2JumpSceneCfg(num_envs=10, env_spacing=10.0)
    rewards: UnitreeGo2JumpRewardsCfg = UnitreeGo2JumpRewardsCfg()
    terminations: UnitreeGo2JumpTerminationsCfg = UnitreeGo2JumpTerminationsCfg()
    commands: UnitreeGo2JumpCommandsCfg = UnitreeGo2JumpCommandsCfg()
    events: UnitreeGo2JumpEventsCfg = UnitreeGo2JumpEventsCfg()
    actions: Go2BaseActionsCfg = Go2BaseActionsCfg()
    observations: UnitreeGo2JumpObservationsCfg = UnitreeGo2JumpObservationsCfg()
    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
        self.scene.robot.init_state.pos = (-1.0, 0.0, PLATFORM_HEIGHT + 0.32)

@configclass
class UnitreeGo2JumpEnvCfg_PLAY(UnitreeGo2JumpEnvCfg):
    """Configuration for the play-time Unitree Go2 jumping environment."""
    def __post_init__(self):
        super().__post_init__()
        self.viewer.eye = (10.0, 0.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)
        self.scene.num_envs = 2
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.ranges.lin_vel_x = (2.0, 2.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
