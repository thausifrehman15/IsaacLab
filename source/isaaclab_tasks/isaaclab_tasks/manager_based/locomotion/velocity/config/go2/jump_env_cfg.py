# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for a Unitree Go2 jumping task with curriculum learning.
The robot starts on a green platform and must learn to jump over a gap
to land on a red platform. Curriculum starts with small gap for walking.
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
INITIAL_TARGET_PLATFORM_POS = (PLATFORM_LENGTH, 0.0, PLATFORM_HEIGHT / 2.0)  # Start with zero gap

# -- Custom Reward & Termination Functions (Vectorized) --
def approach_target(env: ManagerBasedRLEnvCfg, std: float = 2.0) -> torch.Tensor:
    """Reward for approaching the target platform."""
    target_x = env.scene["target_platform"].data.root_pos_w[:, 0]
    robot_x = env.scene["robot"].data.root_pos_w[:, 0]
    distance_to_target = torch.abs(target_x - robot_x)
    return torch.exp(-distance_to_target / std)

def landed_on_target(env: ManagerBasedRLEnvCfg) -> torch.Tensor:
    """Fixed: Strict target platform bounds."""
    robot_pos = env.scene["robot"].data.root_pos_w
    target_pos = env.scene["target_platform"].data.root_pos_w
    on_target_x = (robot_pos[:, 0] > target_pos[:, 0] - PLATFORM_LENGTH / 2.0) & (
        robot_pos[:, 0] < target_pos[:, 0] + PLATFORM_LENGTH / 2.0
    )
    on_target_y = (robot_pos[:, 1] > target_pos[:, 1] - PLATFORM_WIDTH / 2.0) & (
        robot_pos[:, 1] < target_pos[:, 1] + PLATFORM_WIDTH / 2.0
    )
    latest_forces = env.scene["contact_forces"].data.net_forces_w_history[:, 0, :, :]
    in_contact = torch.any(torch.sum(latest_forces, dim=-1).abs() > 1.0, dim=-1)
    should_terminate = on_target_x & on_target_y & in_contact
    return should_terminate

def fell_off_start_platform(env: ManagerBasedRLEnvCfg) -> torch.Tensor:
    """Terminate if the robot falls off the starting platform."""
    robot_pos = env.scene["robot"].data.root_pos_w
    start_pos = env.scene["start_platform"].data.root_pos_w
    on_start_side = robot_pos[:, 0] < start_pos[:, 0] + PLATFORM_LENGTH / 2.0 + 0.5
    is_low = robot_pos[:, 2] < PLATFORM_HEIGHT * 0.8
    should_terminate = on_start_side & is_low
    return should_terminate

def penalize_standing_still(env: ManagerBasedRLEnvCfg) -> torch.Tensor:
    """Penalty for standing still (low velocity)."""
    lin_vel = env.scene["robot"].data.root_lin_vel_w[:, 0]
    return -torch.clamp(lin_vel.abs(), max=0.1)  # Negative if speed < 0.1 m/s

def reset_robot_to_start(env: ManagerBasedRLEnvCfg, env_ids: torch.Tensor):
    """Explicitly resets the robot to a random position on its own starting platform."""
    if len(env_ids) == 0:
        return
    env_origins = env.scene.env_origins[env_ids]
    x_min, x_max = -PLATFORM_LENGTH / 4, -PLATFORM_LENGTH / 4 + 1.0
    y_min, y_max = -PLATFORM_WIDTH / 4, PLATFORM_WIDTH / 4
    num_resets = len(env_ids)
    root_states = env.scene["robot"].data.root_state_w[env_ids]
    local_x = x_min + (x_max - x_min) * torch.rand(num_resets, device=root_states.device)
    local_y = y_min + (y_max - y_min) * torch.rand(num_resets, device=root_states.device)
    root_states[:, 0] = env_origins[:, 0] + local_x
    root_states[:, 1] = env_origins[:, 1] + local_y
    root_states[:, 2] = env_origins[:, 2] + PLATFORM_HEIGHT + 0.32
    root_states[:, 3] = 1.0
    root_states[:, 4:7] = 0.0
    root_states[:, 7:13] = 0.0
    env.scene["robot"].write_root_state_to_sim(root_states, env_ids)

# NEW: Function to adjust gap based on curriculum (step-based)
def adjust_gap(env: ManagerBasedRLEnvCfg, env_ids: torch.Tensor, threshold_steps: int, increment: float, max_gap: float, initial_gap: float):
    if len(env_ids) == 0:
        return
    level = env.common_step_counter // threshold_steps
    gap = initial_gap + level * increment
    gap = min(gap, max_gap)
    target_state = env.scene["target_platform"].data.root_state_w.clone()
    start_pos = env.scene["start_platform"].data.root_pos_w[:, 0]
    target_state[:, 0] = start_pos + PLATFORM_LENGTH + gap
    env.scene["target_platform"].write_root_state_to_sim(target_state, torch.arange(env.num_envs, device=env.device))

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
        init_state=RigidObjectCfg.InitialStateCfg(pos=INITIAL_TARGET_PLATFORM_POS),
    )

@configclass
class UnitreeGo2JumpObservationsCfg(Go2BaseObservationsCfg):
    """A minimal observation space for the jumping task."""
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))  # CHANGED: Add noise for robustness
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
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
    """Enhanced reward terms for the jumping task."""
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-7) 
    
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)  # Add this line
    
    # Re-enable velocity tracking to encourage forward movement
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", 
                "std": 0.25,
                "asset_cfg": SceneEntityCfg("robot")},
    ) 

    # Multi-stage jumping rewards with higher weights
    approach_target = RewTerm(func=approach_target, weight=15.0)  # CHANGED: Boost
    landing_bonus = RewTerm(func=landed_on_target, weight=200.0) 
    
    standing_still_penalty = RewTerm(func=penalize_standing_still, weight=-0.1)

    fell_off_penalty = RewTerm(func=fell_off_start_platform, weight=-5.0)
    
    # Disable conflicting rewards
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
        resampling_time_range=(6.0, 8.0),
        rel_standing_envs=0.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 1.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )

@configclass
class UnitreeGo2JumpEventsCfg:
    """Event terms for the jumping task."""
    reset_robot_position = EventTerm(
        func=reset_robot_to_start,
        mode="reset",
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot")},
    )
    curriculum_gap = EventTerm(
        func=adjust_gap,
        mode="interval",
        params={
            "threshold_steps": 5000000,
            "increment": 0.1,
            "max_gap": 2.0,
            "initial_gap": 0.0,
        },
        interval_range_s=(5.0, 5.0),
    )

@configclass
class UnitreeGo2JumpEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Unitree Go2 jumping environment."""
    decimation = 4
    episode_length_s = 15.0

    scene = UnitreeGo2JumpSceneCfg(num_envs=4000, env_spacing=12.0)
    rewards = UnitreeGo2JumpRewardsCfg()
    terminations = UnitreeGo2JumpTerminationsCfg()
    commands = UnitreeGo2JumpCommandsCfg()
    events = UnitreeGo2JumpEventsCfg()
    actions = Go2BaseActionsCfg()
    observations = UnitreeGo2JumpObservationsCfg()

    def __post_init__(self):
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,
            dynamic_friction=1.0,
        )
        self.scene.robot.init_state.pos = (-1.5, 0.0, PLATFORM_HEIGHT + 0.4)

@configclass
class UnitreeGo2JumpEnvCfg_PLAY(UnitreeGo2JumpEnvCfg):
    """Configuration for the play-time Unitree Go2 jumping environment."""
    def __post_init__(self):
        super().__post_init__()
        self.viewer.eye = (10.0, 0.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False
        self.events.push_robot_interval = None
        self.events.randomize_base_mass = None