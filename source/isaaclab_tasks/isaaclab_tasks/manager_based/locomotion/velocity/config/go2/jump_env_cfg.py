# /opt/isaaclab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/jump_env_cfg.py

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sim.spawners.shapes import CuboidCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.sim.schemas import (
    RigidBodyPropertiesCfg,
    MassPropertiesCfg,
    CollisionPropertiesCfg,
)
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.shapes import CuboidCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import PreviewSurfaceCfg
from isaaclab.sim.schemas import (
    RigidBodyPropertiesCfg,
    MassPropertiesCfg,
    CollisionPropertiesCfg,
)
from dataclasses import MISSING
from isaaclab.managers import EventTermCfg as EventTerm


#
# Custom reward/termination functions
#
def is_airborne(env) -> float:
    """Return 1.0 if robot is fully airborne, else 0.0"""
    return float((env.contact_forces.sum(dim=-1) < 1e-3).all())


def landed_on_target(env) -> float:
    """Return 1.0 if robot COM is inside landing platform bounds, else 0.0"""
    base_pos = env.robot.data.root_pos_w
    x, y, z = base_pos[0], base_pos[1], base_pos[2]
    return float((5.0 < x < 7.0) and (-2.0 < y < 2.0) and (z > 0.2))


def fell_off(env) -> bool:
    """Terminate if robot falls below ground level"""
    return env.robot.data.root_pos_w[2] < 0.2


def missed_jump(env) -> bool:
    """Terminate if robot overshoots X > 8.0 without being on target platform"""
    base_pos = env.robot.data.root_pos_w
    return (base_pos[0] > 8.0) and not landed_on_target(env)

def approach_target_x(env) -> float:
    """Reward for decreasing the distance to the target in the x-direction."""
    return -abs(env.robot.data.root_pos_w[:, 0] - 6.0)


def track_y_zero(env) -> float:
    """Penalize for deviating from the y=0 line."""
    return -abs(env.robot.data.root_pos_w[:, 1])


def maintain_forward_velocity(env) -> float:
    """Reward for maintaining a positive forward velocity."""
    return env.robot.data.root_lin_vel_w[:, 0]


#
# Env Config
#
@configclass
class UnitreeGo2JumpEnvCfg(UnitreeGo2FlatEnvCfg):
    """Jump env: two cuboid platforms with a gap in between."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.env_spacing = 10.0
        platform_size = (4.0, 1.0, 0.5)
        self.scene.num_envs = 15
        # Start platform
        cuboid_start = RigidObjectCfg(
            prim_path="/World/envs/env_.*/StartCuboid",
            spawn=CuboidCfg(
                size=platform_size,
                rigid_props=RigidBodyPropertiesCfg(),
                mass_props=MassPropertiesCfg(mass=200.0),
                collision_props=CollisionPropertiesCfg(),
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.3, 0.8, 0.3)),
                physics_material=RigidBodyMaterialCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)),
        )

        # Landing platform
        cuboid_target = RigidObjectCfg(
            prim_path="/World/envs/env_.*/TargetCuboid",
            spawn=CuboidCfg(
                size=platform_size,
                rigid_props=RigidBodyPropertiesCfg(),
                mass_props=MassPropertiesCfg(mass=200.0),
                collision_props=CollisionPropertiesCfg(),
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.3)),
                physics_material=RigidBodyMaterialCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(6.0, 0.0, 0.25)),
        )

        # Attach platforms to scene
        if not hasattr(self.scene, "start_platform"):
            setattr(self.scene, "start_platform", MISSING)
        self.scene.start_platform = cuboid_start

        if not hasattr(self.scene, "target_platform"):
            setattr(self.scene, "target_platform", MISSING)
        self.scene.target_platform = cuboid_target 
        self.events.reset_robot = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.9, 0.9), "yaw": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

#
# Rewards
#
@configclass
class UnitreeGo2JumpRewardsCfg:
    """Rewards for the jump task."""

    # Base rewards from flat terrain config
    # These encourage stable locomotion
    lin_vel_z = RewTerm(func="mdp.track_lin_vel_z_l2", weight=-2.0, params={"command_name": "base_velocity"})
    ang_vel_xy = RewTerm(func="mdp.track_ang_vel_xy_l2", weight=-0.05, params={"command_name": "base_velocity"})
    dof_acc = RewTerm(func="mdp.dof_acc_l2", weight=-2.5e-7)
    action_rate = RewTerm(func="mdp.action_rate_l2", weight=-0.01)
    feet_air_time = RewTerm(
        func="mdp.feet_air_time",
        weight=0.125,
        params={"command_name": "base_velocity", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 0.5},
    )

    # Task-specific rewards
    # Encourage running towards the target
    approach_target = RewTerm(func=approach_target_x, weight=1.5)
    track_y = RewTerm(func=track_y_zero, weight=-0.5)
    forward_velocity = RewTerm(func=maintain_forward_velocity, weight=0.5)

    # Encourage jumping and landing
    jump_air_time = RewTerm(
        func=is_airborne,
        weight=0.2,
    )
    landing_success = RewTerm(
        func=landed_on_target,
        weight=20.0,
    )

#
# Terminations
#
@configclass
class UnitreeGo2JumpTerminationsCfg:
    time_out = DoneTerm(func="mdp.time_out", time_out=True)
    fall_off = DoneTerm(func=fell_off)
    missed = DoneTerm(func=missed_jump)

#
# Final EnvCfg class
#
@configclass
class UnitreeGo2JumpEnvCfg_PLAY(UnitreeGo2JumpEnvCfg):
    rewards: UnitreeGo2JumpRewardsCfg = UnitreeGo2JumpRewardsCfg()
    terminations: UnitreeGo2JumpTerminationsCfg = UnitreeGo2JumpTerminationsCfg()
