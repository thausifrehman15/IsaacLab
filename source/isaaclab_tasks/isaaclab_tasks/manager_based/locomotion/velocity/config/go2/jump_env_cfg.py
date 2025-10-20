import torch
import math

from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg 
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg,
)
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.envs.mdp.rewards import ang_vel_xy_l2

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ActionsCfg as Go2BaseActionsCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG as GO2_CFG
from isaaclab.assets import AssetBaseCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs.mdp import reset_scene_to_default


@configclass
class Go2JumpSceneCfg(InteractiveSceneCfg):
    """Scene for jump-while-running task."""
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=GroundPlaneCfg())
    robot: ArticulationCfg = GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=GO2_CFG.init_state.replace(pos=(0.0, 0.0, 0.34)),
    )
    feet_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*_foot", track_air_time=True)
    base_contact  = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/base")


@configclass
class Go2JumpObservationsCfg:
    """Observation terms for the Go2 jump environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""
        base_lin_vel      = ObsTerm(func="isaaclab.envs.mdp.observations:base_lin_vel")
        base_ang_vel      = ObsTerm(func="isaaclab.envs.mdp.observations:base_ang_vel")
        projected_gravity = ObsTerm(func="isaaclab.envs.mdp.observations:projected_gravity")
        velocity_commands = ObsTerm(func="isaaclab.envs.mdp.observations:generated_commands", params={"command_name":"base_velocity"})
        dof_pos           = ObsTerm(func="isaaclab.envs.mdp.observations:joint_pos_rel")
        last_action       = ObsTerm(func="isaaclab.envs.mdp.observations:last_action")

        def __post_init__(self):
            super().__post_init__()
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class Go2JumpRewardsCfg:
    track_lin_vel_xy = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"std":0.5, "command_name": "base_velocity"})
    track_ang_vel_z = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.75, params={"std":0.5, "command_name": "base_velocity"})

    lin_vel_z_reward = RewTerm(func=mdp.lin_vel_z_l2, weight=2.0)
    feet_air_time = RewTerm(func=mdp.feet_air_time, weight=1.0,
                            params={"sensor_cfg":SceneEntityCfg("feet_contacts"), "command_name": "base_velocity", "threshold":1.0})
    # Regularization
    action_rate    = RewTerm(func=mdp.action_rate_l2, weight=-0.05) 
    dof_torques    = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    ang_vel_xy_penalty = RewTerm(func=ang_vel_xy_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class Go2JumpTerminationsCfg:
    time_out      = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact  = DoneTerm(func=mdp.illegal_contact,
                             params={"sensor_cfg":SceneEntityCfg("base_contact"), "threshold":1.0})


@configclass
class Go2JumpCommandsCfg:
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=True,
        resampling_time_range=(4.0,8.0),
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5,1.5),
            lin_vel_y=(-0.5,0.5),
            ang_vel_z=(-1.0,1.0),
            heading=(-math.pi,math.pi),
        ),
    )


@configclass
class Go2JumpEventsCfg:
    """Event terms for the Go2 locomotion environment."""
    reset_scene = EventTerm(func=reset_scene_to_default, mode="reset")
    physics_randomization = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="interval",
        interval_range_s=(10.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.0), 
            "num_buckets": 64, 
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class UnitreeGo2JumpEnvCfg(ManagerBasedRLEnvCfg):
    """Jump-while-running environment."""
    decimation        = 4
    episode_length_s  = 20.0

    scene             = Go2JumpSceneCfg(num_envs=4096, env_spacing=4.0)
    rewards           = Go2JumpRewardsCfg()
    terminations      = Go2JumpTerminationsCfg()
    commands          = Go2JumpCommandsCfg()
    events            = Go2JumpEventsCfg()
    actions           = Go2BaseActionsCfg()
    observations      = Go2JumpObservationsCfg()


    def __post_init__(self):
        self.sim.dt             = 0.005
        self.sim.render_interval= self.decimation
        self.sim.physics_material=RigidBodyMaterialCfg(static_friction=1.0,
                                                      dynamic_friction=1.0,
                                                      restitution=0.0)

        self.actions.joint_pos.scale = 0.3


@configclass
class UnitreeGo2JumpEnvCfg_PLAY(UnitreeGo2JumpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs     = 50
        self.observations.policy.enable_corruption = False
        self.events.physics_randomization   = None
        self.events.push_robot             = None
        self.commands.base_velocity.resampling_time_range=(2.0,5.0)
        self.actions.joint_pos.scale       = 0.35