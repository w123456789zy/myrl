"""MDP building blocks for Genesis-based environments.

Mirrors the structure of ``unitree_rl_lab.tasks.locomotion.mdp`` so that the
myLab Genesis envs can re-use the same set of reward / termination / event /
command / curriculum / observation functions that were originally written
for Isaac Lab.

All public callables take the *env* as their first argument. The env is
expected to expose the same attribute surface as :class:`envs.genesis.go2_base.Go2BaseEnv`
(``base_lin_vel``, ``base_ang_vel``, ``projected_gravity``, ``dof_pos``,
``dof_vel``, ``last_actions``, ``actions``, ``link_contact_forces``,
``foot_positions``, ``foot_velocities``, ``base_euler``, ``commands``,
``commands_body``, ``episode_length_buf``, ``dt``, ``device``, etc.).

This indirection lets the env-side code stay declarative (env_cfg/obs_cfg/
reward_cfg/command_cfg) while the actual computation lives in pure
``torch`` functions that can be tested or re-used across envs.
"""

from envs.genesis.mdp import (
    commands,
    curriculums,
    events,
    math_utils,
    observations,
    rewards,
    terminations,
)

from envs.genesis.mdp.commands import UniformLevelVelocityCommand
from envs.genesis.mdp.curriculums import (
    lin_vel_cmd_levels,
    ang_vel_cmd_levels,
    terrain_levels_vel,
)
from envs.genesis.mdp.events import (
    push_by_setting_velocity,
    randomize_rigid_body_material,
    randomize_rigid_body_mass,
    reset_root_state_uniform,
    reset_joints_by_scale,
)
from envs.genesis.mdp.observations import (
    base_ang_vel,
    base_lin_vel,
    projected_gravity,
    joint_pos_rel,
    joint_vel_rel,
    height_scan,
    last_action,
    generated_commands,
    gait_phase,
)
from envs.genesis.mdp.rewards import (
    track_lin_vel_xy_exp,
    track_ang_vel_z_exp,
    lin_vel_z_l2,
    ang_vel_xy_l2,
    joint_vel_l2,
    joint_acc_l2,
    joint_torques_l2,
    action_rate_l2,
    joint_pos_limits,
    energy,
    flat_orientation_l2,
    joint_position_penalty,
    feet_air_time,
    air_time_variance_penalty,
    feet_slide,
    undesired_contacts,
    feet_stumble,
    feet_too_near,
    feet_contact_without_cmd,
    feet_gait,
    feet_height_body,
    foot_clearance_reward,
    stand_still,
    upward,
)
from envs.genesis.mdp.terminations import (
    time_out,
    illegal_contact,
    bad_orientation,
    root_height_below_minimum,
)

__all__ = [
    # math
    "math_utils",
    # commands
    "commands",
    "UniformLevelVelocityCommand",
    # curriculums
    "curriculums",
    "lin_vel_cmd_levels",
    "ang_vel_cmd_levels",
    "terrain_levels_vel",
    # events
    "events",
    "push_by_setting_velocity",
    "randomize_rigid_body_material",
    "randomize_rigid_body_mass",
    "reset_root_state_uniform",
    "reset_joints_by_scale",
    # observations
    "observations",
    "base_ang_vel",
    "base_lin_vel",
    "projected_gravity",
    "joint_pos_rel",
    "joint_vel_rel",
    "height_scan",
    "last_action",
    "generated_commands",
    "gait_phase",
    # rewards
    "rewards",
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "lin_vel_z_l2",
    "ang_vel_xy_l2",
    "joint_vel_l2",
    "joint_acc_l2",
    "joint_torques_l2",
    "action_rate_l2",
    "joint_pos_limits",
    "energy",
    "flat_orientation_l2",
    "joint_position_penalty",
    "feet_air_time",
    "air_time_variance_penalty",
    "feet_slide",
    "undesired_contacts",
    "feet_stumble",
    "feet_too_near",
    "feet_contact_without_cmd",
    "feet_gait",
    "feet_height_body",
    "foot_clearance_reward",
    "stand_still",
    "upward",
    # terminations
    "terminations",
    "time_out",
    "illegal_contact",
    "bad_orientation",
    "root_height_below_minimum",
]
