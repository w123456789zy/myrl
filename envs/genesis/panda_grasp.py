# Copyright (c) 2025, Your Name
# All rights reserved.

from __future__ import annotations

import math
from typing import Literal

import genesis as gs
import torch
from genesis.utils.geom import transform_by_quat, transform_quat_by_quat, xyz_to_quat

from mylab.env.vec_env import VecEnv, VecEnvObs

"""
https://github.com/Genesis-Embodied-AI/Genesis/tree/0119348c47c9dcf29276af1ec383f50c5c0ade15/examples/manipulation
Adapted to mylab/envs/vec_env format.
"""


class FrankaPandaGraspEnv(VecEnv):
    """Franka Panda grasping environment, adapted to VecEnv interface."""

    def __init__(self, num_envs: int, show_viewer: bool = False) -> None:
        self.name = "panda_grasp"
        self.num_envs = num_envs
        self._env_cfg, self._obs_cfg, self._reward_cfg, self._robot_cfg = self._default_configs()

        self.num_obs = self._obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = self._env_cfg["num_actions"]
        self.device = gs.device

        self.ctrl_dt = self._env_cfg["ctrl_dt"]
        self.max_episode_length = math.ceil(self._env_cfg["episode_length_s"] / self.ctrl_dt)

        self.action_scales = torch.tensor(self._env_cfg["action_scales"], device=self.device)

        # setup scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.ctrl_dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(min(10, num_envs)))),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.Plane())

        self.robot = Manipulator(
            num_envs=self.num_envs, scene=self.scene, args=self._robot_cfg, device=gs.device
        )

        self.object = self.scene.add_entity(
            gs.morphs.Box(
                size=self._env_cfg["box_size"],
                fixed=self._env_cfg["box_fixed"],
                collision=self._env_cfg["box_collision"],
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(1.0, 0.0, 0.0)),
            ),
        )

        if self._env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(1280, 720), pos=(1.5, 0.0, 0.2), lookat=(0, 0, 0.2), fov=50, GUI=True
            )

        self.scene.build(n_envs=num_envs)
        self.robot.set_pd_gains()

        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self._reward_cfg.keys():
            self._reward_cfg[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        self.keypoints_offset = self.get_keypoint_offsets(batch_size=self.num_envs, device=self.device, unit_length=0.5)
        self._init_buffers()

    @classmethod
    def _default_configs(cls):
        env_cfg = {
            "num_actions": 6,
            "action_scales": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
            "action_range": 1.0,
            "episode_length_s": 3.0,
            "ctrl_dt": 0.01,
            "box_size": [0.04, 0.04, 0.06],
            "box_collision": False,
            "box_fixed": True,
            "visualize_camera": True,
        }
        obs_cfg = {"num_obs": 14, "num_priv_obs": None}
        reward_scales = {"keypoints": 1.0, "table_contact": -1.0}
        robot_cfg = {
            "ee_link_name": "hand",
            "gripper_link_names": ["left_finger", "right_finger"],
            "default_arm_dof": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
            "default_gripper_dof": [0.04, 0.04],
            "ik_method": "dls_ik",
        }
        return env_cfg, obs_cfg, reward_scales, robot_cfg

    def _init_buffers(self):
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=gs.device)
        self.goal_pose = torch.zeros(self.num_envs, 7, device=gs.device)
        self.extras = dict()
        self.extras["observations"] = dict()

    # ------------------------------------------------------------------
    # VecEnv abstract methods
    # ------------------------------------------------------------------

    def get_observations(self):
        finger_pos, finger_quat = (
            self.robot.center_finger_pose[:, :3], self.robot.center_finger_pose[:, 3:7]
        )
        obj_pos, obj_quat = self.object.get_pos(), self.object.get_quat()
        obs_components = [finger_pos - obj_pos, finger_quat, obj_pos, obj_quat]
        obs_tensor = torch.cat(obs_components, dim=-1)
        self.extras["observations"]["critic"] = obs_tensor
        return {"state": obs_tensor}

    def get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=gs.device, dtype=gs.tc_float)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=gs.device)
        self.reset_buf[env_ids] = True
        self.reset_idx(env_ids)
        obs = self.get_observations()
        return obs, self.extras

    def step(self, actions: torch.Tensor):
        actions = torch.clip(actions, -self._env_cfg["action_range"], self._env_cfg["action_range"])
        self.episode_length_buf += 1

        actions = self.rescale_action(actions)
        self.robot.apply_action(actions, open_gripper=True)
        self.scene.step()

        reward = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self._reward_cfg[name]
            reward += rew
            self.episode_sums[name] += rew

        env_reset_idx = self.is_episode_complete()
        final_obs = self.get_observations()
        if len(env_reset_idx) > 0:
            self.reset_idx(env_reset_idx)

        obs = self.get_observations()
        self.extras["final_observations"] = final_obs["state"]

        time_out_buf = self.episode_length_buf > self.max_episode_length
        done = time_out_buf.to(dtype=gs.tc_float)

        return obs, reward, done, self.extras

    def seed(self, seed: int = -1) -> int:
        if seed != -1:
            torch.manual_seed(seed)
        return seed

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        self.episode_length_buf[envs_idx] = 0
        self.robot.reset(envs_idx)

        num_reset = len(envs_idx)
        random_x = torch.rand(num_reset, device=self.device) * 0.4 + 0.2
        random_y = (torch.rand(num_reset, device=self.device) - 0.5) * 0.5
        random_z = torch.ones(num_reset, device=self.device) * 0.025
        random_pos = torch.stack([random_x, random_y, random_z], dim=-1)

        q_downward = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat(num_reset, 1)
        PI = 3.1415926
        random_yaw = (torch.rand(num_reset, device=self.device) * 2 * PI - PI) * 0.25
        q_yaw = torch.stack(
            [torch.cos(random_yaw / 2), torch.zeros(num_reset, device=self.device),
             torch.zeros(num_reset, device=self.device), torch.sin(random_yaw / 2)],
            dim=-1,
        )
        goal_yaw = transform_quat_by_quat(q_yaw, q_downward)
        self.goal_pose[envs_idx] = torch.cat([random_pos, goal_yaw], dim=-1)
        self.object.set_pos(random_pos, envs_idx=envs_idx)
        self.object.set_quat(goal_yaw, envs_idx=envs_idx)

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self._env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def is_episode_complete(self):
        time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = time_out_buf
        time_out_idx = time_out_buf.nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0
        if torch.any(self.reset_buf):
            self.extras["episode_length"] = (
                (self.episode_length_buf * self.reset_buf).sum() / self.reset_buf.sum()
            ).item()
        return self.reset_buf.nonzero(as_tuple=True)[0]

    def rescale_action(self, action: torch.Tensor) -> torch.Tensor:
        return action * self.action_scales

    def _reward_keypoints(self):
        keypoints_offset = self.keypoints_offset
        finger_tip_z_offset = torch.tensor([0.0, 0.0, -0.06], device=self.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        finger_pos_keypoints = self._to_world_frame(
            self.robot.center_finger_pose[:, :3] + finger_tip_z_offset,
            self.robot.center_finger_pose[:, 3:7],
            keypoints_offset,
        )
        object_pos_keypoints = self._to_world_frame(
            self.object.get_pos(), self.object.get_quat(), keypoints_offset
        )
        dist = torch.norm(finger_pos_keypoints - object_pos_keypoints, p=2, dim=-1).sum(-1)
        return torch.exp(-dist)

    def _reward_table_contact(self):
        gripper_dofs = self.robot._robot_entity.get_qpos()[:, self.robot._fingers_dof]
        expected_open_pos = torch.tensor([0.04, 0.04], device=self.device).repeat(self.num_envs, 1)
        dof_error = torch.norm(gripper_dofs - expected_open_pos, dim=-1)
        contact_threshold = 0.001
        contact_penalty = torch.where(dof_error > contact_threshold, -dof_error, torch.zeros_like(dof_error))
        return contact_penalty

    def _to_world_frame(self, position, quaternion, keypoints_offset):
        world = torch.zeros_like(keypoints_offset)
        for k in range(keypoints_offset.shape[1]):
            world[:, k] = position + transform_by_quat(keypoints_offset[:, k], quaternion)
        return world

    @staticmethod
    def get_keypoint_offsets(batch_size, device, unit_length=0.5):
        keypoint_offsets = (
            torch.tensor(
                [
                    [0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0],
                    [0, -1.0, 0], [0, 1.0, 0], [0, 0, -1.0], [0, 0, 1.0],
                ],
                device=device, dtype=torch.float32,
            )
            * unit_length
        )
        return keypoint_offsets.unsqueeze(0).repeat(batch_size, 1, 1)

    def render(self):
        frame, _, _, _ = self.cam.render()
        return frame


class Manipulator:
    def __init__(self, num_envs: int, scene: gs.Scene, args: dict, device: str = "cpu"):
        self._device = device
        self._scene = scene
        self._num_envs = num_envs
        self._args = args

        material: gs.materials.Rigid = gs.materials.Rigid()
        morph: gs.morphs.URDF = gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml", pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0))
        self._robot_entity: gs.Entity = scene.add_entity(material=material, morph=morph)

        self._ik_method: Literal["rel_pose", "dls"] = args["ik_method"]
        self._init()

    def set_pd_gains(self):
        self._robot_entity.set_dofs_kp(torch.tensor([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
        self._robot_entity.set_dofs_kv(torch.tensor([450, 450, 350, 350, 200, 200, 200, 10, 10]))
        self._robot_entity.set_dofs_force_range(
            torch.tensor([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            torch.tensor([87, 87, 87, 87, 12, 12, 12, 100, 100]),
        )

    def _init(self):
        self._arm_dof_dim = self._robot_entity.n_dofs - 2
        self._gripper_dim = 2
        self._arm_dof_idx = torch.arange(self._arm_dof_dim, device=self._device)
        self._fingers_dof = torch.arange(self._arm_dof_dim, self._arm_dof_dim + self._gripper_dim, device=self._device)
        self._left_finger_dof = self._fingers_dof[0]
        self._right_finger_dof = self._fingers_dof[1]
        self._ee_link = self._robot_entity.get_link(self._args["ee_link_name"])
        self._left_finger_link = self._robot_entity.get_link(self._args["gripper_link_names"][0])
        self._right_finger_link = self._robot_entity.get_link(self._args["gripper_link_names"][1])
        self._default_joint_angles = self._args["default_arm_dof"]
        if self._args["default_gripper_dof"] is not None:
            self._default_joint_angles += self._args["default_gripper_dof"]

    def reset(self, envs_idx: torch.IntTensor):
        if len(envs_idx) == 0:
            return
        self.reset_home(envs_idx)

    def reset_home(self, envs_idx: torch.IntTensor | None = None):
        if envs_idx is None:
            envs_idx = torch.arange(self._num_envs, device=self._device)
        default_joint_angles = torch.tensor(self._default_joint_angles, dtype=torch.float32, device=self._device).repeat(len(envs_idx), 1)
        self._robot_entity.set_qpos(default_joint_angles, envs_idx=envs_idx)

    def apply_action(self, action: torch.Tensor, open_gripper: bool) -> None:
        q_pos = self._robot_entity.get_qpos()
        if self._ik_method == "gs_ik":
            q_pos = self._gs_ik(action)
        elif self._ik_method == "dls_ik":
            q_pos = self._dls_ik(action)
        else:
            raise ValueError(f"Invalid control mode: {self._ik_method}")
        if open_gripper:
            q_pos[:, self._fingers_dof] = +0.04
        else:
            q_pos[:, self._fingers_dof] = +0.02
        self._robot_entity.control_dofs_position(position=q_pos)

    def _gs_ik(self, action: torch.Tensor) -> torch.Tensor:
        delta_position = action[:, :3]
        delta_orientation = action[:, 3:6]
        target_position = delta_position + self._ee_link.get_pos()
        quat_rel = xyz_to_quat(delta_orientation, rpy=True, degrees=False)
        target_orientation = transform_quat_by_quat(quat_rel, self._ee_link.get_quat())
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link, pos=target_position, quat=target_orientation, dofs_idx_local=self._arm_dof_idx
        )
        return q_pos

    def _dls_ik(self, action: torch.Tensor) -> torch.Tensor:
        delta_pose = action[:, :6]
        lambda_val = 0.01
        jacobian = self._robot_entity.get_jacobian(link=self._ee_link)
        jacobian_T = jacobian.transpose(1, 2)
        lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=self._device)
        delta_joint_pos = (
            jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
        ).squeeze(-1)
        return self._robot_entity.get_qpos() + delta_joint_pos

    def go_to_goal(self, goal_pose: torch.Tensor, open_gripper: bool = True):
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link, pos=goal_pose[:, :3], quat=goal_pose[:, 3:7], dofs_idx_local=self._arm_dof_idx
        )
        if open_gripper:
            q_pos[:, self._fingers_dof] = 0.04
        else:
            q_pos[:, self._fingers_dof] = +0.00
        self._robot_entity.control_dofs_position(position=q_pos)

    @property
    def ee_pose(self) -> torch.Tensor:
        pos, quat = self._ee_link.get_pos(), self._ee_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def left_finger_pose(self) -> torch.Tensor:
        pos, quat = self._left_finger_link.get_pos(), self._left_finger_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def right_finger_pose(self) -> torch.Tensor:
        pos, quat = self._right_finger_link.get_pos(), self._right_finger_link.get_quat()
        return torch.cat([pos, quat], dim=-1)

    @property
    def center_finger_pose(self) -> torch.Tensor:
        left_finger_pose = self.left_finger_pose
        right_finger_pose = self.right_finger_pose
        center_finger_pos = (left_finger_pose[:, :3] + right_finger_pose[:, :3]) / 2
        center_finger_quat = left_finger_pose[:, 3:7]
        return torch.cat([center_finger_pos, center_finger_quat], dim=-1)


def get_env(num_envs: int) -> FrankaPandaGraspEnv:
    try:
        gs.init(logging_level="warning", precision="32")
    except Exception:
        pass
    return FrankaPandaGraspEnv(num_envs=num_envs)