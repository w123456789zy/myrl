"""Pure-torch math helpers used across the mdp functions.

These are intentionally framework-agnostic: they take plain ``torch.Tensor``
inputs and produce plain ``torch.Tensor`` outputs. They mirror the helpers
in ``unitree_rl_lab`` (quat_apply_inverse, quat_rotate, etc.) but rewritten
against the tensor conventions used by the existing Go2 environments.

Backward-compat aliases: many legacy modules (``go2_walk``, ``go2_backflip``,
…) import helper names with the ``gs_`` prefix. Both naming styles are
exported here so older imports keep working.
"""

from __future__ import annotations

import math

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Random sampling helpers
# ---------------------------------------------------------------------------

def rand_float(lower: float, upper: float, shape, device) -> torch.Tensor:
    """Uniform random tensor in ``[lower, upper]``."""
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


# ``gs_`` prefix aliases for backward compat
gs_rand_float = rand_float


# ---------------------------------------------------------------------------
# Quaternion algebra
# ---------------------------------------------------------------------------

def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


def inv_quat(quat: torch.Tensor) -> torch.Tensor:
    """Inverse of a quaternion ``(w, x, y, z)``."""
    qw, qx, qy, qz = quat.unbind(-1)
    return torch.stack([qw, -qx, -qy, -qz], dim=-1)


def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    shape = quat.shape
    q = quat.reshape(-1, 4)
    return torch.cat((q[:, :1], -q[:, 1:]), dim=-1).view(shape)


def quat_apply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Rotate vector ``b`` by quaternion ``a``."""
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, 1:]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, :1] * t + xyz.cross(t, dim=-1)).view(shape)


def quat_apply_yaw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate ``vec`` by the yaw component of ``quat`` only."""
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, 1:3] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([w, x, y, z], dim=-1).view(shape)


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Build a quaternion from a (per-env) angle and a (broadcastable) axis."""
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return normalize(torch.cat([w, xyz], dim=-1))


def quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to (roll, pitch, yaw) in radians."""
    qw, qx, qy, qz = quat.unbind(-1)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = torch.where(
        torch.abs(sinp) >= 1,
        torch.sign(sinp) * (math.pi / 2),
        torch.asin(sinp),
    )
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([roll, pitch, yaw], dim=-1)


def euler_to_quat(xyz: torch.Tensor) -> torch.Tensor:
    """Convert (roll, pitch, yaw) euler angles to a quaternion ``(w, x, y, z)``."""
    roll, pitch, yaw = xyz.unbind(-1)
    cosr = (roll * 0.5).cos()
    sinr = (roll * 0.5).sin()
    cosp = (pitch * 0.5).cos()
    sinp = (pitch * 0.5).sin()
    cosy = (yaw * 0.5).cos()
    siny = (yaw * 0.5).sin()
    qw = cosr * cosp * cosy + sinr * sinp * siny
    qx = sinr * cosp * cosy - cosr * sinp * siny
    qy = cosr * sinp * cosy + sinr * cosp * siny
    qz = cosr * cosp * siny - sinr * sinp * cosy
    return torch.stack([qw, qx, qy, qz], dim=-1)


def transform_by_quat(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Rotate ``pos`` by the rotation encoded in ``quat``."""
    qw, qx, qy, qz = quat.unbind(-1)
    rot_matrix = torch.stack(
        [
            1.0 - 2 * qy**2 - 2 * qz**2,
            2 * qx * qy - 2 * qz * qw,
            2 * qx * qz + 2 * qy * qw,
            2 * qx * qy + 2 * qz * qw,
            1 - 2 * qx**2 - 2 * qz**2,
            2 * qy * qz - 2 * qx * qw,
            2 * qx * qz - 2 * qy * qw,
            2 * qy * qz + 2 * qx * qw,
            1 - 2 * qx**2 - 2 * qy**2,
        ],
        dim=-1,
    ).reshape(*quat.shape[:-1], 3, 3)
    rotated_pos = torch.matmul(rot_matrix, pos.unsqueeze(-1)).squeeze(-1)
    return rotated_pos


# ``gs_`` prefix aliases for backward compat with go2_*.py legacy imports
gs_inv_quat = inv_quat
gs_quat_conjugate = quat_conjugate
gs_quat_apply = quat_apply
gs_quat_apply_yaw = quat_apply_yaw
gs_quat_mul = quat_mul
gs_quat_from_angle_axis = quat_from_angle_axis
gs_quat2euler = quat_to_euler
gs_euler2quat = euler_to_quat
gs_transform_by_quat = transform_by_quat


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def wrap_to_pi(angles):
    """Wrap an angle (or batch of angles) to ``[-pi, pi]``."""
    angles = np.asarray(angles) if not isinstance(angles, torch.Tensor) else angles
    if isinstance(angles, torch.Tensor):
        angles = angles % (2 * math.pi)
        angles = angles - 2 * math.pi * (angles > math.pi)
        return angles
    angles = angles % (2 * math.pi)
    angles -= 2 * math.pi * (angles > math.pi)
    return angles
