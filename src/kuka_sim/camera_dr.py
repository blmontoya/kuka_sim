"""Slight domain randomization of the D455's intrinsics and extrinsics around its EasyHec calibration.

Each :meth:`CameraRandomizer.randomize` call samples
  * intrinsics: fx, fy scaled by a common random factor (focal length / zoom) plus a small independent fx/fy
    jitter, and the principal point (cx, cy) shifted by a few pixels;
  * extrinsics: a small random rotation (tilt/pan/roll) and translation of the camera, in the camera's own frame.

The pose is written to the sim camera. The intrinsics are not: RTX ignores principal-point offsets, so the scene
renders a centred pinhole and :func:`kuka_sim.scene.to_calibrated_k` warps it onto a K. Pass the K returned here
(``to_calibrated_k(img, cal, nearest, K=sample["K"])``); for a pinhole camera that warp is exact. A smaller focal
length sees more than the nominal render, so build the cell with ``pad=cfg.required_pad(cal)``.

This is the slight, calibration-centred version of the camera DR on gigastrap's dev/jay/domain_randomization
branch (shadow_hand_dr_vision_env.py), which samples much wider focal ranges and whole new viewpoints.

Import this only after the Isaac Sim app is running (``AppLauncher``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from isaaclab.sensors import Camera

from kuka_sim.cell import CellCalibration, pose_from_matrix


@dataclass
class CameraDRCfg:
    focal_scale: float = 0.02
    """fx and fy are both scaled by 1 + U(-focal_scale, focal_scale) (0.02 = +-2 %, about +-13 px on the D455)."""
    aspect_jitter: float = 0.002
    """Extra independent scale on fy only, +-this fraction (the real fx/fy differ by ~0.1 %)."""
    principal_px: float = 4.0
    """cx and cy are each shifted by U(-principal_px, principal_px) pixels."""
    rot_deg: float = 1.0
    """Rotation about each camera axis (x = tilt, y = pan, z = roll), U(-rot_deg, rot_deg) degrees."""
    trans_m: float = 0.01
    """Translation along each camera axis, U(-trans_m, trans_m) metres."""
    seed: int | None = None

    def required_pad(self, cal: CellCalibration) -> int:
        """Rendered border (pixels per side) needed so the widest sampled K never samples outside the render."""
        f_r = cal.K[0, 0]
        s = f_r / (cal.K[1, 1] * (1 - self.focal_scale) * (1 - self.aspect_jitter))  # largest src/dst scale
        reach_x = max(cal.K[0, 2], cal.width - 1 - cal.K[0, 2]) + self.principal_px
        reach_y = max(cal.K[1, 2], cal.height - 1 - cal.K[1, 2]) + self.principal_px
        need = max(s * reach_x - (cal.width - 1) / 2, s * reach_y - (cal.height - 1) / 2)
        return int(np.ceil(need)) + 2


class CameraRandomizer:
    def __init__(self, camera: Camera, cal: CellCalibration, cfg: CameraDRCfg = CameraDRCfg()):
        self.camera, self.cal, self.cfg = camera, cal, cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.T_nominal = cal.T_arm1_cam_cv.copy()  # OpenCV/ROS optical frame in the world (= arm1 base)

    def _set_pose(self, T: np.ndarray):
        pos, quat = pose_from_matrix(T)
        dev = self.camera.device
        self.camera.set_world_poses(torch.tensor([pos], device=dev), torch.tensor([quat], device=dev), convention="ros")

    def randomize(self) -> dict:
        """Sample and apply a new camera pose; return {"K": 3x3 intrinsics to warp onto, "T": 4x4 pose, ...}."""
        c, r = self.cfg, self.rng
        K = self.cal.K.copy()
        s = 1 + r.uniform(-c.focal_scale, c.focal_scale)
        K[0, 0] *= s
        K[1, 1] *= s * (1 + r.uniform(-c.aspect_jitter, c.aspect_jitter))
        K[0, 2] += r.uniform(-c.principal_px, c.principal_px)
        K[1, 2] += r.uniform(-c.principal_px, c.principal_px)

        rotvec_deg = r.uniform(-c.rot_deg, c.rot_deg, size=3)
        delta = np.eye(4)
        delta[:3, :3] = Rotation.from_rotvec(np.deg2rad(rotvec_deg)).as_matrix()
        delta[:3, 3] = r.uniform(-c.trans_m, c.trans_m, size=3)
        T = self.T_nominal @ delta  # perturbation in the camera's own frame
        self._set_pose(T)
        return {"K": K, "T": T, "focal_scale": float(s), "rot_deg_xyz": rotvec_deg.tolist(),
                "trans_m_xyz": delta[:3, 3].tolist()}

    def restore(self):
        """Put the camera back at its calibrated pose (intrinsics: just call to_calibrated_k without K)."""
        self._set_pose(self.T_nominal)
