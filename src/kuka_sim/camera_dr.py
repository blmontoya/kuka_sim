"""Slight domain randomization of the D455's intrinsics and extrinsics around its EasyHec calibration.

Each :meth:`CameraRandomizer.randomize` call samples
  * intrinsics: fx, fy scaled by a common random factor (focal length / zoom) plus a small independent fx/fy
    jitter, and the principal point (cx, cy) shifted by a few pixels;
  * extrinsics: a small random rotation (tilt/pan/roll) and translation of the camera, in the camera's own frame.

The pose and the focal-length change are written to the sim camera's USD prim (like the per-env camera DR on gigastrap's
dev/jay/domain_randomization branch), so they show up in the sim itself, e.g. in the viewport looking through the D455.
RTX ignores principal-point offsets and non-square pixels, so the sim renders a centred pinhole with the sampled fx and
:func:`kuka_sim.scene.to_calibrated_k` warps it onto the exact sampled K. Pass both values returned here
(``to_calibrated_k(img, cal, nearest, K=sample["K"], f_render=sample["f_render"])``); for a pinhole camera that warp is
exact. Build the cell with ``pad=cfg.required_pad(cal)``.

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
        h_r, w_r = camera.image_shape
        self.K_render_nominal = np.array([[cal.K[0, 0], 0.0, w_r / 2], [0.0, cal.K[0, 0], h_r / 2], [0.0, 0.0, 1.0]])

    def _set_render_focal(self, f: float):
        """Re-render the centred pinhole with focal length f [px] by rewriting the USD camera's focal length/aperture."""
        K = self.K_render_nominal.copy()
        K[0, 0] = K[1, 1] = f
        self.camera.set_intrinsic_matrices(torch.tensor(K[None], dtype=torch.float32))

    def _set_pose(self, T: np.ndarray):
        pos, quat = pose_from_matrix(T)
        dev = self.camera.device
        self.camera.set_world_poses(torch.tensor([pos], device=dev), torch.tensor([quat], device=dev), convention="ros")

    def randomize(self) -> dict:
        """Sample and apply a new camera pose and focal length; return {"K": 3x3 intrinsics to warp onto,
        "f_render": the focal length [px] the sim now renders with, "T": 4x4 pose, ...}."""
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
        f_render = float(K[0, 0])
        self._set_render_focal(f_render)
        return {"K": K, "f_render": f_render, "T": T, "focal_scale": float(s), "rot_deg_xyz": rotvec_deg.tolist(),
                "trans_m_xyz": delta[:3, 3].tolist()}

    def restore(self):
        """Put the camera back at its calibrated pose and focal length (then call to_calibrated_k without K)."""
        self._set_pose(self.T_nominal)
        self._set_render_focal(float(self.cal.K[0, 0]))
