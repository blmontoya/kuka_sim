"""Two KUKA iiwa7 + HandUMI grippers and the D455, placed from the EasyHec calibration.

World frame = arm1 base (``lbr_link_0``): +X forward, +Z up, origin on the mounting surface.

Import this only after the Isaac Sim app is running (``AppLauncher``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverterCfg

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "iiwa7_handumi"
DEFAULT_CALIB_DIR = Path(__file__).resolve().parents[3] / "camera_calibration_20260924"  # bare-arm calibration
# gripper-aware EasyHec runs (one folder per arm, each with its own dataset.npz)
DEFAULT_GRIPPER_RUNS = tuple(
    Path(__file__).resolve().parents[3] / "kuka_easyhec" / "results" / f"gripper_{a}" for a in ("arm1", "arm2")
)

ARM_JOINTS = [f"lbr_A{i}" for i in range(1, 8)]
FINGER_JOINTS = ["gripper_finger_left_joint", "gripper_finger_right_joint"]
FINGER_MAX = 0.037  # per-finger travel [m]; opening = 2 * q


IIWA7_HANDUMI_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(ASSET_DIR / "iiwa7_handumi.urdf"),
        fix_base=True,
        root_link_name="lbr_link_0",
        # keep gripper parts as separate prims so they can be told apart in segmentation
        merge_fixed_joints=False,
        self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=1e4, damping=1e2)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={"lbr_A.*": 0.0, "gripper_finger_.*": 0.0}),
    actuators={
        "arm": ImplicitActuatorCfg(joint_names_expr=["lbr_A[1-7]"], stiffness=1e4, damping=1e2),
        "gripper": ImplicitActuatorCfg(joint_names_expr=["gripper_finger_.*_joint"], stiffness=1e4, damping=1e2),
    },
)
"""One iiwa7 R800 with the HandUMI gripper on the flange. Finger q = opening/2 [m]."""


@dataclass
class CellCalibration:
    K: np.ndarray  # 3x3 colour intrinsics
    width: int
    height: int
    T_arm1_cam_cv: np.ndarray  # D455 colour optical frame (OpenCV: +Z fwd, +Y down) in arm1 base
    T_arm1_arm2: np.ndarray  # arm2 base in arm1 base
    images: np.ndarray  # (N, H, W, 3) RGB calibration photos
    joints_deg: dict[str, np.ndarray]  # "arm1"/"arm2" -> (N, 7), KUKA A1..A7 in degrees
    sam_masks: dict[str, dict[int, np.ndarray]]  # "arm1"/"arm2" -> {pose index: (H, W) SAM2 mask EasyHec was fit to}
    masks_include_gripper: bool  # True if the SAM2 masks cover the gripper too (calibrated with the gripper in the URDF)

    @staticmethod
    def load(calib_dir: Path = DEFAULT_CALIB_DIR) -> CellCalibration:
        calib_dir = Path(calib_dir)
        data = np.load(calib_dir / "dataset.npz")
        cell = json.loads((calib_dir / "cell_calibration.json").read_text())
        T_cam_arm1_cv = np.load(calib_dir / "arm1" / "camera_extrinsic_opencv.npy").astype(np.float64)
        images = data["images"]
        return CellCalibration(
            K=data["intrinsic"].astype(np.float64),
            width=images.shape[2],
            height=images.shape[1],
            T_arm1_cam_cv=np.linalg.inv(T_cam_arm1_cv),
            T_arm1_arm2=np.array(cell["arm2_base_in_arm1_base"]["matrix"], dtype=np.float64),
            images=images,
            joints_deg={a: data[f"joints_deg_{a}"] for a in ("arm1", "arm2")},
            sam_masks={a: dict(enumerate(np.load(calib_dir / a / "masks.npy") > 0.5)) for a in ("arm1", "arm2")},
            masks_include_gripper=False,
        )

    @staticmethod
    def load_gripper_runs(arm1_run: Path = DEFAULT_GRIPPER_RUNS[0], arm2_run: Path = DEFAULT_GRIPPER_RUNS[1]) -> CellCalibration:
        """Load two separate gripper-aware EasyHec runs (``<run>/dataset.npz`` + ``<run>/<arm>/``).

        The photos and joint angles come from arm1's run. Each run may use a subset of those photos; its masks are
        matched back to the photo they were drawn on. arm2's base pose is chained through the camera, as in
        kuka_easyhec/combine.py.
        """
        runs = {"arm1": Path(arm1_run), "arm2": Path(arm2_run)}
        data = np.load(runs["arm1"] / "dataset.npz")
        images = data["images"]
        T_cam_ros = {a: np.load(r / a / "camera_pose_ros.npy").astype(np.float64) for a, r in runs.items()}
        sam_masks = {}
        for a, r in runs.items():
            run_images = np.load(r / "dataset.npz")["images"]
            masks = np.load(r / a / "masks.npy") > 0.5
            sam_masks[a] = {}
            for img, m in zip(run_images, masks):
                match = [i for i, ref in enumerate(images) if np.array_equal(ref, img)]
                if not match:
                    raise ValueError(f"{r}: a photo in its dataset.npz is not in {runs['arm1'] / 'dataset.npz'}")
                sam_masks[a][match[0]] = m
        return CellCalibration(
            K=data["intrinsic"].astype(np.float64),
            width=images.shape[2],
            height=images.shape[1],
            T_arm1_cam_cv=np.linalg.inv(np.load(runs["arm1"] / "arm1" / "camera_extrinsic_opencv.npy").astype(np.float64)),
            T_arm1_arm2=T_cam_ros["arm1"] @ np.linalg.inv(T_cam_ros["arm2"]),
            images=images,
            joints_deg={a: data[f"joints_deg_{a}"] for a in ("arm1", "arm2")},
            sam_masks=sam_masks,
            masks_include_gripper=True,
        )


def pose_from_matrix(T: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """4x4 -> (xyz, quat wxyz) as plain tuples, for Isaac Lab configs."""
    x, y, z, w = Rotation.from_matrix(T[:3, :3]).as_quat()
    return tuple(float(v) for v in T[:3, 3]), (float(w), float(x), float(y), float(z))
