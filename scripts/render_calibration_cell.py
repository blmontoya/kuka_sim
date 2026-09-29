"""Rebuild the calibrated two-arm KUKA cell in Isaac Lab and compare it with the real calibration photos.

For every calibration pose this sets both arms to the recorded joint angles, renders the simulated D455
(calibrated intrinsics + extrinsics) and writes:

    sim_<i>.png       what the simulated D455 sees
    overlay_<i>.png   real photo with the sim silhouettes (arm1 green, arm2 blue, grippers yellow)
                      and the SAM2 masks EasyHec was fit to (white outline)
    blend_<i>.png     50/50 blend of real photo and sim render
    contact_sheet.png all overlays in one image
    setup_overview.png an outside view of the sim cell, with the D455 drawn as a box
    metrics.json      per-pose IoU of the sim arm silhouette vs. the SAM2 mask

Run from kuka_sim/:

    pixi r -e isaaclab-gpu python scripts/render_calibration_cell.py --headless

Drop --headless to open the Isaac Sim window; it stays open on --view-pose (default 0) until you close it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--calib-dir", type=Path, default=None, help="EasyHec output folder (default: camera_calibration_20260924)")
parser.add_argument("--out", type=Path, default=Path("outputs/calibration_cell"))
parser.add_argument("--gripper-opening", type=float, default=0.06, help="fingertip gap in the photos [m], 0..0.074")
parser.add_argument("--render-frames", type=int, default=12, help="RTX frames to accumulate per pose")
parser.add_argument("--view-pose", type=int, default=0, help="with the GUI: calibration pose to hold after rendering")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import json  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

from kuka_sim.cell import (  # noqa: E402
    ARM_JOINTS,
    DEFAULT_CALIB_DIR,
    FINGER_JOINTS,
    FINGER_MAX,
    IIWA7_HANDUMI_CFG,
    CellCalibration,
    pose_from_matrix,
)

PAD = 16  # extra rendered pixels per side, so the principal-point shift never samples outside the render
COLORS = {"arm1": (60, 220, 60), "arm2": (60, 140, 255), "gripper": (255, 210, 0)}  # RGB


def build_scene(cal: CellCalibration):
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=args.device))

    dome = sim_utils.DomeLightCfg(intensity=1500.0)
    dome.func("/World/DomeLight", dome)
    light = sim_utils.DistantLightCfg(intensity=2500.0, angle=0.5)
    light.func("/World/KeyLight", light, orientation=(0.87, 0.2, -0.3, 0.3))
    # visual-only tabletop at the bases' mounting surface (no collider, so it can't push on the arms)
    table = sim_utils.CuboidCfg(
        size=(1.6, 2.2, 0.02), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.80, 0.66, 0.48), roughness=0.6)
    )
    table.func("/World/Table", table, translation=(0.55, -0.36, -0.0101))

    arm1 = Articulation(IIWA7_HANDUMI_CFG.replace(prim_path="/World/Arm1"))
    pos2, rot2 = pose_from_matrix(cal.T_arm1_arm2)
    arm2_cfg = IIWA7_HANDUMI_CFG.replace(prim_path="/World/Arm2")
    arm2_cfg.init_state = arm2_cfg.init_state.replace(pos=pos2, rot=rot2)
    arm2 = Articulation(arm2_cfg)

    # D455 colour camera. RTX ignores principal-point offsets and uses fy = fx, so render a centred pinhole with
    # the calibrated fx and a PAD border, then warp it onto the exact calibrated K (see to_calibrated_k).
    fx = cal.K[0, 0]
    w_r, h_r = cal.width + 2 * PAD, cal.height + 2 * PAD
    K_render = [fx, 0.0, w_r / 2, 0.0, fx, h_r / 2, 0.0, 0.0, 1.0]
    cam_pos, cam_rot = pose_from_matrix(cal.T_arm1_cam_cv)
    d455 = Camera(
        CameraCfg(
            prim_path="/World/D455/color",
            width=w_r,
            height=h_r,
            data_types=["rgb", "instance_id_segmentation_fast"],
            colorize_instance_id_segmentation=False,
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(K_render, w_r, h_r, clipping_range=(0.05, 20.0)),
            offset=CameraCfg.OffsetCfg(pos=cam_pos, rot=cam_rot, convention="ros"),
        )
    )
    # D455 body (124 x 29 x 26 mm) so the camera shows up in the overview shot; hidden by the near clip in its own view
    body = sim_utils.CuboidCfg(size=(0.124, 0.029, 0.026), visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)))
    body.func("/World/D455_body", body, translation=cam_pos, orientation=cam_rot)

    overview = Camera(
        CameraCfg(
            prim_path="/World/Overview",
            width=1600,
            height=1000,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 50.0)),
        )
    )
    sim.reset()
    overview.set_world_poses_from_view(
        torch.tensor([[1.9, 0.9, 1.35]], device=sim.device), torch.tensor([[0.35, -0.36, 0.25]], device=sim.device)
    )
    return sim, {"arm1": arm1, "arm2": arm2}, d455, overview


def set_pose(arm: Articulation, q_deg: np.ndarray, finger_q: float):
    q = arm.data.default_joint_pos.clone()
    ids_arm, _ = arm.find_joints(ARM_JOINTS, preserve_order=True)
    ids_fing, _ = arm.find_joints(FINGER_JOINTS, preserve_order=True)
    q[:, ids_arm] = torch.tensor(np.deg2rad(q_deg), dtype=q.dtype, device=q.device)
    q[:, ids_fing] = finger_q
    arm.write_joint_state_to_sim(q, torch.zeros_like(q))
    arm.set_joint_position_target(q)
    arm.write_data_to_sim()
    return q


def to_calibrated_k(img: np.ndarray, cal: CellCalibration, nearest: bool) -> np.ndarray:
    """Resample a centred-pinhole render (fx = fy, PAD border) onto the calibrated K (cx, cy, fy)."""
    fx, fy, cx, cy = cal.K[0, 0], cal.K[1, 1], cal.K[0, 2], cal.K[1, 2]
    h_r, w_r = img.shape[:2]
    # dst pixel (u, v) -> src (u - cx + c_rx, fx/fy * (v - cy) + c_ry); pixel centres at integers (OpenCV)
    M = np.array([[1.0, 0.0, (w_r - 1) / 2 - cx], [0.0, fx / fy, (h_r - 1) / 2 - fx / fy * cy]])
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (cal.width, cal.height), flags=interp | cv2.WARP_INVERSE_MAP)


def label_masks(seg: np.ndarray, id_to_path: dict) -> dict[str, np.ndarray]:
    """Split the instance segmentation into arm1/arm2 arm-links and grippers, by prim path."""
    out = {k: np.zeros(seg.shape, bool) for k in ("arm1", "arm2", "arm1_gripper", "arm2_gripper")}
    for sid, path in id_to_path.items():
        path = str(path)
        for arm, root in (("arm1", "/World/Arm1/"), ("arm2", "/World/Arm2/")):
            if path.startswith(root):
                key = f"{arm}_gripper" if "/gripper_" in path else arm
                out[key] |= seg == int(sid)
    return out


def draw_overlay(real: np.ndarray, m: dict[str, np.ndarray], sam: dict[str, np.ndarray]) -> np.ndarray:
    out = real.astype(np.float32)
    for key, col in (("arm1", "arm1"), ("arm2", "arm2"), ("arm1_gripper", "gripper"), ("arm2_gripper", "gripper")):
        out[m[key]] = 0.45 * out[m[key]] + 0.55 * np.array(COLORS[col], np.float32)
    out = out.astype(np.uint8)
    for arm in ("arm1", "arm2"):
        for mask, col, th in ((m[arm] | m[f"{arm}_gripper"], COLORS[arm], 2), (sam[arm], (255, 255, 255), 1)):
            cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, cs, -1, col, th, cv2.LINE_AA)
    return out


def main():
    cal = CellCalibration.load(args.calib_dir or DEFAULT_CALIB_DIR)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    finger_q = float(np.clip(args.gripper_opening / 2, 0.0, FINGER_MAX))

    sim, arms, d455, overview = build_scene(cal)
    n = len(cal.images)
    metrics, overlays = [], []
    for i in range(n):
        targets = {a: set_pose(arm, cal.joints_deg[a][i], finger_q) for a, arm in arms.items()}
        for _ in range(args.render_frames):
            sim.step(render=True)
            for arm in arms.values():
                arm.update(sim.get_physics_dt())
        d455.update(0.0, force_recompute=True)
        joint_err = {a: float((arms[a].data.joint_pos - targets[a]).abs().max()) for a in arms}

        rgb = to_calibrated_k(d455.data.output["rgb"][0, ..., :3].cpu().numpy(), cal, nearest=False)
        seg_raw = d455.data.output["instance_id_segmentation_fast"][0, ..., 0].cpu().numpy().astype(np.int32)
        seg = to_calibrated_k(seg_raw, cal, nearest=True)
        masks = label_masks(seg, d455.data.info[0]["instance_id_segmentation_fast"]["idToLabels"])
        sam = {a: cal.sam_masks[a][i] for a in arms}

        real = cal.images[i]
        ov = draw_overlay(real, masks, sam)
        blend = cv2.addWeighted(real, 0.5, rgb, 0.5, 0)
        for name, img in (("sim", rgb), ("overlay", ov), ("blend", blend)):
            cv2.imwrite(str(out / f"{name}_{i}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        overlays.append(ov)

        row = {"pose": i, "max_joint_error_rad": joint_err}
        for a in arms:
            inter, union = (masks[a] & sam[a]).sum(), (masks[a] | sam[a]).sum()
            row[f"iou_{a}_vs_sam"] = float(inter / union) if union else None
        metrics.append(row)
        print(f"pose {i}: IoU arm1 {row['iou_arm1_vs_sam']:.3f}  arm2 {row['iou_arm2_vs_sam']:.3f}  "
              f"joint err {max(joint_err.values()):.2e} rad", flush=True)

    overview.update(0.0, force_recompute=True)
    cv2.imwrite(str(out / "setup_overview.png"), cv2.cvtColor(overview.data.output["rgb"][0, ..., :3].cpu().numpy(), cv2.COLOR_RGB2BGR))

    cols = 4
    thumbs = [cv2.resize(o, (640, 360), interpolation=cv2.INTER_AREA) for o in overlays]
    thumbs += [np.zeros_like(thumbs[0])] * (-len(thumbs) % cols)
    sheet = np.vstack([np.hstack(thumbs[r:r + cols]) for r in range(0, len(thumbs), cols)])
    cv2.imwrite(str(out / "contact_sheet.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    summary = {
        "mean_iou_arm1_vs_sam": float(np.mean([m["iou_arm1_vs_sam"] for m in metrics])),
        "mean_iou_arm2_vs_sam": float(np.mean([m["iou_arm2_vs_sam"] for m in metrics])),
        "gripper_opening_m": 2 * finger_q,
        "poses": metrics,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"mean IoU vs SAM2 masks: arm1 {summary['mean_iou_arm1_vs_sam']:.3f}, arm2 {summary['mean_iou_arm2_vs_sam']:.3f}")
    print(f"wrote {out}", flush=True)

    if not args.headless:
        # keep the window open on one calibration pose; close the window to exit
        for a, arm in arms.items():
            set_pose(arm, cal.joints_deg[a][args.view_pose], finger_q)
        sim.set_camera_view(eye=[1.9, 0.9, 1.35], target=[0.35, -0.36, 0.25])
        print(f"holding pose {args.view_pose}; close the Isaac Sim window to exit", flush=True)
        while app.is_running():
            sim.step(render=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
