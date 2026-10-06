"""Rebuild the calibrated two-arm KUKA cell in Isaac Lab and compare it with the real calibration photos.

For every calibration pose this sets both arms to the recorded joint angles, renders the simulated D455
(calibrated intrinsics + extrinsics) and writes:

    sim_<i>.png       what the simulated D455 sees
    overlay_<i>.png   real photo with the sim silhouettes (arm1 green, arm2 blue, grippers yellow)
                      and the SAM2 masks EasyHec was fit to (white outline)
    blend_<i>.png     50/50 blend of real photo and sim render
    contact_sheet.png all overlays in one image
    setup_overview.png an outside view of the sim cell, with the D455 drawn as a box
    sim_masks.npz     sim silhouettes per pose (arm1, arm2, arm1_gripper, arm2_gripper), for scripts/compare_overlays.py
    metrics.json      per-pose IoU of the sim silhouette vs. the SAM2 mask (arm + gripper when the
                      calibration included the gripper; poses an arm wasn't calibrated on have no IoU)

Run from kuka_sim/:

    pixi r -e isaaclab-gpu python scripts/render_calibration_cell.py --headless

Drop --headless to open the Isaac Sim window; it stays open on --view-pose (default 0) until you close it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--runs", type=Path, nargs=2, default=None, metavar=("ARM1_RUN", "ARM2_RUN"),
                    help="gripper-aware EasyHec runs (default: kuka_easyhec/results/gripper_arm1 gripper_arm2)")
parser.add_argument("--calib-dir", type=Path, default=None,
                    help="use an older combined bare-arm calibration folder instead, e.g. camera_calibration_20260924")
parser.add_argument("--out", type=Path, default=Path("outputs/calibration_cell"))
parser.add_argument("--gripper-opening", type=float, default=0.06, help="fingertip gap in the photos [m], 0..0.074")
parser.add_argument("--render-frames", type=int, default=12, help="RTX frames to accumulate per pose")
parser.add_argument("--warmup-frames", type=int, default=120, help="frames rendered before the first pose")
parser.add_argument("--flange-mm", type=float, default=None,
                    help="override the media-flange thickness between the iiwa flange and the gripper mount [mm]")
parser.add_argument("--view-pose", type=int, default=0, help="with the GUI: calibration pose to hold after rendering")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import json  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from kuka_sim.cell import DEFAULT_GRIPPER_RUNS, FINGER_MAX, CellCalibration  # noqa: E402
from kuka_sim.scene import build_cell, set_pose, to_calibrated_k  # noqa: E402

COLORS = {"arm1": (60, 220, 60), "arm2": (60, 140, 255), "gripper": (255, 210, 0)}  # RGB, overlay outlines


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


def draw_overlay(real: np.ndarray, m: dict[str, np.ndarray], sam: dict[str, np.ndarray | None]) -> np.ndarray:
    out = real.astype(np.float32)
    for key, col in (("arm1", "arm1"), ("arm2", "arm2"), ("arm1_gripper", "gripper"), ("arm2_gripper", "gripper")):
        out[m[key]] = 0.45 * out[m[key]] + 0.55 * np.array(COLORS[col], np.float32)
    out = out.astype(np.uint8)
    for arm in ("arm1", "arm2"):
        outlines = [(m[arm] | m[f"{arm}_gripper"], COLORS[arm], 2)]
        if sam[arm] is not None:
            outlines.append((sam[arm], (255, 255, 255), 1))
        for mask, col, th in outlines:
            cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(out, cs, -1, col, th, cv2.LINE_AA)
    return out


def main():
    if args.calib_dir is not None:
        cal = CellCalibration.load(args.calib_dir)
    else:
        cal = CellCalibration.load_gripper_runs(*(args.runs or DEFAULT_GRIPPER_RUNS))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    finger_q = float(np.clip(args.gripper_opening / 2, 0.0, FINGER_MAX))

    sim, arms, d455, overview = build_cell(cal, args.device, args.flange_mm, out)
    # MDL materials (the wood table) compile and stream their textures over the first frames; render a few
    # throwaway frames so pose 0 isn't captured with the placeholder material
    for _ in range(args.warmup_frames):
        sim.step(render=True)
    n = len(cal.images)
    metrics, overlays, sim_masks = [], [], []
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
        sam = {a: cal.sam_masks[a].get(i) for a in arms}
        sim_masks.append(masks)

        real = cal.images[i]
        ov = draw_overlay(real, masks, sam)
        blend = cv2.addWeighted(real, 0.5, rgb, 0.5, 0)
        for name, img in (("sim", rgb), ("overlay", ov), ("blend", blend)):
            cv2.imwrite(str(out / f"{name}_{i}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        overlays.append(ov)

        row = {"pose": i, "max_joint_error_rad": joint_err}
        for a in arms:
            sim_mask = masks[a] | masks[f"{a}_gripper"] if cal.masks_include_gripper else masks[a]
            if sam[a] is None:
                row[f"iou_{a}_vs_sam"] = None
                continue
            inter, union = (sim_mask & sam[a]).sum(), (sim_mask | sam[a]).sum()
            row[f"iou_{a}_vs_sam"] = float(inter / union) if union else None
        metrics.append(row)
        ious = "  ".join(f"{a} {'-' if row[f'iou_{a}_vs_sam'] is None else format(row[f'iou_{a}_vs_sam'], '.3f')}" for a in arms)
        print(f"pose {i}: IoU {ious}  joint err {max(joint_err.values()):.2e} rad", flush=True)

    overview.update(0.0, force_recompute=True)
    cv2.imwrite(str(out / "setup_overview.png"), cv2.cvtColor(overview.data.output["rgb"][0, ..., :3].cpu().numpy(), cv2.COLOR_RGB2BGR))

    cols = 4
    thumbs = [cv2.resize(o, (640, 360), interpolation=cv2.INTER_AREA) for o in overlays]
    thumbs += [np.zeros_like(thumbs[0])] * (-len(thumbs) % cols)
    sheet = np.vstack([np.hstack(thumbs[r:r + cols]) for r in range(0, len(thumbs), cols)])
    cv2.imwrite(str(out / "contact_sheet.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    np.savez_compressed(out / "sim_masks.npz", **{k: np.stack([m[k] for m in sim_masks]) for k in sim_masks[0]})

    summary = {
        **{f"mean_iou_{a}_vs_sam": float(np.mean([m[f"iou_{a}_vs_sam"] for m in metrics if m[f"iou_{a}_vs_sam"] is not None]))
           for a in arms},
        "masks_include_gripper": cal.masks_include_gripper,
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
