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
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

from kuka_sim.cell import (  # noqa: E402
    ARM_JOINTS,
    DEFAULT_GRIPPER_RUNS,
    FINGER_JOINTS,
    FINGER_MAX,
    ASSET_DIR,
    IIWA7_HANDUMI_CFG,
    CellCalibration,
    pose_from_matrix,
)

TABLE_HEIGHT = 0.004  # tabletop above the base mounting surface [m] (measured)
PAD = 16  # extra rendered pixels per side, so the principal-point shift never samples outside the render
COLORS = {"arm1": (60, 220, 60), "arm2": (60, 140, 255), "gripper": (255, 210, 0)}  # RGB, overlay outlines

# Sim appearance. The URDF colours are lost in Isaac Lab's URDF->USD conversion, so they are bound here.
# Keys are prim-path regexes (one per link), values linear RGB 0..1. Edit these to recolour the sim.
PART_COLORS = {
    "/World/Arm1/lbr_link_.*": (0.20, 0.45, 0.85),  # arm1 (left): blue
    "/World/Arm2/lbr_link_.*": (0.90, 0.42, 0.10),  # arm2 (right): orange
    "/World/Arm.*/gripper_.*": (0.03, 0.03, 0.03),  # grippers: black
    "/World/Arm.*/gripper_assembly_root": (0.75, 0.75, 0.78),  # media flange: metal (after the gripper line, so it wins)
}
TABLE_COLOR = (0.80, 0.66, 0.48)  # used only if TABLE_WOOD_TEXTURE is None
# light birch wood image (from NVIDIA Materials/Base/Wood/Birch, copied locally); set to None for the flat TABLE_COLOR
TABLE_WOOD_TEXTURE = Path(__file__).resolve().parents[1] / "src/kuka_sim/assets/materials/Birch/Birch_BaseColor.png"
TABLE_WOOD_TILE = 0.6  # metres of table covered by one copy of the texture (smaller = finer grain)
DOME_INTENSITY = 500.0  # ambient light
KEY_INTENSITY = 800.0  # directional light; lower both if the render looks washed out


def spawn_table(path: str, size: tuple[float, float, float], center: tuple[float, float, float]):
    """Visual-only box. With TABLE_WOOD_TEXTURE it is a UV-mapped mesh with a textured UsdPreviewSurface
    (Isaac Lab's CuboidCfg has no UVs, so image textures can't be shown on it)."""
    if TABLE_WOOD_TEXTURE is None:
        cfg = sim_utils.CuboidCfg(size=size, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=TABLE_COLOR, roughness=0.6))
        cfg.func(path, cfg, translation=center)
        return
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    stage = sim_utils.get_current_stage()
    hx, hy, hz = (v / 2 for v in size)
    cx, cy, cz = center
    pts, uvs, counts, idx = [], [], [], []
    # (normal axis, sign): one quad per face, UVs = the face's two in-plane coordinates in metres / TABLE_WOOD_TILE
    for axis in range(3):
        for sign in (-1, 1):
            u_ax, v_ax = [a for a in range(3) if a != axis]
            half = (hx, hy, hz)
            corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)] if sign > 0 else [(-1, -1), (-1, 1), (1, 1), (1, -1)]
            for su, sv in corners:
                q = [0.0, 0.0, 0.0]
                q[axis], q[u_ax], q[v_ax] = sign * half[axis], su * half[u_ax], sv * half[v_ax]
                pts.append(Gf.Vec3f(q[0] + cx, q[1] + cy, q[2] + cz))
                uvs.append(Gf.Vec2f(q[u_ax] / TABLE_WOOD_TILE, q[v_ax] / TABLE_WOOD_TILE))
            idx += list(range(len(pts) - 4, len(pts)))
            counts.append(4)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    st.Set(Vt.Vec2fArray(uvs))

    mat = UsdShade.Material.Define(stage, f"{path}_Looks/wood")
    surf = UsdShade.Shader.Define(stage, f"{path}_Looks/wood/surface")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    reader = UsdShade.Shader.Define(stage, f"{path}_Looks/wood/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, f"{path}_Looks/wood/diffuse")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(TABLE_WOOD_TEXTURE))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)


def apply_part_colors():
    """Bind one PreviewSurface per PART_COLORS entry to the visuals of the matching links."""
    stage = sim_utils.get_current_stage()
    for k, (pattern, rgb) in enumerate(PART_COLORS.items()):
        mat_path = f"/World/Looks/part_color_{k}"
        mat = sim_utils.PreviewSurfaceCfg(diffuse_color=rgb, roughness=0.5)
        mat.func(mat_path, mat)
        for path in sim_utils.find_matching_prim_paths(pattern + "/visuals"):
            # the URDF importer makes visuals instanceable, and materials can't be bound to instanced prims
            stage.GetPrimAtPath(path).SetInstanceable(False)
            sim_utils.bind_visual_material(path, mat_path)


def robot_cfg(out: Path):
    """IIWA7_HANDUMI_CFG, or a copy whose media flange is --flange-mm thick (gripper moved to match)."""
    if args.flange_mm is None:
        return IIWA7_HANDUMI_CFG
    import xml.etree.ElementTree as ET

    t = args.flange_mm / 1000
    tree = ET.parse(ASSET_DIR / "iiwa7_handumi.urdf")
    root = tree.getroot()
    for mesh in root.iter("mesh"):  # the copy lives in the output folder, so make mesh paths absolute
        mesh.set("filename", str(ASSET_DIR / mesh.get("filename")))
    for j in root.iter("joint"):
        if j.get("name") == "gripper_flange__gripper":
            o = j.find("origin")
            x, y, _ = o.get("xyz").split()
            o.set("xyz", f"{x} {y} {t}")
    flange_link = next(lk for lk in root.iter("link") if lk.get("name") == "gripper_assembly_root")
    for el in list(flange_link):
        cyl = el.find("geometry/cylinder")
        if cyl is None:
            continue
        if t <= 0:
            flange_link.remove(el)
        else:
            cyl.set("length", str(t))
            el.find("origin").set("xyz", f"0 0 {-t / 2}")
    path = out / f"iiwa7_handumi_flange{args.flange_mm:g}mm.urdf"
    tree.write(path, xml_declaration=True, encoding="utf-8")
    return IIWA7_HANDUMI_CFG.replace(spawn=IIWA7_HANDUMI_CFG.spawn.replace(asset_path=str(path)))


def build_scene(cal: CellCalibration, out: Path):
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=args.device))

    # the visible dome is the background: grey, and brighter with DOME_INTENSITY.
    # visible_in_primary_ray=False makes the background black (the dome still lights the scene)
    dome = sim_utils.DomeLightCfg(intensity=DOME_INTENSITY, visible_in_primary_ray=True)
    dome.func("/World/DomeLight", dome)
    light = sim_utils.DistantLightCfg(intensity=KEY_INTENSITY, angle=0.5)
    light.func("/World/KeyLight", light, orientation=(0.87, 0.2, -0.3, 0.3))
    # visual-only tabletop, top face TABLE_HEIGHT above the bases' mounting surface (no collider, so it can't push on the arms)
    spawn_table("/World/Table", size=(1.6, 2.2, 0.02), center=(0.55, -0.36, TABLE_HEIGHT - 0.0101))

    robot = robot_cfg(out)
    arm1 = Articulation(robot.replace(prim_path="/World/Arm1"))
    pos2, rot2 = pose_from_matrix(cal.T_arm1_arm2)
    arm2_cfg = robot.replace(prim_path="/World/Arm2")
    arm2_cfg.init_state = arm2_cfg.init_state.replace(pos=pos2, rot=rot2)
    arm2 = Articulation(arm2_cfg)
    apply_part_colors()

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

    sim, arms, d455, overview = build_scene(cal, out)
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
