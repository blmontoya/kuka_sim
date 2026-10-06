"""Shared Isaac Lab scene for the calibrated two-arm KUKA cell: lights, wood table, both iiwa7 + HandUMI arms
(coloured per arm) and the calibrated D455. Used by scripts/render_calibration_cell.py and scripts/live_twin.py.

Import this only after the Isaac Sim app is running (``AppLauncher``).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg

from kuka_sim.cell import ARM_JOINTS, ASSET_DIR, FINGER_JOINTS, IIWA7_HANDUMI_CFG, CellCalibration, pose_from_matrix

TABLE_HEIGHT = 0.004  # tabletop above the base mounting surface [m] (measured)
PAD = 16  # extra rendered pixels per side, so the principal-point shift never samples outside the render

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
TABLE_WOOD_TEXTURE = Path(__file__).resolve().parent / "assets/materials/Birch/Birch_BaseColor.png"
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


def robot_cfg(flange_mm: float | None = None, out: Path | None = None):
    """IIWA7_HANDUMI_CFG, or a copy (written to `out`) whose media flange is `flange_mm` thick (gripper moved to match)."""
    if flange_mm is None:
        return IIWA7_HANDUMI_CFG
    import xml.etree.ElementTree as ET

    t = flange_mm / 1000
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
    path = Path(out) / f"iiwa7_handumi_flange{flange_mm:g}mm.urdf"
    tree.write(path, xml_declaration=True, encoding="utf-8")
    return IIWA7_HANDUMI_CFG.replace(spawn=IIWA7_HANDUMI_CFG.spawn.replace(asset_path=str(path)))


def build_cell(cal: CellCalibration, device: str, flange_mm: float | None = None, out: Path | None = None,
               overview: bool = True):
    """Lights, table, both arms (arm2 at its calibrated pose), the calibrated D455 and optionally an overview camera.

    Returns (sim, {"arm1": .., "arm2": ..}, d455, overview_camera_or_None).
    """
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 60, device=device))

    # the visible dome is the background: grey, and brighter with DOME_INTENSITY.
    # visible_in_primary_ray=False makes the background black (the dome still lights the scene)
    dome = sim_utils.DomeLightCfg(intensity=DOME_INTENSITY, visible_in_primary_ray=True)
    dome.func("/World/DomeLight", dome)
    light = sim_utils.DistantLightCfg(intensity=KEY_INTENSITY, angle=0.5)
    light.func("/World/KeyLight", light, orientation=(0.87, 0.2, -0.3, 0.3))
    # visual-only tabletop, top face TABLE_HEIGHT above the bases' mounting surface (no collider, so it can't push on the arms)
    spawn_table("/World/Table", size=(1.6, 2.2, 0.02), center=(0.55, -0.36, TABLE_HEIGHT - 0.0101))

    robot = robot_cfg(flange_mm, out)
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

    if overview:
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
    if overview:
        overview.set_world_poses_from_view(
            torch.tensor([[1.9, 0.9, 1.35]], device=sim.device), torch.tensor([[0.35, -0.36, 0.25]], device=sim.device)
        )
    return sim, {"arm1": arm1, "arm2": arm2}, d455, overview or None


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
