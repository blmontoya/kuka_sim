"""Render the calibrated cell with randomized arm appearance (visual domain randomization).

For each sample this puts both arms in a calibration pose, gives every arm/gripper link a random flat colour or a
random image texture (see kuka_sim/visual_dr.py), renders the simulated D455 and writes:

    dr_<i>.png         the randomized D455 render
    contact_sheet.png  all renders in one image
    samples.json       the pose and per-link material sample behind each render

Run from kuka_sim/:

    # colours only
    pixi r -e isaaclab-gpu python scripts/render_visual_dr.py --headless
    # colours + procedurally generated textures (written to <out>/textures)
    pixi r -e isaaclab-gpu python scripts/render_visual_dr.py --headless --procedural-textures 64
    # colours + your own images (any folder of png/jpg, e.g. the DTD texture dataset)
    pixi r -e isaaclab-gpu python scripts/render_visual_dr.py --headless --texture-dir /path/to/images

Drop --headless to see it in the Isaac Sim window: after the batch, the arms get one random look (one "episode")
and hold it until you close the window (--num-samples 0 skips the batch; --live-every N starts a new episode every N s).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--runs", type=Path, nargs=2, default=None, metavar=("ARM1_RUN", "ARM2_RUN"),
                    help="gripper-aware EasyHec runs (default: kuka_easyhec/results/gripper_arm1 gripper_arm2)")
parser.add_argument("--out", type=Path, default=Path("outputs/visual_dr"))
parser.add_argument("--num-samples", type=int, default=16)
parser.add_argument("--texture-dir", type=Path, default=None, help="folder of images to use as textures")
parser.add_argument("--procedural-textures", type=int, default=0, help="generate this many random textures instead")
parser.add_argument("--texture-prob", type=float, default=0.5, help="chance a link gets a texture instead of a colour")
parser.add_argument("--per-arm", action="store_true", help="one sample per arm instead of one per link")
parser.add_argument("--gripper-opening", type=float, default=0.06, help="fingertip gap [m], 0..0.074")
parser.add_argument("--render-frames", type=int, default=30, help="RTX frames per sample (textures stream in)")
parser.add_argument("--warmup-frames", type=int, default=120)
parser.add_argument("--seed", type=int, default=None, help="fix the random looks (default: different every run)")
parser.add_argument("--pose", type=int, default=None, help="hold this calibration pose instead of a random one per sample")
parser.add_argument("--live-every", type=float, default=0.0,
                    help="with the GUI: seconds per episode (new look each); 0 = one look, held")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import json  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from kuka_sim.cell import DEFAULT_GRIPPER_RUNS, FINGER_MAX, CellCalibration  # noqa: E402
from kuka_sim.scene import build_cell, set_pose, to_calibrated_k  # noqa: E402
from kuka_sim.visual_dr import ArmVisualRandomizer, VisualDRCfg, make_procedural_textures  # noqa: E402


def main():
    cal = CellCalibration.load_gripper_runs(*(args.runs or DEFAULT_GRIPPER_RUNS))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    finger_q = float(np.clip(args.gripper_opening / 2, 0.0, FINGER_MAX))
    texture_dir = args.texture_dir
    if args.procedural_textures:
        texture_dir = make_procedural_textures(out / "textures", n=args.procedural_textures, seed=args.seed)

    sim, arms, d455, _ = build_cell(cal, args.device, out=out, overview=False)
    dr = ArmVisualRandomizer(VisualDRCfg(texture_dir=texture_dir, texture_prob=args.texture_prob,
                                         per_link=not args.per_arm, seed=args.seed))
    print(f"randomizing {len(dr.visuals)} link visuals, {len(dr.textures)} textures", flush=True)
    for _ in range(args.warmup_frames):
        sim.step(render=True)

    rng = np.random.default_rng(args.seed)
    renders, log = [], []
    for i in range(args.num_samples):
        pose = int(rng.integers(len(cal.images))) if args.pose is None else args.pose
        for a, arm in arms.items():
            set_pose(arm, cal.joints_deg[a][pose], finger_q)
        samples = dr.randomize()
        for _ in range(args.render_frames):
            sim.step(render=True)
            for arm in arms.values():
                arm.update(sim.get_physics_dt())
        d455.update(0.0, force_recompute=True)
        rgb = to_calibrated_k(d455.data.output["rgb"][0, ..., :3].cpu().numpy(), cal, nearest=False)
        cv2.imwrite(str(out / f"dr_{i}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        renders.append(rgb)
        log.append({"sample": i, "pose": pose, "links": samples})
        print(f"sample {i}: pose {pose}", flush=True)

    if renders:
        cols = 4
        thumbs = [cv2.resize(r, (640, 360), interpolation=cv2.INTER_AREA) for r in renders]
        thumbs += [np.zeros_like(thumbs[0])] * (-len(thumbs) % cols)
        sheet = np.vstack([np.hstack(thumbs[r:r + cols]) for r in range(0, len(thumbs), cols)])
        cv2.imwrite(str(out / "contact_sheet.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        (out / "samples.json").write_text(json.dumps(log, indent=2))
        print(f"wrote {out}", flush=True)

    if not args.headless:
        # one random look (one "episode"), held until the window is closed; --live-every > 0 starts a new
        # episode (new look, and a new pose unless --pose) every that many seconds
        sim.set_camera_view(eye=[1.9, 0.9, 1.35], target=[0.35, -0.36, 0.25])
        if args.live_every > 0:
            print(f"live: new episode every {args.live_every:g} s; close the Isaac Sim window to exit", flush=True)
        else:
            print("live: holding one random look; close the Isaac Sim window to exit", flush=True)
        steps_per_sample = round(args.live_every / sim.get_physics_dt()) if args.live_every > 0 else None
        step = 0
        while app.is_running():
            if step == 0 or (steps_per_sample and step % steps_per_sample == 0):
                pose = int(rng.integers(len(cal.images))) if args.pose is None else args.pose
                for a, arm in arms.items():
                    set_pose(arm, cal.joints_deg[a][pose], finger_q)
                dr.randomize()
            sim.step(render=True)
            step += 1


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
