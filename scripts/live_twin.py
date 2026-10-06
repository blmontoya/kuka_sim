"""Live digital twin of the two-arm KUKA cell: the sim follows the real arms' joint angles (ROS 2 JointState)
and the live D455 feed is blended with the simulated D455 view, so any mismatch shows up immediately.

Joint angles come from one sensor_msgs/JointState topic per arm (e.g. lbr_fri_ros2_stack in FRI monitoring
mode). Joints are matched by name suffix A1..A7, so "A1", "lbr_A1" and "arm1_A1" all work.

Run from kuka_sim/ (needs the ROS 2 environment):

    pixi r -e isaaclab-ros2-gpu python scripts/live_twin.py
    # without the robots, replay the calibration poses as fake joint data in a second terminal:
    pixi r -e isaaclab-ros2-gpu python scripts/fake_joint_states.py

The "Live twin" window shows the blend; drag "sim" between 0 (camera only) and 1 (sim only).
--headless --snapshot-dir DIR saves the blend every --snapshot-every seconds instead (for testing).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--runs", type=Path, nargs=2, default=None, metavar=("ARM1_RUN", "ARM2_RUN"),
                    help="gripper-aware EasyHec runs (default: kuka_easyhec/results/gripper_arm1 gripper_arm2)")
parser.add_argument("--calib-dir", type=Path, default=None, help="use a combined calibration folder instead")
parser.add_argument("--arm1-topic", default="/arm1/joint_states")
parser.add_argument("--arm2-topic", default="/arm2/joint_states")
parser.add_argument("--serial", default="234222302015", help="RealSense serial of the calibrated D455")
parser.add_argument("--no-camera", action="store_true", help="sim only, no live D455 feed")
parser.add_argument("--gripper-opening", type=float, default=0.06, help="fingertip gap to show [m], 0..0.074")
parser.add_argument("--flange-mm", type=float, default=None, help="override the media-flange thickness [mm]")
parser.add_argument("--blend", type=float, default=0.5, help="initial sim weight in the blend, 0..1")
parser.add_argument("--snapshot-dir", type=Path, default=None, help="save blend_<n>.png here every --snapshot-every s")
parser.add_argument("--snapshot-every", type=float, default=5.0)
parser.add_argument("--duration", type=float, default=0.0, help="stop after this many seconds (0 = until closed)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import re  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402

from kuka_sim.cell import DEFAULT_GRIPPER_RUNS, FINGER_MAX, CellCalibration  # noqa: E402
from kuka_sim.scene import build_cell, set_pose, to_calibrated_k  # noqa: E402

AXIS = re.compile(r"A([1-7])$")


class JointListener(Node):
    """Keeps the latest A1..A7 (degrees) and receive time per arm."""

    def __init__(self, topics: dict[str, str]):
        super().__init__("kuka_live_twin")
        self.q_deg: dict[str, np.ndarray] = {}
        self.stamp: dict[str, float] = {}
        for arm, topic in topics.items():
            self.create_subscription(JointState, topic, lambda msg, a=arm: self._on_msg(a, msg), qos_profile_sensor_data)
            self.get_logger().info(f"{arm}: listening on {topic}")

    def _on_msg(self, arm: str, msg: JointState):
        q = np.full(7, np.nan)
        for name, pos in zip(msg.name, msg.position):
            m = AXIS.search(name)
            if m:
                q[int(m.group(1)) - 1] = np.rad2deg(pos)
        if not np.isnan(q).any():
            self.q_deg[arm], self.stamp[arm] = q, time.monotonic()


class Camera455:
    """Live colour frames from one RealSense at the calibration resolution."""

    def __init__(self, serial: str, width: int, height: int):
        import pyrealsense2 as rs

        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)
        self.pipe.start(cfg)
        self.last = np.zeros((height, width, 3), np.uint8)

    def latest(self) -> np.ndarray:
        frames = self.pipe.poll_for_frames()  # non-blocking; keep the previous frame if none is new
        if frames:
            color = frames.get_color_frame()
            if color:
                self.last = np.asanyarray(color.get_data()).copy()
        return self.last

    def stop(self):
        self.pipe.stop()


class BlendWindow:
    """omni.ui window showing the camera/sim blend with a slider and per-arm status."""

    def __init__(self, width: int, height: int, blend: float):
        import omni.ui as ui

        self.provider = ui.ByteImageProvider()
        self.window = ui.Window("Live twin: D455 vs sim", width=width // 2 + 20, height=height // 2 + 90)
        with self.window.frame:
            with ui.VStack(spacing=4):
                with ui.HStack(height=22):
                    ui.Label("sim", width=40)
                    self.slider = ui.FloatSlider(min=0.0, max=1.0)
                    self.slider.model.set_value(blend)
                self.status = ui.Label("", height=20)
                ui.ImageWithProvider(self.provider)

    @property
    def blend(self) -> float:
        return self.slider.model.as_float

    def show(self, rgb: np.ndarray, status: str):
        rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, np.uint8)])
        self.provider.set_data_array(rgba, [rgba.shape[1], rgba.shape[0]])
        self.status.text = status


def main():
    if args.calib_dir is not None:
        cal = CellCalibration.load(args.calib_dir)
    else:
        cal = CellCalibration.load_gripper_runs(*(args.runs or DEFAULT_GRIPPER_RUNS))
    finger_q = float(np.clip(args.gripper_opening / 2, 0.0, FINGER_MAX))
    sim, arms, d455, _ = build_cell(cal, args.device, args.flange_mm, args.snapshot_dir, overview=False)
    for a, arm in arms.items():  # start at calibration pose 0 until joint data arrives
        set_pose(arm, cal.joints_deg[a][0], finger_q)

    rclpy.init()
    listener = JointListener({"arm1": args.arm1_topic, "arm2": args.arm2_topic})
    cam = None if args.no_camera else Camera455(args.serial, cal.width, cal.height)
    window = None if args.headless else BlendWindow(cal.width, cal.height, args.blend)
    if args.snapshot_dir:
        args.snapshot_dir.mkdir(parents=True, exist_ok=True)

    t0 = last_snap = time.monotonic()
    n_snap = 0
    try:
        while app.is_running():
            rclpy.spin_once(listener, timeout_sec=0.0)
            for a, arm in arms.items():
                if a in listener.q_deg:
                    set_pose(arm, listener.q_deg[a], finger_q)
            sim.step(render=True)
            d455.update(0.0, force_recompute=True)

            sim_rgb = to_calibrated_k(d455.data.output["rgb"][0, ..., :3].cpu().numpy(), cal, nearest=False)
            alpha = window.blend if window else args.blend
            out = sim_rgb if cam is None else cv2.addWeighted(cam.latest(), 1.0 - alpha, sim_rgb, alpha, 0)
            now = time.monotonic()
            status = "   ".join(
                f"{a}: {'no data' if a not in listener.stamp else f'{now - listener.stamp[a]:.1f}s old'}" for a in arms
            )
            if window:
                window.show(out, status)
            if args.snapshot_dir and now - last_snap >= args.snapshot_every:
                cv2.putText(out, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(str(args.snapshot_dir / f"blend_{n_snap}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
                print(f"snapshot {n_snap}: {status}", flush=True)
                n_snap, last_snap = n_snap + 1, now
            if args.duration and now - t0 > args.duration:
                break
    finally:
        if cam:
            cam.stop()
        listener.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
