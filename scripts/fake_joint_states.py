"""Publish fake JointState messages for both arms, smoothly cycling through the calibration poses, so
scripts/live_twin.py can be tested without the robots. Names and topics mimic lbr_fri_ros2_stack.

    pixi r -e isaaclab-ros2-gpu python scripts/fake_joint_states.py            # loops forever
    pixi r -e isaaclab-ros2-gpu python scripts/fake_joint_states.py --hold 3   # stay on pose 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

DATASET = Path(__file__).resolve().parents[2] / "kuka_easyhec" / "results" / "gripper_arm1" / "dataset.npz"

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dataset", type=Path, default=DATASET, help="EasyHec dataset.npz with joints_deg_arm1/arm2")
parser.add_argument("--rate", type=float, default=50.0, help="publish rate [Hz]")
parser.add_argument("--seconds-per-pose", type=float, default=6.0, help="time to move between consecutive poses")
parser.add_argument("--hold", type=int, default=None, help="publish only this calibration pose")
args = parser.parse_args()

data = np.load(args.dataset)
poses = {a: np.deg2rad(data[f"joints_deg_{a}"]) for a in ("arm1", "arm2")}
n = len(poses["arm1"])


def main():
    rclpy.init()
    node = Node("fake_kuka_joint_states")
    pubs = {a: node.create_publisher(JointState, f"/{a}/joint_states", qos_profile_sensor_data) for a in poses}
    t = 0.0

    def tick():
        nonlocal t
        if args.hold is not None:
            i, j, s = args.hold, args.hold, 0.0
        else:
            phase = t / args.seconds_per_pose
            i, s = int(phase) % n, phase % 1.0
            j = (i + 1) % n
            s = min(1.0, s / 0.6)  # move for 60 % of the time, then rest on the pose
            s = 0.5 - 0.5 * np.cos(np.pi * s)  # ease in/out
        for a, pub in pubs.items():
            msg = JointState()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.name = [f"{a}_A{k}" for k in range(1, 8)]
            msg.position = ((1 - s) * poses[a][i] + s * poses[a][j]).tolist()
            pub.publish(msg)
        t += 1.0 / args.rate

    node.create_timer(1.0 / args.rate, tick)
    node.get_logger().info(f"publishing /arm1/joint_states and /arm2/joint_states ({n} poses from {args.dataset})")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
