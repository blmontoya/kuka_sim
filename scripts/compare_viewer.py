"""Fade between the real calibration photos and the Isaac renders written by render_calibration_cell.py.

    conda activate easyhec          # needs an OpenCV with GUI support (the pixi env's is headless)
    python scripts/compare_viewer.py --out outputs/calibration_cell_connected

Slider "sim %": 0 = real photo, 100 = sim render. Slider "pose": calibration pose.
Keys: left/right arrows (or a/d) = previous/next pose, up/down arrows (or w/s) = more/less sim,
      o = toggle the outline overlay, q / Esc = quit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_CALIB_DIR = Path(__file__).resolve().parents[2] / "camera_calibration_20260924"  # same as kuka_sim.cell

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", type=Path, default=Path("outputs/calibration_cell"), help="render_calibration_cell.py output folder")
parser.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR, help="EasyHec folder with dataset.npz")
args = parser.parse_args()

real = np.load(args.calib_dir / "dataset.npz")["images"][..., ::-1]  # RGB -> BGR
n = len(real)
sims = [cv2.imread(str(args.out / f"sim_{i}.png")) for i in range(n)]
overlays = [cv2.imread(str(args.out / f"overlay_{i}.png")) for i in range(n)]
if any(s is None for s in sims):
    raise SystemExit(f"missing sim_<i>.png in {args.out}; run render_calibration_cell.py first")

WIN = "real vs sim"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1280, 760)
cv2.createTrackbar("pose", WIN, 0, n - 1, lambda _: None)
cv2.createTrackbar("sim %", WIN, 50, 100, lambda _: None)
show_overlay = False
# arrow-key codes from cv2.waitKeyEx: Qt backend, GTK backend
LEFT, RIGHT = {0x1000012, 65361, ord("a")}, {0x1000014, 65363, ord("d")}
UP, DOWN = {0x1000013, 65362, ord("w")}, {0x1000015, 65364, ord("s")}

while cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) >= 1:
    i = cv2.getTrackbarPos("pose", WIN)
    a = cv2.getTrackbarPos("sim %", WIN) / 100.0
    base = overlays[i] if show_overlay and overlays[i] is not None else np.ascontiguousarray(real[i])
    img = cv2.addWeighted(base, 1.0 - a, sims[i], a, 0)
    cv2.putText(img, f"pose {i}/{n - 1}   sim {a:.0%}   [<- ->] pose   [up/down] fade   [o] outlines   [q] quit", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(WIN, img)
    key = cv2.waitKeyEx(30)
    if key in (ord("q"), 27):
        break
    if key == ord("o"):
        show_overlay = not show_overlay
    if key in LEFT or key in RIGHT:
        cv2.setTrackbarPos("pose", WIN, (i + (1 if key in RIGHT else -1)) % n)
    if key in UP or key in DOWN:
        cv2.setTrackbarPos("sim %", WIN, int(np.clip(round(a * 100) + (10 if key in UP else -10), 0, 100)))
cv2.destroyAllWindows()
