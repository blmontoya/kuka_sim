"""Score several render_calibration_cell.py outputs against the same reference masks.

The reference is the SAM2 masks (specks under MIN_BLOB_PX removed) of the gripper-aware EasyHec runs (arm + gripper), so renders made with different
calibrations or gripper models are compared on equal terms. Per arm and pose it reports:

    iou        overlap of the sim silhouette (arm + gripper) with the SAM2 mask
    edge_px    mean distance between the two outlines [px], both directions averaged
    grip_iou   IoU inside a window around the sim gripper (how well the gripper itself lines up)
    grip_px    edge_px restricted to that window

and writes <first out>/../gripper_crops.png: the gripper region of each pose with every render's outline.

    pixi r -e isaaclab-gpu python scripts/compare_overlays.py outputs/compare/bare_22mm outputs/compare/gripper_22mm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

SIM2REAL = Path(__file__).resolve().parents[2]
DEFAULT_RUNS = [SIM2REAL / "kuka_easyhec" / "results" / f"gripper_{a}" for a in ("arm1", "arm2")]
MIN_BLOB_PX = 500  # SAM2 masks have stray specks (people, pendants); smaller pieces are dropped from the reference
WINDOW_PX = 40  # gripper window = sim gripper mask grown by this many pixels
CROP_COLORS = [(255, 80, 80), (80, 220, 80), (80, 160, 255), (255, 200, 0), (220, 80, 255), (0, 220, 220)]  # RGB

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("outs", type=Path, nargs="+", help="render_calibration_cell.py output folders (with sim_masks.npz)")
parser.add_argument("--runs", type=Path, nargs=2, default=DEFAULT_RUNS, metavar=("ARM1_RUN", "ARM2_RUN"))
parser.add_argument("--exclude", nargs="*", default=["arm2:6"], help="arm:pose pairs left out of the means (bad masks)")
args = parser.parse_args()


def reference(runs: list[Path]) -> tuple[np.ndarray, dict[str, dict[int, np.ndarray]]]:
    """Photos from arm1's run; each run's masks matched back to the photo they were drawn on."""
    images = np.load(runs[0] / "dataset.npz")["images"]
    refs = {}
    for arm, run in zip(("arm1", "arm2"), runs):
        run_images = np.load(run / "dataset.npz")["images"]
        masks = np.load(run / arm / "masks.npy") > 0.5
        refs[arm] = {next(i for i, ref in enumerate(images) if np.array_equal(ref, img)): drop_specks(m) for img, m in zip(run_images, masks)}
    return images, refs


def drop_specks(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= MIN_BLOB_PX]
    return np.isin(labels, keep)


def edge(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)


def edge_distance(a: np.ndarray, b: np.ndarray, window: np.ndarray | None = None) -> float:
    """Mean symmetric distance [px] between the outlines of masks a and b (optionally only inside window)."""
    ea, eb = edge(a), edge(b)
    if window is not None:
        ea, eb = ea & window, eb & window
    if not ea.any() or not eb.any():
        return float("nan")
    da = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 5)  # distance to b's outline
    db = cv2.distanceTransform((~ea).astype(np.uint8), cv2.DIST_L2, 5)
    return float((da[ea].mean() + db[eb].mean()) / 2)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else float("nan")


def main():
    images, refs = reference(args.runs)
    excluded = {(e.split(":")[0], int(e.split(":")[1])) for e in args.exclude}
    sims = {out: np.load(out / "sim_masks.npz") for out in args.outs}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * WINDOW_PX + 1, 2 * WINDOW_PX + 1))

    report = {}
    for out, sm in sims.items():
        rows = []
        for arm, ref_by_pose in refs.items():
            for i, ref in sorted(ref_by_pose.items()):
                full = sm[arm][i] | sm[f"{arm}_gripper"][i]
                win = cv2.dilate(sm[f"{arm}_gripper"][i].astype(np.uint8), kernel).astype(bool)
                rows.append({
                    "arm": arm, "pose": i, "excluded": (arm, i) in excluded,
                    "iou": iou(full, ref), "edge_px": edge_distance(full, ref),
                    "grip_iou": iou(full & win, ref & win), "grip_px": edge_distance(full, ref, win),
                })
        report[str(out)] = rows

    keys = ("iou", "edge_px", "grip_iou", "grip_px")
    print(f"{'render':<28}{'arm':<6}" + "".join(f"{k:>10}" for k in keys) + "   (means; excluded: " + " ".join(args.exclude) + ")")
    for out, rows in report.items():
        for arm in refs:
            sel = [r for r in rows if r["arm"] == arm and not r["excluded"]]
            print(f"{Path(out).name:<28}{arm:<6}" + "".join(f"{np.nanmean([r[k] for r in sel]):>10.3f}" for k in keys))
    print("\nper pose (iou / edge_px / grip_px):")
    for arm in refs:
        for i in sorted(refs[arm]):
            cells = []
            for rows in report.values():
                r = next(r for r in rows if r["arm"] == arm and r["pose"] == i)
                cells.append(f"{r['iou']:.3f}/{r['edge_px']:5.1f}/{r['grip_px']:5.1f}")
            flag = "  (excluded)" if (arm, i) in excluded else ""
            print(f"  {arm} pose {i}: " + "   ".join(cells) + flag)

    base = Path(args.outs[0]).parent
    (base / "compare_metrics.json").write_text(json.dumps(report, indent=2))

    # gripper close-ups: real photo, SAM2 outline white, one coloured outline per render
    tiles = []
    for arm in refs:
        for i in sorted(refs[arm]):
            img = images[i].copy()
            cs, _ = cv2.findContours(refs[arm][i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cs, -1, (255, 255, 255), 1, cv2.LINE_AA)
            for k, sm in enumerate(sims.values()):
                full = (sm[arm][i] | sm[f"{arm}_gripper"][i]).astype(np.uint8)
                cs, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cs, -1, CROP_COLORS[k % len(CROP_COLORS)], 1, cv2.LINE_AA)
            ys, xs = np.nonzero(sims[args.outs[0]][f"{arm}_gripper"][i])
            if len(xs) == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            x0, y0 = np.clip(cx - 120, 0, img.shape[1] - 240), np.clip(cy - 90, 0, img.shape[0] - 180)
            tile = cv2.resize(img[y0:y0 + 180, x0:x0 + 240], (480, 360), interpolation=cv2.INTER_CUBIC)
            cv2.putText(tile, f"{arm} pose {i}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(tile)
    legend = np.zeros((40 * (len(sims) + 1), 480, 3), np.uint8)
    cv2.putText(legend, "white: SAM2 mask", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    for k, out in enumerate(sims):
        cv2.putText(legend, Path(out).name, (8, 68 + 40 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CROP_COLORS[k % len(CROP_COLORS)], 2, cv2.LINE_AA)
    cols = 4
    tiles.append(cv2.resize(legend, (480, 360)) if legend.shape[0] > 360 else np.pad(legend, ((0, 360 - legend.shape[0]), (0, 0), (0, 0))))
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
    sheet = np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])
    cv2.imwrite(str(base / "gripper_crops.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"\nwrote {base / 'compare_metrics.json'} and {base / 'gripper_crops.png'}")


if __name__ == "__main__":
    main()
