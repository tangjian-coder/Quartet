"""GoPro-IMU adjacent-frame alignment check: image displacement (phase correlation) vs IMU pose-difference prediction

Renders adjacent frames locked to their own mid poses, measures image displacement, and
compares with the quaternion-difference prediction (RMSE residual). Validates the pose chain.

Note: adjacent-frame differences are insensitive to time_offset, so this is a whole-chain
diagnostic only (apple3 baseline: RMSE ~13px, dominated by translation parallax).

Usage: python3 adapters/gopro_imu/gopro_imu_align_check.py <seq_dir> <out_dir> [--shift S] [--step K]
  seq_dir: sequence dir (parent must contain camera_parameters.h5)
"""
import sys
import os
import argparse

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # same dir: gopro_common + gopro_imu_* modules

from gopro_common import quat_mul, quat_conj, load_gopro_calib, warp_pixels
from gopro_imu_to_input import build_pose_chain


def _warp_img(img, calib, quat_at, t0, q_out, W, H):
    """Lock the image to q_out through the shared inverse warper (single implementation)."""
    mx, my = warp_pixels(calib, quat_at, t0, q_out, W, H, row_ref="output_row")
    mx = np.nan_to_num(mx, nan=-1.0)
    my = np.nan_to_num(my, nan=-1.0)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--shift", type=int, default=0, help="frame-level time_offset shift (frames, image vs IMU)")
    ap.add_argument("--step", type=int, default=6, help="sampling stride")
    args = ap.parse_args()

    # camera calibration always read from dataset-root camera_parameters.h5
    calib = load_gopro_calib(os.path.join(os.path.dirname(args.seq_dir), "camera_parameters.h5"))

    ch = build_pose_chain(args.seq_dir)     # same chain the adapter uses
    time_offset, quat_at = ch["time_offset"], ch["quat_at"]

    cap = cv2.VideoCapture(f"{args.seq_dir}/video.mp4")
    fps = 29.97
    W, H = int(calib["size"][0]), int(calib["size"][1])
    # central 60% region for correlation (avoid edge black borders / rotated-out content)
    cy0, cy1, cx0, cx1 = H // 5, H * 4 // 5, W // 5, W * 4 // 5

    prev_off = args.shift * (1.0 / fps)
    errs, resps = [], []
    i = 0
    while True:
        ok, f1 = cap.read()
        if not ok:
            break
        if i % args.step != 0:
            i += 1
            continue
        ok, f2 = cap.read()
        if not ok:
            break
        t1 = i / fps + time_offset + prev_off
        t2 = (i + 1) / fps + time_offset + prev_off
        q1 = quat_at(t1 + calib["readout"] / 2)
        q2 = quat_at(t2 + calib["readout"] / 2)
        # Same map the renderer uses, locked to each frame's own mid pose (row_ref="output_row"
        # is the legacy single-pass variant; the render default for this mode is identical)
        w1 = _warp_img(f1, calib, quat_at, t1, q1, W, H)
        w2 = _warp_img(f2, calib, quat_at, t2, q2, W, H)
        g1 = cv2.cvtColor(w1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(w2, cv2.COLOR_BGR2GRAY)
        # phase correlation: (dx,dy) = content displacement of frame 2 relative to frame 1
        (dx, dy), resp = cv2.phaseCorrelate(g1[cy0:cy1, cx0:cx1].astype(np.float32),
                                            g2[cy0:cy1, cx0:cx1].astype(np.float32))
        # IMU prediction from the w2c difference rotation (axis-angle).
        # Derivation: content moves by M = R(q2)*R(q1)^T (w2c: pc = R*pw); for M ~ I+[ang]x
        # the small-angle center displacement is (fx*ang_y, -fx*ang_x)
        dq = quat_mul(q2, quat_conj(q1))  # camera-1 frame -> camera-2 frame
        ang = 2 * np.array([dq[1], dq[2], dq[3]]) / max(dq[0], 1e-12)
        px_pred, py_pred = calib["fx"] * ang[1], -calib["fx"] * ang[0]
        errs.append(np.hypot(dx - px_pred, dy - py_pred))
        resps.append(resp)
        i += 1
    cap.release()

    errs = np.array(errs)
    n = len(errs)
    bad = int(np.ceil(n * 0.1))
    if bad > 0:
        errs_sorted = np.sort(errs)
        mean_90 = errs_sorted[: n - bad].mean() if n - bad > 0 else np.nan
    else:
        mean_90 = np.nan
    print(f"shift={args.shift}: pairs={n} RMSE={np.sqrt((errs ** 2).mean()):.2f}px "
          f"mean={errs.mean():.2f}px p90(top-10% removed)={mean_90:.2f}px "
          f"resp-mean={np.mean(resps):.3f}")


if __name__ == "__main__":
    main()
