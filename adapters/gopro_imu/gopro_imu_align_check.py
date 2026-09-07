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

from gopro_common import (axis_angle_to_R, integrate_gyro, make_quat_at, fix_timestamps,
                          quat_mul, quat_conj, quat_to_R, load_gopro_calib)
from gopro_imu_to_input import load_imu, load_calib
from gopro_imu_validate import warp_virtual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--shift", type=int, default=0, help="frame-level time_offset shift (frames, image vs IMU)")
    ap.add_argument("--step", type=int, default=6, help="sampling stride")
    args = ap.parse_args()

    # camera calibration always read from dataset-root camera_parameters.h5
    calib = load_gopro_calib(os.path.join(os.path.dirname(args.seq_dir), "camera_parameters.h5"))

    t, acc, gyro = load_imu(args.seq_dir)
    gyro_rate_nom, time_offset, rot, gbias, abias = load_calib(args.seq_dir)
    R_gc = axis_angle_to_R(rot)
    # official timestamp fix: keep all samples, linear-fit uniform axis (only timestamps broken)
    t_fix = fix_timestamps(t)
    dt = t_fix[1] - t_fix[0]
    du = np.vstack([gyro, acc]).T
    qs = integrate_gyro((du[:, :3] + gbias) @ R_gc.T, dt)
    quat_at = make_quat_at(qs, 1.0 / dt)

    cap = cv2.VideoCapture(f"{args.seq_dir}/video.mp4")
    fps = 29.97
    W, H = 1920, 1080
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
        w1 = warp_virtual(f1, calib, quat_at, t1, q1, True, True, q_ref=q1)
        w2 = warp_virtual(f2, calib, quat_at, t2, q2, True, True, q_ref=q2)
        g1 = cv2.cvtColor(np.clip(w1, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(np.clip(w2, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        # phase correlation: (dx,dy) = content displacement of frame 2 relative to frame 1
        (dx, dy), resp = cv2.phaseCorrelate(g1[cy0:cy1, cx0:cx1].astype(np.float32),
                                            g2[cy0:cy1, cx0:cx1].astype(np.float32))
        # IMU prediction from c2w difference rotation axis-angle.
        # Derivation: H = K·R(conj(q1) x q2)·K⁻¹; small-angle center displacement = K·(ang x e_z)
        #            = (fx·ang_y, -fx·ang_x)
        dq = quat_mul(q1, quat_conj(q2))  # frame-1 frame -> frame-2 frame (c2w difference)
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
