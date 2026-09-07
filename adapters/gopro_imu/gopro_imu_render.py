"""GoPro-IMU lock-frame-0 exact renderer (per-pixel, no grid interpolation)

Pipeline: output frame (frame-0 mid pose view) -> row pose rotation (incl. RS) -> atan
distortion -> sample original frame. Same as gopro_imu_validate.warp_virtual with
q_ref = frame-0 mid pose.

Usage: python3 adapters/gopro_imu/gopro_imu_render.py <seq_dir> <out_dir> [--max-frames N] [--out-name name]
  seq_dir: sequence dir (parent must contain camera_parameters.h5)
"""
import sys
import os
import time
import subprocess

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # same dir: gopro_common + gopro_imu_* modules

from gopro_common import axis_angle_to_R, integrate_gyro, make_quat_at, fix_timestamps, load_gopro_calib
from gopro_imu_to_input import load_imu, load_calib
from gopro_imu_validate import warp_virtual

W, H = 1920, 1080


def main():
    seq_dir, out_dir = sys.argv[1], sys.argv[2]
    max_frames = 0
    if "--max-frames" in sys.argv:
        max_frames = int(sys.argv[sys.argv.index("--max-frames") + 1])
    out_name = "fix0_dense.mp4"
    if "--out-name" in sys.argv:
        out_name = sys.argv[sys.argv.index("--out-name") + 1]
    # camera calibration always read from dataset-root camera_parameters.h5
    calib = load_gopro_calib(os.path.join(os.path.dirname(seq_dir), "camera_parameters.h5"))

    # pose chain identical to the adapter
    t, acc, gyro = load_imu(seq_dir)
    gyro_rate_nom, time_offset, rot, gbias, abias = load_calib(seq_dir)
    R_gc = axis_angle_to_R(rot)
    # official timestamp fix: keep all samples, linear-fit uniform axis (only timestamps broken)
    t_fix = fix_timestamps(t)
    dt = t_fix[1] - t_fix[0]
    du = np.vstack([gyro, acc]).T
    qs = integrate_gyro((du[:, :3] + gbias) @ R_gc.T, dt)
    quat_at = make_quat_at(qs, 1.0 / dt)

    # video info via ffmpeg
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    r = subprocess.run([ff, "-i", f"{seq_dir}/video.mp4"], capture_output=True, text=True)
    fps, dur = 29.97, 0.0
    for line in r.stderr.splitlines():
        if "Video:" in line and "fps" in line:
            fps = float(line.split("fps")[0].split(",")[-1].strip())
        if "Duration:" in line:
            h, m, s = [float(x) for x in line.split("Duration:")[1].split(",")[0].strip().split(":")]
            dur = h * 3600 + m * 60 + s
    n_frames = int(dur * fps)
    print(f"fps {fps} duration {dur:.2f}s -> {n_frames} frames")

    q_ref0 = quat_at(time_offset + calib["readout"] / 2)  # frame-0 mid pose (lock-frame-0 reference)

    cap = cv2.VideoCapture(f"{seq_dir}/video.mp4")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    limit = max_frames or n_frames
    t0 = time.time()
    for i in range(limit):
        ok, frame = cap.read()
        if not ok:
            break
        t0i = i / fps + time_offset
        q_mid = quat_at(t0i + calib["readout"] / 2)
        w = warp_virtual(frame, calib, quat_at, t0i, q_mid, True, True, q_ref=q_ref0)
        out8 = np.clip(w, 0, 255).astype(np.uint8)
        # green center-crop rect (20% crop)
        x0, y0, x1, y1 = int(W * 0.1), int(H * 0.1), int(W * 0.9), int(H * 0.9)
        cv2.rectangle(out8, (x0, y0), (x1, y1), (0, 255, 0), 1)
        label = str(i)
        cv2.putText(out8, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out8, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(out8)
        if (i + 1) % 50 == 0 or i == limit - 1:
            el = time.time() - t0
            print(f"{i+1}/{limit} | {el:.0f}s | ETA {el/(i+1)*(limit-i-1):.0f}s", flush=True)
    cap.release()
    writer.release()
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
