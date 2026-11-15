"""GoPro-IMU renderer: lock-frame-0 (per-pixel) or stabilized output from optimized poses.

Pipeline: output frame (q_out view) -> row pose rotation (incl. RS) -> atan distortion ->
sample original frame, always through common.warp_pixels (single inverse-warp path).
One mode must be chosen explicitly:
  --fix0           : q_out = frame-0 mid pose. Verification: a correct pose chain / time
                     offset / calibration holds the picture still, a wrong one drifts.
  --opt-quat file  : q_out = opt[i] (already w2c, same convention as quat_at). Final render.
Both modes iterate the source row (row_ref="pixel_row") so the de-RS solve is self-consistent.
Usage: python3 adapters/gopro_imu/gopro_imu_render.py <seq_dir> <out_dir> (--fix0 | --opt-quat opt.csv)
       [--max-frames N] [--out-name name] [--f0 N] [--f1 N] [--workers N]
  seq_dir: sequence dir (parent must contain camera_parameters.h5)
"""
import sys
import os
import time
import subprocess

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # same dir: gopro_common + gopro_imu_*; common.py is one level up

from gopro_common import load_gopro_calib, warp_pixels
from gopro_imu_to_input import build_pose_chain


def main():
    seq_dir, out_dir = sys.argv[1], sys.argv[2]
    max_frames = 0
    if "--max-frames" in sys.argv:
        max_frames = int(sys.argv[sys.argv.index("--max-frames") + 1])
    opt_quat_path = None
    if "--opt-quat" in sys.argv:
        opt_quat_path = sys.argv[sys.argv.index("--opt-quat") + 1]
    fix0 = "--fix0" in sys.argv
    # fix0 is a verification mode, not a default: the two modes answer different questions
    if fix0 and opt_quat_path:
        sys.exit("--fix0 and --opt-quat are mutually exclusive")
    if not fix0 and not opt_quat_path:
        sys.exit("pick a mode: --fix0 (lock frame 0, verify the pose chain) "
                 "or --opt-quat <csv> (stabilized render)")
    # the default file name states which question the output answers
    out_name = "fix0_dense.mp4" if fix0 else "stab_dense.mp4"
    if "--out-name" in sys.argv:
        out_name = sys.argv[sys.argv.index("--out-name") + 1]
    f0 = int(sys.argv[sys.argv.index("--f0") + 1]) if "--f0" in sys.argv else 0
    f1 = int(sys.argv[sys.argv.index("--f1") + 1]) if "--f1" in sys.argv else 0
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 0
    # camera calibration always read from dataset-root camera_parameters.h5
    calib = load_gopro_calib(os.path.join(os.path.dirname(seq_dir), "camera_parameters.h5"))
    # frame size comes from the calibration, never from a hardcoded constant
    W, H = int(calib["size"][0]), int(calib["size"][1])

    ch = build_pose_chain(seq_dir)          # same chain the adapter writes to rcs.csv
    time_offset, quat_at = ch["time_offset"], ch["quat_at"]

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
    opt_quats = None
    if opt_quat_path:
        # optimized poses are already w2c, the same convention as quat_at: no conjugation
        opt_quats = []
        with open(opt_quat_path) as fq:
            for row in fq:
                v = [float(x) for x in row.replace(',', ' ').split()]
                if len(v) >= 4:
                    opt_quats.append(np.array(v[:4]))

    cap = cv2.VideoCapture(f"{seq_dir}/video.mp4")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    limit = max_frames or n_frames
    first = f0
    # --max-frames counts from --f0 (it used to ignore f0 when both were given)
    last = f1 if f1 > f0 else (min(first + limit, n_frames) - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    t0 = time.time()
    short_opt = False
    for i in range(first, last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        t0i = i / fps + time_offset
        # One inverse-warp path for both modes; only the output view differs.
        if opt_quats is not None:
            # optimizer CSV can be one row shorter than the video (e.g. playground1); hold
            # the last pose instead of aborting the whole render
            if i >= len(opt_quats):
                if not short_opt:
                    print(f"  [WARN] opt-quat has {len(opt_quats)} rows < frame {i}; holding last pose")
                    short_opt = True
                q_out = opt_quats[-1]
            else:
                q_out = opt_quats[i]
        else:
            q_out = q_ref0
        if q_out is not None:
            mx, my = warp_pixels(calib, quat_at, t0i, q_out, W, H, workers=workers)
            # border fill in one pass (map is already float32)
            mx = np.nan_to_num(mx, nan=-1.0)
            my = np.nan_to_num(my, nan=-1.0)
            out8 = cv2.remap(frame, mx, my, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        # green center-crop rect (20% crop)
        x0, y0, x1, y1 = int(W * 0.1), int(H * 0.1), int(W * 0.9), int(H * 0.9)
        cv2.rectangle(out8, (x0, y0), (x1, y1), (0, 255, 0), 1)
        label = str(i)
        cv2.putText(out8, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out8, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(out8)
        if (i + 1) % 50 == 0 or i == last:
            done = i - first + 1
            el = time.time() - t0
            print(f"{done}/{last - first + 1} | {el:.0f}s | "
                  f"ETA {el / done * (last - i):.0f}s", flush=True)
    cap.release()
    writer.release()
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
