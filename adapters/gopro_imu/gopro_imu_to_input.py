"""GoPro-IMU dataset (LiU CVPR'18) -> quaternion_optimizer input adapter

Pipeline: imu.h5 --fix_timestamps(keep all samples)--> gyro --minus gbias-->
          camera-frame angular velocity (rot extrinsics) --left-multiply integration-->
          pose chain (c2w); per-frame output is w2c
          Row time t = n/fps + time_offset + readout*j/NROW -> slerp -> per-row pose
          rcs.csv = real-K corrected grid points (LDC + de-RS, row pose vs reference pose),
                    normalized as (p - (W/H-1)/2)/((W/H-1)/2) — center-origin [-1,1]
          Frame pose = w2c conjugate of q_mid

Usage: python3 adapters/gopro_imu/gopro_imu_to_input.py <data_dir> <seq> <out_dir>
       [--readout ms] [--calib-h5 path] [--verify-only] [--lock-ref]

<data_dir> is the gopro-imu-dataset root, containing <seq>/{imu.h5,calibration.yml,video.mp4}
"""
import sys
import os
import json
import subprocess

import numpy as np
import h5py

# gopro_common.py lives in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gopro_common import (axis_angle_to_R, integrate_gyro, make_quat_at,
                          fix_timestamps, warp_grid, write_conf)

# LiU GoPro-IMU camera calibration (gopro-hero3.h5 = barn3 sequence; all sequences share it)
CALIB = dict(fx=868.10363082, fy=881.57627866, cx=987.87297848, cy=529.7808111,
             lgamma=0.8923827203724195, wc=(0.00038555, -0.00340777),
             readout=0.0316698, fps=29.97)
W, H = 1920, 1080
# Grid density: 11x11 control points. The C++ optimizer hardcodes kGridSide=11
# (boundary indices assume an 11x11 grid); 65x65 was both slow and boundary-index-mismatched
GRID_H, GRID_V = 11, 11


def probe_video(video_path):
    """Probe video fps/duration via ffmpeg; fall back to (29.97, 0) on failure"""
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        r = subprocess.run([ff, "-i", video_path], capture_output=True, text=True, timeout=30)
        fps, dur = 29.97, 0.0
        for line in r.stderr.splitlines():
            if "Video:" in line and "fps" in line:
                fps = float(line.split("fps")[0].split(",")[-1].strip())
            if "Duration:" in line:
                hms = line.split("Duration:")[1].split(",")[0].strip()
                h, m, s = [float(x) for x in hms.split(":")]
                dur = h * 3600 + m * 60 + s
        return fps, dur
    except Exception:
        return 29.97, 0.0


def load_imu(seq_dir):
    """Read imu.h5; returns raw timestamps and acc/gyro measurements (NaN included)"""
    with h5py.File(f"{seq_dir}/imu.h5", "r") as f:
        t = f["time"][:]
        acc = f["acc"][:]
        gyro = f["gyro"][:]
    return t, acc, gyro


def load_calib(seq_dir):
    """Read calibration.yml (bias/rate/time offset/extrinsics)"""
    import yaml
    with open(f"{seq_dir}/calibration.yml") as f:
        c = yaml.safe_load(f)
    rot = np.array([c["rot_x"], c["rot_y"], c["rot_z"]])
    gbias = np.array([c["gbias_x"], c["gbias_y"], c["gbias_z"]])
    abias = np.array([c["abias_x"], c["abias_y"], c["abias_z"]])
    return c["gyro_rate"], c["time_offset"], rot, gbias, abias


def static_verify(t, acc, gyro, gbias, abias):
    """Static-segment sanity: gyro+bias residual ~0, |acc| ~9.81; checks bias sign convention"""
    # take the 1s window with lowest gyro energy as the static segment
    omega_norm2 = np.sum(gyro ** 2, axis=0)
    win = int(1.0 * 1000)  # ~1s @1kHz
    if len(omega_norm2) > win:
        i0 = np.argmin(np.convolve(omega_norm2, np.ones(win) / win, mode="valid"))
        sl = slice(i0, i0 + win)
    else:
        sl = slice(0, len(omega_norm2))

    g_plus = np.mean(np.abs(gyro[:, sl] + gbias[:, None]), axis=1)
    g_minus = np.mean(np.abs(gyro[:, sl] - gbias[:, None]), axis=1)
    a_mean = np.mean(acc[:, sl] + abias[:, None], axis=1)
    a_norm = np.linalg.norm(a_mean)

    print("== Static-segment check (1s low-energy window) ==")
    print(f"  gyro raw mean: {np.mean(gyro[:, sl], axis=1)}")
    print(f"  |gyro+gbias| mean: {g_plus}   (should be ~0)")
    print(f"  |gyro-gbias| mean: {g_minus}")
    print(f"  acc+abias mean: {a_mean}  norm={a_norm:.3f} (should be ~9.81)")
    if np.mean(g_minus) < 0.5 * np.mean(g_plus):
        print("  [WARN] bias sign may be flipped: gyro-gbias residual is smaller")
    if abs(a_norm - 9.81) > 0.3:
        print("  [WARN] acc norm deviates from gravity; check abias/units")
    return g_plus, g_minus, a_norm


def main():
    data_dir, seq, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    readout = CALIB["readout"]
    if "--readout" in sys.argv:
        readout = float(sys.argv[sys.argv.index("--readout") + 1]) / 1000.0
    verify_only = "--verify-only" in sys.argv
    # Camera calibration: default reads data_dir/camera_parameters.h5; --calib-h5 overrides;
    # CALIB constant is only a fallback if the file is missing
    calib = dict(CALIB)
    calib_h5 = None
    if "--calib-h5" in sys.argv:
        calib_h5 = sys.argv[sys.argv.index("--calib-h5") + 1]
    if not calib_h5:
        default_h5 = f"{data_dir}/camera_parameters.h5"
        if os.path.exists(default_h5):
            calib_h5 = default_h5
    if calib_h5:
        with h5py.File(calib_h5, "r") as f:
            K = f["K"][:]
            calib.update(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                         lgamma=float(f["lgamma"][()]), wc=tuple(f["wc"][:]),
                         readout=float(f["readout"][()]), fps=float(f["fps"][()]))
    else:
        print("  [WARN] camera_parameters.h5 not found; using embedded CALIB")

    seq_dir = f"{data_dir}/{seq}"
    t, acc, gyro = load_imu(seq_dir)
    gyro_rate_nom, time_offset, rot, gbias, abias = load_calib(seq_dir)
    R_gc = axis_angle_to_R(rot)

    # effective sample rate from median dt (nominal gyro_rate may be unreliable)
    valid = np.isfinite(t)
    dt_med = np.median(np.diff(t[valid]))
    gyro_rate_eff = 1.0 / dt_med
    print(f"== {seq} basics ==")
    print(f"  IMU samples: {len(t)} (valid {valid.sum()}, NaN {len(t)-valid.sum()})")
    print(f"  duration: {t[valid][-1]-t[valid][0]:.2f}s  median dt={dt_med*1000:.4f}ms")
    print(f"  effective rate (median dt): {gyro_rate_eff:.1f}Hz  nominal gyro_rate: {gyro_rate_nom}Hz")
    print(f"  time_offset={time_offset:.4f}s  rot(axis-angle)={rot}")
    print(f"  camera: fx={calib['fx']:.3f} fy={calib['fy']:.3f} cx={calib['cx']:.3f} cy={calib['cy']:.3f}"
          f" lgamma={calib['lgamma']:.5f} readout={calib['readout']*1000:.4f}ms")

    static_verify(t, acc, gyro, gbias, abias)

    # Official timestamp fix: keep every sample (NaN rows hold valid measurements);
    # index->time global linear fit gives a uniform 1000Hz axis (README Timestamp correction)
    t_uni = fix_timestamps(t)
    dt = t_uni[1] - t_uni[0]
    rate = 1.0 / dt  # only dt/rate used afterwards
    gyro_uni, _ = gyro.T, acc.T  # all samples kept (acc only used for checks)

    # subtract bias + IMU frame -> camera frame; crisp convention: corrected = raw + bias
    omega = (gyro_uni + gbias) @ R_gc.T

    qs = integrate_gyro(omega, dt)
    quat_at = make_quat_at(qs, rate)

    fps, vid_dur = probe_video(f"{seq_dir}/video.mp4")
    print(f"  video fps (ffmpeg): {fps}  duration: {vid_dur:.2f}s")

    if verify_only:
        print(f"[verify-only] alignment/integration done, no input files written: {seq}")
        return

    os.makedirs(out_dir, exist_ok=True)

    # frame count from actual video duration (gyro stream covers video via time_offset)
    t_end = (len(qs) - 1) / rate
    n_frames = int(vid_dur * fps) if vid_dur > 0 else int((t_end - time_offset - readout) * fps)
    # lock-frame-0 grid: q_to = frame-0 mid pose (pose lock baked into the grid, render with Ht=I).
    # Sample coordinates live in the original frame (row) frame; only output rays may be
    # rotated, never the sample positions
    lock_ref = quat_at(time_offset + readout / 2) if "--lock-ref" in sys.argv else None

    # RCS grid: GRID_H x GRID_V regular source grid corrected by the shared warp_grid chain
    # (real-K LDC + de-RS, per-row poses), then normalized about the frame center
    # (p - (W/H-1)/2) / ((W/H-1)/2) — center-origin [-1,1] grid definition
    cx0, cy0 = (W - 1) / 2, (H - 1) / 2
    grid_path = f"{out_dir}/rcs.csv" if lock_ref is None else f"{out_dir}/rcs_lock0.csv"
    with open(f"{out_dir}/quat.csv", "w") as fq, open(grid_path, "w") as fr:
        fq.write("time,w,x,y,z\n")
        for n in range(n_frames):
            t0 = n / fps + time_offset
            q_mid = quat_at(t0 + readout / 2)
            q_to = lock_ref if lock_ref is not None else q_mid
            # frame pose: w2c = conjugate of c2w
            fq.write(f"{int(n / fps * 1e9)},{q_mid[0]:.9f},{-q_mid[1]:.9f},{-q_mid[2]:.9f},{-q_mid[3]:.9f}\n")
            # row pose per content row handled inside warp_grid; normalize to center origin
            grid = warp_grid(calib, quat_at, t0, q_to, W, H, GRID_H, GRID_V)
            vals = []
            for j in range(GRID_V):
                for i in range(GRID_H):
                    vals.append(f"{(grid[j, i, 0] - cx0) / cx0:.6f},{(grid[j, i, 1] - cy0) / cy0:.6f}")
            fr.write(",".join(vals) + "\n")

    # conf.yaml: real-K intrinsics; the RCS grid is center-origin normalized, so the
    # optimizer converts the principal point back to the frame center when loading
    write_conf(f"{out_dir}/conf.yaml",
               fx=calib["fx"], fy=calib["fy"], cx=calib["cx"], cy=calib["cy"],
               width=W, height=H, crop_h=0.2, crop_v=0.2, grid_h=GRID_H, grid_v=GRID_V)
    print(f"  wrote: {out_dir}/quat.csv ({n_frames} frames) {grid_path} conf.yaml")
    print(f"  calibration table printed above; cross-check time_offset/bias with crisp if in doubt")


if __name__ == "__main__":
    main()
