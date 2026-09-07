"""GoPro-IMU image-level correction verification: LDC (undistortion) / RSC (de-rolling-shutter) / full

Usage: python3 adapters/gopro_imu/gopro_imu_validate.py <seq_dir> [frame_n] [out_dir]
  seq_dir: sequence dir (parent must contain camera_parameters.h5)
  frame_n: frame to extract (0-based)
  out_dir: output directory

Output: 2x2 compare image (original / LDC only / RSC only / LDC+RSC) -> <out_dir>/compare_<frame_n>.png
"""
import sys
import os
import json
import subprocess
import cv2
from scipy.interpolate import griddata

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # same dir: gopro_common + gopro_imu_* modules

import numpy as np

from gopro_common import quat_mul, quat_conj, quat_to_R, load_gopro_calib, warp_grid

W, H = 1920, 1080


def extract_frame(video_path, frame_n, out_png):
    """Extract frame n via ffmpeg (-ss fast seek, position error <1 frame)"""
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-ss", f"{frame_n / 29.97:.3f}", "-i", video_path,
                    "-frames:v", "1", "-q:v", "2", out_png],
                   capture_output=True, timeout=60)


def load_frame(png_path):
    """PNG -> RGB float32 (0..255)"""
    import matplotlib.image as mpimg
    return (mpimg.imread(png_path)[:, :, :3] * 255.0).astype(np.float32)


def warp_virtual(src, calib, quat_at, t0, q_mid, use_rs=True, use_ldc=True, q_ref=None):
    """Inverse mapping: virtual output frame (center principal point, f=fx, undistorted) -> input frame pixels, bilinear

    Legacy renderer kept for lock-frame renders (align_check/render). Note: its virtual
    ray y-scale (f=fx) enters the 3D de-RS rotation -> up to ~8px edge error on dynamic
    frames; forward_warp (real-K chain) is the physically exact counterpart.

    use_rs=False: row pose = frame mid pose (distortion only, verify LDC)
    use_ldc=False: skip atan distortion (RS only, verify RSC)
    q_ref: reference pose (default frame mid q_mid = lock current frame; frame-0 mid = lock frame 0)
    """
    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
    lgamma, wc = calib["lgamma"], np.asarray(calib["wc"])
    readout = calib["readout"]
    q_to = q_ref if q_ref is not None else q_mid
    out = np.zeros((H, W, 3), np.float32)
    # output pixel grid (normalized, center principal point, isotropic f=fx, y uses (H-1)/2)
    ny, nx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    rays = np.stack([nx * (W - 1) / 2 / fx, ny * (H - 1) / 2 / fx, np.ones_like(nx)], axis=-1)
    for j in range(H):
        q_row = quat_at(t0 + readout * (j + 0.5) / H) if use_rs else q_mid
        # rotation direction: R(q_to x conj(q_row)) - reference-frame ray -> row-frame ray
        # (reversed direction drifted the lock-frame-0 render by 15px; this one by 0.9px)
        R = quat_to_R(quat_mul(q_to, quat_conj(q_row)))
        v = rays[j] @ R.T
        xu, yu = v[:, 0] / v[:, 2], v[:, 1] / v[:, 2]
        if use_ldc:
            dx, dy = xu - wc[0], yu - wc[1]
            ru = np.hypot(dx, dy)
            rd = np.arctan(ru * lgamma) / lgamma
            scale = np.where(ru > 1e-12, rd / np.where(ru > 1e-12, ru, 1.0), 1.0)
            xd, yd = wc[0] + dx * scale, wc[1] + dy * scale
        else:
            xd, yd = xu, yu
        sx, sy = fx * xd + cx, fy * yd + cy
        # bilinear sample
        x0 = np.floor(sx).astype(int)
        y0 = np.floor(sy).astype(int)
        ax, ay = sx - x0, sy - y0
        ok = (x0 >= 0) & (x0 < W - 1) & (y0 >= 0) & (y0 < H - 1)
        x1, y1 = np.clip(x0 + 1, 0, W - 1), np.clip(y0 + 1, 0, H - 1)
        x0c, y0c = np.clip(x0, 0, W - 1), np.clip(y0, 0, H - 1)
        row = (src[y0c, x0c] * (1 - ax)[:, None] + src[y0c, x1] * ax[:, None]) * (1 - ay)[:, None] \
              + (src[y1, x0c] * (1 - ax)[:, None] + src[y1, x1] * ax[:, None]) * ay[:, None]
        row[~ok] = 0
        out[j] = row
    return out


def forward_warp(src, calib, quat_at, t0, q_mid, use_rs=True, use_ldc=True, q_ref=None,
                gh=31, gv=31):
    """Forward correction mesh: source grid -> LDC (real-K undistort) -> de-RS to q_to frame.
    The warp field is inverse-interpolated (griddata) into a remap table, then resampled (no scatter holes).

    Real K throughout (fx/fy/cx/cy); corrected points may leave the frame. De-RS uses per-source-row
    poses (row time readout*j/(gv-1)) with H = K@R(q_to)@R(q_row)^T@K^-1 (product paradigm).
    gh/gv: source-grid density (default 31; 31 is already converged for rendering, denser adds nothing).
    use_rs=False / use_ldc=False isolate LDC / RS; q_ref overrides the reference (default q_mid).
    Correction grid comes from the shared gopro_common.warp_grid (single implementation,
    same chain to_input uses for rcs.csv).
    Counterpart of warp_virtual, which samples the same correction inversely on the virtual frame.
    """
    q_to = q_ref if q_ref is not None else q_mid

    frame_h, frame_w = src.shape[:2]
    grid = warp_grid(calib, quat_at, t0, q_to, frame_w, frame_h, gh, gv, use_rs, use_ldc)
    grid_uniform = np.stack(np.meshgrid(np.linspace(0, frame_w - 1, gh, dtype=np.float32),
                                        np.linspace(0, frame_h - 1, gv, dtype=np.float32)),
                            axis=-1)

    # Warp field: each corrected grid point (dst) carries its source pixel (src)
    pts_dst = grid.reshape(-1, 2)
    pts_src = grid_uniform.reshape(-1, 2)
    # Inverse-interpolate: per-output-pixel source location over the canvas (NaN outside hull)
    canvas_x, canvas_y = np.meshgrid(np.linspace(0, frame_w - 1, frame_w),
                                     np.linspace(0, frame_h - 1, frame_h))
    grid_map = griddata(pts_dst, pts_src, (canvas_x, canvas_y))

    out = cv2.remap(src,
                    grid_map[..., 0].astype(np.float32),
                    grid_map[..., 1].astype(np.float32),
                    cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0))

    return out


def main():
    # CLI: <seq_dir> [frame_n] [out_dir]; defaults target apple3 for debugging
    seq_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/tangjian5/data/gopro_imu/gopro-imu-dataset/apple3"
    frame_n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "/home/tangjian5/data/gopro_imu/output/apple3"
    # camera calibration always read from dataset-root camera_parameters.h5
    calib_path = os.path.join(os.path.dirname(seq_dir), "camera_parameters.h5")
    calib = load_gopro_calib(calib_path)
    os.makedirs(out_dir, exist_ok=True)

    # rebuild the pose chain exactly like the adapter (rcs.csv grid is too coarse to reuse)
    from gopro_imu_to_input import load_imu, load_calib
    from gopro_common import axis_angle_to_R, integrate_gyro, make_quat_at, fix_timestamps
    import h5py

    t, acc, gyro = load_imu(seq_dir)
    gyro_rate_nom, time_offset, rot, gbias, abias = load_calib(seq_dir)
    R_gc = axis_angle_to_R(rot)
    # official timestamp fix: keep all samples, linear-fit uniform axis (only timestamps broken)
    t_fix = fix_timestamps(t)
    dt = t_fix[1] - t_fix[0]
    du = np.vstack([gyro, acc]).T
    qs = integrate_gyro((du[:, :3] + gbias) @ R_gc.T, dt)
    quat_at = make_quat_at(qs, 1.0 / dt)

    fps = 29.97
    t0 = frame_n / fps + time_offset
    q_mid = quat_at(t0 + calib["readout"] / 2)

    png = os.path.join(out_dir, f"frame_{frame_n}.png")
    extract_frame(f"{seq_dir}/video.mp4", frame_n, png)
    src = load_frame(png)
    print(f"frame {frame_n}: t0={t0:.3f}s (IMU time), extracted {png}")

    # all three use forward_warp (real-K window): ldc/rsc isolate LDC / RS by toggles,
    # so the quadrants share one geometry and are directly comparable
    full = forward_warp(src, calib, quat_at, t0, q_mid, True, True)
    ldc = forward_warp(src, calib, quat_at, t0, q_mid, False, True)
    rsc = forward_warp(src, calib, quat_at, t0, q_mid, True, False)

    import matplotlib.pyplot as plt
    titles = ["Original (distorted+RS)", "LDC only (undistorted)", "RSC only (de-RS)", "LDC+RSC (full)"]
    imgs = [src, ldc, rsc, full]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for ax, img, ti in zip(axes.ravel(), imgs, titles):
        ax.imshow(np.clip(img / 255.0, 0, 1))
        ax.set_title(ti, fontsize=12)
        ax.axis("off")
    out_png = os.path.join(out_dir, f"compare_{frame_n}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=80)
    print(f"saved compare image: {out_png}")


if __name__ == "__main__":
    main()
