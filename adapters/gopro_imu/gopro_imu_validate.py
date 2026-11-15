"""GoPro-IMU image-level correction verification: LDC (undistortion) / RSC (de-rolling-shutter) / full

Usage: python3 adapters/gopro_imu/gopro_imu_validate.py <seq_dir> [frame_n] [out_dir]
  seq_dir: sequence dir (parent must contain camera_parameters.h5)
  frame_n: frame to extract (0-based)
  out_dir: output directory

Output: 2x2 compare image (original / LDC only / RSC only / LDC+RSC) -> <out_dir>/compare_<frame_n>.png
"""
import argparse
import os
import sys

import cv2
from scipy.interpolate import griddata

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # same dir: gopro_common + gopro_imu_*; common.py is one level up

import numpy as np

from gopro_common import load_gopro_calib, warp_grid
from gopro_imu_to_input import GRID_H, GRID_V, probe_video


def extract_frame(video_path, frame_n, out_png):
    """Extract frame n by index so the QA figure matches the rendered frame exactly.

    Index seek avoids the <1 frame position error of ffmpeg fast seek. cv2.imwrite
    expects BGR and encodes RGB itself, so pass the frame through unchanged.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame_n} from {video_path}")
    cv2.imwrite(out_png, bgr)


def load_frame(png_path):
    """PNG -> RGB float32 (0..255)"""
    import matplotlib.image as mpimg
    return (mpimg.imread(png_path)[:, :, :3] * 255.0).astype(np.float32)


def forward_warp(src, calib, quat_at, t0, q_mid, use_rs=True, ldc_type="fisheye", q_ref=None,
                gh=GRID_H, gv=GRID_V):
    """Forward correction mesh: source grid -> LDC (real-K undistort) -> de-RS to q_to frame.
    The warp field is inverse-interpolated (griddata) into a remap table, then resampled (no scatter holes).

    Real K throughout (fx/fy/cx/cy); corrected points may leave the frame. De-RS uses per-source-row
    poses (row time readout*j/(gv-1)) with H = K@R(q_to)@R(q_row)^T@K^-1 (w2c forward: the source
    point's position in the q_to view, the exact inverse of warp_pixels' gather).
    gh/gv: source-grid density, default = the optimizer grid (GRID_H/GRID_V). Measured against a
           241x241 reference remap (frame 560): 11x11 p99=11.8 px, 31x31 p99=1.15 px, 55x55 p99=0.36 px,
           so 31 is NOT converged; keep the adapter density.
    use_rs=False / ldc_type=None isolate LDC / RS; q_ref overrides the reference (default q_mid).
    Correction grid comes from the shared common.warp_grid (single implementation,
    same chain to_input uses for rcs.csv).
    Counterpart of common.warp_pixels, which applies the same correction inversely per output pixel.
    """
    q_to = q_ref if q_ref is not None else q_mid

    frame_h, frame_w = src.shape[:2]
    grid = warp_grid(calib, quat_at, t0, q_to, frame_w, frame_h, gh, gv, use_rs, ldc_type)
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
    ap = argparse.ArgumentParser(description="LDC / RSC / full correction comparison for one frame")
    ap.add_argument("seq_dir", help="sequence dir; the parent must hold camera_parameters.h5")
    ap.add_argument("frame_n", nargs="?", type=int, default=200)
    ap.add_argument("out_dir", nargs="?", default=None)
    a = ap.parse_args()
    seq_dir, frame_n = a.seq_dir, a.frame_n
    out_dir = a.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(seq_dir)), "output", os.path.basename(seq_dir))
    # camera calibration always read from dataset-root camera_parameters.h5
    calib_path = os.path.join(os.path.dirname(seq_dir), "camera_parameters.h5")
    calib = load_gopro_calib(calib_path)
    os.makedirs(out_dir, exist_ok=True)

    # same pose chain as the adapter (single definition)
    from gopro_imu_to_input import build_pose_chain
    ch = build_pose_chain(seq_dir)
    time_offset, quat_at = ch["time_offset"], ch["quat_at"]

    fps, _ = probe_video(f"{seq_dir}/video.mp4")
    t0 = frame_n / fps + time_offset
    q_mid = quat_at(t0 + calib["readout"] / 2)

    png = os.path.join(out_dir, f"frame_{frame_n}.png")
    extract_frame(f"{seq_dir}/video.mp4", frame_n, png)
    src = load_frame(png)
    print(f"frame {frame_n}: t0={t0:.3f}s (IMU time), extracted {png}")

    # all three use forward_warp (real-K window): ldc/rsc isolate LDC / RS by toggles,
    # so the quadrants share one geometry and are directly comparable
    full = forward_warp(src, calib, quat_at, t0, q_mid, True, calib["ldc_type"])
    ldc = forward_warp(src, calib, quat_at, t0, q_mid, False, calib["ldc_type"])
    rsc = forward_warp(src, calib, quat_at, t0, q_mid, True, None)

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
