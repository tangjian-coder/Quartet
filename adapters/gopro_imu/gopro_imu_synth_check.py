"""GoPro-IMU synthetic correction check: ideal grid -> simulated distortion+RS -> recovered correction

Usage: python3 adapters/gopro_imu/gopro_imu_synth_check.py <seq_dir> <frame_n> <out_dir>
  seq_dir: sequence dir (parent must contain camera_parameters.h5)

Output:
- numeric: distortion/RS correction magnitudes (pixels), round-trip error (should be ~0)
- image: ideal grid / simulated input / recovered / error magnified
"""
import sys
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # same dir: gopro_common + gopro_imu_*; common.py is one level up

from gopro_common import quat_mul, quat_conj, quat_to_R, load_gopro_calib
from gopro_imu_to_input import build_pose_chain


def _size(calib):
    """Frame size from the calibration (single source; was a hardcoded 1920x1080)."""
    return int(calib["size"][0]), int(calib["size"][1])


def make_grid_image(W, H):
    """Ideal straight-line grid (center 80%, one line per 100px)"""
    img = np.full((H, W, 3), 255.0, np.float32)
    x0, x1 = int(W * 0.1), int(W * 0.9)
    y0, y1 = int(H * 0.1), int(H * 0.9)
    img[y0:y1, x0:x1] = 0.0
    for x in range(x0, x1 + 1, 100):
        img[y0:y1, x:x + 3] = 255.0
    for y in range(y0, y1 + 1, 100):
        img[y:y + 3, x0:x1] = 255.0
    return img


def sim_warp(src, calib, quat_at, t0, q_mid, use_rs=True, use_ldc=True, direction="fwd"):
    """Bidirectional warp for simulation only (local copy, do NOT confuse with
    common.warp_pixels / gopro_imu_validate.forward_warp): direction selects simulate vs recover

    Both directions use the real K on the ideal (corrected) frame, so the pair stays
    inverse to each other and matches the production chain's coordinate domain.

    direction="fwd": apply atan distortion before sampling (correction: ideal frame -> distorted input frame)
    direction="inv": apply tan undistortion (simulation: ideal frame -> input frame), inverse RS
    """
    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
    lgamma, wc = calib["lgamma"], np.asarray(calib["wc"])
    readout = calib["readout"]
    H, W = src.shape[:2]
    cx0, cy0 = (W - 1) / 2, (H - 1) / 2
    out = np.zeros((H, W, 3), np.float32)
    ny, nx = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    # Ideal frame rays come from the real K (centered principal point), like forward_warp
    rays = np.stack([(nx * cx0 + cx0 - cx) / fx, (ny * cy0 + cy0 - cy) / fy,
                     np.ones_like(nx)], axis=-1)
    for j in range(H):
        q_row = quat_at(t0 + readout * (j + 0.5) / H) if use_rs else q_mid
        # gather direction (ideal view -> raw row frame), the same matrix warp_pixels uses
        R = quat_to_R(quat_mul(q_row, quat_conj(q_mid)))
        if direction == "inv":
            # simulate input frame: ideal pixel -> real-K unproject -> undistort -> inverse RS -> ideal pixel
            px, py = nx[j] * cx0 + cx0, ny[j] * cy0 + cy0
            xu, yu = (px - cx) / fx, (py - cy) / fy
            if use_ldc:
                dx, dy = xu - wc[0], yu - wc[1]
                ru = np.hypot(dx, dy)
                rd = np.tan(ru * lgamma) / lgamma  # undistort (atan inverse)
                scale = np.where(ru > 1e-12, rd / np.where(ru > 1e-12, ru, 1.0), 1.0)
                xd, yd = wc[0] + dx * scale, wc[1] + dy * scale
            else:
                xd, yd = xu, yu
            v = (R.T @ np.stack([xd, yd, np.ones_like(xd)])).T
            sx, sy = fx * v[:, 0] / v[:, 2] + cx, fy * v[:, 1] / v[:, 2] + cy
        else:
            # correct: real-K ideal pixel -> row rotation -> distort -> real pixel
            v = rays[j] @ R.T
            xu, yu = v[:, 0] / v[:, 2], v[:, 1] / v[:, 2]
            if use_ldc:
                dx, dy = xu - wc[0], yu - wc[1]
                ru = np.hypot(dx, dy)
                rd = np.arctan(ru * lgamma) / lgamma  # distort: sample points move inward
                scale = np.where(ru > 1e-12, rd / np.where(ru > 1e-12, ru, 1.0), 1.0)
                xd, yd = wc[0] + dx * scale, wc[1] + dy * scale
            else:
                xd, yd = xu, yu
            sx, sy = fx * xd + cx, fy * yd + cy
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


def map_point(calib, q_row, q_mid, nx, ny, direction="fwd"):
    """Single-point coordinate mapping: ideal-frame <-> input-frame pixels (no resampling)

    fwd (correct): ideal pixel -> real-K ray -> row rotation -> atan distortion -> real pixel
    inv (simulate): real pixel -> real-K unproject -> undistort (tan inverse) -> inverse rotation
                    -> ideal pixel
    Strictly inverse: inv(fwd(p)) == p (rotation/distortion order symmetric)
    """
    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
    lgamma, wc = calib["lgamma"], np.asarray(calib["wc"])
    W, H = _size(calib)
    cx0, cy0 = (W - 1) / 2, (H - 1) / 2
    # gather: R maps an ideal-view point into the raw row frame (inverse of warp_grid)
    R = quat_to_R(quat_mul(q_row, quat_conj(q_mid)))
    if direction == "fwd":
        # ideal pixel -> ray via the real K (centered principal point)
        px, py = nx * cx0 + cx0, ny * cy0 + cy0
        ray = np.array([(px - cx) / fx, (py - cy) / fy, 1.0])
        v = R @ ray  # row rotation
        xu, yu = v[0] / v[2], v[1] / v[2]
        dx, dy = xu - wc[0], yu - wc[1]
        ru = np.hypot(dx, dy)
        if ru > 1e-12:
            rd = np.arctan(ru * lgamma) / lgamma  # distort: sample points move inward
            dx, dy = dx * rd / ru, dy * rd / ru
        # plane -> real pixel
        px, py = fx * (wc[0] + dx) + cx, fy * (wc[1] + dy) + cy
        return (px - cx0) / cx0, (py - cy0) / cy0
    # inv: real pixel -> real-K unproject
    px, py = nx * cx0 + cx0, ny * cy0 + cy0
    xu, yu = (px - cx) / fx, (py - cy) / fy
    dx, dy = xu - wc[0], yu - wc[1]
    ru = np.hypot(dx, dy)
    if ru > 1e-12:
        rd = np.tan(ru * lgamma) / lgamma  # undistort (atan inverse)
        dx, dy = dx * rd / ru, dy * rd / ru
    xd, yd = wc[0] + dx, wc[1] + dy
    v = R.T @ np.array([xd, yd, 1.0])  # inverse rotation
    x_v, y_v = v[0] / v[2], v[1] / v[2]
    # ideal plane -> ideal pixel (real K)
    return (fx * x_v + cx - cx0) / cx0, (fy * y_v + cy - cy0) / cy0


def geometric_roundtrip(calib, quat_at, t0, q_mid, use_rs=True, use_ldc=True):
    """Self-consistency: p -> fwd(p) -> inv(p1) should return p (pure coordinate mapping, no resampling)"""
    readout = calib["readout"]
    W, H = _size(calib)
    errs_fwd_inv, errs_inv_fwd = [], []
    for j in range(0, H, 54):  # 20 sampled rows
        q_row = quat_at(t0 + readout * (j + 0.5) / H) if use_rs else q_mid
        for i in range(0, W, 96):  # 20 sampled columns
            nx, ny = 2 * i / (W - 1) - 1, 2 * j / (H - 1) - 1
            # The round trip must also run with use_ldc=False (RSC-only check); gating it
            # on use_ldc used to record nothing and report a vacuous 0.0
            p1 = map_point(calib, q_row, q_mid, nx, ny, "fwd")
            p2 = map_point(calib, q_row, q_mid, p1[0], p1[1], "inv")
            errs_fwd_inv.append(np.hypot(p2[0] - nx, p2[1] - ny))
            p1 = map_point(calib, q_row, q_mid, nx, ny, "inv")
            p2 = map_point(calib, q_row, q_mid, p1[0], p1[1], "fwd")
            errs_inv_fwd.append(np.hypot(p2[0] - nx, p2[1] - ny))
    e1 = np.max(errs_fwd_inv) if errs_fwd_inv else 0.0
    e2 = np.max(errs_inv_fwd) if errs_inv_fwd else 0.0
    return e1, e2



def main():
    seq_dir, frame_n, out_dir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    # camera calibration always read from dataset-root camera_parameters.h5
    calib = load_gopro_calib(os.path.join(os.path.dirname(seq_dir), "camera_parameters.h5"))
    W, H = _size(calib)
    os.makedirs(out_dir, exist_ok=True)

    ch = build_pose_chain(seq_dir)          # same chain the adapter uses
    time_offset, quat_at = ch["time_offset"], ch["quat_at"]

    t0 = frame_n / 29.97 + time_offset
    q_mid = quat_at(t0 + calib["readout"] / 2)

    # geometric self-consistency (pure coordinate mapping, unaffected by resampling)
    e_fi, e_if = geometric_roundtrip(calib, quat_at, t0, q_mid, True, True)
    e_ldc, _ = geometric_roundtrip(calib, quat_at, t0, q_mid, False, True)
    e_rsc, _ = geometric_roundtrip(calib, quat_at, t0, q_mid, True, False)
    print(f"  geometric round-trip (normalized, ~0): LDC+RSC: fwd->inv={e_fi:.2e} inv->fwd={e_if:.2e}")
    print(f"  geometric round-trip: LDC only={e_ldc:.2e}  RSC only={e_rsc:.2e}")

    ideal = make_grid_image(W, H)

    # 1) simulate input frame: ideal image -> distortion+RS (undistort direction + inverse RS,
    #    row poses from this frame's real motion)
    sim = sim_warp(ideal, calib, quat_at, t0, q_mid, True, True, "inv")
    # 2) recover (distort direction + forward RS)
    rec = sim_warp(sim, calib, quat_at, t0, q_mid, True, True, "fwd")
    # 3) per-module self-consistency
    rec_ldc = sim_warp(sim_warp(ideal, calib, quat_at, t0, q_mid, False, True, "inv"),
                       calib, quat_at, t0, q_mid, False, True, "fwd")
    rec_rsc = sim_warp(sim_warp(ideal, calib, quat_at, t0, q_mid, True, False, "inv"),
                       calib, quat_at, t0, q_mid, True, False, "fwd")

    # numeric magnitudes
    fx, fy = calib["fx"], calib["fy"]
    # distortion magnitude: corner normalized point radial displacement under atan model (pixels)
    rd_corner = np.arctan(np.hypot(1, 1) * calib["lgamma"]) / calib["lgamma"] / np.hypot(1, 1)
    ldc_mag = (1 - rd_corner) * np.hypot((W - 1) / 2, (H - 1) / 2)
    # RS magnitude: first-vs-last row pose angle -> pixel displacement at focal length
    q_first, q_last = quat_at(t0 + calib["readout"] / (2 * H)), quat_at(t0 + calib["readout"] * (H - 0.5) / H)
    dq = quat_mul(q_first, quat_conj(q_last))
    rs_ang = 2 * np.arccos(np.clip(abs(dq[0]), 0, 1))
    rs_mag = np.tan(rs_ang) * fx
    # round-trip error (gray difference; grid lines give 255 when misaligned)
    err_full = np.abs(rec - ideal)
    err_ldc = np.abs(rec_ldc - ideal)
    err_rsc = np.abs(rec_rsc - ideal)
    print(f"== frame {frame_n} synthetic correction check ==")
    print(f"  distortion magnitude (corner radial): {ldc_mag:.1f} px   (LDC correction scale)")
    print(f"  RS magnitude (first-last row angle {np.degrees(rs_ang):.4f} deg): ~{rs_mag:.1f} px  (RSC correction scale)")
    for name, e in [("LDC+RSC", err_full), ("LDC only", err_ldc), ("RSC only", err_rsc)]:
        print(f"  image round-trip {name}: mean={e.mean():.3f} max={e.max():.1f} "
              f"(resampling-limited; judge by the geometric error above)")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for ax, img, ti in zip(axes.ravel(), [ideal, sim, rec, np.abs(rec - ideal) * 8],
                           ["ideal grid", "simulated input (distortion+RS)", "recovered", "error x8"]):
        ax.imshow(np.clip(img / 255.0, 0, 1), cmap="gray")
        ax.set_title(ti, fontsize=12)
        ax.axis("off")
    out_png = os.path.join(out_dir, f"synth_{frame_n}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=80)
    print(f"compare image: {out_png}")


if __name__ == "__main__":
    main()
