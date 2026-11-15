"""GoPro-IMU (LiU) adapter: h5 calibration, IMU pose chain, timestamps.

The shared math lives in adapters/common.py and is re-exported here, so existing
`from gopro_common import ...` call sites keep working.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import (SLERP_LIN_D, MAP_CHUNK, THREAD_MIN_PIXELS,   # noqa: E402,F401
                    quat_mul, quat_conj, quat_to_R, quat_to_R_many, R_to_quat,
                    axis_angle_to_R, R_to_rotvec, rotvec_to_R, slerp,
                    integrate_gyro, row_time,
                    fisheye_distort, fisheye_undistort,
                    radtan_distort, radtan_undistort,
                    warp_grid, warp_pixels, write_conf)


def make_quat_at(qs, rate):
    """Return quat_at(t) sampling the pose chain by slerp; it also exposes .many(times).

    quat_at(t)          : scalar sample (cached-closure indexing, one value at a time)
    quat_at.many(times) : vectorised sample over an array of times; warp_pixels requires it,
                          otherwise the per-row pose table costs one Python call per row.
    """
    n = len(qs)

    def quat_at(t):
        kf = t * rate
        k0 = int(np.clip(np.floor(kf), 0, n - 2))
        frac = kf - k0
        a, b = qs[k0], qs[k0 + 1]
        d = np.dot(a, b)
        if d < 0:
            # q and -q are the same rotation; negate to interpolate the short arc
            b = -b
            d = -d
        if d > SLERP_LIN_D:
            r = a + frac * (b - a)
        else:
            th = np.arccos(np.clip(d, -1, 1))
            r = (np.sin((1 - frac) * th) * a + np.sin(frac * th) * b) / np.sin(th)
        return r / np.linalg.norm(r)

    def many(times):
        """Vectorised quat_at for all scan rows at once (the scalar path is n Python calls)."""
        kf = np.asarray(times, dtype=np.float64) * rate
        k0 = np.clip(np.floor(kf).astype(np.int64), 0, n - 2)
        frac = kf - k0
        a, b = qs[k0], qs[k0 + 1]
        d = np.sum(a * b, axis=1)
        flip = d < 0        # q and -q are the same rotation; negate to take the short arc
        b = np.where(flip[:, None], -b, b)
        d = np.abs(d)
        out = np.empty_like(a)
        lin = d > SLERP_LIN_D
        if lin.any():
            f = frac[lin][:, None]
            out[lin] = a[lin] + f * (b[lin] - a[lin])
        if (~lin).any():
            th = np.arccos(np.clip(d[~lin], -1, 1))[:, None]
            f = frac[~lin][:, None]
            out[~lin] = (np.sin((1 - f) * th) * a[~lin] + np.sin(f * th) * b[~lin]) / np.sin(th)
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    quat_at.many = many
    return quat_at


def fix_timestamps(t):
    """Rebuild timestamps as a uniform axis by global linear fit of index -> time.

    Per LiU GoPro-IMU README, timestamps are occasionally repeated or NaN while the
    samples themselves are valid; sampling rate is a constant 1000Hz. Keeping every
    sample (instead of dropping NaN rows) preserves all measurements.
    """
    good = np.isfinite(t)
    k, m = np.polyfit(np.arange(len(t))[good], t[good], 1)
    return np.arange(len(t)) * k + m


def load_gopro_calib(h5_path):
    """Read camera calibration from camera_parameters.h5 (LiU GoPro-IMU dataset root)

    h5 contents: K(3x3), lgamma, wc(2), readout, fps, size(2)
    Returns dict(fx,fy,cx,cy,lgamma,wc,readout,fps,size,ldc_type)
    """
    import h5py
    with h5py.File(h5_path, "r") as f:
        K = f["K"][()]
        return dict(
            fx=float(K[0, 0]), fy=float(K[1, 1]),
            cx=float(K[0, 2]), cy=float(K[1, 2]),
            lgamma=float(f["lgamma"][()]),
            wc=f["wc"][()].tolist(),
            readout=float(f["readout"][()]),
            fps=float(f["fps"][()]),
            size=f["size"][()].tolist(),
            ldc_type="fisheye")


