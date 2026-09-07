"""Shared math for GoPro-IMU adapters -> quaternion_optimizer input.

Conventions (matching crisp/Kontiki):
- gyro raw measurement: body-frame angular velocity (rad/s)
- bias is a correction: omega_corrected = raw + bias
- rot axis-angle -> R_gc: camera-frame angular velocity = R_gc.T @ gyro-frame
- left-multiply body-frame integration -> pose chain (c2w); frame output uses w2c (conjugate)
- warp_grid: the single LDC + de-RS grid chain (real-K), shared by validate.forward_warp
  and to_input rcs.csv
"""
import numpy as np


def quat_mul(a, b):
    """Hamilton product, (w,x,y,z)"""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])


def quat_conj(a):
    """Quaternion conjugate (w,x,y,z)"""
    return np.array([a[0], -a[1], -a[2], -a[3]])


def axis_angle_to_R(r):
    """Axis-angle -> rotation matrix (Rodrigues)"""
    ang = np.linalg.norm(r)
    n = r / ang
    x, y, z = n
    c, s, C = np.cos(ang), np.sin(ang), 1 - np.cos(ang)
    return np.array([[c + x*x*C, x*y*C - z*s, x*z*C + y*s],
                     [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
                     [z*x*C - y*s, z*y*C + x*s, c + z*z*C]])


def quat_to_R(q):
    """Quaternion (w,x,y,z) -> rotation matrix"""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])


def integrate_gyro(omega, dt):
    """Left-multiply integration, gyro -> pose chain (equivalent to right-multiply).

    Left ODE dq/dt = [0, w_w/2] x q equals the physical right ODE q x [0, w_b/2]
    only when w_w = R(q)·w_b, so body-frame omega is rotated to world frame each step.
    (Directly left-multiplying body-frame omega is an approximation: it drifted the
    lock-frame-0 render from 1.7px to 3.4px in our tests.)
    """
    n = len(omega)
    qs = np.empty((n, 4))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    half_dt = 0.5 * dt
    for k in range(n):
        # body-frame angular velocity -> world frame, then left-multiply
        w_w = quat_to_R(q) @ omega[k]
        dq = np.array([1.0, w_w[0] * half_dt, w_w[1] * half_dt, w_w[2] * half_dt])
        dq /= np.linalg.norm(dq)
        q = quat_mul(dq, q)
        q /= np.linalg.norm(q)
        qs[k] = q
    return qs


def make_quat_at(qs, rate):
    """Return quat_at(t): c2w pose at time t (seconds), slerp between samples"""
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
        if d > 0.9995:
            r = a + frac * (b - a)
        else:
            th = np.arccos(np.clip(d, -1, 1))
            r = (np.sin((1 - frac) * th) * a + np.sin(frac * th) * b) / np.sin(th)
        return r / np.linalg.norm(r)

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


def warp_grid(calib, quat_at, t0, q_to, frame_w, frame_h, gh=31, gv=31,
              use_rs=True, use_ldc=True):
    """LDC + de-RS correction of a regular source-frame grid, real K throughout.

    Grid points are source-frame pixels (top-left origin, linspace 0..W-1 / H-1); each
    content row j uses its own pose at readout*j/(gv-1). De-RS homography
    H = K@R(q_to)@R(q_row)^T@K^-1 (product paradigm; R maps world -> current frame).
    Returns corrected grid (gv, gh, 2) in real-K pixels; points may leave the frame.
    This is the optimizer-input RCS chain (to_input) and the correction core of
    validate.forward_warp; single implementation shared by both.
    """
    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
    lgamma, wc = calib["lgamma"], np.asarray(calib["wc"])
    readout = calib["readout"]

    x = np.linspace(0, frame_w - 1, gh, dtype=np.float32)
    y = np.linspace(0, frame_h - 1, gv, dtype=np.float32)
    grid = np.stack(np.meshgrid(x, y), axis=-1)

    if use_ldc:
        # Unproject with real K (no virtual-camera assumption), then undistort about wc
        xn = (grid[..., 0] - cx) / fx
        yn = (grid[..., 1] - cy) / fy
        dx, dy = xn - wc[0], yn - wc[1]
        rd = np.hypot(dx, dy)
        # tan is the inverse of the forward atan model (needs rd*lgamma < pi/2)
        ru = np.tan(rd * lgamma) / lgamma
        scale = np.where(rd > 1e-12, ru / np.where(rd > 1e-12, rd, 1.0), 1.0)
        # Undistorted direction is relative to wc; add back, then back to real-K pixels
        xu, yu = wc[0] + dx * scale, wc[1] + dy * scale
        grid[..., 0], grid[..., 1] = fx * xu + cx, fy * yu + cy

    if use_rs:
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float32)
        K_inv = np.linalg.inv(K)
        for j in range(gv):
            q_row = quat_at(t0 + readout * j / (gv - 1))
            # Row content is captured at its own row time; rotate it into the q_to frame
            R = quat_to_R(quat_mul(q_to, quat_conj(q_row)))
            H = K @ R @ K_inv
            xy1 = np.concatenate([grid[j], np.ones((gh, 1), dtype=np.float32)], axis=-1)
            warped = (H @ xy1.T).T
            grid[j] = warped[..., :2] / warped[..., 2:3]
    return grid


def write_conf(out_path, focal, width, height, crop_h, crop_v, grid_h, grid_v,
               fy=None, cx=None, cy=None):
    """Write conf.yaml (field names must match quaternion_optimizer DataLoader)

    calFocalLength: virtual output camera f=fx (center principal point, isotropic),
    the only intrinsic the optimizer needs (RCS grid is output-frame coordinates).
    calFocalY/calPrincipalX/Y: real sensor intrinsics, optional (Python side only).
    """
    with open(out_path, "w") as f:
        f.write(f"calFocalLength: {focal:.6f}\nfrmWidth: {width}\nfrmHeight: {height}\n"
                f"gridFrmCropFctrH: {crop_h}\ngridFrmCropFctrV: {crop_v}\n"
                f"gridCntH: {grid_h}\ngridCntV: {grid_v}\n")
        if fy is not None:
            f.write(f"calFocalY: {fy:.6f}\n")
        if cx is not None and cy is not None:
            f.write(f"calPrincipalX: {cx:.6f}\ncalPrincipalY: {cy:.6f}\n")


def load_gopro_calib(h5_path):
    """Read camera calibration from camera_parameters.h5 (LiU GoPro-IMU dataset root)

    h5 contents: K(3x3), lgamma, wc(2), readout, fps, size(2)
    Returns dict(fx,fy,cx,cy,lgamma,wc,readout,fps,size)
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
            size=f["size"][()].tolist())
