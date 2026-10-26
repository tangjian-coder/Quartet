"""Shared math for GoPro-IMU adapters -> quaternion_optimizer input.

Conventions (matching crisp/Kontiki, single convention end to end):
- gyro raw measurement: body-frame angular velocity (rad/s)
- bias is a correction: omega_corrected = raw + bias
- rot axis-angle -> R_gc: camera-frame angular velocity = R_gc.T @ gyro-frame
- ALL poses satisfy pc = R(q) * pw (world -> camera, w2c). Hence
  forward (source -> view) = R(view) * R(row)^T and gather = R(row) * R(view)^T.
  quat.csv / opt.csv hold these same w2c poses (no conjugation anywhere).
- warp_grid: forward LDC + de-RS map of the source grid (real-K), shared by
  validate.forward_warp, to_input rcs.csv and the C++ boundary polygon
"""
import os
import atexit
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Slerp switches to a normalised lerp when the samples are this close (cos of ~1.8 deg).
SLERP_LIN_D = 0.9995
# Render block size: measured fastest at 32; larger blocks lose cache locality.
MAP_CHUNK = 32
# Below this pixel count the thread-pool overhead outweighs the win.
THREAD_MIN_PIXELS = 200000


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
    if ang < 1e-12:
        return np.eye(3)   # no extrinsic rotation; n = r/ang would turn the whole chain into NaN
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


def quat_to_R_many(q):
    """Batch quat_to_R for (n,4) quaternions: same formula, one vectorised pass."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), dtype=q.dtype)
    R[:, 0, 0] = 1 - 2*(y*y + z*z)
    R[:, 0, 1] = 2*(x*y - w*z)
    R[:, 0, 2] = 2*(x*z + w*y)
    R[:, 1, 0] = 2*(x*y + w*z)
    R[:, 1, 1] = 1 - 2*(x*x + z*z)
    R[:, 1, 2] = 2*(y*z - w*x)
    R[:, 2, 0] = 2*(x*z - w*y)
    R[:, 2, 1] = 2*(y*z + w*x)
    R[:, 2, 2] = 1 - 2*(x*x + y*y)
    return R


def integrate_gyro(omega, dt):
    """Body-rate integration of a world->camera pose chain: pc = R(q) * pw.

    Left ODE dq/dt = -[0, w_b/2] x q  (i.e. R_new = exp(-[w_b] dt) R) integrates to
    R(t) = exp(-[w_b] t) for a constant rate, which is exactly the w2c attitude: a fixed
    world point appears to counter-rotate in the camera frame. The sign is what makes
    warp_grid's forward map R(view)*R(row)^T and warp_pixels' gather its true inverse.
    """
    n = len(omega)
    qs = np.empty((n, 4))
    q = np.array([1.0, 0.0, 0.0, 0.0])
    half_dt = 0.5 * dt
    for k in range(n):
        # Negated body rate, left-multiplied: world appears to rotate the other way
        dq = np.array([1.0, -omega[k][0] * half_dt, -omega[k][1] * half_dt,
                       -omega[k][2] * half_dt])
        dq /= np.linalg.norm(dq)
        q = quat_mul(dq, q)
        q /= np.linalg.norm(q)
        qs[k] = q
    return qs


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


def ldc_undistort(xn, yn, wc, lgamma):
    """Distorted normalized coords -> undistorted, about the distortion center wc."""
    dx, dy = xn - wc[0], yn - wc[1]
    rd = np.hypot(dx, dy)
    # tan inverts the forward atan model; it diverges as rd*lgamma -> pi/2, so reject
    # configurations whose corner would blow up instead of producing inf maps
    rc = float(np.max(rd)) * lgamma
    if rc >= 0.98 * np.pi / 2:
        raise ValueError(f"LDC model out of range: max(rd*lgamma)={rc:.4f} >= pi/2; "
                         f"check lgamma={lgamma} or the crop window")
    ru = np.tan(rd * lgamma) / lgamma
    scale = np.where(rd > 1e-12, ru / np.where(rd > 1e-12, rd, 1.0), 1.0)
    return wc[0] + dx * scale, wc[1] + dy * scale


def ldc_distort(xu, yu, wc, lgamma):
    """Undistorted normalized coords -> distorted (forward atan model), about wc."""
    dx, dy = xu - wc[0], yu - wc[1]
    # sqrt/maximum instead of hypot/nested where: this runs per pixel per iteration
    ru = np.sqrt(dx * dx + dy * dy)
    rd = np.arctan(ru * lgamma) / lgamma
    scale = np.where(ru > 1e-12, rd / np.maximum(ru, 1e-12), 1.0)
    return wc[0] + dx * scale, wc[1] + dy * scale


def row_time(t0, readout, s, n, ref="pixel_row"):
    """Exposure time of row s of n rows.

    ref="pixel_row": s is a real sensor row, exposed over its own slot, so the
    lookup point is the slot centre. ref="grid_node": s is a sparse grid node;
    nodes span the readout end to end, so the lookup points are the slot edges.
    """
    if ref == "grid_node":
        return t0 + readout * s / (n - 1)
    return t0 + readout * (s + 0.5) / n


def warp_grid(calib, quat_at, t0, q_to, frame_w, frame_h, gh=31, gv=31,
              use_rs=True, use_ldc=True):
    """LDC + de-RS correction of a regular source-frame grid, real K throughout.

    Grid points are source-frame pixels (top-left origin, linspace 0..W-1 / H-1); each
    content row j uses its own pose at readout*j/(gv-1). De-RS homography
    H = K@R(q_to)@R(q_row)^T@K^-1 (w2c convention: R maps world -> current frame, so this
    is the forward map of a source point into the q_to view).
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
        xu, yu = ldc_undistort(xn, yn, wc, lgamma)
        grid[..., 0], grid[..., 1] = fx * xu + cx, fy * yu + cy

    if use_rs:
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float32)
        K_inv = np.linalg.inv(K)
        for j in range(gv):
            q_row = quat_at(row_time(t0, readout, j, gv, ref="grid_node"))
            # Row content is captured at its own row time; maps row frame -> q_to frame
            R = quat_to_R(quat_mul(q_to, quat_conj(q_row)))
            H = K @ R @ K_inv
            xy1 = np.concatenate([grid[j], np.ones((gh, 1), dtype=np.float32)], axis=-1)
            warped = (H @ xy1.T).T
            grid[j] = warped[..., :2] / warped[..., 2:3]
    return grid


_POOLS = {}
atexit.register(lambda: [p.shutdown(wait=False) for p in list(_POOLS.values())])


def _dispatch(blocks, fn, workers):
    """Run per-block work serially or on a shared pool (each block owns its map slice).

    The pool is cached per worker count: building one per frame cost 1-3 ms of the render.
    """
    if workers > 1 and len(blocks) > 1:
        pool = _POOLS.get(workers)
        if pool is None:
            pool = _POOLS[workers] = ThreadPoolExecutor(workers)
        list(pool.map(fn, blocks))
    else:
        for b in blocks:
            fn(b)


def warp_pixels(calib, quat_at, t0, q_out, frame_w, frame_h, use_rs=True, use_ldc=True,
                row_ref="pixel_row", max_iter=3, tol=0.01, chunk=MAP_CHUNK, workers=0,
                undistort=None, scan_axis="row"):
    """Inverse per-pixel mapping for rendering: output pixels -> source pixels (gather).

    Real K throughout. De-RS applies the INVERSE of the forward correction: the ray of an
    output pixel is rotated by R(q_row)*R(q_out)^T (w2c; the exact inverse of warp_grid).
    De-RS needs the exposure of the source row that is actually sampled, so sy is solved by
    fixed-point iteration with row_ref="pixel_row"; row_ref="output_row" keeps the legacy
    approximation that reuses the output row (A/B only). Row poses come from one batched
    slerp over all source rows (quat_at.many), so the pixel loop stays vectorised.
    Returns float32 (map_x, map_y) in source pixels, NaN where the source pixel falls
    outside the frame (caller maps NaN to a border value).

    quat_at   : callable returning a pose for a time; MUST expose .many(times) -> (n,4)
                (both make_quat_at implementations do).
    undistort : optional callable (xu, yu) -> (xd, yd) overriding the fisheye model, for
                cameras whose distortion is not the GoPro LDC (e.g. ADVIO radtan).
    scan_axis : "row" for landscape capture (default, unchanged) or "col" for portrait
                capture, where the sensor reads columns and de-RS iterates the source column.
    """
    if not hasattr(quat_at, "many"):
        raise TypeError("quat_at must expose .many(times) (use make_quat_at, or add a "
                        "vectorised sibling to your sampler)")
    fx, fy, cx, cy = calib["fx"], calib["fy"], calib["cx"], calib["cy"]
    lgamma, wc = calib["lgamma"], np.asarray(calib["wc"])
    readout = calib["readout"]

    # float32 halves the memory traffic of the per-pixel einsum (measured 6x faster)
    map_x = np.full((frame_h, frame_w), np.nan, np.float32)
    map_y = np.full((frame_h, frame_w), np.nan, np.float32)

    def _apply_dist(xu, yu):
        if undistort is not None:
            return undistort(xu, yu)
        if use_ldc:
            return ldc_distort(xu, yu, wc, lgamma)
        return xu, yu

    # Per-source-row poses: one scalar interpolation per row, shared by every pixel.
    scan_n = frame_h if scan_axis == "row" else frame_w
    scan_times = [row_time(t0, readout, s, scan_n, ref="pixel_row") for s in range(scan_n)]
    # Gather map: forward is R(view)*R(row)^T, so the gather is R(row)*R(view)^T with the
    # row pose taken at that source row's exposure time.
    rows = quat_at.many(np.asarray(scan_times)).astype(np.float32)
    out_w = np.broadcast_to(np.asarray(q_out, dtype=np.float32), rows.shape)
    conj_out = out_w * np.array([1, -1, -1, -1], dtype=np.float32)   # conjugate of a batch
    a, b = rows, conj_out
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    q_rel = np.stack([aw*bw - ax*bx - ay*by - az*bz,
                      aw*bx + ax*bw + ay*bz - az*by,
                      aw*by - ax*bz + ay*bw + az*bx,
                      aw*bz + ax*by - ay*bx + az*bw], axis=1)
    R_scan = quat_to_R_many(q_rel)
    # Flat 9-coefficient table: per-pixel gather + elementwise matvec beats einsum on
    # a (chunk, W, 3, 3) gather, which was the hottest loop of the render.
    R_flat = R_scan.reshape(scan_n, 9)
    center_scan = R_flat[(scan_n - 1) // 2]
    # Convergence is checked per block; max_iter only bounds the worst-case block.
    iters = max_iter if (use_rs and row_ref == "pixel_row") else 1
    # Blocks are independent and numpy releases the GIL, so fan out without changing values.
    if workers <= 0:
        workers = min(4, os.cpu_count() or 1) if frame_w * frame_h >= THREAD_MIN_PIXELS else 1

    if scan_axis == "row":
        xs = np.arange(frame_w, dtype=np.float32)

        def _block(span):
            y0, y1 = span
            yy, xx = np.meshgrid(np.arange(y0, y1, dtype=np.float32), xs, indexing="ij")
            ray = np.stack([(xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx)], axis=-1)
            sy = yy.copy()          # the output row is the initial source-row guess
            sx = xx
            for _ in range(iters):
                if use_rs:
                    idx = np.clip(np.rint(sy).astype(np.int32), 0, frame_h - 1)
                    c = R_flat[idx]
                else:
                    c = center_scan
                vx = c[..., 0] * ray[..., 0] + c[..., 1] * ray[..., 1] + c[..., 2]
                vy = c[..., 3] * ray[..., 0] + c[..., 4] * ray[..., 1] + c[..., 5]
                vz = c[..., 6] * ray[..., 0] + c[..., 7] * ray[..., 1] + c[..., 8]
                xu, yu = vx / vz, vy / vz
                xd, yd = _apply_dist(xu, yu)
                sx, sy_new = fx * xd + cx, fy * yd + cy
                # always take the new guess so a single pass is a valid (non-RS) mapping
                converged = np.max(np.abs(sy_new - sy)) < tol
                sy = sy_new
                if not use_rs or converged:
                    break
            inside = (sx >= 0) & (sx <= frame_w - 1) & (sy >= 0) & (sy <= frame_h - 1)
            map_x[y0:y1] = np.where(inside, sx, np.nan)
            map_y[y0:y1] = np.where(inside, sy, np.nan)

        _dispatch([(y0, min(y0 + chunk, frame_h)) for y0 in range(0, frame_h, chunk)],
                  _block, workers)
    else:
        ys = np.arange(frame_h, dtype=np.float32)

        def _block(span):
            x0, x1 = span
            yy, xx = np.meshgrid(ys, np.arange(x0, x1, dtype=np.float32), indexing="ij")
            ray = np.stack([(xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx)], axis=-1)
            sx = xx          # the output column is the initial source-column guess
            sy = yy.copy()
            for _ in range(iters):
                if use_rs:
                    idx = np.clip(np.rint(sx).astype(np.int32), 0, frame_w - 1)
                    c = R_flat[idx]
                else:
                    c = center_scan
                vx = c[..., 0] * ray[..., 0] + c[..., 1] * ray[..., 1] + c[..., 2]
                vy = c[..., 3] * ray[..., 0] + c[..., 4] * ray[..., 1] + c[..., 5]
                vz = c[..., 6] * ray[..., 0] + c[..., 7] * ray[..., 1] + c[..., 8]
                xu, yu = vx / vz, vy / vz
                xd, yd = _apply_dist(xu, yu)
                sx_new, sy = fx * xd + cx, fy * yd + cy
                # always take the new guess so a single pass is a valid (non-RS) mapping
                converged = np.max(np.abs(sx_new - sx)) < tol
                sx = sx_new
                if not use_rs or converged:
                    break
            inside = (sx >= 0) & (sx <= frame_w - 1) & (sy >= 0) & (sy <= frame_h - 1)
            map_x[:, x0:x1] = np.where(inside, sx, np.nan)
            map_y[:, x0:x1] = np.where(inside, sy, np.nan)

        _dispatch([(x0, min(x0 + chunk, frame_w)) for x0 in range(0, frame_w, chunk)],
                  _block, workers)
    return map_x, map_y


def write_conf(out_path, fx, fy, cx, cy, width, height, crop_h, crop_v, grid_h, grid_v):
    """Write conf.yaml (field names must match quaternion_optimizer DataLoader)

    Intrinsics are real K in top-left origin pixels. The RCS grid stores corrected
    real-K pixels normalized to center-origin [-1,1], so the loader must convert the
    principal point back: cx_centered = calPrincipalX - (frameWidth-1)/2
    (same for y, with frameHeight).
    """
    with open(out_path, "w") as f:
        f.write(f"# camera intrinsics (real K, top-left origin pixels)\n"
                f"calFocalX: {fx:.6f}\ncalFocalY: {fy:.6f}\n"
                f"calPrincipalX: {cx:.6f}\ncalPrincipalY: {cy:.6f}\n"
                f"# frame geometry\n"
                f"frameWidth: {width}\nframeHeight: {height}\n"
                f"# RCS grid: control-point counts (gridRows x gridCols)\n"
                f"gridCols: {grid_h}\ngridRows: {grid_v}\n"
                f"# output crop window: fraction cropped away (cropFactor/2 per side)\n"
                f"cropFactorH: {crop_h}\ncropFactorV: {crop_v}\n")


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
