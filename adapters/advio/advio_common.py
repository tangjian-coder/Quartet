"""ADVIO dataset adapter: raw iPhone IMU + frames.mov, reusing the shared warp core (adapters/common.py).

ADVIO (advio-XX) ships per-device streams under iphone/ (accelerometer.csv, gyro.csv,
frames.csv, arkit.csv, frames.mov) plus ground-truth/pose.csv. Timestamps are seconds on
one device clock, so no LiU-style offset repair is needed.

Calibration comes from the official repo (AaltoVision/ADVIO) calibration/ yaml matching
this sequence's device batch: pinhole intrinsics, radial-tangential distortion, T_cam_imu
(the camera frame is the IMU frame rotated 180 deg about x) and approximate IMU biases.
ARKit and ground-truth files both store camera->world attitudes, so both are conjugated on
import; the gyro stream instead goes through integrate_gyro and R_cam_imu.
"""
import io
import os
import sys
import zipfile

import numpy as np

# shared math lives in adapters/common.py; only ADVIO-specific I/O stays here
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import (SLERP_LIN_D, quat_mul, quat_conj, integrate_gyro,   # noqa: E402
                    R_to_quat, slerp)

ADVIO_DIR = '/home/tangjian5/data/ADVIO'
CALIB_DIR = os.path.join(ADVIO_DIR, 'calibration')

# Official batch -> calibration file mapping (calibration/README.md table):
# 1-12 -> iphone-2, 13-17 -> iphone-3, 18-19 -> iphone-1, 20-23 -> iphone-4
BATCHES = ((1, 12, 'iphone-02.yaml'), (13, 17, 'iphone-03.yaml'),
           (18, 19, 'iphone-01.yaml'), (20, 23, 'iphone-04.yaml'))
# the README table labels its bias rows by sequence range
BIAS_ROWS = ((1, 12, '1-12'), (13, 17, '13-17'), (18, 19, '18-19'), (20, 23, '20-23'))

# ARKit/GT store camera->world in ARKit camera axes (x right, y up, z back); video pixels
# are x right, y down, z forward, so poses need a 180 deg roll about x before rendering.
Q_VIDEO_FROM_ARKIT = np.array([0.0, 1.0, 0.0, 0.0])

_CALIB_CACHE = {}

_R_to_quat = R_to_quat
_slerp = slerp
_BIAS_CACHE = {}


def _seq_number(seq):
    """'advio-07' -> 7, so the calibration batch can be selected."""
    return int(str(seq).rstrip('/').split('-')[-1])


def _calib_file(seq):
    n = _seq_number(seq)
    for lo, hi, name in BATCHES:
        if lo <= n <= hi:
            return name
    raise ValueError(f'{seq}: no official calibration batch for sequence {n}')


def load_official_calib(seq):
    """Parse the official calibration yaml covering this sequence (cached).

    The shipped yamls contain hard tabs, which YAML rejects, so tabs are normalised to
    spaces before parsing. Distortion is radial-tangential in OpenCV order [k1,k2,p1,p2]
    (the README table labels the same four numbers r1,r2,k1,k2).
    """
    name = _calib_file(seq)
    if name in _CALIB_CACHE:
        return _CALIB_CACHE[name]
    import yaml
    path = os.path.join(CALIB_DIR, name)
    with open(path) as f:
        txt = f.read().replace('\t', ' ')
    cam = yaml.safe_load(txt)['cameras'][0]['camera']
    fx, fy, cx, cy = (float(v) for v in cam['intrinsics']['data'])
    k1, k2, p1, p2 = (float(v) for v in cam['distortion']['parameters']['data'])
    T = np.array(cam['T_cam_imu']['data'], dtype=np.float64)
    out = dict(fx=fx, fy=fy, cx=cx, cy=cy,
               w=int(cam['image_width']), h=int(cam['image_height']),
               radtan=(k1, k2, p1, p2),
               R_cam_imu=T[:3, :3].copy(),
               # every official file sets line-delay-nanoseconds: 0 (no RS calibration)
               readout=float(cam.get('line-delay-nanoseconds', 0)) * 1e-9,
               source=name)
    _CALIB_CACHE[name] = out
    return out


def load_imu_bias(seq):
    """Approximate additive IMU biases for this batch, parsed from calibration/README.md."""
    name = _calib_file(seq)
    if name in _BIAS_CACHE:
        return _BIAS_CACHE[name]
    # the README numbers the rows by sequence range, not by yaml file name
    n = _seq_number(seq)
    row = next((label for lo, hi, label in BIAS_ROWS if lo <= n <= hi), None)
    if row is None:
        raise ValueError(f'{seq}: no bias row for sequence {n}')
    with open(os.path.join(CALIB_DIR, 'README.md')) as f:
        for line in f:
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if cells and cells[0] == row and len(cells) == 7:
                vals = [float(c) for c in cells[1:]]
                out = dict(acc=np.array(vals[:3]), gyro=np.array(vals[3:]), source=row)
                _BIAS_CACHE[name] = out
                return out
    raise ValueError(f'{name}: bias row {row} not found in calibration/README.md')


def load_csv(seq, rel):
    """Read a headerless ADVIO csv, from the zip when present, else from the extracted dir."""
    zp = os.path.join(ADVIO_DIR, f'{seq}.zip')
    if os.path.exists(zp):
        with zipfile.ZipFile(zp) as zf:
            return np.loadtxt(io.StringIO(zf.read(f'{seq}/{rel}').decode()), delimiter=',')
    return np.loadtxt(os.path.join(ADVIO_DIR, seq, rel), delimiter=',')


def frames_mov(seq):
    """Path to the extracted frames.mov (cv2 cannot read inside a zip)."""
    return os.path.join(ADVIO_DIR, seq, 'iphone', 'frames.mov')


def video_info(seq):
    """Return (W, H, n_frames, fps) read from the shipped video."""
    import cv2
    cap = cv2.VideoCapture(frames_mov(seq))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return W, H, n, fps


def calib_for(seq, readout):
    """Calibration dict for the shared warp core, read from the official files.

    ldc_type tells warp_grid/warp_pixels which correction model to apply; ADVIO is a
    pinhole camera with radial-tangential distortion, so lgamma/wc are placeholders and
    the radtan parameters are the real ones.
    """
    c = load_official_calib(seq)
    b = load_imu_bias(seq)
    return dict(fx=c['fx'], fy=c['fy'], cx=c['cx'], cy=c['cy'],
                lgamma=1.0, wc=(0.0, 0.0), readout=readout,
                radtan=c['radtan'], ldc_type='radtan', R_cam_imu=c['R_cam_imu'],
                gyro_bias=b['gyro'], acc_bias=b['acc'], source=c['source'])


def _quat_sampler(ts, qs):
    """Return quat_at(t) interpolating a sign-continuous quaternion series.

    Nearest-neighbour sampling quantises a 100Hz attitude onto 60fps frame times, which
    shows up as per-frame jitter during fast motion; slerp removes that quantisation.
    """
    qs = np.asarray(qs, dtype=np.float64).copy()
    for i in range(1, len(qs)):
        if np.dot(qs[i], qs[i - 1]) < 0:
            qs[i] = -qs[i]
    ts = np.asarray(ts, dtype=np.float64)
    n = len(ts)

    def quat_at(t):
        k = int(np.clip(np.searchsorted(ts, t) - 1, 0, n - 2))
        span = ts[k + 1] - ts[k]
        a = 0.0 if span <= 0 else float(np.clip((t - ts[k]) / span, 0.0, 1.0))
        return _slerp(qs[k], qs[k + 1], a)

    def many(times):
        """Vectorised quat_at for all scan rows at once (same searchsorted + slerp)."""
        t = np.asarray(times, dtype=np.float64)
        k = np.clip(np.searchsorted(ts, t) - 1, 0, n - 2)
        span = ts[k + 1] - ts[k]
        a = np.where(span > 0,
                     np.clip((t - ts[k]) / np.where(span > 0, span, 1.0), 0.0, 1.0), 0.0)
        q0, q1 = qs[k], qs[k + 1]
        d = np.sum(q0 * q1, axis=1)
        flip = d < 0
        q1 = np.where(flip[:, None], -q1, q1)
        d = np.abs(d)
        out = np.empty_like(q0)
        lin = d > SLERP_LIN_D
        if lin.any():
            f = a[lin][:, None]
            out[lin] = q0[lin] + f * (q1[lin] - q0[lin])
        if (~lin).any():
            th = np.arccos(np.clip(d[~lin], -1.0, 1.0))[:, None]
            f = a[~lin][:, None]
            out[~lin] = (np.sin((1 - f) * th) * q0[~lin] + np.sin(f * th) * q1[~lin]) / np.sin(th)
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    quat_at.many = many
    return quat_at


def make_quat_at(seq, pose='arkit'):
    """Return (t_frames, quat_at, n_frames) with CAMERA attitudes in the pipeline convention
    (pc = R(q) * pw, i.e. world -> camera).

    pose='arkit' : per-frame ARKit attitude, sampled on the frame timestamps. Validated:
                   fix0 frame-to-frame residual drops 4.12 px (raw) -> 0.34 px
    pose='gt'    : ground-truth camera attitude, sampled on the frame timestamps
    pose='imu'   : gyro (100Hz) minus the official bias, integrated then rotated by R_cam_imu
                   (NOT yet validated against the video: fix0 still shows ~5 px/frame residual)
    ARKit and ground-truth files both store camera->world (c2w) attitudes, so they are
    conjugated on import and re-axed into the video frame before sampling.
    """
    t_frames = load_csv(seq, 'iphone/frames.csv')[:, 0]
    n = len(t_frames)

    if pose == 'arkit':
        ak = load_csv(seq, 'iphone/arkit.csv')
        qs = np.stack([quat_conj(q) for q in ak[:, 4:8]])
        qs = np.stack([quat_mul(Q_VIDEO_FROM_ARKIT, q) for q in qs])
        qs = np.stack([q / np.linalg.norm(q) for q in qs])
        return t_frames, _quat_sampler(ak[:, 0], qs), n

    if pose == 'gt':
        gt = load_csv(seq, 'ground-truth/pose.csv')
        qs = gt[:, 4:8] / np.linalg.norm(gt[:, 4:8], axis=1, keepdims=True)
        qs = np.stack([quat_conj(q) for q in qs])
        qs = np.stack([quat_mul(Q_VIDEO_FROM_ARKIT, q) for q in qs])
        return t_frames, _quat_sampler(gt[:, 0], qs), n

    if pose == 'imu':
        g = load_csv(seq, 'iphone/gyro.csv')
        dt = float(np.median(np.diff(g[:, 0])))
        # Bias and camera-IMU rotation come from the official files for this sequence's batch
        c = load_official_calib(seq)
        b = load_imu_bias(seq)
        # world->camera attitude of the IMU, then the constant camera-IMU rotation on the left
        qs_imu = integrate_gyro(g[:, 1:4] - b['gyro'], dt)
        q_ci = _R_to_quat(c['R_cam_imu'])
        qs = np.stack([quat_mul(q_ci, q) for q in qs_imu])
        return t_frames, _quat_sampler(g[:, 0], qs), n

    raise ValueError(f"unknown pose '{pose}' (expected arkit / gt / imu)")
