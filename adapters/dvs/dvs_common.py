"""DVS (Deep Online Fused Video Stabilization) data access and pose chain.

Forward chain in the order the sensor pipeline uses: OIS (lens shift) -> RSC (rows).
DVS ships no distortion parameters, so the lens model is `ldc_type=None` and the
shared warp core skips it. OIS is stored as a lens shift in active-array pixels
(4032x2272); the official code folds it into the principal point, a pure translation.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from common import quat_mul, quat_mul_many, rotvec_to_quat, rotvec_to_quat_many  # noqa: E402

DVS_DIR = os.environ.get('DVS_DIR',
                         os.path.expanduser('~/data/DeepFVS/dataset_release'))
ACTIVE_W, ACTIVE_H = 4032, 2272
DEFAULT_FOV = 1.27


def find_clip(seq):
    hits = glob.glob(os.path.join(DVS_DIR, '*', seq))
    if not hits:
        raise FileNotFoundError(f'no clip {seq} under {DVS_DIR}')
    return hits[0]


def load_clip(seq):
    d = find_clip(seq)
    ts = np.loadtxt(glob.glob(d + '/*frame_timestamps_*.txt')[0])
    gyro = np.loadtxt(glob.glob(d + '/*gyro_log_*.txt')[0])
    ois = np.loadtxt(glob.glob(d + '/*ois_log_*.txt')[0])[:, -3:]   # x, y, time_ns
    calib = {}
    cfiles = glob.glob(d + '/calib.txt')
    if cfiles:
        with open(cfiles[0]) as f:
            calib = json.load(f)
    return {'dir': d, 'ts': ts, 'gyro': gyro, 'ois': ois, 'calib': calib}


def frame_info(d):
    W, H, n, fps = video_info(d)
    ts = d['ts']
    # official loader shifts the frame and ois clocks by half the exposure; seconds keep
    # the whole chain in the same unit as readout (ns here would silently drop the de-RS term)
    t_frame = (ts[:, 0] - ts[:, 1] / 2) * 1e-9
    t_ois = (ts[:, 4] - ts[:, 1] / 2) * 1e-9
    # rs_time_us is stored in microseconds; the timestamp column is in ns
    rs_s = (float(d['calib']['rs_time_us']) * 1e-6 if 'rs_time_us' in d['calib']
            else float(ts[0, 3]) * 1e-9)
    fov = float(d['calib'].get('fov', DEFAULT_FOV))
    return {'W': W, 'H': H, 'n': n, 'fps': fps, 't_frame': t_frame,
            't_ois': t_ois, 'readout': rs_s, 'fov': fov, 'f': W / fov}


def video_info(d):
    import cv2
    cap = cv2.VideoCapture(video_path(d)[0])
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    ts = d['ts']
    dur = (ts[min(n, len(ts)) - 1, 0] - ts[0, 0]) / 1e9
    return W, H, n, n / dur if dur > 0 else 60.0


def video_path(d):
    return [p for p in glob.glob(d['dir'] + '/*.mp4') if 'no_ois' not in p]


def integrate_gyro(gyro):
    """Integrate the gyro log into (w,x,y,z) poses, official DVS convention.

    Kept apart from common.integrate_gyro on purpose: DVS reorders the axes to
    [col2, col1, col3] and left-multiplies the increment, which the shared
    right-multiplying integrator does not express.
    """
    t = gyro[:, 0] * 1e-6                     # us -> s
    w = gyro[:, [2, 1, 3]]
    q = np.zeros((len(t), 4))                  # (w, x, y, z)
    q[0] = (1.0, 0.0, 0.0, 0.0)
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        wv = w[i]
        L = np.linalg.norm(wv)
        if L < 1e-9 or dt <= 0:
            q[i] = q[i - 1]
            continue
        h = L * dt / 2
        dq = np.array([np.cos(h), wv[0] / L * np.sin(h),
                       wv[1] / L * np.sin(h), wv[2] / L * np.sin(h)])
        qq = quat_mul(dq, q[i - 1])
        q[i] = qq / np.linalg.norm(qq)
    return t, q


def make_quat_at(seq, ois_scale=1.0, exp_offset=0.0, use_ois=True, f_scale=1.0):
    """Return (t_frame, quat_at, info) with our pipeline convention (w,x,y,z, w2c).

    OIS is folded into an equivalent small rotation, which reproduces the official
    principal-point shift for distant content:
        dx = ois_x/4032*W, dy = ois_y/2272*H  ->  ry = dx/f, rx = -dy/f
    """
    d = load_clip(seq)
    info = frame_info(d)
    # fov is the only intrinsic DVS ships, so a scale knob absorbs its calibration error
    info['f'] = info['f'] * f_scale
    gt, qg = integrate_gyro(d['gyro'])
    # OIS log carries ns; store it in seconds so every lookup matches the pose clock
    ois = d['ois'].copy()
    ois[:, 2] *= 1e-9
    f = info['f']
    W, H = info['W'], info['H']

    def _ois_quat(o):
        dx = o[0] / ACTIVE_W * W * ois_scale
        dy = o[1] / ACTIVE_H * H * ois_scale
        return rotvec_to_quat(np.array([-dy / f, dx / f, 0.0]))

    def _quat(t):
        i = min(np.searchsorted(gt, t), len(gt) - 1)
        q = qg[i].copy()
        if use_ois:
            j = min(np.searchsorted(ois[:, 2], t + exp_offset), len(ois) - 1)
            q = quat_mul(_ois_quat(ois[j, :2]), q)
            q = q / np.linalg.norm(q)
        return q

    def _many(tts):
        """Same result for a whole row/column of times, without the per-time loop."""
        tts = np.asarray(tts, dtype=float)
        idx = np.clip(np.searchsorted(gt, tts), 0, len(gt) - 1)
        q = qg[idx].copy()
        if use_ois:
            j = np.clip(np.searchsorted(ois[:, 2], tts + exp_offset), 0, len(ois) - 1)
            o = ois[j, :2]
            dx = o[:, 0] / ACTIVE_W * W * ois_scale
            dy = o[:, 1] / ACTIVE_H * H * ois_scale
            rv = np.stack([-dy / f, dx / f, np.zeros(len(tts))], axis=1)
            q = quat_mul_many(rotvec_to_quat_many(rv), q)
            q = q / np.linalg.norm(q, axis=1, keepdims=True)
        return q

    # warp_pixels batches row times, so the sampler must expose .many
    _quat.many = _many
    return info['t_frame'][:info['n']], _quat, info


def ois_at(seq, ois_scale=1.0, exp_offset=0.0):
    """Return ois(t) -> (dx, dy) in output pixels: the logged lens shift as a translation.

    Kept separate from the pose: OIS is a measured physical shift, not a degree of freedom
    the optimizer should touch, so the warp core applies it as a translation.
    """
    d = load_clip(seq)
    info = frame_info(d)
    ois = d['ois'].copy()
    ois[:, 2] *= 1e-9
    W, H = info['W'], info['H']

    def _ois(t):
        j = min(np.searchsorted(ois[:, 2], t + exp_offset), len(ois) - 1)
        return np.array([ois[j, 0] / ACTIVE_W * W * ois_scale,
                         ois[j, 1] / ACTIVE_H * H * ois_scale])

    def _many(tts):
        tts = np.asarray(tts, dtype=float)
        j = np.clip(np.searchsorted(ois[:, 2], tts + exp_offset), 0, len(ois) - 1)
        o = ois[j, :2]
        return np.stack([o[:, 0] / ACTIVE_W * W * ois_scale,
                         o[:, 1] / ACTIVE_H * H * ois_scale], axis=1)

    _ois.many = _many
    return _ois


def calib_dict(info):
    """Calib record for common.warp_pixels; DVS has no lens model, so ldc_type is None."""
    W, H, f = info['W'], info['H'], info['f']
    return {'fx': f, 'fy': f, 'cx': (W - 1) / 2.0, 'cy': (H - 1) / 2.0,
            'readout': info['readout'], 'ldc_type': None}
