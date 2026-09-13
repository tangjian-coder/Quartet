#!/usr/bin/env python3
"""Check that the SDF chain is sensitive to the real K as expected.

Reproduces the C++ verifyConstraints forward warp for a case and reports the
SDF statistics under three intrinsics. The real K must differ from the virtual
(principal point at zero) and from a non-centered principal point; identical
values would mean K never reaches the warp.

Usage: python3 tests/sdf_scale_check.py <case_dir> [n_frames]
"""
import os
import re
import sys

import numpy as np

GRID_SIDE = 11
FRAMES_DEFAULT = 20


def parse_conf(path):
    """Read the camera/grid keys, accepting the legacy product names too."""
    fields = {'calFocalX': 'calFocalLength', 'frameWidth': 'frmWidth',
              'frameHeight': 'frmHeight', 'cropFactorH': 'gridFrmCropFctrH',
              'cropFactorV': 'gridFrmCropFctrV', 'gridCols': 'gridCntH',
              'gridRows': 'gridCntV', 'calFocalY': None,
              'calPrincipalX': None, 'calPrincipalY': None}
    with open(path) as f:
        text = f.read()
    conf = {}
    for name, legacy in fields.items():
        for key in (name,) if legacy is None else (name, legacy):
            m = re.search(r'^\s*' + key + r'\s*:\s*([0-9.eE+-]+)', text, re.M)
            if m is not None:
                conf[name] = float(m.group(1))
                break
    return conf


def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def sdf_to_polygon(p, poly):
    """Signed distance with the winding-number sign, as GeometryTools.cpp."""
    min_d, inside = np.inf, 0
    for i in range(len(poly)):
        p1, p2 = poly[i], poly[(i + 1) % len(poly)]
        ab = p2 - p1
        t = np.clip(np.dot(p - p1, ab) / np.dot(ab, ab), 0.0, 1.0)
        min_d = min(min_d, np.linalg.norm(p - (p1 + t * ab)))
        cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
        if p1[1] <= p[1]:
            inside += 1 if (p2[1] > p[1] and cross > 0) else 0
        else:
            inside -= 1 if (p2[1] <= p[1] and cross < 0) else 0
    return -min_d if inside != 0 else min_d


def boundary_indices():
    es = GRID_SIDE - 1
    idx = list(range(GRID_SIDE))
    idx += [i * GRID_SIDE + es for i in range(1, es + 1)]
    idx += [es * GRID_SIDE + j for j in range(es - 1, -1, -1)]
    idx += [i * GRID_SIDE for i in range(es - 1, 0, -1)]
    return idx


def locate(case, names):
    """Accept both the processed layout and the legacy DbgFile_* names."""
    for name in names:
        path = os.path.join(case, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'none of {names} under {case}')


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else '/home/tangjian5/data/gopro_imu/processed/apple3'
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else FRAMES_DEFAULT

    conf = parse_conf(locate(case, ('conf.yaml',)))
    W, H = int(conf['frameWidth']), int(conf['frameHeight'])
    fx = conf['calFocalX']
    fy = conf.get('calFocalY', fx)
    cx = conf.get('calPrincipalX', (W - 1) / 2)
    cy = conf.get('calPrincipalY', (H - 1) / 2)
    crop_h = conf.get('cropFactorH', 0.0)
    crop_v = conf.get('cropFactorV', 0.0)
    cx0, cy0 = (W - 1) / 2, (H - 1) / 2

    rcs = np.loadtxt(locate(case, ('rcs.csv', 'DbgFile_WarpGrids_RSCFRMMid.csv')),
                     delimiter=',')[:n_frames]
    grids = np.zeros((rcs.shape[0], GRID_SIDE * GRID_SIDE, 2))
    grids[..., 0] = rcs[:, 0::2] * (W - 1) / 2
    grids[..., 1] = rcs[:, 1::2] * (H - 1) / 2

    quats = np.loadtxt(locate(case, ('quat.csv', 'DbgFile_Frame_OriginalCamOrient.csv')),
                       delimiter=',', skiprows=1)[:, 1:5]
    bidx = boundary_indices()
    hw, hh = (W - 1) / 2, (H - 1) / 2
    corners = np.array([[-hw * (1 - crop_h), -hh * (1 - crop_v)],
                        [hw * (1 - crop_h), -hh * (1 - crop_v)],
                        [hw * (1 - crop_h), hh * (1 - crop_v)],
                        [-hw * (1 - crop_h), hh * (1 - crop_v)]])

    variants = {
        'real K (centered)': np.array([[fx, 0, cx - cx0], [0, fy, cy - cy0], [0, 0, 1.0]]),
        'not centered': np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]),
        'virtual f=fx, pp=0': np.array([[fx, 0, 0], [0, fx, 0], [0, 0, 1.0]]),
        'fy swapped with fx': np.array([[fy, 0, cx - cx0], [0, fx, cy - cy0], [0, 0, 1.0]]),
    }

    # Inter-frame relative rotation: keeps dq away from identity so that K is
    # exercised. The state is the next frame's raw pose against the current raw.
    ref = min(n_frames - 1, len(quats) - 1)
    print(f'case={case}  frames={ref + 1}  fx={fx:.3f} fy={fy:.3f} '
          f'pp_center=({cx - cx0:+.3f}, {cy - cy0:+.3f})')

    results = {}
    for name, K in variants.items():
        Kinv = np.linalg.inv(K)
        all_sdf = []
        for i in range(ref):
            R_dq = quat_to_R(quat_mul(quats[i + 1], quat_conj(quats[i])))
            Ht = K @ R_dq @ Kinv
            poly = grids[i][bidx]
            warped = (Ht @ np.c_[poly, np.ones(len(poly))].T).T
            warped = warped[:, :2] / warped[:, 2:3]
            all_sdf += [sdf_to_polygon(c, warped) for c in corners]
        results[name] = (max(all_sdf), min(all_sdf))
        print(f'  {name:22s} max_sdf={results[name][0]:10.4f}  min_sdf={results[name][1]:10.4f}')

    base = results['real K (centered)']
    print('deltas vs real K (px):')
    for name, val in results.items():
        if name != 'real K (centered)':
            print(f'  {name:22s} dmax={val[0] - base[0]:+9.4f}  dmin={val[1] - base[1]:+9.4f}')

    # Identity pose (state == raw) puts the corners exactly on the grid, so the
    # margin is purely geometric and K-independent; it must be negative (inside).
    geo = []
    for i in range(min(len(quats), len(grids))):
        geo += [sdf_to_polygon(c, grids[i][bidx]) for c in corners]
    print(f'identity-pose margin (K-independent): max_sdf={max(geo):.4f} '
          f'min_sdf={min(geo):.4f} -> {"inside" if max(geo) < 0 else "OUTSIDE"}')

    # The real K must not coincide with any of the approximations. A legacy conf
    # carries no real-K fields, so the variants coincide by construction there.
    has_real_k = 'calFocalY' in conf and 'calPrincipalX' in conf
    if not has_real_k:
        print('LEGACY conf: no real-K fields; variants coincide by construction (fallback OK)')
        return 0
    ok = all(abs(v[0] - base[0]) > 1e-6 for k, v in results.items() if k != 'real K (centered)')
    print('PASS: real K differs from every variant' if ok else 'FAIL: some variant is indistinguishable')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
