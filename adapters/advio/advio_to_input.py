"""ADVIO -> C++ pipeline inputs (quat csv + RCS grid + conf.yaml).

The optimizer expects raw poses as (time,w,x,y,z) in the pipeline w2c convention,
a normalized [-1,1] RCS grid per frame, and intrinsics in top-left pixel coords.
The RCS grid is a regular source-frame grid pushed through the shared forward chain
(lens distortion removal, plus de-RS when the camera has a readout), so its border is
curved and no longer the frame rectangle. ADVIO ships no RS calibration (readout=0), so
the grid does not depend on the frame and is written once, then shared by every sequence.

Usage: python3 adapters/advio/advio_to_input.py --seq advio-01 [--out-dir DIR]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import advio_common as ac  # noqa: E402
from common import warp_grid, write_conf  # noqa: E402

GRID_H, GRID_V = 55, 55  # must match kGridSide in include/GridLayout.h
MAX_FRAMES = 20000      # rows in the shared grid file; longer than any sequence, the loader stops early


def write_quat_csv(path, t_frames, quat_at):
    """Raw attitudes; the C++ loader skips the header row and needs exactly 5 columns."""
    with open(path, 'w') as f:
        f.write('time,w,x,y,z\n')
        for t in t_frames:
            w, x, y, z = quat_at(t)
            f.write(f'{int(round(t * 1e9))},{w:.9f},{x:.9f},{y:.9f},{z:.9f}\n')


def grid_row(calib, quat_at, t0, width, height):
    """One RCS row: regular source grid -> correction chain -> center normalization.

    The same definition as the GoPro chain, so both share one convention: the border is
    not axis aligned after distortion removal, and the normalization base is (W-1)/2.
    """
    readout = calib['readout']
    cx0, cy0 = (width - 1) / 2.0, (height - 1) / 2.0
    q_mid = quat_at(t0 + readout / 2)
    grid = warp_grid(calib, quat_at, t0, q_mid, width, height, GRID_H, GRID_V,
                     use_rs=True, ldc_type=calib['ldc_type'])
    vals = []
    for j in range(GRID_V):
        for i in range(GRID_H):
            vals.append(f'{(grid[j, i, 0] - cx0) / cx0:.6f},'
                        f'{(grid[j, i, 1] - cy0) / cy0:.6f}')
    return ','.join(vals)


def write_rcs_csv(path, calib, quat_at, t_frames, width, height, n_rows=MAX_FRAMES):
    """Write the per-frame grid rows consumed by the optimizer.

    With readout=0 the de-RS homography is the identity, so every frame has the same grid
    (only the lens model acts, and it does not depend on time). Calling warp_grid per frame
    would return bit-identical rows 15589 times, so the row is computed once and repeated.
    A non-zero readout keeps the per-frame loop: rows really differ then.
    """
    if calib['readout'] == 0:
        row = grid_row(calib, quat_at, t_frames[0], width, height)
        with open(path, 'w') as f:
            for _ in range(n_rows):
                f.write(row + '\n')
        return
    with open(path, 'w') as f:
        for t0 in t_frames:
            f.write(grid_row(calib, quat_at, t0, width, height) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='advio-01')
    ap.add_argument('--pose', choices=['arkit', 'gt'], default='arkit')
    ap.add_argument('--crop', type=float, default=0.2,
                    help='cropFactor; keeps (1-crop) of each side inside the frame')
    ap.add_argument('--debug', action='store_true',
                    help='write under output/pipeline_debug/<seq> so the production dir is untouched')
    ap.add_argument('--out-dir', default=None)
    a = ap.parse_args()

    base = 'pipeline_debug' if a.debug else 'pipeline'
    out_dir = a.out_dir or os.path.join(ac.ADVIO_DIR, 'output', base, a.seq)
    os.makedirs(out_dir, exist_ok=True)

    t_frames, quat_at, n = ac.make_quat_at(a.seq, a.pose, True, 0.0)
    W, H, n_video, fps = ac.video_info(a.seq)
    if n != n_video:
        # the optimizer indexes poses and frames 1:1, so a mismatch would shift the video
        print(f'[WARN] {a.seq}: {n} pose rows vs {n_video} video frames')
    calib = ac.calib_for(a.seq, 0.0)

    write_quat_csv(os.path.join(out_dir, 'quat.csv'), t_frames, quat_at)
    # One shared grid for every sequence: with readout=0 the rows are identical. The shape
    # and model live in conf.yaml, so the name stays plain rcs.csv.
    shared = os.path.join(ac.ADVIO_DIR, 'output', base, 'rcs.csv')
    if not os.path.exists(shared):
        write_rcs_csv(shared, calib, quat_at, t_frames, W, H)
    write_conf(os.path.join(out_dir, 'conf.yaml'),
               calib['fx'], calib['fy'], calib['cx'], calib['cy'],
               W, H, a.crop, a.crop, GRID_H, GRID_V)
    print(f'{a.seq}: {len(t_frames)} frames {W}x{H} f={calib["fx"]:.1f} -> {out_dir}')
    print(f'  rcs: {shared}  (shared {GRID_H}x{GRID_V} {calib["ldc_type"]} grid, pass to --rcs)')
    if a.debug:
        # Two things are easy to get wrong here: the c2w/w2c reading of the pose files and
        # whether the grid matches the frame, so print both instead of guessing.
        _, raw_at, _ = ac.make_quat_at(a.seq, a.pose, False, 0.0)
        print(f'  debug: frame0 quat written (converted) = {np.round(quat_at(t_frames[0]), 5)}')
        print(f'  debug: frame0 quat unconverted        = {np.round(raw_at(t_frames[0]), 5)}')
        print(f'  debug: principal ({calib["cx"]:.3f}, {calib["cy"]:.3f}) vs '
              f'frame center ({W/2:.1f}, {H/2:.1f})')
        print(f'  debug: radtan={calib["radtan"]} source={calib["source"]}')
        g = warp_grid(calib, quat_at, t_frames[0], quat_at(t_frames[0]), W, H,
                      GRID_H, GRID_V, use_rs=True, ldc_type=calib['ldc_type'])
        print(f'  debug: grid x[{g[...,0].min():.2f},{g[...,0].max():.2f}] '
              f'y[{g[...,1].min():.2f},{g[...,1].max():.2f}] px '
              f'row0 y ptp={np.ptp(g[0,:,1]):.4f} (0 = rectangle)')


if __name__ == '__main__':
    main()
