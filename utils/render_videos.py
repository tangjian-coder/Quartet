#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the stabilized video with scattered-point interpolation.

Pipeline (same as the reference product implementation):
  for each frame:
    gridFinal = Ht * rcs_grid       # grid vertices warped by the attitude fix
    grid_map  = griddata(gridFinal -> regular grid) on a coarse query grid
    grid_map  = remap(grid_map, 0..coarse-1 index space)   # coarse -> full res
    out       = remap(frame, grid_map)                     # sample the raw frame

  Coords: gridFinal (griddata sample points) is normalized [-1,1]; the
  interpolation values are output-frame pixels, which bakes the content->
  video vertical compression (e.g. 1200->1080) into the map.

Attitude warp: Ht = K * R(q_opt * q_raw^-1) * K^-1  (same as verifyConstraints)

Usage:
  python3 utils/render_videos.py [--video ...] [--grid ...] [--quat-raw ...]
      [--quat-opt ...] [--yaml ...] [--out ...] [--draw-grid] [--max-frames N]
  All paths default to the rotation_20250711_100523 case.
"""

import argparse
import csv
import math
import os
import sys
import time
import re

import numpy as np
import cv2
from scipy.interpolate import griddata

# Quaternion helpers (w, x, y, z) -- Eigen order, w first.
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def qconj(q):
    return (q[0], -q[1], -q[2], -q[3])


def qnormalize(q):
    n = math.sqrt(sum(v * v for v in q))
    return tuple(v / n for v in q)


def q_to_R(q):
    w, x, y, z = q
    return ((1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
            (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
            (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)))


def parse_conf(yaml_path):
    """Read the camera/grid keys with regex; no yaml dependency.

    Both the explicit key names and the legacy product names are accepted, so
    configs written before the real-K switch keep loading.
    """
    fields = [('calFocalX', 'calFocalLength'), ('frameWidth', 'frmWidth'),
              ('frameHeight', 'frmHeight'), ('cropFactorH', 'gridFrmCropFctrH'),
              ('cropFactorV', 'gridFrmCropFctrV'), ('gridCols', 'gridCntH'),
              ('gridRows', 'gridCntV'), ('calFocalY', None),
              ('calPrincipalX', None), ('calPrincipalY', None)]
    keys = {}
    with open(yaml_path) as f:
        text = f.read()
    for key, legacy in fields:
        names = (key,) if legacy is None else (key, legacy)
        for name in names:
            m = re.search(r'^\s*' + name + r'\s*:\s*([0-9.eE+-]+)', text, re.M)
            if m is not None:
                keys[key] = float(m.group(1))
                break
    return keys


def load_quats(path, skip_first):
    """CSV rows of quaternions; skip the header row if skip_first is True."""
    quats = []
    with open(path) as f:
        for row in csv.reader(f):
            if not row:
                continue
            if skip_first:
                skip_first = False
                continue
            quats.append(qnormalize(tuple(float(v) for v in row[-4:])))
    return quats


def load_grids(path, gv, gh):
    """Each CSV row -> gv x gh grid of normalized coords in [-1, 1]."""
    grids = []
    with open(path) as f:
        for row in csv.reader(f):
            vals = [float(v) for v in row]
            assert len(vals) == gv * gh * 2, \
                f'row has {len(vals)} values, expected {gv * gh * 2}'
            pts = [(vals[k], vals[k + 1]) for k in range(0, len(vals), 2)]
            grids.append([pts[r * gh:(r + 1) * gh] for r in range(gv)])
    return grids


def get_warped_frame(frame, Ht, gridRSC, W, H, frame_w, frame_h, coarse=256, use_coarse=False):
    """Attitude-correct the raw frame via scattered-point interpolation.

    use_coarse: True  -> two-stage pipeline: coarse griddata table, then
                 remap-upscale to full res (fast, sub-pixel smear).
                 False (default) -> direct full-res griddata (exact, slow).
    """
    gridRSC = np.array(gridRSC)
    # Normalized [-1,1] -> pixel coords (center-origin) for the Ht warp.
    gridRSC[:,:,0] *= (W - 1)/2
    gridRSC[:,:,1] *= (H - 1)/2
    gv, gh = gridRSC.shape[:2]   # grid dims come from the loaded data

    # gridFinal = Ht * rcs_grid: warp the grid vertices in pixel space,
    # then back to normalized coords -- where the grid lands after the
    # attitude correction (this becomes the griddata sample positions).
    gridRSC = np.concatenate([gridRSC, np.ones((gv, gh, 1))], axis=-1)
    gridFinal = Ht @ gridRSC.reshape(-1, 3).T
    gridFinal = gridFinal[:2, :] / gridFinal[-1, :]
    gridFinal[0] /= (W - 1) / 2
    gridFinal[1] /= (H - 1) / 2
    gridFinal = gridFinal.T.reshape(gv, gh, -1)

    # Regular grid on the output frame: the interpolation target values.
    # Mixed domains: point lives in normalized [-1,1] (gridFinal, H-space
    # semantic), value lives in output pixel space (frame_w x frame_h) --
    # griddata allows any value domain, and this bakes the 1200->1080
    # vertical compression into the interpolation target.
    x = np.linspace(0, frame_w - 1, gh, dtype=np.float32)
    y = np.linspace(0, frame_h - 1, gv, dtype=np.float32)
    map_x, map_y = np.meshgrid(x, y)
    grid = np.concatenate([map_x[..., None], map_y[..., None]], axis=-1).astype(np.float32)

    # griddata samples: warped grid vertices (positions) -> regular output
    # grid pixels (values); queries are the output pixel grid in [-1,1].
    point = gridFinal.reshape(-1, 2)
    value = grid.reshape(-1, 2)

    # Query grid: full-res direct, or coarse table upsampled via remap.
    if use_coarse:
        grid_map_coarse_x, grid_map_coarse_y = np.meshgrid(np.linspace(-1, 1, coarse),
                                                           np.linspace(-1, 1, coarse))
        grid_map_coarse = griddata(point, value, (grid_map_coarse_x, grid_map_coarse_y))
        # Upscale the coarse map to full res; map1/map2 must be float32
        # (CV_32FC1) and live in the coarse table's index space (0..coarse-1).
        grid_map_x, grid_map_y = np.meshgrid(np.linspace(0, coarse - 1, frame_w),
                                             np.linspace(0, coarse - 1, frame_h))
        grid_map = cv2.remap(grid_map_coarse.astype(np.float32),
                             grid_map_x.astype(np.float32), grid_map_y.astype(np.float32),
                             cv2.INTER_LINEAR)
    else:
        # Direct full-res interpolation: query = output pixel grid, so no
        # upsampling smear; hull-outside pixels stay NaN (true black border).
        grid_map_x, grid_map_y = np.meshgrid(np.linspace(-1, 1, frame_w),
                                             np.linspace(-1, 1, frame_h))
        grid_map = griddata(point, value, (grid_map_x, grid_map_y))

    # Sample the raw frame; black borders outside the interpolation hull.
    warped_frame = cv2.remap(frame,
                             grid_map[..., 0].astype(np.float32),
                             grid_map[..., 1].astype(np.float32),
                             cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))
    return warped_frame


def draw_crop_frame(img, fctrH, fctrV, frame_w, frame_h,
                    color=(0, 255, 0), thickness=1):
    """Overlay the retained crop rect (1 - cropFactor) to check black borders.

    cropFactor is the fraction cropped away (fctr/2 per side); the
    retained rect must stay fully covered after the attitude correction --
    black borders invading it mean the correction is off.
    """
    x0 = int(round(frame_w * fctrH / 2))   # crop fctr/2 per side
    y0 = int(round(frame_h * fctrV / 2))
    cv2.rectangle(img, (x0, y0), (frame_w - x0, frame_h - y0), color, thickness)


def main():
    # Parse CLI args; all paths default to the current test case.
    ap = argparse.ArgumentParser(description='Render stabilized video.')
    case = '/mnt/c/A_TangJian/rotation_20250711_100523'
    ap.add_argument('--video', default=case + '/VID_20250711_100523.mp4')
    ap.add_argument('--grid', default=case + '/Dump/DbgFile_WarpGrids_RSCFRMMid.csv')
    ap.add_argument('--quat-raw', default=case + '/Dump/DbgFile_Frame_OriginalCamOrient.csv')
    ap.add_argument('--quat-opt', default=case + '/Dump/DbgFile_Frame_OptimizedCamOrient.csv')
    ap.add_argument('--yaml', default=case + '/conf.yaml')
    ap.add_argument('--out', default=case + '/out', help='output directory')
    ap.add_argument('--draw-grid', action='store_false',
                    help='overlay the center crop rect (black-border check)')
    ap.add_argument('--no-frame-number', action='store_true',
                    help='do not overlay the frame number (top-left)')
    ap.add_argument('--max-frames', type=int, default=0, help='limit frames (debug)')
    args = ap.parse_args()

    # Read camera intrinsics and grid layout from the case config.
    conf = parse_conf(args.yaml)
    f = conf['calFocalX']
    W = int(conf['frameWidth'])
    H = int(conf['frameHeight'])

    # Center-origin principal point when the config is real-K; the legacy product
    # config has none and keeps the virtual camera (principal point at zero).
    cx0, cy0 = (W - 1) / 2, (H - 1) / 2
    if 'calFocalY' in conf and 'calPrincipalX' in conf:
        K = np.array([[f, 0.0, conf['calPrincipalX'] - cx0],
                      [0.0, conf['calFocalY'], conf['calPrincipalY'] - cy0],
                      [0.0, 0.0, 1.0]])
    else:
        K = np.array([[f, 0.0, 0.0], [0.0, f, 0.0], [0.0, 0.0, 1.0]])
    Kinv = np.linalg.inv(K)

    # Load per-frame data: raw quats, optimized quats, distortion grids.
    quats_raw = load_quats(args.quat_raw, skip_first=True)
    quats_opt = load_quats(args.quat_opt, skip_first=False)
    grids = load_grids(args.grid, int(conf['gridRows']), int(conf['gridCols']))
    n = min(len(quats_raw), len(quats_opt), len(grids))   # common frame count

    # Open the input video and the output writer (same size as the video).
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    vn = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('Camera: f=%.3f content=%dx%d frame=%dx%d | frames: raw=%d opt=%d grid=%d video=%d -> %d'
          % (f, W, H, frame_w, frame_h, len(quats_raw), len(quats_opt), len(grids), vn, n))

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, 'final_stabilized_render2.mp4')
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h))

    # Render loop: attitude-correct every frame and write it out.
    limit = args.max_frames or n
    t0 = time.time()
    for i in range(limit):
        ok, frame = cap.read()
        if not ok:
            break

        # Attitude warp: Ht = K * R(q_opt * q_raw^-1) * K^-1, with the real K when
        # the config carries it (its center-origin principal point matches the
        # RCS grid); otherwise the legacy virtual camera diag(f, f, 1).
        dq = qnormalize(qmul(quats_opt[i], qconj(quats_raw[i])))
        Ht = K @ np.array(q_to_R(dq), dtype=np.float64) @ Kinv

        # Warp the raw frame; optionally overlay the crop rect for checking.
        warped_frame = get_warped_frame(frame, Ht, grids[i], W, H, frame_w, frame_h)

        if args.draw_grid:
            draw_crop_frame(warped_frame, conf['cropFactorH'],
                            conf['cropFactorV'], frame_w, frame_h)
        # Frame number overlay, top-left, 0-based (black outline + white text).
        if not args.no_frame_number:
            label = str(i)
            cv2.putText(warped_frame, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(warped_frame, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(warped_frame)

        # Progress bar: [####------]  42% 700/1661 | 12.3s | ETA 17.1s
        pct = (i + 1) / limit
        filled = int(40 * pct)
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (limit - i - 1)
        sys.stdout.write('\r[%s%s] %3d%% %d/%d | %.1fs | ETA %.1fs'
                         % ('#' * filled, '-' * (40 - filled), pct * 100, i + 1, limit, elapsed, eta))
        sys.stdout.flush()

    sys.stdout.write('\n')
    sys.stdout.flush()

    cap.release()
    writer.release()
    print('Saved: ' + out_path)


if __name__ == '__main__':
    main()
