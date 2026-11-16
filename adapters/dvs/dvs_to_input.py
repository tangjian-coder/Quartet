"""DVS -> C++ pipeline inputs (quat csv + RCS grid + conf.yaml).

The RCS grid follows the shared definition: a regular source grid pushed through the
forward chain (de-RS for the rolling-shutter readout, no lens model) and normalized to
centre-origin [-1, 1]. DVS readout is ~16 ms, so the rows differ per frame and each
clip keeps its own rcs.csv next to its quat.csv.

Usage: python3 adapters/dvs/dvs_to_input.py --seq s14_indoor_rotation_... [--out-dir DIR]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import dvs_common as dc  # noqa: E402
from common import warp_grid, write_conf  # noqa: E402

GRID_H, GRID_V = 55, 55  # must match kGridSide in include/GridLayout.h


def grid_row(calib, quat_at, t0, width, height, ois_fn=None):
    readout = calib['readout']
    cx0, cy0 = (width - 1) / 2.0, (height - 1) / 2.0
    q_mid = quat_at(t0 + readout / 2)
    grid = warp_grid(calib, quat_at, t0, q_mid, width, height, GRID_H, GRID_V,
                     use_rs=True, ldc_type=calib['ldc_type'], ois_fn=ois_fn)
    return ','.join(f'{(grid[j, i, 0] - cx0) / cx0:.6f},'
                    f'{(grid[j, i, 1] - cy0) / cy0:.6f}'
                    for j in range(GRID_V) for i in range(GRID_H))


def write_rcs_csv(path, calib, quat_at, t_frames, width, height, ois_fn=None):
    with open(path, 'w') as f:
        for t0 in t_frames:
            f.write(grid_row(calib, quat_at, t0, width, height, ois_fn) + '\n')


def write_quat_csv(path, t_frames, quat_at):
    with open(path, 'w') as f:
        f.write('time,w,x,y,z\n')
        for t in t_frames:
            w, x, y, z = quat_at(t)
            # ns integers keep the column identical in shape to GoPro and ADVIO
            f.write(f'{int(round(t * 1e9))},{w:.9f},{x:.9f},{y:.9f},{z:.9f}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True)
    ap.add_argument('--crop', type=float, default=0.2)
    ap.add_argument('--ois-scale', type=float, default=1.0,
                    help='scale on the logged OIS shift (1.0 = official active-array pixels)')
    ap.add_argument('--no-ois', action='store_true',
                    help='drop the OIS term (keep for A/B checks; the shift is real data)')
    ap.add_argument('--out-dir', default=None)
    a = ap.parse_args()

    out_dir = a.out_dir or os.path.join(dc.DVS_DIR, 'output', 'pipeline', a.seq)
    os.makedirs(out_dir, exist_ok=True)

    # poses stay pure-gyro; OIS is a measured shift, applied by the warp core as a translation
    t_frames, quat_at, info = dc.make_quat_at(a.seq, a.ois_scale, 0.0, use_ois=False)
    ois_fn = None if a.no_ois else dc.ois_at(a.seq, a.ois_scale)
    calib = dc.calib_dict(info)

    write_quat_csv(os.path.join(out_dir, 'quat.csv'), t_frames, quat_at)
    write_rcs_csv(os.path.join(out_dir, 'rcs.csv'), calib, quat_at, t_frames,
                  info['W'], info['H'], ois_fn)
    write_conf(os.path.join(out_dir, 'conf.yaml'), calib['fx'], calib['fy'],
               ((info['W'] - 1) / 2.0), ((info['H'] - 1) / 2.0),
               info['W'], info['H'], a.crop, a.crop, GRID_H, GRID_V)

    print(f'{a.seq}: {len(t_frames)} frames {info["W"]}x{info["H"]} {info["fps"]:.1f}fps'
          f' f={calib["fx"]:.1f} readout={info["readout"]*1e3:.2f}ms'
          f' ois={"off" if a.no_ois else f"x{a.ois_scale:g}"}')
    print(f'  src: {dc.find_clip(a.seq)}')
    print(f'  out: {out_dir}')


if __name__ == '__main__':
    main()
