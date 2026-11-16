"""DVS renderer: warp the clip through the shared inverse warper (warp_pixels).

Two modes share this entry point and one must be chosen explicitly:
  * --fix0          - q_out = the f0 pose: locks the view to the first rendered frame, which
                      makes pose / calibration / direction errors show up as waviness.
  * --opt-quat file - q_out = optimized poses: the final stabilized render.

Only data loading is DVS-specific; the warp core is shared with GoPro and ADVIO.

Usage: python3 adapters/dvs/dvs_render.py --seq s14_rotation_static \
           (--fix0 | --opt-quat opt.csv) [--f0 0] [--f1 1099]
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import dvs_common as dc  # noqa: E402
from common import warp_pixels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', required=True)
    ap.add_argument('--ois-scale', type=float, default=1.0,
                    help='scale on the logged OIS shift (1.0 = official active-array pixels)')
    ap.add_argument('--no-ois', action='store_true',
                    help='drop the OIS term (keep for A/B checks; the shift is real data)')
    ap.add_argument('--f0', type=int, default=0)
    ap.add_argument('--f1', type=int, default=None,
                    help='last frame, inclusive (default: last frame of the video)')
    ap.add_argument('--scan-axis', choices=['row', 'col'], default='row')
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-label', action='store_true',
                    help='skip the burnt-in frame counter (metrics lock onto it otherwise)')
    ap.add_argument('--fix0', action='store_true',
                    help='lock the view to the f0 pose (verification: a correct chain holds still)')
    ap.add_argument('--opt-quat', default=None,
                    help='optimized w2c poses (w,x,y,z per row); renders the stabilized view')
    a = ap.parse_args()
    # fix0 is a verification mode, not a default: the two modes answer different questions
    if a.fix0 and a.opt_quat:
        ap.error('--fix0 and --opt-quat are mutually exclusive')
    if not a.fix0 and not a.opt_quat:
        ap.error('pick a mode: --fix0 (verify the pose chain) or --opt-quat <csv> (stabilized render)')

    # poses stay pure-gyro; OIS is a measured shift, applied by the warp core as a translation
    t_frames, quat_at, info = dc.make_quat_at(a.seq, a.ois_scale, 0.0, use_ois=False)
    ois_fn = None if a.no_ois else dc.ois_at(a.seq, a.ois_scale)
    calib = dc.calib_dict(info)
    W, H, n_video, fps = info['W'], info['H'], info['n'], info['fps']
    if len(t_frames) != n_video:
        # the optimizer indexes poses and frames 1:1, so a mismatch would shift the video
        print(f'[WARN] {a.seq}: {len(t_frames)} pose rows vs {n_video} video frames')
    f0 = a.f0
    f1 = n_video - 1 if a.f1 is None else min(a.f1, n_video - 1)
    if not 0 <= f0 <= f1:
        ap.error(f'empty frame range: f0={f0} f1={f1} (video has {n_video} frames)')
    use_rs = info['readout'] != 0.0
    print(f'{a.seq}: video {W}x{H} {fps:.2f}fps {n_video} frames | '
          f'f={calib["fx"]:.1f} readout={info["readout"]*1e3:.2f}ms '
          f'ois={"off" if a.no_ois else f"x{a.ois_scale:g}"}')

    mode = 'stab' if a.opt_quat else 'fix0'
    ois_tag = '' if a.no_ois else f'_ois{int(round(a.ois_scale * 100))}'
    out = a.out or os.path.join(dc.DVS_DIR, 'output', mode,
                                f'{a.seq}_{mode}_rs{int(round(info["readout"] * 1000))}'
                                f'{ois_tag}_f{f0}-{f1}.mp4')
    os.makedirs(os.path.dirname(out), exist_ok=True)

    q_ref0 = quat_at(t_frames[f0] + info['readout'] / 2)   # first rendered frame = fixed view
    opt = None
    opt_abs = False
    short_opt = False
    if a.opt_quat:
        # full-sequence csv indexes by absolute frame; a csv built for this clip indexes from f0
        opt_all = np.loadtxt(a.opt_quat, delimiter=',', ndmin=2)[:, :4]
        opt_abs = len(opt_all) == n_video
        opt = opt_all
        print(f'optimized poses: {len(opt_all)} rows '
              f'({"absolute" if opt_abs else "clip-relative"} indexing)')

    cap = cv2.VideoCapture(dc.video_path({'dir': dc.find_clip(a.seq)})[0])
    # POS_FRAMES seek lands on the wrong frame for these open-GOP clips, so decode forward
    for _ in range(f0):
        if not cap.grab():
            break
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

    for i in range(f0, f1 + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if opt is not None:
            k = i if opt_abs else i - f0
            if k >= len(opt):
                # a short csv silently freezes the view, which looks like a failed render
                if not short_opt:
                    print(f'  [WARN] opt-quat has {len(opt)} rows < frame {k}; holding last pose')
                    short_opt = True
                k = len(opt) - 1
            q_out = opt[k]
            q_out = q_out / np.linalg.norm(q_out)
        else:
            q_out = q_ref0
        mx, my = warp_pixels(calib, quat_at, t_frames[i], q_out, W, H,
                             use_rs=use_rs, row_ref='pixel_row', scan_axis=a.scan_axis,
                             ldc_type=calib['ldc_type'], ois_fn=ois_fn)
        mx = np.where(np.isnan(mx), -1.0, mx).astype(np.float32)
        my = np.where(np.isnan(my), -1.0, my).astype(np.float32)
        out8 = cv2.remap(frame, mx, my, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        # green center-crop rect (20% crop, same as the GoPro and ADVIO renderers)
        x0, y0, x1, y1 = int(W * 0.1), int(H * 0.1), int(W * 0.9), int(H * 0.9)
        cv2.rectangle(out8, (x0, y0), (x1, y1), (0, 255, 0), 1)
        if not a.no_label:
            label = f'{mode} {i}'
            cv2.putText(out8, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(out8, label, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        vw.write(out8)
        if (i - f0) % 100 == 0:
            print(f'  {i}/{f1}', flush=True)
    cap.release()
    vw.release()
    print('saved:', out)


if __name__ == '__main__':
    main()
