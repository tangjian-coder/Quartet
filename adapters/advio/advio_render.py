"""ADVIO renderer: warp the iPhone video through the shared inverse warper (warp_pixels).

Two modes share this entry point and one must be chosen explicitly:
  * --fix0          - q_out = the f0 pose: locks the view to the first rendered frame, which
                      makes pose / calibration / direction errors show up as waviness.
  * --opt-quat file - q_out = optimized poses: the final stabilized render.

Nothing here is ADVIO-specific except data loading; the warp core is shared with GoPro.

Usage: python3 adapters/advio/advio_render.py --seq advio-01 --pose arkit --readout 0 \
           (--fix0 | --opt-quat opt.csv) \
           [--f0 3343] [--f1 3703]
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import advio_common as ac  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gopro_imu'))
from common import warp_pixels  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='advio-01')
    ap.add_argument('--pose', choices=['arkit', 'gt', 'imu'], default='arkit')
    ap.add_argument('--readout', type=float, default=0.0)
    ap.add_argument('--scan-axis', choices=['row', 'col'], default='col')
    ap.add_argument('--no-dist', action='store_true', help='skip the radtan model (debug)')
    ap.add_argument('--f0', type=int, default=0)
    ap.add_argument('--f1', type=int, default=None,
                    help='last frame, inclusive (default: last frame of the video)')
    ap.add_argument('--out', default=None)
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

    t_frames, quat_at, n = ac.make_quat_at(a.seq, a.pose)
    W, H, n_video, fps = ac.video_info(a.seq)
    f0 = a.f0
    f1 = n_video - 1 if a.f1 is None else min(a.f1, n_video - 1)
    if not 0 <= f0 <= f1:
        ap.error(f'empty frame range: f0={f0} f1={f1} (video has {n_video} frames)')
    calib = ac.calib_for(a.seq, a.readout)
    print(f'{a.seq}: video {W}x{H} {fps:.2f}fps {n_video} frames | pose={a.pose}'
          f' readout={a.readout*1e3:.2f}ms f={calib["fx"]:.1f}px')

    mode = 'stab' if a.opt_quat else 'fix0'
    out = a.out or os.path.join(ac.ADVIO_DIR, 'output', 'fix0',
                                f'{a.seq}_{mode}_{a.pose}'
                                f'_rs{int(round(a.readout*1000))}_f{f0}-{f1}.mp4')
    os.makedirs(os.path.dirname(out), exist_ok=True)

    q_ref0 = quat_at(t_frames[f0] + a.readout / 2)   # first rendered frame = fixed view
    opt = None
    opt_abs = False
    short_opt = False
    if a.opt_quat:
        # full-sequence csv indexes by absolute frame; a csv built for this clip indexes from f0
        opt_all = np.loadtxt(a.opt_quat, delimiter=',', ndmin=2)[:, :4]
        opt_abs = len(opt_all) == n_video
        opt = opt_all
        print(f'optimized poses: {len(opt_all)} rows ({"absolute" if opt_abs else "clip-relative"} indexing)')

    cap = cv2.VideoCapture(ac.frames_mov(a.seq))
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
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
        # official ADVIO intrinsics + radtan; portrait capture means the sensor reads columns
        # readout=0 means every column shares one exposure, so de-RS has no fixed point to
        # solve: asking for it only buys a redundant second iteration (measured 31 ms/frame)
        use_rs = calib['readout'] != 0.0
        mx, my = warp_pixels(calib, quat_at, t_frames[i], q_out, W, H,
                             use_rs=use_rs, row_ref='pixel_row', scan_axis=a.scan_axis,
                             ldc_type=None if a.no_dist else calib['ldc_type'])
        mx = np.where(np.isnan(mx), -1.0, mx).astype(np.float32)
        my = np.where(np.isnan(my), -1.0, my).astype(np.float32)
        out8 = cv2.remap(frame, mx, my, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        # green center-crop rect (20% crop, same as the GoPro renderer and the optimizer SDF)
        x0, y0, x1, y1 = int(W * 0.1), int(H * 0.1), int(W * 0.9), int(H * 0.9)
        cv2.rectangle(out8, (x0, y0), (x1, y1), (0, 255, 0), 1)
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
