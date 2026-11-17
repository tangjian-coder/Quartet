#!/usr/bin/env python3
"""Build a side-by-side original/stabilized GIF for the README.

Both inputs are read by frame index, so the stabilized clip must cover the same frames as
the original one. Frames are subsampled to the requested rate and downscaled per panel:
a full-resolution 10 s clip is tens of MB as a GIF, which is unusable in a README.
"""
import argparse
import os
import sys

import cv2
import numpy as np

from PIL import Image


def _open(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f'cannot open video: {path}')
    return cap


def _label(img, text, color=(0, 255, 255)):
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4,
                cv2.LINE_AA)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description='Side-by-side original vs stabilized GIF.')
    ap.add_argument('--ori', required=True, help='original video')
    ap.add_argument('--stab', required=True, help='stabilized video (same frame count)')
    ap.add_argument('--out', required=True, help='output .gif')
    ap.add_argument('--dur-sec', type=float, default=10.0, help='clip length in seconds')
    ap.add_argument('--start-sec', type=float, default=None,
                    help='clip start in seconds (default: centred in the clip)')
    ap.add_argument('--fps', type=float, default=10.0, help='output frame rate')
    ap.add_argument('--width', type=int, default=320, help='width of each panel in pixels')
    ap.add_argument('--colors', type=int, default=64,
                    help='GIF palette size; fewer colours shrink noisy footage a lot')
    ap.add_argument('--no-blur', action='store_true',
                    help='skip the light denoise that keeps sensor noise out of the GIF')
    ap.add_argument('--labels', default='Original,Stabilized',
                    help='panel captions, comma separated')
    ap.add_argument('--crop-rect', type=float, default=0.1,
                    help='redraw the green crop rect at this inset fraction (0 = off); the '
                         'burned-in 1 px rect vanishes once the frame is downscaled')
    ap.add_argument('--rect-width', type=int, default=2, help='crop rect line width in pixels')
    a = ap.parse_args()

    cap_o, cap_s = _open(a.ori), _open(a.stab)
    src_fps = cap_o.get(cv2.CAP_PROP_FPS) or 30.0
    n_o = int(cap_o.get(cv2.CAP_PROP_FRAME_COUNT))
    n_s = int(cap_s.get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n_o, n_s)
    if n_o != n_s:
        print(f'[WARN] frame counts differ (ori {n_o} / stab {n_s}); using {n}')
    total_s = n / src_fps
    dur = min(a.dur_sec, total_s)
    start = a.start_sec if a.start_sec is not None else max(0.0, (total_s - dur) / 2.0)
    f0 = int(round(start * src_fps))
    step = max(1, int(round(src_fps / a.fps)))
    print(f'{os.path.basename(a.ori)}: {n} frames @ {src_fps:.2f} fps '
          f'| clip {start:.1f}-{start + dur:.1f}s, {a.fps:g} fps, step {step}')

    # Keep the source aspect ratio: ADVIO is portrait 720x1280 and a fixed 16:9 squashes it
    cap_ar = cv2.VideoCapture(a.ori)
    sw = cap_ar.get(cv2.CAP_PROP_FRAME_WIDTH) or 16
    sh = cap_ar.get(cv2.CAP_PROP_FRAME_HEIGHT) or 9
    cap_ar.release()
    w = a.width
    h = int(round(w * sh / sw))
    labels = a.labels.split(',')
    frames = []
    for i in range(f0, min(f0 + int(round(dur * src_fps)), n), step):
        cap_o.set(cv2.CAP_PROP_POS_FRAMES, i)
        cap_s.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok_o, fo = cap_o.read()
        ok_s, fs = cap_s.read()
        if not (ok_o and ok_s):
            break
        fo = cv2.resize(fo, (w, h), interpolation=cv2.INTER_AREA)
        fs = cv2.resize(fs, (w, h), interpolation=cv2.INTER_AREA)
        if not a.no_blur:
            fo = cv2.GaussianBlur(fo, (3, 3), 0.7)
            fs = cv2.GaussianBlur(fs, (3, 3), 0.7)
        if a.crop_rect > 0:
            # Stabilized panel only: the original frame has no reason to mark a crop
            cv2.rectangle(fs, (int(w * a.crop_rect), int(h * a.crop_rect)),
                          (int(w * (1 - a.crop_rect)), int(h * (1 - a.crop_rect))),
                          (0, 255, 0), a.rect_width)
        _label(fo, labels[0])
        _label(fs, labels[1])
        img = Image.fromarray(np.hstack([fo, fs])[:, :, ::-1])      # BGR -> RGB
        frames.append(img.convert('P', palette=Image.ADAPTIVE, colors=a.colors))
    cap_o.release()
    cap_s.release()
    if not frames:
        sys.exit('no frames read')

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out_fps = max(1.0, src_fps / step)
    frames[0].save(a.out, save_all=True, append_images=frames[1:],
                   duration=int(round(1000.0 / out_fps)), loop=0, optimize=True,
                   disposal=1)
    print(f'saved: {a.out} ({len(frames)} frames, {os.path.getsize(a.out) / 1048576:.1f} MB)')


if __name__ == '__main__':
    main()
