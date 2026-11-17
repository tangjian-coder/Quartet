#!/usr/bin/env python3
"""Save a side-by-side original/stabilized still of the most dynamic moment.

A still keeps full detail at a fraction of an animation's size, so it is the honest way to
show crop bounds and per-frame alignment in a README. The frame is picked by the largest
frame-to-frame shift of the original clip (phase correlation), which is where stabilisation
differs most; --at-sec overrides that choice.
"""
import argparse
import os
import sys

import cv2
import numpy as np


def pick_frame(ori, step, n_limit):
    """Return the frame index whose predecessor-to-frame shift is largest.

    n_limit keeps the search inside the frames both clips share: a stabilized render may
    cover only part of the original clip.
    """
    cap = cv2.VideoCapture(ori)
    n = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), n_limit)
    best_i, best_mag, prev = 0, -1.0, None
    for i in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(f, (480, 270)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            (dx, dy), resp = cv2.phaseCorrelate(prev, g)
            mag = np.hypot(dx, dy) * (resp ** 2)
            if mag > best_mag:
                best_mag, best_i = mag, i
        prev = g
    cap.release()
    return best_i


def read_frame(path, idx, width, height):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f'cannot read frame {idx} from {path}')
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(description='Side-by-side original vs stabilized still.')
    ap.add_argument('--ori', required=True)
    ap.add_argument('--stab', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--at-sec', type=float, default=None,
                    help='frame time to grab (default: the largest frame-to-frame shift)')
    ap.add_argument('--panel-width', type=int, default=1280, help='width of each panel')
    ap.add_argument('--crop-rect', type=float, default=0.1,
                    help='redraw the green crop rect at this inset fraction (0 = off)')
    ap.add_argument('--rect-width', type=int, default=2)
    ap.add_argument('--quality', type=int, default=92, help='JPEG quality')
    ap.add_argument('--labels', default='Original,Stabilized')
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.ori)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    cap_s = cv2.VideoCapture(a.stab)
    n_s = int(cap_s.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_s.release()
    n = min(n, n_s)
    if a.at_sec is not None:
        idx = int(round(a.at_sec * fps))
    else:
        idx = pick_frame(a.ori, max(1, n // 120), n)
    idx = max(0, min(idx, n - 1))

    # Keep the source aspect ratio: ADVIO is portrait 720x1280 and a fixed 16:9 squashes it
    cap_ar = cv2.VideoCapture(a.ori)
    sw = cap_ar.get(cv2.CAP_PROP_FRAME_WIDTH) or 16
    sh = cap_ar.get(cv2.CAP_PROP_FRAME_HEIGHT) or 9
    cap_ar.release()
    w = a.panel_width
    h = int(round(w * sh / sw))
    panels = [read_frame(a.ori, idx, w, h), read_frame(a.stab, idx, w, h)]
    labels = a.labels.split(',')
    for img, text in zip(panels, labels):
        if a.crop_rect > 0:
            # Burned-in rect is 1 px of the source frame; redraw it at this scale
            cv2.rectangle(img, (int(w * a.crop_rect), int(h * a.crop_rect)),
                          (int(w * (1 - a.crop_rect)), int(h * (1 - a.crop_rect))),
                          (0, 255, 0), a.rect_width)
        for col, th in (((0, 0, 0), 6), ((0, 255, 255), 2)):
            cv2.putText(img, text, (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.3, col, th, cv2.LINE_AA)
        # Caption on a solid bar: drawing the same text twice (outline + fill) leaves a
        # ghost on narrow glyphs
        tag = f'frame {idx}  t={idx / fps:.2f}s'
        cv2.rectangle(img, (0, h - 34), (330, h), (0, 0, 0), -1)
        cv2.putText(img, tag, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1, cv2.LINE_AA)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cv2.imwrite(a.out, np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, a.quality])
    print(f'saved: {a.out} (frame {idx}, {w * 2}x{h}, '
          f'{os.path.getsize(a.out) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
