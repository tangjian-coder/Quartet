"""Residual shake metric for rendered clips (median inter-frame translation).

Translation only: in-plane rotation measured on a warped output is contaminated by
the perspective warp, so it is not a fair comparison between raw and stabilized.

Usage: python3 adapters/dvs/dvs_eval.py --video a.mp4 [--video b.mp4] [--f0 0 --f1 -1]
"""
import argparse
import os
import sys

import numpy as np
import cv2


def frame_residuals(video, f0=0, f1=None):
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f1 = n - 1 if f1 is None or f1 < 0 else min(f1, n - 1)
    # seek lands on the wrong frame for open-GOP clips; decode forward instead
    for _ in range(f0):
        if not cap.grab():
            break
    orb = cv2.ORB_create(1500)
    prev = None
    vals = []
    for i in range(f0, f1 + 1):
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            ka, da = orb.detectAndCompute(prev, None)
            kb, db = orb.detectAndCompute(g, None)
            if da is not None and db is not None:
                m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
                good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
                if len(good) >= 10:
                    p1 = np.float32([ka[x.queryIdx].pt for x in good])
                    p2 = np.float32([kb[x.trainIdx].pt for x in good])
                    A, _ = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC,
                                                       ransacReprojThreshold=1.5)
                    if A is not None:
                        vals.append(float(np.hypot(A[0, 2], A[1, 2])))
        prev = g
    cap.release()
    return np.array(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', action='append', required=True,
                    help='path, or path@f0:f1 when the clip does not start at frame 0')
    ap.add_argument('--f0', type=int, default=0)
    ap.add_argument('--f1', type=int, default=-1)
    a = ap.parse_args()
    base = None
    for spec in a.video:
        v, f0, f1 = spec, a.f0, a.f1
        if '@' in spec:
            v, rng = spec.rsplit('@', 1)
            f0, f1 = (int(x) for x in rng.split(':'))
        r = frame_residuals(v, f0, f1)
        if len(r) == 0:
            print(f'{os.path.basename(v)}: no frame pairs matched')
            continue
        med = float(np.median(r))
        if base is None:
            base = med
            print(f'{os.path.basename(v):52s} {med:6.2f}px  (baseline)')
        else:
            print(f'{os.path.basename(v):52s} {med:6.2f}px  {base/med:4.1f}x  (n={len(r)})')


if __name__ == '__main__':
    main()
