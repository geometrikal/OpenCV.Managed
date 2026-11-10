#!/usr/bin/env python3
"""Golden-data generator for the managed histogram and statistics port.

Runs cv2 4.5.5 calcHist, calcBackProject, compareHist, equalizeHist, CLAHE,
integral and normalize over a seeded battery and writes
``TestData/histogram_golden.json``. ``ImgProc/HistogramGoldenTests.cs``
replays each case.

Regenerate with:
    python tools/generate_histogram_golden.py
(run from the ``Geometrikal.OpenCV.Managed.Tests`` directory).

Environment and battery constraints:
 * OPENCV_CPU_DISABLE removes the AVX/AVX2/FMA dispatch; IPP, OpenCL and
   threads are disabled. The battery also avoids the cv2 4.5.5 IPP paths
   when IPP is on: accumulate cases pass a mask, and 32F->32F sum-only
   integrals come from integral2 (sqsum requested).
 * Integral 8U->32F cases keep row sums below 2^24 (OpenCV 5.0 uses a
   CPU-dependent vector order above that; the port keeps the scalar order).
 * cv2 4.5.5 cannot produce non-uniform histograms (the Python overload
   fixes uniform=true) or the OpenCV 5.0 CLAHE bit shift. Those cases come
   from scalar models of the OpenCV 5.0 source in this script and have
   "model": true.
 * compareHist CHISQR / CHISQR_ALT: OpenCV 5.0 vectorizes the sum with the
   term (a*a)*(1/b). Cases whose cv2 4.5.5 value differs from the 5.0 model
   are suspects; "values5" holds the 5.0 model value, pinned in
   HistogramOpenCv5DifferenceTests.
 * cv2 Python converts a 3-D ndarray to a 2-D multi-channel Mat. compareHist
   of 3-D histograms therefore uses the flat 1 x n view, and 3-D back
   projections come from the uniform model (cv2 would use channel 0 only).
 * ROI cases store the parent array and apply "roi" ([x, y, w, h]) to the
   inputs listed in "roiInputs".

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import json
import math
import os

os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX,AVX2,AVX512F,FMA3")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

f32 = np.float32
DTYPES = {"u8": np.uint8, "u16": np.uint16, "i16": np.int16, "i32": np.int32, "f32": np.float32, "f64": np.float64}
CVDEPTH = {"u8": cv2.CV_8U, "u16": cv2.CV_16U, "i16": cv2.CV_16S, "i32": cv2.CV_32S, "f32": cv2.CV_32F, "f64": cv2.CV_64F}
DBL_EPS = float(np.finfo(np.float64).eps)


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def dt_of(arr):
    for k, v in DTYPES.items():
        if arr.dtype == v:
            return k
    raise ValueError(arr.dtype)


def jnum(v):
    v = float(v)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    return v


class Gen:
    def __init__(self):
        self.cases = []
        self.names = set()
        self.arrays = []
        self.index = {}

    def ref(self, arr):
        arr = np.ascontiguousarray(arr)
        m = {"dtype": dt_of(arr), "width": int(arr.shape[1]) if arr.ndim >= 2 else int(arr.shape[0]),
             "height": int(arr.shape[0]) if arr.ndim >= 2 else 1,
             "channels": 1 if arr.ndim <= 2 else int(arr.shape[2]), "data": b64(arr)}
        key = (m["dtype"], m["width"], m["height"], m["channels"], m["data"])
        if key not in self.index:
            self.index[key] = len(self.arrays)
            self.arrays.append(m)
        return self.index[key]

    def add(self, name, op, inputs, params, expected=(), values=None, model=False, suspect=None, values5=None):
        assert name not in self.names, name
        self.names.add(name)
        case = {"name": name, "op": op, "inputs": [self.ref(a) for a in inputs], "params": params,
                "expected": [self.ref(e) for e in expected], "model": model, "suspect": suspect is not None}
        if suspect is not None:
            case["suspectReason"] = suspect
        if values is not None:
            case["values"] = [jnum(v) for v in values]
        if values5 is not None:
            case["values5"] = [jnum(v) for v in values5]
        self.cases.append(case)


def flat_ranges(ranges):
    return [float(v) for r in ranges for v in r]


# ---------------------------------------------------------------- models (OpenCV 5.0 generic C++ path)

def cv_ceil(x):
    return int(math.ceil(x))


def tab8u_nonuniform(sizes, ranges):
    tab = []
    for i, sz in enumerate(sizes):
        R = [float(f32(r)) for r in ranges[i]]
        t = [None] * 256
        limit = min(cv_ceil(R[0]), 256)
        idx, written, j = -1, None, 0
        while True:
            while j < limit:
                t[j] = written
                j += 1
            idx += 1
            if idx < sz:
                limit = min(cv_ceil(R[idx + 1]), 256)
                written = idx
            else:
                while j < 256:
                    t[j] = None
                    j += 1
                break
        tab.append(t)
    return tab


def bin_nonuniform(v, R, sz):
    v = f32(v)
    idx = -1
    while v >= f32(R[idx + 1]):
        idx += 1
        if not idx < sz:
            break
    return idx if 0 <= idx < sz else None


def planes_of(images, channels):
    planes = []
    for im in images:
        im = im if im.ndim == 3 else im[:, :, None]
        for c in range(im.shape[2]):
            planes.append(im[:, :, c])
    return [planes[c] for c in channels]


def model_hist_nonuniform(images, channels, mask, sizes, ranges):
    sel = planes_of(images, channels)
    H = np.zeros(sizes, np.int64)
    tab = tab8u_nonuniform(sizes, ranges) if sel[0].dtype == np.uint8 else None
    for y in range(sel[0].shape[0]):
        for x in range(sel[0].shape[1]):
            if mask is not None and not mask[y, x]:
                continue
            idx = []
            for i in range(len(sizes)):
                v = sel[i][y, x]
                k = tab[i][int(v)] if tab is not None else bin_nonuniform(v, ranges[i], sizes[i])
                if k is None:
                    break
                idx.append(k)
            else:
                H[tuple(idx)] += 1
    return H.astype(np.float32)


def sat(x, hi):
    return min(max(int(np.rint(x)), 0), hi)


def model_backproject_nonuniform(images, channels, hist, ranges, scale):
    sel = planes_of(images, channels)
    dt = sel[0].dtype
    sizes = list(hist.shape)
    tab = tab8u_nonuniform(sizes, ranges) if dt == np.uint8 else None
    out = np.zeros(sel[0].shape, dt)
    scale = f32(scale)
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            idx = []
            for i in range(len(sizes)):
                v = sel[i][y, x]
                k = tab[i][int(v)] if tab is not None else bin_nonuniform(v, ranges[i], sizes[i])
                if k is None:
                    break
                idx.append(k)
            else:
                p = f32(f32(hist[tuple(idx)]) * scale)
                out[y, x] = sat(p, 255) if dt == np.uint8 else (sat(p, 65535) if dt == np.uint16 else p)
    return out


def tab8u_uniform(sizes, ranges):
    tab = []
    for i, sz in enumerate(sizes):
        lo, hi = float(f32(ranges[i][0])), float(f32(ranges[i][1]))
        t = sz / (hi - lo)
        a, b = t, -t * lo
        tab.append([min(max(int(math.floor(j * a + b)), 0), sz - 1) if lo <= j < hi else None for j in range(256)])
    return tab


def model_backproject_uniform(images, channels, hist, ranges, scale):
    """calcBackProj_8u / calcBackProj_ with uniform bins (no range test for 16U/32F)."""
    sel = planes_of(images, channels)
    dt = sel[0].dtype
    sizes = list(hist.shape)
    tab = tab8u_uniform(sizes, ranges) if dt == np.uint8 else None
    out = np.zeros(sel[0].shape, dt)
    scale = f32(scale)
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            idx = []
            for i in range(len(sizes)):
                v = sel[i][y, x]
                if tab is not None:
                    k = tab[i][int(v)]
                else:
                    lo, hi = float(f32(ranges[i][0])), float(f32(ranges[i][1]))
                    t = sizes[i] / (hi - lo)
                    z = float(v) * t + (-t * lo)
                    k = None if math.isnan(z) or math.isinf(z) else int(math.floor(z))
                    k = k if k is not None and 0 <= k < sizes[i] else None
                if k is None:
                    break
                idx.append(k)
            else:
                p = f32(f32(hist[tuple(idx)]) * scale)
                out[y, x] = sat(p, 255) if dt == np.uint8 else (sat(p, 65535) if dt == np.uint16 else p)
    return out


def model_chisqr_v5(h1, h2, alt):
    a1 = h1.astype(np.float32).ravel()
    a2 = h2.astype(np.float32).ravel()
    n = a1.size
    nv = (n // 4) * 4
    acc = [0.0, 0.0]
    for j0 in range(0, nv, 4):
        for half in (0, 2):
            for lane in range(2):
                x = float(a1[j0 + half + lane])
                y = float(a2[j0 + half + lane])
                a = x - y
                b = x + y if alt else x
                if abs(b) > DBL_EPS:
                    acc[lane] = acc[lane] + (a * a) * (1.0 / b)
    res = acc[0] + acc[1]
    for k in range(nv, n):
        a = float(f32(a1[k]) - f32(a2[k]))
        b = float(f32(a1[k]) + f32(a2[k])) if alt else float(a1[k])
        if abs(b) > DBL_EPS:
            res += a * a / b
    return res * 2 if alt else res


def reflect101(p, n):
    if n == 1:
        return 0
    while p < 0 or p >= n:
        p = -p if p < 0 else 2 * n - 2 - p
    return p


def model_clahe(src, clip_limit, grid, bit_shift):
    tx_n, ty_n = grid
    is8 = src.dtype == np.uint8
    hist_size = (256 if is8 else 65536) >> bit_shift
    H, W = src.shape
    if W % tx_n == 0 and H % ty_n == 0:
        ext = src
    else:
        rows = [reflect101(r, H) for r in range(H + ty_n - H % ty_n)]
        cols = [reflect101(c, W) for c in range(W + tx_n - W % tx_n)]
        ext = src[np.ix_(rows, cols)]
    tw, th = ext.shape[1] // tx_n, ext.shape[0] // ty_n
    area = tw * th
    lut_scale = f32(f32(hist_size - 1) / f32(area))
    clip = max(int(clip_limit * area / hist_size), 1) if clip_limit > 0 else 0
    lut = np.zeros((ty_n * tx_n, hist_size), dtype=src.dtype)
    for k in range(tx_n * ty_n):
        ty, tx = divmod(k, tx_n)
        tile = ext[ty * th:(ty + 1) * th, tx * tw:(tx + 1) * tw].astype(np.int64) >> bit_shift
        hist = np.bincount(tile.ravel(), minlength=hist_size).astype(np.int64)
        if clip_limit > 0:
            over = hist > clip
            clipped = int((hist[over] - clip).sum())
            hist[over] = clip
            batch = clipped // hist_size
            residual = clipped - batch * hist_size
            hist += batch
            if residual:
                step = max(hist_size // residual, 1)
                i = 0
                while i < hist_size and residual > 0:
                    hist[i] += 1
                    i += step
                    residual -= 1
        cs = np.cumsum(hist)
        lut[k] = np.clip(np.rint(cs.astype(np.float32) * lut_scale), 0, 255 if is8 else 65535).astype(src.dtype)
    inv_tw = f32(f32(1.0) / f32(tw))
    inv_th = f32(f32(1.0) / f32(th))
    txf = np.arange(W, dtype=np.float32) * inv_tw - f32(0.5)
    tx1 = np.floor(txf).astype(np.int64)
    xa = (txf - tx1.astype(np.float32)).astype(np.float32)
    xa1 = (f32(1.0) - xa).astype(np.float32)
    tx2 = np.minimum(tx1 + 1, tx_n - 1)
    tx1 = np.maximum(tx1, 0)
    out = np.empty_like(src)
    for y in range(H):
        tyf = f32(f32(y) * inv_th - f32(0.5))
        ty1 = int(math.floor(tyf))
        ya = f32(tyf - f32(ty1))
        ya1 = f32(f32(1.0) - ya)
        ty2 = min(ty1 + 1, ty_n - 1)
        ty1 = max(ty1, 0)
        v = src[y].astype(np.int64) >> bit_shift
        p1 = lut[ty1 * tx_n:(ty1 + 1) * tx_n]
        p2 = lut[ty2 * tx_n:(ty2 + 1) * tx_n]
        res = ((p1[tx1, v].astype(np.float32) * xa1 + p1[tx2, v].astype(np.float32) * xa) * ya1
               + (p2[tx1, v].astype(np.float32) * xa1 + p2[tx2, v].astype(np.float32) * xa) * ya)
        r = np.clip(np.rint(res), 0, 255 if is8 else 65535).astype(np.int64)
        out[y] = ((r << bit_shift) & (255 if is8 else 65535)).astype(src.dtype)
    return out


# ---------------------------------------------------------------- battery

def calc_hist(g, rng):
    a8 = rng.integers(0, 256, (37, 41), dtype=np.uint8)
    m = (rng.random((37, 41)) > 0.3).astype(np.uint8)
    for hs, rg in [(256, [0, 256]), (16, [0, 256]), (10, [3, 200]), (7, [0.5, 255.5]),
                   (30, [-10.25, 300.75]), (5, [10, 11])]:
        for mm in (None, m):
            r = cv2.calcHist([a8], [0], mm, [hs], rg)
            name = f"calcHist_u8_1d_hs{hs}_{rg[0]}_{rg[1]}" + ("_mask" if mm is not None else "")
            ins = [a8] + ([mm] if mm is not None else [])
            g.add(name, "calcHist", ins, {"images": 1, "mask": mm is not None, "channels": [0], "histSize": [hs],
                                           "ranges": flat_ranges([rg]), "uniform": True}, [r.ravel()])
    r = cv2.calcHist([a8], [0], None, [64], [])
    g.add("calcHist_u8_1d_implicit", "calcHist", [a8],
          {"images": 1, "mask": False, "channels": [0], "histSize": [64], "ranges": None, "uniform": True}, [r.ravel()])

    c3 = rng.integers(0, 256, (23, 19, 3), dtype=np.uint8)
    m3 = m[:23, :19].copy()
    r = cv2.calcHist([c3], [0, 2], m3, [12, 9], [0, 256, 20, 240])
    g.add("calcHist_u8_2d_mask", "calcHist", [c3, m3], {"images": 1, "mask": True, "channels": [0, 2], "histSize": [12, 9],
                                                        "ranges": [0, 256, 20, 240], "uniform": True}, [r.ravel()])
    r = cv2.calcHist([c3], [1, 2], None, [16, 16], [])
    g.add("calcHist_u8_2d_implicit", "calcHist", [c3], {"images": 1, "mask": False, "channels": [1, 2], "histSize": [16, 16],
                                                        "ranges": None, "uniform": True}, [r.ravel()])
    a8s = a8[:23, :19].copy()
    r = cv2.calcHist([c3, a8s], [1, 3, 0], None, [4, 5, 6], [0, 256, 0, 256, 50, 250])
    g.add("calcHist_u8_3d_two_images", "calcHist", [c3, a8s],
          {"images": 2, "mask": False, "channels": [1, 3, 0], "histSize": [4, 5, 6], "ranges": [0, 256, 0, 256, 50, 250],
           "uniform": True}, [r.ravel()])
    r = cv2.calcHist([c3], [0, 1, 2], m3, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    g.add("calcHist_u8_3d_hsv_style_mask", "calcHist", [c3, m3],
          {"images": 1, "mask": True, "channels": [0, 1, 2], "histSize": [8, 8, 8], "ranges": [0, 256] * 3, "uniform": True},
          [r.ravel()])
    r = cv2.calcHist([c3], [2], None, [20], [5, 250])
    g.add("calcHist_u8_1d_interleaved_ch2", "calcHist", [c3],
          {"images": 1, "mask": False, "channels": [2], "histSize": [20], "ranges": [5, 250], "uniform": True}, [r.ravel()])

    c4 = rng.integers(0, 256, (17, 13, 4), dtype=np.uint8)
    rg4 = [0, 256, 10, 200, 0.5, 255.5, -5, 300]
    r = cv2.calcHist([c4], [0, 1, 2, 3], None, [3, 4, 2, 5], rg4)
    g.add("calcHist_u8_4d", "calcHist", [c4], {"images": 1, "mask": False, "channels": [0, 1, 2, 3], "histSize": [3, 4, 2, 5],
                                               "ranges": rg4, "uniform": True}, [r.ravel()])
    m4 = (rng.random((17, 13)) > 0.4).astype(np.uint8)
    r = cv2.calcHist([c4], [3, 1, 0, 2], m4, [2, 3, 4, 5], rg4)
    g.add("calcHist_u8_4d_mask", "calcHist", [c4, m4], {"images": 1, "mask": True, "channels": [3, 1, 0, 2],
                                                        "histSize": [2, 3, 4, 5], "ranges": rg4, "uniform": True}, [r.ravel()])

    # ROI (non-continuous) source and mask.
    parent = rng.integers(0, 256, (50, 60, 3), dtype=np.uint8)
    pmask = (rng.random((50, 60)) > 0.5).astype(np.uint8)
    roi = [5, 7, 31, 23]
    sub = parent[7:30, 5:36]
    subm = pmask[7:30, 5:36]
    r = cv2.calcHist([sub], [1], subm, [32], [0, 256])
    g.add("calcHist_u8_1d_roi_mask", "calcHist", [parent, pmask],
          {"images": 1, "mask": True, "channels": [1], "histSize": [32], "ranges": [0, 256], "uniform": True,
           "roi": roi, "roiInputs": [0, 1]}, [r.ravel()])
    r = cv2.calcHist([sub], [0, 2], None, [6, 7], [0, 256, 30, 230])
    g.add("calcHist_u8_2d_roi", "calcHist", [parent],
          {"images": 1, "mask": False, "channels": [0, 2], "histSize": [6, 7], "ranges": [0, 256, 30, 230], "uniform": True,
           "roi": roi, "roiInputs": [0]}, [r.ravel()])

    # 16U.
    a16 = rng.integers(0, 65536, (31, 29), dtype=np.uint16)
    a16.ravel()[:4] = [0, 65535, 100, 60000]
    m16 = m[:31, :29].copy()
    for hs, rg in [(256, [0, 65536]), (100, [100, 60000]), (13, [0.5, 65535.5]), (65536, [0, 65536])]:
        for mm in ((None, m16) if hs != 65536 else (None,)):
            r = cv2.calcHist([a16], [0], mm, [hs], rg)
            name = f"calcHist_u16_1d_hs{hs}_{rg[0]}_{rg[1]}" + ("_mask" if mm is not None else "")
            ins = [a16] + ([mm] if mm is not None else [])
            g.add(name, "calcHist", ins, {"images": 1, "mask": mm is not None, "channels": [0], "histSize": [hs],
                                           "ranges": flat_ranges([rg]), "uniform": True}, [r.ravel()])
    c16 = rng.integers(0, 65536, (19, 21, 2), dtype=np.uint16)
    r = cv2.calcHist([c16], [1, 0], None, [9, 11], [0, 65536, 1000, 50000])
    g.add("calcHist_u16_2d_interleaved", "calcHist", [c16], {"images": 1, "mask": False, "channels": [1, 0],
                                                             "histSize": [9, 11], "ranges": [0, 65536, 1000, 50000],
                                                             "uniform": True}, [r.ravel()])

    # 32F with edge values.
    a32 = (rng.random((33, 35)) * 1.2 - 0.1).astype(np.float32)
    edge = np.array([0, 1, 0.1, 0.2, 0.3, 0.7, 0.9999999, np.nextafter(f32(1), f32(0)), -0.0, 0.5, 0.8,
                     np.nextafter(f32(0.1), f32(0))], np.float32)
    a32.ravel()[:edge.size] = edge
    m32 = m[:33, :35].copy()
    for hs, rg in [(10, [0, 1]), (3, [0, 1]), (7, [0.1, 0.8]), (256, [0, 256]), (20, [0.5, 20.5])]:
        for mm in (None, m32):
            src = a32 if hs != 20 else (a32 * 21).astype(np.float32)
            r = cv2.calcHist([src], [0], mm, [hs], rg)
            name = f"calcHist_f32_1d_hs{hs}_{rg[0]}_{rg[1]}" + ("_mask" if mm is not None else "")
            ins = [src] + ([mm] if mm is not None else [])
            g.add(name, "calcHist", ins, {"images": 1, "mask": mm is not None, "channels": [0], "histSize": [hs],
                                           "ranges": flat_ranges([rg]), "uniform": True}, [r.ravel()])
    n32 = np.array([[np.nan, np.inf, -np.inf, 0.5, 1.0, np.nan, 0.25, -0.0, 0.0, 0.999]], np.float32)
    for mm in (None, np.ones_like(n32, np.uint8)):
        r = cv2.calcHist([n32], [0], mm, [4], [0, 1])
        g.add("calcHist_f32_nan_inf" + ("_mask" if mm is not None else ""), "calcHist", [n32] + ([mm] if mm is not None else []),
              {"images": 1, "mask": mm is not None, "channels": [0], "histSize": [4], "ranges": [0, 1], "uniform": True},
              [r.ravel()])
    w32 = (rng.random((40, 37)) * 10 - 1).astype(np.float32)
    f2 = (rng.random((29, 31, 2)) * 3 - 0.5).astype(np.float32)
    r = cv2.calcHist([f2], [0, 1], None, [5, 8], [0, 2, -0.5, 2.5])
    g.add("calcHist_f32_2d_interleaved", "calcHist", [f2], {"images": 1, "mask": False, "channels": [0, 1], "histSize": [5, 8],
                                                            "ranges": [0, 2, -0.5, 2.5], "uniform": True}, [r.ravel()])
    ws = w32[:29, :31].copy()
    r = cv2.calcHist([f2, ws], [2, 1, 0], None, [4, 3, 5], [0, 8, -0.5, 2.5, 0, 2])
    g.add("calcHist_f32_3d_two_images", "calcHist", [f2, ws],
          {"images": 2, "mask": False, "channels": [2, 1, 0], "histSize": [4, 3, 5], "ranges": [0, 8, -0.5, 2.5, 0, 2],
           "uniform": True}, [r.ravel()])
    r = cv2.calcHist([f2, ws, ws], [0, 1, 2, 3], None, [2, 3, 4, 2], [0, 2, 0, 2.5, -1, 9, 0, 5])
    g.add("calcHist_f32_4d", "calcHist", [f2, ws, ws],
          {"images": 3, "mask": False, "channels": [0, 1, 2, 3], "histSize": [2, 3, 4, 2], "ranges": [0, 2, 0, 2.5, -1, 9, 0, 5],
           "uniform": True}, [r.ravel()])
    pf = (rng.random((40, 50)) * 4).astype(np.float32)
    r = cv2.calcHist([pf[3:30, 4:41]], [0], None, [9], [0.5, 3.5])
    g.add("calcHist_f32_1d_roi", "calcHist", [pf], {"images": 1, "mask": False, "channels": [0], "histSize": [9],
                                                    "ranges": [0.5, 3.5], "uniform": True, "roi": [4, 3, 37, 27],
                                                    "roiInputs": [0]}, [r.ravel()])
    wide = (rng.random((3, 203)) * 2).astype(np.float32)
    r = cv2.calcHist([wide], [0], None, [17], [0, 2])
    g.add("calcHist_f32_1d_wide", "calcHist", [wide], {"images": 1, "mask": False, "channels": [0], "histSize": [17],
                                                       "ranges": [0, 2], "uniform": True}, [r.ravel()])
    w16 = rng.integers(0, 65536, (3, 211), dtype=np.uint16)
    r = cv2.calcHist([w16], [0], None, [37], [10, 65000])
    g.add("calcHist_u16_1d_wide", "calcHist", [w16], {"images": 1, "mask": False, "channels": [0], "histSize": [37],
                                                      "ranges": [10, 65000], "uniform": True}, [r.ravel()])

    # Accumulate with fractional prior values (mask keeps cv2 4.5.5 off the IPP path).
    prior = np.array([0.5, 1.5, 2.5, -0.5, 3.49, 1e9] + [0.0] * 250, np.float32)
    h = prior.reshape(-1, 1).copy()
    cv2.calcHist([a8], [0], m, [256], [0, 256], hist=h, accumulate=True)
    g.add("calcHist_u8_accumulate_mask", "calcHist", [a8, m, prior],
          {"images": 1, "mask": True, "channels": [0], "histSize": [256], "ranges": [0, 256], "uniform": True,
           "accumulate": True}, [h.ravel()])
    prior32 = (rng.random(10) * 5).astype(np.float32)
    h = prior32.reshape(-1, 1).copy()
    cv2.calcHist([a32], [0], m32, [10], [0, 1], hist=h, accumulate=True)
    g.add("calcHist_f32_accumulate_mask", "calcHist", [a32, m32, prior32],
          {"images": 1, "mask": True, "channels": [0], "histSize": [10], "ranges": [0, 1], "uniform": True,
           "accumulate": True}, [h.ravel()])

    # Non-uniform: model-derived.
    nu = [[0, 10.5, 50, 51, 200, 255.5]]
    g.add("calcHist_u8_nonuniform_model", "calcHist", [a8],
          {"images": 1, "mask": False, "channels": [0], "histSize": [5], "ranges": flat_ranges(nu), "uniform": False},
          [model_hist_nonuniform([a8], [0], None, [5], nu).ravel()], model=True)
    nu2 = [[-3, 0.25, 100, 256], [20, 30, 31.5, 128, 129, 300]]
    g.add("calcHist_u8_2d_nonuniform_mask_model", "calcHist", [c3, m3],
          {"images": 1, "mask": True, "channels": [2, 0], "histSize": [3, 5], "ranges": flat_ranges(nu2), "uniform": False},
          [model_hist_nonuniform([c3], [2, 0], m3, [3, 5], nu2).ravel()], model=True)
    nf = [[0, 0.1, 0.15, 0.5, 0.9, 1.0]]
    a32n = a32.copy()
    a32n.ravel()[20:23] = [np.nan, np.inf, -np.inf]
    g.add("calcHist_f32_nonuniform_model", "calcHist", [a32n],
          {"images": 1, "mask": False, "channels": [0], "histSize": [5], "ranges": flat_ranges(nf), "uniform": False},
          [model_hist_nonuniform([a32n], [0], None, [5], nf).ravel()], model=True)
    n16 = [[0, 1000, 1000.5, 30000, 65535]]
    g.add("calcHist_u16_nonuniform_mask_model", "calcHist", [a16, m16],
          {"images": 1, "mask": True, "channels": [0], "histSize": [4], "ranges": flat_ranges(n16), "uniform": False},
          [model_hist_nonuniform([a16], [0], m16, [4], n16).ravel()], model=True)


def back_project(g, rng):
    a8 = rng.integers(0, 256, (27, 31, 2), dtype=np.uint8)
    hist = cv2.calcHist([a8], [0, 1], None, [8, 6], [0, 256, 10, 250])
    for sc in (1.0, 0.37, 255.0 / float(hist.max())):
        r = cv2.calcBackProject([a8], [0, 1], hist, [0, 256, 10, 250], sc)
        g.add(f"backProject_u8_2d_s{sc:.4f}", "calcBackProject", [a8, hist],
              {"images": 1, "channels": [0, 1], "histSize": [8, 6], "ranges": [0, 256, 10, 250], "scale": sc,
               "uniform": True}, [r])
    h1 = cv2.calcHist([a8], [0], None, [16], [0, 256])
    r = cv2.calcBackProject([a8], [0], h1, [0, 256], 3.3)
    g.add("backProject_u8_1d_interleaved", "calcBackProject", [a8, h1],
          {"images": 1, "channels": [0], "histSize": [16], "ranges": [0, 256], "scale": 3.3, "uniform": True}, [r])
    g1 = rng.integers(0, 256, (21, 133), dtype=np.uint8)
    hg = cv2.calcHist([g1], [0], None, [32], [8, 240])
    r = cv2.calcBackProject([g1], [0], hg, [8, 240], 0.5)
    g.add("backProject_u8_1d_wide", "calcBackProject", [g1, hg],
          {"images": 1, "channels": [0], "histSize": [32], "ranges": [8, 240], "scale": 0.5, "uniform": True}, [r])
    c3 = rng.integers(0, 256, (19, 23, 3), dtype=np.uint8)
    # cv2 Python passes a 3-D histogram as a 2-D multi-channel Mat, and calcBackProject then uses
    # only channel 0 of the first two dimensions. 3-D expectations come from the model.
    h3 = cv2.calcHist([c3], [0, 1, 2], None, [4, 5, 6], [0, 256, 20, 240, 0, 256])
    r = model_backproject_uniform([c3], [0, 1, 2], h3, [[0, 256], [20, 240], [0, 256]], 2.0)
    g.add("backProject_u8_3d_model", "calcBackProject", [c3, h3.reshape(-1)],
          {"images": 1, "channels": [0, 1, 2], "histSize": [4, 5, 6], "ranges": [0, 256, 20, 240, 0, 256], "scale": 2.0,
           "uniform": True}, [r], model=True)
    c4 = rng.integers(0, 256, (11, 13, 4), dtype=np.uint8)
    h4 = cv2.calcHist([c4], [0, 1, 2, 3], None, [2, 3, 2, 3], [0, 256, 0, 200, 50, 256, 0, 256])
    r = cv2.calcBackProject([c4], [0, 1, 2, 3], h4, [0, 256, 0, 200, 50, 256, 0, 256], 1.5)
    g.add("backProject_u8_4d", "calcBackProject", [c4, h4.reshape(-1)],
          {"images": 1, "channels": [0, 1, 2, 3], "histSize": [2, 3, 2, 3], "ranges": [0, 256, 0, 200, 50, 256, 0, 256],
           "scale": 1.5, "uniform": True}, [r])

    a32 = (rng.random((20, 22)) * 1.2 - 0.1).astype(np.float32)
    a32[0, :6] = [1.0, np.nextafter(f32(1), f32(0)), 0.0, np.nan, -0.0, np.nextafter(f32(0), f32(-1))]
    h32 = (rng.random((10, 1)) * 5).astype(np.float32)
    r = cv2.calcBackProject([a32], [0], h32, [0, 1], 1.7)
    g.add("backProject_f32_1d", "calcBackProject", [a32, h32],
          {"images": 1, "channels": [0], "histSize": [10], "ranges": [0, 1], "scale": 1.7, "uniform": True}, [r])
    w32 = (rng.random((5, 77)) * 3).astype(np.float32)
    hw = (rng.random((7, 1)) * 2).astype(np.float32)
    r = cv2.calcBackProject([w32], [0], hw, [0.25, 2.75], 1.0)
    g.add("backProject_f32_1d_wide", "calcBackProject", [w32, hw],
          {"images": 1, "channels": [0], "histSize": [7], "ranges": [0.25, 2.75], "scale": 1.0, "uniform": True}, [r])
    f2 = (rng.random((17, 19, 2)) * 3 - 1).astype(np.float32)
    hf2 = (rng.random((5, 7)) * 4).astype(np.float32)
    r = cv2.calcBackProject([f2], [0, 1], hf2, [0, 1, -1, 2], 0.75)
    g.add("backProject_f32_2d", "calcBackProject", [f2, hf2],
          {"images": 1, "channels": [0, 1], "histSize": [5, 7], "ranges": [0, 1, -1, 2], "scale": 0.75, "uniform": True}, [r])
    f1 = (rng.random((17, 19)) * 2).astype(np.float32)
    hf3 = (rng.random((3, 4, 5)) * 4).astype(np.float32)
    r = model_backproject_uniform([f2, f1], [1, 0, 2], hf3, [[-1, 2], [0, 1], [0, 2]], 1.25)
    g.add("backProject_f32_3d_two_images_model", "calcBackProject", [f2, f1, hf3.reshape(-1)],
          {"images": 2, "channels": [1, 0, 2], "histSize": [3, 4, 5], "ranges": [-1, 2, 0, 1, 0, 2], "scale": 1.25,
           "uniform": True}, [r], model=True)
    # The uniform model reproduces cv2 for the 1-D and 2-D cases.
    assert np.array_equal(model_backproject_uniform([f2], [0, 1], hf2, [[0, 1], [-1, 2]], 0.75),
                          cv2.calcBackProject([f2], [0, 1], hf2, [0, 1, -1, 2], 0.75))
    assert np.array_equal(model_backproject_uniform([a8], [0, 1], hist, [[0, 256], [10, 250]], 0.37),
                          cv2.calcBackProject([a8], [0, 1], hist, [0, 256, 10, 250], 0.37))
    assert np.array_equal(model_backproject_uniform([a32], [0], h32.reshape(-1), [[0, 1]], 1.7),
                          cv2.calcBackProject([a32], [0], h32, [0, 1], 1.7))
    a16 = rng.integers(0, 65536, (20, 22), dtype=np.uint16)
    h16 = (rng.random((50, 1)) * 30000).astype(np.float32)
    r = cv2.calcBackProject([a16], [0], h16, [0, 65536], 2.5)
    g.add("backProject_u16_1d_saturate", "calcBackProject", [a16, h16],
          {"images": 1, "channels": [0], "histSize": [50], "ranges": [0, 65536], "scale": 2.5, "uniform": True}, [r])
    c16 = rng.integers(0, 65536, (13, 41, 2), dtype=np.uint16)
    h16b = (rng.random((6, 4)) * 900).astype(np.float32)
    r = cv2.calcBackProject([c16], [1, 0], h16b, [100, 60000, 0, 65536], 3.0)
    g.add("backProject_u16_2d", "calcBackProject", [c16, h16b],
          {"images": 1, "channels": [1, 0], "histSize": [6, 4], "ranges": [100, 60000, 0, 65536], "scale": 3.0,
           "uniform": True}, [r])
    parent = rng.integers(0, 256, (40, 45), dtype=np.uint8)
    hp = cv2.calcHist([parent], [0], None, [12], [0, 256])
    r = cv2.calcBackProject([parent[5:31, 3:40]], [0], hp, [0, 256], 0.1)
    g.add("backProject_u8_1d_roi", "calcBackProject", [parent, hp],
          {"images": 1, "channels": [0], "histSize": [12], "ranges": [0, 256], "scale": 0.1, "uniform": True,
           "roi": [3, 5, 37, 26], "roiInputs": [0]}, [r])
    pf = (rng.random((30, 40)) * 2).astype(np.float32)
    hpf = (rng.random((9, 1)) * 3).astype(np.float32)
    r = cv2.calcBackProject([pf[2:27, 6:39]], [0], hpf, [0, 2], 1.0)
    g.add("backProject_f32_1d_roi", "calcBackProject", [pf, hpf],
          {"images": 1, "channels": [0], "histSize": [9], "ranges": [0, 2], "scale": 1.0, "uniform": True,
           "roi": [6, 2, 33, 25], "roiInputs": [0]}, [r])

    # Non-uniform: model-derived.
    nu = [[0, 10.5, 50, 51, 200, 255.5]]
    hn = (rng.random((5, 1)) * 100).astype(np.float32)
    g1s = g1[:9, :40].copy()
    g.add("backProject_u8_nonuniform_model", "calcBackProject", [g1s, hn],
          {"images": 1, "channels": [0], "histSize": [5], "ranges": flat_ranges(nu), "scale": 1.3, "uniform": False},
          [model_backproject_nonuniform([g1s], [0], hn.reshape(-1), nu, 1.3)], model=True)
    nf = [[0, 0.1, 0.15, 0.5, 0.9, 1.0]]
    g.add("backProject_f32_nonuniform_model", "calcBackProject", [a32, hn],
          {"images": 1, "channels": [0], "histSize": [5], "ranges": flat_ranges(nf), "scale": 0.5, "uniform": False},
          [model_backproject_nonuniform([a32], [0], hn.reshape(-1), nf, 0.5)], model=True)


def compare_hist(g, rng):
    for n in (256, 255, 7, 1000, 259, 3):
        h1 = (rng.random(n) * 100).astype(np.float32)
        h2 = (rng.random(n) * 100).astype(np.float32)
        h1[:min(5, n)] = 0
        h2[min(3, n - 1):min(8, n)] = 0
        for meth in range(6):
            r = cv2.compareHist(h1, h2, meth)
            name = f"compareHist_n{n}_m{meth}"
            susp = None
            v5 = None
            if meth in (1, 4):
                v5 = model_chisqr_v5(h1, h2, meth == 4)
                if v5 != r:
                    susp = "OpenCV 5.0 vectorizes CHISQR/CHISQR_ALT with (a*a)*(1/b); cv2 4.5.5 sums a*a/b in a scalar loop"
            g.add(name, "compareHist", [h1, h2], {"method": meth, "sizes": [n]}, values=[r], suspect=susp,
                  values5=[v5] if susp else None)
    for sizes in ((16, 16), (4, 5, 6)):
        n = int(np.prod(sizes))
        h1 = (rng.random(sizes) * 10).astype(np.float32)
        h2 = (rng.random(sizes) * 10).astype(np.float32)
        h1.ravel()[::7] = 0
        for meth in range(6):
            # A 3-D ndarray reaches cv2 as a 2-D multi-channel Mat (total() = rows * cols); the flat
            # 1 x n view gives the dense N-d histogram result.
            r = cv2.compareHist(h1.reshape(1, n), h2.reshape(1, n), meth)
            name = f"compareHist_{'x'.join(map(str, sizes))}_m{meth}"
            susp = None
            v5 = None
            if meth in (1, 4):
                v5 = model_chisqr_v5(h1, h2, meth == 4)
                if v5 != r:
                    susp = "OpenCV 5.0 vectorizes CHISQR/CHISQR_ALT with (a*a)*(1/b); cv2 4.5.5 sums a*a/b in a scalar loop"
            g.add(name, "compareHist", [h1.reshape(1, n), h2.reshape(1, n)], {"method": meth, "sizes": list(sizes)},
                  values=[r], suspect=susp, values5=[v5] if susp else None)


def equalize(g, rng):
    for shp, lo, hi in [((37, 41), 0, 256), ((64, 64), 40, 90), ((5, 5), 7, 8), ((480, 640), 0, 256),
                        ((3, 300), 200, 256), ((1, 1), 9, 10), ((7, 129), 0, 2)]:
        a = rng.integers(lo, hi, shp, dtype=np.uint8)
        g.add(f"equalizeHist_{shp[0]}x{shp[1]}_{lo}_{hi}", "equalizeHist", [a], {}, [cv2.equalizeHist(a)])
    for v in (0, 255):
        a = np.full((9, 11), v, np.uint8)
        a[3, 4] = 128
        g.add(f"equalizeHist_two_values_{v}", "equalizeHist", [a], {}, [cv2.equalizeHist(a)])
    a = np.zeros((6, 70), np.uint8)
    g.add("equalizeHist_all_zero", "equalizeHist", [a], {}, [cv2.equalizeHist(a)])
    a = np.full((6, 70), 255, np.uint8)
    g.add("equalizeHist_all_255", "equalizeHist", [a], {}, [cv2.equalizeHist(a)])
    parent = rng.integers(0, 256, (50, 80), dtype=np.uint8)
    g.add("equalizeHist_roi", "equalizeHist", [parent], {"roi": [7, 3, 65, 41], "roiInputs": [0]},
          [cv2.equalizeHist(parent[3:44, 7:72])])


def clahe(g, rng):
    cases = [((64, 64), 40.0, (8, 8)), ((67, 53), 2.0, (8, 8)), ((100, 77), 3.0, (5, 3)), ((31, 33), 0.0, (4, 4)),
             ((48, 40), 1.0, (8, 8)), ((17, 9), 4.0, (2, 2)), ((128, 96), 1e6, (8, 8)), ((67, 64), 2.5, (8, 8)),
             ((5, 5), 40.0, (8, 8)), ((1, 37), 3.0, (3, 1)), ((150, 211), 2.0, (8, 8))]
    for shp, cl, gr in cases:
        a = rng.integers(0, 256, shp, dtype=np.uint8)
        a[: shp[0] // 2] //= 3
        r = cv2.createCLAHE(cl, gr).apply(a)
        g.add(f"clahe_u8_{shp[0]}x{shp[1]}_c{cl}_g{gr[0]}x{gr[1]}", "clahe", [a],
              {"clipLimit": cl, "grid": list(gr), "bitShift": 0}, [r])
    a = np.full((40, 48), 77, np.uint8)
    g.add("clahe_u8_constant", "clahe", [a], {"clipLimit": 40.0, "grid": [8, 8], "bitShift": 0},
          [cv2.createCLAHE(40.0, (8, 8)).apply(a)])
    parent = rng.integers(0, 256, (90, 100), dtype=np.uint8)
    g.add("clahe_u8_roi", "clahe", [parent], {"clipLimit": 3.0, "grid": [4, 4], "bitShift": 0,
                                              "roi": [5, 9, 77, 61], "roiInputs": [0]},
          [cv2.createCLAHE(3.0, (4, 4)).apply(parent[9:70, 5:82])])
    for shp, cl, gr in [((64, 64), 40.0, (8, 8)), ((67, 53), 2.0, (4, 4)), ((40, 30), 0.0, (3, 3))]:
        a = rng.integers(0, 4096, shp, dtype=np.uint16) * 16
        r = cv2.createCLAHE(cl, gr).apply(a)
        g.add(f"clahe_u16_{shp[0]}x{shp[1]}_c{cl}_g{gr[0]}x{gr[1]}", "clahe", [a],
              {"clipLimit": cl, "grid": list(gr), "bitShift": 0}, [r])
    a = rng.integers(0, 65536, (70, 90), dtype=np.uint16)
    g.add("clahe_u16_full_range_c40", "clahe", [a], {"clipLimit": 40.0, "grid": [8, 8], "bitShift": 0},
          [cv2.createCLAHE(40.0, (8, 8)).apply(a)])

    # OpenCV 5.0 bit shift: model-derived.
    a = rng.integers(0, 256, (45, 51), dtype=np.uint8)
    for s in (2, 5):
        g.add(f"clahe_u8_bitshift{s}_model", "clahe", [a], {"clipLimit": 3.0, "grid": [4, 4], "bitShift": s},
              [model_clahe(a, 3.0, (4, 4), s)], model=True)
    a = rng.integers(0, 65536, (33, 40), dtype=np.uint16)
    g.add("clahe_u16_bitshift4_model", "clahe", [a], {"clipLimit": 2.0, "grid": [3, 4], "bitShift": 4},
          [model_clahe(a, 2.0, (3, 4), 4)], model=True)
    assert np.array_equal(model_clahe(a, 2.0, (3, 4), 0), cv2.createCLAHE(2.0, (3, 4)).apply(a))


def integral(g, rng):
    a8 = rng.integers(0, 256, (9, 11), dtype=np.uint8)
    for sd in ("i32", "f32", "f64"):
        g.add(f"integral_u8_{sd}", "integral", [a8], {"sdepth": sd, "sqdepth": None, "tilted": False},
              [cv2.integral(a8, sdepth=CVDEPTH[sd])])
    for sd, sq in (("i32", "f64"), ("i32", "f32"), ("i32", "i32"), ("f32", "f64"), ("f32", "f32"), ("f64", "f64")):
        s, q = cv2.integral2(a8, sdepth=CVDEPTH[sd], sqdepth=CVDEPTH[sq])
        g.add(f"integral2_u8_{sd}_{sq}", "integral", [a8], {"sdepth": sd, "sqdepth": sq, "tilted": False}, [s, q])
    wide8 = rng.integers(0, 256, (13, 257), dtype=np.uint8)
    s, q = cv2.integral2(wide8, sdepth=cv2.CV_32S, sqdepth=cv2.CV_64F)
    g.add("integral2_u8_wide_i32_f64", "integral", [wide8], {"sdepth": "i32", "sqdepth": "f64", "tilted": False}, [s, q])
    s, q = cv2.integral2(wide8, sdepth=cv2.CV_32F, sqdepth=cv2.CV_32F)
    g.add("integral2_u8_wide_f32_f32", "integral", [wide8], {"sdepth": "f32", "sqdepth": "f32", "tilted": False}, [s, q])
    s, q, t = cv2.integral3(a8)
    g.add("integral3_u8c1", "integral", [a8], {"sdepth": "i32", "sqdepth": "f64", "tilted": True}, [s, q, t])
    c3 = rng.integers(0, 256, (6, 7, 3), dtype=np.uint8)
    s, q, t = cv2.integral3(c3)
    g.add("integral3_u8c3", "integral", [c3], {"sdepth": "i32", "sqdepth": "f64", "tilted": True}, [s, q, t])
    for nm, sub in (("1row", a8[:1].copy()), ("1col", a8[:, :1].copy()), ("1x1", a8[:1, :1].copy()),
                    ("2col", a8[:, :2].copy())):
        s, q, t = cv2.integral3(sub)
        g.add(f"integral3_u8_{nm}", "integral", [sub], {"sdepth": "i32", "sqdepth": "f64", "tilted": True}, [s, q, t])
    c4 = rng.integers(0, 256, (5, 9, 4), dtype=np.uint8)
    g.add("integral_u8c4_i32", "integral", [c4], {"sdepth": "i32", "sqdepth": None, "tilted": False},
          [cv2.integral(c4, sdepth=cv2.CV_32S)])
    s, q = cv2.integral2(c4, sdepth=cv2.CV_64F, sqdepth=cv2.CV_64F)
    g.add("integral2_u8c4_f64_f64", "integral", [c4], {"sdepth": "f64", "sqdepth": "f64", "tilted": False}, [s, q])
    c2 = rng.integers(0, 256, (7, 5, 2), dtype=np.uint8)
    s, q = cv2.integral2(c2, sdepth=cv2.CV_32F, sqdepth=cv2.CV_64F)
    g.add("integral2_u8c2_f32_f64", "integral", [c2], {"sdepth": "f32", "sqdepth": "f64", "tilted": False}, [s, q])
    u16 = rng.integers(0, 65536, (8, 10), dtype=np.uint16)
    s, q = cv2.integral2(u16)
    g.add("integral2_u16_default", "integral", [u16], {"sdepth": "f64", "sqdepth": "f64", "tilted": False}, [s, q])
    s, q, t = cv2.integral3(u16, sdepth=cv2.CV_64F, sqdepth=cv2.CV_64F)
    g.add("integral3_u16", "integral", [u16], {"sdepth": "f64", "sqdepth": "f64", "tilted": True}, [s, q, t])
    s16 = rng.integers(-32768, 32768, (8, 9), dtype=np.int16)
    s, q = cv2.integral2(s16)
    g.add("integral2_i16_default", "integral", [s16], {"sdepth": "f64", "sqdepth": "f64", "tilted": False}, [s, q])
    f = (rng.random((40, 50)) * 3).astype(np.float32)
    g.add("integral_f32_f64", "integral", [f], {"sdepth": "f64", "sqdepth": None, "tilted": False},
          [cv2.integral(f, sdepth=cv2.CV_64F)])
    s, q = cv2.integral2(f, sdepth=cv2.CV_32F, sqdepth=cv2.CV_32F)
    g.add("integral2_f32_f32_f32", "integral", [f], {"sdepth": "f32", "sqdepth": "f32", "tilted": False}, [s, q])
    s, q = cv2.integral2(f, sdepth=cv2.CV_32F, sqdepth=cv2.CV_64F)
    g.add("integral2_f32_f32_f64", "integral", [f], {"sdepth": "f32", "sqdepth": "f64", "tilted": False}, [s, q])
    # Sum-only 32F->32F golden from integral2 (cv2 4.5.5 IPP path differs when IPP is on).
    g.add("integral_f32_f32_from_integral2", "integral", [f], {"sdepth": "f32", "sqdepth": None, "tilted": False}, [s])
    f2 = (rng.random((7, 8, 2)) * 3 - 1).astype(np.float32)
    s, q, t = cv2.integral3(f2, sdepth=cv2.CV_32F, sqdepth=cv2.CV_32F)
    g.add("integral3_f32c2_f32_f32", "integral", [f2], {"sdepth": "f32", "sqdepth": "f32", "tilted": True}, [s, q, t])
    s, q, t = cv2.integral3(f, sdepth=cv2.CV_32F, sqdepth=cv2.CV_64F)
    g.add("integral3_f32_f32_f64", "integral", [f], {"sdepth": "f32", "sqdepth": "f64", "tilted": True}, [s, q, t])
    d = rng.standard_normal((9, 12)) * 100
    s, q = cv2.integral2(d)
    g.add("integral2_f64", "integral", [d], {"sdepth": "f64", "sqdepth": "f64", "tilted": False}, [s, q])
    s, q, t = cv2.integral3(d, sdepth=cv2.CV_64F, sqdepth=cv2.CV_64F)
    g.add("integral3_f64", "integral", [d], {"sdepth": "f64", "sqdepth": "f64", "tilted": True}, [s, q, t])
    parent = rng.integers(0, 256, (30, 40), dtype=np.uint8)
    sub = parent[4:25, 3:36]
    s, q = cv2.integral2(sub, sdepth=cv2.CV_32S, sqdepth=cv2.CV_64F)
    g.add("integral2_u8_roi", "integral", [parent], {"sdepth": "i32", "sqdepth": "f64", "tilted": False,
                                                     "roi": [3, 4, 33, 21], "roiInputs": [0]}, [s, q])
    pf = (rng.random((20, 30)) * 2 - 1).astype(np.float32)
    s, q = cv2.integral2(pf[2:17, 5:28], sdepth=cv2.CV_32F, sqdepth=cv2.CV_64F)
    g.add("integral2_f32_roi", "integral", [pf], {"sdepth": "f32", "sqdepth": "f64", "tilted": False,
                                                  "roi": [5, 2, 23, 15], "roiInputs": [0]}, [s, q])


def normalize(g, rng):
    a = (rng.random((10, 12)) * 300 - 20).astype(np.float32)
    mk = (rng.random((10, 12)) > 0.5).astype(np.uint8)
    u8 = rng.integers(10, 200, (9, 13), dtype=np.uint8)
    u16 = rng.integers(0, 65536, (7, 11), dtype=np.uint16)
    c3 = rng.integers(0, 256, (6, 7, 3), dtype=np.uint8)
    const = np.full((5, 6), 3.5, np.float32)
    cases = [
        ("minmax_u8_u8", u8, 0, 255, cv2.NORM_MINMAX, "u8", None),
        ("minmax_u8_u8_reversed", u8, 255, 10, cv2.NORM_MINMAX, "u8", None),
        ("minmax_f32_f32", a, 0, 1, cv2.NORM_MINMAX, "f32", None),
        ("minmax_f32_f32_mask", a, -1, 1, cv2.NORM_MINMAX, "f32", mk),
        ("minmax_f32_u8_mask", a, 0, 255, cv2.NORM_MINMAX, "u8", mk),
        ("minmax_f32_f64", a, 0, 10, cv2.NORM_MINMAX, "f64", None),
        ("minmax_constant", const, 2, 9, cv2.NORM_MINMAX, "f32", None),
        ("minmax_u8c3", c3, 0, 100, cv2.NORM_MINMAX, "u8", None),
        ("minmax_u16_f32", u16, 0, 1, cv2.NORM_MINMAX, "f32", None),
        ("l1_f32", a, 1, 0, cv2.NORM_L1, "f32", None),
        ("l2_f32_mask", a, 1, 0, cv2.NORM_L2, "f32", mk),
        ("inf_f32_mask", a, 5, 0, cv2.NORM_INF, "f32", mk),
        ("l2_u8_f32", u8, 1, 0, cv2.NORM_L2, "f32", None),
        ("l1_u8_u8", u8, 5000, 0, cv2.NORM_L1, "u8", None),
        ("inf_u16_f64", u16, 2, 0, cv2.NORM_INF, "f64", None),
        ("l2_u8c3_f32", c3, 3, 0, cv2.NORM_L2, "f32", None),
    ]
    for name, src, al, be, nt, dt, mm in cases:
        r = cv2.normalize(src, None, al, be, nt, CVDEPTH[dt], mm)
        ins = [src] + ([mm] if mm is not None else [])
        g.add(f"normalize_{name}", "normalize", ins,
              {"alpha": al, "beta": be, "normType": nt, "dtype": dt, "mask": mm is not None}, [r])


def main():
    rng = np.random.default_rng(20261012)
    g = Gen()
    calc_hist(g, rng)
    back_project(g, rng)
    compare_hist(g, rng)
    equalize(g, rng)
    clahe(g, rng)
    integral(g, rng)
    normalize(g, rng)

    # Sanity checks on the reference itself.
    n32 = np.array([[np.nan, np.inf, -np.inf, 0.5, 1.0]], np.float32)
    assert cv2.calcHist([n32], [0], np.ones_like(n32, np.uint8), [4], [0, 1]).ravel().tolist() == [1, 0, 1, 0]
    assert cv2.equalizeHist(np.full((3, 3), 7, np.uint8)).ravel().tolist() == [7] * 9
    assert cv2.integral(np.ones((2, 2), np.uint8))[2, 2] == 4

    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "histogram_golden.json")
    with open(out, "w") as fh:
        json.dump({"arrays": g.arrays, "cases": g.cases}, fh, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases ({sum(c['suspect'] for c in g.cases)} suspect, "
          f"{sum(c['model'] for c in g.cases)} model) to {out}")
    for c in g.cases:
        if c["suspect"]:
            print("  suspect:", c["name"])


if __name__ == "__main__":
    main()
