#!/usr/bin/env python3
"""Golden-data generator for the managed edge detection port.

Runs cv2 4.5.5 Canny (image and dx/dy overloads) and spatialGradient over a
seeded battery and writes ``TestData/edge_golden.json``.
``ImgProc/EdgeGoldenTests.cs`` replays each case.

Regenerate with:
    python tools/generate_edge_golden.py
(run from the ``Geometrikal.OpenCV.Managed.Tests`` directory).

Environment and battery constraints (see Docs/Notes/EdgePortNotes.md):
 * IPP is disabled. cv2's ipp_Canny (8UC1, 1 thread, aperture 3/5) receives
   the thresholds before the L2 clamp and cvFloor and differs from the native
   path for out-of-range thresholds. OpenCL is disabled; one thread.
 * OPENCV_CPU_DISABLE keeps the Sobel calls on the SSE code path, as for the
   filtering goldens (the aperture-7 float sums are exact either way).
 * L1 dx/dy cases contain no -32768. OpenCV's SIMD magnitude loop maps
   |-32768| to -32768 and its scalar tail to 32768, so the result depends on
   the column position and the vector width. The port uses the scalar rule.
 * Arrays are stored once in a pool and referenced by index.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import json
import os

os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX,AVX2,AVX512F,FMA3")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

SUSPECT_CASES = {}

DTYPES = {"u8": np.uint8, "s16": np.int16}
THRESHOLDS = [(50, 150), (150, 50), (0, 0), (10.5, 30.7), (-5, 20), (100, 100), (300, 900),
              (2000, 50000), (1e10, 1e10 + 1), (-1e10, 10)]
APERTURES = [(3, False), (3, True), (5, False), (5, True), (7, False), (7, True), (-1, True)]


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def dt_of(arr):
    for k, v in DTYPES.items():
        if arr.dtype == v:
            return k
    raise ValueError(arr.dtype)


class Gen:
    def __init__(self):
        self.cases = []
        self.names = set()
        self.arrays = []
        self.index = {}

    def ref(self, arr):
        arr = np.ascontiguousarray(arr)
        m = {"dtype": dt_of(arr), "width": int(arr.shape[1]), "height": int(arr.shape[0]),
             "channels": 1 if arr.ndim == 2 else int(arr.shape[2]), "data": b64(arr)}
        key = (m["dtype"], m["width"], m["height"], m["channels"], m["data"])
        if key not in self.index:
            self.index[key] = len(self.arrays)
            self.arrays.append(m)
        return self.index[key]

    def add(self, name, op, inputs, params, expected, roi=None):
        assert name not in self.names, name
        self.names.add(name)
        case = {"name": name, "op": op, "inputs": [self.ref(a) for a in inputs], "params": params,
                "expected": [self.ref(e) for e in expected], "suspect": name in SUSPECT_CASES}
        if roi is not None:
            case["roi"] = [int(v) for v in roi]
        if case["suspect"]:
            case["suspectReason"] = SUSPECT_CASES[name]
        self.cases.append(case)

    def canny(self, name, img, lo, hi, ap, l2, roi=None):
        src = img if roi is None else img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        out = cv2.Canny(src, lo, hi, apertureSize=ap, L2gradient=l2)
        self.add(name, "canny", [img], {"threshold1": float(lo), "threshold2": float(hi),
                                        "apertureSize": int(ap), "l2": bool(l2)}, [out], roi)

    def canny_deriv(self, name, dx, dy, lo, hi, l2):
        if not l2:
            assert not (dx == -32768).any() and not (dy == -32768).any(), name
        out = cv2.Canny(dx, dy, lo, hi, L2gradient=l2)
        self.add(name, "cannyDeriv", [dx, dy], {"threshold1": float(lo), "threshold2": float(hi),
                                                "l2": bool(l2)}, [out])

    def spatial_gradient(self, name, img, border, roi=None):
        src = img if roi is None else img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        gx, gy = cv2.spatialGradient(src, ksize=3, borderType=border)
        sx = cv2.Sobel(src, cv2.CV_16S, 1, 0, ksize=3, borderType=border | cv2.BORDER_ISOLATED)
        sy = cv2.Sobel(src, cv2.CV_16S, 0, 1, ksize=3, borderType=border | cv2.BORDER_ISOLATED)
        assert np.array_equal(gx, sx) and np.array_equal(gy, sy), name
        self.add(name, "spatialGradient", [img], {"borderType": int(border)}, [gx, gy], roi)


def structured(rng):
    st = np.zeros((60, 75), np.uint8)
    cv2.rectangle(st, (8, 8), (36, 28), 200, -1)
    cv2.circle(st, (52, 36), 16, 120, -1)
    cv2.line(st, (0, 59), (74, 0), 255, 2)
    st[40:, :24] = np.linspace(0, 255, 24, dtype=np.uint8)
    noisy = cv2.add(st, rng.randint(0, 20, st.shape).astype(np.uint8))
    return st, noisy


def main():
    g = Gen()
    rng = np.random.RandomState(20261012)

    smooth = cv2.GaussianBlur(rng.randint(0, 256, (64, 80)).astype(np.uint8), (0, 0), 2.0)
    struct, struct_noisy = structured(rng)
    rand = rng.randint(0, 256, (31, 50)).astype(np.uint8)
    const = np.full((12, 17), 77, np.uint8)
    smooth3 = cv2.GaussianBlur(rng.randint(0, 256, (40, 53, 3)).astype(np.uint8), (0, 0), 1.5)
    smooth4 = cv2.GaussianBlur(rng.randint(0, 256, (23, 37, 4)).astype(np.uint8), (0, 0), 1.5)
    rand2 = rng.randint(0, 256, (11, 19, 2)).astype(np.uint8)
    parent = cv2.GaussianBlur(rng.randint(0, 256, (70, 90)).astype(np.uint8), (0, 0), 1.2)
    parent3 = rng.randint(0, 256, (30, 40, 3)).astype(np.uint8)

    # Full threshold and aperture sweep on medium images.
    for img_name, img in [("smooth", smooth), ("struct", struct), ("structnoisy", struct_noisy),
                          ("smooth3", smooth3)]:
        for ap, l2 in APERTURES:
            for ti, (lo, hi) in enumerate(THRESHOLDS):
                g.canny(f"canny_{img_name}_ap{ap}_{'l2' if l2 else 'l1'}_t{ti}", img, lo, hi, ap, l2)

    # Other images with representative thresholds.
    for img_name, img in [("rand", rand), ("const", const), ("smooth4", smooth4), ("rand2", rand2)]:
        for ap, l2 in APERTURES:
            for ti in (0, 1, 3, 6):
                lo, hi = THRESHOLDS[ti]
                g.canny(f"canny_{img_name}_ap{ap}_{'l2' if l2 else 'l1'}_t{ti}", img, lo, hi, ap, l2)

    # Small sizes and odd widths.
    for (h, w) in [(1, 1), (1, 7), (2, 3), (3, 3), (4, 4), (5, 9), (7, 5), (17, 33), (9, 47), (3, 70)]:
        img = rng.randint(0, 256, (h, w)).astype(np.uint8)
        for ap, l2 in APERTURES:
            for ti in (0, 2):
                lo, hi = THRESHOLDS[ti]
                g.canny(f"canny_r{h}x{w}_ap{ap}_{'l2' if l2 else 'l1'}_t{ti}", img, lo, hi, ap, l2)
    tiny3 = rng.randint(0, 256, (3, 4, 3)).astype(np.uint8)
    for ap, l2 in APERTURES:
        g.canny(f"canny_tiny3_ap{ap}_{'l2' if l2 else 'l1'}", tiny3, 50, 150, ap, l2)

    # Stride ROIs (numpy views are isolated in cv2, as in the port).
    for ap, l2 in [(3, False), (5, True), (7, False)]:
        g.canny(f"canny_roi_ap{ap}_{'l2' if l2 else 'l1'}", parent, 20, 60, ap, l2, roi=(7, 5, 61, 43))
    g.canny("canny_roi3_ap3_l1", parent3, 50, 150, 3, False, roi=(3, 2, 29, 21))

    # Legacy flag: bit 31 of apertureSize selects L2 when L2gradient is false.
    for ap in (3, 5, 7):
        g.canny(f"canny_smooth_ap{ap}_legacyflag", smooth, 30, 90, ap | -2147483648, False)

    # dx/dy overload from Sobel derivatives.
    for img_name, img in [("smooth", smooth), ("structnoisy", struct_noisy), ("smooth3", smooth3)]:
        for ap in (3, 5):
            dx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=ap, borderType=cv2.BORDER_REPLICATE)
            dy = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=ap, borderType=cv2.BORDER_REPLICATE)
            for l2 in (False, True):
                for ti, (lo, hi) in enumerate(THRESHOLDS):
                    g.canny_deriv(f"cannyd_{img_name}_ap{ap}_{'l2' if l2 else 'l1'}_t{ti}",
                                  dx, dy, lo, hi, l2)

    # Random full-range int16: |x| >= 27146 overflows tg67x; dx = dy = -32768 wraps L2.
    for shp in [(50, 61), (37, 8), (20, 13, 3), (50, 61, 3)]:
        dx = rng.randint(-32768, 32768, shp).astype(np.int16)
        dy = rng.randint(-32768, 32768, shp).astype(np.int16)
        dx.flat[::97] = -32768
        dy.flat[::89] = -32768
        dx.flat[::89 * 97] = -32768
        assert ((dx == -32768) & (dy == -32768)).any()
        assert (np.abs(dx.astype(np.int32)) >= 27146).any()
        dx1 = np.where(dx == -32768, -32767, dx).astype(np.int16)
        dy1 = np.where(dy == -32768, -32767, dy).astype(np.int16)
        tag = "x".join(str(s) for s in shp)
        for ti, (lo, hi) in enumerate([(1000, 30000), (20000, 40000), (1e9, 2e9), (50, 150)]):
            g.canny_deriv(f"cannyd_rand{tag}_l2_t{ti}", dx, dy, lo, hi, True)
            g.canny_deriv(f"cannyd_rand{tag}_l1_t{ti}", dx1, dy1, lo, hi, False)

    # spatialGradient: sizes from the spec, both borders, and a stride ROI.
    for (h, w) in [(1, 1), (1, 2), (2, 1), (1, 40), (40, 1), (2, 2), (3, 3), (2, 17), (2, 18), (3, 33),
                   (16, 16), (17, 17), (31, 50), (64, 129), (101, 77)]:
        img = rng.randint(0, 256, (h, w)).astype(np.uint8)
        for border, bname in [(cv2.BORDER_DEFAULT, "reflect101"), (cv2.BORDER_REPLICATE, "replicate")]:
            g.spatial_gradient(f"sg_{h}x{w}_{bname}", img, border)
    g.spatial_gradient("sg_roi_reflect101", parent, cv2.BORDER_DEFAULT, roi=(5, 7, 33, 25))
    g.spatial_gradient("sg_roi_replicate", parent, cv2.BORDER_REPLICATE, roi=(11, 3, 47, 19))

    # Sanity checks.
    assert not cv2.Canny(const, 1, 2).any()
    edges = sum(int(np.count_nonzero(cv2.Canny(smooth, 20, 60))) for _ in range(1))
    assert edges > 100, edges
    print("sanity checks passed")

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "TestData", "edge_golden.json"))
    with open(out_path, "w") as f:
        json.dump({"arrays": g.arrays, "cases": g.cases}, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases, {len(g.arrays)} arrays to {out_path}")
    print(f"file size: {os.path.getsize(out_path)} bytes")


if __name__ == "__main__":
    main()
