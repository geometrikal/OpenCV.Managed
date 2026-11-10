#!/usr/bin/env python3
"""Golden-data generator for the managed corner detector port.

Runs cv2 5.0.0 ``cornerMinEigenVal``, ``cornerHarris``, ``cornerEigenValsAndVecs``,
``goodFeaturesToTrack`` and ``cornerSubPix`` over a seeded battery and writes
``TestData/corners_golden.json``. ``ImgProc/CornersGoldenTests.cs`` replays the cases and
compares the outputs exactly (float bits).

Required environment (the script stops if cv2 is not 5.0.0):
 * cv2 5.0.0, for example
       pip install --no-deps --target <dir> opencv-contrib-python-headless==5.0.0.93
   and run with ``PYTHONPATH=<dir>``. numpy 1.22 or later.
 * OPENCV_CPU_DISABLE=AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL before cv2 is
   imported (the script sets it when it is not set). The AVX line functions of corner.cpp and
   the AVX2 filter kernels are then not used; the port follows the SSE3 baseline.
 * IPP, OpenCL and threads are disabled in the script (IPP has a cornerHarris path).

Regenerate with (from the ``Geometrikal.OpenCV.Managed.Tests`` directory):
    PYTHONPATH=<cv2 5.0.0 dir> python tools/generate_corners_golden.py

Battery notes:
 * Images are synthetic: filled rectangles and triangles, checkerboards, smoothed noise and
   plain noise, as byte and float. Sizes with ``width·height mod 4 != 0`` check the Harris
   scalar tail.
 * Outputs with more than 4096 elements are stored as SHA-256 (``sha``).
 * ``goodFeaturesToTrack`` outputs are corner lists ``[x, y, quality bits]``.
 * ``cornerSubPix`` starts from integer corners plus offsets and stores the refined float bits.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import hashlib
import json
import os
import platform
import sys

os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

if cv2.__version__ != "5.0.0":
    sys.exit(f"generate_corners_golden.py requires cv2 5.0.0 (found {cv2.__version__}); "
             "set PYTHONPATH to the opencv-contrib-python-headless==5.0.0.93 install")
if cv2.checkHardwareSupport(11) or cv2.checkHardwareSupport(13):
    sys.exit("AVX2/AVX-512 dispatch is enabled; set OPENCV_CPU_DISABLE before importing cv2")

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

BORDERS = {"Reflect101": cv2.BORDER_REFLECT_101, "Replicate": cv2.BORDER_REPLICATE,
           "Reflect": cv2.BORDER_REFLECT, "Constant": cv2.BORDER_CONSTANT}
SUSPECT_CASES = {}

ARRAYS = []
ARRAY_INDEX = {}


def ref(a):
    a = np.ascontiguousarray(a)
    m = {"dtype": "u8" if a.dtype == np.uint8 else "f32", "width": int(a.shape[1]), "height": int(a.shape[0]),
         "channels": 1 if a.ndim == 2 else int(a.shape[2]), "data": base64.b64encode(a.tobytes()).decode("ascii")}
    key = (m["dtype"], m["width"], m["height"], m["channels"], m["data"])
    if key not in ARRAY_INDEX:
        ARRAY_INDEX[key] = len(ARRAYS)
        ARRAYS.append(m)
    return ARRAY_INDEX[key]


def expected(out):
    out = np.ascontiguousarray(out)
    if out.size > 4096:
        return {"sha": hashlib.sha256(out.tobytes()).hexdigest()}
    return {"array": ref(out)}


def fbits(v):
    return int(np.array(v, np.float32).view(np.uint32))


class Gen:
    def __init__(self):
        self.cases = []
        self.names = set()

    def add(self, name, case):
        assert name not in self.names, name
        self.names.add(name)
        case["name"] = name
        case["suspect"] = name in SUSPECT_CASES
        if case["suspect"]:
            case["suspectReason"] = SUSPECT_CASES[name]
        self.cases.append(case)


def shapes_image(rng, w, h):
    img = np.full((h, w), 40, np.uint8)
    for _ in range(4):
        x0, y0 = int(rng.integers(0, w - 2)), int(rng.integers(0, h - 2))
        x1, y1 = int(rng.integers(x0 + 2, w + 1)), int(rng.integers(y0 + 2, h + 1))
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), int(rng.integers(60, 255)), -1)
    pts = rng.integers(0, [w, h], (3, 2)).astype(np.int32)
    cv2.fillPoly(img, [pts], int(rng.integers(0, 255)))
    return img


def checker_image(w, h, cell):
    ys, xs = np.mgrid[0:h, 0:w]
    return (((xs // cell + ys // cell) % 2) * 200 + 30).astype(np.uint8)


def images(rng):
    out = []
    for (w, h) in ((7, 5), (13, 9), (31, 17), (40, 30), (64, 48)):
        out.append((f"shapes{w}x{h}", shapes_image(rng, w, h)))
    out.append(("checker33x27", checker_image(33, 27, 5)))
    out.append(("checker48x48", checker_image(48, 48, 8)))
    noise = rng.integers(0, 256, (23, 29), dtype=np.uint8)
    out.append(("noise29x23", noise))
    out.append(("smooth37x21", cv2.GaussianBlur(rng.integers(0, 256, (21, 37), dtype=np.uint8), (5, 5), 1.5)))
    return out


def as_type(img, dt, rng):
    if dt == "u8":
        return img
    return (img.astype(np.float32) / 255.0 * 3.0 + rng.standard_normal(img.shape).astype(np.float32) * 0.01).astype(np.float32)


def responses(g, rng):
    for dt in ("u8", "f32"):
        for name, img in images(rng):
            src = as_type(img, dt, rng)
            for border in BORDERS:
                for block in (2, 3, 5):
                    for ksize in (1, 3, 5, 7, -1):
                        if border != "Reflect101" and (block != 3 or ksize not in (3, -1)):
                            continue
                        if name not in ("shapes40x30", "checker33x27", "noise29x23") and (block, ksize) not in ((3, 3), (2, -1), (5, 5)):
                            continue
                        b = BORDERS[border]
                        p = {"blockSize": block, "ksize": ksize, "border": border}
                        g.add(f"mineig_{dt}_{name}_b{block}_k{ksize}_{border}",
                              {"op": "minEig", "input": ref(src), "params": p,
                               "outputs": [expected(cv2.cornerMinEigenVal(src, block, ksize=ksize, borderType=b))]})
                        for k in (0.04, 0.12):
                            pk = dict(p, k=k)
                            g.add(f"harris_{dt}_{name}_b{block}_k{ksize}_{border}_{k}",
                                  {"op": "harris", "input": ref(src), "params": pk,
                                   "outputs": [expected(cv2.cornerHarris(src, block, ksize, k, borderType=b))]})
                        g.add(f"eigvv_{dt}_{name}_b{block}_k{ksize}_{border}",
                              {"op": "eigenValsVecs", "input": ref(src), "params": p,
                               "outputs": [expected(cv2.cornerEigenValsAndVecs(src, block, ksize, borderType=b))]})


def gftt(g, rng):
    for dt in ("u8", "f32"):
        for name, img in images(rng):
            src = as_type(img, dt, rng)
            h, w = img.shape
            mask = (rng.random((h, w)) < 0.8).astype(np.uint8) * 255
            for (maxc, q, md, use_mask, block, grad, harris) in (
                    (0, 0.01, 0, False, 3, 3, False), (10, 0.01, 1, False, 3, 3, False),
                    (25, 0.05, 3.5, False, 3, 3, False), (0, 0.001, 2, True, 3, 3, False),
                    (5, 0.1, 5, False, 5, 5, True), (0, 0.01, 1.4, False, 3, -1, True),
                    (100, 0.02, 0.5, True, 2, 3, True), (0, 0.3, 10, False, 7, 7, False)):
                kw = dict(maxCorners=maxc, qualityLevel=q, minDistance=md, blockSize=block, gradientSize=grad,
                          useHarrisDetector=harris, k=0.04)
                corners, quality = cv2.goodFeaturesToTrackWithQuality(src, mask=mask if use_mask else None, **kw)
                lst = [] if corners is None else [[fbits(c[0][0]), fbits(c[0][1]), fbits(qv)]
                                                  for c, qv in zip(corners, quality.reshape(-1))]
                p = dict(kw, mask=ref(mask) if use_mask else None)
                g.add(f"gftt_{dt}_{name}_m{maxc}_q{q}_d{md}_mask{int(use_mask)}_b{block}_g{grad}_h{int(harris)}",
                      {"op": "gftt", "input": ref(src), "params": p, "corners": lst})


def subpix(g, rng):
    for dt in ("u8", "f32"):
        for name, img in images(rng):
            h, w = img.shape
            src = as_type(img, dt, rng)
            pts = cv2.goodFeaturesToTrack(img, 20, 0.01, 2)
            base = [] if pts is None else [tuple(p[0]) for p in pts]
            base += [(0.0, 0.0), (w - 1.0, h - 1.0), (w / 2.0, h / 2.0)]
            start = np.array([[x + float(rng.uniform(-1.5, 1.5)), y + float(rng.uniform(-1.5, 1.5))] for x, y in base],
                             np.float32)
            start[:, 0] = np.clip(start[:, 0], 0, w - 1)
            start[:, 1] = np.clip(start[:, 1], 0, h - 1)
            for (win, zero, crit) in (((1, 1), (-1, -1), (3, 30, 0.01)), ((2, 2), (-1, -1), (3, 40, 0.001)),
                                      ((3, 2), (1, 0), (1, 5, 0.0)), ((5, 5), (-1, -1), (2, 0, 0.1))):
                if w < win[0] * 2 + 5 or h < win[1] * 2 + 5:
                    continue
                refined = cv2.cornerSubPix(src, start.copy().reshape(-1, 1, 2), win, zero, crit).reshape(-1, 2)
                g.add(f"subpix_{dt}_{name}_w{win[0]}x{win[1]}_z{zero[0]}x{zero[1]}_t{crit[0]}_{crit[1]}_{crit[2]}",
                      {"op": "subpix", "input": ref(src),
                       "params": {"win": list(win), "zero": list(zero), "type": crit[0], "maxCount": crit[1],
                                  "epsilon": crit[2]},
                       "start": [[fbits(x), fbits(y)] for x, y in start],
                       "refined": [[fbits(x), fbits(y)] for x, y in refined]})


def large(g, rng):
    img = cv2.GaussianBlur(rng.integers(0, 256, (301, 517), dtype=np.uint8), (7, 7), 2.0)
    for dt in ("u8", "f32"):
        src = as_type(img, dt, rng)
        g.add(f"large_mineig_{dt}", {"op": "minEig", "input": ref(src),
                                       "params": {"blockSize": 3, "ksize": 3, "border": "Reflect101"},
                                       "outputs": [expected(cv2.cornerMinEigenVal(src, 3, ksize=3))]})
        g.add(f"large_harris_{dt}", {"op": "harris", "input": ref(src),
                                       "params": {"blockSize": 2, "ksize": 3, "border": "Reflect101", "k": 0.04},
                                       "outputs": [expected(cv2.cornerHarris(src, 2, 3, 0.04))]})
        corners, quality = cv2.goodFeaturesToTrackWithQuality(src, 500, 0.01, 4.0, mask=None, blockSize=3,
                                                              gradientSize=3)
        g.add(f"large_gftt_{dt}", {"op": "gftt", "input": ref(src),
                                     "params": {"maxCorners": 500, "qualityLevel": 0.01, "minDistance": 4.0,
                                                "blockSize": 3, "gradientSize": 3, "useHarrisDetector": False,
                                                "k": 0.04, "mask": None},
                                     "corners": [[fbits(c[0][0]), fbits(c[0][1]), fbits(qv)]
                                                 for c, qv in zip(corners, quality.reshape(-1))]})


def main():
    rng = np.random.default_rng(20261017)
    g = Gen()
    responses(g, rng)
    gftt(g, rng)
    subpix(g, rng)
    large(g, rng)
    assert set(SUSPECT_CASES) <= g.names, set(SUSPECT_CASES) - g.names
    doc = {"generator": {"cv2": cv2.__version__, "numpy": np.__version__, "machine": platform.processor()},
           "arrays": ARRAYS, "cases": g.cases}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TestData", "corners_golden.json")
    with open(out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases, {len(ARRAYS)} arrays to {out}")


if __name__ == "__main__":
    main()
