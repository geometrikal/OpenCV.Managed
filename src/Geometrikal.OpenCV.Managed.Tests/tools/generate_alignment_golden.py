#!/usr/bin/env python3
"""Golden-data generator for the managed alignment port.

Runs cv2 5.0.0 ``matchTemplate`` (all methods, 8U/32F, 1-4 channels, byte and float masks),
``phaseCorrelate`` (with and without a window), ``createHanningWindow``,
``phaseCorrelateIterative``, ``computeECC`` and ``findTransformECC`` /
``findTransformECCWithMask`` over a seeded battery and writes ``TestData/alignment_golden.json``.
``ImgProc/TemplateMatchingGoldenTests.cs``, ``PhaseCorrelationGoldenTests.cs`` and
``EccGoldenTests.cs`` replay the cases.

Required environment (the script stops if cv2 is not 5.0.0):
 * cv2 5.0.0, run with ``PYTHONPATH=<dir>`` (see ``generate_advfilter_golden.py``).
 * OPENCV_CPU_DISABLE=AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL before cv2 is
   imported (the script sets it when it is not set).
 * IPP, OpenCL and threads are disabled in the script.

Regenerate with (from the ``Geometrikal.OpenCV.Managed.Tests`` directory):
    PYTHONPATH=<cv2 5.0.0 dir> python tools/generate_alignment_golden.py

Battery notes:
 * Inputs are smooth textures (blurred seeded noise plus sinusoids) so that the alignment
   problems have one clear optimum.
 * matchTemplate outputs with more than 4096 elements are stored as SHA-256 (``expectedSha``)
   of the float32 bytes. Large cases use several DFT tiles.
 * Float results of phaseCorrelate, computeECC and findTransformECC are stored as IEEE-754 bit
   patterns. The runners compare them exactly (see AlignmentPortNotes.md).

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import hashlib
import json
import os
import struct
import sys

os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

if cv2.__version__ != "5.0.0":
    sys.exit(f"generate_alignment_golden.py requires cv2 5.0.0 (found {cv2.__version__}); "
             "set PYTHONPATH to the opencv-contrib-python-headless==5.0.0.93 install")
if cv2.checkHardwareSupport(11) or cv2.checkHardwareSupport(13):
    sys.exit("AVX2/AVX-512 dispatch is enabled; set OPENCV_CPU_DISABLE before importing cv2")

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

DTYPES = {"u8": np.uint8, "f32": np.float32, "f64": np.float64}
METHODS = ["SqDiff", "SqDiffNormed", "CCorr", "CCorrNormed", "CCoeff", "CCoeffNormed"]
SUSPECT_CASES = {}


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def dbits(v):
    return struct.unpack("<q", struct.pack("<d", float(v)))[0]


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

    def add(self, name, op, inputs, params, out=None, scalars=None):
        assert name not in self.names, name
        self.names.add(name)
        case = {"name": name, "op": op, "params": params, "suspect": name in SUSPECT_CASES,
                "inputs": {k: self.ref(v) for k, v in inputs.items() if v is not None}}
        if case["suspect"]:
            case["suspectReason"] = SUSPECT_CASES[name]
        if out is not None:
            out = np.ascontiguousarray(out)
            if out.size > 4096:
                case["expectedSha"] = sha(out)
                case["expectedShape"] = [int(out.shape[1]), int(out.shape[0])]
            else:
                case["expected"] = self.ref(out)
        if scalars is not None:
            case["scalars"] = scalars
        self.cases.append(case)


def texture(rng, h, w, cn=1, scale=255.0):
    planes = []
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    for c in range(cn):
        n = rng.standard_normal((h, w)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), 2.0)
        n = n / (np.abs(n).max() + 1e-9)
        s = np.sin(xs * (0.11 + 0.03 * c)) * np.cos(ys * (0.07 + 0.02 * c))
        v = 0.5 + 0.35 * n + 0.15 * s
        planes.append(np.clip(v, 0, 1) * scale)
    a = np.dstack(planes) if cn > 1 else planes[0]
    return a.astype(np.float32)


def as_dtype(a, dt):
    if dt == "u8":
        return np.clip(np.rint(a), 0, 255).astype(np.uint8)
    return a.astype(DTYPES[dt])


def match_template(g, rng):
    # Small images: one DFT tile; outputs stored in full.
    for dt in ("u8", "f32"):
        for cn in (1, 2, 3, 4):
            img = as_dtype(texture(rng, 41, 47, cn), dt)
            for (tw, th) in ((1, 1), (5, 7), (16, 16), (47, 3)):
                ty, tx = (41 - th) // 3, (47 - tw) // 2
                templ = img[ty:ty + th, tx:tx + tw].copy()
                if dt == "f32":
                    templ = templ + rng.standard_normal(templ.shape).astype(np.float32)
                for m, meth in enumerate(METHODS):
                    out = cv2.matchTemplate(img, templ, m)
                    g.add(f"mt_{dt}_c{cn}_t{tw}x{th}_{meth}", "matchTemplate", {"image": img, "templ": templ},
                          {"method": meth}, out)

    # Constant template: CCOEFF_NORMED returns ones; constant image window: normed rounding guards.
    for dt in ("u8", "f32"):
        for cn in (1, 3):
            img = as_dtype(texture(rng, 30, 33, cn), dt)
            img[5:20, 4:20] = 100
            templ = np.full((7, 6, cn) if cn > 1 else (7, 6), 100, DTYPES[dt])
            for m, meth in enumerate(METHODS):
                out = cv2.matchTemplate(img, templ, m)
                g.add(f"mt_const_{dt}_c{cn}_{meth}", "matchTemplate", {"image": img, "templ": templ}, {"method": meth}, out)

    # Template larger than the image in both dimensions: OpenCV swaps the inputs.
    for dt in ("u8", "f32"):
        big = as_dtype(texture(rng, 20, 24), dt)
        small = big[3:15, 5:20].copy()
        for m, meth in enumerate(METHODS):
            out = cv2.matchTemplate(small, big, m)
            g.add(f"mt_swap_{dt}_{meth}", "matchTemplate", {"image": small, "templ": big}, {"method": meth}, out)

    # Several DFT tiles (8x8 template: 249-column blocks; 64x64 template: 297-column blocks).
    for dt in ("u8", "f32"):
        for cn in (1, 3):
            for (iw, ih, tw, th) in ((300, 280, 8, 8), (400, 380, 64, 64)):
                img = as_dtype(texture(rng, ih, iw, cn), dt)
                templ = img[100:100 + th, 120:120 + tw].copy()
                for m, meth in enumerate(METHODS):
                    out = cv2.matchTemplate(img, templ, m)
                    g.add(f"mt_large_{dt}_c{cn}_{iw}x{ih}_t{tw}_{meth}", "matchTemplate",
                          {"image": img, "templ": templ}, {"method": meth}, out)

    # Masks: byte masks (values 0, 1 and above 1), float weight masks; 1 or cn channels.
    for dt in ("u8", "f32"):
        for cn in (1, 3):
            img = as_dtype(texture(rng, 37, 43, cn), dt)
            templ = img[9:20, 13:26].copy()
            th, tw = templ.shape[:2]
            for mcn in sorted({1, cn}):
                mshape = (th, tw) if mcn == 1 else (th, tw, mcn)
                mask8 = (rng.integers(0, 4, mshape) * 60).astype(np.uint8)
                maskf = rng.uniform(0.0, 1.5, mshape).astype(np.float32)
                maskf[rng.uniform(size=mshape) < 0.2] = 0
                for mname, mask in (("m8", mask8), ("mf", maskf)):
                    for m, meth in enumerate(METHODS):
                        out = cv2.matchTemplate(img, templ, m, mask=mask)
                        g.add(f"mt_mask_{dt}_c{cn}_{mname}{mcn}_{meth}", "matchTemplate",
                              {"image": img, "templ": templ, "mask": mask}, {"method": meth}, out)
    for dt in ("u8", "f32"):
        img = as_dtype(texture(rng, 300, 290), dt)
        templ = img[50:62, 70:84].copy()
        mask = (rng.integers(0, 2, templ.shape) * 255).astype(np.uint8)
        for m, meth in enumerate(METHODS):
            out = cv2.matchTemplate(img, templ, m, mask=mask)
            g.add(f"mt_mask_large_{dt}_{meth}", "matchTemplate", {"image": img, "templ": templ, "mask": mask},
                  {"method": meth}, out)


def shifted_pair(rng, h, w, dx, dy, dt):
    base = texture(rng, h + 40, w + 40)
    m = np.float64([[1, 0, dx], [0, 1, dy]])
    moved = cv2.warpAffine(base, m, (w + 40, h + 40), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    a = base[20:20 + h, 20:20 + w]
    b = moved[20:20 + h, 20:20 + w]
    return np.ascontiguousarray(a).astype(DTYPES[dt]), np.ascontiguousarray(b).astype(DTYPES[dt])


def phase_correlate(g, rng):
    for dt in ("f32", "f64"):
        cvt = cv2.CV_32F if dt == "f32" else cv2.CV_64F
        for (w, h) in ((2, 2), (3, 5), (17, 9), (64, 48), (101, 77)):
            win = cv2.createHanningWindow((w, h), cvt)
            g.add(f"hann_{dt}_{w}x{h}", "createHanningWindow", {}, {"width": w, "height": h, "dtype": dt}, win)
        for (w, h, dx, dy) in ((64, 64, 3.0, -5.0), (100, 75, 7.25, 2.5), (128, 96, -10.6, 4.3), (45, 61, 1.5, -2.75),
                               (40, 1, 2.0, 0.0)):
            if w == 1 or h == 1:
                a = texture(rng, h, w).astype(DTYPES[dt])
                b = np.roll(a, (int(dy), int(dx)), axis=(0, 1))
            else:
                a, b = shifted_pair(rng, h, w, dx, dy, dt)
            for use_win in (False, True):
                if use_win and (w < 2 or h < 2):
                    continue
                win = cv2.createHanningWindow((w, h), cvt) if use_win else None
                # OpenCV multiplies the window into unpadded sources in place; pass copies.
                (sx, sy), resp = (cv2.phaseCorrelate(a.copy(), b.copy(), win) if use_win
                                  else cv2.phaseCorrelate(a.copy(), b.copy()))
                name = f"pc_{dt}_{w}x{h}_{'win' if use_win else 'nowin'}"
                g.add(name, "phaseCorrelate", {"a": a, "b": b, "window": win}, {},
                      scalars={"x": dbits(sx), "y": dbits(sy), "response": dbits(resp)})
        for (w, h, dx, dy) in ((64, 64, 3.3, -5.6), (96, 80, -7.25, 2.5), (128, 128, 12.4, 9.1)):
            a, b = shifted_pair(rng, h, w, dx, dy, dt)
            for (l2, iters) in ((7, 10), (5, 3)):
                sx, sy = cv2.phaseCorrelateIterative(a, b, l2, iters)
                g.add(f"pci_{dt}_{w}x{h}_L{l2}_i{iters}", "phaseCorrelateIterative", {"a": a, "b": b},
                      {"L2size": l2, "maxIters": iters}, scalars={"x": dbits(sx), "y": dbits(sy)})


MOTIONS = {"Translation": cv2.MOTION_TRANSLATION, "Euclidean": cv2.MOTION_EUCLIDEAN,
           "Affine": cv2.MOTION_AFFINE, "Homography": cv2.MOTION_HOMOGRAPHY}


def ecc(g, rng):
    truth = {
        "Translation": np.float32([[1, 0, 2.6], [0, 1, -1.8]]),
        "Euclidean": cv2.getRotationMatrix2D((40, 30), 3.0, 1.0).astype(np.float32),
        "Affine": np.float32([[1.02, 0.03, -1.5], [-0.02, 0.98, 2.2]]),
        "Homography": np.float32([[1.01, 0.02, -1.2], [-0.015, 0.99, 1.7], [1e-4, -5e-5, 1]]),
    }
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-6)
    for dt in ("u8", "f32"):
        for cn in (1, 3):
            base = texture(rng, 60, 80, cn)
            templ = as_dtype(base, dt)
            for mname, mt in MOTIONS.items():
                w = truth[mname]
                if mt == cv2.MOTION_HOMOGRAPHY:
                    moved = cv2.warpPerspective(base, w, (80, 60), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                else:
                    moved = cv2.warpAffine(base, w, (80, 60), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                inp = as_dtype(moved, dt)
                rows = 3 if mt == cv2.MOTION_HOMOGRAPHY else 2
                for gauss in (5, 1):
                    init = np.eye(rows, 3, dtype=np.float32)
                    rho, warp = cv2.findTransformECC(templ, inp, init, mt, criteria, None, gauss)
                    g.add(f"ecc_{dt}_c{cn}_{mname}_g{gauss}", "findTransformECC", {"templ": templ, "input": inp},
                          {"motion": mname, "gauss": gauss, "maxCount": 40, "eps": 1e-6},
                          scalars={"rho": dbits(rho), "warp": [dbits(v) for v in warp.ravel()]})
                if cn == 1:
                    mask = np.full((60, 80), 255, np.uint8)
                    mask[:, :12] = 0
                    mask[45:, 60:] = 0
                    init = np.eye(rows, 3, dtype=np.float32)
                    rho, warp = cv2.findTransformECC(templ, inp, init, mt, criteria, mask, 5)
                    g.add(f"ecc_{dt}_{mname}_inmask", "findTransformECC", {"templ": templ, "input": inp, "inputMask": mask},
                          {"motion": mname, "gauss": 5, "maxCount": 40, "eps": 1e-6},
                          scalars={"rho": dbits(rho), "warp": [dbits(v) for v in warp.ravel()]})
                    init = np.eye(rows, 3, dtype=np.float32)
                    rho, warp = cv2.findTransformECCWithMask(templ, inp, mask, mask, init, mt, criteria, 5)
                    g.add(f"ecc_{dt}_{mname}_bothmasks", "findTransformECC",
                          {"templ": templ, "input": inp, "inputMask": mask, "templateMask": mask},
                          {"motion": mname, "gauss": 5, "maxCount": 40, "eps": 1e-6},
                          scalars={"rho": dbits(rho), "warp": [dbits(v) for v in warp.ravel()]})
            # Count-only criteria: a fixed number of iterations.
            init = np.eye(2, 3, dtype=np.float32)
            inp = as_dtype(cv2.warpAffine(base, truth["Affine"], (80, 60), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REFLECT), dt)
            rho, warp = cv2.findTransformECC(templ, inp, init, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_COUNT, 3, 0), None, 5)
            g.add(f"ecc_{dt}_c{cn}_Affine_count3", "findTransformECC", {"templ": templ, "input": inp},
                  {"motion": "Affine", "gauss": 5, "maxCount": 3, "eps": 0.0, "countOnly": True},
                  scalars={"rho": dbits(rho), "warp": [dbits(v) for v in warp.ravel()]})

    for dt in ("u8", "f32"):
        for cn in (1, 3):
            a = as_dtype(texture(rng, 31, 29, cn), dt)
            b = as_dtype(texture(rng, 31, 29, cn), dt)
            g.add(f"cecc_{dt}_c{cn}", "computeECC", {"templ": a, "input": b}, {},
                  scalars={"ecc": dbits(cv2.computeECC(a, b))})
            mask = (rng.integers(0, 3, (31, 29)) > 0).astype(np.uint8) * 255
            g.add(f"cecc_{dt}_c{cn}_mask", "computeECC", {"templ": a, "input": b, "inputMask": mask}, {},
                  scalars={"ecc": dbits(cv2.computeECC(a, b, mask))})


def main():
    rng = np.random.default_rng(20261016)
    g = Gen()
    match_template(g, rng)
    phase_correlate(g, rng)
    ecc(g, rng)
    assert any(c["op"] == "matchTemplate" and "mask" in c["inputs"] for c in g.cases)
    out = {"generator": "generate_alignment_golden.py", "opencv": cv2.__version__, "cases": g.cases, "arrays": g.arrays}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TestData", "alignment_golden.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases, {len(g.arrays)} arrays to {os.path.normpath(path)}")


if __name__ == "__main__":
    main()
