#!/usr/bin/env python3
"""Golden-data generator for the managed robust geometric estimation port.

Runs cv2 5.0.0 ``estimateAffine2D``, ``estimateAffinePartial2D``,
``estimateTranslation2D`` and ``findHomography`` over a seeded battery and writes
``TestData/estimation_golden.json``. ``Calib/EstimationGoldenTests.cs`` replays
each case.

Required environment (the script stops if cv2 is not 5.0.0):
 * cv2 5.0.0 built with the internal LAPACK (``Lapack: YES (Built-In
   libclapack)``). Run with ``PYTHONPATH=<dir>``.
 * OPENCV_CPU_DISABLE=AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL
   before cv2 is imported (the script sets it when it is not set).
 * IPP, OpenCL and threads are disabled in the script.

Regenerate with (from the ``Geometrikal.OpenCV.Managed.Tests`` directory):
    PYTHONPATH=<cv2 5.0.0 dir> python tools/generate_estimation_golden.py

Battery notes:
 * RANSAC and LMEDS create ``RNG((uint64)-1)`` in every call, so a call does
   not depend on earlier calls. The script runs every case twice and asserts
   equal outputs.
 * Inputs are stored with their own dtype (``f32`` or ``f64``); cv2 converts
   both to float32 first.
 * ``model`` is the float64 model, or absent when cv2 returns an empty model.
   ``mask`` is the uint8 inlier mask. ``error`` marks cases where cv2 raises.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import json
import math
import os
import sys

os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

if cv2.__version__ != "5.0.0":
    sys.exit(f"generate_estimation_golden.py requires cv2 5.0.0 (found {cv2.__version__})")
if cv2.checkHardwareSupport(11) or cv2.checkHardwareSupport(13):
    sys.exit("AVX2/AVX-512 dispatch is enabled; set OPENCV_CPU_DISABLE before importing cv2")
if "Built-In libclapack" not in cv2.getBuildInformation():
    sys.exit("cv2 must be built with the internal LAPACK (Lapack: YES (Built-In libclapack))")

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

# Adjudicated differences (case name -> reason). Empty when the port matches every case.
SUSPECT_CASES = {}

RANSAC, LMEDS, RHO = cv2.RANSAC, cv2.LMEDS, cv2.RHO


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def run(func, src, dst, method, thr, max_iters, conf, refine):
    try:
        if func == "affine":
            m, mask = cv2.estimateAffine2D(src, dst, method=method, ransacReprojThreshold=thr,
                                           maxIters=max_iters, confidence=conf, refineIters=refine)
        elif func == "partial":
            m, mask = cv2.estimateAffinePartial2D(src, dst, method=method, ransacReprojThreshold=thr,
                                                  maxIters=max_iters, confidence=conf, refineIters=refine)
        elif func == "translation":
            t, mask = cv2.estimateTranslation2D(src, dst, method=method, ransacReprojThreshold=thr,
                                                maxIters=max_iters, confidence=conf, refineIters=refine)
            m = np.array([[t[0], t[1]]], np.float64)
        else:
            m, mask = cv2.findHomography(src, dst, method, thr, maxIters=max_iters, confidence=conf)
    except cv2.error:
        return None
    if m is not None:
        m = np.asarray(m, np.float64)
    mask = np.zeros((0,), np.uint8) if mask is None else np.asarray(mask, np.uint8).reshape(-1)
    return m, mask


class Gen:
    def __init__(self):
        self.cases = []
        self.names = set()

    def add(self, name, func, src, dst, method, thr=3.0, max_iters=2000, conf=0.99, refine=10):
        assert name not in self.names, name
        self.names.add(name)
        dtype = "f32" if src.dtype == np.float32 else "f64"
        r1 = run(func, src.copy(), dst.copy(), method, thr, max_iters, conf, refine)
        r2 = run(func, src.copy(), dst.copy(), method, thr, max_iters, conf, refine)
        assert (r1 is None) == (r2 is None), name
        case = {"name": name, "func": func, "method": int(method), "threshold": thr, "maxIters": max_iters,
                "confidence": conf, "refineIters": refine, "dtype": dtype, "count": int(src.shape[0]),
                "src": b64(src), "dst": b64(dst), "suspect": name in SUSPECT_CASES}
        if case["suspect"]:
            case["suspectReason"] = SUSPECT_CASES[name]
        if r1 is None:
            case["error"] = True
        else:
            m, mask = r1
            if r2[0] is not None:
                assert np.array_equal(m, r2[0]) and np.array_equal(mask, r2[1]), f"{name} is not deterministic"
            if m is not None and m.size > 0:
                case["model"] = b64(m)
                case["modelShape"] = list(m.shape)
            case["mask"] = b64(mask)
            case["maskLength"] = int(mask.size)
        self.cases.append(case)


def transform_points(kind, src, rng):
    if kind == "similarity":
        a = math.radians(rng.uniform(-30, 30))
        s = rng.uniform(0.8, 1.25)
        m = np.array([[s * math.cos(a), -s * math.sin(a), rng.uniform(-80, 80)],
                      [s * math.sin(a), s * math.cos(a), rng.uniform(-80, 80)], [0, 0, 1]])
    elif kind == "affine":
        m = np.array([[rng.uniform(0.8, 1.2), rng.uniform(-0.2, 0.2), rng.uniform(-80, 80)],
                      [rng.uniform(-0.2, 0.2), rng.uniform(0.8, 1.2), rng.uniform(-80, 80)], [0, 0, 1]])
    elif kind == "translation":
        m = np.array([[1, 0, rng.uniform(-80, 80)], [0, 1, rng.uniform(-80, 80)], [0, 0, 1]])
    else:
        m = np.array([[rng.uniform(0.9, 1.1), rng.uniform(-0.1, 0.1), rng.uniform(-50, 50)],
                      [rng.uniform(-0.1, 0.1), rng.uniform(0.9, 1.1), rng.uniform(-50, 50)],
                      [rng.uniform(-2e-4, 2e-4), rng.uniform(-2e-4, 2e-4), 1.0]])
    p = np.hstack([src.astype(np.float64), np.ones((src.shape[0], 1))]) @ m.T
    return p[:, :2] / p[:, 2:3]


def make_set(rng, kind, n, noise, outliers, dtype=np.float32, extent=1000.0):
    src = rng.uniform(0, extent, (n, 2))
    dst = transform_points(kind, src, rng)
    if noise > 0:
        dst = dst + rng.normal(0, noise, dst.shape)
    k = int(round(n * outliers))
    if k > 0:
        idx = rng.choice(n, k, replace=False)
        dst[idx] = rng.uniform(-100, extent + 100, (k, 2))
    if dtype == np.float32:
        return src.astype(np.float32), dst.astype(np.float32)
    return src.astype(np.float64), dst.astype(np.float64)


FUNCS = {
    "affine": ("affine", 3),
    "partial": ("similarity", 2),
    "translation": ("translation", 1),
    "homography": ("homography", 4),
}


def battery(g, rng):
    for func, (kind, mp) in FUNCS.items():
        methods = [("ransac", RANSAC), ("lmeds", LMEDS)]
        if func == "homography":
            methods = [("none", 0)] + methods + [("rho", RHO)]
        default_refine = 0 if func == "translation" else 10
        for mname, method in methods:
            pre = f"{func}_{mname}"
            # Exact and noisy transforms.
            for n in (mp + 2, 20, 100):
                s, d = make_set(rng, kind, n, 0.0, 0.0)
                g.add(f"{pre}_exact_{n}", func, s, d, method, refine=default_refine)
            s, d = make_set(rng, kind, 100, 0.5, 0.0)
            g.add(f"{pre}_noise_100", func, s, d, method, refine=default_refine)
            # Outliers.
            for frac in (0.1, 0.3, 0.45, 0.6):
                for n in (60, 200):
                    s, d = make_set(rng, kind, n, 0.7, frac)
                    g.add(f"{pre}_out{int(frac * 100)}_{n}", func, s, d, method, refine=default_refine)
            s, d = make_set(rng, kind, 500, 1.0, 0.3)
            g.add(f"{pre}_out30_500", func, s, d, method, refine=default_refine)
            # Double inputs.
            s, d = make_set(rng, kind, 150, 0.7, 0.3, np.float64)
            g.add(f"{pre}_f64_out30_150", func, s, d, method, refine=default_refine)
            # Minimal counts.
            for n in (mp, mp + 1):
                if func == "homography" and n < 4:
                    continue
                s, d = make_set(rng, kind, n, 0.3, 0.0)
                g.add(f"{pre}_min_{n}", func, s, d, method, refine=default_refine)
            # Refinement settings.
            s, d = make_set(rng, kind, 120, 0.8, 0.25)
            if func != "homography":
                for refine in (0, 1, 3, 30):
                    g.add(f"{pre}_refine{refine}", func, s, d, method, refine=refine)
            # Threshold, iteration and confidence settings.
            s, d = make_set(rng, kind, 150, 1.5, 0.3)
            for thr in (0.5, 1.0, 5.0) + ((0.0, -1.0) if func == "homography" else ()):
                g.add(f"{pre}_thr{thr}", func, s, d, method, thr=thr, refine=default_refine)
            for iters in (1, 5, 50):
                g.add(f"{pre}_iters{iters}", func, s, d, method, max_iters=iters, refine=default_refine)
            for conf in (0.5, 0.9, 0.999):
                g.add(f"{pre}_conf{conf}", func, s, d, method, conf=conf, refine=default_refine)
            # Degenerate sets.
            n = 40
            t = rng.uniform(0, 500, n)
            line = np.stack([t, 0.5 * t + 10], axis=1).astype(np.float32)
            s2, d2 = make_set(rng, kind, n, 0.0, 0.0)
            g.add(f"{pre}_collinear_src", func, line, d2, method, refine=default_refine)
            g.add(f"{pre}_collinear_both", func, line, line * 1.5 + 3, method, refine=default_refine)
            same = np.repeat(rng.uniform(0, 100, (1, 2)), n, axis=0).astype(np.float32)
            g.add(f"{pre}_same_points", func, same, same + 5, method, refine=default_refine)
            dup = np.concatenate([s2[:5]] * 8).astype(np.float32)
            dd = np.concatenate([d2[:5]] * 8).astype(np.float32)
            g.add(f"{pre}_duplicates", func, dup, dd, method, refine=default_refine)
            # Integer grid coordinates (exact float values), as from feature matchers on pixel grids.
            gx, gy = np.meshgrid(np.arange(0, 640, 64), np.arange(0, 480, 48))
            grid = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
            gd = transform_points(kind, grid, rng).astype(np.float32)
            gd[::7] += 40
            g.add(f"{pre}_grid", func, grid, gd, method, refine=default_refine)
        # Too few points.
        s, d = make_set(rng, kind, max(mp - 1, 1), 0.0, 0.0)
        g.add(f"{func}_too_few", func, s[: max(mp - 1, 0)], d[: max(mp - 1, 0)], RANSAC, refine=default_refine)

    # Nadir drone frames: near-similarity transforms with many matches.
    for n in (300, 1000):
        s, d = make_set(rng, "similarity", n, 0.4, 0.35)
        for func in ("affine", "partial", "homography"):
            for mname, method in (("ransac", RANSAC), ("lmeds", LMEDS), ("rho", RHO)):
                if method == RHO and func != "homography":
                    continue
                g.add(f"drone_{func}_{mname}_{n}", func, s, d, method, refine=10)
    # RHO on sets sorted by match quality (PROSAC order) and on large sets.
    for n in (100, 2000):
        s, d = make_set(rng, "homography", n, 0.5, 0.0)
        k = int(n * 0.4)
        d[n - k:] = rng.uniform(-100, 1100, (k, 2)).astype(np.float32)
        g.add(f"homography_rho_sorted_{n}", "homography", s, d, RHO)
    for n in (5, 6, 8, 12, 21, 22, 30):
        s, d = make_set(rng, "homography", n, 0.3, 0.2 if n > 8 else 0.0)
        g.add(f"homography_rho_small_{n}", "homography", s, d, RHO)
    # Unsupported method values.
    s, d = make_set(rng, "affine", 30, 0.0, 0.0)
    g.add("partial_usac_error", "partial", s, d, cv2.USAC_DEFAULT)
    g.add("affine_rho_error", "affine", s, d, RHO)


def main():
    rng = np.random.default_rng(20261016)
    g = Gen()
    battery(g, rng)

    s = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float32)
    h, _ = cv2.findHomography(s, s * 2)
    assert abs(h[0, 0] - 2) < 1e-9
    assert set(SUSPECT_CASES) <= g.names, set(SUSPECT_CASES) - g.names

    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "estimation_golden.json")
    with open(out, "w") as f:
        json.dump({"cases": g.cases}, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases to {out}")


if __name__ == "__main__":
    main()
