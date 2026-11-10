#!/usr/bin/env python3
"""Golden-data generator for the managed two-view (epipolar) geometry port.

Runs cv2 5.0.0 ``findEssentialMat``, ``recoverPose``, ``triangulatePoints``,
``decomposeEssentialMat``, ``findFundamentalMat``, ``computeCorrespondEpilines``,
``convertPointsToHomogeneous`` and ``convertPointsFromHomogeneous`` over a
seeded battery and writes ``TestData/epipolar_golden.json``.
``Calib/EpipolarGoldenTests.cs`` replays each case.

Required environment (the script stops if cv2 is not 5.0.0):
 * cv2 5.0.0 built with the internal LAPACK. Run with ``PYTHONPATH=<dir>``.
 * OPENCV_CPU_DISABLE=AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL
   before cv2 is imported (the script sets it when it is not set).
 * IPP, OpenCL and threads are disabled in the script.

Regenerate with (from the ``Geometrikal.OpenCV.Managed.Tests`` directory):
    PYTHONPATH=<cv2 5.0.0 dir> python tools/generate_epipolar_golden.py

Scenes: synthetic 3-D points seen by two pinhole cameras (f = 1000 px,
1920 x 1080). ``nadir``: a drone at about 100 m looking down, terrain relief
of +-10 m, a 10 m baseline. ``oblique``: depths 20-200 m, a 15 degree view
change. ``planar``: the nadir scene with all points on one plane (degenerate
for the fundamental matrix). Noise is Gaussian in pixels; outliers are
uniform random image points in the second view.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import json
import os
import sys

os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

if cv2.__version__ != "5.0.0":
    sys.exit(f"generate_epipolar_golden.py requires cv2 5.0.0 (found {cv2.__version__})")
if cv2.checkHardwareSupport(11) or cv2.checkHardwareSupport(13):
    sys.exit("AVX2/AVX-512 dispatch is enabled; set OPENCV_CPU_DISABLE before importing cv2")
if "Built-In libclapack" not in cv2.getBuildInformation():
    sys.exit("cv2 must be built with the internal LAPACK (Lapack: YES (Built-In libclapack))")

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

# Adjudicated differences (case name -> reason).
SUSPECT_CASES = {}

K = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1]])


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def scene(rng, kind, n, noise, outliers, dtype):
    if kind == "oblique":
        z = rng.uniform(20, 200, n)
        rvec = np.array([0.05, 0.26, 0.03])
        t = np.array([-8.0, 1.0, 2.0])
    else:
        z = 100.0 + (0 if kind == "planar" else rng.uniform(-10, 10, n))
        rvec = np.array([0.01, -0.02, 0.03])
        t = np.array([-10.0, 0.5, 0.3])
    u = rng.uniform(0, 1920, n)
    v = rng.uniform(0, 1080, n)
    X = np.stack([(u - 960) / 1000 * z, (v - 540) / 1000 * z, z * np.ones(n)], 1)
    R, _ = cv2.Rodrigues(rvec)
    X2 = X @ R.T + t
    p1 = np.stack([u, v], 1)
    p2 = X2[:, :2] / X2[:, 2:3] * 1000 + np.array([960, 540])
    p1 = p1 + rng.normal(0, noise, p1.shape)
    p2 = p2 + rng.normal(0, noise, p2.shape)
    k = int(round(n * outliers))
    if k:
        idx = rng.choice(n, k, replace=False)
        p2[idx] = np.stack([rng.uniform(0, 1920, k), rng.uniform(0, 1080, k)], 1)
    return p1.astype(dtype), p2.astype(dtype), R, t


def tag(arr):
    return "f32" if arr.dtype == np.float32 else "f64"


class Golden:
    def __init__(self):
        self.cases = []

    def add(self, case):
        case["suspect"] = case["name"] in SUSPECT_CASES
        self.cases.append(case)


def essential(g, name, p1, p2, method, prob, thr, max_iters, focal_pp=False):
    def call():
        try:
            if focal_pp:
                return cv2.findEssentialMat(p1, p2, focal=1000.0, pp=(960.0, 540.0), method=method,
                                            prob=prob, threshold=thr, maxIters=max_iters)
            return cv2.findEssentialMat(p1, p2, K, method=method, prob=prob, threshold=thr, maxIters=max_iters)
        except cv2.error:
            return None, None
    e, mask = call()
    e2, mask2 = call()
    assert (e is None) == (e2 is None) and (e is None or np.array_equal(e, e2))
    case = {"name": name, "kind": "essential", "dtype": tag(p1), "n": len(p1), "p1": b64(p1), "p2": b64(p2),
            "method": int(method), "prob": prob, "thr": thr, "maxIters": max_iters, "focalPp": focal_pp}
    if e is not None and e.size:
        case["model"] = b64(np.asarray(e, np.float64))
        case["rows"] = int(e.shape[0])
        case["mask"] = b64(mask.ravel().astype(np.uint8))
    g.add(case)
    return e, mask


def recover(g, name, E, p1, p2, mask, dist):
    m = None if mask is None else mask.copy()
    if dist is None:
        ret, R, t, mo = cv2.recoverPose(E, p1, p2, K, mask=m)
        tri = None
    else:
        ret, R, t, mo, tri = cv2.recoverPose(E, p1, p2, K, distanceThresh=dist, mask=m)
    case = {"name": name, "kind": "recover", "dtype": tag(p1), "n": len(p1), "p1": b64(p1), "p2": b64(p2),
            "E": b64(np.asarray(E, np.float64)), "dist": dist, "ret": int(ret), "R": b64(R), "t": b64(t),
            "outMask": b64(mo.ravel().astype(np.uint8))}
    if mask is not None:
        case["inMask"] = b64(mask.ravel().astype(np.uint8))
    if tri is not None:
        case["tri"] = b64(np.asarray(tri, np.float64))
    g.add(case)


def fundamental(g, name, p1, p2, method, thr=3.0, conf=0.99, max_iters=1000):
    try:
        f, mask = cv2.findFundamentalMat(p1, p2, method, thr, conf, max_iters)
    except cv2.error:
        f, mask = None, None
    case = {"name": name, "kind": "fundamental", "dtype": tag(p1), "n": len(p1), "p1": b64(p1), "p2": b64(p2),
            "method": int(method), "thr": thr, "conf": conf, "maxIters": max_iters}
    if f is not None and f.size:
        case["model"] = b64(np.asarray(f, np.float64))
        case["rows"] = int(f.shape[0])
        case["mask"] = b64(mask.ravel().astype(np.uint8))
    g.add(case)
    return f


def battery(g, rng):
    # Essential matrix and pose.
    for kind in ("nadir", "oblique", "planar"):
        for n, outl in ((5, 0.0), (6, 0.0), (10, 0.0), (50, 0.0), (200, 0.3), (1000, 0.3), (300, 0.5)):
            for method, mname in ((cv2.RANSAC, "ransac"), (cv2.LMEDS, "lmeds")):
                if method == cv2.LMEDS and outl >= 0.5:
                    continue
                dtype = np.float64 if (n % 2 == 0) else np.float32
                p1, p2, _, _ = scene(rng, kind, n, 0.5, outl, dtype)
                nm = f"em_{kind}_{n}_{int(outl * 100)}_{mname}"
                e, mask = essential(g, nm, p1, p2, method, 0.999, 1.0, 1000)
                if e is not None and e.shape == (3, 3) and n >= 10:
                    recover(g, nm + "_pose", e, p1, p2, mask, None)
                    recover(g, nm + "_pose_nomask", e, p1, p2, None, None)
                    recover(g, nm + "_pose_d1000", e, p1, p2, mask, 1000.0)
                    recover(g, nm + "_pose_d5", e, p1, p2, None, 5.0)
    p1, p2, _, _ = scene(rng, "nadir", 400, 1.0, 0.2, np.float32)
    essential(g, "em_focalpp", p1, p2, cv2.RANSAC, 0.999, 1.0, 1000, focal_pp=True)
    for thr in (0.3, 3.0):
        essential(g, f"em_thr_{thr}", p1, p2, cv2.RANSAC, 0.999, thr, 1000)
    essential(g, "em_maxiters_20", p1, p2, cv2.RANSAC, 0.99, 1.0, 20)
    essential(g, "em_prob_0.9", p1, p2, cv2.LMEDS, 0.9, 1.0, 1000)
    essential(g, "em_too_few", p1[:4], p2[:4], cv2.RANSAC, 0.999, 1.0, 1000)

    # Decomposition.
    for i in range(6):
        p1, p2, _, _ = scene(rng, ("nadir", "oblique")[i % 2], 60, 0.3, 0.0, np.float64)
        e, _ = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        r1, r2, t = cv2.decomposeEssentialMat(e[:3])
        g.add({"name": f"decompose_{i}", "kind": "decompose", "E": b64(e[:3]), "R1": b64(r1), "R2": b64(r2),
               "t": b64(t)})

    # Triangulation.
    for kind in ("nadir", "oblique"):
        for dtype in (np.float32, np.float64):
            p1, p2, R, t = scene(rng, kind, 100, 0.5, 0.0, dtype)
            P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = K @ np.hstack([R, t.reshape(3, 1)])
            X = cv2.triangulatePoints(P1, P2, p1.T.copy(), p2.T.copy())
            g.add({"name": f"tri_{kind}_{tag(p1)}", "kind": "triangulate", "dtype": tag(p1), "n": len(p1),
                   "p1": b64(p1), "p2": b64(p2), "P1": b64(P1), "P2": b64(P2), "X": b64(np.asarray(X, np.float64))})

    # Fundamental matrix.
    FM = {"7point": cv2.FM_7POINT, "8point": cv2.FM_8POINT, "ransac": cv2.FM_RANSAC, "lmeds": cv2.FM_LMEDS}
    for kind in ("nadir", "oblique", "planar"):
        for n, outl in ((7, 0.0), (8, 0.0), (10, 0.0), (14, 0.0), (15, 0.0), (50, 0.1), (500, 0.3), (2000, 0.3)):
            for mname, method in FM.items():
                if mname == "7point" and n > 8:
                    continue
                if mname == "8point" and n > 500:
                    continue
                dtype = np.float32 if n % 2 == 0 else np.float64
                p1, p2, _, _ = scene(rng, kind, n, 0.5, outl, dtype)
                f = fundamental(g, f"fm_{kind}_{n}_{int(outl * 100)}_{mname}", p1, p2, method)
                if f is not None and f.shape == (3, 3) and mname == "ransac" and n >= 50:
                    for which in (1, 2):
                        pts = p1 if which == 1 else p2
                        lines = cv2.computeCorrespondEpilines(pts.reshape(-1, 1, 2), which, f).reshape(-1, 3)
                        g.add({"name": f"epilines_{kind}_{n}_{which}", "kind": "epilines", "dtype": tag(pts),
                               "n": len(pts), "pts": b64(pts), "which": which, "F": b64(f),
                               "lines": b64(lines)})
    p1, p2, _, _ = scene(rng, "oblique", 300, 1.0, 0.3, np.float32)
    for thr in (0.5, 1.0, -1.0):
        fundamental(g, f"fm_thr_{thr}", p1, p2, cv2.FM_RANSAC, thr=thr)
    fundamental(g, "fm_conf_out_of_range", p1, p2, cv2.FM_RANSAC, conf=1.5)
    fundamental(g, "fm_maxiters_10", p1, p2, cv2.FM_RANSAC, max_iters=10)
    fundamental(g, "fm_too_few", p1[:6], p2[:6], cv2.FM_RANSAC)

    # Homogeneous conversions.
    for dtype in (np.float32, np.float64):
        pts3 = rng.uniform(-10, 10, (50, 3)).astype(dtype)
        pts3[::7, 2] = 0
        pts3[3, 2] = dtype(1e-9)
        from_h = cv2.convertPointsFromHomogeneous(pts3).reshape(-1, 2)
        pts2 = rng.uniform(-10, 10, (20, 2)).astype(dtype)
        to_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1, 3)
        g.add({"name": f"homog_{tag(pts3)}", "kind": "homogeneous", "dtype": tag(pts3), "pts3": b64(pts3),
               "from": b64(from_h), "pts2": b64(pts2), "to": b64(to_h)})


def main():
    g = Golden()
    battery(g, np.random.default_rng(20261016))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TestData", "epipolar_golden.json")
    with open(out, "w") as f:
        json.dump({"cases": g.cases}, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases to {out}")


if __name__ == "__main__":
    main()
