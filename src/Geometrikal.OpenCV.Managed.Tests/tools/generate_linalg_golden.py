#!/usr/bin/env python3
"""Golden-data generator for the managed linear algebra port.

Runs cv2 5.0.0 ``gemm``, ``determinant``, ``invert``, ``solve``, ``eigen``,
``SVDecomp``, ``SVBackSubst``, ``solveCubic``, ``solvePoly``, ``transform``,
``perspectiveTransform``, ``mulTransposed`` and ``trace`` over a seeded battery
and writes ``TestData/linalg_golden.json``. ``Core/LinearAlgebraGoldenTests.cs``
replays each case.

Required environment (the script stops if cv2 is not 5.0.0):
 * cv2 5.0.0 built with the internal LAPACK (``Lapack: YES (Built-In
   libclapack)`` in the build information). Run with ``PYTHONPATH=<dir>``.
 * OPENCV_CPU_DISABLE=AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL
   before cv2 is imported (the script sets it when it is not set). The
   transform kernels then run the 128-bit SSE4.1 dispatch, which the port
   reproduces.
 * IPP, OpenCL and threads are disabled in the script.

Regenerate with (from the ``Geometrikal.OpenCV.Managed.Tests`` directory):
    PYTHONPATH=<cv2 5.0.0 dir> python tools/generate_linalg_golden.py

Battery notes:
 * Every case has a ``tolerance``. 0 means a byte-exact comparison. A
   positive tolerance is relative to the largest magnitude of the expected
   array. Positive tolerances are used only where cv2 calls LAPACK
   (``gesdd`` for SVD when the larger dimension is 25 or more, ``geqrf``-based
   QR when the row count is 30 or more), for the transcendental functions of
   solveCubic/solvePoly, and for the eigenvector signs below.
 * ``lapackSvd`` cases compare only the singular values (cv2 ``gesdd``
   vectors can differ in sign and order from the Jacobi vectors); the runner
   checks the managed vectors by reconstruction.
 * Arrays are stored once in a shared ``arrays`` list.

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
    sys.exit(f"generate_linalg_golden.py requires cv2 5.0.0 (found {cv2.__version__})")
if cv2.checkHardwareSupport(11) or cv2.checkHardwareSupport(13):
    sys.exit("AVX2/AVX-512 dispatch is enabled; set OPENCV_CPU_DISABLE before importing cv2")
if "Built-In libclapack" not in cv2.getBuildInformation():
    sys.exit("cv2 must be built with the internal LAPACK (Lapack: YES (Built-In libclapack))")

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

DTYPES = {"f32": np.float32, "f64": np.float64}
FT = {"f32": np.float32, "f64": np.float64}

# Adjudicated differences. cv2 solves a square system of 30 or more rows with DECOMP_QR through the
# built-in LAPACK HAL (lapack_QR), which returns a wrong solution (residual far above the LU
# solution). The port runs the OpenCV QRImpl, which gives the least-squares solution. The port
# behaviour is pinned in LinearAlgebraBasicTests.Solve_Qr_LargeSquare_MatchesLu.
SUSPECT_CASES = {
    f"solve_{dt}_qr_50_nb1": "cv2 lapack_QR returns a wrong solution for square systems with 30 or more rows"
    for dt in ("f32", "f64")
}


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
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        m = {"dtype": dt_of(arr), "width": int(arr.shape[1]), "height": int(arr.shape[0]),
             "channels": 1 if arr.ndim == 2 else int(arr.shape[2]), "data": b64(arr)}
        key = (m["dtype"], m["width"], m["height"], m["channels"], m["data"])
        if key not in self.index:
            self.index[key] = len(self.arrays)
            self.arrays.append(m)
        return self.index[key]

    def add(self, name, op, inputs, params, outputs=None, values=None, tol=0.0):
        assert name not in self.names, name
        self.names.add(name)
        case = {"name": name, "op": op, "inputs": [self.ref(a) for a in inputs], "params": params,
                "suspect": name in SUSPECT_CASES, "tolerance": tol,
                "outputs": {k: self.ref(v) for k, v in (outputs or {}).items()},
                "values": [float(v) for v in (values or [])]}
        if case["suspect"]:
            case["suspectReason"] = SUSPECT_CASES[name]
        self.cases.append(case)


def rnd(rng, dt, rows, cols, lo=-1.0, hi=1.0):
    return rng.uniform(lo, hi, (rows, cols)).astype(FT[dt])


def spd(rng, dt, n):
    m = rng.uniform(-1, 1, (n, n))
    return (m @ m.T + n * np.eye(n)).astype(FT[dt])


def sym(rng, dt, n):
    m = rng.uniform(-1, 1, (n, n))
    return ((m + m.T) * 0.5).astype(FT[dt])


def gemm_cases(g, rng):
    shapes = [(2, 2, 2), (3, 3, 3), (4, 4, 4), (3, 3, 1), (4, 4, 20), (20, 4, 4), (9, 9, 9), (8, 9, 1),
              (5, 3, 7), (7, 4, 20), (1, 6, 5), (6, 1, 5), (9, 1, 1), (3, 2, 9), (50, 50, 50), (33, 70, 5),
              (70, 200, 130), (104, 40, 36), (100, 3, 3)]
    for dt in ("f32", "f64"):
        for (dh, ln, dw) in shapes:
            for flags in (0, 1, 2, 3):
                if (dh, ln, dw) == (70, 200, 130) and flags in (1, 3):
                    continue
                a_shape = (ln, dh) if flags & 1 else (dh, ln)
                b_shape = (dw, ln) if flags & 2 else (ln, dw)
                a = rnd(rng, dt, *a_shape)
                b = rnd(rng, dt, *b_shape)
                name = f"gemm_{dt}_{dh}x{ln}x{dw}_f{flags}"
                d = cv2.gemm(a, b, 1.0, None, 0.0, flags=flags)
                g.add(name, "gemm", [a, b], {"flags": flags, "alpha": 1.0, "beta": 0.0}, {"d": d})
                if dh * dw <= 400 or ((dh, ln, dw) == (104, 40, 36) and flags in (0, 3)):
                    for ctrans in (0, 4):
                        c = rnd(rng, dt, dw, dh) if ctrans else rnd(rng, dt, dh, dw)
                        alpha, beta = 0.7, -1.3
                        d = cv2.gemm(a, b, alpha, c, beta, flags=flags | ctrans)
                        g.add(f"{name}_c{ctrans}", "gemm", [a, b, c],
                              {"flags": flags | ctrans, "alpha": alpha, "beta": beta}, {"d": d})
        # Absent C with a non-zero beta, and a zero alpha.
        for rows in (4, 120):
            a = rnd(rng, dt, rows, 4)
            b = rnd(rng, dt, 4, 5)
            c = rnd(rng, dt, rows, 5)
            g.add(f"gemm_{dt}_{rows}_alpha0", "gemm", [a, b, c], {"flags": 0, "alpha": 0.0, "beta": 2.0},
                  {"d": cv2.gemm(a, b, 0.0, c, 2.0)})
            g.add(f"gemm_{dt}_{rows}_noc_beta", "gemm", [a, b], {"flags": 0, "alpha": 1.5, "beta": -2.0},
                  {"d": cv2.gemm(a, b, 1.5, None, -2.0)})


def det_cases(g, rng):
    for dt in ("f32", "f64"):
        for n in (1, 2, 3, 4, 5, 6, 9, 12, 50):
            a = rnd(rng, dt, n, n)
            g.add(f"det_{dt}_{n}", "det", [a], {}, values=[cv2.determinant(a)])
        s = rnd(rng, dt, 4, 4)
        s[3] = s[0] + s[1]
        g.add(f"det_{dt}_singular4", "det", [s], {}, values=[cv2.determinant(s)])
        s3 = rnd(rng, dt, 3, 3)
        s3[2] = s3[0] * 2
        g.add(f"det_{dt}_singular3", "det", [s3], {}, values=[cv2.determinant(s3)])


METHODS = {"lu": cv2.DECOMP_LU, "chol": cv2.DECOMP_CHOLESKY, "svd": cv2.DECOMP_SVD, "eig": cv2.DECOMP_EIG,
           "qr": cv2.DECOMP_QR}


def invert_cases(g, rng):
    for dt in ("f32", "f64"):
        for n in (1, 2, 3, 4, 6, 9, 50):
            for mname in ("lu", "chol", "svd", "eig"):
                if mname in ("chol", "eig"):
                    a = spd(rng, dt, n) if mname == "chol" else sym(rng, dt, n)
                else:
                    a = rnd(rng, dt, n, n)
                ret, inv = cv2.invert(a, flags=METHODS[mname])
                tol = 0.0
                if mname == "svd" and n >= 25:
                    tol = 1e-3 if dt == "f32" else 1e-9
                g.add(f"inv_{dt}_{mname}_{n}", "invert", [a], {"method": METHODS[mname]}, {"inv": inv},
                      values=[ret], tol=tol)
        for (r, c) in ((5, 3), (3, 5), (12, 9)):
            a = rnd(rng, dt, r, c)
            ret, inv = cv2.invert(a, flags=cv2.DECOMP_SVD)
            g.add(f"inv_{dt}_svd_{r}x{c}", "invert", [a], {"method": cv2.DECOMP_SVD}, {"inv": inv}, values=[ret])
        s = rnd(rng, dt, 5, 5)
        s[4] = s[0] - s[2]
        for mname in ("lu", "svd"):
            ret, inv = cv2.invert(s, flags=METHODS[mname])
            g.add(f"inv_{dt}_{mname}_singular5", "invert", [s], {"method": METHODS[mname]}, {"inv": inv},
                  values=[ret])
        s2 = np.array([[1, 2], [2, 4]], FT[dt])
        ret, inv = cv2.invert(s2, flags=cv2.DECOMP_LU)
        g.add(f"inv_{dt}_lu_singular2", "invert", [s2], {"method": cv2.DECOMP_LU}, {"inv": inv}, values=[ret])


def solve_cases(g, rng):
    for dt in ("f32", "f64"):
        for n in (1, 2, 3, 4, 6, 9, 50):
            for nb in (1, 3):
                for mname in ("lu", "chol", "qr", "svd", "eig"):
                    if n == 50 and nb == 3:
                        continue
                    if mname == "chol":
                        a = spd(rng, dt, n)
                    elif mname == "eig":
                        a = sym(rng, dt, n) + np.eye(n, dtype=FT[dt]) * 3
                    else:
                        a = rnd(rng, dt, n, n)
                    b = rnd(rng, dt, n, nb)
                    ok, x = cv2.solve(a, b, flags=METHODS[mname])
                    tol = 0.0
                    if (mname == "svd" and n >= 25) or (mname == "qr" and n >= 30):
                        tol = 2e-3 if dt == "f32" else 1e-8
                    g.add(f"solve_{dt}_{mname}_{n}_nb{nb}", "solve", [a, b], {"flags": METHODS[mname]}, {"x": x},
                          values=[1.0 if ok else 0.0], tol=tol)
        for (m, n) in ((12, 5), (40, 6), (9, 8)):
            for nb in (1, 2):
                a = rnd(rng, dt, m, n)
                b = rnd(rng, dt, m, nb)
                for mname, flags in (("qr", cv2.DECOMP_QR), ("svd", cv2.DECOMP_SVD),
                                     ("lu_n", cv2.DECOMP_LU | cv2.DECOMP_NORMAL),
                                     ("chol_n", cv2.DECOMP_CHOLESKY | cv2.DECOMP_NORMAL),
                                     ("qr_n", cv2.DECOMP_QR | cv2.DECOMP_NORMAL),
                                     ("svd_n", cv2.DECOMP_SVD | cv2.DECOMP_NORMAL)):
                    ok, x = cv2.solve(a, b, flags=flags)
                    tol = 0.0
                    if (mname == "svd" and m >= 25) or (mname == "qr" and m >= 30):
                        tol = 2e-3 if dt == "f32" else 1e-8
                    g.add(f"solve_{dt}_{mname}_{m}x{n}_nb{nb}", "solve", [a, b], {"flags": flags}, {"x": x},
                          values=[1.0 if ok else 0.0], tol=tol)
        s = rnd(rng, dt, 5, 5)
        s[4] = s[1] * 3
        b = rnd(rng, dt, 5, 1)
        for mname in ("lu", "qr", "svd"):
            ok, x = cv2.solve(s, b, flags=METHODS[mname])
            g.add(f"solve_{dt}_{mname}_singular5", "solve", [s, b], {"flags": METHODS[mname]}, {"x": x},
                  values=[1.0 if ok else 0.0])
        # Homography DLT normal equations: 8 x 8 system.
        a = rnd(rng, dt, 8, 8, -100, 100)
        b = rnd(rng, dt, 8, 1, -100, 100)
        for mname in ("lu", "svd"):
            ok, x = cv2.solve(a, b, flags=METHODS[mname])
            g.add(f"solve_{dt}_{mname}_8_scaled", "solve", [a, b], {"flags": METHODS[mname]}, {"x": x},
                  values=[1.0 if ok else 0.0])


def eigen_cases(g, rng):
    for dt in ("f32", "f64"):
        for n in (1, 2, 3, 5, 9, 50):
            a = sym(rng, dt, n)
            ok, w, v = cv2.eigen(a)
            assert ok
            g.add(f"eigen_{dt}_{n}", "eigen", [a], {}, {"w": w, "v": v})
        a = np.diag(np.array([3, 1, 2, 2], FT[dt]))
        ok, w, v = cv2.eigen(a)
        g.add(f"eigen_{dt}_diag", "eigen", [a], {}, {"w": w, "v": v})


def svd_cases(g, rng):
    for dt in ("f32", "f64"):
        shapes = [(2, 2), (3, 3), (5, 5), (9, 9), (7, 4), (4, 7), (8, 9), (1, 5), (5, 1)]
        for (r, c) in shapes:
            a = rnd(rng, dt, r, c)
            for fname, flags in (("none", 0), ("nouv", cv2.SVD_NO_UV), ("full", cv2.SVD_FULL_UV)):
                w, u, vt = cv2.SVDecomp(a, flags=flags)
                outs = {"w": w} if flags == cv2.SVD_NO_UV else {"w": w, "u": u, "vt": vt}
                g.add(f"svd_{dt}_{r}x{c}_{fname}", "svd", [a], {"flags": flags}, outs)
        # Rank-deficient: zero singular values take the random-vector branch.
        m = rnd(rng, dt, 6, 3)
        a = (m @ m.T).astype(FT[dt])
        for fname, flags in (("none", 0), ("full", cv2.SVD_FULL_UV)):
            w, u, vt = cv2.SVDecomp(a, flags=flags)
            g.add(f"svd_{dt}_rank3_{fname}", "svd", [a], {"flags": flags}, {"w": w, "u": u, "vt": vt})
        z = np.zeros((4, 3), FT[dt])
        w, u, vt = cv2.SVDecomp(z, flags=cv2.SVD_FULL_UV)
        g.add(f"svd_{dt}_zero_full", "svd", [z], {"flags": cv2.SVD_FULL_UV}, {"w": w, "u": u, "vt": vt})
        # LAPACK gesdd in cv2 (larger dimension >= 25).
        for (r, c) in ((30, 20), (20, 30)):
            a = rnd(rng, dt, r, c)
            w, u, vt = cv2.SVDecomp(a)
            g.add(f"svd_{dt}_{r}x{c}_lapack", "svd", [a], {"flags": 0, "lapackSvd": True}, {"w": w},
                  tol=1e-4 if dt == "f32" else 1e-12)
        # Back substitution with the cv2 decomposition.
        a = rnd(rng, dt, 7, 4)
        w, u, vt = cv2.SVDecomp(a)
        rhs = rnd(rng, dt, 7, 2)
        g.add(f"svbksb_{dt}_rhs", "svbksb", [w, u, vt, rhs], {}, {"x": cv2.SVBackSubst(w, u, vt, rhs)})
        rhs1 = rnd(rng, dt, 7, 1)
        g.add(f"svbksb_{dt}_rhs1", "svbksb", [w, u, vt, rhs1], {}, {"x": cv2.SVBackSubst(w, u, vt, rhs1)})
        # solveZ: last row of Vt, as cv::SVD::solveZ.
        for (r, c) in ((8, 9), (9, 9), (5, 3)):
            a = rnd(rng, dt, r, c)
            w, u, vt = cv2.SVDecomp(a, flags=0 if r >= c else cv2.SVD_FULL_UV)
            g.add(f"solvez_{dt}_{r}x{c}", "solvez", [a], {}, {"x": vt[-1].reshape(-1, 1).copy()})


def poly_cases(g, rng):
    cubic = [[1, -6, 11, -6], [2, -4, -22, 24], [1, 0, 0, -1], [1, -3, 3, -1], [0, 1, -3, 2], [0, 0, 2, -4],
             [0, 0, 0, 1], [0, 0, 0, 0], [-6, 11, -6], [1e-20, 1, -3, 2], [3, -2, 5, 7]]
    for i, c in enumerate(cubic):
        for dt in ("f32", "f64"):
            co = np.array(c, FT[dt]).reshape(1, -1)
            n, roots = cv2.solveCubic(co)
            g.add(f"cubic_{dt}_{i}", "cubic", [co], {}, {"roots": roots}, values=[n],
                  tol=1e-6 if dt == "f32" else 1e-13)
    polys = [[-6, 11, -6, 1], [2, 0, 1], [1, 0, 0, 0, 1], [-1, 0, 0, 0, 0, 1], [24, -50, 35, -10, 1],
             [1, 2, 3, 4, 5, 6, 7], [0.5, -1.5, 1]]
    for i, c in enumerate(polys):
        co = np.array(c, np.float64).reshape(1, -1)
        md, roots = cv2.solvePoly(co)
        g.add(f"poly_f64_{i}", "poly", [co], {"maxIters": 300}, {"roots": roots.reshape(-1, 2).copy()},
              values=[md], tol=1e-9)


def transform_cases(g, rng):
    for dt in ("f32", "f64"):
        h = np.array([[1.1, 0.2, 5.0], [-0.1, 0.9, -3.0], [1e-3, -2e-3, 1.0]], np.float64)
        for n in (1, 7, 8):
            p = rnd(rng, dt, n, 2, -50, 50).reshape(n, 1, 2)
            g.add(f"persp_{dt}_2d_{n}", "persp", [p, h], {}, {"d": cv2.perspectiveTransform(p, h)})
        h4 = np.array([[1, 0.1, 0.2, 1], [0.3, 1, 0, 2], [0, 0.1, 1, 3], [0.01, 0.02, 0.03, 1]], np.float64)
        p3 = rnd(rng, dt, 5, 3, -10, 10).reshape(5, 1, 3)
        g.add(f"persp_{dt}_3d", "persp", [p3, h4], {}, {"d": cv2.perspectiveTransform(p3, h4)})
        h32 = h4[1:].copy()
        g.add(f"persp_{dt}_3to2", "persp", [p3, h32], {}, {"d": cv2.perspectiveTransform(p3, h32)})
        h23 = np.array([[1, 0.5, 2], [0.3, 1, -1], [0.2, 0.1, 1], [0.01, 0.02, 1]], np.float64)
        p2 = rnd(rng, dt, 6, 2, -5, 5).reshape(6, 1, 2)
        g.add(f"persp_{dt}_2to3", "persp", [p2, h23], {}, {"d": cv2.perspectiveTransform(p2, h23)})
        pz = np.array([[[100, 0]], [[1, 1]]], FT[dt])
        hz = np.array([[1, 0, 0], [0, 1, 0], [0.01, 0, -1]], np.float64)
        g.add(f"persp_{dt}_wzero", "persp", [pz, hz], {}, {"d": cv2.perspectiveTransform(pz, hz)})
        img = rnd(rng, dt, 5 * 4, 2, -20, 20).reshape(5, 4, 2)
        g.add(f"persp_{dt}_mat", "persp", [img, h], {}, {"d": cv2.perspectiveTransform(img, h)})

        mats = {
            "2to2": (2, np.array([[0.9, -0.2, 3.0], [0.25, 1.1, -2.0]])),
            "3to3": (3, np.array([[0.5, 0.25, 0.125, 1], [0.1, 0.7, 0.3, -2], [0.3, 0.2, 0.9, 0.5]])),
            "3to3_noofs": (3, np.array([[0.5, 0.25, 0.125], [0.1, 0.7, 0.3], [0.3, 0.2, 0.9]])),
            "3to3_diag": (3, np.array([[0.5, 0, 0, 1], [0, 0.7, 0, -2], [0, 0, 0.9, 0.5]])),
            "3to1": (3, np.array([[0.299, 0.587, 0.114, 0.5]])),
            "4to4": (4, np.array([[1, 0.1, 0.2, 0.3, 1], [0.4, 1, 0.5, 0.6, 2], [0.7, 0.8, 1, 0.9, 3],
                                  [0.11, 0.12, 0.13, 1, 4]])),
            "2to3": (2, np.array([[1, 0.5, 2], [0.3, 1, -1], [0.2, 0.1, 1]])),
            "2to2_diag": (2, np.array([[2.0, 0, 1], [0, 3.0, -1]])),
        }
        for key, (scn, m) in mats.items():
            for n in (1, 8, 9):
                src = rnd(rng, dt, n, scn, -30, 30).reshape(n, 1, scn)
                d = cv2.transform(src, m)
                if d.ndim == 2:
                    d = d.reshape(n, 1, -1)
                g.add(f"transform_{dt}_{key}_{n}", "transform", [src, m], {}, {"d": d})
        img = rnd(rng, dt, 3 * 5, 3, -30, 30).reshape(3, 5, 3)
        g.add(f"transform_{dt}_mat3", "transform", [img, mats["3to3"][1]], {},
              {"d": cv2.transform(img, mats["3to3"][1])})


def misc_cases(g, rng):
    for dt in ("f32", "f64"):
        a = rnd(rng, dt, 7, 5)
        for ata in (True, False):
            for scale in (1.0, 0.5):
                d = cv2.mulTransposed(a, ata, scale=scale)
                g.add(f"multrans_{dt}_{int(ata)}_{scale}", "multrans", [a], {"aTa": ata, "scale": scale}, {"d": d})
        a = rnd(rng, dt, 5, 4)
        g.add(f"trace_{dt}", "trace", [a], {}, values=[cv2.trace(a)[0]])


def main():
    rng = np.random.default_rng(20261015)
    g = Gen()
    gemm_cases(g, rng)
    det_cases(g, rng)
    invert_cases(g, rng)
    solve_cases(g, rng)
    eigen_cases(g, rng)
    svd_cases(g, rng)
    poly_cases(g, rng)
    transform_cases(g, rng)
    misc_cases(g, rng)

    a = np.array([[2, 0], [0, 4]], np.float64)
    assert abs(cv2.determinant(a) - 8) < 1e-12
    assert set(SUSPECT_CASES) <= g.names, set(SUSPECT_CASES) - g.names

    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "linalg_golden.json")
    with open(out, "w") as f:
        json.dump({"arrays": g.arrays, "cases": g.cases}, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases to {out}")


if __name__ == "__main__":
    main()
