#!/usr/bin/env python3
"""Golden-data generator for the managed PnP solver port.

Runs cv2 5.0.0 ``solvePnP`` (all non-USAC methods), ``solvePnPGeneric`` (with the
reprojection errors), ``solveP3P``, ``solvePnPRansac`` (classic RANSAC),
``solvePnPRefineLM`` and ``solvePnPRefineVVS`` over a seeded battery and writes
``TestData/pnp_golden.json``. ``Calib/PnpGoldenTests.cs`` replays each case.

Required environment (the script stops if cv2 is not 5.0.0):
 * cv2 5.0.0. Run with ``PYTHONPATH=<dir>``.
 * OPENCV_CPU_DISABLE=AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL
   before cv2 is imported (the script sets it when it is not set).
 * IPP, OpenCL and threads are disabled in the script.

Regenerate with (from the ``Geometrikal.OpenCV.Managed.Tests`` directory):
    PYTHONPATH=<cv2 5.0.0 dir> python tools/generate_pnp_golden.py

Battery notes:
 * Scenes look like ground control points seen from a drone: a DJI FC6310
   camera (5472 x 3648, 8.8 mm on 13.2 mm) with an off-centre principal point,
   flat ("planar", Z = 0) and relief (Z in [-4, 4] m) point sets within
   +-60 m, nadir and oblique poses at 60-120 m.
 * Distortion sets: none, k5 and k8 (rational).
 * Image noise 0 or 0.5 px; RANSAC sets have 10 %, 30 % and 45 % outliers.
 * Numbers are JSON doubles (Python repr, exact round trip); NaN and infinities
   are the strings "nan", "inf" and "-inf". A ``float``
   input is stored as the double value of the float.
 * Every case runs twice and the script asserts equal outputs.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import json
import os
import sys

os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

if cv2.__version__ != "5.0.0":
    sys.exit(f"generate_pnp_golden.py requires cv2 5.0.0 (found {cv2.__version__})")
if cv2.checkHardwareSupport(11) or cv2.checkHardwareSupport(13):
    sys.exit("AVX2/AVX-512 dispatch is enabled; set OPENCV_CPU_DISABLE before importing cv2")

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

METHODS = {"iterative": cv2.SOLVEPNP_ITERATIVE, "epnp": cv2.SOLVEPNP_EPNP, "p3p": cv2.SOLVEPNP_P3P,
           "ap3p": cv2.SOLVEPNP_AP3P, "ippe": cv2.SOLVEPNP_IPPE, "ippe_square": cv2.SOLVEPNP_IPPE_SQUARE,
           "sqpnp": cv2.SOLVEPNP_SQPNP}

DIST = {
    "none": [],
    "k5": [-0.2713, 0.1125, 0.00071, -0.00118, -0.0315],
    "k8": [0.31, -0.12, 0.0006, -0.0009, 0.021, 0.58, -0.05, 0.012],
}

POSES = {
    "nadir": (np.array([0.03, -0.02, 0.4]), np.array([1.5, -2.0, 120.0])),
    "oblique": (np.array([0.35, 0.1, -1.2]), np.array([-4.0, 3.0, 90.0])),
    "low": (np.array([-0.2, 0.25, 2.1]), np.array([6.0, -1.0, 60.0])),
}


def camera():
    fx = 8.8 / 13.2 * 5472.0
    return np.array([[fx, 0, 2735.5 + 26.0], [0, fx * 1.0007, 1823.5 - 20.0], [0, 0, 1]], np.float64)


def num(v):
    v = float(v)
    return v if np.isfinite(v) else repr(v)


def flat(a):
    return [num(v) for v in np.asarray(a, np.float64).ravel()]


class Gen:
    def __init__(self):
        self.cases = []
        self.names = set()

    def add(self, name, op, params, inputs, outputs):
        assert name not in self.names, name
        self.names.add(name)
        self.cases.append({"name": name, "op": op, "params": params,
                           "inputs": {k: flat(v) for k, v in inputs.items()}, "outputs": outputs})


def scene(rng, kind, n):
    xy = rng.uniform(-60, 60, (n, 2))
    z = np.zeros((n, 1)) if kind == "planar" else rng.uniform(-4, 4, (n, 1))
    return np.hstack([xy, z])


def project(obj, rv, tv, a, d):
    return cv2.projectPoints(obj, rv, tv, a, np.array(d, np.float64) if d else None)[0].reshape(-1, 2)


def dist_arg(d):
    return np.array(d, np.float64) if d else None


def twice(fn):
    r1 = fn()
    r2 = fn()
    assert repr(r1) == repr(r2) or 'nan' in repr(r1)
    return r1


def generic_out(res):
    n, rvecs, tvecs, errs = res
    return {"n": int(n), "rvecs": [flat(r) for r in rvecs], "tvecs": [flat(t) for t in tvecs],
            "errs": flat(errs) if errs is not None and n > 0 else []}


def pnp_cases(g, rng, a):
    for kind in ("planar", "relief"):
        for n in (4, 6, 10, 30, 100):
            obj0 = scene(rng, kind, n)
            for pname, (rv, tv) in POSES.items():
                for dname, d in DIST.items():
                    ip = project(obj0, rv, tv, a, d)
                    for noise in (0.0, 0.5):
                        img = ip + rng.normal(0, noise, ip.shape) if noise else ip.copy()
                        for mname, m in METHODS.items():
                            if mname == "ippe_square":
                                continue
                            if mname == "ippe" and kind != "planar":
                                continue
                            if mname in ("p3p", "ap3p") and n != 4:
                                continue
                            if mname == "iterative" and kind == "relief" and n < 6:
                                continue
                            ins = {"obj": obj0, "img": img, "a": a, "dist": d}
                            tag = f"{kind}_{n}_{pname}_{dname}_{'noisy' if noise else 'exact'}_{mname}"
                            res = twice(lambda: cv2.solvePnPGeneric(obj0, img, a, dist_arg(d), flags=m))
                            g.add("generic_" + tag, "generic", {"method": mname}, ins, generic_out(res))
                            ok, r, t = twice(lambda: cv2.solvePnP(obj0, img, a, dist_arg(d), flags=m))
                            g.add("pnp_" + tag, "pnp", {"method": mname}, ins,
                                  {"ok": bool(ok), "rvec": flat(r) if ok else [], "tvec": flat(t) if ok else []})


def square_cases(g, rng, a):
    for side in (0.5, 2.0):
        h = side / 2
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)
        for pname, (rv, tv) in POSES.items():
            tv2 = tv * 0.05
            for dname, d in DIST.items():
                ip = project(obj, rv, tv2, a, d)
                for noise in (0.0, 0.5):
                    img = ip + rng.normal(0, noise, ip.shape) if noise else ip.copy()
                    ins = {"obj": obj, "img": img, "a": a, "dist": d}
                    tag = f"square_{side}_{pname}_{dname}_{'noisy' if noise else 'exact'}"
                    for mname in ("ippe_square", "ippe", "p3p", "ap3p", "epnp", "sqpnp"):
                        res = twice(lambda: cv2.solvePnPGeneric(obj, img, a, dist_arg(d), flags=METHODS[mname]))
                        g.add(f"generic_{tag}_{mname}", "generic", {"method": mname}, ins, generic_out(res))


def p3p_cases(g, rng, a):
    for kind in ("planar", "relief"):
        for n in (3, 4):
            for rep in range(3):
                obj = scene(rng, kind, n)
                for pname, (rv, tv) in POSES.items():
                    for dname in ("none", "k5"):
                        d = DIST[dname]
                        img = project(obj, rv, tv, a, d)
                        for mname in ("p3p", "ap3p"):
                            res = twice(lambda: cv2.solveP3P(obj, img, a, dist_arg(d), METHODS[mname]))
                            nsol, rvecs, tvecs = res
                            g.add(f"p3p_{kind}_{n}_{rep}_{pname}_{dname}_{mname}", "p3p", {"method": mname},
                                  {"obj": obj, "img": img, "a": a, "dist": d},
                                  {"n": int(nsol), "rvecs": [flat(r) for r in rvecs], "tvecs": [flat(t) for t in tvecs]})


def ransac_cases(g, rng, a):
    for kind in ("planar", "relief"):
        for n in (4, 5, 20, 100, 300):
            obj = scene(rng, kind, n)
            for pname in ("nadir", "oblique"):
                rv, tv = POSES[pname]
                for dname in ("none", "k5"):
                    d = DIST[dname]
                    ip = project(obj, rv, tv, a, d) + rng.normal(0, 0.5, (n, 2))
                    for frac in ((0.0,) if n <= 5 else (0.1, 0.3, 0.45)):
                        img = ip.copy()
                        nout = int(round(frac * n))
                        if nout:
                            sel = rng.choice(n, nout, replace=False)
                            img[sel] += rng.uniform(-400, 400, (nout, 2))
                        methods = ("iterative", "epnp", "p3p", "ap3p", "sqpnp") + (("ippe",) if kind == "planar" else ())
                        settings = ((100, 8.0, 0.99), (300, 3.0, 0.999))
                        if n >= 300:
                            methods = ("iterative", "epnp")
                            settings = settings[:1]
                        for mname in methods:
                            for dtype in ("f64", "f32"):
                                if dtype == "f32" and mname not in ("epnp", "iterative"):
                                    continue
                                o = obj.astype(np.float32) if dtype == "f32" else obj
                                im = img.astype(np.float32) if dtype == "f32" else img
                                for iters, thr, conf in settings:
                                    res = twice(lambda: cv2.solvePnPRansac(o, im, a, dist_arg(d), iterationsCount=iters,
                                                                           reprojectionError=thr, confidence=conf,
                                                                           flags=METHODS[mname]))
                                    ok, r, t, inl = res
                                    g.add(f"ransac_{kind}_{n}_{pname}_{dname}_{frac}_{mname}_{dtype}_{iters}", "ransac",
                                          {"method": mname, "dtype": dtype, "iters": iters, "thr": thr, "conf": conf},
                                          {"obj": o, "img": im, "a": a, "dist": d},
                                          {"ok": bool(ok), "rvec": flat(r) if ok else [], "tvec": flat(t) if ok else [],
                                           "inliers": [int(v) for v in np.asarray(inl).ravel()] if ok and inl is not None else []})


def ransac_guess_cases(g, rng, a):
    for kind in ("planar", "relief"):
        obj = scene(rng, kind, 60)
        for pname, (rv, tv) in POSES.items():
            d = DIST["k5"]
            img = project(obj, rv, tv, a, d) + rng.normal(0, 0.5, (60, 2))
            sel = rng.choice(60, 18, replace=False)
            img[sel] += rng.uniform(-400, 400, (18, 2))
            r0 = (rv + np.array([0.01, -0.02, 0.015])).reshape(3, 1)
            t0 = (tv + np.array([0.5, -0.4, 2.0])).reshape(3, 1)
            res = twice(lambda: cv2.solvePnPRansac(obj, img, a, dist_arg(d), r0.copy(), t0.copy(), True, 100, 8.0, 0.99,
                                                   flags=cv2.SOLVEPNP_ITERATIVE))
            ok, r, t, inl = res
            g.add(f"ransacguess_{kind}_{pname}", "ransac",
                  {"method": "iterative", "dtype": "f64", "iters": 100, "thr": 8.0, "conf": 0.99},
                  {"obj": obj, "img": img, "a": a, "dist": d, "rvec0": r0, "tvec0": t0},
                  {"ok": bool(ok), "rvec": flat(r), "tvec": flat(t), "inliers": [int(v) for v in np.asarray(inl).ravel()]})


def refine_cases(g, rng, a):
    for kind in ("planar", "relief"):
        for n in (3, 8, 50):
            obj = scene(rng, kind, n)
            for pname, (rv, tv) in POSES.items():
                for dname, d in DIST.items():
                    img = project(obj, rv, tv, a, d) + rng.normal(0, 0.5, (n, 2))
                    r0 = (rv + np.array([0.01, -0.02, 0.015])).reshape(3, 1)
                    t0 = (tv + np.array([0.5, -0.4, 2.0])).reshape(3, 1)
                    ins = {"obj": obj, "img": img, "a": a, "dist": d, "rvec0": r0, "tvec0": t0}
                    for crit, cname in (((cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 20, 1.1920928955078125e-07), "default"),
                                        ((cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 5, 1e-3), "short")):
                        def lm():
                            r, t = r0.copy(), t0.copy()
                            r, t = cv2.solvePnPRefineLM(obj, img, a, dist_arg(d), r, t, crit)
                            return r, t

                        r, t = twice(lm)
                        p = {"type": crit[0], "max": crit[1], "eps": crit[2]}
                        g.add(f"lm_{kind}_{n}_{pname}_{dname}_{cname}", "lm", p, ins, {"rvec": flat(r), "tvec": flat(t)})
                        for lam in (1.0, 0.5):
                            def vvs():
                                r, t = r0.copy(), t0.copy()
                                r, t = cv2.solvePnPRefineVVS(obj, img, a, dist_arg(d), r, t, crit, lam)
                                return r, t

                            r, t = twice(vvs)
                            p2 = dict(p)
                            p2["lambda"] = lam
                            g.add(f"vvs_{kind}_{n}_{pname}_{dname}_{cname}_{lam}", "vvs", p2, ins,
                                  {"rvec": flat(r), "tvec": flat(t)})


def main():
    rng = np.random.default_rng(20261016)
    a = camera()
    g = Gen()
    pnp_cases(g, rng, a)
    square_cases(g, rng, a)
    p3p_cases(g, rng, a)
    ransac_cases(g, rng, a)
    refine_cases(g, rng, a)
    ransac_guess_cases(g, rng, a)
    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "pnp_golden.json")
    with open(out, "w") as f:
        json.dump({"cases": g.cases}, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases to {out}")


if __name__ == "__main__":
    main()
