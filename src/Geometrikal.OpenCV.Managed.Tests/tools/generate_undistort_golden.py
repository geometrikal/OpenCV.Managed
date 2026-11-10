#!/usr/bin/env python3
"""Golden-data generator for the managed lens distortion and camera model port.

Runs cv2 5.0.0 ``initUndistortRectifyMap``, ``undistort``, ``undistortPoints``,
``undistortImagePoints``, ``getDefaultNewCameraMatrix``,
``getOptimalNewCameraMatrix``, ``Rodrigues`` and ``projectPoints`` over a seeded
battery and writes ``TestData/undistort_golden.json``.
``Calib/UndistortGoldenTests.cs`` and ``Calib/CameraModelGoldenTests.cs`` replay
each case.

Required environment (the script stops if cv2 is not 5.0.0):
 * cv2 5.0.0. Run with ``PYTHONPATH=<dir>``.
 * OPENCV_CPU_DISABLE=AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL
   before cv2 is imported (the script sets it when it is not set). The map
   computer then runs the baseline (SSE3, 128-bit, no FMA) body, which the port
   reproduces.
 * IPP, OpenCL and threads are disabled in the script.

Regenerate with (from the ``Geometrikal.OpenCV.Managed.Tests`` directory):
    PYTHONPATH=<cv2 5.0.0 dir> python tools/generate_undistort_golden.py

Battery notes:
 * Camera intrinsics are a DJI FC6310 (5472 x 3648, 8.8 mm on 13.2 mm) scaled
   down, with an off-centre principal point. Coefficient sets have 0, 4, 5, 8,
   12 and 14 values; the "extreme" set produces coordinates outside the 16-bit
   range to cover the fixed-point saturation (blocks) and wrap (tail) rules.
 * Maps and undistorted images larger than 4096 elements are stored as
   SHA-256 hashes of the raw buffer; smaller ones are stored raw.
 * Every comparison is byte-exact.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import hashlib
import json
import os
import sys

os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

if cv2.__version__ != "5.0.0":
    sys.exit(f"generate_undistort_golden.py requires cv2 5.0.0 (found {cv2.__version__})")
if cv2.checkHardwareSupport(11) or cv2.checkHardwareSupport(13):
    sys.exit("AVX2/AVX-512 dispatch is enabled; set OPENCV_CPU_DISABLE before importing cv2")

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

HASH_LIMIT = 4096

DTYPES = {"u8": np.uint8, "i16": np.int16, "u16": np.uint16, "f32": np.float32, "f64": np.float64}

# Adjudicated differences (none).
SUSPECT_CASES = {}


def dt_of(arr):
    for k, v in DTYPES.items():
        if arr.dtype == v:
            return k
    raise ValueError(arr.dtype)


def arr_json(arr, allow_hash=True):
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    m = {"dtype": dt_of(arr), "width": int(arr.shape[1]), "height": int(arr.shape[0]),
         "channels": 1 if arr.ndim == 2 else int(arr.shape[2])}
    raw = arr.tobytes()
    if allow_hash and arr.size > HASH_LIMIT:
        m["sha256"] = hashlib.sha256(raw).hexdigest()
    else:
        m["data"] = base64.b64encode(raw).decode("ascii")
    return m


class Gen:
    def __init__(self):
        self.cases = []
        self.names = set()
        self.shared = {}

    def input_json(self, arr):
        m = arr_json(arr, False)
        if len(m["data"]) <= HASH_LIMIT:
            return m
        key = hashlib.sha256(m["data"].encode("ascii")).hexdigest()
        self.shared.setdefault(key, m)
        return {"ref": key}

    def add(self, name, op, params, inputs=None, outputs=None):
        assert name not in self.names, name
        self.names.add(name)
        case = {"name": name, "op": op, "params": params, "suspect": name in SUSPECT_CASES,
                "inputs": {k: self.input_json(v) for k, v in (inputs or {}).items()},
                "outputs": {k: arr_json(v) for k, v in (outputs or {}).items()}}
        if case["suspect"]:
            case["suspectReason"] = SUSPECT_CASES[name]
        self.cases.append(case)


def camera(w, h):
    s = w / 5472.0
    fx = 8.8 / 13.2 * 5472.0 * s
    return np.array([[fx, 0, (w - 1) * 0.5 + 3.25 * s * 8], [0, fx * 1.0007, (h - 1) * 0.5 - 2.5 * s * 8], [0, 0, 1]],
                    np.float64)


DIST = {
    "none": [],
    "k4": [-0.2713, 0.1125, 0.00071, -0.00118],
    "k5": [-0.2713, 0.1125, 0.00071, -0.00118, -0.0315],
    "k8": [0.31, -0.12, 0.0006, -0.0009, 0.021, 0.58, -0.05, 0.012],
    "k12": [-0.2713, 0.1125, 0.00071, -0.00118, -0.0315, 0.012, -0.004, 0.0015, 0.0011, -0.0004, 0.0008,
            -0.0003],
    "k14": [-0.2713, 0.1125, 0.00071, -0.00118, -0.0315, 0.012, -0.004, 0.0015, 0.0011, -0.0004, 0.0008,
            -0.0003, 0.012, -0.018],
    "extreme": [55.0, 30.0, 0.5, -0.4, 10.0],
}


def rot(rx, ry, rz):
    return cv2.Rodrigues(np.array([rx, ry, rz], np.float64))[0]


def map_cases(g):
    sizes = [(37, 23), (101, 64), (257, 171), (684, 456)]
    for (w, h) in sizes:
        a = camera(w, h)
        for dname, d in DIST.items():
            dist = np.array(d, np.float64)
            rs = {"rnone": None, "rot": rot(0.021, -0.013, 0.0042)}
            if (w, h) == (101, 64):
                rs["rid"] = np.eye(3)
            for rname, r in rs.items():
                news = {"nnone": None, "nk": a.copy()}
                news["nk"][0, 0] *= 0.83
                news["nk"][1, 1] *= 0.83
                if (w, h) in ((37, 23), (101, 64)):
                    p = np.hstack([news["nk"], np.array([[3.5], [-1.25], [0.0]])])
                    news["np"] = p
                for nname, nk in news.items():
                    if (w, h) == (684, 456) and dname not in ("k5", "k14", "extreme"):
                        continue
                    for mname, mtype in (("f1", cv2.CV_32FC1), ("f2", cv2.CV_32FC2), ("fx", cv2.CV_16SC2)):
                        m1, m2 = cv2.initUndistortRectifyMap(a, dist if len(d) else None, r, nk, (w, h), mtype)
                        outs = {"map1": m1}
                        if m2 is not None:
                            outs["map2"] = m2
                        ins = {"a": a}
                        if len(d):
                            ins["dist"] = dist
                        if r is not None:
                            ins["r"] = r
                        if nk is not None:
                            ins["newk"] = nk
                        g.add(f"map_{w}x{h}_{dname}_{rname}_{nname}_{mname}", "map",
                              {"width": w, "height": h, "mtype": mname}, ins, outs)


def undistort_cases(g, rng):
    images = {}
    for (w, h) in [(64, 48), (257, 171), (517, 301)]:
        a = camera(w, h)
        for dname in ("none", "k5", "k8", "k14"):
            d = DIST[dname]
            dist = np.array(d, np.float64)
            for nname in ("nnone", "nk"):
                nk = None
                if nname == "nk":
                    nk = cv2.getOptimalNewCameraMatrix(a, dist if len(d) else None, (w, h), 0.4)[0]
                for tname, (dtype, cn) in {"u8c1": (np.uint8, 1), "u8c3": (np.uint8, 3), "f32c1": (np.float32, 1)}.items():
                    if (w, h) == (517, 301) and tname != "u8c3":
                        continue
                    if (w, h, tname) not in images:
                        if dtype == np.uint8:
                            img = rng.integers(0, 256, (h, w, cn), dtype=np.uint8)
                        else:
                            img = rng.uniform(-10, 10, (h, w, cn)).astype(np.float32)
                        images[(w, h, tname)] = img[:, :, 0].copy() if cn == 1 else img
                    img = images[(w, h, tname)]
                    out = cv2.undistort(img, a, dist if len(d) else None, None, nk)
                    ins = {"src": img, "a": a}
                    if len(d):
                        ins["dist"] = dist
                    if nk is not None:
                        ins["newk"] = nk
                    g.add(f"undistort_{w}x{h}_{dname}_{nname}_{tname}", "undistort", {}, ins, {"dst": out})


def points_cases(g, rng):
    w, h = 684, 456
    a = camera(w, h)
    pts = np.vstack([rng.uniform([-20, -20], [w + 20, h + 20], (200, 2)),
                     np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1], [a[0, 2], a[1, 2]]])])
    r = rot(0.021, -0.013, 0.0042)
    p = a.copy()
    p[0, 0] *= 0.9
    p34 = np.hstack([p, np.array([[2.0], [-1.0], [0.5]])])
    crits = {
        "cdef": None,
        "c20": (cv2.TERM_CRITERIA_COUNT, 20, 0.0),
        "ce": (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-4),
    }
    for dname, d in DIST.items():
        if dname == "extreme":
            continue
        dist = np.array(d, np.float64)
        for dt in ("f32", "f64"):
            src = pts.astype(DTYPES[dt]).reshape(-1, 1, 2)
            for rpname, (rr, pp) in {"none": (None, None), "r": (r, None), "p": (None, p), "rp": (r, p),
                                     "p34": (None, p34)}.items():
                for cname, crit in crits.items():
                    if cname != "cdef" and rpname not in ("none", "rp"):
                        continue
                    if crit is None:
                        out = cv2.undistortPoints(src, a, dist if len(d) else None, R=rr, P=pp)
                    else:
                        out = cv2.undistortPoints(src, a, dist if len(d) else None, R=rr, P=pp, criteria=crit)
                    ins = {"src": src.reshape(-1, 2), "a": a}
                    if len(d):
                        ins["dist"] = dist
                    if rr is not None:
                        ins["r"] = rr
                    if pp is not None:
                        ins["p"] = pp
                    params = {"crit": [] if crit is None else [crit[0], crit[1], crit[2]]}
                    g.add(f"upts_{dname}_{dt}_{rpname}_{cname}", "upts", params, ins, {"dst": out.reshape(-1, 2)})
            out = cv2.undistortImagePoints(src, a, dist if len(d) else np.zeros(4))
            ins = {"src": src.reshape(-1, 2), "a": a, "dist": dist if len(d) else np.zeros(4)}
            g.add(f"uimgpts_{dname}_{dt}", "uimgpts", {}, ins, {"dst": out.reshape(-1, 2)})


def newk_cases(g):
    for (w, h) in [(684, 456), (257, 171)]:
        a = camera(w, h)
        for center in (False, True):
            g.add(f"defk_{w}x{h}_{int(center)}", "defk", {"width": w, "height": h, "center": center}, {"a": a},
                  {"k": cv2.getDefaultNewCameraMatrix(a, (w, h), center)})
        for dname, d in DIST.items():
            if dname == "extreme":
                continue
            dist = np.array(d, np.float64)
            for alpha in (0.0, 0.37, 1.0):
                for nsz in ((0, 0), (w // 2 + 3, h // 2 - 1)):
                    for center in (False, True):
                        m, roi = cv2.getOptimalNewCameraMatrix(a, dist if len(d) else None, (w, h), alpha, nsz,
                                                               centerPrincipalPoint=center)
                        ins = {"a": a}
                        if len(d):
                            ins["dist"] = dist
                        g.add(f"optk_{w}x{h}_{dname}_{alpha}_{nsz[0]}_{int(center)}", "optk",
                              {"width": w, "height": h, "alpha": alpha, "nw": nsz[0], "nh": nsz[1], "center": center,
                               "roi": [int(v) for v in roi]}, ins, {"k": m})


def rodrigues_cases(g, rng):
    vecs = [rng.uniform(-np.pi, np.pi, 3) for _ in range(12)]
    vecs += [np.array([0.0, 0.0, 0.0]), np.array([1e-17, 0, 0]), np.array([np.pi, 0, 0]),
             np.array([0, -np.pi, 0]), np.array([0.3, 0.4, 0.5]) * (np.pi / np.linalg.norm([0.3, 0.4, 0.5])),
             np.array([1e-6, -2e-6, 3e-7]), np.array([0.0, 0.0, 2.5])]
    for i, v in enumerate(vecs):
        for dt in ("f32", "f64"):
            src = v.astype(DTYPES[dt]).reshape(3, 1)
            dst, jac = cv2.Rodrigues(src)
            g.add(f"rod_v2m_{dt}_{i}", "rod", {}, {"src": src}, {"dst": dst, "jac": jac})
            if i == 0:
                src = v.astype(DTYPES[dt]).reshape(1, 3)
                dst, jac = cv2.Rodrigues(src)
                g.add(f"rod_v2m_{dt}_{i}_row", "rod", {}, {"src": src}, {"dst": dst, "jac": jac})
            m = cv2.Rodrigues(v.astype(np.float64))[0]
            for nname, noise in (("exact", 0.0), ("noisy", 1e-3)):
                mm = (m + rng.uniform(-noise, noise, (3, 3))).astype(DTYPES[dt])
                dst, jac = cv2.Rodrigues(mm)
                g.add(f"rod_m2v_{dt}_{i}_{nname}", "rod", {}, {"src": mm}, {"dst": dst, "jac": jac})
    for dt in ("f32", "f64"):
        bad = np.eye(3, dtype=DTYPES[dt])
        bad[0, 1] = 150
        dst, jac = cv2.Rodrigues(bad)
        g.add(f"rod_m2v_{dt}_range", "rod", {}, {"src": bad}, {"dst": dst, "jac": jac})
        flip = np.diag([1.0, -1.0, -1.0]).astype(DTYPES[dt])
        dst, jac = cv2.Rodrigues(flip)
        g.add(f"rod_m2v_{dt}_flip", "rod", {}, {"src": flip}, {"dst": dst, "jac": jac})


def project_cases(g, rng):
    w, h = 684, 456
    a = camera(w, h)
    obj = np.hstack([rng.uniform(-30, 30, (60, 2)), rng.uniform(-2, 2, (60, 1))])
    poses = {"p0": (np.array([0.05, -0.03, 0.01]), np.array([0.5, -0.2, 80.0])),
             "p1": (np.array([2.9, 0.1, -0.2]), np.array([-1.0, 0.3, 60.0])),
             "zero": (np.zeros(3), np.array([0.0, 0.0, 50.0]))}
    for dname, d in DIST.items():
        if dname == "extreme":
            continue
        dist = np.array(d, np.float64)
        for pname, (rv, tv) in poses.items():
            for dt in ("f32", "f64"):
                src = obj.astype(DTYPES[dt]).reshape(-1, 1, 3)
                for ar in (0.0, 1.25):
                    if ar and (pname != "p0" or dt != "f64"):
                        continue
                    ip, jac = cv2.projectPoints(src, rv, tv, a, dist if len(d) else None, aspectRatio=ar)
                    ins = {"src": src.reshape(-1, 3), "a": a, "rvec": rv.reshape(3, 1), "tvec": tv.reshape(3, 1)}
                    if len(d):
                        ins["dist"] = dist
                    g.add(f"proj_{dname}_{pname}_{dt}_{ar}", "proj", {"aspect": ar}, ins,
                          {"dst": ip.reshape(-1, 2), "jac": jac})
            if dname in ("k5", "k14"):
                rm = cv2.Rodrigues(rv)[0]
                ip, jac = cv2.projectPoints(obj.reshape(-1, 1, 3), rm, tv, a, dist)
                ins = {"src": obj, "a": a, "rmat": rm, "tvec": tv.reshape(3, 1), "dist": dist}
                g.add(f"proj_{dname}_{pname}_rmat", "proj", {"aspect": 0.0}, ins,
                      {"dst": ip.reshape(-1, 2), "jac": jac})


def pnp_cases(g):
    rng = np.random.default_rng(20261017)
    w, h = 684, 456
    a = camera(w, h)
    planar = np.hstack([rng.uniform(-40, 40, (12, 2)), np.zeros((12, 1))])
    relief = np.hstack([rng.uniform(-40, 40, (12, 2)), rng.uniform(-3, 3, (12, 1))])
    poses = {"nadir": (np.array([0.03, -0.02, 0.4]), np.array([1.5, -2.0, 120.0])),
             "oblique": (np.array([0.35, 0.1, -1.2]), np.array([-4.0, 3.0, 90.0]))}
    for oname, obj in (("planar", planar), ("relief", relief)):
        for pname, (rv, tv) in poses.items():
            for dname in ("none", "k5", "k8"):
                d = DIST[dname]
                dist = np.array(d, np.float64)
                ip = cv2.projectPoints(obj, rv, tv, a, dist if len(d) else None)[0].reshape(-1, 2)
                for nname, noise in (("exact", 0.0), ("noisy", 0.4)):
                    img = ip + rng.normal(0, noise, ip.shape) if noise else ip.copy()
                    ok, r, t = cv2.solvePnP(obj, img, a, dist if len(d) else None, flags=cv2.SOLVEPNP_ITERATIVE)
                    assert ok
                    ins = {"obj": obj, "img": img, "a": a}
                    if len(d):
                        ins["dist"] = dist
                    g.add(f"pnp_{oname}_{pname}_{dname}_{nname}", "pnp", {"guess": False}, ins,
                          {"rvec": r.reshape(3, 1), "tvec": t.reshape(3, 1)})
                    r0 = (rv + np.array([0.01, -0.02, 0.015])).reshape(3, 1)
                    t0 = (tv + np.array([0.5, -0.4, 2.0])).reshape(3, 1)
                    ok, r, t = cv2.solvePnP(obj, img, a, dist if len(d) else None, r0.copy(), t0.copy(), True,
                                            cv2.SOLVEPNP_ITERATIVE)
                    ins = dict(ins)
                    ins["rvec0"] = r0
                    ins["tvec0"] = t0
                    g.add(f"pnp_{oname}_{pname}_{dname}_{nname}_guess", "pnp", {"guess": True}, ins,
                          {"rvec": r.reshape(3, 1), "tvec": t.reshape(3, 1)})


def main():
    rng = np.random.default_rng(20261016)
    g = Gen()
    map_cases(g)
    undistort_cases(g, rng)
    points_cases(g, rng)
    newk_cases(g)
    rodrigues_cases(g, rng)
    project_cases(g, rng)
    pnp_cases(g)

    m1, m2 = cv2.initUndistortRectifyMap(np.eye(3), None, None, np.eye(3), (8, 4), cv2.CV_32FC1)
    assert m1[0, 5] == 5.0 and m2[3, 0] == 3.0
    assert set(SUSPECT_CASES) <= g.names, set(SUSPECT_CASES) - g.names

    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "undistort_golden.json")
    with open(out, "w") as f:
        json.dump({"shared": g.shared, "cases": g.cases}, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases to {out}")


if __name__ == "__main__":
    main()
