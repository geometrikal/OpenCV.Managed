#!/usr/bin/env python3
"""Golden-data generator for the managed sparse optical flow port.

Runs cv2 5.0.0 ``buildOpticalFlowPyramid`` and ``calcOpticalFlowPyrLK`` over a seeded battery
and writes ``TestData/lk_golden.json``. ``Video/OpticalFlowGoldenTests.cs`` replays the cases.

Required environment (the script stops if cv2 is not 5.0.0):
 * cv2 5.0.0, run with ``PYTHONPATH=<dir>`` (see ``generate_advfilter_golden.py``).
 * OPENCV_CPU_DISABLE=AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL before cv2 is
   imported (the script sets it when it is not set).
 * IPP, OpenCL and threads are disabled in the script.

Regenerate with (from the ``Geometrikal.OpenCV.Managed.Tests`` directory):
    PYTHONPATH=<cv2 5.0.0 dir> python tools/generate_lk_golden.py

Battery notes:
 * Inputs are textures (blurred seeded noise plus sinusoids). The next image is the previous
   image moved by a known sub-pixel shift, a rotation or a shift plus noise.
 * Points come from goodFeaturesToTrack, from a grid, from the image borders and from outside
   the image.
 * Points and errors are stored as float32 bit patterns. ``err`` is compared only where cv2
   writes it: for every point with OPTFLOW_LK_GET_MIN_EIGENVALS, otherwise for status 1 only
   (cv2 leaves the other entries uninitialised).
 * Pyramid levels and derivatives are the ROI contents. Python cannot return the padding.

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
    sys.exit(f"generate_lk_golden.py requires cv2 5.0.0 (found {cv2.__version__}); "
             "set PYTHONPATH to the opencv-contrib-python-headless==5.0.0.93 install")
if cv2.checkHardwareSupport(11) or cv2.checkHardwareSupport(13):
    sys.exit("AVX2/AVX-512 dispatch is enabled; set OPENCV_CPU_DISABLE before importing cv2")

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

BORDERS = {"Reflect101": cv2.BORDER_REFLECT_101, "Replicate": cv2.BORDER_REPLICATE,
           "Reflect": cv2.BORDER_REFLECT, "Constant": cv2.BORDER_CONSTANT, "Wrap": cv2.BORDER_WRAP,
           "Transparent": cv2.BORDER_TRANSPARENT}
SUSPECT_CASES = {}

ARRAYS = []
ARRAY_INDEX = {}


def ref(a):
    a = np.ascontiguousarray(a)
    dt = {np.dtype(np.uint8): "u8", np.dtype(np.int16): "s16", np.dtype(np.float32): "f32"}[a.dtype]
    m = {"dtype": dt, "width": int(a.shape[1]), "height": int(a.shape[0]),
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


def fbits_list(a):
    return [int(v) for v in np.ascontiguousarray(a, np.float32).reshape(-1).view(np.uint32)]


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


def texture(rng, w, h, cn=1):
    noise = rng.integers(0, 256, (h, w, cn) if cn > 1 else (h, w), dtype=np.uint8)
    img = cv2.GaussianBlur(noise, (0, 0), 1.6).astype(np.float32)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    wave = 40 * np.sin(xs / 5.3 + 0.7) * np.cos(ys / 7.1)
    if cn > 1:
        wave = wave[:, :, None]
    img = (img - 128) * 3.0 + 128 + wave
    return np.clip(img, 0, 255).astype(np.uint8)


def moved(img, dx, dy, angle=0.0, noise=0, rng=None):
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    m[0, 2] += dx
    m[1, 2] += dy
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if noise:
        out = np.clip(out.astype(np.int16) + rng.integers(-noise, noise + 1, out.shape), 0, 255).astype(np.uint8)
    return out


def points(rng, img, kind):
    h, w = img.shape[:2]
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if kind == "gftt":
        p = cv2.goodFeaturesToTrack(gray, 120, 0.01, 4)
        return p.reshape(-1, 2).astype(np.float32)
    if kind == "grid":
        ys, xs = np.mgrid[2:h - 1:7, 3:w - 1:9]
        p = np.stack([xs.ravel() + 0.25, ys.ravel() + 0.5], 1).astype(np.float32)
        return p
    if kind == "border":
        p = [(0, 0), (w - 1, h - 1), (0, h - 1), (w - 1, 0), (0.5, h / 2), (w - 1.5, h / 3), (w / 2, 0.2),
             (w / 3, h - 0.7), (-3, 5), (w + 2, 4), (5, -4.5), (7, h + 3), (-30, -30), (w + 40, h / 2),
             (-10.5, h / 2), (w / 2, h + 9.25), (1.3, 1.7), (w - 2.2, h - 2.6)]
        return np.array(p, np.float32)
    raise ValueError(kind)


def lk_case(g, name, prev, nxt, pts, win, max_level, criteria, flags=0, min_eig=1e-4, init=None):
    kw = dict(winSize=win, maxLevel=max_level, criteria=criteria, flags=flags, minEigThreshold=min_eig)
    if flags & cv2.OPTFLOW_USE_INITIAL_FLOW:
        nxt_pts, st, err = cv2.calcOpticalFlowPyrLK(prev, nxt, pts, init.copy(), **kw)
    else:
        nxt_pts, st, err = cv2.calcOpticalFlowPyrLK(prev, nxt, pts, None, **kw)
    st = st.reshape(-1).astype(np.uint8)
    err = err.reshape(-1).astype(np.float32)
    mask = np.ones_like(st, bool) if flags & cv2.OPTFLOW_LK_GET_MIN_EIGENVALS else st == 1
    g.add(name, {
        "op": "lk", "prev": ref(prev), "next": ref(nxt),
        "params": {"win": list(win), "maxLevel": max_level, "type": criteria[0], "maxCount": criteria[1],
                   "epsilon": criteria[2], "flags": flags, "minEig": min_eig},
        "prevPts": fbits_list(pts), "initPts": fbits_list(init) if init is not None else None,
        "nextPts": fbits_list(nxt_pts), "status": [int(v) for v in st],
        "errMask": [int(v) for v in mask], "err": [int(b) if m else 0 for b, m in zip(fbits_list(err), mask)],
    })


def lk_battery(g, rng):
    default = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01)
    scenes = []
    for (w, h) in ((160, 120), (97, 83), (64, 48)):
        img = texture(rng, w, h)
        scenes.append((f"t{w}x{h}_shift", img, moved(img, 2.37, -1.61)))
        scenes.append((f"t{w}x{h}_rot", img, moved(img, 0.5, 0.25, angle=3.0)))
    big = texture(rng, 200, 150)
    scenes.append(("t200x150_far", big, moved(big, 9.4, 6.2, noise=3, rng=rng)))
    small = texture(rng, 40, 30)
    scenes.append(("t40x30_shift", small, moved(small, 1.2, 0.7)))
    tiny = texture(rng, 13, 11)
    scenes.append(("t13x11_shift", tiny, moved(tiny, 0.6, -0.3)))
    for sname, prev, nxt in scenes:
        for kind in ("gftt", "grid", "border"):
            pts = points(rng, prev, kind)
            if len(pts) == 0:
                continue
            for win in ((21, 21), (9, 9), (15, 7), (5, 5), (3, 3)):
                for lvl in (0, 1, 3, 5):
                    if (win, lvl) not in (((21, 21), 3), ((21, 21), 0), ((9, 9), 1), ((15, 7), 3), ((5, 5), 5), ((3, 3), 1)):
                        continue
                    lk_case(g, f"lk_{sname}_{kind}_w{win[0]}x{win[1]}_l{lvl}", prev, nxt, pts, win, lvl, default)
            lk_case(g, f"lk_{sname}_{kind}_mineig", prev, nxt, pts, (21, 21), 3, default,
                    flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS)
            lk_case(g, f"lk_{sname}_{kind}_mineig_thr", prev, nxt, pts, (11, 11), 2, default,
                    flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS, min_eig=5e-2)
            lk_case(g, f"lk_{sname}_{kind}_thr", prev, nxt, pts, (11, 11), 2, default, min_eig=5e-2)
            init = (pts + np.array([2.0, -1.5], np.float32)).astype(np.float32)
            lk_case(g, f"lk_{sname}_{kind}_init", prev, nxt, pts, (21, 21), 3, default,
                    flags=cv2.OPTFLOW_USE_INITIAL_FLOW, init=init)
            lk_case(g, f"lk_{sname}_{kind}_init_mineig", prev, nxt, pts, (13, 13), 2, default,
                    flags=cv2.OPTFLOW_USE_INITIAL_FLOW | cv2.OPTFLOW_LK_GET_MIN_EIGENVALS, init=init)
            for cname, crit in (("count5", (cv2.TERM_CRITERIA_COUNT, 5, 0.0)),
                                ("eps", (cv2.TERM_CRITERIA_EPS, 0, 0.3)),
                                ("tight", (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 0.0)),
                                ("count0", (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 0, 0.01)),
                                ("big", (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 500, 50.0))):
                if kind != "gftt":
                    continue
                lk_case(g, f"lk_{sname}_{kind}_{cname}", prev, nxt, pts, (15, 15), 2, crit)
    # Multi-channel images.
    for cn in (3, 4):
        img = texture(rng, 90, 70, cn)
        nxt = moved(img, -1.8, 2.3)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if cn == 3 else cv2.COLOR_BGRA2GRAY)
        for kind in ("gftt", "grid", "border"):
            pts = points(rng, gray, kind)
            for win, lvl in (((21, 21), 3), ((7, 9), 1), ((4, 4), 0)):
                lk_case(g, f"lk_c{cn}_{kind}_w{win[0]}x{win[1]}_l{lvl}", img, nxt, pts, win, lvl, default)
            lk_case(g, f"lk_c{cn}_{kind}_mineig", img, nxt, pts, (21, 21), 2, default,
                    flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS)
    # Flat images: zero gradient rejects every point.
    flat = np.full((50, 60), 77, np.uint8)
    pts = points(rng, texture(rng, 60, 50), "grid")
    lk_case(g, "lk_flat_grid", flat, flat, pts, (21, 21), 3, default)
    lk_case(g, "lk_flat_grid_mineig", flat, flat, pts, (21, 21), 3, default, flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS)


def pyr_case(g, name, img, win, max_level, with_deriv, pyr_border, deriv_border, reuse):
    ret, pyr = cv2.buildOpticalFlowPyramid(img, win, max_level, withDerivatives=with_deriv,
                                           pyrBorder=BORDERS[pyr_border], derivBorder=BORDERS[deriv_border],
                                           tryReuseInputImage=reuse)
    g.add(name, {
        "op": "pyramid", "input": ref(img),
        "params": {"win": list(win), "maxLevel": max_level, "withDerivatives": with_deriv,
                   "pyrBorder": pyr_border, "derivBorder": deriv_border, "reuse": reuse},
        "returned": int(ret), "outputs": [expected(m) for m in pyr],
    })


def pyr_battery(g, rng):
    for (w, h, cn) in ((160, 120, 1), (97, 83, 1), (40, 30, 1), (13, 11, 1), (33, 47, 3), (25, 19, 4), (3, 3, 1),
                       (1, 7, 1), (8, 1, 2)):
        img = texture(rng, w, h, cn)
        for win, lvl in (((21, 21), 3), ((5, 5), 4), ((3, 3), 6), ((9, 4), 2), ((4, 9), 0)):
            for with_deriv in (True, False):
                pyr_case(g, f"pyr_{w}x{h}c{cn}_w{win[0]}x{win[1]}_l{lvl}_d{int(with_deriv)}", img, win, lvl,
                         with_deriv, "Reflect101", "Constant", True)
        for pb in ("Replicate", "Reflect", "Constant", "Wrap"):
            for db in ("Constant", "Reflect101", "Replicate"):
                pyr_case(g, f"pyr_{w}x{h}c{cn}_{pb}_{db}", img, (7, 7), 3, True, pb, db, False)


def main():
    rng = np.random.default_rng(20261018)
    g = Gen()
    pyr_battery(g, rng)
    lk_battery(g, rng)
    assert set(SUSPECT_CASES) <= g.names, set(SUSPECT_CASES) - g.names
    doc = {"generator": {"cv2": cv2.__version__, "numpy": np.__version__, "machine": platform.processor()},
           "arrays": ARRAYS, "cases": g.cases}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TestData", "lk_golden.json")
    with open(out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases, {len(ARRAYS)} arrays to {out}")


if __name__ == "__main__":
    main()
