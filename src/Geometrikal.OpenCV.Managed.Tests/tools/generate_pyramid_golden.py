#!/usr/bin/env python3
"""Golden-data generator for the managed image pyramid and keypoint port.

Runs cv2 5.0.0 ``pyrDown``, ``pyrUp``, ``buildPyramid``, ``KeyPoint_convert`` and
``KeyPoint_overlap`` over a seeded battery and writes ``TestData/pyramid_golden.json``.
``ImgProc/PyramidGoldenTests.cs`` and ``Features/KeyPointTests.cs`` replay the cases and
compare the output bytes exactly.

Required environment (the script stops if cv2 is not 5.0.0):
 * cv2 5.0.0, for example
       pip install --no-deps --target <dir> opencv-contrib-python-headless==5.0.0.93
   and run with ``PYTHONPATH=<dir>``. numpy 1.22 or later.
 * OPENCV_CPU_DISABLE=AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL before cv2 is
   imported (the script sets it when it is not set). pyramids.cpp is compiled for the SSE3
   baseline only, so the setting does not change its output; it keeps the environment equal to
   the other generators.
 * IPP, OpenCL and threads are disabled in the script.

Regenerate with (from the ``Geometrikal.OpenCV.Managed.Tests`` directory):
    PYTHONPATH=<cv2 5.0.0 dir> python tools/generate_pyramid_golden.py

Battery notes:
 * Float inputs are random normal values, so the OpenCV vector and scalar sum orders give
   different results on many elements. The cases check that each element uses the order of
   the OpenCV 128-bit loop that computes it.
 * ``roi`` cases store the parent array and the view rectangle; cv2 receives the numpy view,
   which is an isolated image for OpenCV.
 * pyrUp with a one-column source and a three-column destination is not in the battery: OpenCV
   leaves column 2 uninitialized. A one-row source with a three-row destination is defined (row 2
   is a copy of row 0) and is in the battery.
 * Large cases build their input from ``hash_input`` (``"gen"`` in the case). Outputs with more
   than 4096 elements are stored as SHA-256 (``expectedSha``).
 * cv2 has no ``buildPyramid`` binding; the script applies ``pyrDown`` per level, as OpenCV
   ``buildPyramid`` does without IPP.
 * cv2 has no ``KeyPointsFilter`` binding; ``Features/KeyPointTests.cs`` tests it analytically.

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
    sys.exit(f"generate_pyramid_golden.py requires cv2 5.0.0 (found {cv2.__version__}); "
             "set PYTHONPATH to the opencv-contrib-python-headless==5.0.0.93 install")
if cv2.checkHardwareSupport(11) or cv2.checkHardwareSupport(13):
    sys.exit("AVX2/AVX-512 dispatch is enabled; set OPENCV_CPU_DISABLE before importing cv2")

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

BORDERS = {"Replicate": cv2.BORDER_REPLICATE, "Reflect": cv2.BORDER_REFLECT,
           "Reflect101": cv2.BORDER_REFLECT_101, "Wrap": cv2.BORDER_WRAP}
DTYPES = {"u8": np.uint8, "u16": np.uint16, "i16": np.int16, "f32": np.float32, "f64": np.float64}
SUSPECT_CASES = {}


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def dt_of(arr):
    for k, v in DTYPES.items():
        if arr.dtype == v:
            return k
    raise ValueError(arr.dtype)


def hash_u32(x, y):
    h = (np.asarray(x, dtype=np.uint32) * np.uint32(73856093)) ^ (np.asarray(y, dtype=np.uint32) * np.uint32(19349663))
    h ^= h >> np.uint32(13)
    h = h * np.uint32(0x5BD1E995)
    h ^= h >> np.uint32(15)
    return h


def hash_input(dt, w, h, cn, seed):
    """Element (x, y, c) from Hash(x + 131·seed, y·cn + c): u8 takes the low 8 bits, u16 the low
    16 bits, i16 the low 16 bits as a signed value; f32/f64 ((h & 0xFFFF) / 65536 − 0.5) · 200
    rounded to float32 (then widened for f64)."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.uint32)
    planes = []
    for c in range(cn):
        hv = hash_u32(xs + np.uint32(131 * seed), ys * np.uint32(cn) + np.uint32(c))
        if dt == "u8":
            planes.append((hv & np.uint32(255)).astype(np.uint8))
        elif dt == "u16":
            planes.append((hv & np.uint32(0xFFFF)).astype(np.uint16))
        elif dt == "i16":
            planes.append((hv & np.uint32(0xFFFF)).astype(np.uint16).view(np.int16))
        else:
            f = (((hv & np.uint32(0xFFFF)).astype(np.float64) / 65536.0 - 0.5) * 200.0).astype(np.float32)
            planes.append(f.astype(DTYPES[dt]))
    a = np.dstack(planes) if cn > 1 else planes[0]
    return np.ascontiguousarray(a)


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

    def expected(self, out):
        out = np.ascontiguousarray(out)
        if out.size > 4096:
            return {"sha": sha(out), "dtype": dt_of(out), "width": int(out.shape[1]), "height": int(out.shape[0])}
        return {"array": self.ref(out)}

    def add(self, name, op, src, params, outs, gen=None):
        assert name not in self.names, name
        self.names.add(name)
        case = {"name": name, "op": op, "params": params, "suspect": name in SUSPECT_CASES}
        if case["suspect"]:
            case["suspectReason"] = SUSPECT_CASES[name]
        if gen is not None:
            case["gen"] = gen
        else:
            case["input"] = self.ref(src)
        case["outputs"] = [self.expected(o) for o in outs]
        self.cases.append(case)


def rand(rng, dt, h, w, cn, scale=40.0):
    shape = (h, w) if cn == 1 else (h, w, cn)
    if dt == "u8":
        return rng.integers(0, 256, shape, dtype=np.uint8)
    if dt == "u16":
        return rng.integers(0, 65536, shape, dtype=np.uint16)
    if dt == "i16":
        return rng.integers(-32768, 32768, shape, dtype=np.int16)
    if dt == "f64":
        return rng.standard_normal(shape) * scale
    return (rng.standard_normal(shape) * scale).astype(np.float32)


SIZES = [(1, 1), (2, 2), (3, 3), (1, 5), (5, 1), (2, 7), (4, 7), (7, 4), (13, 9), (37, 23), (40, 32), (67, 11)]


def pyrdown(g, rng):
    for dt in DTYPES:
        for cn in (1, 2, 3, 4):
            for (w, h) in SIZES:
                src = rand(rng, dt, h, w, cn)
                out = cv2.pyrDown(src)
                assert out.shape[:2] == ((h + 1) // 2, (w + 1) // 2)
                g.add(f"down_{dt}_c{cn}_{w}x{h}", "pyrDown", src, {"border": "Reflect101", "dw": 0, "dh": 0}, [out])
            for border in ("Reflect", "Replicate", "Wrap"):
                for (w, h) in ((5, 4), (13, 9), (40, 32)):
                    src = rand(rng, dt, h, w, cn)
                    out = cv2.pyrDown(src, borderType=BORDERS[border])
                    g.add(f"down_{dt}_c{cn}_{w}x{h}_{border}", "pyrDown", src, {"border": border, "dw": 0, "dh": 0}, [out])

    # Float widths around the 128-bit vector blocks of every channel count.
    for dt in ("f32", "f64"):
        for cn in (1, 2, 3, 4):
            for w in range(16, 30):
                src = rand(rng, dt, 6, w, cn)
                out = cv2.pyrDown(src)
                g.add(f"down_{dt}_c{cn}_w{w}", "pyrDown", src, {"border": "Reflect101", "dw": 0, "dh": 0}, [out])

    # Custom destination sizes: |2·d − s| ≤ 2.
    custom = [((8, 8), (5, 5)), ((8, 8), (3, 3)), ((9, 9), (4, 5)), ((10, 6), (6, 4)), ((11, 7), (6, 3)),
              ((4, 4), (3, 3)), ((2, 2), (2, 2)), ((1, 1), (1, 1)), ((2, 1), (2, 1)), ((3, 3), (2, 1)),
              ((20, 14), (11, 8)), ((21, 15), (10, 7)), ((6, 2), (4, 2))]
    for dt in DTYPES:
        for cn in (1, 3):
            for (w, h), (dw, dh) in custom:
                src = rand(rng, dt, h, w, cn)
                for border in ("Reflect101", "Replicate"):
                    out = cv2.pyrDown(src, dstsize=(dw, dh), borderType=BORDERS[border])
                    assert out.shape[:2] == (dh, dw)
                    g.add(f"down_{dt}_c{cn}_{w}x{h}_to_{dw}x{dh}_{border}", "pyrDown", src,
                          {"border": border, "dw": dw, "dh": dh}, [out])

    # Saturation extremes.
    for dt, lo, hi in (("u8", 0, 255), ("u16", 0, 65535), ("i16", -32768, 32767)):
        for cn in (1, 3):
            src = np.full((9, 11, cn) if cn > 1 else (9, 11), hi, DTYPES[dt])
            g.add(f"down_{dt}_c{cn}_max", "pyrDown", src, {"border": "Reflect101", "dw": 0, "dh": 0}, [cv2.pyrDown(src)])
            src = rng.choice(np.array([lo, hi], DTYPES[dt]), size=(17, 19, cn) if cn > 1 else (17, 19))
            g.add(f"down_{dt}_c{cn}_extremes", "pyrDown", src, {"border": "Reflect101", "dw": 0, "dh": 0}, [cv2.pyrDown(src)])


def pyrup(g, rng):
    for dt in DTYPES:
        for cn in (1, 2, 3, 4):
            for (w, h) in SIZES:
                src = rand(rng, dt, h, w, cn)
                out = cv2.pyrUp(src)
                assert out.shape[:2] == (2 * h, 2 * w)
                g.add(f"up_{dt}_c{cn}_{w}x{h}", "pyrUp", src, {"border": "Reflect101", "dw": 0, "dh": 0}, [out])
    for dt in ("f32", "f64"):
        for cn in (1, 2, 3, 4):
            for w in range(5, 12):
                src = rand(rng, dt, 3, w, cn)
                g.add(f"up_{dt}_c{cn}_w{w}", "pyrUp", src, {"border": "Reflect101", "dw": 0, "dh": 0}, [cv2.pyrUp(src)])

    # Custom destination sizes 2n − 1, 2n, 2n + 1 (no one-column source with three columns).
    for dt in DTYPES:
        for cn in (1, 3):
            for (w, h) in ((1, 1), (2, 1), (1, 2), (2, 3), (5, 4), (9, 7), (17, 6)):
                for ddw in (-1, 0, 1):
                    for ddh in (-1, 0, 1):
                        if ddw == 0 and ddh == 0:
                            continue
                        dw, dh = 2 * w + ddw, 2 * h + ddh
                        if w == 1 and dw == 3:
                            continue
                        src = rand(rng, dt, h, w, cn)
                        out = cv2.pyrUp(src, dstsize=(dw, dh))
                        assert out.shape[:2] == (dh, dw)
                        g.add(f"up_{dt}_c{cn}_{w}x{h}_to_{dw}x{dh}", "pyrUp", src,
                              {"border": "Reflect101", "dw": dw, "dh": dh}, [out])

    for dt, lo, hi in (("u8", 0, 255), ("u16", 0, 65535), ("i16", -32768, 32767)):
        for cn in (1, 3):
            src = rng.choice(np.array([lo, hi], DTYPES[dt]), size=(9, 13, cn) if cn > 1 else (9, 13))
            g.add(f"up_{dt}_c{cn}_extremes", "pyrUp", src, {"border": "Reflect101", "dw": 0, "dh": 0}, [cv2.pyrUp(src)])


def roi(g, rng):
    for dt in DTYPES:
        for cn in (1, 3):
            parent = rand(rng, dt, 30, 41, cn)
            x, y, w, h = 3, 5, 27, 19
            view = parent[y:y + h, x:x + w]
            params = {"border": "Reflect101", "dw": 0, "dh": 0, "roi": [x, y, w, h]}
            g.add(f"roi_down_{dt}_c{cn}", "pyrDown", parent, params, [cv2.pyrDown(view)])
            g.add(f"roi_up_{dt}_c{cn}", "pyrUp", parent, params, [cv2.pyrUp(view)])
            p2 = dict(params, border="Reflect")
            g.add(f"roi_down_{dt}_c{cn}_Reflect", "pyrDown", parent, p2, [cv2.pyrDown(view, borderType=cv2.BORDER_REFLECT)])


def build_pyramid(src, levels, border=cv2.BORDER_REFLECT_101):
    """cv2 has no buildPyramid binding; OpenCV buildPyramid without IPP calls pyrDown per level."""
    outs = [src]
    for _ in range(levels):
        outs.append(cv2.pyrDown(outs[-1], borderType=border))
    return outs


def build(g, rng):
    for dt in DTYPES:
        for cn in (1, 3):
            src = rand(rng, dt, 37, 45, cn)
            for border, levels in (("Reflect101", 4), ("Replicate", 3), ("Wrap", 6)):
                outs = build_pyramid(src, levels, BORDERS[border])
                assert len(outs) == levels + 1
                assert np.array_equal(outs[0], src)
                g.add(f"build_{dt}_c{cn}_{border}_{levels}", "buildPyramid", src,
                      {"border": border, "levels": levels}, list(outs[1:]))


def large(g):
    for dt in DTYPES:
        for cn in (1, 3, 4):
            for (w, h) in ((1031, 517), (640, 481)):
                seed = 7 * w + cn
                src = hash_input(dt, w, h, cn, seed)
                gen = {"dtype": dt, "width": w, "height": h, "channels": cn, "seed": seed}
                g.add(f"large_down_{dt}_c{cn}_{w}x{h}", "pyrDown", None,
                      {"border": "Reflect101", "dw": 0, "dh": 0}, [cv2.pyrDown(src)], gen=gen)
                if w == 640:
                    g.add(f"large_up_{dt}_c{cn}_{w}x{h}", "pyrUp", None,
                          {"border": "Reflect101", "dw": 0, "dh": 0}, [cv2.pyrUp(src)], gen=gen)
    src = hash_input("u8", 1024, 1024, 3, 99)
    g.add("large_build_u8_c3_1024", "buildPyramid", None, {"border": "Reflect101", "levels": 4},
          build_pyramid(src, 4)[1:], gen={"dtype": "u8", "width": 1024, "height": 1024, "channels": 3, "seed": 99})


def keypoints(rng):
    """KeyPoint.overlap on random and degenerate pairs; float32 results stored as bits."""
    pairs = []
    fixed = [
        ((0, 0, 10), (0, 0, 10)), ((0, 0, 10), (100, 0, 10)), ((0, 0, 10), (5, 0, 10)),
        ((0, 0, 10), (1, 1, 4)), ((3.5, -2.25, 7), (4.0, -1.0, 3)), ((0, 0, 0), (0, 0, 0)),
        ((0, 0, 10), (10, 0, 10)), ((1, 2, 6), (4, 6, 4)), ((0, 0, 1e-3), (1e-4, 0, 1e-3)),
    ]
    for (x1, y1, s1), (x2, y2, s2) in fixed:
        pairs.append((np.float32(x1), np.float32(y1), np.float32(s1), np.float32(x2), np.float32(y2), np.float32(s2)))
    for _ in range(300):
        v = rng.standard_normal(6).astype(np.float32)
        x1, y1, x2, y2 = v[0] * 8, v[1] * 8, v[2] * 8, v[3] * 8
        s1, s2 = np.float32(abs(v[4]) * 12 + 0.5), np.float32(abs(v[5]) * 12 + 0.5)
        pairs.append((np.float32(x1), np.float32(y1), s1, np.float32(x2), np.float32(y2), s2))
    out = []
    with np.errstate(all="ignore"):
        for p in pairs:
            k1 = cv2.KeyPoint(float(p[0]), float(p[1]), float(p[2]))
            k2 = cv2.KeyPoint(float(p[3]), float(p[4]), float(p[5]))
            r = np.float32(cv2.KeyPoint_overlap(k1, k2))
            out.append([int(np.array(q, np.float32).view(np.uint32)) for q in p] + [int(np.array(r, np.float32).view(np.uint32))])
    # convert with indices and the reverse conversion defaults.
    kps = [cv2.KeyPoint(float(i) * 1.5, float(-i) * 0.25, 3.0, 10.0 * i, 0.5, i, i + 1) for i in range(6)]
    pts = cv2.KeyPoint_convert(kps, [4, 0, 4, 2])
    back = cv2.KeyPoint_convert(np.array([[1.25, 2.5], [-3.0, 4.75]], np.float32), 2.0, 0.75, 3, 9)
    conv = {"indexed": [[float(p[0]), float(p[1])] for p in pts],
            "reverse": [[k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id] for k in back]}
    return {"overlap": out, "convert": conv}


def main():
    rng = np.random.default_rng(20261015)
    g = Gen()
    pyrdown(g, rng)
    pyrup(g, rng)
    roi(g, rng)
    build(g, rng)
    large(g)
    kp = keypoints(np.random.default_rng(20261016))
    assert set(SUSPECT_CASES) <= g.names, set(SUSPECT_CASES) - g.names
    doc = {"generator": {"cv2": cv2.__version__, "numpy": np.__version__, "machine": platform.processor()},
           "arrays": g.arrays, "cases": g.cases, "keypoints": kp}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TestData", "pyramid_golden.json")
    with open(out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases, {len(g.arrays)} arrays to {out}")


if __name__ == "__main__":
    main()
