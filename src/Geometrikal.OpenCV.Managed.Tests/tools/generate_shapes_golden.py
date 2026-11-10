#!/usr/bin/env python3
"""Golden-data generator for the managed shape-descriptor port.

Runs cv2 5.0.0 approxPolyDP, arcLength, contourArea, boundingRect (point sets and masks),
moments (contours and images), HuMoments and matchShapes over a seeded battery and writes
``TestData/shapes_golden.json``. ``ImgProc/ShapeDescriptorsGoldenTests.cs`` replays each case.

Regenerate with (from the ``Geometrikal.OpenCV.Managed.Tests`` directory):
    PYTHONPATH=<opencv-contrib-python-headless==5.0.0 install> python tools/generate_shapes_golden.py

Environment and battery constraints:
 * cv2 must be 5.0.0. IPP, OpenCL and threads are disabled. OPENCV_CPU_DISABLE removes the
   AVX/AVX2/AVX-512 dispatch; the shape functions use baseline code only, so it has no effect on
   the results.
 * Contours come from findContours of seeded random and analytic masks. Float point sets are
   seeded random sets, scaled contours and noisy circles. Degenerate sets (1 and 2 points,
   collinear points, duplicate points) are included. cv2 rejects empty point sets in some
   functions; the generator records only the calls that succeed.
 * cv2.moments treats a continuous single-channel float or int32 array with two columns as a point
   vector. Float images in the battery therefore never have width 2.
 * NaN coordinates are excluded: the cv2 vector min/max of boundingRect has platform NaN rules.
 * ROI cases store the parent array and apply "roi" ([x, y, w, h]).
 * Values are Python floats; JSON keeps their shortest round-trip form, so the C# runner compares
   doubles exactly.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import json
import math
import os
import platform
import sys

os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

if cv2.__version__ != "5.0.0":
    sys.exit(f"generate_shapes_golden.py requires cv2 5.0.0 (found {cv2.__version__}); "
             "set PYTHONPATH to the opencv-contrib-python-headless==5.0.0 install")

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

DTYPES = {"u8": np.uint8, "u16": np.uint16, "i16": np.int16, "i32": np.int32, "f32": np.float32, "f64": np.float64}
MOMENT_KEYS = ["m00", "m10", "m01", "m20", "m11", "m02", "m30", "m21", "m12", "m03",
               "mu20", "mu11", "mu02", "mu30", "mu21", "mu12", "mu03",
               "nu20", "nu11", "nu02", "nu30", "nu21", "nu12", "nu03"]


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def dt_of(arr):
    for k, v in DTYPES.items():
        if arr.dtype == v:
            return k
    raise ValueError(arr.dtype)


def jnum(v):
    v = float(v)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    return v


class Gen:
    def __init__(self):
        self.cases = []
        self.names = set()
        self.arrays = []
        self.index = {}

    def ref(self, arr):
        arr = np.ascontiguousarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        m = {"dtype": dt_of(arr), "width": int(arr.shape[1]), "height": int(arr.shape[0]),
             "channels": 1 if arr.ndim == 2 else int(arr.shape[2]), "data": b64(arr)}
        key = (m["dtype"], m["width"], m["height"], m["channels"], m["data"])
        if key not in self.index:
            self.index[key] = len(self.arrays)
            self.arrays.append(m)
        return self.index[key]

    def add(self, name, op, inputs, params, values=None, expected=None):
        assert name not in self.names, name
        self.names.add(name)
        case = {"name": name, "op": op, "inputs": [self.ref(a) for a in inputs], "params": params}
        if values is not None:
            case["values"] = [jnum(v) for v in values]
        if expected is not None:
            case["expected"] = self.ref(expected)
        self.cases.append(case)


def pts(arr):
    """Returns an (N, 2) int32 or float32 point array stored as a 2-channel 1 x N array."""
    a = np.ascontiguousarray(np.asarray(arr).reshape(-1, 2))
    return a


def as_stored(p):
    # Point arrays are stored as width N, height 1, 2 channels.
    return p.reshape(1, -1, 2)


def moment_values(m):
    return [m[k] for k in MOMENT_KEYS]


# ---------------------------------------------------------------- inputs

def random_mask(rng, w, h, density, blur):
    noise = (rng.random((h, w)) * 255).astype(np.uint8)
    noise = cv2.GaussianBlur(noise, (0, 0), blur)
    thr = np.percentile(noise, 100 * (1 - density))
    return (noise > thr).astype(np.uint8) * 255


def analytic_masks():
    out = {}
    m = np.zeros((64, 80), np.uint8)
    cv2.circle(m, (30, 30), 18, 255, -1)
    cv2.rectangle(m, (55, 5), (75, 50), 255, -1)
    cv2.circle(m, (62, 30), 4, 0, -1)
    out["circleRect"] = m
    m = np.zeros((90, 90), np.uint8)
    cv2.ellipse(m, (45, 45), (35, 15), 30, 0, 360, 255, -1)
    out["ellipse"] = m
    m = np.zeros((70, 70), np.uint8)
    star = []
    for k in range(10):
        r = 30 if k % 2 == 0 else 12
        a = k * math.pi / 5
        star.append((int(round(35 + r * math.cos(a))), int(round(35 + r * math.sin(a)))))
    cv2.fillPoly(m, [np.array(star, np.int32)], 255)
    out["star"] = m
    m = np.zeros((40, 40), np.uint8)
    m[10:30, 10:11] = 255
    m[5, 5] = 255
    m[35, 20:36] = 255
    out["thin"] = m
    return out


def contour_battery(rng):
    contours = []
    masks = dict(analytic_masks())
    for k, (w, h, d, b) in enumerate([(96, 80, 0.4, 3.0), (120, 100, 0.3, 5.0), (64, 64, 0.5, 2.0)]):
        masks[f"rand{k}"] = random_mask(rng, w, h, d, b)
    for mname, mask in masks.items():
        for meth, mtag in [(cv2.CHAIN_APPROX_NONE, "none"), (cv2.CHAIN_APPROX_SIMPLE, "simple")]:
            cs, _ = cv2.findContours(mask, cv2.RETR_LIST, meth)
            cs = sorted(cs, key=lambda c: -len(c))
            for i, c in enumerate(cs[:4]):
                contours.append((f"{mname}_{mtag}{i}", pts(c).astype(np.int32)))
    return contours, masks


def float_battery(rng, int_contours):
    sets = []
    for name, c in int_contours[:6]:
        sets.append((f"f_{name}", (c.astype(np.float32) * np.float32(0.37) + np.float32(1.25)).astype(np.float32)))
    for k in range(3):
        n = [7, 40, 129][k]
        sets.append((f"f_rand{k}", (rng.random((n, 2)) * 200 - 50).astype(np.float32)))
    t = np.linspace(0, 2 * math.pi, 90, endpoint=False)
    circ = np.stack([60 + 40 * np.cos(t), 50 + 30 * np.sin(t)], 1) + rng.normal(0, 0.7, (90, 2))
    sets.append(("f_noisyCircle", circ.astype(np.float32)))
    sets.append(("f_negative", np.array([[-3.5, -2.25], [4.75, -7.5], [9.0, 3.0], [-1.5, 6.5]], np.float32)))
    return sets


def degenerate_battery():
    i = np.int32
    f = np.float32
    return [
        ("d_one", np.array([[5, 7]], i)),
        ("d_two", np.array([[1, 2], [9, 4]], i)),
        ("d_twoSame", np.array([[3, 3], [3, 3]], i)),
        ("d_collinear", np.array([[0, 0], [2, 2], [4, 4], [7, 7], [9, 9]], i)),
        ("d_collinearBack", np.array([[0, 0], [5, 0], [10, 0], [5, 0]], i)),
        ("d_duplicates", np.array([[0, 0], [0, 0], [10, 0], [10, 0], [10, 10], [10, 10], [0, 10], [0, 0]], i)),
        ("d_allSame", np.array([[4, 4]] * 6, i)),
        ("d_closedOpen", np.array([[0, 0], [10, 1], [20, 0], [20, 10], [0, 10], [0, 0]], i)),
        ("d_large", np.array([[100000, -200000], [3000000, 5], [-70000, 4000000], [16777217, 33554435]], i)),
        ("d_fone", np.array([[2.5, -1.5]], f)),
        ("d_ftwo", np.array([[0.1, 0.2], [0.30000001, 0.4]], f)),
        ("d_fcollinear", np.array([[0, 0], [0.5, 0.5], [1, 1], [1.5, 1.5]], f)),
        ("d_fduplicates", np.array([[1.5, 1.5], [1.5, 1.5], [8.25, 1.5], [8.25, 9.75], [1.5, 9.75]], f)),
    ]


# ---------------------------------------------------------------- cases

def try_call(fn, *args):
    try:
        return fn(*args), True
    except cv2.error:
        return None, False


def point_cases(g, name, p):
    stored = as_stored(p)
    n = len(p)
    r, ok = try_call(cv2.arcLength, p, True)
    if ok:
        g.add(f"arcLength_{name}_closed", "arcLength", [stored], {"closed": True}, values=[r])
    r, ok = try_call(cv2.arcLength, p, False)
    if ok:
        g.add(f"arcLength_{name}_open", "arcLength", [stored], {"closed": False}, values=[r])
    r, ok = try_call(cv2.contourArea, p, True)
    if ok:
        g.add(f"contourArea_{name}_oriented", "contourArea", [stored], {"oriented": True}, values=[r])
        r2 = cv2.contourArea(p, False)
        assert r2 == abs(r)
        g.add(f"contourArea_{name}_abs", "contourArea", [stored], {"oriented": False}, values=[r2])
    r, ok = try_call(cv2.boundingRect, p)
    if ok:
        g.add(f"boundingRect_{name}", "boundingRect", [stored], {}, values=list(r))
    r, ok = try_call(cv2.moments, p)
    if ok:
        hu = cv2.HuMoments(r).ravel()
        g.add(f"moments_{name}", "moments", [stored], {}, values=moment_values(r) + list(hu))
    for eps in [0.0, 0.5, 1.0, 2.0, 3.5, 10.0]:
        for closed in [True, False]:
            r, ok = try_call(cv2.approxPolyDP, p, eps, closed)
            if not ok:
                continue
            ctag = "closed" if closed else "open"
            params = {"epsilon": eps, "closed": closed}
            if r is None:
                g.add(f"approx_{name}_e{eps}_{ctag}", "approx", [stored], params, values=[0])
            else:
                out = r.reshape(-1, 2)
                assert out.dtype == p.dtype
                g.add(f"approx_{name}_e{eps}_{ctag}", "approx", [stored], params, values=[len(out)],
                      expected=as_stored(out))
    del n


def image_moment_cases(g, rng):
    def add(name, img, roi=None, binaries=(False, True)):
        view = img if roi is None else img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        for binary in binaries:
            m = cv2.moments(view, binary)
            hu = cv2.HuMoments(m).ravel()
            params = {"binary": binary}
            if roi is not None:
                params["roi"] = list(roi)
            g.add(f"imageMoments_{name}_{'bin' if binary else 'gray'}", "imageMoments", [img], params,
                  values=moment_values(m) + list(hu))

    add("u8_1x1", np.array([[200]], np.uint8))
    add("u8_zero", np.zeros((17, 23), np.uint8))
    add("u8_full255_64x64", np.full((64, 64), 255, np.uint8))
    for (w, h) in [(75, 53), (31, 33), (32, 32), (100, 64), (129, 1), (1, 70), (257, 97)]:
        add(f"u8_{w}x{h}", rng.integers(0, 256, (h, w), dtype=np.uint8))
    add("u8_sparse_90x70", (rng.random((70, 90)) < 0.05).astype(np.uint8) * rng.integers(1, 256, (70, 90), dtype=np.uint8))
    add("u8_roi", rng.integers(0, 256, (80, 120), dtype=np.uint8), roi=(13, 7, 77, 61))
    add("u16_75x53", rng.integers(0, 65536, (53, 75), dtype=np.uint16))
    add("u16_full_40x40", np.full((40, 40), 65535, np.uint16))
    add("u16_roi", rng.integers(0, 65536, (70, 90), dtype=np.uint16), roi=(5, 9, 66, 47))
    add("i16_75x53", rng.integers(-32768, 32768, (53, 75), dtype=np.int16))
    add("i16_min_33x35", np.full((35, 33), -32768, np.int16))
    add("i16_roi", rng.integers(-32768, 32768, (64, 64), dtype=np.int16), roi=(3, 3, 40, 37))
    f = (rng.random((53, 75)) * 10 - 3).astype(np.float32)
    f[f < 0] = 0
    add("f32_75x53", f)
    add("f32_signed_47x39", (rng.random((39, 47)) * 200 - 100).astype(np.float32))
    fz = (rng.random((40, 41)) * 5).astype(np.float32)
    fz[::3, ::2] = 0.0
    fz[1::5, ::3] = -0.0
    add("f32_zeros_41x40", fz)
    add("f32_roi", (rng.random((60, 70)) * 1000).astype(np.float32), roi=(7, 11, 50, 33))
    add("f64_75x53", rng.random((53, 75)) * 1e3)
    add("f64_signed_33x65", rng.random((65, 33)) * 2 - 1)
    add("f64_roi", rng.random((50, 50)), roi=(1, 2, 45, 40))
    masks = analytic_masks()
    add("mask_star", masks["star"])
    add("mask_ellipse_roi", masks["ellipse"], roi=(10, 20, 64, 50))


def mask_rect_cases(g, rng):
    def add(name, img, roi=None):
        view = img if roi is None else img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        r = cv2.boundingRect(view)
        params = {} if roi is None else {"roi": list(roi)}
        g.add(f"maskRect_{name}", "maskBoundingRect", [img], params, values=list(r))

    add("empty", np.zeros((30, 40), np.uint8))
    add("full", np.full((30, 41), 1, np.uint8))
    one = np.zeros((50, 67), np.uint8)
    one[23, 41] = 9
    add("single", one)
    corner = np.zeros((33, 70), np.uint8)
    corner[0, 0] = 1
    corner[32, 69] = 1
    add("corners", corner)
    for k, (w, h, p) in enumerate([(100, 80, 0.001), (257, 64, 0.0005), (63, 63, 0.02), (1000, 20, 0.0002)]):
        add(f"sparse{k}", ((rng.random((h, w)) < p) * rng.integers(1, 256, (h, w))).astype(np.uint8))
    for name, m in analytic_masks().items():
        add(f"analytic_{name}", m)
    big = np.zeros((90, 130), np.uint8)
    big[40:45, 3] = 255
    big[60, 100:120] = 255
    add("roi", big, roi=(2, 30, 101, 40))
    add("roiEmpty", big, roi=(10, 0, 50, 30))
    stripe = np.zeros((20, 300), np.uint8)
    stripe[5, 150] = 1
    stripe[15, 37] = 1
    stripe[10, 263] = 1
    add("stripe", stripe)


def match_cases(g, contours):
    pairs = [(0, 1), (2, 5), (3, 3), (7, 12), (10, 20)]
    for a, b in pairs:
        if a >= len(contours) or b >= len(contours):
            continue
        na, ca = contours[a]
        nb, cb = contours[b]
        for method in (1, 2, 3):
            r = cv2.matchShapes(ca, cb, method, 0)
            g.add(f"matchShapes_{na}_{nb}_m{method}", "matchShapes",
                  [as_stored(ca), as_stored(cb)], {"method": method}, values=[r])
    fa = np.array([[0, 0], [10.5, 0.25], [12, 9], [1, 11]], np.float32)
    fb = np.array([[0, 0], [21, 0.5], [24, 18], [2, 22]], np.float32)
    for method in (1, 2, 3):
        g.add(f"matchShapes_float_m{method}", "matchShapes", [as_stored(fa), as_stored(fb)],
              {"method": method}, values=[cv2.matchShapes(fa, fb, method, 0)])
    # One shape with all Hu moments 0 (degenerate) against a real one gives DBL_MAX.
    deg = np.array([[0, 0], [5, 5]], np.int32)
    for method in (1, 2, 3):
        g.add(f"matchShapes_degenerate_m{method}", "matchShapes", [as_stored(deg), as_stored(contours[0][1])],
              {"method": method}, values=[cv2.matchShapes(deg, contours[0][1], method, 0)])
    masks = analytic_masks()
    for method in (1, 2, 3):
        a = masks["star"]
        b = cv2.resize(masks["star"], (90, 77), interpolation=cv2.INTER_NEAREST)
        g.add(f"matchShapes_images_m{method}", "matchShapesImage", [a, b], {"method": method},
              values=[cv2.matchShapes(a, b, method, 0)])


def main():
    rng = np.random.default_rng(20261015)
    g = Gen()
    contours, _ = contour_battery(rng)
    for name, c in contours:
        point_cases(g, name, c)
    for name, c in float_battery(rng, contours):
        point_cases(g, name, c)
    for name, c in degenerate_battery():
        point_cases(g, name, c)
    image_moment_cases(g, rng)
    mask_rect_cases(g, rng)
    match_cases(g, contours)

    # Sanity: the analytic square has area 100 and a positive oriented area for screen-clockwise order.
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.int32)
    assert cv2.contourArea(sq, True) == 100.0
    assert cv2.boundingRect(sq) == (0, 0, 11, 11)

    doc = {"generator": {"cv2": cv2.__version__, "numpy": np.__version__, "machine": platform.processor()},
           "momentKeys": MOMENT_KEYS, "arrays": g.arrays, "cases": g.cases}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TestData", "shapes_golden.json")
    with open(out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases, {len(g.arrays)} arrays to {out}")


if __name__ == "__main__":
    main()
