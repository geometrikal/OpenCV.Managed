#!/usr/bin/env python3
"""Golden-data generator for the managed thresholding port.

Runs cv2 4.5.5 ``threshold`` (fixed, Otsu, triangle) and ``adaptiveThreshold``
over a seeded battery and writes ``TestData/threshold_golden.json``.
``ImgProc/ThresholdGoldenTests.cs`` replays each case.

Regenerate with:
    python tools/generate_threshold_golden.py
(run from the ``Geometrikal.OpenCV.Managed.Tests`` directory).

Environment and battery constraints:
 * OPENCV_CPU_DISABLE removes AVX/AVX2/FMA dispatch. The 32F Gaussian of
   adaptiveThreshold is FMA-dependent, and exact .5 means occur.
 * IPP is disabled: the IPP 32F TRUNC/TOZERO/TOZERO_INV paths give other
   NaN results than the OpenCV 5.0 vector code that the port follows.
 * 4.5.5 computes the scalar tail of 32F/64F TRUNC with std::min, which
   differs from the vector body for equal-valued zeros of opposite sign.
   TRUNC cases with a zero threshold therefore give every zero source the
   sign of the threshold.
 * thresholdWithMask and THRESH_DRYRUN do not exist in 4.5.5. They are
   covered by analytic tests in ThresholdBasicTests.
 * The large Otsu case stores no pixels: both sides synthesize the image
   from the formula in ``synthetic_otsu`` and compare the threshold only.
 * Input and output arrays are stored once in a shared ``arrays`` list.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import json
import math
import os

os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX,AVX2,AVX512F,FMA3")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

SUSPECT_CASES = {}

DTYPES = {"u8": np.uint8, "i16": np.int16, "u16": np.uint16, "f32": np.float32, "f64": np.float64}
TYPE_NAMES = ["bin", "binv", "trunc", "tozero", "tozinv"]
FLT_MAX = float(np.finfo(np.float32).max)
NAN, INF = float("nan"), float("inf")

INT_THRESHOLDS = {
    "u8": [-1e12, -INF, NAN, INF, 1e12, 3e9, -1, -0.5, -0.0, 0, 0.5, 1, 127, 127.5, 253.9, 254,
           254.5, 254.999, 255, 255.5, 256],
    "i16": [-32769, -32768.5, -32768, -32767.5, -0.5, 0, 100.25, 32766, 32766.5, 32767, 32768,
            NAN, 1e12, -1e12],
    "u16": [-1, -0.5, 0, 0.5, 1000.75, 65534, 65534.5, 65535, 65536, NAN, 1e12, -1e12],
}
FLOAT_THRESHOLDS = [0.1, 0.3, 1.5, -1.5, 0.0, -0.0, FLT_MAX, -FLT_MAX, 1e300, INF, -INF, NAN]
MAXVALS = [0, 1, 127.5, 128.5, 254.5, 255, 256, 300, -5, -1e6, 1e6, 32767, 40000, 65535.5, NAN, INF]
FLOAT_MAXVALS = [255, 1, -5, 0.3, 1e300, NAN, INF, -0.0]


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def jnum(v):
    v = float(v)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    return v


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

    def add(self, name, op, inputs, params, expected=None, values=None):
        assert name not in self.names, name
        self.names.add(name)
        case = {"name": name, "op": op, "inputs": [self.ref(a) for a in inputs],
                "params": {k: jnum(v) if isinstance(v, float) else v for k, v in params.items()},
                "suspect": name in SUSPECT_CASES}
        if case["suspect"]:
            case["suspectReason"] = SUSPECT_CASES[name]
        if expected is not None:
            case["expected"] = self.ref(expected)
        if values is not None:
            case["values"] = [jnum(v) for v in values]
        self.cases.append(case)


def int_source(rng, dt, cn, t, w, h):
    info = np.iinfo(DTYPES[dt])
    lo, hi = int(info.min), int(info.max)
    shape = (h, w) if cn == 1 else (h, w, cn)
    a = rng.integers(lo, hi, size=shape, endpoint=True, dtype=np.int64)
    flat = a.reshape(-1)
    if math.isfinite(t):
        f = math.floor(t)
        near = rng.integers(f - 2, f + 3, size=flat.size // 3)
        pos = rng.choice(flat.size, size=near.size, replace=False)
        flat[pos] = near
    specials = [lo, hi, lo + 1, hi - 1, 0]
    if math.isfinite(t):
        specials += [math.floor(t), math.floor(t) + 1]
    for i, v in enumerate(specials):
        flat[(i * 7) % flat.size] = v
    return np.clip(a, lo, hi).astype(DTYPES[dt])


def float_source(rng, dt, cn, t, w, h, zero_sign=None):
    shape = (h, w) if cn == 1 else (h, w, cn)
    a = (rng.standard_normal(shape) * 3).astype(DTYPES[dt])
    flat = a.reshape(-1)
    with np.errstate(over="ignore"):
        tv = DTYPES[dt](t) if not math.isnan(t) else DTYPES[dt](0.5)
        near = [tv, np.nextafter(tv, DTYPES[dt](INF)), np.nextafter(tv, DTYPES[dt](-INF))]
    specials = [NAN, INF, -INF, -0.0, 0.0, FLT_MAX, -FLT_MAX, *near, 0.3, 0.1]
    for i, v in enumerate(specials):
        flat[(i * 5 + 1) % flat.size] = v
        flat[(flat.size - 1 - i * 3) % flat.size] = v
    if zero_sign is not None:
        flat[flat == 0] = math.copysign(0.0, zero_sign)
    return a


def thresholds(g, rng):
    k = 0
    for dt in ("u8", "i16", "u16"):
        for t in INT_THRESHOLDS[dt]:
            for ty in range(5):
                cn = (1, 3)[k % 2]
                w, h = ((37, 3), (67, 2))[(k // 2) % 2]
                mv = MAXVALS[k % len(MAXVALS)]
                k += 1
                src = int_source(rng, dt, cn, t, w, h)
                ret, out = cv2.threshold(src, t, mv, ty)
                g.add(f"thr_{dt}_c{cn}_{TYPE_NAMES[ty]}_t{t!r}_m{mv!r}", "threshold", [src],
                      {"thresh": float(t), "maxval": float(mv), "type": ty}, out, [ret])

    for dt in ("f32", "f64"):
        for t in FLOAT_THRESHOLDS:
            for ty in range(5):
                cn = (1, 3)[k % 2]
                w, h = ((37, 3), (67, 2))[(k // 2) % 2]
                mv = FLOAT_MAXVALS[k % len(FLOAT_MAXVALS)]
                k += 1
                zero_sign = t if ty == cv2.THRESH_TRUNC and t == 0 else None
                src = float_source(rng, dt, cn, t, w, h, zero_sign)
                ret, out = cv2.threshold(src, t, mv, ty)
                g.add(f"thr_{dt}_c{cn}_{TYPE_NAMES[ty]}_t{t!r}_m{mv!r}", "threshold", [src],
                      {"thresh": float(t), "maxval": float(mv), "type": ty}, out, [ret])

    for dt, cn, w, h in (("u8", 1, 1, 1), ("u8", 4, 17, 1), ("u16", 1, 1, 5), ("f32", 4, 1, 1),
                         ("f64", 1, 1, 17), ("i16", 2, 3, 3)):
        for ty in range(5):
            src = (int_source(rng, dt, cn, 100, w, h) if dt in INT_THRESHOLDS
                   else float_source(rng, dt, cn, 0.5, w, h))
            ret, out = cv2.threshold(src, 100.5 if dt in INT_THRESHOLDS else 0.25, 200, ty)
            g.add(f"thr_small_{dt}_c{cn}_{w}x{h}_{TYPE_NAMES[ty]}", "threshold", [src],
                  {"thresh": 100.5 if dt in INT_THRESHOLDS else 0.25, "maxval": 200.0, "type": ty}, out, [ret])


def bimodal(rng, dt, w, h, m1, m2, s1, s2, frac):
    hi = 255 if dt == "u8" else 65535
    n = w * h
    k = int(n * frac)
    v = np.concatenate([rng.normal(m1, s1, k), rng.normal(m2, s2, n - k)])
    rng.shuffle(v)
    return np.clip(np.round(v), 0, hi).astype(DTYPES[dt]).reshape(h, w)


def auto_images(rng):
    u8 = {
        "bimodal_a": bimodal(rng, "u8", 67, 23, 60, 180, 15, 20, 0.5),
        "bimodal_b": bimodal(rng, "u8", 37, 31, 30, 120, 8, 30, 0.8),
        "bimodal_c": bimodal(rng, "u8", 64, 64, 200, 230, 5, 5, 0.3),
        "uniform": rng.integers(0, 256, (29, 37)).astype(np.uint8),
        "constant": np.full((5, 7), 77, np.uint8),
        "zeros": np.zeros((4, 9), np.uint8),
        "full": np.full((4, 9), 255, np.uint8),
        "twolevel": np.where(rng.random((9, 13)) < 0.4, 40, 200).astype(np.uint8),
        "threelevel": rng.choice(np.array([10, 100, 190], np.uint8), (11, 17)),
        "single": np.array([[123]], np.uint8),
        "row": rng.integers(0, 256, (1, 67)).astype(np.uint8),
        "col": rng.integers(0, 256, (67, 1)).astype(np.uint8),
        "leftskew": np.clip(np.round(rng.exponential(20, (31, 37))), 0, 255).astype(np.uint8),
        "rightskew": (255 - np.clip(np.round(rng.exponential(20, (31, 37))), 0, 255)).astype(np.uint8),
        "peak0": np.where(rng.random((19, 23)) < 0.7, 0, rng.integers(0, 256, (19, 23))).astype(np.uint8),
        "peak255": np.where(rng.random((19, 23)) < 0.7, 255, rng.integers(0, 256, (19, 23))).astype(np.uint8),
        "twovalues": np.array([[0, 0, 200, 200]], np.uint8),
    }
    u16 = {
        "bimodal16": bimodal(rng, "u16", 67, 19, 10000, 40000, 3000, 5000, 0.5),
        "uniform16": rng.integers(0, 65536, (23, 37)).astype(np.uint16),
        "fewlevel16": (rng.integers(0, 6, (17, 29)) * 9000).astype(np.uint16),
    }
    return u8, u16


def synthetic_otsu(w, h):
    y, x = np.mgrid[0:h, 0:w]
    img = (50 + (x * 7 + y * 13) % 200).astype(np.uint8)
    for i in range(10):
        img[(i * 991) % h, (i * 997) % w] = i
    return img


def automatic(g, rng):
    u8, u16 = auto_images(rng)
    k = 0
    for flag, fname in ((cv2.THRESH_OTSU, "otsu"), (cv2.THRESH_TRIANGLE, "triangle")):
        for name, img in u8.items():
            for ty in range(5):
                mv = (255, 99)[k % 2]
                k += 1
                ret, out = cv2.threshold(img, 0, mv, ty | flag)
                g.add(f"{fname}_u8_{name}_{TYPE_NAMES[ty]}", "threshold", [img],
                      {"thresh": 0.0, "maxval": float(mv), "type": ty | flag}, out, [ret])
    for name, img in u16.items():
        for ty in range(5):
            mv = (65535, 999)[k % 2]
            k += 1
            ret, out = cv2.threshold(img, 0, mv, ty | cv2.THRESH_OTSU)
            g.add(f"otsu_u16_{name}_{TYPE_NAMES[ty]}", "threshold", [img],
                  {"thresh": 0.0, "maxval": float(mv), "type": ty | cv2.THRESH_OTSU}, out, [ret])

    big = synthetic_otsu(3001, 3002)
    ret, _ = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    g.add("otsu_u8_synthetic_3001x3002", "otsuSynthetic", [], {"width": 3001, "height": 3002}, values=[ret])
    return u8


def adaptive_images(rng):
    smooth = cv2.GaussianBlur(rng.integers(0, 256, (23, 37)).astype(np.uint8), (7, 7), 2)
    y, x = np.mgrid[0:29, 0:41]
    return {
        "random": rng.integers(0, 256, (23, 37)).astype(np.uint8),
        "smooth": smooth,
        "gradient": ((x * 5 + y * 3) % 256).astype(np.uint8),
        "blocks": np.where(((x // 6) + (y // 5)) % 2 == 0, 90, 160).astype(np.uint8)
                  + rng.integers(0, 3, (29, 41)).astype(np.uint8),
    }


def adaptive(g, rng):
    imgs = adaptive_images(rng)
    names = list(imgs)
    cs = [0, 2, -3, 2.5, -2.5, 0.4, -0.4, 7.9, 1e12, -1e12, NAN]
    maxvals = [255, 1, 0.49, 100.5, 101.5, 255.5, 300, -0.0, NAN, INF, -1]
    k = 0

    def run(tag, img, mv, method, ty, bs, c):
        out = cv2.adaptiveThreshold(img, mv, method, ty, bs, c)
        g.add(f"adapt_{tag}_{('mean', 'gauss')[method]}_{TYPE_NAMES[ty]}_b{bs}_c{c!r}_m{mv!r}",
              "adaptive", [img], {"maxval": float(mv), "method": method, "type": ty,
                                  "blockSize": bs, "c": float(c)}, out)

    for method in (0, 1):
        for ty in (0, 1):
            for bs in (3, 5, 7, 9, 11, 15, 17, 21, 31, 51):
                name = names[k % len(names)]
                run(name, imgs[name], 255, method, ty, bs, cs[k % len(cs)])
                k += 1
            for c in cs:
                name = names[k % len(names)]
                run(name, imgs[name], 255, method, ty, (5, 11)[k % 2], c)
                k += 1
            for mv in maxvals:
                name = names[k % len(names)]
                run(name, imgs[name], mv, method, ty, 7, 1.5)
                k += 1
            for tag, img in (("single", np.array([[77]], np.uint8)),
                             ("row", rng.integers(0, 256, (1, 67)).astype(np.uint8)),
                             ("col", rng.integers(0, 256, (37, 1)).astype(np.uint8)),
                             ("tiny", rng.integers(0, 256, (5, 5)).astype(np.uint8))):
                run(tag, img, 200, method, ty, 9, 1)
            big = rng.integers(0, 256, (150, 150)).astype(np.uint8)
            run("150", big, 255, method, ty, 31, -1)


def sanity():
    a = np.array([[0, 0, 200, 200]], np.uint8)
    r, o = cv2.threshold(a, 0, 255, cv2.THRESH_TRUNC | cv2.THRESH_OTSU)
    assert r == 0 and o.tolist() == [[0, 0, 0, 0]], (r, o)
    r, _ = cv2.threshold(np.zeros((3, 3), np.uint8), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    assert r == 1, r
    r, o = cv2.threshold(np.array([[5]], np.uint8), NAN, 255, cv2.THRESH_BINARY)
    assert r == -2147483648.0 and o[0, 0] == 255, (r, o)
    r, o = cv2.threshold(np.array([[5]], np.uint8), -INF, 255, cv2.THRESH_BINARY)
    assert r == 2147483647.0 and o[0, 0] == 0, (r, o)
    for bad in (lambda: cv2.threshold(np.zeros((3, 3, 3), np.uint8), 0, 255, cv2.THRESH_OTSU),
                lambda: cv2.threshold(np.zeros((3, 3), np.float32), 0, 255, cv2.THRESH_OTSU),
                lambda: cv2.threshold(np.zeros((3, 3), np.uint16), 0, 255, cv2.THRESH_TRIANGLE),
                lambda: cv2.adaptiveThreshold(np.zeros((3, 3), np.uint8), 255, 0, 0, 4, 0),
                lambda: cv2.adaptiveThreshold(np.zeros((3, 3), np.uint8), 255, 0, 2, 3, 0)):
        try:
            bad()
        except cv2.error:
            continue
        raise AssertionError("expected cv2.error")
    z = cv2.adaptiveThreshold(np.full((3, 3), 9, np.uint8), -1, 5, 2, 3, 0)
    assert not z.any()


def main():
    rng = np.random.default_rng(20261012)
    sanity()
    g = Gen()
    thresholds(g, rng)
    automatic(g, rng)
    adaptive(g, rng)
    assert set(SUSPECT_CASES) <= g.names, set(SUSPECT_CASES) - g.names

    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "threshold_golden.json")
    with open(out, "w") as f:
        json.dump({"arrays": g.arrays, "cases": g.cases}, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases, {len(g.arrays)} arrays to {out}")


if __name__ == "__main__":
    main()
