#!/usr/bin/env python3
"""Golden-data generator for the managed core operations port.

Runs cv2 4.5.5 core functions (arithmetic, bitwise, compare, inRange,
conversions, LUT, copyTo, reductions) over a seeded battery and writes
``TestData/core_golden.json``. ``Core/CoreGoldenTests.cs`` replays each case.

Regenerate with:
    python tools/generate_core_golden.py
(run from the ``Geometrikal.OpenCV.Managed.Tests`` directory).

Environment and battery constraints:
 * OPENCV_CPU_DISABLE removes AVX/AVX2/FMA dispatch, so v_fma is a separate
   multiply and add. IPP, OpenCL and threads are disabled.
 * Scalars are always explicit 4-tuples (a bare number changes channel 0 only).
 * The port follows OpenCV 5.0. Where 4.5.5 differs, the battery avoids the
   input range (int multiply overflow, int divide operands above 2^12,
   int absdiff above INT_MAX, INT_MIN in norms, 16-bit reduce sums above
   2^24) or the case is listed in SUSPECT_CASES and pinned in
   CoreOpenCv5DifferenceTests.
 * 4.5.5 lane-dependent results are avoided: addWeighted uses dyadic
   weights; min/max inputs hold no NaN; 64F sources of scaled conversions to
   integer types hold float-representable values; 16U products stay below 2^31.
 * Reductions of floating sources are compared with a relative tolerance:
   the summation order is not reproducible.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import json
import os

os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX,AVX2,AVX512F,FMA3")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

SUSPECT_CASES = {
    f"multiply_f32_c{cn}_s{s}": "4.5.5 computes 32F multiply with scale != 1 in double; OpenCV 5.0 uses float"
    for cn in (1, 3) for s in (1 / 3, 0.1)
}

DTYPES = {
    "u8": np.uint8, "i8": np.int8, "u16": np.uint16, "i16": np.int16,
    "i32": np.int32, "f32": np.float32, "f64": np.float64,
}
CVDEPTH = {
    "u8": cv2.CV_8U, "i8": cv2.CV_8S, "u16": cv2.CV_16U, "i16": cv2.CV_16S,
    "i32": cv2.CV_32S, "f32": cv2.CV_32F, "f64": cv2.CV_64F,
}
W, H = 37, 9


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def mat(arr, dt):
    arr = np.ascontiguousarray(arr)
    assert arr.dtype == DTYPES[dt], (arr.dtype, dt)
    return {
        "dtype": dt, "width": int(arr.shape[1]), "height": int(arr.shape[0]),
        "channels": 1 if arr.ndim == 2 else int(arr.shape[2]), "data": b64(arr),
    }


def jnum(v):
    if np.isnan(v):
        return "NaN"
    if np.isinf(v):
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
        m = mat(arr, dt_of(arr))
        key = (m["dtype"], m["width"], m["height"], m["channels"], m["data"])
        if key not in self.index:
            self.index[key] = len(self.arrays)
            self.arrays.append(m)
        return self.index[key]

    def add(self, name, op, inputs, params, expected=None, values=None, tol=0.0):
        assert name not in self.names, name
        self.names.add(name)
        case = {"name": name, "op": op, "inputs": [self.ref(a) for a in inputs],
                "params": params, "suspect": name in SUSPECT_CASES, "tolerance": tol}
        if case["suspect"]:
            case["suspectReason"] = SUSPECT_CASES[name]
        if expected is not None:
            case["expected"] = self.ref(expected)
        if values is not None:
            case["values"] = [jnum(float(v)) for v in values]
        self.cases.append(case)


def rand(rng, dt, cn, lo=None, hi=None, w=W, h=H, specials=False):
    shape = (h, w) if cn == 1 else (h, w, cn)
    t = DTYPES[dt]
    if dt in ("f32", "f64"):
        a = (rng.standard_normal(shape) * 100).astype(t)
        if lo is not None:
            a = rng.uniform(lo, hi, shape).astype(t)
        if specials:
            flat = a.reshape(-1)
            flat[3] = np.nan
            flat[7] = np.inf
            flat[11] = -np.inf
            flat[13] = -0.0
        return a
    info = np.iinfo(t)
    lo = info.min if lo is None else lo
    hi = info.max if hi is None else hi
    a = rng.integers(lo, hi, size=shape, endpoint=True, dtype=np.int64).astype(t)
    flat = a.reshape(-1)
    flat[0], flat[1] = lo, hi
    return a


def mask_for(rng, w=W, h=H):
    m = (rng.integers(0, 3, size=(h, w)) > 0).astype(np.uint8)
    m[m > 0] = rng.integers(1, 256, size=int(m.sum()), dtype=np.int64).astype(np.uint8)
    return m


def arithmetic(g, rng):
    scalars = [(1.5, 2.5, -1.5, 0), (2, 300, -7, 0), (0.5, -0.5, 3.5, 0), (-3e9, 1e6, 2.25, 0)]
    for dt in DTYPES:
        for cn in (1, 3):
            lim = {"i32": (-(1 << 20), 1 << 20)}.get(dt, (None, None))
            a = rand(rng, dt, cn)
            b = rand(rng, dt, cn)
            g.add(f"add_{dt}_c{cn}", "add", [a, b], {}, cv2.add(a, b))
            g.add(f"subtract_{dt}_c{cn}", "subtract", [a, b], {}, cv2.subtract(a, b))
            ad, bd = rand(rng, dt, cn, *lim), rand(rng, dt, cn, *lim)
            g.add(f"absdiff_{dt}_c{cn}", "absdiff", [ad, bd], {}, cv2.absdiff(ad, bd))
            g.add(f"min_{dt}_c{cn}", "min", [a, b], {}, cv2.min(a, b))
            g.add(f"max_{dt}_c{cn}", "max", [a, b], {}, cv2.max(a, b))
            for si, s in enumerate(scalars):
                if dt == "i32" and abs(s[0]) > 1e9:
                    continue
                g.add(f"addS_{dt}_c{cn}_{si}", "addS", [a], {"scalar": list(s)}, cv2.add(a, s))
                g.add(f"subtractS_{dt}_c{cn}_{si}", "subtractS", [a], {"scalar": list(s)}, cv2.subtract(a, s))
                g.add(f"subtractSR_{dt}_c{cn}_{si}", "subtractSR", [a], {"scalar": list(s)}, cv2.subtract(s, a))
                if si != 3:  # |x - s| beyond INT_MAX: 4.5.5 and 5.0 differ (CoreOpenCv5DifferenceTests).
                    g.add(f"absdiffS_{dt}_c{cn}_{si}", "absdiffS", [ad], {"scalar": list(s)}, cv2.absdiff(ad, s))
                g.add(f"minS_{dt}_c{cn}_{si}", "minS", [a], {"scalar": list(s)}, cv2.min(a, s))
                g.add(f"maxS_{dt}_c{cn}_{si}", "maxS", [a], {"scalar": list(s)}, cv2.max(a, s))
                g.add(f"multiplyS_{dt}_c{cn}_{si}", "multiplyS", [a], {"scalar": list(s), "scale": 0.5},
                      cv2.multiply(a, s, scale=0.5))
                g.add(f"divideS_{dt}_c{cn}_{si}", "divideS", [a], {"scalar": list(s), "scale": 3.0},
                      cv2.divide(a, s, scale=3.0))
                g.add(f"divideSR_{dt}_c{cn}_{si}", "divideSR", [ad], {"scalar": list(s), "scale": 1.0},
                      cv2.divide(s, ad))

            # multiply: int operands small enough to avoid 32-bit overflow (wrap in 4.5.5,
            # double in 5.0); 16U products below 2^31.
            mlim = {"i32": (-30000, 30000), "u16": (0, 40000)}.get(dt, (None, None))
            ma, mb = rand(rng, dt, cn, *mlim), rand(rng, dt, cn, *mlim)
            scales = [1.0, 0.5, 2.0, 0.25] if dt == "i32" else [1.0, 0.5, 1 / 3, 0.1, 2.0]
            for s in scales:
                g.add(f"multiply_{dt}_c{cn}_s{s}", "multiply", [ma, mb], {"scale": s},
                      cv2.multiply(ma, mb, scale=s))

            # divide: zeros in the divisor; int operands below 2^12 (float in 4.5.5, double in 5.0).
            dlim = {"i32": (-4096, 4096)}.get(dt, (None, None))
            da = rand(rng, dt, cn, *dlim)
            db = rand(rng, dt, cn, *({"i32": (-200, 200)}.get(dt, dlim)))
            db.reshape(-1)[2::5] = 0
            for s in (1.0, 0.5, 1 / 3, 255.0):
                g.add(f"divide_{dt}_c{cn}_s{s}", "divide", [da, db], {"scale": s}, cv2.divide(da, db, scale=s))
            for s in (1.0, 1000.0):
                g.add(f"reciprocal_{dt}_c{cn}_s{s}", "reciprocal", [db], {"scale": s},
                      cv2.divide(scale=s, src2=db))

            # addWeighted / scaleAdd: dyadic weights; float inputs hold integers.
            wa, wb = a, b
            if dt in ("f32", "f64"):
                wa = rand(rng, dt, cn, -1000, 1000).round().astype(DTYPES[dt])
                wb = rand(rng, dt, cn, -1000, 1000).round().astype(DTYPES[dt])
            if dt == "i32":
                wa, wb = rand(rng, dt, cn, -(1 << 20), 1 << 20), rand(rng, dt, cn, -(1 << 20), 1 << 20)
            for (al, be, ga) in ((0.5, 0.5, 0.0), (0.75, 0.25, 3.0), (1.0, 1.0, 0.0), (0.125, 2.0, -2.25),
                                 (0.25, 1.0, 0.0)):
                g.add(f"addWeighted_{dt}_c{cn}_{al}_{be}_{ga}", "addWeighted", [wa, wb],
                      {"alpha": al, "beta": be, "gamma": ga}, cv2.addWeighted(wa, al, wb, be, ga))
            g.add(f"scaleAdd_{dt}_c{cn}", "scaleAdd", [wa, wb], {"alpha": 0.5}, cv2.scaleAdd(wa, 0.5, wb))

    # Masked add/subtract into an existing destination.
    for dt in ("u8", "i16", "f32"):
        for cn in (1, 3):
            a, b, d0 = rand(rng, dt, cn), rand(rng, dt, cn), rand(rng, dt, cn)
            m = mask_for(rng)
            g.add(f"addMask_{dt}_c{cn}", "addMask", [a, b, d0, m], {}, cv2.add(a, b, dst=d0.copy(), mask=m))
            g.add(f"subtractMask_{dt}_c{cn}", "subtractMask", [a, b, d0, m], {},
                  cv2.subtract(a, b, dst=d0.copy(), mask=m))
            s = (2.5, -1.5, 7, 0)
            g.add(f"addSMask_{dt}_c{cn}", "addSMask", [a, d0, m], {"scalar": list(s)},
                  cv2.add(a, s, dst=d0.copy(), mask=m))


def logic(g, rng):
    for dt in DTYPES:
        for cn in (1, 3):
            a, b = rand(rng, dt, cn), rand(rng, dt, cn)
            for op, fn in (("and", cv2.bitwise_and), ("or", cv2.bitwise_or), ("xor", cv2.bitwise_xor)):
                g.add(f"bitwise_{op}_{dt}_c{cn}", f"bitwise_{op}", [a, b], {}, fn(a, b))
                s = (3.7, 1.0, -2.0, 0)
                g.add(f"bitwise_{op}S_{dt}_c{cn}", f"bitwise_{op}S", [a], {"scalar": list(s)}, fn(a, s))
            g.add(f"bitwise_not_{dt}_c{cn}", "bitwise_not", [a], {}, cv2.bitwise_not(a))
            m = mask_for(rng)
            d0 = rand(rng, dt, cn)
            g.add(f"bitwise_andMask_{dt}_c{cn}", "bitwise_andMask", [a, b, d0, m], {},
                  cv2.bitwise_and(a, b, dst=d0.copy(), mask=m))

            ca = rand(rng, dt, cn, specials=True)
            cb = rand(rng, dt, cn, specials=True)
            cb.reshape(-1)[::4] = ca.reshape(-1)[::4]
            for op in range(6):
                g.add(f"compare_{dt}_c{cn}_op{op}", "compare", [ca, cb], {"cmp": op}, cv2.compare(ca, cb, op))
                for vi, v in enumerate((2.5, -1.0, 300.0, 0.0, 1e6, -2.5, 0.1)):
                    g.add(f"compareS_{dt}_c{cn}_op{op}_{vi}", "compareS", [ca], {"cmp": op, "value": v},
                          cv2.compare(ca, (v, v, v, v), op))

            lo, hi = rand(rng, dt, cn), rand(rng, dt, cn)
            lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
            g.add(f"inRange_{dt}_c{cn}", "inRange", [ca, lo, hi], {}, cv2.inRange(ca, lo, hi))
            for bi, (l, u) in enumerate((((1.5, -20.5, 0, 0), (3.5, 100.5, 50, 0)),
                                         ((2.5, 2.5, 2.5, 0), (2.5, 2.5, 2.5, 0)),
                                         ((5, 5, 5, 0), (1, 1, 1, 0)),
                                         ((-10, -1e6, -40, 0), (1000, 1e6, 120.5, 0)))):
                g.add(f"inRangeS_{dt}_c{cn}_{bi}", "inRangeS", [ca], {"lower": list(l), "upper": list(u)},
                      cv2.inRange(ca, l, u))

            m3 = mask_for(rng)
            g.add(f"copyToMask_{dt}_c{cn}", "copyToMask", [a, d0, m3], {}, cv2.copyTo(a, m3, d0.copy()))
            if cn == 3:
                mc = np.dstack([mask_for(rng) for _ in range(3)])
                g.add(f"copyToMaskCn_{dt}", "copyToMask", [a, d0, mc], {}, cv2.copyTo(a, mc, d0.copy()))


def conversions(g, rng):
    for sdt in DTYPES:
        for cn in (1, 3):
            src = rand(rng, sdt, cn)
            if sdt in ("f32", "f64"):
                src = rand(rng, sdt, cn, -400, 70000)
                src.reshape(-1)[:8] = [0.5, 1.5, 2.5, -0.5, -2.5, 127.5, 32767.5, 65535.5]
                if sdt == "f64":
                    src = src.astype(np.float32).astype(np.float64)
            for ddt in DTYPES:
                if ddt == sdt:
                    continue
                plain = cv2.add(src, np.zeros_like(src), dtype=CVDEPTH[ddt])
                g.add(f"convertTo_{sdt}_{ddt}_c{cn}", "convertTo", [src], {"to": ddt, "alpha": 1.0, "beta": 0.0}, plain)
                smin, smax = float(src.min()), float(src.max())
                for (da, db) in ((0.0, 255.0), (-100.0, 100.0)):
                    dmin, dmax = min(da, db), max(da, db)
                    scale = (dmax - dmin) * (1.0 / (smax - smin))
                    if ddt == "f32":
                        scale = float(np.float32(scale))
                        shift = float(np.float32(np.float32(dmin) - np.float32(smin * scale)))
                    else:
                        shift = dmin - smin * scale
                    out = cv2.normalize(src, None, da, db, cv2.NORM_MINMAX, dtype=CVDEPTH[ddt])
                    g.add(f"convertToScaled_{sdt}_{ddt}_c{cn}_{da}", "convertTo", [src],
                          {"to": ddt, "alpha": scale, "beta": shift}, out)
            for (al, be) in ((1.0, 0.0), (0.5, 0.25), (-1.3, 7.0)):
                g.add(f"convertScaleAbs_{sdt}_c{cn}_{al}_{be}", "convertScaleAbs", [src],
                      {"alpha": al, "beta": be}, cv2.convertScaleAbs(src, alpha=al, beta=be))

    img = rand(rng, "u8", 3)
    for tdt in ("u8", "i16", "f32", "f64"):
        tab = rand(rng, tdt, 1, w=256, h=1)
        g.add(f"lut_{tdt}", "lut", [img, tab], {}, cv2.LUT(img, tab))
        tab3 = rand(rng, tdt, 3, w=256, h=1)
        g.add(f"lut3_{tdt}", "lut", [img, tab3], {}, cv2.LUT(img, tab3))
    planes = [rand(rng, "u16", 1), rand(rng, "u16", 2), rand(rng, "u16", 1)]
    g.add("merge_u16", "merge", planes, {}, cv2.merge(planes))


def reductions(g, rng):
    for dt in DTYPES:
        ftol = 1e-9 if dt in ("f32", "f64", "i32") else 0.0
        for cn in (1, 3, 4):
            lim = {"i32": (-(1 << 30), 1 << 30)}.get(dt, (None, None))
            a = rand(rng, dt, cn, *lim)
            m = mask_for(rng)
            g.add(f"sum_{dt}_c{cn}", "sum", [a], {}, values=cv2.sumElems(a), tol=ftol)
            g.add(f"mean_{dt}_c{cn}", "mean", [a], {}, values=cv2.mean(a), tol=ftol)
            g.add(f"meanMask_{dt}_c{cn}", "meanMask", [a, m], {}, values=cv2.mean(a, mask=m), tol=ftol)
            mu, sd = cv2.meanStdDev(a)
            g.add(f"meanStdDev_{dt}_c{cn}", "meanStdDev", [a], {},
                  values=list(mu.ravel()) + list(sd.ravel()), tol=max(ftol, 1e-12))
            mu, sd = cv2.meanStdDev(a, mask=m)
            g.add(f"meanStdDevMask_{dt}_c{cn}", "meanStdDevMask", [a, m], {},
                  values=list(mu.ravel()) + list(sd.ravel()), tol=max(ftol, 1e-12))
            b = rand(rng, dt, cn, *lim)
            for nt in (cv2.NORM_INF, cv2.NORM_L1, cv2.NORM_L2, cv2.NORM_L2SQR):
                ntol = 0.0 if nt == cv2.NORM_INF else max(ftol, 1e-12)
                g.add(f"norm_{dt}_c{cn}_{nt}", "norm", [a], {"normType": nt}, values=[cv2.norm(a, nt)], tol=ntol)
                g.add(f"normMask_{dt}_c{cn}_{nt}", "normMask", [a, m], {"normType": nt},
                      values=[cv2.norm(a, nt, mask=m)], tol=ntol)
                g.add(f"normDiff_{dt}_c{cn}_{nt}", "normDiff", [a, b], {"normType": nt},
                      values=[cv2.norm(a, b, nt)], tol=ntol)
                g.add(f"normRel_{dt}_c{cn}_{nt}", "normDiff", [a, b], {"normType": nt | cv2.NORM_RELATIVE},
                      values=[cv2.norm(a, b, nt | cv2.NORM_RELATIVE)], tol=max(ntol, 1e-12))

        s = rand(rng, dt, 1, specials=True)
        flat = s.reshape(-1)
        flat[20] = flat.max() if dt not in ("f32", "f64") else 1e30
        flat[40] = flat[20]
        mv = cv2.minMaxLoc(s)
        g.add(f"minMaxLoc_{dt}", "minMaxLoc", [s], {}, values=[mv[0], mv[1], *mv[2], *mv[3]])
        m = mask_for(rng)
        mv = cv2.minMaxLoc(s, mask=m)
        g.add(f"minMaxLocMask_{dt}", "minMaxLocMask", [s, m], {}, values=[mv[0], mv[1], *mv[2], *mv[3]])
        mv = cv2.minMaxLoc(s, mask=np.zeros_like(m))
        g.add(f"minMaxLocEmptyMask_{dt}", "minMaxLocMask", [s, np.zeros_like(m)], {},
              values=[mv[0], mv[1], *mv[2], *mv[3]])
        z = rand(rng, dt, 1, specials=True)
        z.reshape(-1)[::3] = 0
        g.add(f"countNonZero_{dt}", "countNonZero", [z], {}, values=[cv2.countNonZero(z)])

    nan = np.full((3, 5), np.nan, np.float32)
    mv = cv2.minMaxLoc(nan)
    g.add("minMaxLoc_allNaN", "minMaxLoc", [nan], {}, values=[mv[0], mv[1], *mv[2], *mv[3]])

    combos = [("u8", "i32", "sum"), ("u8", "f32", "sum"), ("u8", "f64", "sum"), ("u16", "f32", "sum"),
              ("u16", "f64", "sum"), ("i16", "f32", "sum"), ("i16", "f64", "sum"), ("f32", "f32", "sum"),
              ("f32", "f64", "sum"), ("f64", "f64", "sum"), ("u8", "u8", "avg"), ("u8", "f32", "avg"),
              ("f32", "f32", "avg"), ("f64", "f64", "avg"), ("u16", "f64", "avg"),
              ("u8", "u8", "max"), ("u16", "u16", "max"), ("i16", "i16", "min"), ("f32", "f32", "min"),
              ("f64", "f64", "max")]
    rop = {"sum": cv2.REDUCE_SUM, "avg": cv2.REDUCE_AVG, "max": cv2.REDUCE_MAX, "min": cv2.REDUCE_MIN}
    for sdt, ddt, op in combos:
        for cn in (1, 3):
            lim = (0, 2000) if sdt == "u16" else (-2000, 2000) if sdt == "i16" else (None, None)
            a = rand(rng, sdt, cn, *lim)
            for dim in (0, 1):
                out = cv2.reduce(a, dim, rop[op], dtype=CVDEPTH[ddt])
                exact = sdt not in ("f32", "f64") or op in ("max", "min")
                g.add(f"reduce_{op}_{sdt}_{ddt}_c{cn}_d{dim}", "reduce", [a],
                      {"dim": dim, "rtype": rop[op], "to": ddt}, out, tol=0.0 if exact else 1e-5)


def main():
    rng = np.random.default_rng(20261011)
    g = Gen()
    arithmetic(g, rng)
    logic(g, rng)
    conversions(g, rng)
    reductions(g, rng)

    # Sanity checks on the reference itself.
    assert cv2.add(np.array([[250]], np.uint8), (10, 0, 0, 0))[0, 0] == 255
    assert cv2.divide(np.array([[5, 7]], np.uint8), np.array([[2, 2]], np.uint8))[0].tolist() == [2, 4]
    assert all(c["name"] in g.names for c in g.cases)
    assert set(SUSPECT_CASES) <= g.names, set(SUSPECT_CASES) - g.names

    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "core_golden.json")
    with open(out, "w") as f:
        json.dump({"arrays": g.arrays, "cases": g.cases}, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases to {out}")


if __name__ == "__main__":
    main()
