#!/usr/bin/env python3
"""Golden-data generator for the managed math functions port.

Runs cv2 4.5.5 magnitude, phase, cartToPolar, polarToCart, exp, log, pow and
sqrt over a seeded battery and writes ``TestData/mathfuncs_golden.json``.
``Core/MathFunctionsGoldenTests.cs`` replays each case byte for byte.

Regenerate with:
    python tools/generate_mathfuncs_golden.py
(run from the ``Geometrikal.OpenCV.Managed.Tests`` directory).

Environment and battery constraints:
 * OPENCV_CPU_DISABLE removes AVX/AVX2/FMA dispatch, so vector blocks are two
   SSE registers (8 floats) and v_fma is a separate multiply and add. IPP,
   OpenCL and threads are disabled.
 * All arrays are continuous, so each matrix is one plane.
 * The port follows OpenCV 5.0. Cases where 4.5.5 differs are listed in
   SUSPECT_CASES and pinned in MathFunctionsOpenCv5DifferenceTests.

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

_SIZES = ("1x5", "2x1030", "3x13c3", "9x37")
_VEC = ("2x1030", "3x13c3", "9x37")
SUSPECT_CASES = {}
SUSPECT_CASES.update({f"cartToPolar_f64_{s}_{u}": "4.5.5 computes the 64F magnitude in double; OpenCV 5.0 cartToPolar64f converts to float"
                      for s in _SIZES for u in ("deg", "rad")})
SUSPECT_CASES.update({f"polarToCart_f64_{s}_{u}": "4.5.5 computes 64F polarToCart in double; OpenCV 5.0 polarToCart64f converts to float"
                      for s in _SIZES for u in ("deg", "rad")})
SUSPECT_CASES.update({f"polarToCartUnit_f64_{s}_{u}": "4.5.5 returns uninitialised values for 64F polarToCart without magnitudes"
                      for s in _SIZES for u in ("deg", "rad")})
SUSPECT_CASES.update({f"polarToCart_f32_{s}_{u}": "4.5.5 uses the sin table for every element and a float magnitude product; "
                      "OpenCV 5.0 uses v_sincos for vector blocks and a double product"
                      for s in _VEC for u in ("deg", "rad")})
SUSPECT_CASES["polarToCart_f32_1x5_deg"] = "4.5.5 multiplies the magnitude in float; OpenCV 5.0 in double before the cast"
SUSPECT_CASES.update({f"polarToCartUnit_f32_{s}_{u}": "4.5.5 uses the sin table for every element; OpenCV 5.0 uses v_sincos for vector blocks"
                      for s in _VEC for u in ("deg", "rad")})
_MATHLIB = "the cv2 4.5.5 wheel computes exp/log with an approximate vector routine; OpenCV 5.0 for MSVC calls the C runtime per element"
SUSPECT_CASES.update({n: _MATHLIB for n in (
    "exp_f32_1x5", "exp_f32_2x1030", "exp_f32_3x13c3", "exp_f32_9x37", "exp_f64_2x1030", "exp_f64_3x13c3", "exp_f64_9x37",
    "log_f32_2x1030", "log_f32_3x13c3", "log_f32_9x37", "log_f64_2x1030",
    "pow_f32_2x1030_p-2.3", "pow_f32_2x1030_p1.7", "pow_f32_3x13c3_p-2.3", "pow_f32_3x13c3_p1.7",
    "pow_f32_9x37_p-2.3", "pow_f32_9x37_p1.7", "pow_f64_2x1030_p-2.3")})

DTYPES = {"u8": np.uint8, "i8": np.int8, "u16": np.uint16, "i16": np.int16,
          "i32": np.int32, "f32": np.float32, "f64": np.float64}


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def mat(arr):
    arr = np.ascontiguousarray(arr)
    dt = next(k for k, v in DTYPES.items() if v == arr.dtype)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return {"dtype": dt, "width": int(arr.shape[1]), "height": int(arr.shape[0]),
            "channels": 1 if arr.ndim == 2 else int(arr.shape[2]), "data": b64(arr)}


class Gen:
    def __init__(self):
        self.cases = []

    def add(self, name, op, inputs, outputs, **params):
        assert all(c["name"] != name for c in self.cases), name
        self.cases.append({"name": name, "op": op, "params": params,
                           "inputs": [None if a is None else mat(a) for a in inputs],
                           "outputs": [mat(o) for o in outputs],
                           "suspect": name in SUSPECT_CASES})


def rand_f(rng, dt, shape, lo=-100.0, hi=100.0):
    return rng.uniform(lo, hi, shape).astype(DTYPES[dt])


def main():
    rng = np.random.default_rng(20261023)
    g = Gen()
    shapes = {"9x37": (9, 37), "1x5": (1, 5), "3x13c3": (3, 13, 3), "2x1030": (2, 1030)}

    for dt in ("f32", "f64"):
        for sn, shape in shapes.items():
            x = rand_f(rng, dt, shape)
            y = rand_f(rng, dt, shape)
            x.reshape(-1)[:4] = [0, -0.0, 3, -3]
            y.reshape(-1)[:4] = [0, 0, -4, 0]
            g.add(f"magnitude_{dt}_{sn}", "magnitude", [x, y], [cv2.magnitude(x, y)])
            for deg in (False, True):
                tag = "deg" if deg else "rad"
                g.add(f"phase_{dt}_{sn}_{tag}", "phase", [x, y], [cv2.phase(x, y, angleInDegrees=deg)], degrees=deg)
                m, a = cv2.cartToPolar(x, y, angleInDegrees=deg)
                g.add(f"cartToPolar_{dt}_{sn}_{tag}", "cartToPolar", [x, y], [m, a], degrees=deg)

                angle = rand_f(rng, dt, shape, -400 if deg else -7, 400 if deg else 7)
                angle.reshape(-1)[:3] = [0, 90 if deg else np.pi / 2, -1e4 if deg else -200]
                mag = rand_f(rng, dt, shape, 0, 50)
                px, py = cv2.polarToCart(mag, angle, angleInDegrees=deg)
                g.add(f"polarToCart_{dt}_{sn}_{tag}", "polarToCart", [mag, angle], [px, py], degrees=deg)
                ux, uy = cv2.polarToCart(None, angle, angleInDegrees=deg)
                g.add(f"polarToCartUnit_{dt}_{sn}_{tag}", "polarToCart", [None, angle], [ux, uy], degrees=deg)

            e = rand_f(rng, dt, shape, -20, 20)
            g.add(f"exp_{dt}_{sn}", "exp", [e], [cv2.exp(e)])
            pos = rand_f(rng, dt, shape, 1e-3, 1e3)
            pos.reshape(-1)[:2] = [1, 0]
            g.add(f"log_{dt}_{sn}", "log", [pos], [cv2.log(pos)])
            g.add(f"sqrt_{dt}_{sn}", "sqrt", [pos], [cv2.sqrt(pos)])
            withneg = rand_f(rng, dt, shape, -5, 5)
            withneg.reshape(-1)[:3] = [0, -2, 2]
            for p in (0, 1, 2, 3, -3, 0.5, -0.5, 1.7, -2.3):
                src = pos if p in (0.5, -0.5) else withneg
                g.add(f"pow_{dt}_{sn}_p{p}", "pow", [src], [cv2.pow(src, p)], power=p)

    for dt, lo, hi in (("u8", 0, 256), ("i8", -128, 128), ("u16", 0, 65536), ("i16", -32768, 32768), ("i32", -50, 50)):
        for sn, shape in (("9x37", (9, 37)), ("1x5", (1, 5)), ("3x13c3", (3, 13, 3))):
            a = rng.integers(lo, hi, shape).astype(DTYPES[dt])
            a.reshape(-1)[:3] = [0, 1, 2]
            small = rng.integers(max(lo, -3), min(hi, 4), shape).astype(DTYPES[dt])
            # 16U products of 2^31 and more are outside the port (core multiply limitation).
            below = np.minimum(a, 46340) if dt == "u16" else a
            for p in (0, 1, 2, 3, 5, -1, -2, -3):
                src = small if p < 0 else below if p == 2 else a
                g.add(f"pow_{dt}_{sn}_p{p}", "pow", [src], [cv2.pow(src, p)], power=p)

    # Sanity checks on the reference itself.
    assert abs(cv2.magnitude(np.float32([[3]]), np.float32([[4]]))[0, 0] - 5) < 1e-6
    assert cv2.pow(np.uint8([[2, 3]]), 3)[0].tolist() == [8, 27]
    assert set(SUSPECT_CASES) <= {c["name"] for c in g.cases}

    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "mathfuncs_golden.json")
    with open(out, "w") as f:
        json.dump({"cases": g.cases}, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases to {out}")


if __name__ == "__main__":
    main()
