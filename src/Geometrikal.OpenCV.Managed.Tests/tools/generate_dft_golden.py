#!/usr/bin/env python3
"""Golden-data generator for the managed DFT/DCT port.

Runs cv2 4.5.5 ``dft``, ``dct`` and ``mulSpectrums`` over a seeded battery and
writes ``TestData/dft_golden.json``. ``Core/DftGoldenTests.cs`` replays each
case and compares the output bytes exactly.

Regenerate with:
    python tools/generate_dft_golden.py
(run from the ``Geometrikal.OpenCV.Managed.Tests`` directory).

Environment and battery constraints:
 * IPP is disabled. With IPP, 32F DFTs with n*count >= 64, 2D DFTs with
   W*H > 64, 64F 1D DFTs and the 32F DCT use other algorithms. The managed
   port follows the dxt.cpp code without IPP.
 * OPENCV_CPU_DISABLE has no effect on dxt.cpp (baseline SSE3 code only); it
   is set for consistency with the other generators. One thread.
 * DFT_ROWS with DFT_COMPLEX_OUTPUT and nonzeroRows is not generated: cv2
   leaves the upper half of the rows after nonzeroRows uninitialized.
 * Non-continuous single columns are generated as ROI cases (``roi`` holds
   x, y, width, height in the parent array). OpenCV transforms them with the
   column algorithm, which gives other bits than the continuous case.
 * Input and output arrays are stored once in a shared ``arrays`` list.

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

SUSPECT_CASES = {}

DTYPES = {"f32": np.float32, "f64": np.float64}
INV, SCALE, ROWS, COUT, ROUT, CIN = 1, 2, 4, 16, 32, 64

SIZES_1D = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 25, 27, 30, 49, 64, 77,
            121, 128, 169, 210, 243, 256, 257, 289, 360, 512, 1000, 1024, 1155, 2048]
SHAPES_2D = [(2, 2), (3, 3), (4, 6), (5, 7), (8, 8), (6, 9), (7, 10), (12, 15), (16, 16),
             (9, 14), (30, 25), (64, 48), (21, 22), (2, 5), (5, 2), (3, 64), (17, 4), (49, 36)]
DCT_SIZES = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 26, 30, 32, 36, 44, 50, 64, 66, 96, 100,
             128, 154, 210, 256, 500, 512, 1024]


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
            arr = arr[None]
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
                "params": params, "suspect": name in SUSPECT_CASES}
        if case["suspect"]:
            case["suspectReason"] = SUSPECT_CASES[name]
        if expected is not None:
            case["expected"] = self.ref(expected)
        if values is not None:
            case["values"] = values
        self.cases.append(case)

    def dft(self, name, src, flags=0, nz=0, roi=None):
        view = src if roi is None else src[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        out = cv2.dft(view, flags=flags, nonzeroRows=nz)
        out = out.reshape(view.shape[0], view.shape[1], -1)
        params = {"flags": flags, "nonzeroRows": nz}
        if roi is not None:
            params["roi"] = list(roi)
        self.add(name, "dft", [src], params, out)

    def dct(self, name, src, flags=0, roi=None):
        view = src if roi is None else src[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        out = cv2.dct(view, flags=flags).reshape(view.shape[0], view.shape[1])
        params = {"flags": flags}
        if roi is not None:
            params["roi"] = list(roi)
        self.add(name, "dct", [src], params, out)

    def mul(self, name, a, b, flags, conj):
        out = cv2.mulSpectrums(a, b, flags, conjB=conj).reshape(a.shape)
        self.add(name, "mulSpectrums", [a, b], {"flags": flags, "conjB": conj}, out)


def rnd(rng, dt, shape, scale=10.0):
    return (rng.standard_normal(shape) * scale).astype(DTYPES[dt])


def battery_1d(g, rng):
    for dt in DTYPES:
        for n in SIZES_1D:
            x = rnd(rng, dt, (1, n))
            z = rnd(rng, dt, (1, n, 2))
            g.dft(f"{dt}_row_real_{n}", x)
            g.dft(f"{dt}_row_real_scale_{n}", x, SCALE)
            g.dft(f"{dt}_row_real_cout_{n}", x, COUT)
            g.dft(f"{dt}_row_cplx_{n}", z)
            g.dft(f"{dt}_row_cplx_inv_scale_{n}", z, INV | SCALE)
            g.dft(f"{dt}_row_cplx_inv_{n}", z, INV)
            ccs = cv2.dft(x)
            g.dft(f"{dt}_row_ccs_inv_{n}", ccs, INV)
            g.dft(f"{dt}_row_ccs_inv_scale_{n}", ccs, INV | SCALE)
            g.dft(f"{dt}_row_arb_inv_{n}", rnd(rng, dt, (1, n)), INV)
            g.dft(f"{dt}_row_c2r_{n}", cv2.dft(x, flags=COUT), INV | SCALE | ROUT)
            if n <= 300:
                g.dft(f"{dt}_col_real_{n}", rnd(rng, dt, (n, 1)), SCALE if n % 2 else 0)
                g.dft(f"{dt}_col_cplx_inv_{n}", rnd(rng, dt, (n, 1, 2)), INV)


def battery_rows(g, rng):
    for dt in DTYPES:
        for (h, w) in [(3, 8), (5, 7), (9, 16), (4, 1), (6, 30), (11, 13), (16, 64), (7, 100)]:
            x = rnd(rng, dt, (h, w))
            z = rnd(rng, dt, (h, w, 2))
            g.dft(f"{dt}_rows_real_{h}x{w}", x, ROWS)
            g.dft(f"{dt}_rows_real_scale_cout_{h}x{w}", x, ROWS | SCALE | COUT)
            g.dft(f"{dt}_rows_cplx_inv_scale_{h}x{w}", z, ROWS | INV | SCALE)
            g.dft(f"{dt}_rows_ccs_inv_{h}x{w}", cv2.dft(x, flags=ROWS), ROWS | INV | SCALE)
            g.dft(f"{dt}_rows_c2r_{h}x{w}", cv2.dft(x, flags=ROWS | COUT), ROWS | INV | ROUT)
            if w > 1:
                g.dft(f"{dt}_rows_real_nz_{h}x{w}", x, ROWS, nz=max(1, h // 2))
                g.dft(f"{dt}_rows_cplx_nz_{h}x{w}", z, ROWS, nz=max(1, h // 3))
                g.dft(f"{dt}_rows_ccs_inv_nz_{h}x{w}", cv2.dft(x, flags=ROWS), ROWS | INV, nz=max(1, h // 2))


def battery_2d(g, rng):
    for dt in DTYPES:
        for (h, w) in SHAPES_2D:
            x = rnd(rng, dt, (h, w))
            z = rnd(rng, dt, (h, w, 2))
            tag = f"{h}x{w}"
            g.dft(f"{dt}_2d_real_{tag}", x)
            g.dft(f"{dt}_2d_real_scale_{tag}", x, SCALE)
            g.dft(f"{dt}_2d_real_cout_{tag}", x, COUT)
            g.dft(f"{dt}_2d_cplx_{tag}", z)
            g.dft(f"{dt}_2d_cplx_inv_scale_{tag}", z, INV | SCALE)
            ccs = cv2.dft(x)
            g.dft(f"{dt}_2d_ccs_inv_{tag}", ccs, INV)
            g.dft(f"{dt}_2d_ccs_inv_scale_{tag}", ccs, INV | SCALE)
            g.dft(f"{dt}_2d_arb_inv_scale_{tag}", rnd(rng, dt, (h, w)), INV | SCALE)
            g.dft(f"{dt}_2d_c2r_{tag}", cv2.dft(x, flags=COUT), INV | SCALE | ROUT)
            g.dft(f"{dt}_2d_cin_c2r_{tag}", z, INV | ROUT | CIN)
            nz = max(1, h // 2)
            xz = x.copy()
            xz[nz:] = 0
            g.dft(f"{dt}_2d_real_nz_{tag}", x, 0, nz=nz)
            g.dft(f"{dt}_2d_real_cout_nz_{tag}", x, COUT, nz=nz)
            g.dft(f"{dt}_2d_cplx_nz_{tag}", z, 0, nz=nz)
            g.dft(f"{dt}_2d_ccs_inv_nz_{tag}", ccs, INV | SCALE, nz=nz)
            g.dft(f"{dt}_2d_cplx_inv_nz_{tag}", z, INV, nz=nz)
            g.dft(f"{dt}_2d_c2r_nz_{tag}", z, INV | ROUT, nz=nz)
        big = rnd(rng, dt, (120, 90))
        g.dft(f"{dt}_2d_real_120x90", big)
        g.dft(f"{dt}_2d_ccs_inv_scale_120x90", cv2.dft(big), INV | SCALE)
        g.dft(f"{dt}_2d_zero_6x8", np.zeros((6, 8), DTYPES[dt]))
        g.dft(f"{dt}_2d_zero_cout_5x7", np.zeros((5, 7), DTYPES[dt]), COUT)


def battery_roi(g, rng):
    for dt in DTYPES:
        parent = rnd(rng, dt, (40, 9))
        g.dft(f"{dt}_roi_col_real", parent, 0, roi=(3, 2, 1, 30))
        g.dft(f"{dt}_roi_col_real_cout", parent, COUT, roi=(4, 1, 1, 17))
        g.dft(f"{dt}_roi_col_ccs_inv", parent, INV | SCALE, roi=(2, 5, 1, 24))
        g.dft(f"{dt}_roi_2d_real", parent, 0, roi=(1, 3, 6, 20))
        g.dft(f"{dt}_roi_2d_ccs_inv", parent, INV, roi=(2, 2, 7, 15))
        pc = rnd(rng, dt, (33, 6, 2))
        g.dft(f"{dt}_roi_col_cplx", pc, 0, roi=(1, 0, 1, 33))
        g.dft(f"{dt}_roi_col_c2r", pc, INV | ROUT, roi=(2, 3, 1, 28))
        g.dct(f"{dt}_roi_dct_col", parent, 0, roi=(5, 4, 1, 32))
        g.dct(f"{dt}_roi_dct_2d", parent, 0, roi=(1, 2, 8, 20))


def battery_dct(g, rng):
    for dt in DTYPES:
        for n in DCT_SIZES:
            x = rnd(rng, dt, (1, n))
            g.dct(f"{dt}_dct_row_{n}", x)
            g.dct(f"{dt}_idct_row_{n}", x, INV)
            if n <= 256:
                g.dct(f"{dt}_dct_col_{n}", rnd(rng, dt, (n, 1)))
        for (h, w) in [(2, 2), (4, 6), (8, 8), (6, 10), (16, 12), (30, 18), (64, 64), (10, 1), (1, 1), (3, 8), (100, 26)]:
            x = rnd(rng, dt, (h, w))
            tag = f"{h}x{w}"
            if h % 2 == 0 or h == 1:
                g.dct(f"{dt}_dct2d_{tag}", x)
                g.dct(f"{dt}_idct2d_{tag}", x, INV)
            g.dct(f"{dt}_dct_rows_{tag}", x, ROWS)
            g.dct(f"{dt}_idct_rows_{tag}", x, ROWS | INV)


def battery_mul(g, rng):
    for dt in DTYPES:
        for (h, w) in [(1, 8), (1, 9), (2, 2), (3, 5), (4, 6), (5, 7), (8, 8), (6, 9), (7, 10), (12, 15), (16, 16), (30, 25), (9, 1), (8, 1), (40, 70)]:
            tag = f"{h}x{w}"
            a = cv2.dft(rnd(rng, dt, (h, w), 1.0))
            b = cv2.dft(rnd(rng, dt, (h, w), 1.0))
            za = rnd(rng, dt, (h, w, 2), 1.0)
            zb = rnd(rng, dt, (h, w, 2), 1.0)
            for conj in (False, True):
                c = "conj" if conj else "plain"
                g.mul(f"{dt}_mul_ccs_{c}_{tag}", a, b, 0, conj)
                g.mul(f"{dt}_mul_ccs_rows_{c}_{tag}", a, b, ROWS, conj)
                g.mul(f"{dt}_mul_cplx_{c}_{tag}", za, zb, 0, conj)
                g.mul(f"{dt}_mul_cplx_rows_{c}_{tag}", za, zb, ROWS, conj)


def optimal_sizes(g):
    probes = [0, -1, -5, 1, 2, 7, 11, 97, 100, 101, 127, 129, 250, 1000, 1023, 1025, 4097, 10007,
              123456, 4782969, 14348907, 2125763999, 2125764000, 2125764001, 2147483647]
    g.add("optimal_dft_size", "optimalSize", [], {"sizes": probes}, values=[cv2.getOptimalDFTSize(p) for p in probes])


def sanity():
    x = np.arange(8, dtype=np.float32)[None]
    y = cv2.dft(x)
    assert y[0, 0] == 28 and y.shape == (1, 8), y
    z = cv2.dft(x, flags=COUT)
    assert z.shape == (1, 8, 2), z.shape
    r = cv2.idft(y, flags=SCALE)
    assert np.allclose(r, x), r
    assert cv2.getOptimalDFTSize(97) == 100
    for bad in (lambda: cv2.dft(np.zeros((4, 1), np.float32), nonzeroRows=2),
                lambda: cv2.dct(np.zeros((1, 7), np.float32)),
                lambda: cv2.dft(np.zeros((4, 4), np.uint8))):
        try:
            bad()
        except cv2.error:
            continue
        raise AssertionError("expected cv2.error")


def main():
    rng = np.random.default_rng(20261011)
    sanity()
    g = Gen()
    battery_1d(g, rng)
    battery_rows(g, rng)
    battery_2d(g, rng)
    battery_roi(g, rng)
    battery_dct(g, rng)
    battery_mul(g, rng)
    optimal_sizes(g)
    assert set(SUSPECT_CASES) <= g.names, set(SUSPECT_CASES) - g.names

    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "dft_golden.json")
    with open(out, "w") as f:
        json.dump({"arrays": g.arrays, "cases": g.cases}, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases, {len(g.arrays)} arrays to {out}")


if __name__ == "__main__":
    main()
