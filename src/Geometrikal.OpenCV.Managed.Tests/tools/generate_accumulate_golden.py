#!/usr/bin/env python3
"""Golden-data generator for the managed accumulate port.

Runs cv2 4.5.5 accumulate, accumulateSquare, accumulateProduct and
accumulateWeighted for every source/accumulator depth pair, 1-4 channels,
with and without a mask, and writes ``TestData/accumulate_golden.json``.
``ImgProc/AccumulationGoldenTests.cs`` replays each case; accumulators must
match bit for bit.

Regenerate with:
    python tools/generate_accumulate_golden.py
(run from the ``Geometrikal.OpenCV.Managed.Tests`` directory).

Environment: OPENCV_CPU_DISABLE removes AVX/AVX2/FMA dispatch; IPP, OpenCL
and threads are disabled (IPP has its own 8U/16U/32F to 32F kernels).

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

DEPTHS = {np.uint8: "u8", np.uint16: "u16", np.float32: "f32", np.float64: "f64"}
PAIRS = [(np.uint8, np.float32), (np.uint8, np.float64), (np.uint16, np.float32), (np.uint16, np.float64),
         (np.float32, np.float32), (np.float32, np.float64), (np.float64, np.float64)]


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def mat(a):
    ch = 1 if a.ndim == 2 else a.shape[2]
    return {"width": int(a.shape[1]), "height": int(a.shape[0]), "channels": ch,
            "depth": DEPTHS[a.dtype.type], "data": b64(a)}


def make(rng, shape, dtype):
    if dtype == np.uint8:
        return rng.integers(0, 256, shape).astype(np.uint8)
    if dtype == np.uint16:
        return rng.integers(0, 65536, shape).astype(np.uint16)
    return (rng.normal(0, 100, shape)).astype(dtype)


def weighted_alphas():
    # Alphas where (float)(1 - alpha) differs from 1 - (float)alpha, plus common values.
    found = []
    a = 0.0013
    while len(found) < 2:
        if np.float32(1.0 - a) != np.float32(1) - np.float32(a):
            found.append(round(a, 6))
        a += 0.0137
    return [0.1, 0.5] + found


def main():
    rng = np.random.default_rng(20261023)
    cases = []
    alphas = weighted_alphas()
    for sdt, ddt in PAIRS:
        for cn in (1, 2, 3, 4):
            for w, h in ((29, 5), (16, 2)):
                shape = (h, w) if cn == 1 else (h, w, cn)
                for masked in (False, True):
                    mask = (rng.random((h, w)) < 0.6).astype(np.uint8) * rng.integers(1, 256, (h, w)).astype(np.uint8) if masked else None
                    src1 = make(rng, shape, sdt)
                    src2 = make(rng, shape, sdt)
                    acc0 = rng.normal(0, 1000, shape).astype(ddt)
                    group = {"src1": mat(src1), "src2": mat(src2), "acc": mat(acc0), "results": []}
                    if masked:
                        group["mask"] = mat(mask)
                    ops = [("acc", None), ("sqr", None), ("prod", None)] + [("w", a) for a in alphas]
                    for op, alpha in ops:
                        dst = acc0.copy()
                        if op == "acc":
                            cv2.accumulate(src1, dst, mask)
                        elif op == "sqr":
                            cv2.accumulateSquare(src1, dst, mask)
                        elif op == "prod":
                            cv2.accumulateProduct(src1, src2, dst, mask)
                        else:
                            cv2.accumulateWeighted(src1, dst, alpha, mask)
                        name = f"{op}{'' if alpha is None else alpha}_{DEPTHS[sdt]}_{DEPTHS[ddt]}_c{cn}_{w}x{h}{'_m' if masked else ''}"
                        group["results"].append({"name": name, "op": op, "alpha": alpha or 0, "result": b64(dst),
                                                 "suspect": name in SUSPECT_CASES})
                    cases.append(group)

    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "accumulate_golden.json")
    with open(out, "w") as f:
        json.dump({"groups": cases}, f, separators=(",", ":"))
    print(f"wrote {sum(len(g['results']) for g in cases)} cases to {out}; alphas {alphas}")


if __name__ == "__main__":
    main()
