#!/usr/bin/env python3
"""Golden-data generator for the managed warpPolar port.

Runs cv2 4.5.5 warpPolar (linear and semi-log, forward and inverse) over a
seeded battery and writes ``TestData/polar_golden.json``.
``ImgProc/WarpPolarGoldenTests.cs`` replays each case; outputs must match
exactly.

Regenerate with:
    python tools/generate_polar_golden.py
(run from the ``Geometrikal.OpenCV.Managed.Tests`` directory).

Environment: OPENCV_CPU_DISABLE removes AVX/AVX2/FMA dispatch; IPP, OpenCL
and threads are disabled. Cases without WARP_FILL_OUTLIERS pass a zero
destination, because cv2 leaves the outlier pixels of a new destination
uninitialised; they use INTER_NEAREST only (OpenCV 5.0 renormalises the
partial bilinear neighbourhoods of a transparent border).

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

DTYPES = {np.uint8: "u8", np.float32: "f32"}


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def mat(a):
    ch = 1 if a.ndim == 2 else a.shape[2]
    return {"width": int(a.shape[1]), "height": int(a.shape[0]), "channels": ch,
            "depth": DTYPES[a.dtype.type], "data": b64(a)}


def make_image(rng, w, h, ch, dtype):
    a = rng.integers(0, 256, (h, w, ch)).astype(np.uint8)
    a = cv2.GaussianBlur(a, (7, 7), 2)
    if ch == 1:
        a = a.reshape(h, w)
    if dtype == np.float32:
        a = a.astype(np.float32) / 255 + rng.normal(0, 0.01, a.shape).astype(np.float32)
    return a


def main():
    rng = np.random.default_rng(20261023)
    cases = []

    def add(name, src, dsize, center, max_radius, flags, result, zero_dst):
        cases.append({"name": name, "src": mat(src), "dsize": list(dsize), "center": [float(center[0]), float(center[1])],
                      "maxRadius": max_radius, "flags": flags, "zeroDst": zero_dst, "result": mat(result),
                      "suspect": name in SUSPECT_CASES})

    images = {
        "g40x30": make_image(rng, 40, 30, 1, np.uint8),
        "c33": make_image(rng, 33, 33, 3, np.uint8),
        "f50x41": make_image(rng, 50, 41, 1, np.float32),
        "g1100x20": make_image(rng, 1100, 20, 1, np.uint8),
    }
    interps = {"nn": cv2.INTER_NEAREST, "lin": cv2.INTER_LINEAR, "cub": cv2.INTER_CUBIC}
    for key, src in images.items():
        h, w = src.shape[:2]
        centers = [((w - 1) * 0.5, (h - 1) * 0.5), (w * 0.3 + 0.25, h * 0.6 - 0.125)]
        for ci, center in enumerate(centers):
            for mode, mname in ((cv2.WARP_POLAR_LINEAR, "lin"), (cv2.WARP_POLAR_LOG, "log")):
                for iname, interp in interps.items():
                    for radius, dsize in ((min(w, h) * 0.5, (w, h)), (max(w, h) * 0.7, (0, 0)), (17.3, (25, 0))):
                        if key == "g1100x20" and dsize == (0, 0):
                            continue
                        flags = interp | mode | cv2.WARP_FILL_OUTLIERS
                        fwd = cv2.warpPolar(src, dsize, center, radius, flags)
                        tag = f"{key}_c{ci}_{mname}_{iname}_r{radius:g}_d{dsize[0]}x{dsize[1]}"
                        add(f"fwd_{tag}", src, dsize, center, radius, flags, fwd, False)
                        inv = cv2.warpPolar(fwd, (w, h), center, radius, flags | cv2.WARP_INVERSE_MAP)
                        add(f"inv_{tag}", fwd, (w, h), center, radius, flags | cv2.WARP_INVERSE_MAP, inv, False)
            for mode, mname in ((cv2.WARP_POLAR_LINEAR, "lin"), (cv2.WARP_POLAR_LOG, "log")):
                radius = max(w, h) * 0.8
                flags = cv2.INTER_NEAREST | mode
                dst = np.zeros((h, w) + src.shape[2:], src.dtype)
                fwd = cv2.warpPolar(src, (w, h), center, radius, flags, dst=dst)
                add(f"fwdT_{key}_c{ci}_{mname}", src, (w, h), center, radius, flags, fwd, True)

    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "polar_golden.json")
    with open(out, "w") as f:
        json.dump({"cases": cases}, f, separators=(",", ":"))
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
