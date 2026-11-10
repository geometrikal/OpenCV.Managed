#!/usr/bin/env python3
"""Golden-data generator for the managed inpaint port.

Runs cv2 4.5.5 inpaint (Telea and Navier-Stokes) on 8-bit gray and colour,
16-bit and float images with several masks and radii, and writes
``TestData/inpaint_golden.json``. ``Photo/InpaintGoldenTests.cs`` replays each
case; outputs must match exactly.

Regenerate with:
    python tools/generate_inpaint_golden.py
(run from the ``Geometrikal.OpenCV.Managed.Tests`` directory).

Environment: OPENCV_CPU_DISABLE removes AVX/AVX2/FMA dispatch; IPP, OpenCL
and threads are disabled.

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

BORDER_READ = ("OpenCV commit 2688248df7 (2024, C-API cleanup) reads neighbour values at (k-1, l-1); 4.5.5 read "
               "(km, lm), which is row or column 1 for neighbours in image row or column 0")
FLOAT_TELEA = ("OpenCV commit dfb6069ad9 (2024): Telea no longer adds 0.5 to float outputs; 4.5.5 output is "
               "the 5.0 value plus 0.5")


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def scene(rng, w, h):
    a = cv2.resize(rng.random((h // 6 + 2, w // 6 + 2)), (w, h), interpolation=cv2.INTER_CUBIC)
    a = np.clip(a * 200 + 30, 0, 255)
    cv2.line(a, (0, h // 3), (w - 1, h // 2), 250, 2)
    return a


def masks(rng, w, h):
    m1 = np.zeros((h, w), np.uint8)
    cv2.circle(m1, (w // 2, h // 2), min(w, h) // 5, 255, -1)
    m2 = np.zeros((h, w), np.uint8)
    cv2.line(m2, (2, 3), (w - 3, h - 4), 1, 2)
    m2[0:3, w - 6:w] = 7
    m3 = (rng.random((h, w)) < 0.08).astype(np.uint8)
    return {"disc": m1, "edge": m2, "dots": m3}


def main():
    rng = np.random.default_rng(20261023)
    images = []
    w, h = 41, 33
    base = scene(rng, w, h)
    variants = {
        "u8": base.astype(np.uint8),
        "u8c3": np.dstack([base, np.roll(base, 5, 1) * 0.8, 255 - base]).astype(np.uint8),
        "u16": (base * 251).astype(np.uint16),
        "f32": (base / 255).astype(np.float32),
    }
    for vname, src in variants.items():
        results = []
        for mname, mask in masks(rng, w, h).items():
            for radius in (1, 3, 5.6):
                for flag, fname in ((cv2.INPAINT_TELEA, "telea"), (cv2.INPAINT_NS, "ns")):
                    out = cv2.inpaint(src, mask, radius, flag)
                    name = f"{fname}_{vname}_{mname}_r{radius}"
                    rng_px = min(max(int(np.rint(radius)), 1), 100)
                    near_edge = mask[: rng_px + 1, :].any() or mask[:, : rng_px + 1].any()
                    reason = FLOAT_TELEA if (vname == "f32" and flag == cv2.INPAINT_TELEA) else (BORDER_READ if near_edge else None)
                    results.append({"name": name, "mask": b64(mask), "radius": radius, "method": fname,
                                    "result": b64(out), "suspect": reason is not None, "reason": reason})
        images.append({"name": vname, "width": w, "height": h, "channels": 1 if src.ndim == 2 else 3,
                       "data": b64(src), "results": results})

    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "inpaint_golden.json")
    with open(out, "w") as f:
        json.dump({"images": images}, f, separators=(",", ":"))
    print(f"wrote {sum(len(i['results']) for i in images)} cases to {out}")


if __name__ == "__main__":
    main()
