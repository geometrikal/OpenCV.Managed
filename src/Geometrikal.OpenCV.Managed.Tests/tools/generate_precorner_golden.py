#!/usr/bin/env python3
"""Golden-data generator for the managed preCornerDetect port.

Runs cv2 4.5.5 preCornerDetect on 8-bit and float images with every Sobel
aperture and three border types, and writes ``TestData/precorner_golden.json``.
``ImgProc/PreCornerDetectGoldenTests.cs`` replays each case; outputs must match
bit for bit.

Regenerate with:
    python tools/generate_precorner_golden.py
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

SUSPECT_CASES = {}


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def main():
    rng = np.random.default_rng(20261023)
    cases = []
    for w, h in ((37, 23), (16, 9)):
        u8 = cv2.GaussianBlur(rng.integers(0, 256, (h, w)).astype(np.uint8), (3, 3), 0.8)
        cv2.rectangle(u8, (w // 4, h // 4), (w // 2, h // 2), 230, -1)
        # Integer-valued floats: the port's small symmetric float Sobel follows the scalar
        # association (FilteringPortNotes.md, item 2), which equals the SIMD one on exact sums.
        f32 = (u8.astype(np.float32) * 3 - rng.integers(0, 40, (h, w))).astype(np.float32)
        for src, dname in ((u8, "u8"), (f32, "f32")):
            for ksize in (1, 3, 5, 7):
                for border, bname in ((cv2.BORDER_REFLECT_101, "r101"), (cv2.BORDER_REPLICATE, "rep"), (cv2.BORDER_CONSTANT, "const")):
                    out = cv2.preCornerDetect(src, ksize, borderType=border)
                    name = f"{dname}_{w}x{h}_k{ksize}_{bname}"
                    cases.append({"name": name, "depth": dname, "width": w, "height": h, "ksize": ksize, "border": int(border),
                                  "src": b64(src), "result": b64(out.astype(np.float32)), "suspect": name in SUSPECT_CASES})

    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "precorner_golden.json")
    with open(out, "w") as f:
        json.dump({"cases": cases}, f, separators=(",", ":"))
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
