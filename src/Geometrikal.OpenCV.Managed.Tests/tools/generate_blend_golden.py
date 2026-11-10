#!/usr/bin/env python3
"""Golden-data generator for the managed blendLinear port.

Runs cv2 4.5.5 blendLinear on 8-bit and float images with 1-4 channels and
widths that exercise the SSE blocks and the scalar tail, including zero
weights and exact rounding halves. Writes ``TestData/blend_golden.json``;
``ImgProc/BlendingGoldenTests.cs`` replays each case bit for bit.

Regenerate with:
    python tools/generate_blend_golden.py
(run from the ``Geometrikal.OpenCV.Managed.Tests`` directory).

Environment: OPENCV_CPU_DISABLE removes AVX/AVX2/FMA dispatch; OpenCL is off.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import json
import os

os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX,AVX2,AVX512F,FMA3")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.ocl.setUseOpenCL(False)

SUSPECT_CASES = {}


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def main():
    rng = np.random.default_rng(20261023)
    cases = []
    for dtype, dname in ((np.uint8, "u8"), (np.float32, "f32")):
        for cn in (1, 2, 3, 4):
            for w, h in ((37, 5), (16, 3), (5, 4)):
                shape = (h, w) if cn == 1 else (h, w, cn)
                if dtype == np.uint8:
                    a = rng.integers(0, 256, shape).astype(np.uint8)
                    b = rng.integers(0, 256, shape).astype(np.uint8)
                else:
                    a = rng.normal(0, 50, shape).astype(np.float32)
                    b = rng.normal(0, 50, shape).astype(np.float32)
                w1 = rng.random((h, w)).astype(np.float32)
                w2 = (1 - w1).astype(np.float32)
                w1[0, 0] = 0
                w2[0, 0] = 0
                w1[0, 1] = 0.5
                w2[0, 1] = 0.5
                out = cv2.blendLinear(a, b, w1, w2)
                name = f"{dname}_c{cn}_{w}x{h}"
                cases.append({"name": name, "depth": dname, "width": w, "height": h, "channels": cn,
                              "a": b64(a), "b": b64(b), "w1": b64(w1), "w2": b64(w2), "result": b64(out),
                              "suspect": name in SUSPECT_CASES})

    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "blend_golden.json")
    with open(out, "w") as f:
        json.dump({"cases": cases}, f, separators=(",", ":"))
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
