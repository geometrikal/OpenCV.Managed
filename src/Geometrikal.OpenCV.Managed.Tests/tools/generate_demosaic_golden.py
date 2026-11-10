#!/usr/bin/env python3
"""Golden-data generator for the managed Bayer demosaicing port.

Runs cv2 4.5.5 cvtColor with the Bayer codes (gray, BGR, BGRA, VNG, EA) for
the four patterns, 8-bit and 16-bit, over a seeded battery of sizes, and
writes ``TestData/demosaic_golden.json``. ``ImgProc/DemosaicingGoldenTests.cs``
replays each case; outputs must match exactly.

Regenerate with:
    python tools/generate_demosaic_golden.py
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

GRAY_SIMD = ("OpenCV PR #25968 (2024, issue #25823) replaced the 4.5.5 u8 SIMD Bayer2Gray kernel, whose "
             "descale was wrong; OpenCV 5.0 equals the scalar formula (the 16U result)")
VNG_SHIFT = ("OpenCV PR #27226 (2025, issues #5089, #27225): 4.5.5 VNG output is two rows above its "
             "source neighbourhood and copies border pixels; 5.0 pads the source by reflection")
VNG_SMALL = ("OpenCV PR #27226: for images smaller than 8 pixels VNG falls back to bilinear; 4.5.5 used the "
             "RG pattern for every VNG code")

PATTERNS = ["BG", "GB", "RG", "GR"]
METHODS = {"gray": "2GRAY", "bgr": "2BGR", "bgra": "2BGRA", "vng": "2BGR_VNG", "ea": "2BGR_EA"}


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def scene(rng, w, h, dtype):
    # Smooth colour content with edges, sampled as a Bayer mosaic.
    hi = 255 if dtype == np.uint8 else 65535
    base = rng.random((h // 4 + 2, w // 4 + 2)) * hi
    a = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    a[: h // 2, w // 3:] = a[: h // 2, w // 3:] * 0.3
    a += rng.normal(0, hi * 0.03, (h, w))
    return np.clip(a, 0, hi).astype(dtype)


def main():
    rng = np.random.default_rng(20261023)
    images = []
    sizes = [(37, 29), (64, 48), (40, 9), (7, 7), (2, 5), (5, 2), (3, 3), (1, 1)]
    for dtype, dname in ((np.uint8, "u8"), (np.uint16, "u16")):
        for w, h in sizes:
            src = scene(rng, w, h, dtype)
            results = []
            for p in PATTERNS:
                for m, suffix in METHODS.items():
                    if m == "vng" and dtype != np.uint8:
                        continue
                    code = getattr(cv2, f"COLOR_Bayer{p}{suffix}")
                    try:
                        out = cv2.cvtColor(src, code)
                    except cv2.error as e:
                        print(f"skip {dname} {w}x{h} {p} {m}: {str(e).splitlines()[-1]}")
                        continue
                    name = f"{m}_{p}_{dname}_{w}x{h}"
                    reason = None
                    if m == "gray" and dtype == np.uint8:
                        if not np.array_equal(out, cv2.cvtColor(src.astype(np.uint16), code)):
                            reason = GRAY_SIMD
                    elif m == "vng" and min(w, h) >= 8:
                        reason = VNG_SHIFT
                    elif m == "vng":
                        if not np.array_equal(out, cv2.cvtColor(src, getattr(cv2, f"COLOR_Bayer{p}2BGR"))):
                            reason = VNG_SMALL
                    results.append({"name": name, "pattern": p, "method": m, "result": b64(out),
                                    "channels": 1 if out.ndim == 2 else int(out.shape[2]), "suspect": reason is not None,
                                    "reason": reason})
            images.append({"width": w, "height": h, "depth": dname, "data": b64(src), "results": results})

    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "demosaic_golden.json")
    with open(out, "w") as f:
        json.dump({"images": images}, f, separators=(",", ":"))
    print(f"wrote {sum(len(i['results']) for i in images)} cases to {out}")


if __name__ == "__main__":
    main()
