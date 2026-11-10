#!/usr/bin/env python3
"""Golden-data generator for the managed fastNlMeansDenoising port.

Runs cv2 4.5.5 fastNlMeansDenoising (L2 and L1, 8-bit 1-4 channels, 16-bit,
one h or per-channel h, several window sizes) and fastNlMeansDenoisingColored,
and writes ``TestData/nlmeans_golden.json``. ``Photo/DenoisingGoldenTests.cs``
replays each case; outputs must match exactly.

Regenerate with:
    python tools/generate_nlmeans_golden.py
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
cv2.ipp.setUseIPP(False)

SUSPECT_CASES = {}


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def noisy(rng, w, h, cn, hi, sigma):
    base = cv2.resize(rng.random((h // 5 + 2, w // 5 + 2, cn)), (w, h), interpolation=cv2.INTER_CUBIC).reshape(h, w, cn)
    a = base * hi * 0.8 + hi * 0.1 + rng.normal(0, sigma * hi, (h, w, cn))
    a = np.clip(a, 0, hi)
    out = a.astype(np.uint8 if hi == 255 else np.uint16)
    return out.reshape(h, w) if cn == 1 else out


def main():
    rng = np.random.default_rng(20261023)
    cases = []

    def add(name, src, out, **params):
        cases.append({"name": name, "width": int(src.shape[1]), "height": int(src.shape[0]),
                      "channels": 1 if src.ndim == 2 else int(src.shape[2]), "depth": "u8" if src.dtype == np.uint8 else "u16",
                      "src": b64(src), "result": b64(out), "suspect": name in SUSPECT_CASES, **params})

    for cn in (1, 2, 3, 4):
        src = noisy(rng, 37, 29, cn, 255, 0.08)
        for t, s in ((7, 21), (3, 7), (5, 11), (4, 10)):
            for hval in (3.0, 12.5):
                out = cv2.fastNlMeansDenoising(src, None, hval, t, s)
                add(f"l2_u8_c{cn}_t{t}_s{s}_h{hval}", src, out, op="nlm", h=[hval], t=t, s=s, norm="L2")
            out = cv2.fastNlMeansDenoising(src, h=[9.0], templateWindowSize=t, searchWindowSize=s, normType=cv2.NORM_L1)
            add(f"l1_u8_c{cn}_t{t}_s{s}", src, out, op="nlm", h=[9.0], t=t, s=s, norm="L1")
        if cn > 1:
            hs = [4.0 + 5 * i for i in range(cn)]
            out = cv2.fastNlMeansDenoising(src, h=hs, templateWindowSize=5, searchWindowSize=11)
            add(f"l2_u8_c{cn}_perchannel", src, out, op="nlm", h=hs, t=5, s=11, norm="L2")
    for cn in (1, 3):
        src16 = noisy(rng, 31, 23, cn, 65535, 0.06)
        for hval in (900.0, 4000.0):
            out = cv2.fastNlMeansDenoising(src16, h=[hval], templateWindowSize=5, searchWindowSize=13, normType=cv2.NORM_L1)
            add(f"l1_u16_c{cn}_h{hval}", src16, out, op="nlm", h=[hval], t=5, s=13, norm="L1")
    big = noisy(rng, 96, 80, 1, 255, 0.1)
    add("l2_u8_big", big, cv2.fastNlMeansDenoising(big, None, 10, 7, 21), op="nlm", h=[10.0], t=7, s=21, norm="L2")
    for cn in (3, 4):
        col = noisy(rng, 35, 27, cn, 255, 0.07)
        out = cv2.fastNlMeansDenoisingColored(col, None, 6, 9, 5, 15)
        add(f"colored_c{cn}", col, out, op="colored", h=[6.0], hColor=9.0, t=5, s=15)

    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "nlmeans_golden.json")
    with open(out, "w") as f:
        json.dump({"cases": cases}, f, separators=(",", ":"))
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
