#!/usr/bin/env python3
"""Golden-data generator for the managed divSpectrums port.

Runs cv2 4.5.5 divSpectrums on float and double CCS (1-channel) and complex
(2-channel) spectra: 2-D with odd and even sizes, single rows and columns,
DFT_ROWS, with and without conjB. Writes ``TestData/divspectrums_golden.json``;
``Core/DivSpectrumsGoldenTests.cs`` replays each case bit for bit.

Regenerate with:
    python tools/generate_divspectrums_golden.py
(run from the ``Geometrikal.OpenCV.Managed.Tests`` directory).

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import json
import os

import cv2
import numpy as np

FLOAT_ROWS = ("OpenCV commit e1dba2c6d2 (2024) computes the float row-loop pairs in double; 4.5.5 multiplied and "
              "added in float before the double quotient")


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def main():
    rng = np.random.default_rng(20261023)
    cases = []
    for dtype, dname in ((np.float32, "f32"), (np.float64, "f64")):
        for cn in (1, 2):
            for w, h in ((8, 6), (7, 5), (9, 1), (1, 10), (1, 7), (6, 4)):
                shape = (h, w) if cn == 1 else (h, w, 2)
                a = rng.normal(0, 10, shape).astype(dtype)
                b = rng.normal(0, 10, shape).astype(dtype)
                b.flat[3 % b.size] = 0
                for flags, fname in ((0, "none"), (cv2.DFT_ROWS, "rows")):
                    for conj in (False, True):
                        out = cv2.divSpectrums(a, b, flags, conjB=conj)
                        name = f"{dname}_c{cn}_{w}x{h}_{fname}_{'conj' if conj else 'plain'}"
                        cases.append({"name": name, "depth": dname, "width": w, "height": h, "channels": cn, "rows": flags != 0,
                                      "conj": conj, "a": b64(a), "b": b64(b), "result": b64(out), "suspect": dname == "f32", "reason": FLOAT_ROWS if dname == "f32" else None})

    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "divspectrums_golden.json")
    with open(out, "w") as f:
        json.dump({"cases": cases}, f, separators=(",", ":"))
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
