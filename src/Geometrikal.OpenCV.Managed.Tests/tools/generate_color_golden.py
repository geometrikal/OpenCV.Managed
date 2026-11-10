#!/usr/bin/env python3
"""Golden-data generator for the managed cvtColor port.

Writes ``TestData/color_golden.json``; ``ImgProc/ColorGoldenTests.cs`` replays it.

Regenerate with:
    python tools/generate_color_golden.py
(run from the ``Geometrikal.OpenCV.Managed.Tests`` directory).

Environment and battery constraints (see Docs/Notes/ColorPortNotes.md):
 * cv2.setUseOptimized(False) disables IPP and the AVX2/FMA dispatch. The
   dispatched colour kernels then run the 128-bit (SSE baseline) vector code,
   whose vector-body/scalar-tail split the port reproduces.
 * "cube" cases run a conversion on all 2^24 8-bit triples (a 4096x4096
   image, pixel i = (i>>16, (i>>8)&255, i&255)) and store only the SHA-256
   of the output; the runner regenerates the input.
 * The width sweep covers the vector block sizes 4, 8 and 16 and their tails.
 * NaN, Inf and negative float hue are excluded: the clip rules and the
   OpenCV 4.5.5 and 5.0 hue wrap differ there.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import hashlib
import json
import os

import cv2
import numpy as np

cv2.setUseOptimized(False)
cv2.ocl.setUseOpenCL(False)
assert not cv2.ipp.useIPP()

SUSPECT_CASES = {}

DT = {"u8": np.uint8, "u16": np.uint16, "f32": np.float32}

FORWARD = ["BGR2GRAY", "RGB2GRAY", "BGR2HSV", "RGB2HSV", "BGR2HSV_FULL", "RGB2HSV_FULL", "BGR2Lab",
           "RGB2Lab", "LBGR2Lab", "LRGB2Lab", "BGR2XYZ", "RGB2XYZ", "BGR2YCrCb", "RGB2YCrCb",
           "BGR2RGB", "BGR2BGRA", "BGR2RGBA"]
FORWARD4 = ["BGRA2GRAY", "RGBA2GRAY", "BGRA2BGR", "BGRA2RGBA", "RGBA2BGR"]
INVERSE = ["HSV2BGR", "HSV2RGB", "HSV2BGR_FULL", "HSV2RGB_FULL", "Lab2BGR", "Lab2RGB", "Lab2LBGR",
           "Lab2LRGB", "XYZ2BGR", "XYZ2RGB", "YCrCb2BGR", "YCrCb2RGB"]
DEPTHS = {
    "GRAY": ("u8", "u16", "f32"), "HSV": ("u8", "f32"), "Lab": ("u8", "f32"),
    "XYZ": ("u8", "u16", "f32"), "YCrCb": ("u8", "u16", "f32"), "": ("u8", "u16", "f32"),
}
WIDTHS = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100]


def family(code):
    for k in ("GRAY", "HSV", "Lab", "XYZ", "YCrCb"):
        if k in code:
            return k
    return ""


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


class Gen:
    def __init__(self):
        self.arrays, self.index, self.cases, self.names = [], {}, [], set()

    def ref(self, a):
        a = np.ascontiguousarray(a)
        dt = next(k for k, v in DT.items() if a.dtype == v)
        m = {"dtype": dt, "width": int(a.shape[1]), "height": int(a.shape[0]),
             "channels": 1 if a.ndim == 2 else int(a.shape[2]), "data": b64(a)}
        key = (dt, m["width"], m["height"], m["channels"], m["data"])
        if key not in self.index:
            self.index[key] = len(self.arrays)
            self.arrays.append(m)
        return self.index[key]

    def add(self, name, code, src, dcn=0, roi=None):
        assert name not in self.names, name
        self.names.add(name)
        view = src if roi is None else src[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        out = cv2.cvtColor(view, getattr(cv2, "COLOR_" + code), dstCn=dcn)
        case = {"name": name, "code": code, "value": getattr(cv2, "COLOR_" + code), "dcn": dcn, "input": self.ref(src), "expected": self.ref(out),
                "suspect": name in SUSPECT_CASES}
        if roi is not None:
            case["roi"] = list(roi)
        self.cases.append(case)

    def cube(self, code):
        i = np.arange(1 << 24, dtype=np.uint32)
        img = np.stack([(i >> 16) & 255, (i >> 8) & 255, i & 255], axis=-1).astype(np.uint8).reshape(4096, 4096, 3)
        out = cv2.cvtColor(img, getattr(cv2, "COLOR_" + code))
        self.cases.append({"name": f"cube_{code}", "code": code, "value": getattr(cv2, "COLOR_" + code),
                           "dcn": 0, "cube": True,
                           "sha256": hashlib.sha256(np.ascontiguousarray(out).tobytes()).hexdigest(),
                           "outChannels": 1 if out.ndim == 2 else int(out.shape[2]), "suspect": False})
        self.names.add(f"cube_{code}")


def forward_input(rng, dt, w, h, cn):
    if dt == "f32":
        a = rng.uniform(0, 1, (h, w, cn)).astype(np.float32)
        flat = a.reshape(-1)
        specials = [0, 1, 0.04045, 0.0031308, 0.008856, 6 / 29, -0.5, 1.5, 100, 0.5, 0.25]
        n = min(len(specials), flat.size)
        flat[:n] = specials[:n]
        if w * h >= 4 and cn >= 3:
            a[0, -1, :] = a[0, -1, 0]
            a.reshape(-1, cn)[1, 1] = a.reshape(-1, cn)[1, 0]
        return a
    info = np.iinfo(DT[dt])
    a = rng.integers(0, info.max, (h, w, cn), endpoint=True).astype(DT[dt])
    flat = a.reshape(-1)
    ext = [0, 1, info.max // 2, info.max // 2 + 1, info.max - 1, info.max]
    n = min(len(ext), flat.size)
    flat[:n] = ext[:n]
    return a


def inverse_input(rng, dt, fam, w, h):
    if dt != "f32":
        return forward_input(rng, dt, w, h, 3)
    if fam == "HSV":
        a = np.stack([rng.uniform(0, 360, (h, w)), rng.uniform(0, 1, (h, w)), rng.uniform(0, 1, (h, w))], -1)
        a = a.astype(np.float32)
        flat = a.reshape(-1, 3)
        for k, hv in enumerate([0, 60, 120, 180, 240, 300, 360]):
            if k < flat.shape[0]:
                flat[k, 0] = hv
        if flat.shape[0] > 8:
            flat[7, 1] = 0
            flat[8, 1] = 1
        return a
    if fam == "Lab":
        a = np.stack([rng.uniform(0, 100, (h, w)), rng.uniform(-128, 127, (h, w)), rng.uniform(-128, 127, (h, w))], -1)
        a = a.astype(np.float32)
        flat = a.reshape(-1, 3)
        for k, lv in enumerate([0, 7.99, 8, 8.01, 100]):
            if k < flat.shape[0]:
                flat[k, 0] = lv
        if flat.shape[0] > 6:
            flat[5, 1:] = [200, -200]
            flat[6, 1:] = [-200, 200]
        return a
    return rng.uniform(0, 1, (h, w, 3)).astype(np.float32)


def main():
    rng = np.random.default_rng(20261012)
    g = Gen()

    for code in FORWARD + INVERSE:
        if code.startswith("BGR2RGB") or code.startswith("BGR2BGRA") or code.startswith("BGR2RGBA"):
            continue
        g.cube(code)
    g.cube("BGR2RGB")

    for code in FORWARD + FORWARD4 + INVERSE:
        fam = family(code.replace("GRAY", "GRAY"))
        inverse = code in INVERSE
        for dt in DEPTHS[fam]:
            for w in WIDTHS:
                scn = 4 if code in FORWARD4 else 3
                src = inverse_input(rng, dt, fam, w, 3) if inverse else forward_input(rng, dt, w, 3, scn)
                g.add(f"{code}_{dt}_w{w}", code, src)
            big = inverse_input(rng, dt, fam, 41, 9) if inverse else forward_input(rng, dt, 41, 9, 3)
            g.add(f"{code}_{dt}_roi", code, big, roi=(5, 3, 33, 5))
            if inverse:
                g.add(f"{code}_{dt}_dcn4", code, big, dcn=4)
            elif code in FORWARD and code not in ("BGR2RGB", "BGR2BGRA", "BGR2RGBA"):
                src4 = forward_input(rng, dt, 37, 3, 4)
                src4[..., 3] = src4[..., 3][::-1]
                g.add(f"{code}_{dt}_scn4", code, src4)

    for dt in ("u8", "u16", "f32"):
        gray = forward_input(rng, dt, 37, 3, 1)[..., 0]
        g.add(f"GRAY2BGR_{dt}", "GRAY2BGR", gray)
        g.add(f"GRAY2BGRA_{dt}", "GRAY2BGRA", gray)

    assert set(SUSPECT_CASES) <= g.names
    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "color_golden.json")
    meta = {"cv2": cv2.__version__, "useOptimized": cv2.useOptimized(), "useIPP": cv2.ipp.useIPP()}
    with open(out, "w") as f:
        json.dump({"meta": meta, "arrays": g.arrays, "cases": g.cases}, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases ({len(g.arrays)} arrays) to {out}")


if __name__ == "__main__":
    main()
