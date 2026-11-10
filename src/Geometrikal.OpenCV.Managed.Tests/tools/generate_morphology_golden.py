#!/usr/bin/env python3
"""Golden-data generator for the managed morphology port.

Runs the reference OpenCV (cv2 4.5.5) morphology operations over a battery of
images and writes inputs + reference outputs to
``TestData/morphology_golden.json``. The managed runner
(``ImgProc/MorphologyGoldenTests.cs``) replays each case through the managed
``Morphology``/``Borders`` API and asserts raw buffers match byte-for-byte.

Regenerate with:
    python tools/generate_morphology_golden.py
(run from the ``Geometrikal.OpenCV.Managed.Tests`` directory).

Version risk is minimal: OpenCV's morphology pixel behaviour has been stable
for many years. MORPH_DIAMOND structuring elements are new in OpenCV 5 and are
covered by analytic C# tests instead of cv2 goldens. The suspect/adjudication
mechanism mirrors the contours/drawing ports; no suspects are expected.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import json
import os
import sys

import cv2
import numpy as np

SUSPECT_CASES = {}

DTYPES = {"u8": np.uint8, "u16": np.uint16, "s16": np.int16, "f32": np.float32, "f64": np.float64}
BORDER = {"constant": cv2.BORDER_CONSTANT, "replicate": cv2.BORDER_REPLICATE,
          "reflect": cv2.BORDER_REFLECT, "wrap": cv2.BORDER_WRAP, "reflect101": cv2.BORDER_REFLECT_101}
BORDER_INT = {"constant": 0, "replicate": 1, "reflect": 2, "wrap": 3, "reflect101": 4}
MORPH_OPS = {"erode": 0, "dilate": 1, "open": 2, "close": 3, "gradient": 4,
             "tophat": 5, "blackhat": 6, "hitmiss": 7}


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


class Gen:
    def __init__(self):
        self.cases = []
        self._names = set()

    def _add(self, name, case):
        assert name not in self._names, f"duplicate case name: {name}"
        self._names.add(name)
        case["name"] = name
        case["suspect"] = name in SUSPECT_CASES
        if case["suspect"]:
            case["suspectReason"] = SUSPECT_CASES[name]
        self.cases.append(case)

    def morph(self, name, img, dtype, channels, op, kernel=None, anchor=None,
              iterations=1, border="constant", border_value=None):
        kwargs = {"iterations": iterations, "borderType": BORDER[border]}
        if anchor is not None:
            kwargs["anchor"] = tuple(anchor)
        if border_value is not None:
            bv = tuple(border_value) + (0,) * (4 - len(border_value))
            kwargs["borderValue"] = bv

        k = None if kernel is None else np.array(kernel, np.uint8 if op != "hitmiss" else np.int32)
        if op == "erode":
            out = cv2.erode(img, k, **kwargs)
        elif op == "dilate":
            out = cv2.dilate(img, k, **kwargs)
        else:
            out = cv2.morphologyEx(img, MORPH_OPS[op] if False else getattr(cv2, {
                "open": "MORPH_OPEN", "close": "MORPH_CLOSE", "gradient": "MORPH_GRADIENT",
                "tophat": "MORPH_TOPHAT", "blackhat": "MORPH_BLACKHAT", "hitmiss": "MORPH_HITMISS",
                "erodeex": "MORPH_ERODE", "dilateex": "MORPH_DILATE"}[op]), k, **kwargs)

        case = {
            "op": "morphologyEx" if op not in ("erode", "dilate") else op,
            "morphOp": MORPH_OPS.get(op, MORPH_OPS.get({"erodeex": "erode", "dilateex": "dilate"}.get(op, op))),
            "dtype": dtype, "channels": channels,
            "width": int(img.shape[1]), "height": int(img.shape[0]),
            "input": b64(img), "expected": b64(out),
            "kernelRows": None if kernel is None else [[int(v) for v in row] for row in kernel],
            "anchor": None if anchor is None else [int(anchor[0]), int(anchor[1])],
            "iterations": iterations,
            "borderType": BORDER_INT[border],
            "borderValue": None if border_value is None else
                [float(v) for v in (tuple(border_value) + (0,) * (4 - len(border_value)))],
        }
        self._add(name, case)

    def element(self, name, shape_name, ksize, anchor=None):
        shape = {"rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS, "ellipse": cv2.MORPH_ELLIPSE}[shape_name]
        kwargs = {}
        if anchor is not None:
            kwargs["anchor"] = tuple(anchor)
        elem = cv2.getStructuringElement(shape, tuple(ksize), **kwargs)
        self._add(name, {
            "op": "getStructuringElement",
            "shape": {"rect": 0, "cross": 1, "ellipse": 2}[shape_name],
            "ksize": [int(ksize[0]), int(ksize[1])],
            "anchor": None if anchor is None else [int(anchor[0]), int(anchor[1])],
            "width": int(elem.shape[1]), "height": int(elem.shape[0]),
            "expected": b64(elem), "dtype": "u8", "channels": 1,
        })

    def border(self, name, img, dtype, channels, top, bottom, left, right, border, value=None):
        kwargs = {}
        if value is not None:
            kwargs["value"] = tuple(value) + (0,) * (4 - len(value))
        out = cv2.copyMakeBorder(img, top, bottom, left, right, BORDER[border], **kwargs)
        self._add(name, {
            "op": "copyMakeBorder", "dtype": dtype, "channels": channels,
            "width": int(img.shape[1]), "height": int(img.shape[0]),
            "input": b64(img), "expected": b64(out),
            "top": top, "bottom": bottom, "left": left, "right": right,
            "borderType": BORDER_INT[border],
            "borderValue": None if value is None else
                [float(v) for v in (tuple(value) + (0,) * (4 - len(value)))],
        })

    def interp(self, name, p, length, border):
        res = cv2.borderInterpolate(p, length, BORDER[border])
        self._add(name, {
            "op": "borderInterpolate", "p": p, "len": length,
            "borderType": BORDER_INT[border], "expectedInt": int(res),
            "dtype": "u8", "channels": 1, "width": 0, "height": 0,
        })


def images():
    rng = np.random.RandomState(42)
    imgs = {
        "rand16a": rng.randint(0, 256, (16, 16)).astype(np.uint8),
        "rand16b": rng.randint(0, 256, (16, 16)).astype(np.uint8),
        "mask24": (rng.rand(24, 24) < 0.45).astype(np.uint8) * 255,
        "ramp": np.tile(np.arange(16, dtype=np.uint8) * 16, (16, 1)),
    }
    imgs["rand3ch"] = rng.randint(0, 256, (12, 12, 3)).astype(np.uint8)
    imgs["u16"] = rng.randint(0, 65536, (12, 12)).astype(np.uint16)
    imgs["s16"] = rng.randint(-32768, 32768, (12, 12)).astype(np.int16)
    imgs["f32"] = (rng.rand(12, 12).astype(np.float32) - 0.5) * 200
    imgs["f64"] = (rng.rand(12, 12) - 0.5) * 200
    return imgs


def main():
    g = Gen()
    imgs = images()

    rect3 = np.ones((3, 3), np.uint8).tolist()
    rect1x5 = np.ones((5, 1), np.uint8).tolist()   # 5 rows x 1 col
    rect5x1 = np.ones((1, 5), np.uint8).tolist()   # 1 row x 5 cols
    ellipse5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)).tolist()
    cross3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)).tolist()
    lshape = [[1, 0, 0], [1, 0, 0], [1, 1, 1]]

    kernels = {
        "rect3": (rect3, None), "col5": (rect1x5, None), "row5": (rect5x1, None),
        "ellipse5": (ellipse5, None), "cross3": (cross3, None),
        "lshape_a00": (lshape, (0, 0)), "lshape_a21": (lshape, (2, 1)),
    }

    # -- erode/dilate over kernels, default border ---------------------------
    for img_name in ("rand16a", "mask24"):
        img = imgs[img_name]
        for kname, (k, a) in kernels.items():
            for op in ("erode", "dilate"):
                g.morph(f"{op}_{img_name}_{kname}", img, "u8", 1, op, k, a)

    # -- border types (BORDER_WRAP is rejected by OpenCV's FilterEngine for
    # morphology — the managed port throws NotSupportedException likewise) ----
    for border in ("constant", "replicate", "reflect", "reflect101"):
        for op in ("erode", "dilate"):
            g.morph(f"{op}_rand16b_rect3_{border}", imgs["rand16b"], "u8", 1, op,
                    rect3, border=border,
                    border_value=[7] if border == "constant" else None)
            g.morph(f"{op}_ramp_ellipse5_{border}", imgs["ramp"], "u8", 1, op,
                    ellipse5, border=border,
                    border_value=[130] if border == "constant" else None)

    # default-constant border (morphologyDefaultBorderValue) explicit cases
    for op in ("erode", "dilate"):
        g.morph(f"{op}_ramp_rect3_defaultborder", imgs["ramp"], "u8", 1, op, rect3)

    # -- iterations -----------------------------------------------------------
    for op in ("erode", "dilate"):
        for it in (0, 2, 3):
            g.morph(f"{op}_mask24_rect3_it{it}", imgs["mask24"], "u8", 1, op, rect3, iterations=it)
            g.morph(f"{op}_mask24_lshape_it{it}", imgs["mask24"], "u8", 1, op, lshape,
                    (2, 1), iterations=it)
        for it in (1, 2):
            g.morph(f"{op}_rand16a_nokernel_it{it}", imgs["rand16a"], "u8", 1, op,
                    None, iterations=it)

    # -- other dtypes and channels --------------------------------------------
    for dt in ("u16", "s16", "f32", "f64"):
        g.morph(f"erode_{dt}_rect3", imgs[dt], dt, 1, "erode", rect3)
        g.morph(f"dilate_{dt}_ellipse5", imgs[dt], dt, 1, "dilate", ellipse5)
    g.morph("erode_rand3ch_rect3", imgs["rand3ch"], "u8", 3, "erode", rect3)
    g.morph("dilate_rand3ch_ellipse5", imgs["rand3ch"], "u8", 3, "dilate", ellipse5)
    g.morph("erode_rand3ch_rect3_const", imgs["rand3ch"], "u8", 3, "erode", rect3,
            border="constant", border_value=[7, 9, 11])

    # -- morphologyEx ----------------------------------------------------------
    for exop in ("open", "close", "gradient", "tophat", "blackhat"):
        g.morph(f"ex_{exop}_mask24_rect3", imgs["mask24"], "u8", 1, exop, rect3)
        g.morph(f"ex_{exop}_rand16a_ellipse5", imgs["rand16a"], "u8", 1, exop, ellipse5)
    g.morph("ex_erode_rand16a_rect3", imgs["rand16a"], "u8", 1, "erodeex", rect3)
    g.morph("ex_dilate_rand16a_rect3", imgs["rand16a"], "u8", 1, "dilateex", rect3)
    g.morph("ex_open_mask24_rect3_it2", imgs["mask24"], "u8", 1, "open", rect3, iterations=2)
    g.morph("ex_gradient_rand3ch_rect3", imgs["rand3ch"], "u8", 3, "gradient", rect3)

    # hit-or-miss (binary mask input, -1/0/1 kernels)
    binmask = (imgs["mask24"] > 0).astype(np.uint8) * 255
    corner = [[0, 1, 0], [-1, 1, 1], [-1, -1, 0]]
    isolated = [[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]]
    hitonly = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    g.morph("ex_hitmiss_corner", binmask, "u8", 1, "hitmiss", corner)
    g.morph("ex_hitmiss_isolated", binmask, "u8", 1, "hitmiss", isolated)
    g.morph("ex_hitmiss_hitonly", binmask, "u8", 1, "hitmiss", hitonly)

    # -- getStructuringElement --------------------------------------------------
    for shape in ("rect", "cross", "ellipse"):
        for ks in ((1, 1), (3, 3), (5, 5), (7, 4), (4, 7), (6, 6), (9, 9)):
            g.element(f"elem_{shape}_{ks[0]}x{ks[1]}", shape, ks)
    g.element("elem_cross_5x5_a00", "cross", (5, 5), (0, 0))
    g.element("elem_cross_5x5_a21", "cross", (5, 5), (2, 1))

    # -- copyMakeBorder -----------------------------------------------------------
    small = np.arange(20, dtype=np.uint8).reshape(4, 5) * 10
    for border in ("constant", "replicate", "reflect", "reflect101", "wrap"):
        g.border(f"cmb_{border}", small, "u8", 1, 2, 1, 3, 2, border,
                 value=[7] if border == "constant" else None)
    g.border("cmb_constant_3ch", imgs["rand3ch"][:4, :5], "u8", 3, 1, 2, 2, 1,
             "constant", value=[7, 9, 11])
    thin = np.array([[3], [9], [27]], np.uint8)  # 1-wide image, len==1 reflect edge
    g.border("cmb_reflect_thin", thin, "u8", 1, 1, 1, 2, 2, "reflect")
    g.border("cmb_reflect101_thin", thin, "u8", 1, 1, 1, 2, 2, "reflect101")

    # -- borderInterpolate ----------------------------------------------------------
    cases = [(-1, 10), (-5, 10), (-13, 10), (0, 10), (9, 10), (10, 10), (14, 10),
             (25, 10), (-1, 1), (3, 1), (-7, 3), (8, 3)]
    for border in ("replicate", "reflect", "reflect101", "wrap", "constant"):
        for p, length in cases:
            g.interp(f"bi_{border}_p{p}_l{length}", p, length, border)

    # -- sanity ---------------------------------------------------------------
    all255 = np.full((8, 8), 255, np.uint8)
    assert (cv2.erode(all255, np.ones((3, 3), np.uint8)) == 255).all()
    assert (cv2.dilate(np.zeros((8, 8), np.uint8), np.ones((3, 3), np.uint8)) == 0).all()
    grad = cv2.morphologyEx(imgs["mask24"], cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    d = cv2.dilate(imgs["mask24"], np.ones((3, 3), np.uint8))
    e = cv2.erode(imgs["mask24"], np.ones((3, 3), np.uint8))
    assert (grad == cv2.subtract(d, e)).all()
    print("sanity checks passed")

    out_path = os.path.join(os.path.dirname(__file__), "..", "TestData", "morphology_golden.json")
    out_path = os.path.abspath(out_path)
    with open(out_path, "w") as f:
        json.dump({"cases": g.cases}, f, indent=1)
    print(f"wrote {len(g.cases)} cases to {out_path}")
    print(f"file size: {os.path.getsize(out_path)} bytes")


if __name__ == "__main__":
    sys.exit(main())
