#!/usr/bin/env python3
"""Golden-data generator for the managed stitching seam finder port.

Runs the cv2 5.0.0 seam finders (``detail_DpSeamFinder``, ``detail_GraphCutSeamFinder`` and
the Voronoi finder from ``detail.SeamFinder_createDefault``) over seeded synthetic tile sets and
writes ``TestData/seam_golden.json``. ``Stitching/SeamFinderGoldenTests.cs`` replays the cases
and compares the output masks byte for byte.

Required environment (the script stops if cv2 is not 5.0.0):
 * cv2 5.0.0, for example
       pip install --no-deps --target <dir> opencv-contrib-python-headless==5.0.0.93
   and run with ``PYTHONPATH=<dir>``. numpy 1.22 or later.
 * OPENCV_CPU_DISABLE=AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL before cv2 is
   imported (the script sets it when it is not set). cvtColor and Sobel in the COLOR_GRAD
   costs then use the SSE baseline, as the other generators.
 * IPP, OpenCL and threads are disabled in the script.

Regenerate with (from the ``Geometrikal.OpenCV.Managed.Tests`` directory):
    PYTHONPATH=<cv2 5.0.0 dir> python tools/generate_seam_golden.py

Battery notes:
 * A scene is a set of 2 to 4 tiles cut from one seeded synthetic image (smooth gradient,
   blobs, stripes and noise), with a per-tile gain and noise, at given corners. The masks are
   full rectangles, warped-like convex polygons or masks with holes.
 * Images are float32 (the ``images_warped_f`` of the stitching sample). The DP finder also
   gets uint8 copies and 4-channel copies.
 * cv2 ``detail_VoronoiSeamFinder()`` has no constructor binding; calling ``find`` on it fails.
   The script uses ``detail.SeamFinder_createDefault(SeamFinder_VORONOI_SEAM)``.
 * Pair order of the DP finder: ``std::sort`` is not stable, and grid layouts give many equal
   pair distances. The ``grid*x*`` scenes (36 to 300 pairs; the dense grids overlap by more than half a tile) check the MSVC STL order of
   the Windows wheels (insertion sort up to 32 elements, median-guess partitions above).
   A libstdc++ build of cv2 can order equal pairs differently on these scenes.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import json
import os
import platform
import sys

os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

if cv2.__version__ != "5.0.0":
    sys.exit(f"generate_seam_golden.py requires cv2 5.0.0 (found {cv2.__version__}); "
             "set PYTHONPATH to the opencv-contrib-python-headless==5.0.0.93 install")
if cv2.checkHardwareSupport(11) or cv2.checkHardwareSupport(13):
    sys.exit("AVX2/AVX-512 dispatch is enabled; set OPENCV_CPU_DISABLE before importing cv2")

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

SUSPECT_CASES = {}


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def enc(arr):
    arr = np.ascontiguousarray(arr)
    cn = 1 if arr.ndim == 2 else arr.shape[2]
    dtype = {np.dtype(np.uint8): "u8", np.dtype(np.float32): "f32"}[arr.dtype]
    return {"w": int(arr.shape[1]), "h": int(arr.shape[0]), "cn": cn, "dtype": dtype, "data": b64(arr)}


def make_world(rng, w, h):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    world = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        fx, fy = rng.uniform(0.02, 0.12, 2)
        px, py = rng.uniform(0, 6.28, 2)
        world[..., c] = 110 + 50 * np.sin(fx * xx + px) * np.cos(fy * yy + py) + 0.4 * (xx - w / 2) * (c - 1)
    for _ in range(int(rng.integers(6, 14))):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(4, 16)
        col = rng.uniform(-80, 80, 3).astype(np.float32)
        blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r))[..., None]
        world += blob * col
    stripe = (np.sin(xx * 0.7 + yy * 0.3) > 0.6).astype(np.float32)[..., None]
    world += stripe * rng.uniform(-30, 30, 3).astype(np.float32)
    world += rng.normal(0, 4, world.shape).astype(np.float32)
    return np.clip(world, 0, 255).astype(np.float32)


def make_mask(rng, kind, w, h):
    if kind == "full":
        return np.full((h, w), 255, np.uint8)
    m = np.zeros((h, w), np.uint8)
    if kind == "poly":
        pts = np.array([[rng.integers(0, w // 4), rng.integers(0, h // 4)],
                        [w - 1 - rng.integers(0, w // 4), rng.integers(0, h // 5)],
                        [w - 1 - rng.integers(0, w // 5), h - 1 - rng.integers(0, h // 4)],
                        [rng.integers(0, w // 5), h - 1 - rng.integers(0, h // 5)]], np.int32)
        cv2.fillPoly(m, [pts], 255)
        return m
    if kind == "holes":
        m[:] = 255
        for _ in range(int(rng.integers(1, 4))):
            cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
            cv2.circle(m, (cx, cy), int(rng.integers(3, max(4, min(w, h) // 5))), 0, -1)
        return m
    raise ValueError(kind)


# name: (seed, [(w, h, x, y)], mask kinds)
SCENES = {
    "pair_h": (1, [(60, 44, 0, 0), (60, 44, 38, 3)], ["full", "full"]),
    "pair_v": (2, [(50, 56, 0, 0), (52, 56, -4, 34)], ["full", "full"]),
    "pair_diag": (3, [(56, 48, 0, 0), (56, 48, 30, 26)], ["full", "full"]),
    "pair_poly": (4, [(64, 52, 10, 5), (64, 52, 46, 12)], ["poly", "poly"]),
    "pair_holes": (5, [(60, 50, 0, 0), (60, 50, 34, 6)], ["holes", "holes"]),
    "pair_inside": (6, [(70, 60, 0, 0), (30, 24, 20, 18)], ["full", "full"]),
    "pair_sizes": (7, [(48, 40, 5, 0), (72, 30, 30, 22)], ["full", "poly"]),
    "pair_disjoint": (8, [(30, 30, 0, 0), (30, 30, 40, 0)], ["full", "full"]),
    "row3": (9, [(44, 40, 0, 0), (44, 40, 30, 2), (44, 40, 61, -3)], ["full", "full", "full"]),
    "row3_poly": (10, [(50, 46, 0, 0), (50, 46, 34, 4), (50, 46, 67, 1)], ["poly", "poly", "poly"]),
    "grid4": (11, [(40, 36, 0, 0), (40, 36, 28, 0), (40, 36, 0, 26), (40, 36, 28, 26)],
              ["full", "full", "full", "full"]),
    "grid4_poly": (12, [(46, 42, 0, 0), (46, 42, 33, 2), (46, 42, -2, 30), (46, 42, 32, 31)],
                   ["poly", "poly", "poly", "poly"]),
    "grid4_holes": (13, [(48, 44, 0, 0), (48, 44, 30, 0), (48, 44, 0, 28), (48, 44, 30, 28)],
                    ["holes", "full", "holes", "poly"]),
    "mixed4": (14, [(64, 40, 0, 0), (40, 64, 50, -10), (56, 50, 10, 30), (30, 30, 30, 15)],
               ["full", "poly", "holes", "full"]),
}


def grid(seed, cols, rows, w, h, sx, sy, kind="full", jitter=0):
    rng = np.random.default_rng(seed)
    tiles = []
    for r in range(rows):
        for c in range(cols):
            tiles.append((w, h, c * sx + int(rng.integers(-jitter, jitter + 1)), r * sy + int(rng.integers(-jitter, jitter + 1))))
    return (seed, tiles, [kind] * (cols * rows))


SCENES["grid3x3"] = grid(15, 3, 3, 30, 28, 22, 20)
SCENES["grid4x3_poly"] = grid(16, 4, 3, 30, 28, 21, 19, "poly", 1)
SCENES["grid5x5"] = grid(17, 5, 5, 24, 24, 17, 17)
SCENES["grid4x4_dense"] = grid(18, 4, 4, 28, 28, 11, 11)
SCENES["grid3x3_dense"] = grid(19, 3, 3, 36, 32, 14, 12)


def build_scene(name):
    seed, tiles, kinds = SCENES[name]
    rng = np.random.default_rng(1000 + seed)
    xs = [t[2] for t in tiles]
    ys = [t[3] for t in tiles]
    x0, y0 = min(xs), min(ys)
    ww = max(t[2] + t[0] for t in tiles) - x0
    wh = max(t[3] + t[1] for t in tiles) - y0
    world = make_world(rng, ww, wh)
    images, corners, masks = [], [], []
    for (w, h, x, y), kind in zip(tiles, kinds):
        tile = world[y - y0:y - y0 + h, x - x0:x - x0 + w].copy()
        tile = tile * np.float32(rng.uniform(0.85, 1.15)) + rng.normal(0, 3, tile.shape).astype(np.float32)
        images.append(np.clip(tile, 0, 255).astype(np.float32))
        corners.append((x, y))
        masks.append(make_mask(rng, kind, w, h))
    return images, corners, masks


def run_finder(finder, images, corners, masks):
    out = finder.find(images, corners, [m.copy() for m in masks])
    res = []
    for o in out:
        a = o.get() if hasattr(o, "get") else np.asarray(o)
        assert a.dtype == np.uint8 and a.ndim == 2
        res.append(a)
    return res


FINDERS = [
    ("voronoi", "f32", 3, lambda: cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_VORONOI_SEAM)),
    ("dp_color", "f32", 3, lambda: cv2.detail_DpSeamFinder("COLOR")),
    ("dp_colorgrad", "f32", 3, lambda: cv2.detail_DpSeamFinder("COLOR_GRAD")),
    ("dp_color", "u8", 3, lambda: cv2.detail_DpSeamFinder("COLOR")),
    ("dp_colorgrad", "u8", 3, lambda: cv2.detail_DpSeamFinder("COLOR_GRAD")),
    ("dp_colorgrad", "f32", 4, lambda: cv2.detail_DpSeamFinder("COLOR_GRAD")),
    ("dp_color", "u8", 4, lambda: cv2.detail_DpSeamFinder("COLOR")),
    ("gc_color", "f32", 3, lambda: cv2.detail_GraphCutSeamFinder("COST_COLOR")),
    ("gc_colorgrad", "f32", 3, lambda: cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")),
    ("gc_color_custom", "f32", 3, lambda: cv2.detail_GraphCutSeamFinder("COST_COLOR", 500.0, 50.0)),
    ("gc_colorgrad_custom", "f32", 3, lambda: cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD", 2000.0, 10.0)),
]

FINDER_PARAMS = {
    "voronoi": {"finder": "voronoi"},
    "dp_color": {"finder": "dp", "cost": "Color"},
    "dp_colorgrad": {"finder": "dp", "cost": "ColorGrad"},
    "gc_color": {"finder": "graphcut", "cost": "Color", "terminal": 10000.0, "penalty": 1000.0},
    "gc_colorgrad": {"finder": "graphcut", "cost": "ColorGrad", "terminal": 10000.0, "penalty": 1000.0},
    "gc_color_custom": {"finder": "graphcut", "cost": "Color", "terminal": 500.0, "penalty": 50.0},
    "gc_colorgrad_custom": {"finder": "graphcut", "cost": "ColorGrad", "terminal": 2000.0, "penalty": 10.0},
}


def convert(images, dtype, cn):
    out = []
    for im in images:
        if cn == 4:
            alpha = np.full(im.shape[:2] + (1,), 255, np.float32)
            im = np.concatenate([im, alpha], axis=2)
        if dtype == "u8":
            im = np.clip(np.rint(im), 0, 255).astype(np.uint8)
        out.append(np.ascontiguousarray(im))
    return out


def main():
    scenes = []
    cases = []
    changed_any = 0
    for sname in SCENES:
        images, corners, masks = build_scene(sname)
        scene = {"name": sname, "corners": [list(c) for c in corners],
                 "images": [enc(im) for im in images], "masks": [enc(m) for m in masks]}
        si = len(scenes)
        scenes.append(scene)
        for key, dtype, cn, factory in FINDERS:
            if dtype == "u8" and key.startswith("dp") and cn == 3 and sname not in (
                    "pair_h", "pair_holes", "grid4", "grid4_holes", "mixed4", "row3_poly"):
                continue
            if cn == 4 and sname not in ("pair_poly", "grid4_poly", "mixed4"):
                continue
            if key.endswith("_custom") and sname not in ("pair_h", "grid4_holes", "mixed4"):
                continue
            src = convert(images, dtype, cn)
            expected = run_finder(factory(), src, corners, masks)
            changed = sum(int(np.count_nonzero(e != m)) for e, m in zip(expected, masks))
            changed_any += changed > 0
            name = f"{sname}/{key}/{dtype}c{cn}"
            case = {"name": name, "scene": si, "dtype": dtype, "cn": cn, "params": FINDER_PARAMS[key],
                    "expected": [enc(e) for e in expected], "suspect": name in SUSPECT_CASES}
            if name in SUSPECT_CASES:
                case["suspectReason"] = SUSPECT_CASES[name]
            cases.append(case)

    assert changed_any > len(cases) // 2, "too few cases change the masks"
    out = {
        "meta": {"cv2": cv2.__version__, "numpy": np.__version__, "python": platform.python_version(),
                 "platform": platform.platform(), "cpuDisable": os.environ.get("OPENCV_CPU_DISABLE")},
        "scenes": scenes,
        "cases": cases,
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TestData", "seam_golden.json")
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {len(cases)} cases over {len(scenes)} scenes to {os.path.normpath(path)}")


if __name__ == "__main__":
    main()
