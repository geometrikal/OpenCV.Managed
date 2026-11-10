#!/usr/bin/env python3
"""Golden-data generator for the managed stitching warper, blender and exposure-compensator port.

Runs cv2 5.0.0 ``PyRotationWarper`` (plane, affine, cylindrical, spherical),
``detail_FeatherBlender``, ``detail_MultiBandBlender``, the ``detail`` blending helpers and the
``detail`` exposure compensators over synthetic overlapping tiles with known K/R, and writes
``TestData/stitch_warp_golden.json``. ``Stitching/WarperGoldenTests.cs``,
``Stitching/BlenderGoldenTests.cs`` and ``Stitching/ExposureGoldenTests.cs`` replay the cases.

Required environment (the script stops if cv2 is not 5.0.0):
 * cv2 5.0.0, for example
       pip install --no-deps --target <dir> opencv-contrib-python-headless==5.0.0.93
   and run with ``PYTHONPATH=<dir>``. numpy 1.22 or later.
 * OPENCV_CPU_DISABLE=AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL before cv2 is
   imported (the script sets it when it is not set).
 * IPP, OpenCL and threads are disabled in the script.

Regenerate with (from the ``Geometrikal.OpenCV.Managed.Tests`` directory):
    PYTHONPATH=<cv2 5.0.0 dir> python tools/generate_stitch_warp_golden.py

Format: ``arrays`` holds base64 raw buffers; a case refers to its inputs and outputs by index.
Outputs with more than 4096 elements are stored as SHA-256 (``sha``). Float scalars are stored
as their IEEE-754 bit patterns (``bits``) so that the runner compares them exactly.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import hashlib
import json
import os
import platform
import struct
import sys

os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

if cv2.__version__ != "5.0.0":
    sys.exit(f"generate_stitch_warp_golden.py requires cv2 5.0.0 (found {cv2.__version__})")
if cv2.checkHardwareSupport(11) or cv2.checkHardwareSupport(13):
    sys.exit("AVX2/AVX-512 dispatch is enabled; set OPENCV_CPU_DISABLE before importing cv2")

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

DTYPES = {"u8": np.uint8, "i16": np.int16, "f32": np.float32, "f64": np.float64}
# createLaplacePyr with a CV_16S source writes level 0 into the UMat that wraps the input array.
# Through the Python binding the returned level 0 is the unmodified input and level 1 is not
# pyrDown(input). The blender path (a UMat source) is correct and covered by the multiband cases.
_I16_LAPLACE = "cv2 binding: createLaplacePyr with a CV_16S source aliases the input UMat; returned levels are not the C++ result"
SUSPECT_CASES = {"createLaplacePyr_i16_l1": _I16_LAPLACE, "createLaplacePyr_i16_l3": _I16_LAPLACE}


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def dt_of(arr):
    for k, v in DTYPES.items():
        if arr.dtype == v:
            return k
    raise ValueError(arr.dtype)


def fbits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def dbits(x):
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


class Gen:
    def __init__(self):
        self.cases = []
        self.names = set()
        self.arrays = []
        self.index = {}

    def ref(self, arr):
        arr = np.ascontiguousarray(arr)
        m = {"dtype": dt_of(arr), "width": int(arr.shape[1]), "height": int(arr.shape[0]),
             "channels": 1 if arr.ndim == 2 else int(arr.shape[2]), "data": b64(arr)}
        key = (m["dtype"], m["width"], m["height"], m["channels"], m["data"])
        if key not in self.index:
            self.index[key] = len(self.arrays)
            self.arrays.append(m)
        return self.index[key]

    def out(self, arr):
        raw = isinstance(arr, Raw)
        arr = np.ascontiguousarray(arr.a if raw else arr)
        cn = 1 if arr.ndim == 2 else int(arr.shape[2])
        if arr.size > 4096 and not raw:
            return {"sha": sha(arr), "dtype": dt_of(arr), "width": int(arr.shape[1]),
                    "height": int(arr.shape[0]), "channels": cn}
        return {"array": self.ref(arr)}

    def add(self, name, op, params, inputs, outputs, scalars=None):
        assert name not in self.names, name
        self.names.add(name)
        case = {"name": name, "op": op, "params": params, "inputs": [self.ref(a) for a in inputs],
                "outputs": [self.out(o) for o in outputs], "scalars": scalars or {},
                "suspect": name in SUSPECT_CASES}
        if case["suspect"]:
            case["suspectReason"] = SUSPECT_CASES[name]
        self.cases.append(case)


class Raw:
    """Marks an output that is stored raw at any size (the runner compares it with a tolerance)."""

    def __init__(self, a):
        self.a = a


def table_remap(img, xmap, ymap, interp, border):
    """cv2 remap through fixed-point maps: the OpenCV 4.x table path that the managed remap follows."""
    m1, m2 = cv2.convertMaps(xmap, ymap, cv2.CV_16SC2)
    return cv2.remap(img, m1, m2, interp, borderMode=border)


def tile_image(w, h, seed):
    """Smooth colour gradients plus seeded noise, 8UC3."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    planes = []
    for c in range(3):
        base = 128 + 80 * np.sin(xs / (7.0 + 3 * c + seed) + c) * np.cos(ys / (5.0 + 2 * c + seed))
        noise = rng.integers(-20, 21, size=(h, w))
        planes.append(np.clip(base + noise, 0, 255).astype(np.uint8))
    return np.ascontiguousarray(np.dstack(planes))


def tile_mask(w, h, seed):
    """Elliptic blob mask with a ragged edge, 8U (0 or 255)."""
    rng = np.random.default_rng(1000 + seed)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2 + rng.uniform(-3, 3), h / 2 + rng.uniform(-3, 3)
    r = ((xs - cx) / (0.55 * w)) ** 2 + ((ys - cy) / (0.55 * h)) ** 2
    m = (r + rng.uniform(-0.08, 0.08, size=(h, w)) < 1.0).astype(np.uint8) * 255
    return np.ascontiguousarray(m)


def rot(ax, ay, az):
    rv = np.array([ax, ay, az], dtype=np.float64)
    return cv2.Rodrigues(rv)[0].astype(np.float32)


def kmat(f, cx, cy, aspect=1.0, skew=0.0):
    return np.array([[f, skew, cx], [0, f * aspect, cy], [0, 0, 1]], dtype=np.float32)


INTERP = {"Nearest": cv2.INTER_NEAREST, "Linear": cv2.INTER_LINEAR, "Cubic": cv2.INTER_CUBIC}
BORDER = {"Constant": cv2.BORDER_CONSTANT, "Reflect": cv2.BORDER_REFLECT, "Replicate": cv2.BORDER_REPLICATE,
          "Reflect101": cv2.BORDER_REFLECT_101}


def warper_cases(g):
    src_sizes = [(64, 48), (37, 29)]
    cams = [
        ("id", kmat(60, 31.5, 23.5), np.eye(3, dtype=np.float32)),
        ("rot", kmat(70, 30, 22), rot(0.05, -0.12, 0.03)),
        ("rot2", kmat(55, 33, 25, 1.05, 0.5), rot(-0.2, 0.3, -0.1)),
    ]
    affines = [
        ("aff1", np.array([[1.0, 0.05, 12.5], [-0.04, 0.98, -7.25], [0, 0, 1]], dtype=np.float32)),
        ("aff2", np.array([[0.9, -0.2, -20], [0.15, 1.1, 30.5], [0, 0, 1]], dtype=np.float32)),
    ]
    for wtype, scales in (("plane", [1.0, 70.0]), ("cylindrical", [60.0]), ("spherical", [60.0, 33.3])):
        for scale in scales:
            for (sw, sh) in src_sizes:
                for cname, K, R in cams:
                    if wtype == "plane" and scale == 1.0 and cname == "rot2":
                        continue
                    run_warper(g, wtype, scale, sw, sh, cname, K, R)
    for (sw, sh) in src_sizes:
        for aname, H in affines:
            run_warper(g, "affine", 1.0, sw, sh, aname, kmat(1, 0, 0), H)
            run_warper(g, "affine", 1.0, sw, sh, aname + "K", kmat(50, 20, 15), H)


def run_warper(g, wtype, scale, sw, sh, cname, K, R):
    w = cv2.PyRotationWarper(wtype, scale)
    base = f"{wtype}_s{scale:g}_{sw}x{sh}_{cname}"
    params = {"type": wtype, "scale": fbits(scale), "srcWidth": sw, "srcHeight": sh}
    # warpPoint / warpPointBackward over a few points
    pts = [(0.0, 0.0), (sw - 1.0, 0.0), (12.25, 7.5), (sw / 2.0, sh - 1.0)]
    fw, bw = [], []
    for (x, y) in pts:
        u, v = w.warpPoint((x, y), K, R)
        fw += [fbits(u), fbits(v)]
        bx, by = w.warpPointBackward((u, v), K, R)
        bw += [fbits(bx), fbits(by)]
    g.add(base + "_points", "warpPoint", dict(params, points=[[fbits(x), fbits(y)] for x, y in pts]), [K, R], [],
          {"forward": fw, "backward": bw})
    roi = w.warpRoi((sw, sh), K, R)
    rect, xmap, ymap = w.buildMaps((sw, sh), K, R)
    g.add(base + "_maps", "buildMaps", params, [K, R], [xmap, ymap],
          {"roi": list(map(int, roi)), "mapsRect": list(map(int, rect))})
    img = tile_image(sw, sh, sw + sh)
    for interp, border in (("Linear", "Constant"), ("Nearest", "Reflect"), ("Cubic", "Replicate")):
        corner, dst = w.warp(img, K, R, INTERP[interp], BORDER[border])
        outs = [Raw(dst)]
        if interp == "Linear":
            outs.append(table_remap(img, xmap, ymap, INTERP[interp], BORDER[border]))
        g.add(f"{base}_warp_{interp}_{border}", "warp", dict(params, interp=interp, border=border), [K, R, img], outs,
              {"corner": list(map(int, corner))})
    if wtype != "affine":
        corner, dst = w.warp(img, K, R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        back = w.warpBackward(dst, K, R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT, (sw, sh))
        bnear = w.warpBackward(dst, K, R, cv2.INTER_NEAREST, cv2.BORDER_REFLECT, (sw, sh))
        g.add(base + "_back", "warpBackward", dict(params, interp="Linear", border="Reflect"), [K, R, dst], [Raw(back), bnear])


TILES = [  # (corner x, corner y, width, height)
    (0, 0, 48, 40), (30, 4, 44, 38), (5, 27, 50, 36), (38, 30, 41, 37)]


def tiles(seed_base=0, n=4):
    out = []
    for i, (x, y, w, h) in enumerate(TILES[:n]):
        out.append(((x, y), tile_image(w, h, seed_base + i + 1), tile_mask(w, h, seed_base + i)))
    return out


def blender_cases(g):
    for n in (3, 4):
        for sharp in (0.02, 0.1, 1.0):
            t = tiles(0, n)
            fb = cv2.detail_FeatherBlender(sharp)
            corners = [c for c, _, _ in t]
            sizes = [(im.shape[1], im.shape[0]) for _, im, _ in t]
            fb.prepare(cv2.detail.resultRoi(corners, sizes))
            ins = []
            for c, im, m in t:
                fb.feed(im.astype(np.int16), m, c)
                ins += [im, m]
            dst, dmask = fb.blend(None, None)
            g.add(f"feather_n{n}_s{sharp:g}", "feather", {"sharpness": fbits(sharp), "corners": [list(c) for c in corners]},
                  ins, [dst, dmask])
        for bands in (0, 1, 3, 5):
            for wt, wname in ((cv2.CV_32F, "f32"), (cv2.CV_16S, "i16")):
                for src16 in (False, True):
                    t = tiles(10, n)
                    mb = cv2.detail_MultiBandBlender(0, bands, wt)
                    corners = [c for c, _, _ in t]
                    sizes = [(im.shape[1], im.shape[0]) for _, im, _ in t]
                    mb.prepare(cv2.detail.resultRoi(corners, sizes))
                    ins = []
                    for c, im, m in t:
                        mb.feed(im.astype(np.int16) if src16 else im, m, c)
                        ins += [im, m]
                    dst, dmask = mb.blend(None, None)
                    g.add(f"multiband_n{n}_b{bands}_{wname}_{'s16' if src16 else 'u8'}", "multiband",
                          {"bands": bands, "weightType": wname, "src16": src16, "corners": [list(c) for c in corners]},
                          ins, [dst, dmask])
    # Blending helpers.
    for i, sharp in enumerate((0.02, 0.3)):
        m = tile_mask(45, 33, 50 + i)
        wmap = cv2.detail.createWeightMap(m, sharp, np.zeros((33, 45), np.float32))
        g.add(f"createWeightMap_{i}", "createWeightMap", {"sharpness": fbits(sharp)}, [m], [wmap])
    for levels in (0, 1, 3):
        for dt in ("u8", "i16"):
            im = tile_image(40, 32, 60 + levels)
            if dt == "i16":
                im = (im.astype(np.int16) - 100)
            pyr = [u.get() if isinstance(u, cv2.UMat) else u for u in cv2.detail.createLaplacePyr(im, levels, None)]
            g.add(f"createLaplacePyr_{dt}_l{levels}", "createLaplacePyr", {"levels": levels}, [im], list(pyr))
            rest = cv2.detail.restoreImageFromLaplacePyr([cv2.UMat(p.copy()) for p in pyr])
            rest0 = rest[0].get() if isinstance(rest[0], cv2.UMat) else rest[0]
            g.add(f"restoreLaplacePyr_{dt}_l{levels}", "restoreLaplacePyr", {"levels": levels}, list(pyr), [rest0])
    rng = np.random.default_rng(77)
    src = rng.integers(-3000, 3000, size=(21, 34, 3)).astype(np.int16)
    wf = rng.uniform(0, 3, size=(21, 34)).astype(np.float32)
    wf[0, :5] = 0
    ws = rng.integers(0, 600, size=(21, 34)).astype(np.int16)
    g.add("normalize_f32", "normalizeUsingWeightMap", {}, [wf, src], [cv2.detail.normalizeUsingWeightMap(wf, src.copy())])
    g.add("normalize_i16", "normalizeUsingWeightMap", {}, [ws, src], [cv2.detail.normalizeUsingWeightMap(ws, src.copy())])


def exposure_inputs(n, seed, gray=False):
    t = tiles(seed, n)
    corners, imgs, masks = [], [], []
    for i, (c, im, m) in enumerate(t):
        gain = 0.7 + 0.15 * i
        im = np.clip(im.astype(np.float64) * gain, 0, 255).astype(np.uint8)
        if gray:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        corners.append(c)
        imgs.append(im)
        masks.append(m)
    return corners, imgs, masks


def exposure_cases(g):
    def run(name, comp, n, seed, gray=False, params=None):
        corners, imgs, masks = exposure_inputs(n, seed, gray)
        comp.feed(corners, [i.copy() for i in imgs], masks)
        gains = comp.getMatGains()
        outs = []
        for i in range(n):
            outs.append(comp.apply(i, corners[i], imgs[i].copy(), masks[i]))
        scal = {}
        gain_arrays = []
        for gm in gains:
            gm = np.asarray(gm)
            if gm.dtype == np.float64:
                scal.setdefault("gains", []).append([dbits(v) for v in gm.ravel()])
            else:
                gain_arrays.append(np.ascontiguousarray(gm.astype(np.float32)))
        ins = []
        for im, m in zip(imgs, masks):
            ins += [im, m]
        g.add(name, "exposure", dict(params or {}, corners=[list(c) for c in corners], gray=gray), ins,
              gain_arrays + outs, scal)

    for n in (3, 4):
        for nf in (1, 2):
            for thr in (1.0, 0.1):
                c = cv2.detail_GainCompensator(nf)
                c.setSimilarityThreshold(thr)
                run(f"gain_n{n}_f{nf}_t{thr:g}", c, n, 20, params={"kind": "gain", "nrFeeds": nf, "threshold": dbits(thr)})
        c = cv2.detail_GainCompensator(1)
        run(f"gain_gray_n{n}", c, n, 30, gray=True, params={"kind": "gain", "nrFeeds": 1, "threshold": dbits(1.0)})
        for nf in (1, 2):
            c = cv2.detail_ChannelsCompensator(nf)
            run(f"channels_n{n}_f{nf}", c, n, 40, params={"kind": "channels", "nrFeeds": nf, "threshold": dbits(1.0)})
        for bw, bh in ((32, 32), (16, 12)):
            for it in (2, 0):
                c = cv2.detail_BlocksGainCompensator(bw, bh, 1)
                c.setNrGainsFilteringIterations(it)
                run(f"blocksgain_n{n}_{bw}x{bh}_i{it}", c, n, 50,
                    params={"kind": "blocksgain", "nrFeeds": 1, "bw": bw, "bh": bh, "iters": it, "threshold": dbits(1.0)})
                c = cv2.detail_BlocksChannelsCompensator(bw, bh, 1)
                c.setNrGainsFilteringIterations(it)
                run(f"blockschannels_n{n}_{bw}x{bh}_i{it}", c, n, 60,
                    params={"kind": "blockschannels", "nrFeeds": 1, "bw": bw, "bh": bh, "iters": it, "threshold": dbits(1.0)})
        c = cv2.detail_BlocksGainCompensator(16, 16, 2)
        c.setSimilarityThreshold(0.2)
        run(f"blocksgain_n{n}_16x16_f2_t0.2", c, n, 70,
            params={"kind": "blocksgain", "nrFeeds": 2, "bw": 16, "bh": 16, "iters": 2, "threshold": dbits(0.2)})


def main():
    g = Gen()
    warper_cases(g)
    blender_cases(g)
    exposure_cases(g)
    assert len(g.cases) > 100
    out = {"generator": "generate_stitch_warp_golden.py", "cv2": cv2.__version__, "numpy": np.__version__,
           "python": platform.python_version(), "arrays": g.arrays, "cases": g.cases}
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "TestData", "stitch_warp_golden.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases, {len(g.arrays)} arrays to {path}")


if __name__ == "__main__":
    main()
