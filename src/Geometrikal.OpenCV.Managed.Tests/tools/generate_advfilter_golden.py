#!/usr/bin/env python3
"""Golden-data generator for the managed advanced filtering port.

Runs cv2 5.0.0 (with opencv_contrib) ``filter2D`` (DFT path), ``boxFilter``
(normalized int32 sums), ``sqrBoxFilter``, ``bilateralFilter``,
``ximgproc.guidedFilter`` and ``stackBlur`` over a seeded battery and writes
``TestData/advfilter_golden.json``. ``ImgProc/AdvancedFilterGoldenTests.cs``
replays each case and compares the output bytes exactly.

Required environment (the script stops if cv2 is not 5.0.0):
 * cv2 5.0.0 with contrib, for example
       pip install --no-deps --target <dir> opencv-contrib-python-headless==5.0.0.93
   and run with ``PYTHONPATH=<dir>`` (the contrib wheel contains the main
   modules). numpy 1.22 or later.
 * OPENCV_CPU_DISABLE=AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL
   before cv2 is imported (the script sets it when it is not set). The
   AVX-512 groups use hyphenated names; without them bilateralFilter runs its
   AVX-512 FMA code. The port reproduces the SSE2 dispatch.
 * IPP, OpenCL and threads are disabled in the script. stackBlur output
   depends on the thread count in OpenCV; one thread is the reference.

Regenerate with (from the ``Geometrikal.OpenCV.Managed.Tests`` directory):
    PYTHONPATH=<cv2 5.0.0 dir> python tools/generate_advfilter_golden.py

Battery notes:
 * Border types passed to filter2D and boxFilter have BORDER_ISOLATED, which
   matches the port's isolated-ROI model (the box filter shrink rule depends
   on it). sqrBoxFilter rejects BORDER_ISOLATED (it reaches borderInterpolate
   unstripped), so its cases use plain border types; a constant border then
   does not shrink the kernel of single-row or single-column images.
 * Large cases build their input from the integer hash in ``hash_input``
   (``"gen"`` in the case). Outputs with more than 4096 elements are stored as
   SHA-256 (``expectedSha``), except for the RCPPS-dependent cases.
 * The 1-channel-guide guidedFilter cases depend on the processor's RCPPS
   table. ``rcpProbe`` holds RCPPS results of a probe vector computed on the
   generating processor; the runner compares those cases exactly only when its
   processor gives the same probe results.
 * Arrays are stored once in a shared ``arrays`` list.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import ctypes
import hashlib
import json
import mmap
import os
import platform
import sys

os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

if cv2.__version__ != "5.0.0":
    sys.exit(f"generate_advfilter_golden.py requires cv2 5.0.0 with contrib (found {cv2.__version__}); "
             "set PYTHONPATH to the opencv-contrib-python-headless==5.0.0.93 install")
if not hasattr(cv2, "ximgproc"):
    sys.exit("generate_advfilter_golden.py requires the contrib build (cv2.ximgproc)")
if cv2.checkHardwareSupport(11) or cv2.checkHardwareSupport(13):
    sys.exit("AVX2/AVX-512 dispatch is enabled; set OPENCV_CPU_DISABLE before importing cv2")

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

ISO = cv2.BORDER_ISOLATED
BORDERS = {
    "Constant": cv2.BORDER_CONSTANT, "Replicate": cv2.BORDER_REPLICATE, "Reflect": cv2.BORDER_REFLECT,
    "Reflect101": cv2.BORDER_REFLECT_101, "Wrap": cv2.BORDER_WRAP,
}
DTYPES = {"u8": np.uint8, "i16": np.int16, "u16": np.uint16, "i32": np.int32, "f32": np.float32, "f64": np.float64}
CVDEPTH = {"u8": cv2.CV_8U, "u16": cv2.CV_16U, "i16": cv2.CV_16S, "i32": cv2.CV_32S, "f32": cv2.CV_32F, "f64": cv2.CV_64F}

SUSPECT_CASES = {
    f"bilat_nanrandom_c{cn}_w{w}": "OpenCV 5.0 minMaxLoc resets a lane at NaN; the port skips NaN (CorePortNotes), so the LUT scale differs"
    for cn in (1, 3) for w in (37, 40)
}


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def dt_of(arr):
    for k, v in DTYPES.items():
        if arr.dtype == v:
            return k
    raise ValueError(arr.dtype)


def hash_u32(x, y):
    h = (np.asarray(x, dtype=np.uint32) * np.uint32(73856093)) ^ (np.asarray(y, dtype=np.uint32) * np.uint32(19349663))
    h ^= h >> np.uint32(13)
    h = h * np.uint32(0x5BD1E995)
    h ^= h >> np.uint32(15)
    return h


def hash_input(dt, w, h, cn, seed):
    """Value of element (x, y, c): Hash(x + 131·seed, y·cn + c); bytes take the low 8 bits,
    floats ((h & 0xFFFF) / 65536 − 0.5) · 200 rounded to float32."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.uint32)
    planes = []
    for c in range(cn):
        hv = hash_u32(xs + np.uint32(131 * seed), ys * np.uint32(cn) + np.uint32(c))
        if dt == "u8":
            planes.append((hv & np.uint32(255)).astype(np.uint8))
        else:
            planes.append((((hv & np.uint32(0xFFFF)).astype(np.float64) / 65536.0 - 0.5) * 200.0).astype(np.float32))
    a = np.dstack(planes) if cn > 1 else planes[0]
    return np.ascontiguousarray(a)


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

    def add(self, name, op, inputs, params, out, gen=None, rcp=False):
        assert name not in self.names, name
        self.names.add(name)
        case = {"name": name, "op": op, "params": params, "suspect": name in SUSPECT_CASES, "rcp": rcp}
        if case["suspect"]:
            case["suspectReason"] = SUSPECT_CASES[name]
        out = np.ascontiguousarray(out)
        if gen is not None:
            case["gen"] = gen
            case["inputs"] = []
            if rcp:
                case["expected"] = self.ref(out)
            else:
                case["expectedSha"] = sha(out)
                case["expectedDtype"] = dt_of(out)
        else:
            case["inputs"] = [self.ref(a) for a in inputs]
            if out.size > 4096 and not rcp:
                case["expectedSha"] = sha(out)
                case["expectedDtype"] = dt_of(out)
            else:
                case["expected"] = self.ref(out)
        self.cases.append(case)


def rand(rng, dt, h, w, cn, scale=40.0):
    shape = (h, w) if cn == 1 else (h, w, cn)
    if dt == "u8":
        return rng.integers(0, 256, shape, dtype=np.uint8)
    if dt == "u16":
        return rng.integers(0, 65536, shape, dtype=np.uint16)
    if dt == "i16":
        return rng.integers(-32768, 32768, shape, dtype=np.int16)
    if dt == "f64":
        return rng.standard_normal(shape) * scale
    return (rng.standard_normal(shape) * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# RCPPS probe (x86 only): movups xmm0,[arg0]; rcpps xmm0,xmm0; movups [arg1],xmm0; ret
# ---------------------------------------------------------------------------

def rcpps(values):
    if platform.machine().lower() not in ("amd64", "x86_64"):
        return None
    if os.name == "nt":
        code = bytes([0x0F, 0x10, 0x01, 0x0F, 0x53, 0xC0, 0x0F, 0x11, 0x02, 0xC3])
        k32 = ctypes.windll.kernel32
        k32.VirtualAlloc.restype = ctypes.c_void_p
        k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
        addr = k32.VirtualAlloc(None, 4096, 0x3000, 0x40)
        ctypes.memmove(addr, code, len(code))
    else:
        code = bytes([0x0F, 0x10, 0x07, 0x0F, 0x53, 0xC0, 0x0F, 0x11, 0x06, 0xC3])
        buf = mmap.mmap(-1, 4096, prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
        buf.write(code)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    fn = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(addr)
    src = np.ascontiguousarray(values, np.float32)
    out = np.empty_like(src)
    for i in range(0, src.size, 4):
        fn(src[i:i + 4].ctypes.data, out[i:i + 4].ctypes.data)
    return out


# ---------------------------------------------------------------------------
# filter2D (DFT path and the direct path just below the thresholds)
# ---------------------------------------------------------------------------

def filter2d(g, rng):
    pairs = [("u8", "u8"), ("u8", "i16"), ("u8", "f32"), ("f32", "f32")]

    def kernel(kh, kw, dd):
        k = (rng.standard_normal((kh, kw)) / (kh * kw)).astype(np.float32)
        return (k * 8).astype(np.float32) if dd == "i16" else k

    def run(name, src, ker, dd, anchor, delta, border, gen=None):
        out = cv2.filter2D(src, CVDEPTH[dd], ker, anchor=anchor, delta=delta, borderType=BORDERS[border] | ISO)
        params = {"to": dd, "kernel": g.ref(ker), "anchorX": anchor[0], "anchorY": anchor[1], "delta": delta, "border": border}
        if gen is not None:
            g.add(name, "filter2D", [], params, out, gen=gen)
        else:
            g.add(name, "filter2D", [src], params, out)

    n = 0
    shapes = [(12, 11), (13, 11), (11, 12), (25, 25), (31, 5), (5, 31), (3, 60), (60, 3)]
    for sd, dd in pairs:
        for cn in (1, 3):
            for (kh, kw) in shapes:
                for (h, w) in [(41, 37), (37, 64)]:
                    border = list(BORDERS)[n % 5]
                    anchor = [(-1, -1), (0, 0), (kw - 1, kh - 1)][n % 3]
                    delta = [0.0, 3.3][(n // 3) % 2]
                    src = rand(rng, sd, h, w, cn)
                    run(f"f2d_{sd}_{dd}_c{cn}_k{kh}x{kw}_{h}x{w}_{border}_a{anchor[0]}_{anchor[1]}_d{delta}",
                        src, kernel(kh, kw, dd), dd, anchor, delta, border)
                    n += 1
    # Thresholds: 10x5 (area 50) is the DFT path for u8->f32 only; 7x7 (49), 11x11 (121) stay direct.
    for (sd, dd, kh, kw) in [("u8", "f32", 5, 10), ("u8", "f32", 7, 7), ("f32", "f32", 11, 11), ("u8", "u8", 11, 11),
                             ("u8", "i16", 10, 13), ("f32", "f32", 12, 11)]:
        for cn in (1, 3):
            src = rand(rng, sd, 29, 33, cn)
            run(f"f2d_threshold_{sd}_{dd}_c{cn}_k{kh}x{kw}", src, kernel(kh, kw, dd), dd, (-1, -1), 0.0, "Reflect101")
    # One-row strip and the narrow last tile (302 wide, kernel 3 rows x 60 columns, anchor (0, 1)).
    for sd, dd in pairs:
        src = rand(rng, sd, 20, 600, 1)
        run(f"f2d_strip_{sd}_{dd}", src, kernel(13, 11, dd), dd, (-1, -1), 1.5, "Reflect")
        src = rand(rng, sd, 9, 302, 1)
        run(f"f2d_narrowtile_{sd}_{dd}", src, kernel(3, 60, dd), dd, (0, 1), 0.0, "Reflect101")
    # Several tiles in both directions (input from the hash generator).
    for sd, dd in pairs:
        for cn in (1, 3):
            for (kh, kw, border) in [(15, 15, "Reflect101"), (31, 31, "Constant"), (25, 13, "Wrap")]:
                src = hash_input(sd, 777, 513, cn, 3)
                ker = kernel(kh, kw, dd)
                run(f"f2d_large_{sd}_{dd}_c{cn}_k{kh}x{kw}_{border}", src, ker, dd, (-1, -1), 2.25 if cn == 3 else 0.0,
                    border, gen={"dtype": sd, "width": 777, "height": 513, "channels": cn, "seed": 3})
    # Direct-path rejection of BORDER_WRAP and DFT-path acceptance.
    try:
        cv2.filter2D(rand(rng, "u8", 20, 20, 1), -1, np.ones((3, 3), np.float32) / 9, borderType=cv2.BORDER_WRAP)
        raise AssertionError("cv2 accepted BORDER_WRAP for a direct kernel")
    except cv2.error:
        pass
    cv2.filter2D(rand(rng, "u8", 20, 20, 1), -1, np.ones((13, 11), np.float32) / 143, borderType=cv2.BORDER_WRAP)


# ---------------------------------------------------------------------------
# boxFilter with int32 sums (float32 scale on the vector columns) and sqrBoxFilter
# ---------------------------------------------------------------------------

def box(g, rng):
    cases = [("u8", "f32", (5, 4)), ("u8", "f32", (3, 3)), ("u8", "f32", (6, 6)), ("u8", "f32", (7, 5)),
             ("u8", "u8", (20, 14)), ("u8", "u8", (17, 17)), ("u8", "i16", (5, 4)), ("u8", "i16", (4, 6))]
    n = 0
    for sd, dd, k in cases:
        for cn in (1, 3):
            for (h, w) in [(23, 37), (9, 13), (15, 64)]:
                border = ["Reflect101", "Constant", "Replicate", "Reflect"][n % 4]
                src = rand(rng, sd, h, w, cn)
                out = cv2.boxFilter(src, CVDEPTH[dd], k, normalize=True, borderType=BORDERS[border] | ISO)
                g.add(f"box_{sd}_{dd}_c{cn}_k{k[0]}x{k[1]}_{h}x{w}_{border}", "boxFilter", [src],
                      {"to": dd, "kw": k[0], "kh": k[1], "normalize": True, "border": border}, out)
                n += 1
    src = hash_input("u8", 1001, 377, 3, 5)
    out = cv2.boxFilter(src, cv2.CV_32F, (6, 4), normalize=True, borderType=cv2.BORDER_REFLECT_101 | ISO)
    g.add("box_large_u8_f32", "boxFilter", [], {"to": "f32", "kw": 6, "kh": 4, "normalize": True, "border": "Reflect101"},
          out, gen={"dtype": "u8", "width": 1001, "height": 377, "channels": 3, "seed": 5})


def sqrbox(g, rng):
    n = 0
    for sd in ("u8", "f32"):
        dds = ["f32", "f64", "i32"] if sd == "u8" else ["f32", "f64"]
        for cn in (1, 3):
            for (h, w) in [(20, 31), (33, 13), (1, 17), (17, 1)]:
                for k in [(3, 3), (5, 3), (4, 6), (1, 7)]:
                    for norm in (True, False):
                        dd = dds[n % len(dds)]
                        border = ["Reflect101", "Constant", "Replicate", "Reflect"][n % 4]
                        src = rand(rng, sd, h, w, cn, scale=30.0)
                        out = cv2.sqrBoxFilter(src, CVDEPTH[dd], k, normalize=norm, borderType=BORDERS[border])
                        g.add(f"sqrbox_{sd}_{dd}_c{cn}_{h}x{w}_k{k[0]}x{k[1]}_n{int(norm)}_{border}", "sqrBoxFilter", [src],
                              {"to": dd, "kw": k[0], "kh": k[1], "normalize": norm, "border": border}, out)
                        n += 1
    for sd, dd in [("u8", "f32"), ("f32", "f64")]:
        src = hash_input(sd, 1003, 301, 1, 7)
        out = cv2.sqrBoxFilter(src, CVDEPTH[dd], (9, 7), normalize=True, borderType=cv2.BORDER_REFLECT)
        g.add(f"sqrbox_large_{sd}_{dd}", "sqrBoxFilter", [], {"to": dd, "kw": 9, "kh": 7, "normalize": True, "border": "Reflect"},
              out, gen={"dtype": sd, "width": 1003, "height": 301, "channels": 1, "seed": 7})


# ---------------------------------------------------------------------------
# bilateralFilter
# ---------------------------------------------------------------------------

def bilateral(g, rng):
    n = 0
    params8 = [(3, 30.0, 4.0), (5, 75.0, 3.0), (7, 10.0, 2.0), (9, 200.0, 5.0), (0, 25.0, 1.3), (5, 0.0, 2.0)]
    params32 = [(3, 5.0, 4.0), (5, 20.0, 3.0), (7, 80.0, 2.0), (9, 12.0, 5.0), (0, 30.0, 1.3), (5, 20.0, 0.0)]
    for dt in ("u8", "f32"):
        for cn in (1, 3):
            for (h, w) in [(17, 64), (13, 37), (11, 77), (19, 19), (9, 45)]:
                for (d, sc, ss) in (params8 if dt == "u8" else params32):
                    border = list(BORDERS)[n % 5]
                    src = rand(rng, dt, h, w, cn)
                    out = cv2.bilateralFilter(src, d, sc, ss, borderType=BORDERS[border])
                    g.add(f"bilat_{dt}_c{cn}_{h}x{w}_d{d}_sc{sc}_ss{ss}_{border}", "bilateralFilter", [src],
                          {"d": d, "sigmaColor": sc, "sigmaSpace": ss, "border": border}, out)
                    n += 1
    # NaN neighbours and NaN centres. OpenCV 5 minMaxLoc resets a vector lane's minimum and
    # maximum at a NaN; the port skips NaN. NaNs lie in the upper rows and the extremes in the last
    # row, so both rules give the same range ("nan" cases). The "nanrandom" cases scatter NaNs.
    for cn in (1, 3):
        for w in (37, 40):
            for kind in ("nan", "nanrandom"):
                src = rand(rng, "f32", 15, w, cn)
                flat = src.reshape(-1)
                limit = flat.size // 2 if kind == "nan" else flat.size
                idx = rng.choice(np.arange(1, limit), size=flat.size // 25, replace=False)
                flat[idx] = np.nan
                if kind == "nan":
                    flat[-1] = np.nanmax(flat) + 1
                    flat[-2] = np.nanmin(flat) - 1
                out = cv2.bilateralFilter(src, 5, 30.0, 3.0, borderType=cv2.BORDER_REFLECT_101)
                g.add(f"bilat_{kind}_c{cn}_w{w}", "bilateralFilter", [src],
                      {"d": 5, "sigmaColor": 30.0, "sigmaSpace": 3.0, "border": "Reflect101"}, out)
    # Constant float image: copy path.
    src = np.full((7, 9), 3.5, np.float32)
    g.add("bilat_constant_f32", "bilateralFilter", [src], {"d": 5, "sigmaColor": 30.0, "sigmaSpace": 3.0, "border": "Reflect101"},
          cv2.bilateralFilter(src, 5, 30.0, 3.0))
    # Radius 2 (13 taps) on 8-bit 1-channel images: the vector columns accumulate in the order
    # 0, 12, 1, 2, 3, 9, 10, 11, 4..8, which changes a few pixels of a large image.
    for sc in (12.0, 40.0):
        src = hash_input("u8", 1503, 701, 1, 23)
        out = cv2.bilateralFilter(src, 5, sc, 2.0, borderType=cv2.BORDER_REFLECT_101)
        g.add(f"bilat_order13_u8_sc{sc}", "bilateralFilter", [], {"d": 5, "sigmaColor": sc, "sigmaSpace": 2.0, "border": "Reflect101"},
              out, gen={"dtype": "u8", "width": 1503, "height": 701, "channels": 1, "seed": 23})
    for dt, cn in [("u8", 1), ("u8", 3), ("f32", 1), ("f32", 3)]:
        src = hash_input(dt, 333, 211, cn, 11)
        sc = 40.0 if dt == "u8" else 25.0
        out = cv2.bilateralFilter(src, 9, sc, 4.0, borderType=cv2.BORDER_REFLECT_101)
        g.add(f"bilat_large_{dt}_c{cn}", "bilateralFilter", [], {"d": 9, "sigmaColor": sc, "sigmaSpace": 4.0, "border": "Reflect101"},
              out, gen={"dtype": dt, "width": 333, "height": 211, "channels": cn, "seed": 11})


# ---------------------------------------------------------------------------
# ximgproc.guidedFilter
# ---------------------------------------------------------------------------

def guided(g, rng):
    n = 0
    for gdt in ("u8", "u16", "f32"):
        for gcn in (1, 2, 3):
            for sdt in ("u8", "f32"):
                for scn in (1, 3):
                    for (r, eps) in [(1, 1e-3), (2, 100.0), (8, 2500.0), (0, 50.0), (3, 0.0)]:
                        if (n % 3) != 0 and gdt != "u8":
                            n += 1
                            continue
                        h, w = [(23, 37), (31, 30), (20, 45)][n % 3]
                        scale = [1.0, 1.0, 0.5, 1.0, 0.37][n % 5]
                        dd = ["same", "f32", "u8"][(n // 5) % 3]
                        gd = rand(rng, gdt, h, w, gcn, scale=60.0)
                        if gdt == "f32":
                            gd = np.abs(gd)
                        src = rand(rng, sdt, h, w, scn)
                        ddepth = -1 if dd == "same" else CVDEPTH[dd]
                        out = cv2.ximgproc.guidedFilter(gd, src, r, eps, dDepth=ddepth, scale=scale)
                        g.add(f"guided_g{gdt}c{gcn}_s{sdt}c{scn}_r{r}_e{eps}_sc{scale}_{dd}_{h}x{w}", "guidedFilter", [gd, src],
                              {"radius": r, "eps": eps, "scale": scale, "to": dd}, out, rcp=gcn == 1)
                        n += 1
    for gcn in (1, 3):
        gd = hash_input("u8", 401, 233, gcn, 13)
        src = hash_input("u8", 401, 233, 1, 17)
        out = cv2.ximgproc.guidedFilter(gd, src, 8, 400.0, dDepth=-1)
        g.add(f"guided_large_c{gcn}", "guidedFilter", [], {"radius": 8, "eps": 400.0, "scale": 1.0, "to": "same", "srcSeed": 17},
              out, gen={"dtype": "u8", "width": 401, "height": 233, "channels": gcn, "seed": 13}, rcp=gcn == 1)


# ---------------------------------------------------------------------------
# stackBlur
# ---------------------------------------------------------------------------

def stackblur(g, rng):
    for dt in ("u8", "f32"):
        for cn in (1, 3):
            for (h, w) in [(30, 40), (25, 100), (21, 37), (9, 333)]:
                for ks in [(3, 3), (5, 5), (7, 3), (9, 9), (11, 7), (31, 21), (3, 31), (1, 5), (5, 1), (1, 1)]:
                    src = rand(rng, dt, h, w, cn)
                    out = cv2.stackBlur(src, ks)
                    g.add(f"sb_{dt}_c{cn}_{h}x{w}_k{ks[0]}x{ks[1]}", "stackBlur", [src], {"kw": ks[0], "kh": ks[1]}, out)
    # Kernel larger than the image; integer overflow of the vector multiply (radius 181 and 200).
    src = rand(rng, "u8", 6, 8, 1)
    g.add("sb_u8_bigkernel", "stackBlur", [src], {"kw": 11, "kh": 9}, cv2.stackBlur(src, (11, 9)))
    for value, ks in [(250, 363), (140, 401)]:
        src = np.full((450, 40), value, np.uint8)
        g.add(f"sb_u8_overflow_{value}_k{ks}", "stackBlur", [src], {"kw": 1, "kh": ks}, cv2.stackBlur(src, (1, ks)))
    for dt, cn in [("u8", 1), ("u8", 3), ("f32", 1)]:
        src = hash_input(dt, 1031, 263, cn, 19)
        for ks in [(15, 15), (5, 41)]:
            out = cv2.stackBlur(src, ks)
            g.add(f"sb_large_{dt}_c{cn}_k{ks[0]}x{ks[1]}", "stackBlur", [], {"kw": ks[0], "kh": ks[1]}, out,
                  gen={"dtype": dt, "width": 1031, "height": 263, "channels": cn, "seed": 19})


# ---------------------------------------------------------------------------
# filter2D DFT path for the 16U/16S/64F depth pairs (own seed, so earlier cases keep their inputs)
# ---------------------------------------------------------------------------

def filter2d_depths(g, rng):
    """OpenCV routes every pair except 8U->8U/16S and 32F->32F to the DFT at area >= 50.
    Spectra are float64 for 16U/16S/32F/64F sources and for 8U->64F. The kernel is float32 for
    non-double sources; double sources keep a float64 kernel ("kernelDouble")."""
    pairs = [("u8", "u16"), ("u8", "f64"), ("u16", "u16"), ("u16", "f32"), ("u16", "f64"),
             ("i16", "i16"), ("i16", "f32"), ("i16", "f64"), ("f64", "f64")]
    n = 0
    for sd, dd in pairs:
        for cn in (1, 3):
            for (kh, kw) in [(7, 8), (10, 5), (13, 11), (3, 60)]:
                border = list(BORDERS)[n % 5]
                anchor = [(-1, -1), (0, 0), (kw - 1, kh - 1)][n % 3]
                delta = [0.0, 3.3][(n // 3) % 2]
                src = rand(rng, sd, 37, 41, cn)
                k64 = rng.standard_normal((kh, kw)) / (kh * kw)
                ker = k64 if sd == "f64" else k64.astype(np.float32)
                out = cv2.filter2D(src, CVDEPTH[dd], ker, anchor=anchor, delta=delta, borderType=BORDERS[border] | ISO)
                params = {"to": dd, "kernel": g.ref(ker), "kernelDouble": sd == "f64", "anchorX": anchor[0], "anchorY": anchor[1],
                          "delta": delta, "border": border}
                g.add(f"f2d_depth_{sd}_{dd}_c{cn}_k{kh}x{kw}_{border}_a{anchor[0]}_{anchor[1]}_d{delta}", "filter2D", [src], params, out)
                n += 1
    # Below the threshold (area 49): direct path.
    for sd, dd in pairs:
        src = rand(rng, sd, 23, 29, 1)
        ker = (rng.standard_normal((7, 7)) / 49).astype(np.float32)
        out = cv2.filter2D(src, CVDEPTH[dd], ker, borderType=cv2.BORDER_REFLECT_101 | ISO)
        g.add(f"f2d_depth_direct_{sd}_{dd}", "filter2D", [src],
              {"to": dd, "kernel": g.ref(ker), "kernelDouble": False, "anchorX": -1, "anchorY": -1, "delta": 0.0, "border": "Reflect101"}, out)
    # 64F spectra to 16-bit with a delta: convertTo converts in float on rows of at least 8 elements
    # and in double on shorter rows (narrow images, a 1-column last tile).
    for sd, dd, h, w, kh, kw in [("u16", "u16", 30, 5, 10, 5), ("u16", "u16", 2000, 5, 10, 5), ("i16", "i16", 2000, 7, 7, 8), ("i16", "i16", 9, 7, 7, 8), ("i16", "i16", 9, 302, 3, 60),
                                 ("u16", "u16", 9, 302, 3, 60)]:
        src = rand(rng, sd, h, w, 1)
        ker = (rng.standard_normal((kh, kw)) / (kh * kw)).astype(np.float32)
        out = cv2.filter2D(src, CVDEPTH[dd], ker, anchor=(0, 1), delta=2.7, borderType=cv2.BORDER_REFLECT_101 | ISO)
        g.add(f"f2d_depth_narrow_{sd}_{dd}_{h}x{w}_k{kh}x{kw}", "filter2D", [src],
              {"to": dd, "kernel": g.ref(ker), "kernelDouble": False, "anchorX": 0, "anchorY": 1, "delta": 2.7, "border": "Reflect101"}, out)


def main():
    rng = np.random.default_rng(20261012)
    g = Gen()
    filter2d(g, rng)
    box(g, rng)
    sqrbox(g, rng)
    bilateral(g, rng)
    guided(g, rng)
    stackblur(g, rng)
    filter2d_depths(g, np.random.default_rng(20261013))

    probe_in = np.concatenate([
        np.array([1.0, 1.5, 2.0, 3.0, 0.1, 0.3, 7.25, 1e-3, 1e3, 123.456, 0.987654, 65025.0, 1e-6, 2.5e-5, 4096.5, 0.75],
                 np.float32),
        (rng.random(48) * 1000 + 1e-3).astype(np.float32),
    ])
    probe_out = rcpps(probe_in)
    assert set(SUSPECT_CASES) <= g.names, set(SUSPECT_CASES) - g.names
    doc = {"generator": {"cv2": cv2.__version__, "numpy": np.__version__, "machine": platform.processor()},
           "rcpProbe": {"input": b64(probe_in), "output": b64(probe_out) if probe_out is not None else None},
           "arrays": g.arrays, "cases": g.cases}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TestData", "advfilter_golden.json")
    with open(out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases, {len(g.arrays)} arrays to {out}")


if __name__ == "__main__":
    main()
