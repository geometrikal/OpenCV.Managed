#!/usr/bin/env python3
"""Golden-data generator for the managed geometric transforms port.

Runs cv2 4.5.5 ``resize``, ``warpAffine``, ``warpPerspective``, ``remap``,
``convertMaps``, the transform-matrix helpers and ``flip``/``rotate``/
``transpose`` over a seeded battery and writes ``TestData/geometry_golden.json``.
``ImgProc/GeometryGoldenTests.cs`` replays each case.

Regenerate with:
    python tools/generate_geometry_golden.py
(run from the ``Geometrikal.OpenCV.Managed.Tests`` directory).

Environment and battery constraints:
 * IPP is disabled: cv2 4.5.5 resize uses IPP for 8U cubic, 16U/16S/32F
   linear and cubic and 64F linear, and the IPP results differ by 1 LSB.
 * OPENCV_CPU_DISABLE removes the AVX/AVX2/FMA dispatch. The resize vertical
   passes that depend on the vector width are the 128-bit SSE kernels.
 * INTER_NEAREST_EXACT: the port follows the OpenCV 5.0 formula. Cases where
   cv2 4.5.5 gives other indices are listed in SUSPECT_CASES (the generator
   asserts that the list is exact) and pinned in GeometryOpenCv5DifferenceTests.
 * BORDER_TRANSPARENT with INTER_LINEAR: OpenCV 5.0 renormalises pixels whose
   base tap is inside the source and whose right or lower tap is outside;
   4.5.5 leaves them unchanged. The battery rejects matrices and maps that
   produce such pixels (checked with the coordinate model below).
 * getPerspectiveTransform with collinear points: 4.5.5 returns [0,...,0,1];
   OpenCV 5.0 uses an SVD fallback. The case is a suspect.
 * WARP_RELATIVE_MAP does not exist in 4.5.5; it is covered by analytic tests.
 * Input and output arrays are stored once in a shared ``arrays`` list. Cases on
   wide images store the SHA-256 of the expected output (``expectedSha256``)
   instead of the pixels.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import hashlib
import json
import math
import os

os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX,AVX2,AVX512F,FMA3")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)
assert cv2.__version__ == "4.5.5", cv2.__version__

DTYPES = {"u8": np.uint8, "i8": np.int8, "u16": np.uint16, "i16": np.int16,
          "i32": np.int32, "f32": np.float32, "f64": np.float64}
INTERP = {"nn": 0, "lin": 1, "cub": 2, "area": 3, "lan": 4, "linx": 5, "nnx": 6}
BORDERS = {"const": 0, "repl": 1, "refl": 2, "wrap": 3, "r101": 4, "transp": 5}
BVAL = (17.25, 200.5, -3.0, 1000.0)

SUSPECT_REASON_NNX = ("cv2 4.5.5 INTER_NEAREST_EXACT uses the 16.16 fixed-point index formula; "
                      "OpenCV 5.0 accumulates src/dst in double precision")
SUSPECT_REASON_PERSP = ("cv2 4.5.5 returns [0,...,0,1] for collinear points; OpenCV 5.0 solves the "
                        "homogeneous system by SVD")

# Adjudicated list; main() asserts that it equals the cases where cv2 differs from the 5.0 formula.
SUSPECT_NNX = [
    "resize_x_nnx_f64_c1_64x48_to_32x24_synth",
    "resize_x_nnx_f64_c4_70x12_to_97x9_synth",
    "resize_x_nnx_i32_c1_64x48_to_32x24_synth",
    "resize_x_nnx_i32_c4_70x12_to_97x9_synth",
    "resize_x_nnx_i8_c1_64x48_to_32x24_synth",
    "resize_x_nnx_i8_c4_70x12_to_97x9_synth",
    "resize_x_nnx_u16_c1_64x48_to_32x24_synth",
    "resize_x_nnx_u16_c4_70x12_to_97x9_synth",
    "resize_x_nnx_u8_c1_40000x1_to_30000x1_synth",
]
SUSPECT_CASES = {name: SUSPECT_REASON_NNX for name in SUSPECT_NNX}
SUSPECT_CASES["perspectiveTransform_collinear"] = SUSPECT_REASON_PERSP
DETECTED_NNX = set()


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def dt_of(arr):
    for k, v in DTYPES.items():
        if arr.dtype == v:
            return k
    raise ValueError(arr.dtype)


class Gen:
    def __init__(self):
        self.cases = []
        self.names = set()
        self.arrays = []
        self.index = {}

    def ref(self, arr):
        arr = np.ascontiguousarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        m = {"dtype": dt_of(arr), "width": int(arr.shape[1]), "height": int(arr.shape[0]),
             "channels": 1 if arr.ndim == 2 else int(arr.shape[2]), "data": b64(arr)}
        key = (m["dtype"], m["width"], m["height"], m["channels"], m["data"])
        if key not in self.index:
            self.index[key] = len(self.arrays)
            self.arrays.append(m)
        return self.index[key]

    def add(self, name, op, inputs, params, expected, hashed=False):
        assert name not in self.names, name
        self.names.add(name)
        case = {"name": name, "op": op, "inputs": [None if a is None else self.ref(a) for a in inputs],
                "params": params, "suspect": name in SUSPECT_CASES}
        if hashed:
            case["expected"] = []
            case["expectedSha256"] = [hashlib.sha256(np.ascontiguousarray(e).tobytes()).hexdigest() for e in expected]
        else:
            case["expected"] = [self.ref(e) for e in expected]
        if case["suspect"]:
            case["suspectReason"] = SUSPECT_CASES[name]
        self.cases.append(case)


def ipp_off():
    cv2.ipp.setUseIPP(False)


# ---------------------------------------------------------------------------- inputs

def make(rng, dt, h, w, cn, kind):
    shape = (h, w) if cn == 1 else (h, w, cn)
    t = DTYPES[dt]
    floating = dt in ("f32", "f64")
    if kind == "const":
        v = 37.5 if floating else 77
        return np.full(shape, v, t)
    if kind == "ramp":
        y, x = np.mgrid[0:h, 0:w]
        base = (x * 7 + y * 13).astype(np.float64)
        if cn > 1:
            base = base[:, :, None] + np.arange(cn) * 31
        if floating:
            return (base * 0.37 - 20).astype(t)
        info = np.iinfo(t)
        return (np.mod(base, float(info.max) - float(info.min) + 1) + float(info.min)).astype(t)
    if kind == "extreme":
        if floating:
            return rng.choice(np.array([-1e6, 1e6, 0.0, -0.5, 3.25], t), shape)
        info = np.iinfo(t)
        return rng.choice(np.array([info.min, info.max], t), shape)
    if floating:
        return (rng.random(shape) * 300 - 20).astype(t)
    info = np.iinfo(t)
    return rng.integers(int(info.min), int(info.max), shape, endpoint=True).astype(t)


SOURCES = {}


def synth(dt, h, w, cn):
    """Formula image for wide cases; GeometryGoldenTests.Synthetic builds the same pixels."""
    y, x, c = np.meshgrid(np.arange(h, dtype=np.int64), np.arange(w, dtype=np.int64), np.arange(cn, dtype=np.int64),
                          indexing="ij")
    base = (x * 73 + y * 151 + c * 37 + ((x * y * 5) >> 2)) % 65536
    if dt == "u8":
        v = (base & 255).astype(np.uint8)
    elif dt == "i8":
        v = ((base & 255) - 128).astype(np.int8)
    elif dt == "u16":
        v = base.astype(np.uint16)
    elif dt == "i16":
        v = (base - 32768).astype(np.int16)
    elif dt == "i32":
        v = ((base - 32768) * 60001).astype(np.int32)
    else:
        v = (base * 0.0371 - 600.0).astype(DTYPES[dt])
    return v[:, :, 0] if cn == 1 else v


def source(rng, dt, h, w, cn, kind):
    """Shared source images: cases with the same shape and data kind reuse one array."""
    key = (dt, h, w, cn, kind)
    if key not in SOURCES:
        SOURCES[key] = make(rng, dt, h, w, cn, kind)
    return SOURCES[key]


# ---------------------------------------------------------------------------- resize

def nn_exact_50_tab(s, d):
    sc = s / d
    f = sc * 0.5
    out = []
    for _ in range(d):
        out.append(min(math.floor(f), s - 1))
        f += sc
    return out


def nn_exact_50(src, dw, dh):
    h, w = src.shape[:2]
    return src[np.ix_(nn_exact_50_tab(h, dh), nn_exact_50_tab(w, dw))]


SMALL_SIZES = [((13, 17), (29, 23)), ((13, 17), (7, 5)), ((13, 17), (40, 9)), ((17, 13), (9, 30)),
               ((1, 9), (5, 7)), ((9, 1), (4, 3)), ((13, 17), (1, 1)), ((16, 20), (10, 8)),
               ((18, 21), (7, 6)), ((31, 29), (11, 10)), ((24, 37), (61, 19)), ((12, 26), (17, 21))]
LARGE_SIZES = [((24, 100), (141, 18)), ((32, 96), (57, 21)), ((48, 64), (32, 24)), ((30, 50), (17, 11))]
SCALE_FORMS = [((13, 17), 0.37, 0.41), ((16, 20), 0.5, 0.5), ((10, 12), 2.5, 1.75), ((18, 21), 1 / 3, 1 / 3),
               ((25, 23), 0.6, 1.4)]
KINDS = ["random", "ramp", "extreme", "random", "const"]


def resize_dtypes(interp):
    if interp in ("nn", "nnx"):
        return list(DTYPES)
    if interp == "linx":
        return ["u8", "i8", "u16", "i16", "i32", "f32"]
    return ["u8", "u16", "i16", "f32", "f64"]


def resize_case(g, rng, interp, dt, cn, src_hw, dsize, fxfy, kind, group="r"):
    h, w = src_hw
    synthetic = group != "r"
    a = synth(dt, h, w, cn) if synthetic else source(rng, dt, h, w, cn, kind)
    ipp_off()
    if dsize is not None:
        dw, dh = dsize
        try:
            out = cv2.resize(a, (dw, dh), interpolation=INTERP[interp])
        except cv2.error:
            return
        params = {"dw": dw, "dh": dh, "fx": 0.0, "fy": 0.0, "interp": INTERP[interp]}
        tag = f"{w}x{h}_to_{dw}x{dh}"
    else:
        fx, fy = fxfy
        try:
            out = cv2.resize(a, None, fx=fx, fy=fy, interpolation=INTERP[interp])
        except cv2.error:
            return
        params = {"dw": 0, "dh": 0, "fx": fx, "fy": fy, "interp": INTERP[interp]}
        tag = f"{w}x{h}_fx{fx:.4f}_fy{fy:.4f}"
    if out.ndim == 2 and cn > 1:
        out = out.reshape(out.shape + (1,))
    name = f"resize_{group}_{interp}_{dt}_c{cn}_{tag}_{'synth' if synthetic else kind}"
    if interp == "nnx":
        model = nn_exact_50(a, out.shape[1], out.shape[0])
        if not np.array_equal(model.reshape(out.shape), out):
            if group != "x":
                return
            DETECTED_NNX.add(name)
    if synthetic:
        params.update({"synth": dt, "sw": w, "sh": h, "cn": cn})
        g.add(name, "resize", [None], params, [out], hashed=True)
    else:
        g.add(name, "resize", [a], params, [out])


def resizes(g, rng):
    k = 0
    for interp in INTERP:
        for dt in resize_dtypes(interp):
            for cn in (1, 2, 3, 4, 5):
                if cn == 5 and interp == "area":
                    picks = [((16, 20), (10, 8)), ((18, 21), (7, 6)), ((13, 17), (29, 23))]
                else:
                    picks = [SMALL_SIZES[k % len(SMALL_SIZES)], SMALL_SIZES[(k + 5) % len(SMALL_SIZES)]]
                for i, (src_hw, dsize) in enumerate(picks):
                    resize_case(g, rng, interp, dt, cn, src_hw, dsize, None, KINDS[(k + i) % len(KINDS)])
                src_hw, fx, fy = SCALE_FORMS[k % len(SCALE_FORMS)]
                if cn in (1, 3):
                    resize_case(g, rng, interp, dt, cn, src_hw, None, (fx, fy), KINDS[k % len(KINDS)])
                k += 1
    # Full size list for two configurations, including wide images (vector bodies and tails).
    for interp in INTERP:
        for dt, cn in (("u8", 1), ("u8", 3), ("f32", 1), ("i16", 3)):
            if dt not in resize_dtypes(interp):
                continue
            for src_hw, dsize in SMALL_SIZES + LARGE_SIZES:
                resize_case(g, rng, interp, dt, cn, src_hw, dsize, None, "random", "f")
            for src_hw, fx, fy in SCALE_FORMS:
                resize_case(g, rng, interp, dt, cn, src_hw, None, (fx, fy), "random", "f")
    for dt in ("u16", "f64", "i8", "i32"):
        for interp in ("nn", "nnx", "linx", "lin", "cub", "lan", "area"):
            if dt in resize_dtypes(interp):
                resize_case(g, rng, interp, dt, 4, (12, 70), (97, 9), None, "random", "x")
                resize_case(g, rng, interp, dt, 1, (48, 64), (32, 24), None, "random", "x")
    # INTER_NEAREST_EXACT index overflow of the 4.5.5 formula on a wide row.
    resize_case(g, rng, "nnx", "u8", 1, (1, 40000), (30000, 1), None, "random", "x")
    resize_case(g, rng, "nn", "u8", 1, (1, 40000), (30000, 1), None, "random", "x")


# ---------------------------------------------------------------------------- warp coordinate model

def sat_int(v):
    r = np.rint(v)
    if not np.isfinite(r) or r < -2 ** 31 or r > 2 ** 31 - 1:
        return -2 ** 31
    return int(r)


def sat_short(v):
    return max(-32768, min(32767, v))


def invert_affine(m):
    m = [float(v) for v in m.reshape(-1)]
    d = m[0] * m[4] - m[1] * m[3]
    d = 1. / d if d != 0 else 0.
    a11 = m[4] * d
    a22 = m[0] * d
    m0 = a11
    m1 = m[1] * -d
    m3 = m[3] * -d
    m4 = a22
    return [m0, m1, -m0 * m[2] - m1 * m[5], m3, m4, -m3 * m[2] - m4 * m[5]]


def invert3(m):
    s = [[float(m[i, j]) for j in range(3)] for i in range(3)]
    d = (s[0][0] * (s[1][1] * s[2][2] - s[1][2] * s[2][1]) - s[0][1] * (s[1][0] * s[2][2] - s[1][2] * s[2][0])
         + s[0][2] * (s[1][0] * s[2][1] - s[1][1] * s[2][0]))
    if d == 0:
        return [0.0] * 9
    d = 1. / d
    return [(s[1][1] * s[2][2] - s[1][2] * s[2][1]) * d, (s[0][2] * s[2][1] - s[0][1] * s[2][2]) * d,
            (s[0][1] * s[1][2] - s[0][2] * s[1][1]) * d, (s[1][2] * s[2][0] - s[1][0] * s[2][2]) * d,
            (s[0][0] * s[2][2] - s[0][2] * s[2][0]) * d, (s[0][2] * s[1][0] - s[0][0] * s[1][2]) * d,
            (s[1][0] * s[2][1] - s[1][1] * s[2][0]) * d, (s[0][1] * s[2][0] - s[0][0] * s[2][1]) * d,
            (s[0][0] * s[1][1] - s[0][1] * s[1][0]) * d]


def affine_xy(mi, dw, dh):
    ad = [sat_int(mi[0] * x * 1024) for x in range(dw)]
    bd = [sat_int(mi[3] * x * 1024) for x in range(dw)]
    out = []
    for y in range(dh):
        x0 = sat_int((mi[1] * y + mi[2]) * 1024) + 16
        y0 = sat_int((mi[4] * y + mi[5]) * 1024) + 16
        for x in range(dw):
            out.append((sat_short(((x0 + ad[x]) >> 5) >> 5), sat_short(((y0 + bd[x]) >> 5) >> 5)))
    return out


def persp_xy(mi, dw, dh):
    bh0 = min(16, dh)
    bw0 = min(1024 // bh0, dw)
    out = []
    for y in range(dh):
        for xb in range(0, dw, bw0):
            x0 = mi[0] * xb + mi[1] * y + mi[2]
            y0 = mi[3] * xb + mi[4] * y + mi[5]
            w0 = mi[6] * xb + mi[7] * y + mi[8]
            for x1 in range(min(bw0, dw - xb)):
                w = w0 + mi[6] * x1
                w = 32. / w if w else 0.
                fx = max(-2. ** 31, min(2. ** 31 - 1, (x0 + mi[0] * x1) * w))
                fy = max(-2. ** 31, min(2. ** 31 - 1, (y0 + mi[3] * x1) * w))
                out.append((sat_short(sat_int(fx) >> 5), sat_short(sat_int(fy) >> 5)))
    return out


def map_xy(mx, my):
    out = []
    for vx, vy in zip(mx.reshape(-1), my.reshape(-1)):
        out.append((sat_short(sat_int(np.float32(vx * np.float32(32))) >> 5),
                    sat_short(sat_int(np.float32(vy * np.float32(32))) >> 5)))
    return out


def transparent_differs(xy, w, h):
    """True when OpenCV 5.0 renormalises a pixel that 4.5.5 leaves unchanged (INTER_LINEAR)."""
    for sx, sy in xy:
        inside = 0 <= sx < w - 1 and 0 <= sy < h - 1
        if not inside and 0 <= sx <= w - 1 and 0 <= sy <= h - 1:
            return True
    return False


# ---------------------------------------------------------------------------- warps

WARP_DTYPES = ["u8", "u16", "i16", "f32", "f64"]
WARP_INTERP = ["nn", "lin", "cub", "lan"]


def random_affine(rng, w, h):
    ang = rng.uniform(-40, 40)
    sc = rng.uniform(0.6, 1.5)
    m = cv2.getRotationMatrix2D((w / 2 + 0.3, h / 2 - 0.2), ang, sc)
    m[0, 1] += 0.07
    m[:, 2] += rng.uniform(-3, 3, 2)
    return m


def warp_case(g, rng, op, dt, cn, interp, border, m, inverse, dsize, src_hw, kind, tag, m32=False, hashed=False):
    h, w = src_hw
    a = source(rng, dt, h, w, cn, kind)
    dw, dh = dsize
    pre = make(rng, dt, dh, dw, cn, "random") if border == "transp" else None
    mm = m.astype(np.float32) if m32 else m
    flags = INTERP[interp] | (cv2.WARP_INVERSE_MAP if inverse else 0)
    ipp_off()
    fn = cv2.warpAffine if op == "warpAffine" else cv2.warpPerspective
    if pre is not None:
        out = fn(a, mm, (dw, dh), dst=pre.copy(), flags=flags, borderMode=BORDERS[border], borderValue=BVAL)
    else:
        out = fn(a, mm, (dw, dh), flags=flags, borderMode=BORDERS[border], borderValue=BVAL)
    if out.ndim == 2 and cn > 1:
        out = out.reshape(out.shape + (1,))
    name = f"{op}_{interp}_{border}_{dt}_c{cn}_{tag}"
    params = {"dw": dw, "dh": dh, "interp": INTERP[interp], "inverse": bool(inverse), "border": BORDERS[border],
              "borderValue": list(BVAL), "matrix32": bool(m32)}
    g.add(name, op, [a, mm, pre], params, [out], hashed)


def choose_matrix(rng, op, w, h, dw, dh, interp, border, inverse):
    for _ in range(200):
        m = random_affine(rng, w, h)
        if op == "warpPerspective":
            m = np.array([[m[0, 0], m[0, 1], m[0, 2]], [m[1, 0], m[1, 1], m[1, 2]],
                          [rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), 1.0]])
        if border != "transp" or interp != "lin":
            return m
        if op == "warpAffine":
            mi = [float(v) for v in m.reshape(-1)] if inverse else invert_affine(m)
            xy = affine_xy(mi, dw, dh)
        else:
            mi = [float(v) for v in m.reshape(-1)] if inverse else invert3(m)
            xy = persp_xy(mi, dw, dh)
        if not transparent_differs(xy, w, h):
            return m
    raise AssertionError("no matrix without OpenCV 5.0 transparent difference")


def warps(g, rng):
    src_hw = (13, 15)
    dsize = (17, 14)
    k = 0
    for op in ("warpAffine", "warpPerspective"):
        for dt in WARP_DTYPES:
            for interp in WARP_INTERP:
                for border in BORDERS:
                    cn = (1, 2, 3, 4)[k % 4] if dt != "u8" else None
                    cns = (1, 3, 4) if dt == "u8" else (cn,)
                    for c in cns:
                        inverse = k % 3 == 0
                        m = choose_matrix(rng, op, src_hw[1], src_hw[0], dsize[0], dsize[1], interp, border, inverse)
                        warp_case(g, rng, op, dt, c, interp, border, m, inverse, dsize, src_hw, KINDS[k % 4],
                                  f"k{k}", m32=(k % 7 == 3))
                        k += 1
                    if interp in ("nn", "lin"):
                        m = choose_matrix(rng, op, src_hw[1], src_hw[0], dsize[0], dsize[1], interp, border, False)
                        warp_case(g, rng, op, dt, 5, interp, border, m, False, dsize, src_hw, "random", f"k{k}")
                        k += 1
        for dt in ("i8", "i32"):
            m = choose_matrix(rng, op, 15, 13, 17, 14, "nn", "const", False)
            for border in ("const", "refl", "transp"):
                warp_case(g, rng, op, dt, 3, "nn", border, m, False, dsize, src_hw, "random", f"k{k}")
                k += 1

    # Special matrices (affine).
    w, h = 15, 13
    specials = {
        "identity": np.array([[1.0, 0, 0], [0, 1.0, 0]]),
        "translate_int": np.array([[1.0, 0, 3], [0, 1.0, -2]]),
        "translate_half": np.array([[1.0, 0, 0.5], [0, 1.0, -0.5]]),
        "translate_tie": np.array([[1.0, 0, 1 / 64], [0, 1.0, 3 / 64]]),
        "scale_up": np.array([[2.5, 0, 1], [0, 1.75, -1]]),
        "large_offset": np.array([[1.0, 0, 1e6], [0, 1.0, -1e6]]),
    }
    for name, m in specials.items():
        for interp in WARP_INTERP:
            for dt in ("u8", "f32"):
                for border in ("const", "r101"):
                    warp_case(g, rng, "warpAffine", dt, 3, interp, border, m, False, (31, 23), (h, w), "random",
                              f"{name}")
    # Perspective: horizon line inside the destination (W = 0 and negative W).
    horizon = np.array([[1.0, 0.1, 0.0], [0.05, 1.0, 0.0], [0.05, 0.0, -0.5]])
    for interp in WARP_INTERP:
        for border in ("const", "repl", "wrap"):
            warp_case(g, rng, "warpPerspective", "u8", 1, interp, border, horizon, True, (21, 9), (h, w), "random",
                      "horizon")
    # Several column blocks (bw0 = 64 at 16 rows).
    for t in range(3):
        p = np.array([[0.37 + rng.uniform(-.05, .05), 0.031, 1.7], [0.023, 0.81, -2.1],
                      [rng.uniform(-3e-3, 3e-3), rng.uniform(-3e-3, 3e-3), 1.0]])
        for interp in ("nn", "lin", "cub"):
            warp_case(g, rng, "warpPerspective", "u8", 1, interp, "const", p, True, (150, 40), (50, 60), "random",
                      f"blocks{t}", hashed=True)
    # Wide affine (vector coordinate path).
    m = cv2.getRotationMatrix2D((40.3, 20.2), 17, 0.9)
    for dt, cn in (("u8", 1), ("u8", 3), ("f32", 1), ("u16", 4)):
        for interp in ("nn", "lin", "cub", "lan"):
            warp_case(g, rng, "warpAffine", dt, cn, interp, "refl", m, False, (97, 33), (41, 83), "random", "wide",
                      hashed=True)


# ---------------------------------------------------------------------------- remap

def make_maps(rng, dh, dw, w, h, special):
    mx = (rng.random((dh, dw)) * (w + 4) - 2).astype(np.float32)
    my = (rng.random((dh, dw)) * (h + 4) - 2).astype(np.float32)
    if special:
        mx[0, :4] = [0.5, 1.5, 2.5, -0.5]
        my[0, :4] = [0.5, 1.5, 2.5, -0.5]
        mx[1, :6] = [3 + 1 / 64, 3 + 3 / 64, 3 + 5 / 64, 1e6, -1e6, -40000.5]
        my[1, :6] = [2 + 1 / 64, 1e6, -1e6, 5.0, 2 + 63 / 64, 70000.0]
    return mx, my


def remap_case(g, rng, dt, cn, interp, border, maptype, kind, tag):
    h, w = 13, 15
    dh, dw = 14, 17
    a = source(rng, dt, h, w, cn, kind)
    pre = make(rng, dt, dh, dw, cn, "random") if border == "transp" else None
    mx, my = make_maps(rng, dh, dw, w, h, special=border != "transp")
    if border == "transp" and interp == "lin":
        mx = np.where((mx >= w - 1.1) & (mx < w + 0.1), mx + 1.5, mx).astype(np.float32)
        my = np.where((my >= h - 1.1) & (my < h + 0.1), my + 1.5, my).astype(np.float32)
        assert not transparent_differs(map_xy(mx, my), w, h)
    if maptype == "32FC1":
        m1, m2 = mx, my
    elif maptype == "32FC2":
        m1, m2 = np.dstack([mx, my]), None
    elif maptype == "16SC2+16UC1":
        m1, m2 = cv2.convertMaps(mx, my, cv2.CV_16SC2)
    else:
        m1, m2 = cv2.convertMaps(mx, my, cv2.CV_16SC2, nninterpolation=True)
    ipp_off()
    if pre is not None:
        out = cv2.remap(a, m1, m2, INTERP[interp], dst=pre.copy(), borderMode=BORDERS[border], borderValue=BVAL)
    else:
        out = cv2.remap(a, m1, m2, INTERP[interp], borderMode=BORDERS[border], borderValue=BVAL)
    if out.ndim == 2 and cn > 1:
        out = out.reshape(out.shape + (1,))
    name = f"remap_{maptype}_{interp}_{border}_{dt}_c{cn}_{tag}"
    params = {"interp": INTERP[interp], "border": BORDERS[border], "borderValue": list(BVAL), "maptype": maptype}
    g.add(name, "remap", [a, m1, m2, pre], params, [out])


def remaps(g, rng):
    k = 0
    for dt in WARP_DTYPES:
        for interp in WARP_INTERP:
            for border in BORDERS:
                cns = (1, 3, 4) if dt == "u8" else ((1, 2, 3, 4)[k % 4],)
                for c in cns:
                    maptype = ("32FC1", "32FC2", "16SC2+16UC1")[k % 3]
                    remap_case(g, rng, dt, c, interp, border, maptype, KINDS[k % 4], f"k{k}")
                    k += 1
    for dt in ("u8", "i8", "i32", "f64"):
        for border in BORDERS:
            remap_case(g, rng, dt, 3, "nn", border, "16SC2", "random", f"k{k}")
            remap_case(g, rng, dt, 2, "nn", border, "16SC2+16UC1", "random", f"k{k}nnd")
            k += 1
    for interp in WARP_INTERP:
        remap_case(g, rng, "u8", 5, interp if interp in ("nn", "lin") else "lin", "refl", "32FC1", "random", f"c5_{interp}")


def convert_maps(g, rng):
    dh, dw = 9, 23
    mx = (rng.random((dh, dw)) * 60 - 20).astype(np.float32)
    my = (rng.random((dh, dw)) * 60 - 20).astype(np.float32)
    mx[0, :8] = [0.5, 1.5, 2.5, -0.5, 1 / 64, 3 / 64, 1e6, -1e6]
    my[0, :8] = [-1.5, 2.5, 7 / 64, 1e7, -1e7, 0.25, 33.5, -32768.75]
    mxy = np.dstack([mx, my])
    m1, m2 = cv2.convertMaps(mx, my, cv2.CV_16SC2)
    g.add("convertMaps_32FC1_to_16SC2", "convertMaps", [mx, my], {"from": "32FC1", "to": "16SC2", "nn": False}, [m1, m2])
    m1b, m2b = cv2.convertMaps(mxy, None, cv2.CV_16SC2)
    g.add("convertMaps_32FC2_to_16SC2", "convertMaps", [mxy, None], {"from": "32FC2", "to": "16SC2", "nn": False}, [m1b, m2b])
    n1, _ = cv2.convertMaps(mx, my, cv2.CV_16SC2, nninterpolation=True)
    g.add("convertMaps_32FC1_to_16SC2_nn", "convertMaps", [mx, my], {"from": "32FC1", "to": "16SC2", "nn": True}, [n1])
    n2, _ = cv2.convertMaps(mxy, None, cv2.CV_16SC2, nninterpolation=True)
    g.add("convertMaps_32FC2_to_16SC2_nn", "convertMaps", [mxy, None], {"from": "32FC2", "to": "16SC2", "nn": True}, [n2])
    f1, f2 = cv2.convertMaps(m1, m2, cv2.CV_32FC1)
    g.add("convertMaps_16SC2_to_32FC1", "convertMaps", [m1, m2], {"from": "16SC2", "to": "32FC1", "nn": False}, [f1, f2])
    f3, _ = cv2.convertMaps(m1, m2, cv2.CV_32FC2)
    g.add("convertMaps_16SC2_to_32FC2", "convertMaps", [m1, m2], {"from": "16SC2", "to": "32FC2", "nn": False}, [f3])
    f6, _ = cv2.convertMaps(m1, m2, cv2.CV_32FC2, nninterpolation=True)
    g.add("convertMaps_16SC2_nn_to_32FC2", "convertMaps", [m1, None], {"from": "16SC2", "to": "32FC2", "nn": True}, [f6])
    s1, s2 = cv2.convertMaps(mxy, None, cv2.CV_32FC1)
    g.add("convertMaps_32FC2_to_32FC1", "convertMaps", [mxy, None], {"from": "32FC2", "to": "32FC1", "nn": False}, [s1, s2])
    c2, _ = cv2.convertMaps(mx, my, cv2.CV_32FC2)
    g.add("convertMaps_32FC1_to_32FC2", "convertMaps", [mx, my], {"from": "32FC1", "to": "32FC2", "nn": False}, [c2])


# ---------------------------------------------------------------------------- matrix helpers

def helpers(g, rng):
    for t in range(20):
        c = (float(np.float32(rng.uniform(0, 100))), float(np.float32(rng.uniform(0, 100))))
        ang = float(rng.uniform(-360, 360))
        sc = float(rng.uniform(0.1, 3))
        r = cv2.getRotationMatrix2D(c, ang, sc)
        g.add(f"rotationMatrix_{t}", "rotationMatrix", [np.array([c], np.float32)], {"angle": ang, "scale": sc}, [r])
    for t in range(20):
        s = (rng.random((4, 2)) * 100).astype(np.float32)
        d = (s + rng.normal(0, 5, (4, 2))).astype(np.float32)
        g.add(f"affineTransform_{t}", "affineTransform", [s[:3], d[:3]], {}, [cv2.getAffineTransform(s[:3], d[:3])])
        g.add(f"perspectiveTransform_{t}", "perspectiveTransform", [s, d], {}, [cv2.getPerspectiveTransform(s, d)])
        a = cv2.getAffineTransform(s[:3], d[:3])
        g.add(f"invertAffine64_{t}", "invertAffine", [a], {}, [cv2.invertAffineTransform(a)])
        a32 = a.astype(np.float32)
        g.add(f"invertAffine32_{t}", "invertAffine", [a32], {}, [cv2.invertAffineTransform(a32)])
    s = np.float32([[0, 0], [1, 1], [2, 2], [3, 3]])
    d = np.float32([[0, 0], [1, 0], [2, 0], [3, 0]])
    name = "perspectiveTransform_collinear"
    g.add(name, "perspectiveTransform", [s, d], {}, [cv2.getPerspectiveTransform(s, d)])
    sing = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 1.0]])
    g.add("invertAffine64_singular", "invertAffine", [sing], {}, [cv2.invertAffineTransform(sing)])
    col = np.float32([[0, 0], [1, 1], [2, 2]])
    g.add("affineTransform_collinear", "affineTransform", [col, col + 1], {}, [cv2.getAffineTransform(col, col + 1)])


# ---------------------------------------------------------------------------- flip / rotate / transpose

def transforms(g, rng):
    sizes = [(5, 7), (23, 29), (1, 13), (13, 1), (19, 17)]
    k = 0
    for dt in DTYPES:
        for cn in (1, 2, 3, 4):
            h, w = sizes[k % len(sizes)]
            k += 1
            a = make(rng, dt, h, w, cn, "random")
            for code in (-1, 0, 1):
                out = cv2.flip(a, code)
                g.add(f"flip_{dt}_c{cn}_{w}x{h}_{code}", "flip", [a], {"code": code}, [out.reshape(a.shape)])
            for code in (0, 1, 2):
                out = cv2.rotate(a, code)
                if out.ndim == 2 and cn > 1:
                    out = out.reshape(out.shape + (1,))
                g.add(f"rotate_{dt}_c{cn}_{w}x{h}_{code}", "rotate", [a], {"code": code}, [out])
            out = cv2.transpose(a) if cn <= 4 and a.dtype.itemsize * cn <= 32 else None
            if out is not None:
                if out.ndim == 2 and cn > 1:
                    out = out.reshape(out.shape + (1,))
                g.add(f"transpose_{dt}_c{cn}_{w}x{h}", "transpose", [a], {}, [out])
    for dt, cn, (h, w) in (("u8", 1, (67, 131)), ("u8", 3, (35, 70)), ("u16", 1, (41, 39)), ("f32", 1, (33, 50)),
                           ("f64", 1, (19, 21)), ("i16", 4, (18, 27))):
        a = make(rng, dt, h, w, cn, "random")
        for code in (0, 1, 2):
            out = cv2.rotate(a, code)
            if out.ndim == 2 and cn > 1:
                out = out.reshape(out.shape + (1,))
            g.add(f"rotate_wide_{dt}_c{cn}_{w}x{h}_{code}", "rotate", [a], {"code": code}, [out])
        out = cv2.transpose(a)
        if out.ndim == 2 and cn > 1:
            out = out.reshape(out.shape + (1,))
        g.add(f"transpose_wide_{dt}_c{cn}_{w}x{h}", "transpose", [a], {}, [out])
        for code in (-1, 1):
            g.add(f"flip_wide_{dt}_c{cn}_{w}x{h}_{code}", "flip", [a], {"code": code}, [cv2.flip(a, code).reshape(a.shape)])


def sanity():
    a = np.arange(12, dtype=np.uint8).reshape(3, 4)
    assert np.array_equal(cv2.rotate(a, cv2.ROTATE_90_CLOCKWISE), np.rot90(a, -1))
    assert np.array_equal(cv2.flip(a, -1), a[::-1, ::-1])
    s = np.float32([[0, 0], [1, 1], [2, 2], [3, 3]])
    d = np.float32([[0, 0], [1, 0], [2, 0], [3, 0]])
    assert np.array_equal(cv2.getPerspectiveTransform(s, d).ravel(), [0, 0, 0, 0, 0, 0, 0, 0, 1])
    assert not hasattr(cv2, "WARP_RELATIVE_MAP")


def main():
    rng = np.random.default_rng(20261013)
    sanity()
    g = Gen()
    resizes(g, rng)
    warps(g, rng)
    remaps(g, rng)
    convert_maps(g, rng)
    helpers(g, rng)
    transforms(g, rng)
    for c in g.cases:
        c["suspect"] = c["name"] in SUSPECT_CASES
        if c["suspect"]:
            c["suspectReason"] = SUSPECT_CASES[c["name"]]
    assert set(SUSPECT_CASES) <= g.names, set(SUSPECT_CASES) - g.names
    if DETECTED_NNX != set(SUSPECT_NNX):
        print("NNX suspects differ from SUSPECT_NNX; detected:")
        for name in sorted(DETECTED_NNX):
            print(f'    "{name}",')
        raise SystemExit(1)
    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "geometry_golden.json")
    with open(out, "w") as f:
        json.dump({"arrays": g.arrays, "cases": g.cases}, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases ({len(SUSPECT_CASES)} suspect), {len(g.arrays)} arrays to {out}")
    for name in sorted(SUSPECT_CASES):
        print("suspect:", name)


if __name__ == "__main__":
    main()
