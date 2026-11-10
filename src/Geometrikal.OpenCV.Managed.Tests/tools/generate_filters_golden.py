#!/usr/bin/env python3
"""Golden-data generator for the managed filtering port.

Runs the reference OpenCV (cv2 4.5.5) filtering functions over a battery of
images and writes inputs + reference outputs to
``TestData/filters_golden.json``. The managed runner
(``ImgProc/FiltersGoldenTests.cs``) replays each case through the managed
``Filters``/``Derivatives`` API and asserts raw buffers match byte-for-byte.

Regenerate with:
    python tools/generate_filters_golden.py
(run from the ``Geometrikal.OpenCV.Managed.Tests`` directory).

Environment constraints (see Docs/Notes/FilteringPortNotes.md):
 * OPENCV_CPU_DISABLE kills the AVX/AVX2/FMA dispatch so cv2's float
   arithmetic matches the scalar (SSE, mul+add) semantics the port
   implements.
 * IPP and OpenCL are disabled — both produce results that differ from
   OpenCV's own C++ paths.
 * BORDER_ISOLATED is OR'd into every border argument to match the port's
   always-isolated ROI semantics.
 * Normalized box goldens use odd*odd kernels, except 8U->8U with kernel
   area <= 256. That path (ColumnSum<ushort, uchar>) divides with one
   fixed-point rule in SIMD and scalar code. Other paths round exact halves
   differently in OpenCV's SIMD lanes and scalar tail.
 * 8U->8U sepFilter2D bit-exact cases are checked for exact-half outputs
   (which would be lane-dependent in OpenCV) and must avoid them.
 * Float images fed to integer-valued "special" kernels (Sobel-like) hold
   integer values so reassociated SIMD sums stay exact.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import json
import os
import sys

os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX,AVX2,AVX512F,FMA3")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

SUSPECT_CASES = {
    name: "OpenCV 5.0 multiplies the double Laplacian kernel by scale in double "
          "(commit 1d5f28a44f); cv2 4.5.5 multiplies a float kernel"
    for name in ("lap_f64_k3_scale01", "lap_u8f64_k1_scale01")
}

ISO = cv2.BORDER_ISOLATED
BORDER = {"constant": cv2.BORDER_CONSTANT, "replicate": cv2.BORDER_REPLICATE,
          "reflect": cv2.BORDER_REFLECT, "reflect101": cv2.BORDER_REFLECT_101}
BORDER_INT = {"constant": 0, "replicate": 1, "reflect": 2, "reflect101": 4}
DDEPTH = {"u8": cv2.CV_8U, "u16": cv2.CV_16U, "s16": cv2.CV_16S, "f32": cv2.CV_32F,
          "f64": cv2.CV_64F}


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


class Gen:
    def __init__(self):
        self.cases = []
        self._names = set()

    def _add(self, name, img, dtype, out, out_dtype, case):
        assert name not in self._names, f"duplicate case name: {name}"
        self._names.add(name)
        case["name"] = name
        case["suspect"] = name in SUSPECT_CASES
        if case["suspect"]:
            case["suspectReason"] = SUSPECT_CASES[name]
        case["dtype"] = dtype
        case["ddepth"] = out_dtype
        case["channels"] = 1 if img.ndim == 2 else int(img.shape[2])
        case["width"] = int(img.shape[1])
        case["height"] = int(img.shape[0])
        case["input"] = b64(img)
        case["expected"] = b64(out)
        self.cases.append(case)

    def gaussian(self, name, img, dtype, ksize, sigma_x, sigma_y=0, border="reflect101"):
        out = cv2.GaussianBlur(img, tuple(ksize), sigma_x, sigmaY=sigma_y,
                               borderType=BORDER[border] | ISO)
        self._add(name, img, dtype, out, dtype, {
            "op": "gaussianBlur", "ksize": [int(ksize[0]), int(ksize[1])],
            "sigmaX": float(sigma_x), "sigmaY": float(sigma_y),
            "borderType": BORDER_INT[border],
        })

    def box(self, name, img, dtype, ksize, ddepth, anchor=None, normalize=True,
            border="reflect101"):
        fixed_point = dtype == "u8" and ddepth == "u8" and ksize[0] * ksize[1] <= 256
        if normalize and not fixed_point:
            assert ksize[0] % 2 == 1 and ksize[1] % 2 == 1, \
                f"{name}: normalized box goldens must use odd kernels"
        out = cv2.boxFilter(img, DDEPTH[ddepth], tuple(ksize),
                            anchor=tuple(anchor) if anchor else (-1, -1),
                            normalize=normalize, borderType=BORDER[border] | ISO)
        self._add(name, img, dtype, out, ddepth, {
            "op": "boxFilter", "ksize": [int(ksize[0]), int(ksize[1])],
            "anchor": None if anchor is None else [int(anchor[0]), int(anchor[1])],
            "normalize": bool(normalize), "borderType": BORDER_INT[border],
        })

    def median(self, name, img, dtype, ksize):
        out = cv2.medianBlur(img, ksize)
        self._add(name, img, dtype, out, dtype,
                  {"op": "medianBlur", "ksize": [int(ksize), int(ksize)]})

    def filter2d(self, name, img, dtype, kernel, ddepth, anchor=None, delta=0.0,
                 border="reflect101"):
        kernel = np.asarray(kernel, np.float32)
        assert kernel.size < 50, f"{name}: filter2D kernels must stay below the DFT threshold"
        out = cv2.filter2D(img, DDEPTH[ddepth], kernel,
                           anchor=tuple(anchor) if anchor else (-1, -1),
                           delta=delta, borderType=BORDER[border] | ISO)
        self._add(name, img, dtype, out, ddepth, {
            "op": "filter2D",
            "kernel": b64(kernel), "kw": int(kernel.shape[1]), "kh": int(kernel.shape[0]),
            "anchor": None if anchor is None else [int(anchor[0]), int(anchor[1])],
            "delta": float(delta), "borderType": BORDER_INT[border],
        })

    def sepfilter(self, name, img, dtype, kx, ky, ddepth, anchor=None, delta=0.0,
                  border="reflect101"):
        kx = np.asarray(kx, np.float32).reshape(-1, 1)
        ky = np.asarray(ky, np.float32).reshape(-1, 1)
        out = cv2.sepFilter2D(img, DDEPTH[ddepth], kx, ky,
                              anchor=tuple(anchor) if anchor else (-1, -1),
                              delta=delta, borderType=BORDER[border] | ISO)
        if dtype == "u8" and ddepth == "u8":
            self._check_no_half_boundary(name, img, kx.ravel(), ky.ravel(), delta,
                                         BORDER[border])
        self._add(name, img, dtype, out, ddepth, {
            "op": "sepFilter2D",
            "kernelX": b64(kx.ravel()), "kernelY": b64(ky.ravel()),
            "anchor": None if anchor is None else [int(anchor[0]), int(anchor[1])],
            "delta": float(delta), "borderType": BORDER_INT[border],
        })

    def _check_no_half_boundary(self, name, img, kx, ky, delta, border):
        """For 8U->8U bit-exact candidates, assert no output sits on an exact
        half boundary (OpenCV's lanes and scalar tail would disagree there)."""
        kxi = np.round(kx.astype(np.float64) * 256).astype(np.int64)
        kyi = np.round(ky.astype(np.float64) * 256).astype(np.int64)
        if (np.abs(kx.astype(np.float64) * 256 - kxi) > 10 * 1.1920928955078125e-7 * 256).any():
            return  # not bit-exact eligible; float path has no half-shift
        if (np.abs(ky.astype(np.float64) * 256 - kyi) > 10 * 1.1920928955078125e-7 * 256).any():
            return
        if (kx < 0).any() or (ky < 0).any():
            return  # not smooth; float path
        work = img if img.ndim == 3 else img[:, :, None]
        cn = work.shape[2]
        ax, ay = len(kxi) // 2, len(kyi) // 2
        padded = cv2.copyMakeBorder(img, ay, len(kyi) - 1 - ay, ax, len(kxi) - 1 - ax,
                                    border | ISO, value=0)
        padded = padded if padded.ndim == 3 else padded[:, :, None]
        p64 = padded.astype(np.int64)
        rows = sum(kxi[k] * p64[:, k:k + work.shape[1], :] for k in range(len(kxi)))
        sums = sum(kyi[k] * rows[k:k + work.shape[0], :, :] for k in range(len(kyi)))
        sums = sums + int(round(delta * 65536))
        assert not ((sums & 0xFFFF) == 0x8000).any(), \
            f"{name}: exact-half output would be lane-dependent in OpenCV; reseed the case"

    def deriv(self, name, img, dtype, op, dx, dy, ddepth, ksize=3, scale=1.0,
              delta=0.0, border="reflect101"):
        if op == "sobel":
            out = cv2.Sobel(img, DDEPTH[ddepth], dx, dy, ksize=ksize, scale=scale,
                            delta=delta, borderType=BORDER[border] | ISO)
        else:
            out = cv2.Scharr(img, DDEPTH[ddepth], dx, dy, scale=scale, delta=delta,
                             borderType=BORDER[border] | ISO)
        self._add(name, img, dtype, out, ddepth, {
            "op": op, "dx": int(dx), "dy": int(dy), "ksize": [int(ksize), int(ksize)],
            "scale": float(scale), "delta": float(delta), "borderType": BORDER_INT[border],
        })

    def laplacian(self, name, img, dtype, ddepth, ksize=1, scale=1.0, delta=0.0,
                  border="reflect101"):
        out = cv2.Laplacian(img, DDEPTH[ddepth], ksize=ksize, scale=scale, delta=delta,
                            borderType=BORDER[border] | ISO)
        self._add(name, img, dtype, out, ddepth, {
            "op": "laplacian", "ksize": [int(ksize), int(ksize)], "scale": float(scale),
            "delta": float(delta), "borderType": BORDER_INT[border],
        })

    def gaussian_kernel(self, name, n, sigma):
        k = cv2.getGaussianKernel(n, sigma, cv2.CV_64F).ravel()
        self.cases.append({
            "name": name, "suspect": False, "op": "getGaussianKernel",
            "n": int(n), "sigma": float(sigma), "expected": b64(k),
            "dtype": "f64", "ddepth": "f64", "channels": 1, "width": 0, "height": 0,
            "input": "",
        })
        self._names.add(name)


def images():
    rng = np.random.RandomState(20260728)
    imgs = {
        "u8c1": rng.randint(0, 256, (24, 32)).astype(np.uint8),
        "u8c1b": rng.randint(0, 256, (16, 16)).astype(np.uint8),
        "u8c3": rng.randint(0, 256, (13, 19, 3)).astype(np.uint8),
        "u8c4": rng.randint(0, 256, (16, 16, 4)).astype(np.uint8),
        "u8c2": rng.randint(0, 256, (8, 8, 2)).astype(np.uint8),
        "ramp": np.tile(np.arange(16, dtype=np.uint8) * 16, (16, 1)),
        "uniform": np.full((12, 12), 200, np.uint8),
        "row1": rng.randint(0, 256, (1, 16)).astype(np.uint8),
        "col1": rng.randint(0, 256, (16, 1)).astype(np.uint8),
        "tiny": rng.randint(0, 256, (3, 3)).astype(np.uint8),
        "one": rng.randint(0, 256, (1, 1)).astype(np.uint8),
        "small54": rng.randint(0, 256, (4, 5)).astype(np.uint8),
        "row1c3": rng.randint(0, 256, (1, 9, 3)).astype(np.uint8),
    }
    # Random float images (non-integer values; used with generic kernels only).
    imgs["f32c1"] = ((rng.rand(16, 24).astype(np.float32) - 0.5) * 200)
    imgs["f32c3"] = ((rng.rand(9, 11, 3).astype(np.float32) - 0.5) * 50)
    # Integer-valued float images (exact under reassociated special kernels).
    imgs["f32i1"] = rng.randint(0, 256, (16, 24)).astype(np.float32)
    imgs["f32i3"] = rng.randint(-500, 500, (10, 14, 3)).astype(np.float32)
    imgs["f32row"] = rng.randint(0, 256, (1, 13)).astype(np.float32)
    return imgs


def depth_images():
    """Inputs of the 16U/16S/64F derivative cases (a separate RNG keeps the older cases stable)."""
    rng = np.random.RandomState(20261011)
    return {
        "u16c1": rng.randint(0, 65536, (17, 23)).astype(np.uint16),
        "u16c3": rng.randint(0, 65536, (9, 13, 3)).astype(np.uint16),
        "s16c1": rng.randint(-32768, 32768, (16, 21)).astype(np.int16),
        "s16c3": rng.randint(-32768, 32768, (10, 11, 3)).astype(np.int16),
        "s16small": rng.randint(-300, 300, (15, 18)).astype(np.int16),
        "f64c1": (rng.rand(14, 19) - 0.5) * 300.0,
        "f64c3": (rng.rand(8, 12, 3) - 0.5) * 40.0,
        "u8big3": rng.randint(0, 256, (97, 131, 3)).astype(np.uint8),
    }


def main():
    g = Gen()
    imgs = images()
    rng = np.random.RandomState(77)

    # ---------------------------------------------------------- kernels
    for i, (n, s) in enumerate([(1, 0.0), (3, 0.0), (5, 0.0), (7, 0.0), (9, 0.0),
                                (13, 0.0), (21, 0.0), (3, 0.8), (5, 1.1), (7, 2.5),
                                (11, 3.7), (21, 10.0), (7, 0.1), (5, -1.0)]):
        g.gaussian_kernel(f"gk_{i}_n{n}", n, s)

    # ------------------------------------------------------ gaussianBlur u8
    g.gaussian("gauss_u8_3x3_s0", imgs["u8c1"], "u8", (3, 3), 0)
    g.gaussian("gauss_u8_5x5_s0", imgs["u8c1"], "u8", (5, 5), 0)
    g.gaussian("gauss_u8_7x7_s0_repl", imgs["u8c1"], "u8", (7, 7), 0, border="replicate")
    g.gaussian("gauss_u8_9x9_s0_const", imgs["u8c1"], "u8", (9, 9), 0, border="constant")
    g.gaussian("gauss_u8_13x13_s0", imgs["u8c1b"], "u8", (13, 13), 0)
    g.gaussian("gauss_u8_5x5_s11", imgs["u8c1"], "u8", (5, 5), 1.1)
    g.gaussian("gauss_u8_3x7_s12_24", imgs["u8c1"], "u8", (3, 7), 1.2, 2.4)
    g.gaussian("gauss_u8_1x5_s0", imgs["u8c1"], "u8", (1, 5), 0)
    g.gaussian("gauss_u8_auto_s06", imgs["u8c1"], "u8", (0, 0), 0.6)
    g.gaussian("gauss_u8_auto_s20_refl", imgs["u8c1"], "u8", (0, 0), 2.0, border="reflect")
    g.gaussian("gauss_u8c3_5x5_s0", imgs["u8c3"], "u8", (5, 5), 0)
    g.gaussian("gauss_u8c4_3x3_s09", imgs["u8c4"], "u8", (3, 3), 0.9)
    g.gaussian("gauss_u8c2_5x5_s0", imgs["u8c2"], "u8", (5, 5), 0)
    g.gaussian("gauss_u8_row1_5x5_s0", imgs["row1"], "u8", (5, 5), 0)
    g.gaussian("gauss_u8_col1_3x3_s0_repl", imgs["col1"], "u8", (3, 3), 0, border="replicate")
    g.gaussian("gauss_u8_row1_5x5_s0_const", imgs["row1"], "u8", (5, 5), 0, border="constant")
    g.gaussian("gauss_u8_tiny_7x7_s0", imgs["tiny"], "u8", (7, 7), 0)
    g.gaussian("gauss_u8_one_3x3_s0_const", imgs["one"], "u8", (3, 3), 0, border="constant")
    g.gaussian("gauss_u8_uniform_5x5_s0", imgs["uniform"], "u8", (5, 5), 0)
    g.gaussian("gauss_u8_ramp_9x3_s3", imgs["ramp"], "u8", (9, 3), 3.0)

    # ----------------------------------------------------- gaussianBlur f32
    g.gaussian("gauss_f32_3x3_s0", imgs["f32c1"], "f32", (3, 3), 0)
    g.gaussian("gauss_f32_5x5_s0", imgs["f32c1"], "f32", (5, 5), 0)
    g.gaussian("gauss_f32_7x7_s21_repl", imgs["f32c1"], "f32", (7, 7), 2.1, border="replicate")
    g.gaussian("gauss_f32_9x9_s0_const", imgs["f32c1"], "f32", (9, 9), 0, border="constant")
    g.gaussian("gauss_f32c3_5x3_s15_07", imgs["f32c3"], "f32", (5, 3), 1.5, 0.7)
    g.gaussian("gauss_f32_auto_s14", imgs["f32c1"], "f32", (0, 0), 1.4)
    g.gaussian("gauss_f32_row_3x3_s0", imgs["f32row"], "f32", (3, 3), 0)

    # ------------------------------------------------------------- box/blur
    g.box("blur_u8_3x3", imgs["u8c1"], "u8", (3, 3), "u8")
    g.box("blur_u8_5x5_repl", imgs["u8c1"], "u8", (5, 5), "u8", border="replicate")
    g.box("blur_u8_1x3", imgs["u8c1"], "u8", (1, 3), "u8")
    g.box("blur_u8_7x1_const", imgs["u8c1"], "u8", (7, 1), "u8", border="constant")
    g.box("blur_u8_15x15", imgs["u8c1"], "u8", (15, 15), "u8")
    g.box("blur_u8_17x17_reflect", imgs["u8c1"], "u8", (17, 17), "u8", border="reflect")
    g.box("blur_u8_3x3_a00", imgs["u8c1"], "u8", (3, 3), "u8", anchor=(0, 0))
    g.box("blur_u8_5x3_a41", imgs["u8c1"], "u8", (5, 3), "u8", anchor=(4, 1))
    g.box("blur_u8c3_3x3", imgs["u8c3"], "u8", (3, 3), "u8")
    g.box("blur_u8c4_5x5", imgs["u8c4"], "u8", (5, 5), "u8")
    g.box("blur_u8c2_3x3", imgs["u8c2"], "u8", (3, 3), "u8")
    g.box("blur_u8_row1_5x5", imgs["row1"], "u8", (5, 5), "u8")
    g.box("blur_u8_col1_3x3_const", imgs["col1"], "u8", (3, 3), "u8", border="constant")
    g.box("blur_u8_tiny_5x5", imgs["tiny"], "u8", (5, 5), "u8")
    g.box("blur_u8_2x2", imgs["u8c1"], "u8", (2, 2), "u8")
    g.box("blur_u8_2x1", imgs["u8c1"], "u8", (2, 1), "u8")
    g.box("blur_u8_4x4_const", imgs["u8c1"], "u8", (4, 4), "u8", border="constant")
    g.box("blur_u8_16x16", imgs["u8c1"], "u8", (16, 16), "u8")
    g.box("blur_u8c3_6x2", imgs["u8c3"], "u8", (6, 2), "u8")
    g.box("box_u8_4x6_unnorm", imgs["u8c1b"], "u8", (4, 6), "u8", normalize=False)
    g.box("box_u8_2x2_unnorm", imgs["u8c1b"], "u8", (2, 2), "u8", normalize=False)
    g.box("box_u8s16_5x3_unnorm", imgs["u8c1"], "u8", (5, 3), "s16", normalize=False)
    g.box("box_u8s16_3x3_norm", imgs["u8c1"], "u8", (3, 3), "s16")
    g.box("box_u8s16_9x5_unnorm_refl", imgs["u8c1"], "u8", (9, 5), "s16",
          normalize=False, border="reflect")
    g.box("box_u8f32_3x5_unnorm", imgs["u8c1"], "u8", (3, 5), "f32", normalize=False)
    g.box("box_u8f32_6x2_unnorm", imgs["u8c1b"], "u8", (6, 2), "f32", normalize=False)
    g.box("box_f32_3x3", imgs["f32c1"], "f32", (3, 3), "f32")
    g.box("box_f32_5x5_repl", imgs["f32c1"], "f32", (5, 5), "f32", border="replicate")
    g.box("box_f32_7x9", imgs["f32c1"], "f32", (7, 9), "f32")
    g.box("box_f32_4x4_unnorm", imgs["f32c1"], "f32", (4, 4), "f32", normalize=False)
    g.box("box_f32_2x6_unnorm_const", imgs["f32c1"], "f32", (2, 6), "f32",
          normalize=False, border="constant")
    g.box("box_f32_3x3_a01", imgs["f32c1"], "f32", (3, 3), "f32", anchor=(0, 1))
    g.box("box_f32c3_5x5", imgs["f32c3"], "f32", (5, 5), "f32")

    # -------------------------------------------------------------- median
    g.median("median_u8_k3", imgs["u8c1"], "u8", 3)
    g.median("median_u8_k5", imgs["u8c1"], "u8", 5)
    g.median("median_u8_k7", imgs["u8c1"], "u8", 7)
    g.median("median_u8_k9", imgs["u8c1"], "u8", 9)
    g.median("median_u8_k15", imgs["u8c1"], "u8", 15)
    g.median("median_u8c3_k3", imgs["u8c3"], "u8", 3)
    g.median("median_u8c3_k5", imgs["u8c3"], "u8", 5)
    g.median("median_u8c3_k7", imgs["u8c3"], "u8", 7)
    g.median("median_u8c4_k3", imgs["u8c4"], "u8", 3)
    g.median("median_u8c4_k9", imgs["u8c4"], "u8", 9)
    g.median("median_u8c2_k5", imgs["u8c2"], "u8", 5)
    g.median("median_u8_row1_k3", imgs["row1"], "u8", 3)
    g.median("median_u8_row1_k5", imgs["row1"], "u8", 5)
    g.median("median_u8_col1_k5", imgs["col1"], "u8", 5)
    g.median("median_u8_row1c3_k3", imgs["row1c3"], "u8", 3)
    g.median("median_u8_tiny_k5", imgs["tiny"], "u8", 5)
    g.median("median_u8_small54_k7", imgs["small54"], "u8", 7)
    g.median("median_f32_k3", imgs["f32c1"], "f32", 3)
    g.median("median_f32_k5", imgs["f32c1"], "f32", 5)
    g.median("median_f32c3_k3", imgs["f32c3"], "f32", 3)

    # ------------------------------------------------------------- filter2D
    k33 = rng.uniform(-0.2, 0.4, (3, 3)).astype(np.float32)
    k55 = rng.uniform(-0.1, 0.15, (5, 5)).astype(np.float32)
    k17 = rng.uniform(-0.3, 0.3, (1, 7)).astype(np.float32)
    k71 = rng.uniform(-0.3, 0.3, (7, 1)).astype(np.float32)
    k44 = rng.uniform(-0.2, 0.2, (4, 4)).astype(np.float32)
    g.filter2d("filt2d_u8_k33", imgs["u8c1"], "u8", k33, "u8")
    g.filter2d("filt2d_u8_k33_delta", imgs["u8c1"], "u8", k33, "u8", delta=3.5)
    g.filter2d("filt2d_u8_k55_repl", imgs["u8c1"], "u8", k55, "u8", border="replicate")
    g.filter2d("filt2d_u8_k17", imgs["u8c1"], "u8", k17, "u8")
    g.filter2d("filt2d_u8_k71_const", imgs["u8c1"], "u8", k71, "u8", border="constant")
    g.filter2d("filt2d_u8_k44_a00", imgs["u8c1b"], "u8", k44, "u8", anchor=(0, 0))
    g.filter2d("filt2d_u8_k33_zero", imgs["u8c1"], "u8", np.zeros((3, 3), np.float32),
               "u8", delta=7.25)
    g.filter2d("filt2d_u8c3_k33", imgs["u8c3"], "u8", k33, "u8")
    g.filter2d("filt2d_u8s16_k33", imgs["u8c1"], "u8", k33, "s16", delta=-4.0)
    g.filter2d("filt2d_u8f32_k55", imgs["u8c1"], "u8", k55, "f32")
    g.filter2d("filt2d_f32_k33", imgs["f32c1"], "f32", k33, "f32")
    g.filter2d("filt2d_f32_k55_reflect", imgs["f32c1"], "f32", k55, "f32", border="reflect")
    g.filter2d("filt2d_f32c3_k33_delta", imgs["f32c3"], "f32", k33, "f32", delta=0.5)
    g.filter2d("filt2d_u8_tiny_k55", imgs["tiny"], "u8", k55, "u8")

    # ----------------------------------------------------------- sepFilter2D
    gk3 = cv2.getGaussianKernel(3, 0.8, cv2.CV_32F).ravel()
    gk5 = cv2.getGaussianKernel(5, 1.1, cv2.CV_32F).ravel()
    gk7 = cv2.getGaussianKernel(7, 0, cv2.CV_32F).ravel()
    # 8U->8U bit-exact (smooth symmetric, non-integer)
    g.sepfilter("sep_u8_gauss5", imgs["u8c1"], "u8", gk5, gk5, "u8")
    g.sepfilter("sep_u8_gauss3x7_repl", imgs["u8c1"], "u8", gk3, gk7, "u8", border="replicate")
    g.sepfilter("sep_u8_gauss7_const", imgs["u8c1b"], "u8", gk7, gk7, "u8", border="constant")
    g.sepfilter("sep_u8c3_gauss5", imgs["u8c3"], "u8", gk5, gk5, "u8")
    # 8U->8U float path (not smooth-symmetric)
    kasym = np.array([0.2, 0.5, 0.4], np.float32)
    g.sepfilter("sep_u8_general3", imgs["u8c1"], "u8", kasym, kasym, "u8")
    g.sepfilter("sep_u8_identity1", imgs["u8c1"], "u8", [1.0], gk5, "u8")
    # 8U->16S bit-exact (integer symmetric/antisymmetric)
    g.sepfilter("sep_u8s16_smooth_deriv", imgs["u8c1"], "u8", [1, 2, 1], [-1, 0, 1], "s16")
    g.sepfilter("sep_u8s16_binom5", imgs["u8c1"], "u8", [1, 4, 6, 4, 1], [1, -2, 1], "s16",
                delta=100.0)
    # 8U->16S float path (non-integer kernels)
    g.sepfilter("sep_u8s16_float", imgs["u8c1"], "u8", [0.25, 0.5, 0.25], [-0.5, 0.0, 0.5],
                "s16")
    # 8U->32F
    g.sepfilter("sep_u8f32_gauss5", imgs["u8c1"], "u8", gk5, gk3, "f32")
    g.sepfilter("sep_u8f32_deriv", imgs["u8c1b"], "u8", [-1, 0, 1], [1, 2, 1], "f32")
    # 32F->32F: small symmetric special branches need integer-valued images
    g.sepfilter("sep_f32_121", imgs["f32i1"], "f32", [1, 2, 1], [1, 2, 1], "f32")
    g.sepfilter("sep_f32_1m21", imgs["f32i1"], "f32", [1, -2, 1], [1, -2, 1], "f32")
    g.sepfilter("sep_f32_m101", imgs["f32i3"], "f32", [-1, 0, 1], [-1, 0, 1], "f32")
    g.sepfilter("sep_f32_k5special", imgs["f32i1"], "f32", [1, 0, -2, 0, 1], [1, 4, 6, 4, 1],
                "f32", border="replicate")
    # 32F->32F generic float kernels (random floats are fine)
    g.sepfilter("sep_f32_gauss5", imgs["f32c1"], "f32", gk5, gk5, "f32")
    g.sepfilter("sep_f32_gauss3", imgs["f32c1"], "f32", gk3, gk3, "f32")
    # delta on a ksize-3 float column would hit the one branch where OpenCV's
    # own lanes group delta differently from its scalar tail
    # (SymmColumnSmallVec_32f) — use a ksize-5 column, where the grouping
    # agrees, for the delta case.
    g.sepfilter("sep_f32_gauss7_delta", imgs["f32c1"], "f32", gk7, gk5, "f32", delta=1.5)
    g.sepfilter("sep_f32_asym5", imgs["f32c1"], "f32", [-0.3, 0.6, 0.0, 0.6, 0.3],
                [0.1, 0.2, 0.4, 0.2, 0.1], "f32")
    g.sepfilter("sep_f32_anchor", imgs["f32c1"], "f32", gk3, gk3, "f32", anchor=(0, 2))
    g.sepfilter("sep_f32_len9", imgs["f32c1"], "f32",
                cv2.getGaussianKernel(9, 2.0, cv2.CV_32F).ravel(), gk5, "f32")

    # ------------------------------------------------------------ sobel/scharr
    g.deriv("sobel_u8s16_dx1_k3", imgs["u8c1"], "u8", "sobel", 1, 0, "s16")
    g.deriv("sobel_u8s16_dy1_k3_repl", imgs["u8c1"], "u8", "sobel", 0, 1, "s16",
            border="replicate")
    g.deriv("sobel_u8s16_dx1dy1_k3", imgs["u8c1"], "u8", "sobel", 1, 1, "s16")
    g.deriv("sobel_u8s16_dx2_k3", imgs["u8c1"], "u8", "sobel", 2, 0, "s16")
    g.deriv("sobel_u8s16_dx1_k5_const", imgs["u8c1"], "u8", "sobel", 1, 0, "s16", ksize=5,
            border="constant")
    g.deriv("sobel_u8s16_dx2dy2_k7", imgs["u8c1"], "u8", "sobel", 2, 2, "s16", ksize=7)
    g.deriv("sobel_u8s16_dy2_k5", imgs["u8c3"], "u8", "sobel", 0, 2, "s16", ksize=5)
    g.deriv("sobel_u8s16_dx1_k1", imgs["u8c1"], "u8", "sobel", 1, 0, "s16", ksize=1)
    g.deriv("sobel_u8u8_dx1_k3", imgs["u8c1"], "u8", "sobel", 1, 0, "u8", delta=128.0)
    g.deriv("sobel_u8f32_dx1_k3", imgs["u8c1"], "u8", "sobel", 1, 0, "f32")
    g.deriv("sobel_u8s16_dx1_k3_scale2", imgs["u8c1"], "u8", "sobel", 1, 0, "s16", scale=2.0)
    g.deriv("sobel_u8s16_dx1_k3_scale17", imgs["u8c1"], "u8", "sobel", 1, 0, "s16",
            scale=1.7, delta=10.5)
    g.deriv("sobel_f32_dx1_k3", imgs["f32i1"], "f32", "sobel", 1, 0, "f32")
    g.deriv("sobel_f32_dy1_k5", imgs["f32i1"], "f32", "sobel", 0, 1, "f32", ksize=5)
    g.deriv("sobel_f32_dx2_k5_repl", imgs["f32i3"], "f32", "sobel", 2, 0, "f32", ksize=5,
            border="replicate")
    g.deriv("sobel_f32_dx1_k7", imgs["f32c1"], "f32", "sobel", 1, 0, "f32", ksize=7)
    g.deriv("scharr_u8s16_dx1", imgs["u8c1"], "u8", "scharr", 1, 0, "s16")
    g.deriv("scharr_u8s16_dy1_scale05", imgs["u8c1"], "u8", "scharr", 0, 1, "s16", scale=0.5)
    g.deriv("scharr_f32_dx1", imgs["f32i1"], "f32", "scharr", 1, 0, "f32")
    g.deriv("sobel_u8s16_scharr_via_km1", imgs["u8c1"], "u8", "sobel", 1, 0, "s16", ksize=-1)

    # ------------------------------------------------------------- laplacian
    g.laplacian("lap_u8_k1", imgs["u8c1"], "u8", "u8", ksize=1, delta=128.0)
    g.laplacian("lap_u8_k3", imgs["u8c1"], "u8", "u8", ksize=3, delta=128.0)
    g.laplacian("lap_u8s16_k1", imgs["u8c1"], "u8", "s16", ksize=1)
    g.laplacian("lap_u8s16_k3_repl", imgs["u8c1"], "u8", "s16", ksize=3, border="replicate")
    g.laplacian("lap_u8s16_k5", imgs["u8c1"], "u8", "s16", ksize=5)
    g.laplacian("lap_u8s16_k7", imgs["u8c1"], "u8", "s16", ksize=7)
    g.laplacian("lap_u8s16_k5_scale", imgs["u8c1"], "u8", "s16", ksize=5, scale=0.5,
                delta=20.25)
    g.laplacian("lap_u8f32_k3", imgs["u8c1"], "u8", "f32", ksize=3)
    g.laplacian("lap_u8f32_k5_const", imgs["u8c1"], "u8", "f32", ksize=5, border="constant")
    g.laplacian("lap_u8c3_k5", imgs["u8c3"], "u8", "u8", ksize=5, delta=100.0)
    g.laplacian("lap_f32_k1", imgs["f32c1"], "f32", "f32", ksize=1)
    g.laplacian("lap_f32_k3", imgs["f32c1"], "f32", "f32", ksize=3)
    # ksize 5 uses the d² kernel [1,0,-2,0,1], one of the reassociating
    # special row kernels — integer-valued float data keeps all sums exact.
    g.laplacian("lap_f32_k5", imgs["f32i1"], "f32", "f32", ksize=5)
    g.laplacian("lap_f32_k9_repl", imgs["f32c1"], "f32", "f32", ksize=9, border="replicate")
    g.laplacian("lap_f32_k5_scale", imgs["f32i1"], "f32", "f32", ksize=5, scale=2.0,
                delta=-1.5)

    # ------------------------------------------------- derivatives used by Canny
    g.deriv("sobel_u8s16_dx1_k7_scale1_16_repl", imgs["u8c1"], "u8", "sobel", 1, 0, "s16",
            ksize=7, scale=1 / 16, border="replicate")
    g.deriv("sobel_u8s16_dy1_k7_scale1_16_repl", imgs["u8c1"], "u8", "sobel", 0, 1, "s16",
            ksize=7, scale=1 / 16, border="replicate")
    g.deriv("sobel_u8s16_dx1_k5_repl", imgs["u8c1"], "u8", "sobel", 1, 0, "s16", ksize=5,
            border="replicate")
    g.deriv("sobel_u8s16_dy1_k5_repl", imgs["u8c1"], "u8", "sobel", 0, 1, "s16", ksize=5,
            border="replicate")
    g.deriv("sobel_u8c3s16_dx1_k5_repl", imgs["u8c3"], "u8", "sobel", 1, 0, "s16", ksize=5,
            border="replicate")
    g.deriv("scharr_u8s16_dx1_repl", imgs["u8c1"], "u8", "scharr", 1, 0, "s16", border="replicate")
    g.deriv("sobel_u8s16_dy1_km1_repl", imgs["u8c1"], "u8", "sobel", 0, 1, "s16", ksize=-1,
            border="replicate")
    dimg = depth_images()
    sat = cv2.Sobel(dimg["u8big3"], cv2.CV_16S, 1, 0, ksize=7, borderType=cv2.BORDER_REFLECT_101 | ISO)
    assert (np.abs(sat.astype(np.int32)) == 32767).sum() + (sat == -32768).sum() > 0, \
        "the unscaled 7x7 case must saturate"
    g.deriv("sobel_u8c3s16_dx1_k7_saturate", dimg["u8big3"], "u8", "sobel", 1, 0, "s16", ksize=7)

    # ------------------------------------- 16U/16S/64F sources, 16U/64F destinations
    g.deriv("sobel_u8u16_dx1_k3", imgs["u8c1"], "u8", "sobel", 1, 0, "u16")
    g.deriv("sobel_u8u16_dy1_k5_delta", imgs["u8c1"], "u8", "sobel", 0, 1, "u16", ksize=5,
            delta=1000.0)
    g.deriv("sobel_u8f64_dx1_k3", imgs["u8c1"], "u8", "sobel", 1, 0, "f64")
    g.deriv("sobel_u8f64_dx1dy1_k5_scale", imgs["u8c3"], "u8", "sobel", 1, 1, "f64", ksize=5,
            scale=0.1, delta=0.3, border="replicate")
    g.deriv("sobel_u16u8_dx1_k3", dimg["u16c1"], "u16", "sobel", 1, 0, "u8")
    g.deriv("sobel_u16u16_dy1_k3_delta", dimg["u16c1"], "u16", "sobel", 0, 1, "u16", delta=7.0)
    g.deriv("sobel_u16s16_dx1_k5", dimg["u16c1"], "u16", "sobel", 1, 0, "s16", ksize=5)
    g.deriv("sobel_u16f32_dx1_k3", dimg["u16c1"], "u16", "sobel", 1, 0, "f32")
    g.deriv("sobel_u16f32_dy1_k7_repl", dimg["u16c1"], "u16", "sobel", 0, 1, "f32", ksize=7,
            border="replicate")
    g.deriv("sobel_u16f64_dx1_k3_scale", dimg["u16c1"], "u16", "sobel", 1, 0, "f64",
            scale=1 / 3, delta=0.5)
    g.deriv("sobel_u16c3f32_dx2_k5_const", dimg["u16c3"], "u16", "sobel", 2, 0, "f32", ksize=5,
            border="constant")
    g.deriv("sobel_s16u8_dx1_k3", dimg["s16small"], "s16", "sobel", 1, 0, "u8")
    g.deriv("sobel_s16u16_dy1_k3", dimg["s16c1"], "s16", "sobel", 0, 1, "u16")
    g.deriv("sobel_s16s16_dx1_k3", dimg["s16c1"], "s16", "sobel", 1, 0, "s16")
    g.deriv("sobel_s16s16_dx1_k7_reflect", dimg["s16c1"], "s16", "sobel", 1, 0, "s16", ksize=7,
            border="reflect")
    g.deriv("sobel_s16f32_dy1_k5_reflect", dimg["s16c1"], "s16", "sobel", 0, 1, "f32", ksize=5,
            border="reflect")
    g.deriv("sobel_s16f64_dx1_k1", dimg["s16c1"], "s16", "sobel", 1, 0, "f64", ksize=1)
    g.deriv("sobel_s16c3s16_dx1dy1_k3_scale", dimg["s16c3"], "s16", "sobel", 1, 1, "s16",
            scale=0.25, delta=-3.0)
    g.deriv("sobel_f32u8_dx1_k3", imgs["f32i1"], "f32", "sobel", 1, 0, "u8", delta=128.0)
    g.deriv("sobel_f32u16_dy1_k5", imgs["f32i1"], "f32", "sobel", 0, 1, "u16", ksize=5)
    g.deriv("sobel_f32s16_dx1_k3", imgs["f32i3"], "f32", "sobel", 1, 0, "s16")
    g.deriv("sobel_f32f64_dx1_k5", imgs["f32c1"], "f32", "sobel", 1, 0, "f64", ksize=5)
    g.deriv("sobel_f64u8_dx1_k3", dimg["f64c1"], "f64", "sobel", 1, 0, "u8", delta=100.0)
    g.deriv("sobel_f64u16_dy1_k5", dimg["f64c1"], "f64", "sobel", 0, 1, "u16", ksize=5,
            delta=500.0)
    g.deriv("sobel_f64s16_dx1_k3", dimg["f64c1"], "f64", "sobel", 1, 0, "s16")
    g.deriv("sobel_f64_dx1_k3", dimg["f64c1"], "f64", "sobel", 1, 0, "f64")
    g.deriv("sobel_f64_dy2_k5_const", dimg["f64c1"], "f64", "sobel", 0, 2, "f64", ksize=5,
            border="constant")
    g.deriv("sobel_f64_dx1_k7_scale", dimg["f64c1"], "f64", "sobel", 1, 0, "f64", ksize=7,
            scale=0.3, delta=-1.25, border="replicate")
    g.deriv("sobel_f64c3_dx1dy1_k3", dimg["f64c3"], "f64", "sobel", 1, 1, "f64")
    g.deriv("scharr_u8f64_dx1", imgs["u8c1"], "u8", "scharr", 1, 0, "f64")
    g.deriv("scharr_u16f32_dy1", dimg["u16c1"], "u16", "scharr", 0, 1, "f32")
    g.deriv("scharr_s16s16_dx1_repl", dimg["s16c1"], "s16", "scharr", 1, 0, "s16",
            border="replicate")
    g.deriv("scharr_f64_dy1_scale", dimg["f64c1"], "f64", "scharr", 0, 1, "f64", scale=0.25)

    g.laplacian("lap_u8u16_k3_delta", imgs["u8c1"], "u8", "u16", ksize=3, delta=40.0)
    g.laplacian("lap_u8f64_k1_scale05", imgs["u8c1"], "u8", "f64", ksize=1, scale=0.5)
    g.laplacian("lap_u8f64_k1_scale01", imgs["u8c1"], "u8", "f64", ksize=1, scale=0.1)
    g.laplacian("lap_u16u16_k1", dimg["u16c1"], "u16", "u16", ksize=1)
    g.laplacian("lap_u16f32_k3", dimg["u16c1"], "u16", "f32", ksize=3)
    g.laplacian("lap_u16f64_k3", dimg["u16c3"], "u16", "f64", ksize=3)
    g.laplacian("lap_s16s16_k3", dimg["s16c1"], "s16", "s16", ksize=3)
    g.laplacian("lap_s16f32_k1", dimg["s16c1"], "s16", "f32", ksize=1)
    g.laplacian("lap_s16f64_k3_repl", dimg["s16c1"], "s16", "f64", ksize=3, border="replicate")
    g.laplacian("lap_f64_k1_scale2", dimg["f64c1"], "f64", "f64", ksize=1, scale=2.0, delta=0.5)
    g.laplacian("lap_f64_k3_scale01", dimg["f64c1"], "f64", "f64", ksize=3, scale=0.1)
    g.laplacian("lap_u8u16_k7", imgs["u8c1"], "u8", "u16", ksize=7)
    g.laplacian("lap_u8f64_k5", imgs["u8c1"], "u8", "f64", ksize=5, scale=0.1, delta=0.2)
    g.laplacian("lap_u16u8_k5", dimg["u16c1"], "u16", "u8", ksize=5, scale=1 / 64)
    g.laplacian("lap_u16f32_k5", dimg["u16c1"], "u16", "f32", ksize=5)
    g.laplacian("lap_s16s16_k5", dimg["s16small"], "s16", "s16", ksize=5)
    g.laplacian("lap_s16u16_k7_const", dimg["s16c1"], "s16", "u16", ksize=7, border="constant")
    g.laplacian("lap_f32u16_k5", imgs["f32i1"], "f32", "u16", ksize=5)
    g.laplacian("lap_f32f64_k5", imgs["f32i1"], "f32", "f64", ksize=5, scale=0.5)
    g.laplacian("lap_f64f32_k5", dimg["f64c1"], "f64", "f32", ksize=5)
    g.laplacian("lap_f64_k7_scale", dimg["f64c3"], "f64", "f64", ksize=7, scale=0.3, delta=1.0)

    # ---------------------------------------------------------------- sanity
    k = cv2.getGaussianKernel(5, 0, cv2.CV_64F).ravel()
    assert abs(k.sum() - 1.0) < 1e-15
    flat = cv2.GaussianBlur(imgs["uniform"], (5, 5), 0, borderType=cv2.BORDER_REFLECT_101 | ISO)
    assert (flat == 200).all()
    blurred = cv2.blur(imgs["uniform"], (3, 3), borderType=cv2.BORDER_REPLICATE | ISO)
    assert (blurred == 200).all()
    med = cv2.medianBlur(imgs["uniform"], 5)
    assert (med == 200).all()
    print("sanity checks passed")

    out_path = os.path.join(os.path.dirname(__file__), "..", "TestData", "filters_golden.json")
    out_path = os.path.abspath(out_path)
    with open(out_path, "w") as f:
        json.dump({"cases": g.cases}, f, indent=1)
    print(f"wrote {len(g.cases)} cases to {out_path}")
    print(f"file size: {os.path.getsize(out_path)} bytes")


if __name__ == "__main__":
    sys.exit(main())
