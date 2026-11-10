#!/usr/bin/env python3
"""cv2 reference timings for the managed OpenCV port benchmarks.

Runs the same cases as the BenchmarkDotNet project on identical inputs and
writes one JSON file per cv2 configuration to ``results/``.

Configurations (each runs in its own process, because OPENCV_CPU_DISABLE is
read when cv2 is imported):

* ``default``  - cv2 defaults: all threads, AVX2 dispatch, IPP.
* ``single``   - one thread; AVX2 dispatch and IPP stay enabled.
* ``baseline`` - one thread, no IPP, no OpenCL, AVX/AVX2/AVX512/FMA disabled
  (SSE baseline only). This is the closest cv2 configuration to the scalar,
  single-threaded managed port.

Usage (from the Geometrikal.OpenCV.Managed.Benchmark directory):
    python tools/bench_cv2.py                 # all configurations
    python tools/bench_cv2.py --mode single   # one configuration
    python tools/bench_cv2.py --checksums     # input parity check

Timing: each case is calibrated so one repeat takes at least ``--min-time``
seconds; the result is the median per-call time over ``--repeats`` repeats.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

MODES = ("default", "single", "baseline")
SIZES = (256, 1024)
SHAPE_COUNT = 64


def configure(mode):
    if mode == "baseline":
        os.environ["OPENCV_CPU_DISABLE"] = "AVX,AVX2,AVX512F,FMA3"
    import cv2
    if mode in ("single", "baseline"):
        cv2.setNumThreads(1)
    if mode == "baseline":
        cv2.ipp.setUseIPP(False)
    cv2.ocl.setUseOpenCL(False)
    return cv2


# ---------------------------------------------------------------------------
# Inputs. These mirror Inputs.cs exactly.
# ---------------------------------------------------------------------------

def hash_u32(x, y):
    import numpy as np
    h = (np.asarray(x, dtype=np.uint32) * np.uint32(73856093)) ^ (np.asarray(y, dtype=np.uint32) * np.uint32(19349663))
    h ^= h >> np.uint32(13)
    h = h * np.uint32(0x5BD1E995)
    h ^= h >> np.uint32(15)
    return h


def grid(size):
    import numpy as np
    ys, xs = np.mgrid[0:size, 0:size].astype(np.uint32)
    return xs, ys


def gray(size):
    import numpy as np
    xs, ys = grid(size)
    return (((xs ^ ys) + (hash_u32(xs, ys) & 63)) & 255).astype(np.uint8)


def color(size):
    import numpy as np
    xs, ys = grid(size)
    out = np.empty((size, size, 3), dtype=np.uint8)
    for c in range(3):
        h = hash_u32(xs, ys * 3 + c)
        out[:, :, c] = ((xs * (c + 1) + ys * (3 - c) + (h & 63)) & 255).astype(np.uint8)
    return out


def binary(size):
    import numpy as np
    xs, ys = grid(size)
    h = hash_u32(xs >> 4, ys >> 4)
    lx, ly = xs & 15, ys & 15
    on = h % 3 == 0
    hole = ((h >> 8) % 2 == 0) & (lx >= 5) & (lx <= 10) & (ly >= 5) & (ly <= 10)
    return np.where(on & ~hole, 255, 0).astype(np.uint8)


def random_point(index, salt, size):
    return (int(hash_u32(index, 2 * salt) % size), int(hash_u32(index, 2 * salt + 1) % size))


def sparse_zeros(size):
    import numpy as np
    xs, ys = grid(size)
    return np.where(hash_u32(xs + 11, ys) % 1000 == 0, 0, 255).astype(np.uint8)


def seeds(size, count):
    import numpy as np
    out = np.zeros((size, size), dtype=np.int32)
    for k in range(count):
        x, y = random_point(k, 3, size)
        out[y, x] = k + 1
    return out


def find_contours(cv2, img, mode, method):
    # cv2 4.x returns (contours, hierarchy); 3.x returned (image, contours, hierarchy).
    return cv2.findContours(img, mode, method)[-2:]


# ---------------------------------------------------------------------------
# Cases. Names match "<Class without 'Benchmarks'>.<Method>" in the C# project.
# ---------------------------------------------------------------------------

def build_cases(cv2, size):
    import numpy as np

    g = gray(size)
    c3 = color(size)
    border_dst = np.empty((size + 32, size + 32, 3), dtype=np.uint8)
    gf = g.astype(np.float32)
    bw = binary(size)
    sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    binom5 = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16
    rect3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ellipse5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    rect15 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    ellipse15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    box11 = np.full((11, 11), 1 / 128, dtype=np.float32)
    g2 = cv2.bitwise_not(g)
    gf2 = g2.astype(np.float32)
    planes = cv2.split(c3)
    hsv = cv2.cvtColor(c3, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(c3, cv2.COLOR_BGR2Lab)
    cf = c3.astype(np.float32) * np.float32(1 / 255)
    labf = cv2.cvtColor(cf, cv2.COLOR_BGR2Lab)
    down = (size * 2 // 3, size * 2 // 3)
    up = (size * 3 // 2, size * 3 // 2)
    affine = cv2.getRotationMatrix2D((size / 2, size / 2), 17, 0.9)
    perspective = np.vstack([affine, [1e-4, 5e-5, 1.0]])
    gy, gx = np.mgrid[0:size, 0:size].astype(np.float64)
    map_x = (gx * 0.93 + gy * 0.05 + 3.5).astype(np.float32)
    map_y = (gy * 0.97 - gx * 0.04 + 7.25).astype(np.float32)
    cc_labels = np.empty((size, size), dtype=np.int32)
    gd = g.astype(np.float64)
    cplx = cv2.merge([gf, gf2])
    ccs = cv2.dft(gf)
    ccs2 = cv2.dft(gf2)
    edge_dx = cv2.Sobel(g, cv2.CV_16S, 1, 0, borderType=cv2.BORDER_REPLICATE)
    edge_dy = cv2.Sobel(g, cv2.CV_16S, 0, 1, borderType=cv2.BORDER_REPLICATE)
    g16 = g.astype(np.uint16) * np.uint16(256)
    sz = sparse_zeros(size)
    ws_seeds = seeds(size, 200)
    dist8u = np.empty((size, size), dtype=np.uint8)
    dist_f = np.empty((size, size), dtype=np.float32)
    dist_labels = np.empty((size, size), dtype=np.int32)
    hist3d = cv2.calcHist([c3], [0, 1, 2], None, [32, 32, 32], [0, 256] * 3)
    hist3d_other = cv2.calcHist([cv2.bitwise_not(c3)], [0, 1, 2], None, [32, 32, 32], [0, 256] * 3)
    hist_hs = cv2.calcHist([c3], [0, 1], None, [30, 32], [0, 180, 0, 256])
    clahe = cv2.createCLAHE(40.0, (8, 8))
    integral_sum = np.empty((size + 1, size + 1), dtype=np.int32)
    integral_sq = np.empty((size + 1, size + 1), dtype=np.float64)

    cases = {
        "Filtering.GaussianBlur5x5_U8": lambda: cv2.GaussianBlur(g, (5, 5), 1.5),
        "Filtering.GaussianBlur5x5_U8C3": lambda: cv2.GaussianBlur(c3, (5, 5), 1.5),
        "Filtering.GaussianBlur5x5_F32": lambda: cv2.GaussianBlur(gf, (5, 5), 1.5),
        "Filtering.Blur5x5_U8": lambda: cv2.blur(g, (5, 5)),
        "Filtering.MedianBlur3_U8": lambda: cv2.medianBlur(g, 3),
        "Filtering.MedianBlur5_U8": lambda: cv2.medianBlur(g, 5),
        "Filtering.MedianBlur7_U8": lambda: cv2.medianBlur(g, 7),
        "Filtering.Filter2D3x3_U8": lambda: cv2.filter2D(g, cv2.CV_8U, sharpen),
        "Filtering.SepFilter2D5_U8F32": lambda: cv2.sepFilter2D(g, cv2.CV_32F, binom5, binom5),
        "Filtering.Sobel3Dx_U8S16": lambda: cv2.Sobel(g, cv2.CV_16S, 1, 0),
        "Filtering.Sobel3Dxy_F32": lambda: cv2.Sobel(gf, cv2.CV_32F, 1, 1),
        "Filtering.ScharrDx_U8S16": lambda: cv2.Scharr(g, cv2.CV_16S, 1, 0),
        "Filtering.Laplacian3_U8S16": lambda: cv2.Laplacian(g, cv2.CV_16S, ksize=3),
        "Filtering.GaussianBlur15x15_U8": lambda: cv2.GaussianBlur(g, (15, 15), 3.0),
        "Filtering.GaussianBlur15x15_F32": lambda: cv2.GaussianBlur(gf, (15, 15), 3.0),
        "Filtering.Blur15x15_U8": lambda: cv2.blur(g, (15, 15)),
        "Filtering.MedianBlur15_U8": lambda: cv2.medianBlur(g, 15),
        "Filtering.Filter2D11x11_U8": lambda: cv2.filter2D(g, cv2.CV_8U, box11),
        "Filtering.Sobel7Dx_U8S16": lambda: cv2.Sobel(g, cv2.CV_16S, 1, 0, ksize=7),
        "Filtering.CopyMakeBorder16_U8C3": lambda: cv2.copyMakeBorder(c3, 16, 16, 16, 16, cv2.BORDER_REFLECT_101),
        "Filtering.CopyMakeBorderInto16_U8C3": lambda: cv2.copyMakeBorder(c3, 16, 16, 16, 16, cv2.BORDER_REFLECT_101, dst=border_dst),

        "Morphology.ErodeRect3_U8": lambda: cv2.erode(g, rect3),
        "Morphology.ErodeRect3_U8C3": lambda: cv2.erode(c3, rect3),
        "Morphology.DilateEllipse5_U8": lambda: cv2.dilate(g, ellipse5),
        "Morphology.DilateRect3Iter3_U8": lambda: cv2.dilate(g, rect3, iterations=3),
        "Morphology.OpenEllipse5_U8": lambda: cv2.morphologyEx(g, cv2.MORPH_OPEN, ellipse5),
        "Morphology.GradientRect3_U8": lambda: cv2.morphologyEx(g, cv2.MORPH_GRADIENT, rect3),
        "Morphology.ErodeRect15_U8": lambda: cv2.erode(g, rect15),
        "Morphology.DilateEllipse15_U8": lambda: cv2.dilate(g, ellipse15),

        "Core.Add_U8": lambda: cv2.add(g, g2),
        "Core.Add_F32": lambda: cv2.add(gf, gf2),
        "Core.AbsDiff_U8": lambda: cv2.absdiff(g, g2),
        "Core.AddWeighted_U8": lambda: cv2.addWeighted(g, 0.3, g2, 0.7, 1),
        "Core.Multiply_U8": lambda: cv2.multiply(g, g2, scale=0.5),
        "Core.Compare_U8": lambda: cv2.compare(g, g2, cv2.CMP_GT),
        "Core.InRange_U8C3": lambda: cv2.inRange(c3, (10, 20, 30), (200, 210, 220)),
        "Core.ConvertScaleAbs_F32": lambda: cv2.convertScaleAbs(gf, alpha=0.5),
        "Core.Split_U8C3": lambda: cv2.split(c3),
        "Core.Merge_U8C3": lambda: cv2.merge(planes),
        "Core.Sum_U8": lambda: cv2.sumElems(g),
        "Core.MeanStdDev_U8": lambda: cv2.meanStdDev(g),
        "Core.MinMaxLoc_U8": lambda: cv2.minMaxLoc(g),
        "Core.CountNonZero_U8": lambda: cv2.countNonZero(g),
        "Core.NormL2_F32": lambda: cv2.norm(gf),

        "Color.BgrToGray_U8": lambda: cv2.cvtColor(c3, cv2.COLOR_BGR2GRAY),
        "Color.BgrToRgb_U8": lambda: cv2.cvtColor(c3, cv2.COLOR_BGR2RGB),
        "Color.BgrToBgra_U8": lambda: cv2.cvtColor(c3, cv2.COLOR_BGR2BGRA),
        "Color.BgrToHsv_U8": lambda: cv2.cvtColor(c3, cv2.COLOR_BGR2HSV),
        "Color.HsvToBgr_U8": lambda: cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR),
        "Color.BgrToLab_U8": lambda: cv2.cvtColor(c3, cv2.COLOR_BGR2Lab),
        "Color.LabToBgr_U8": lambda: cv2.cvtColor(lab, cv2.COLOR_Lab2BGR),
        "Color.BgrToYCrCb_U8": lambda: cv2.cvtColor(c3, cv2.COLOR_BGR2YCrCb),
        "Color.BgrToGray_F32": lambda: cv2.cvtColor(cf, cv2.COLOR_BGR2GRAY),
        "Color.BgrToHsv_F32": lambda: cv2.cvtColor(cf, cv2.COLOR_BGR2HSV),
        "Color.BgrToLab_F32": lambda: cv2.cvtColor(cf, cv2.COLOR_BGR2Lab),
        "Color.LabToBgr_F32": lambda: cv2.cvtColor(labf, cv2.COLOR_Lab2BGR),

        "Threshold.Binary_U8": lambda: cv2.threshold(g, 127, 255, cv2.THRESH_BINARY),
        "Threshold.Otsu_U8": lambda: cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU),
        "Threshold.Triangle_U8": lambda: cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE),
        "Threshold.Trunc_F32": lambda: cv2.threshold(gf, 100.5, 0, cv2.THRESH_TRUNC),
        "Threshold.AdaptiveMean11_U8": lambda: cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2),
        "Threshold.AdaptiveGaussian11_U8": lambda: cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),

        "Geometry.ResizeDownLinear_U8C3": lambda: cv2.resize(c3, down, interpolation=cv2.INTER_LINEAR),
        "Geometry.ResizeUpLinear_U8C3": lambda: cv2.resize(c3, up, interpolation=cv2.INTER_LINEAR),
        "Geometry.ResizeDownArea_U8C3": lambda: cv2.resize(c3, down, interpolation=cv2.INTER_AREA),
        "Geometry.ResizeUpArea_U8C3": lambda: cv2.resize(c3, up, interpolation=cv2.INTER_AREA),
        "Geometry.ResizeDownCubic_U8C3": lambda: cv2.resize(c3, down, interpolation=cv2.INTER_CUBIC),
        "Geometry.ResizeUpCubic_U8C3": lambda: cv2.resize(c3, up, interpolation=cv2.INTER_CUBIC),
        "Geometry.ResizeDownLinear_F32": lambda: cv2.resize(gf, down, interpolation=cv2.INTER_LINEAR),
        "Geometry.ResizeUpLinear_F32": lambda: cv2.resize(gf, up, interpolation=cv2.INTER_LINEAR),
        "Geometry.ResizeDownArea_F32": lambda: cv2.resize(gf, down, interpolation=cv2.INTER_AREA),
        "Geometry.ResizeUpArea_F32": lambda: cv2.resize(gf, up, interpolation=cv2.INTER_AREA),
        "Geometry.ResizeDownCubic_F32": lambda: cv2.resize(gf, down, interpolation=cv2.INTER_CUBIC),
        "Geometry.ResizeUpCubic_F32": lambda: cv2.resize(gf, up, interpolation=cv2.INTER_CUBIC),
        "Geometry.WarpAffineLinear_U8": lambda: cv2.warpAffine(g, affine, (size, size)),
        "Geometry.WarpPerspectiveLinear_U8": lambda: cv2.warpPerspective(g, perspective, (size, size)),
        "Geometry.RemapLinear_U8": lambda: cv2.remap(g, map_x, map_y, cv2.INTER_LINEAR),
        "Geometry.Flip_U8C3": lambda: cv2.flip(c3, -1),
        "Geometry.Rotate90_U8": lambda: cv2.rotate(g, cv2.ROTATE_90_CLOCKWISE),
        "Geometry.Transpose_U8": lambda: cv2.transpose(g),
        "Dft.RealCcs_F32": lambda: cv2.dft(gf),
        "Dft.RealCcs_F64": lambda: cv2.dft(gd),
        "Dft.RealComplexOutput_F32": lambda: cv2.dft(gf, flags=cv2.DFT_COMPLEX_OUTPUT),
        "Dft.Complex_F32": lambda: cv2.dft(cplx),
        "Dft.InverseCcsScale_F32": lambda: cv2.idft(ccs, flags=cv2.DFT_SCALE),
        "Dft.Rows_F32": lambda: cv2.dft(gf, flags=cv2.DFT_ROWS),
        "Dft.Dct_F32": lambda: cv2.dct(gf),
        "Dft.Dct_F64": lambda: cv2.dct(gd),
        "Dft.MulSpectrums_F32": lambda: cv2.mulSpectrums(ccs, ccs2, 0, conjB=True),

        "Histogram.CalcHist_U8": lambda: cv2.calcHist([g], [0], None, [256], [0, 256]),
        "Histogram.CalcHist3D_U8C3": lambda: cv2.calcHist([c3], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3),
        "Histogram.CalcHist_F32": lambda: cv2.calcHist([gf], [0], None, [256], [0, 256]),
        "Histogram.EqualizeHist_U8": lambda: cv2.equalizeHist(g),
        "Histogram.Clahe_U8": lambda: clahe.apply(g),
        "Histogram.Integral_U8S32": lambda: cv2.integral(g, sdepth=cv2.CV_32S),
        "Histogram.IntegralSqsum_U8S32F64": lambda: cv2.integral2(g, sdepth=cv2.CV_32S, sqdepth=cv2.CV_64F),
        "Histogram.Integral_U8S32_Into": lambda: cv2.integral(g, sum=integral_sum, sdepth=cv2.CV_32S),
        "Histogram.IntegralSqsum_U8S32F64_Into": lambda: cv2.integral2(g, sum=integral_sum, sqsum=integral_sq, sdepth=cv2.CV_32S, sqdepth=cv2.CV_64F),
        "Histogram.CompareHistChisqr_32x32x32": lambda: cv2.compareHist(hist3d, hist3d_other, cv2.HISTCMP_CHISQR),
        "Histogram.CalcBackProject2D_U8C3": lambda: cv2.calcBackProject([c3], [0, 1], hist_hs, [0, 180, 0, 256], 1.0),

        "Contour.FindContoursTreeNone": lambda: find_contours(cv2, bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE),
        "Contour.FindContoursExternalSimple": lambda: find_contours(cv2, bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE),
        "Contour.FindContoursListTc89L1": lambda: find_contours(cv2, bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1),

        "ConnectedComponents.Default8_S32": lambda: cv2.connectedComponentsWithAlgorithm(bw, 8, cv2.CV_32S, cv2.CCL_DEFAULT),
        "ConnectedComponents.Default8Into_S32": lambda: cv2.connectedComponentsWithAlgorithm(bw, 8, cv2.CV_32S, cv2.CCL_DEFAULT, labels=cc_labels),
        "ConnectedComponents.Wu8_S32": lambda: cv2.connectedComponentsWithAlgorithm(bw, 8, cv2.CV_32S, cv2.CCL_WU),
        "ConnectedComponents.Grana8_S32": lambda: cv2.connectedComponentsWithAlgorithm(bw, 8, cv2.CV_32S, cv2.CCL_GRANA),
        "ConnectedComponents.Wu4_S32": lambda: cv2.connectedComponentsWithAlgorithm(bw, 4, cv2.CV_32S, cv2.CCL_WU),
        "ConnectedComponents.Default8_U16": lambda: cv2.connectedComponentsWithAlgorithm(bw, 8, cv2.CV_16U, cv2.CCL_DEFAULT),
        "ConnectedComponents.StatsDefault8_S32": lambda: cv2.connectedComponentsWithStatsWithAlgorithm(bw, 8, cv2.CV_32S, cv2.CCL_DEFAULT),
        "ConnectedComponents.StatsWu8_S32": lambda: cv2.connectedComponentsWithStatsWithAlgorithm(bw, 8, cv2.CV_32S, cv2.CCL_WU),

        "Edge.Canny3_L1_U8": lambda: cv2.Canny(g, 100, 300),
        "Edge.Canny3_L2_U8": lambda: cv2.Canny(g, 100, 300, L2gradient=True),
        "Edge.Canny5_L1_U8": lambda: cv2.Canny(g, 1000, 3000, apertureSize=5),
        "Edge.Canny5_L2_U8": lambda: cv2.Canny(g, 1000, 3000, apertureSize=5, L2gradient=True),
        "Edge.Canny3_L1_U8C3": lambda: cv2.Canny(c3, 100, 300),
        "Edge.Canny5_L2_U8C3": lambda: cv2.Canny(c3, 1000, 3000, apertureSize=5, L2gradient=True),
        "Edge.Canny3_L1_U8Sparse": lambda: cv2.Canny(bw, 100, 300),
        "Edge.CannyDeriv_L2_S16": lambda: cv2.Canny(edge_dx, edge_dy, 100, 300, L2gradient=True),
        "Edge.SpatialGradient_U8": lambda: cv2.spatialGradient(g),
        "Edge.Sobel3Dx_U16F32": lambda: cv2.Sobel(g16, cv2.CV_32F, 1, 0),
        "Edge.Sobel3Dx_F64": lambda: cv2.Sobel(gd, cv2.CV_64F, 1, 0),
        # IPP serves L2Mask5, L2Precise and L1Byte in the default and single modes only.
        "Distance.L2Mask3": lambda: cv2.distanceTransform(sz, cv2.DIST_L2, 3),
        "Distance.L2Mask5": lambda: cv2.distanceTransform(sz, cv2.DIST_L2, 5),
        "Distance.L2Precise": lambda: cv2.distanceTransform(sz, cv2.DIST_L2, cv2.DIST_MASK_PRECISE),
        "Distance.L1Mask3": lambda: cv2.distanceTransform(sz, cv2.DIST_L1, 3),
        "Distance.L1Byte": lambda: cv2.distanceTransform(sz, cv2.DIST_L1, 3, dst=dist8u, dstType=cv2.CV_8U),
        "Distance.LabelsCcomp": lambda: cv2.distanceTransformWithLabels(sz, cv2.DIST_L2, 5, dst=dist_f, labels=dist_labels, labelType=cv2.DIST_LABEL_CCOMP),
        "Distance.LabelsPixel": lambda: cv2.distanceTransformWithLabels(sz, cv2.DIST_L2, 5, dst=dist_f, labels=dist_labels, labelType=cv2.DIST_LABEL_PIXEL),
        "Distance.Watershed": lambda: cv2.watershed(c3, ws_seeds.copy()),
    }

    # Advanced filtering. filter2D kernels of area >= 130 take the DFT path on both sides.
    # guidedFilter needs opencv-contrib (cv2.ximgproc) and stackBlur needs cv2 >= 4.7; the cases
    # are skipped when the installed cv2 lacks them.
    box15 = np.full((15, 15), 1 / 256, dtype=np.float32)
    box31 = np.full((31, 31), 1 / 1024, dtype=np.float32)
    cases.update({
        "AdvancedFilter.Filter2D15x15_U8": lambda: cv2.filter2D(g, cv2.CV_8U, box15),
        "AdvancedFilter.Filter2D15x15_F32": lambda: cv2.filter2D(gf, cv2.CV_32F, box15),
        "AdvancedFilter.Filter2D31x31_U8": lambda: cv2.filter2D(g, cv2.CV_8U, box31),
        "AdvancedFilter.Filter2D31x31_F32": lambda: cv2.filter2D(gf, cv2.CV_32F, box31),
        "AdvancedFilter.Bilateral9_U8": lambda: cv2.bilateralFilter(g, 9, 50, 5),
        "AdvancedFilter.Bilateral9_U8C3": lambda: cv2.bilateralFilter(c3, 9, 50, 5),
        "AdvancedFilter.Bilateral9_F32": lambda: cv2.bilateralFilter(gf, 9, 50, 5),
        "AdvancedFilter.SqrBoxFilter5x5_U8": lambda: cv2.sqrBoxFilter(g, -1, (5, 5)),
        "AdvancedFilter.SqrBoxFilter5x5_F32": lambda: cv2.sqrBoxFilter(gf, -1, (5, 5)),
    })
    if hasattr(cv2, "ximgproc"):
        guided_gray = cv2.ximgproc.createGuidedFilter(g, 8, 400)
        cases.update({
            "AdvancedFilter.GuidedFilter8_Gray": lambda: cv2.ximgproc.guidedFilter(g, g, 8, 400),
            "AdvancedFilter.GuidedFilter8_ColorGuide": lambda: cv2.ximgproc.guidedFilter(c3, g, 8, 400),
            "AdvancedFilter.GuidedFilter8_GrayPrepared": lambda: guided_gray.filter(g),
        })
    if hasattr(cv2, "stackBlur"):
        cases.update({
            "AdvancedFilter.StackBlur15x15_U8": lambda: cv2.stackBlur(g, (15, 15)),
            "AdvancedFilter.StackBlur15x15_U8C3": lambda: cv2.stackBlur(c3, (15, 15)),
            "AdvancedFilter.StackBlur15x15_F32": lambda: cv2.stackBlur(gf, (15, 15)),
        })

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    col = (255, 128, 0)
    p1 = [random_point(i, 0, size) for i in range(SHAPE_COUNT)]
    p2 = [random_point(i, 1, size) for i in range(SHAPE_COUNT)]
    pairs = list(zip(p1, p2))
    polygons = [np.array([random_point(p * 8 + v, 2, size) for v in range(8)], dtype=np.int32) for p in range(16)]
    contours, hierarchy = find_contours(cv2, bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    radius = size // 16
    axes = (size // 8, size // 16)

    def lines(thickness):
        for a, b in pairs:
            cv2.line(canvas, a, b, col, thickness)

    def rectangles():
        for a, b in pairs:
            cv2.rectangle(canvas, a, b, col, -1)

    def circles(thickness):
        for a in p1:
            cv2.circle(canvas, a, radius, col, thickness)

    def ellipses():
        for a in p1:
            cv2.ellipse(canvas, a, axes, 30, 0, 360, col, 2)

    cases.update({
        "Drawing.Line1px": lambda: lines(1),
        "Drawing.Line5px": lambda: lines(5),
        "Drawing.RectangleFilled": rectangles,
        "Drawing.CircleOutline": lambda: circles(2),
        "Drawing.CircleFilled": lambda: circles(-1),
        "Drawing.EllipseOutline": ellipses,
        "Drawing.FillPoly": lambda: cv2.fillPoly(canvas, polygons, col),
        "Drawing.Polylines": lambda: cv2.polylines(canvas, polygons, True, col),
        "Drawing.DrawContours": lambda: cv2.drawContours(canvas, contours, -1, col, 1, cv2.LINE_8, hierarchy),
    })
    # Shape descriptors over the List/None contours of binary(size).
    shape_contours, _ = find_contours(cv2, bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    sparse_mask = np.zeros((size, size), np.uint8)
    for k in range(3):
        x, y = random_point(k, 7, size)
        sparse_mask[y, x] = 255

    def each_contour(fn):
        return lambda: [fn(c) for c in shape_contours]

    cases.update({
        "ShapeDescriptor.ApproxPolyDPAll": each_contour(lambda c: cv2.approxPolyDP(c, 2.0, True)),
        "ShapeDescriptor.ArcLengthAll": each_contour(lambda c: cv2.arcLength(c, True)),
        "ShapeDescriptor.ContourAreaAll": each_contour(cv2.contourArea),
        "ShapeDescriptor.BoundingRectAll": each_contour(cv2.boundingRect),
        "ShapeDescriptor.MomentsContourAll": each_contour(cv2.moments),
        "ShapeDescriptor.MatchShapesI1All": lambda: [
            cv2.matchShapes(shape_contours[i - 1], shape_contours[i], cv2.CONTOURS_MATCH_I1, 0)
            for i in range(1, len(shape_contours))],
        "ShapeDescriptor.MomentsImage_U8": lambda: cv2.moments(g),
        "ShapeDescriptor.MomentsImageBinary_U8": lambda: cv2.moments(bw, True),
        "ShapeDescriptor.MomentsImage_F32": lambda: cv2.moments(gf),
        "ShapeDescriptor.BoundingRectMask_U8": lambda: cv2.boundingRect(bw),
        "ShapeDescriptor.BoundingRectMaskSparse_U8": lambda: cv2.boundingRect(sparse_mask),
    })

    # Flood fill. Hole fills start at the first zero pixel of row 0 of binary(size); gradient fills at the centre of gray(size).
    ff_work = np.empty_like(bw)
    ff_mask = np.empty((size + 2, size + 2), dtype=np.uint8)
    ff_seed = (int(np.argmax(bw[0] == 0)), 0)
    ff_centre = (size // 2, size // 2)

    def ff_hole():
        np.copyto(ff_work, bw)
        return cv2.floodFill(ff_work, None, ff_seed, 128)[0]

    def ff_hole_mask():
        np.copyto(ff_work, bw)
        ff_mask.fill(0)
        return cv2.floodFill(ff_work, ff_mask, ff_seed, 255, 0, 0, 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))[0]

    def ff_grad():
        np.copyto(ff_work, g)
        return cv2.floodFill(ff_work, None, ff_centre, 255, (12,) * 4, (12,) * 4, 8)[0]

    def ff_fixed():
        np.copyto(ff_work, g)
        return cv2.floodFill(ff_work, None, ff_centre, 255, (48,) * 4, (48,) * 4, 8 | cv2.FLOODFILL_FIXED_RANGE)[0]

    cases.update({
        "FloodFill.HoleFill": ff_hole,
        "FloodFill.HoleFillMask": ff_hole_mask,
        "FloodFill.GradientGray": ff_grad,
        "FloodFill.FixedRangeGray": ff_fixed,
    })
    # Shape fitting over the List/None contours of binary(size).
    sf_contours = shape_contours
    sf_ellipse = [c for c in sf_contours if len(c) >= 5]
    sf_defects = []
    for c in sf_contours:
        h = cv2.convexHull(c, returnPoints=False)
        try:
            cv2.convexityDefects(c, h)
        except cv2.error:
            continue
        sf_defects.append((c, h))
    sf_pts = [(float(c[0][0][0]) + 1.5, float(c[0][0][1]) + 1.5) for c in sf_contours]

    cases.update({
        "ShapeFitting.ConvexHull": lambda: [cv2.convexHull(c) for c in sf_contours],
        "ShapeFitting.ConvexHullIndices": lambda: [cv2.convexHull(c, returnPoints=False) for c in sf_contours],
        "ShapeFitting.ConvexityDefects": lambda: [cv2.convexityDefects(c, h) for c, h in sf_defects],
        "ShapeFitting.IsContourConvex": lambda: [cv2.isContourConvex(c) for c in sf_contours],
        "ShapeFitting.MinAreaRect": lambda: [cv2.minAreaRect(c) for c in sf_contours],
        "ShapeFitting.MinEnclosingCircle": lambda: [cv2.minEnclosingCircle(c) for c in sf_contours],
        "ShapeFitting.FitEllipse": lambda: [cv2.fitEllipse(c) for c in sf_ellipse],
        "ShapeFitting.FitEllipseAms": lambda: [cv2.fitEllipseAMS(c) for c in sf_ellipse],
        "ShapeFitting.FitEllipseDirect": lambda: [cv2.fitEllipseDirect(c) for c in sf_ellipse],
        "ShapeFitting.FitLineL2": lambda: [cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01) for c in sf_contours],
        "ShapeFitting.FitLineHuber": lambda: [cv2.fitLine(c, cv2.DIST_HUBER, 0, 0.01, 0.01) for c in sf_contours],
        "ShapeFitting.PointPolygonTest": lambda: [cv2.pointPolygonTest(c, p, False) for c, p in zip(sf_contours, sf_pts)],
        "ShapeFitting.PointPolygonTestDistance": lambda: [cv2.pointPolygonTest(c, p, True) for c, p in zip(sf_contours, sf_pts)],
    })
    # Pyramids. cv2 5.0.0 has no buildPyramid binding; four pyrDown calls are the non-IPP equivalent.
    def build4(a):
        out = [a]
        for _ in range(4):
            out.append(cv2.pyrDown(out[-1]))
        return out

    cases.update({
        "Pyramid.PyrDown_U8": lambda: cv2.pyrDown(g),
        "Pyramid.PyrDown_U8C3": lambda: cv2.pyrDown(c3),
        "Pyramid.PyrDown_F32": lambda: cv2.pyrDown(gf),
        "Pyramid.PyrUp_U8": lambda: cv2.pyrUp(g),
        "Pyramid.PyrUp_U8C3": lambda: cv2.pyrUp(c3),
        "Pyramid.PyrUp_F32": lambda: cv2.pyrUp(gf),
        "Pyramid.BuildPyramid4_U8": lambda: build4(g),
        "Pyramid.BuildPyramid4_U8C3": lambda: build4(c3),
    })

    # Linear algebra.
    def solver_matrix(rows, cols):
        i, j = np.mgrid[0:rows, 0:cols]
        return (((i * 7 + j * 3) % 11) - 5 + np.where(i == j, 10, 0)).astype(np.float64)

    la_a3, la_b3 = solver_matrix(3, 3), solver_matrix(3, 1)
    la_a9, la_b9 = solver_matrix(9, 9), solver_matrix(9, 1)
    la_a8x9 = solver_matrix(8, 9)
    la_idx = np.arange(100000)
    la_pts = np.stack([(la_idx % 317) * 1.5, (la_idx // 317) * 0.75], axis=1).astype(np.float32).reshape(-1, 1, 2)
    la_h = np.array([[0.9, 0.05, 3], [-0.04, 0.97, 7], [1e-4, 5e-5, 1]], np.float64)
    cases.update({
        "LinearAlgebra.Gemm_F32": lambda: cv2.gemm(gf, gf2, 1, None, 0),
        "LinearAlgebra.Solve3x3_F64": lambda: cv2.solve(la_a3, la_b3, flags=cv2.DECOMP_LU),
        "LinearAlgebra.Solve9x9_F64": lambda: cv2.solve(la_a9, la_b9, flags=cv2.DECOMP_LU),
        "LinearAlgebra.Svd8x9_F64": lambda: cv2.SVDecomp(la_a8x9, flags=cv2.SVD_FULL_UV),
        "LinearAlgebra.PerspectiveTransform100k_F32": lambda: cv2.perspectiveTransform(la_pts, la_h),
    })
    # P2-END
    # Features: FAST, ORB, SIFT and corners on gray(size).
    fast9 = cv2.FastFeatureDetector_create(20, True, cv2.FastFeatureDetector_TYPE_9_16)
    fast9_all = cv2.FastFeatureDetector_create(20, False, cv2.FastFeatureDetector_TYPE_9_16)
    fast7 = cv2.FastFeatureDetector_create(20, True, cv2.FastFeatureDetector_TYPE_7_12)
    orb = cv2.ORB_create()
    sift = cv2.SIFT_create()
    cases.update({
        "Feature.Fast9_16_Nms": lambda: fast9.detect(g),
        "Feature.Fast9_16_All": lambda: fast9_all.detect(g),
        "Feature.Fast7_12_Nms": lambda: fast7.detect(g),
        "Feature.OrbDetect": lambda: orb.detect(g),
        "Feature.OrbDetectAndCompute": lambda: orb.detectAndCompute(g, None),
        "Sift.DetectAndCompute_U8": lambda: sift.detectAndCompute(g, None),
        "Corner.CornerHarris_U8": lambda: cv2.cornerHarris(g, 2, 3, 0.04),
        "Corner.CornerMinEigenVal_U8": lambda: cv2.cornerMinEigenVal(g, 3, ksize=3),
        "Corner.GoodFeaturesToTrack_U8": lambda: cv2.goodFeaturesToTrack(g, 1000, 0.01, 10),
    })

    # Brute-force matching. The cases have no size parameter; they run once.
    def matcher_bytes(rows, cols, salt):
        ys, xs = np.mgrid[0:rows, 0:cols].astype(np.uint32)
        return (hash_u32(xs, ys + np.uint32(salt)) & 255).astype(np.uint8)

    def matcher_floats(rows, cols, salt):
        ys, xs = np.mgrid[0:rows, 0:cols].astype(np.uint32)
        return ((hash_u32(xs, ys + np.uint32(salt)) & 0xFFFF).astype(np.float32) / np.float32(256) - np.float32(128))

    if size == SIZES[0]:
        q1, t1 = matcher_bytes(1000, 32, 0), matcher_bytes(1000, 32, 1_000_000)
        q5, t5 = matcher_bytes(5000, 32, 0), matcher_bytes(5000, 32, 1_000_000)
        qf, tf = matcher_floats(1000, 128, 0), matcher_floats(1000, 128, 1_000_000)
        bf_h, bf_l2 = cv2.BFMatcher(cv2.NORM_HAMMING), cv2.BFMatcher(cv2.NORM_L2)
        cases.update({
            "Matcher.Hamming_Match_1000": lambda: bf_h.match(q1, t1),
            "Matcher.Hamming_Match_5000": lambda: bf_h.match(q5, t5),
            "Matcher.Hamming_Knn2_1000": lambda: bf_h.knnMatch(q1, t1, 2),
            "Matcher.Hamming_Knn2_5000": lambda: bf_h.knnMatch(q5, t5, 2),
            "Matcher.L2_Knn2_1000": lambda: bf_l2.knnMatch(qf, tf, 2),
        })
    # P3-END
    # Undistortion with a scaled drone camera.
    ucam = np.array([[size * 0.66, 0, (size - 1) * 0.5 + 2.5], [0, size * 0.665, (size - 1) * 0.5 - 1.5], [0, 0, 1]], np.float64)
    udist = np.array([-0.2713, 0.1125, 0.00071, -0.00118, -0.0315], np.float64)
    uidx = np.arange(10000, dtype=np.uint32)
    upts = np.stack([(hash_u32(uidx, 7) % np.uint32(size * 16)).astype(np.float32) / np.float32(16),
                     (hash_u32(uidx, 11) % np.uint32(size * 16)).astype(np.float32) / np.float32(16)], axis=1).reshape(-1, 1, 2)
    cases.update({
        "Undistort.InitUndistortRectifyMap_32FC1": lambda: cv2.initUndistortRectifyMap(ucam, udist, None, ucam, (size, size), cv2.CV_32FC1),
        "Undistort.InitUndistortRectifyMap_16SC2": lambda: cv2.initUndistortRectifyMap(ucam, udist, None, ucam, (size, size), cv2.CV_16SC2),
        "Undistort.Undistort_U8C3": lambda: cv2.undistort(c3, ucam, udist),
        "Undistort.UndistortPoints10k": lambda: cv2.undistortPoints(upts, ucam, udist),
    })

    # Alignment: templates cut from gray(size) at (size // 3, size // 4); see AlignmentBenchmarks.cs.
    t16 = g[size // 4:size // 4 + 16, size // 3:size // 3 + 16].copy()
    t64 = g[size // 4:size // 4 + 64, size // 3:size // 3 + 64].copy()
    pc_b = np.roll(gf, (7, 5), axis=(0, 1))
    ecc_t = cv2.GaussianBlur(gf, (0, 0), 3.0)
    ecc_i = np.roll(ecc_t, (1, 2), axis=(0, 1))
    ecc_crit = (cv2.TERM_CRITERIA_COUNT, 10, 0)
    cases.update({
        "Alignment.MatchTemplate_CCoeffNormed_U8_T16": lambda: cv2.matchTemplate(g, t16, cv2.TM_CCOEFF_NORMED),
        "Alignment.MatchTemplate_CCoeffNormed_U8_T64": lambda: cv2.matchTemplate(g, t64, cv2.TM_CCOEFF_NORMED),
        "Alignment.PhaseCorrelate_F32": lambda: cv2.phaseCorrelate(gf, pc_b),
        "Alignment.FindTransformEcc_Euclidean_F32": lambda: cv2.findTransformECC(
            ecc_t, ecc_i, np.eye(2, 3, dtype=np.float32), cv2.MOTION_EUCLIDEAN, ecc_crit, None, 5),
    })
    # EstimationBenchmarks is parameterised by point count, not image size; it has no cv2 cases.
    # P4-END
    # Stitching (StitchingBenchmarks); size is the tile size, four tiles overlap by 30 %.
    st_k = np.array([[size, 0, size / 2], [0, size, size / 2], [0, 0, 1]], np.float32)
    st_r = np.array([[np.cos(0.15), 0, np.sin(0.15)], [0, 1, 0], [-np.sin(0.15), 0, np.cos(0.15)]], np.float32)
    st_step = size * 7 // 10
    st_corners = [(0, 0), (st_step, 0), (0, st_step), (st_step, st_step)]
    st_tiles = [c3.copy() for _ in range(4)]
    st_tiles16 = [t.astype(np.int16) for t in st_tiles]
    st_masks = [np.full((size, size), 255, np.uint8) for _ in range(4)]
    st_roi = cv2.detail.resultRoi(st_corners, [(size, size)] * 4)

    def st_blend(blender, images):
        blender.prepare(st_roi)
        for img, mask, corner in zip(images, st_masks, st_corners):
            blender.feed(img, mask, corner)
        return blender.blend(None, None)

    def st_gain():
        comp = cv2.detail_GainCompensator(1)
        comp.feed(st_corners, st_tiles, st_masks)
        return comp.getMatGains()

    cases.update({
        "Stitching.PlaneWarper_Warp_U8C3": lambda: cv2.PyRotationWarper("plane", float(size)).warp(c3, st_k, st_r, cv2.INTER_LINEAR, cv2.BORDER_REFLECT),
        "Stitching.SphericalWarper_BuildMaps": lambda: cv2.PyRotationWarper("spherical", float(size)).buildMaps((size, size), st_k, st_r),
        "Stitching.FeatherBlender_FeedBlend_4Tiles": lambda: st_blend(cv2.detail_FeatherBlender(), st_tiles16),
        "Stitching.MultiBandBlender5_FeedBlend_4Tiles": lambda: st_blend(cv2.detail_MultiBandBlender(0, 5), st_tiles),
        "Stitching.GainCompensator_Feed_4Tiles": st_gain,
    })

    # Seam finders: four float tiles in a 2x2 grid with a step of 3/4 tile, cut from
    # color(size + step); tile i is scaled by 1 + i / 16. Each call starts from full masks.
    seam_step = size * 3 // 4
    seam_world = color(size + seam_step)
    seam_corners = [(0, 0), (seam_step, 0), (0, seam_step), (seam_step, seam_step)]
    seam_tiles = [seam_world[y:y + size, x:x + size].astype(np.float32) * np.float32(1 + i / 16)
                  for i, (x, y) in enumerate(seam_corners)]
    seam_voronoi = cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_VORONOI_SEAM)
    seam_dp = cv2.detail_DpSeamFinder("COLOR_GRAD")
    seam_gc = cv2.detail_GraphCutSeamFinder("COST_COLOR")

    def seam_masks():
        return [np.full((size, size), 255, np.uint8) for _ in range(4)]

    cases.update({
        "Seam.Voronoi_F32C3": lambda: seam_voronoi.find(seam_tiles, seam_corners, seam_masks()),
        "Seam.DpColorGrad_F32C3": lambda: seam_dp.find(seam_tiles, seam_corners, seam_masks()),
        "Seam.GraphCutColor_F32C3": lambda: seam_gc.find(seam_tiles, seam_corners, seam_masks()),
    })

    # Stitching registration: 6 tiles (3 x 2, step 0.6*size) of blurred hash noise; ORB features once.
    def stitch_tiles(n):
        step = n * 6 // 10
        m = n + 2 * step
        ys, xs = np.mgrid[0:m, 0:m].astype(np.uint32)
        src = cv2.GaussianBlur((hash_u32(xs, ys) & 255).astype(np.uint8), (5, 5), 1.5)
        return [np.ascontiguousarray(src[r * step:r * step + n, c * step:c * step + n]) for r in range(2) for c in range(3)]

    orb_st = cv2.ORB_create()
    st_feats = [cv2.detail.computeImageFeatures2(orb_st, t) for t in stitch_tiles(size)]
    st_pw = cv2.detail_BestOf2NearestMatcher(False).apply2(st_feats)

    def st_hom_ray():
        ok, cams = cv2.detail_HomographyBasedEstimator().apply(st_feats, st_pw, None)
        for c in cams:
            c.R = c.R.astype(np.float32)
        return cv2.detail_BundleAdjusterRay().apply(st_feats, st_pw, cams)

    def st_aff():
        pw = cv2.detail_AffineBestOf2NearestMatcher(False, False).apply2(st_feats)
        ok, cams = cv2.detail_AffineBasedEstimator().apply(st_feats, pw, None)
        for c in cams:
            c.R = c.R.astype(np.float32)
        return cv2.detail_BundleAdjusterAffinePartial().apply(st_feats, pw, cams)

    cases.update({
        "StitchRegistration.BestOf2NearestMatcher_Match": lambda: cv2.detail_BestOf2NearestMatcher(False).apply2(st_feats),
        "StitchRegistration.HomographyEstimator_BundleAdjusterRay": st_hom_ray,
        "StitchRegistration.AffineMatcher_BundleAdjusterAffinePartial": st_aff,
    })
    # P5-END
    # Sparse optical flow: next(y, x) = prev(clamp(y - 2), clamp(x - 3)); 500 grid points.
    lk_ys = np.clip(np.arange(size) - 2, 0, size - 1)
    lk_xs = np.clip(np.arange(size) - 3, 0, size - 1)
    lk_next = np.ascontiguousarray(g[lk_ys][:, lk_xs])
    lk_pts = np.array([[size * (0.1 + 0.8 * j / 24.0), size * (0.1 + 0.8 * i / 19.0)]
                       for i in range(20) for j in range(25)], np.float32).reshape(-1, 1, 2)
    lk_crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 0.01)
    cases.update({
        "OpticalFlow.CalcOpticalFlowPyrLK_500": lambda: cv2.calcOpticalFlowPyrLK(
            g, lk_next, lk_pts, None, winSize=(21, 21), maxLevel=3, criteria=lk_crit),
        "OpticalFlow.BuildOpticalFlowPyramid_U8": lambda: cv2.buildOpticalFlowPyramid(g, (21, 21), 3),
    })

    # Anti-aliased drawing on the Drawing canvas and shapes.
    def lines_aa(thickness):
        for a, b in pairs:
            cv2.line(canvas, a, b, col, thickness, cv2.LINE_AA)

    def circles_aa():
        for a in p1:
            cv2.circle(canvas, a, radius, col, 2, cv2.LINE_AA)

    cases.update({
        "DrawingAa.LineAa1px": lambda: lines_aa(1),
        "DrawingAa.LineAa5px": lambda: lines_aa(5),
        "DrawingAa.CircleAaOutline": circles_aa,
        "DrawingAa.FillPolyAa": lambda: cv2.fillPoly(canvas, polygons, col, cv2.LINE_AA),
    })
    # PnpBenchmarks and EpipolarBenchmarks use fixed point counts and have no cv2 cases;
    # see their port notes for equivalent cv2 snippets.
    # P6-END
    return cases


def measure(fn, repeats, min_time):
    for _ in range(3):
        fn()
    number = 1
    while True:
        start = time.perf_counter()
        for _ in range(number):
            fn()
        elapsed = time.perf_counter() - start
        if elapsed >= min_time:
            break
        number = max(number * 2, int(number * min_time / max(elapsed, 1e-9) * 1.2))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(number):
            fn()
        samples.append((time.perf_counter() - start) / number)
    return statistics.median(samples), number


def run_mode(mode, out_dir, repeats, min_time, filter_text):
    cv2 = configure(mode)
    results = []
    for size in SIZES:
        for name, fn in build_cases(cv2, size).items():
            if filter_text and not any(f.strip().lower() in name.lower() for f in filter_text.split(",")):
                continue
            median, number = measure(fn, repeats, min_time)
            results.append({"case": name, "size": size, "median_ns": median * 1e9, "calls_per_repeat": number})
            print(f"[{mode:8}] {name:40} {size:5}  {median * 1e6:12.2f} us", flush=True)
    payload = {
        "mode": mode,
        "cv2": cv2.__version__,
        "threads": cv2.getNumThreads(),
        "ipp": bool(cv2.ipp.useIPP()),
        "cpu_disable": os.environ.get("OPENCV_CPU_DISABLE", ""),
        "python": platform.python_version(),
        "results": results,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"cv2-{mode}.json")
    if filter_text and os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        fresh = {(r["case"], r["size"]) for r in results}
        payload["results"] = [r for r in old["results"] if (r["case"], r["size"]) not in fresh] + results
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)


def print_checksums():
    cv2 = configure("default")
    for size in SIZES:
        bw = binary(size)
        contours, _ = find_contours(cv2, bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(f"size={size} gray={int(gray(size).sum(dtype='int64'))} "
              f"color={int(color(size).sum(dtype='int64'))} "
              f"binary={int((bw != 0).sum())} contours={len(contours)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--min-time", type=float, default=0.05)
    parser.add_argument("--filter", default="")
    parser.add_argument("--checksums", action="store_true")
    args = parser.parse_args()

    if args.checksums:
        print_checksums()
    elif args.mode:
        run_mode(args.mode, args.out, args.repeats, args.min_time, args.filter)
    else:
        for mode in MODES:
            subprocess.run([sys.executable, os.path.abspath(__file__), "--mode", mode, "--out", args.out,
                            "--repeats", str(args.repeats), "--min-time", str(args.min_time),
                            "--filter", args.filter], check=True)


if __name__ == "__main__":
    main()
