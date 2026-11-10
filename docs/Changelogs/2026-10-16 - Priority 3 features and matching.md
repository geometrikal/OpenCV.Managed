# Priority 3: feature detection, description and matching

**Date:** 2026-10-16 01:05:00

## Summary

The session ported the OpenCV 5.0 feature pipeline for image registration: FAST, ORB, SIFT,
brute-force matching and corner detectors. Every golden case matches cv2 5.0.0 bit for bit
(4,016 cases, no tolerances, no suspect cases). The full test suite passes (28,923 tests). An
optimisation pass made FAST with non-maximum suppression about 10× faster and ORB about 8×
faster, with identical output.

## Changes

### FAST and ORB (`Features/Fast`, `FastFeatureDetector`, `Orb`, new)

- FAST types 5_8, 7_12 and 9_16 with non-maximum suppression and mask.
- ORB detect, compute and detectAndCompute with all `ORB::create` parameters (WTA_K 2/3/4,
  Harris or FAST score, patch size, first level, mask).
- OpenCV blurs a pyramid level (a submatrix) with `sepFilter2D` and the float Gaussian kernel,
  not the fixed-point path. The port does the same.
- Keypoint order inside a level comes from `std::nth_element` in OpenCV. The tests compare
  `detectAndCompute` keypoints as sets; `compute` and FAST are compared in exact order.

### SIFT (`Features/Sift`, new)

- Detect, compute and detectAndCompute with mask; all `SIFT::create` parameters; float or byte
  descriptors.
- `Core/Internal/FastMath`: `cv::fastAtan2`, `hal::fastAtan32f`, `hal::magnitude32f` and the
  MSVC `hal::exp32f` (C runtime `expf`). Exactness holds against the Windows cv2 wheel; Linux
  and macOS builds use a table-based `exp`.

### Brute-force matching (`Features/BFMatcher`, new)

- `Match`, `KnnMatch`, `RadiusMatch` with masks, cross-check and a train collection; L1, L2,
  L2Sqr for byte and float; Hamming and Hamming2 for byte (`NormType.Hamming`/`Hamming2`).
- Float L2 uses the OpenCV SSE3 partial-sum order. Radius-match lists use the MSVC
  `std::sort` order (`Core/Internal/MsvcSort`), so tie order matches the Windows wheel.

### Corners (`ImgProc/Corners/Corners`, new)

- `CornerMinEigenVal`, `CornerHarris`, `CornerEigenValsAndVecs`, `GoodFeaturesToTrack` (with
  quality output), `CornerSubPix`; `Core/TermCriteria`.

### Performance

- FAST corner score: one `Vector256<short>` pass replaces a zero-initialised stack buffer and
  144 scalar min/max operations per corner. FAST 9_16 with NMS at 1024: 67 ms → 4.8 ms.
- `KeyPointsFilter.RetainBest`: quickselect for the boundary, then a sort of the kept points.
  48,000 keypoints to 200: 6.7 ms → 0.9 ms. The order is unchanged.
- ORB: pooled pyramid resize, FAST without the excluded border, descriptor pattern rotation in
  `Vector256<float>`. `ResizeLinearExact` has a single-channel 8-bit horizontal pass.
- `GoodFeaturesToTrack` thresholds in place.

### Benchmarks

- New suites `FeatureBenchmarks`, `SiftBenchmarks`, `MatcherBenchmarks`, `CornerBenchmarks`
  and their OpenCvSharp twins.
- Before optimisation, port ÷ OpenCvSharp 4.13 at 1024, 1 thread: FAST 9_16 NMS 6.4×, ORB
  detect 7.7×, ORB detectAndCompute 6.4×.
- After optimisation: see the results page and the table below. Other agents used the machine
  during the runs; treat single-case ratios below 2× as noise.
- Results page: https://claude.ai/artifact/D7rGGiR4dJnBS18FUKpqKP (groups "Features (FAST,
  ORB)", "SIFT", "Brute-force matching", "Corners").

After optimisation, port ÷ OpenCvSharp 4.13 (BenchmarkDotNet `--fast`):

| Case | 256, 1 thread | 1024, 1 thread | 256, 16 threads | 1024, 16 threads |
|---|---|---|---|---|
| FAST 9_16 NMS | 1.47 | 1.14 | 1.28 | 0.53 |
| FAST 7_12 NMS | 0.37 | 0.24 | 0.23 | 0.12 |
| ORB detect | 3.45 | 1.37 | 2.05 | 1.47 |
| ORB detectAndCompute | 3.86 | 1.56 | 2.60 | 1.32 |
| SIFT detectAndCompute | 1.88 | 1.01 | 2.17 | 0.85 |
| goodFeaturesToTrack | 1.64 | 1.74 | 1.82 | 1.31 |
| cornerHarris | 1.01 | 1.25 | 1.19 | 0.76 |
| Hamming match 5000 | – | 0.27 | – | 0.29 |
| L2 knnMatch 1000 | – | 0.76 | – | 0.84 |

Geometric mean, 1 thread / 16 threads: FAST and ORB 1.19 / 0.83, SIFT 1.38 / 1.36, matchers
0.36 / 0.51, corners 1.28 / 0.99. Four agents started builds during this run. An isolated
minimum-time measurement gave ORB detectAndCompute 2.34 ms at 256 and 24.3 ms at 1024 on one
thread, at or under OpenCvSharp.

## Open items

- SIFT allocates about 263 MB per call at 1024 (non-pooled `GaussianBlur` and `Subtract`
  results). Pooled or destination overloads of the shared filters would remove this.
- `GoodFeaturesToTrack` at 256 with 16 threads is slower than OpenCvSharp (large-object-heap
  allocations in `BoxFilter` and `Dilate`).
- ORB level blur through the generic `SepFilterEngine` costs about 6 ms at 1024.
- FAST without NMS grows a `List` and copies it (9 MB at 1024).
- FLANN is not ported. The port uses exhaustive matching.
