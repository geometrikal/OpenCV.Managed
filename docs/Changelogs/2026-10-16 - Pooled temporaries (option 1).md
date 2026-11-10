# Pooled temporaries (option 1)

**Date:** 2026-10-16 12:30:00

## Summary

Hot functions of the feature, alignment, calibration and stitching paths no longer allocate
per-call temporary buffers. They rent them with the internal `PooledMat<T>` or
`ArrayPool<T>.Shared` and return them before the call ends. Returned results are still new
allocations. The public API does not change. Output is bit-exact (35,876 tests pass).

## Changes

### Features
- **SIFT:** the base image, the Gaussian pyramid and the difference-of-Gaussian pyramid are
  pooled. New internal `Filters.GaussianBlurInto` (float). Keypoints are collected per parallel
  stripe; the order does not depend on the thread count. Orientation and descriptor sample buffers
  are pooled.
- **ORB:** the pyramid, the mask pyramid and the blurred levels are pooled. Keypoint filters run
  on pooled spans (new internal span forms in `KeyPointsFilter`); public `List` methods keep the
  same kept set and order.
- **FAST:** stripes write to a pooled growable buffer; the result is copied once.
- **Optical flow:** LK tracker window buffers, Scharr row buffers and the border column map are
  pooled.

### Corners, alignment, morphology
- `GoodFeaturesToTrack`, `CornerHarris`, `CornerMinEigenVal`, `CornerSubPix`: covariance,
  response, dilate and sort temporaries are pooled. New internal `BoxFilterEngine.ApplyInto`,
  `Morphology.ErodeInto`/`DilateInto`, `Corners.CornerResponseInto`.
- `MatchTemplate`: integral images, padded template and template spectra are pooled.
- `PhaseCorrelate`, `PhaseCorrelateIterative`: four pooled buffers for the whole chain.
- `FindTransformEcc`: all image-size temporaries are pooled, including blurs and gradients (new
  internal `Filters.Filter2DInto`; `Conversions.ConvertInto` is internal).
- `MorphologyEx`: gradient and hit-or-miss intermediates are pooled; the second step reuses the
  result.

### Calibration and stitching
- `LevMarq` Jacobian and residual buffers; `GeometricEstimation` input copies and inlier
  compression; `RhoEstimator`; `PnpRansac` projection buffer (new internal
  `CameraModel.ProjectPointsInto`); EPnP M matrix; `recoverPose` triangulation buffers;
  five-point kernel; `Undistort` chunk maps.
- `DpSeamFinder` is disposable and pools its gray images, derivatives, masks and cost arrays.
- Blenders: feed weight maps are pooled; `MultiBandBlender` reuses its state buffers when
  `Prepare` is called again with the same size, and skips the result copy when no padding is
  needed.
- Bundle adjusters use scratch 3 × 3 matrices. New internal `LinearAlgebra.Invert3x3` is the one
  3 × 3 inverse formula for `Core` and `Stitching`.

## Benchmarks

Allocated bytes per call, 1 thread, BenchmarkDotNet `--fast --inProcess`:

| Case | 256 before → after | 1024 before → after |
|---|---|---|
| SIFT DetectAndCompute | 17.9 MB → 0.41 MB | 275.5 MB → 5.6 MB |
| FindTransformEcc Euclidean | 19.7 MB → 0.04 MB | 313.6 MB → 0.07 MB |
| PhaseCorrelate F32 | 2.37 MB → 0.01 MB | 37.8 MB → 0.03 MB |
| GoodFeaturesToTrack | 1.50 MB → 0.01 MB | 23.1 MB → 0.04 MB |
| ORB DetectAndCompute | 0.98 MB → 0.08 MB | 12.0 MB → 0.15 MB |
| FAST 9/16, no suppression | 0.58 MB → 0.12 MB | 9.45 MB → 2.11 MB (result) |
| MatchTemplate CCoeffNormed T16 | 1.82 MB → 0.24 MB | 21.5 MB → 4.1 MB (result) |
| CornerHarris / MinEigenVal | 1.05 MB → 0.26 MB | 16.8 MB → 4.2 MB (result) |
| DP seam COLOR_GRAD | 6.25 MB → 0.31 MB | 96.1 MB → 1.13 MB |
| FeatherBlender, 4 tiles | 3.33 MB → 2.09 MB | 53.1 MB → 33.3 MB |
| MultiBandBlender, 4 tiles | 5.31 MB → 4.06 MB | 82.5 MB → 62.6 MB |
| FindHomography RANSAC (500 / 5000 points) | 0.09 MB → 0.01 MB | 0.78 MB → 0.01 MB |
| RecoverPose (500 / 2000 points) | 0.13 MB → 0.01 MB | 0.52 MB → 0.01 MB |
| Homography estimator + BundleAdjusterRay | 8.30 MB → 0.95 MB | 3.33 MB → 0.22 MB |

Time, new build ÷ old build (7ed8c0e), BenchmarkDotNet `--fast --inProcess`, three alternating
rounds, minimum of the three medians. Values from 0.9 to 1.1 are within the run-to-run variation
of this machine:

| Case | 1 thread 256 / 1024 | 16 threads 256 / 1024 |
|---|---|---|
| DP seam COLOR_GRAD | 0.71 / 0.76 | 1.07 / 1.11 |
| FeatherBlender, 4 tiles | 0.81 / 0.75 | 1.10 / 0.87 |
| GoodFeaturesToTrack | 0.87 / 1.16 | 0.60 / 0.84 |
| FindTransformEcc | 1.05 / 0.90 | 0.80 / 0.92 |
| PhaseCorrelate | 0.81 / 1.18 | 0.88 / 1.00 |
| Undistort U8C3 | 0.90 / 0.99 | 0.88 / 0.95 |

Steady-state harness (50 warm-up calls, minimum per call, three rounds), ORB and
MatchTemplate: new ÷ base is 0.89–0.97 for ORB Detect and DetectAndCompute at both sizes and
thread counts, and 0.88–1.00 for MatchTemplate T16.

Time changes are small, because the main cost of a new allocation (page commit and zeroing) is
replaced by a rent from a pool that is mostly warm. The gain is lower GC pressure and memory
traffic in long pipelines; the benchmarks run one function at a time.

## Open items

- **Benchmark warm-up.** `--fast --inProcess` runs the 256 case first, while the JIT still runs
  unoptimised (tier 0 or OSR) code. ORB at 256 measures 3.5 ms during the first 0.4 s and 1.4 ms
  after. Single 256 results in `--fast` runs vary 2–5× for this reason. A longer warm-up or
  `DOTNET_TC_QuickJitForLoops=0` on the benchmark job removes this. Applied to the benchmark
  project after this change (see the benchmark `README.md`, measurement notes).
- Results are still new allocations. Destination overloads or a pooled result scope (options 2
  and 3) are needed to remove them.
- Not pooled: `Svd`, `Eigen`, `Solve`, `SolvePoly` and `Gemm` results inside the epipolar and PnP
  kernels; `BFMatcher.KnnMatch` result lists; masked ECC; `FastFeatureDetector.Detect` with a mask;
  the `Filter2DInto` DFT path.
- `MultiBandBlender` with padding still copies the unpadded result region (18 MB at 1024).
