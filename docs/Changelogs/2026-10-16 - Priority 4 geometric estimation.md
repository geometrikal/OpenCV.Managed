# Priority 4: geometric estimation, undistortion and alignment

**Date:** 2026-10-16 01:45:00

## Summary

The session ported the OpenCV 5.0 functions that register two images and correct the camera:
robust affine and homography estimation, lens undistortion, the pinhole camera model, template
matching, phase correlation and ECC alignment. Every golden case matches cv2 5.0.0 bit for bit
(1,796 cases, no tolerances). The full test suite passes (30,780 tests). An optimisation pass
made the 16SC2 undistortion map and the affine RANSAC refinement about 2× faster with identical
output.

## Changes

### Geometric estimation (`Calib/GeometricEstimation`, new)

- `EstimateAffine2D`, `EstimateAffinePartial2D`, `EstimateTranslation2D`, `FindHomography`;
  methods 0, RANSAC, LMEDS and RHO; OpenCV 5.0 LevMarq refinement; `Point2f` and `Point2d`
  inputs. Inlier masks and models are bit-exact.
- RANSAC and LMEDS create `RNG((uint64)-1)` per call, as OpenCV does, so results are
  deterministic.
- USAC (method values 32–38) is not ported. In OpenCV 5.0, plain `RANSAC` does not use USAC.

### Undistortion and camera model (`Calib/Undistortion`, `Calib/CameraModel`, new)

- `InitUndistortRectifyMap` (32FC1, 32FC2, 16SC2), `Undistort`, `UndistortPoints`,
  `UndistortImagePoints`, `GetDefaultNewCameraMatrix`, `GetOptimalNewCameraMatrix`.
- `Rodrigues` (with jacobians), `ProjectPoints` (with jacobian), `SolvePnP` and
  `SolvePnPRefine` (SOLVEPNP_ITERATIVE).
- Map rows follow the OpenCV SSE2 block order and the scalar tail order. `Undistort` keeps the
  OpenCV row stripes, so its output does not always equal one full map followed by `Remap`.

### Alignment (`ImgProc/Alignment`, new)

- `MatchTemplate` (all six methods, 8U/32F, 1–4 channels, byte and float masks),
  `PhaseCorrelate`, `PhaseCorrelateIterative`, `CreateHanningWindow`, `FindTransformEcc`,
  `FindTransformEccWithMask`, `ComputeEcc`.
- `FloatReductions`: OpenCV-order float sum, mean, meanStdDev, norm and dot.
- `EccWarp`: the OpenCV 5.0 float-coordinate warp kernels, used by ECC only. The public
  `Geometry` warps keep the 4.x table path.

### Performance

- 16SC2 undistortion map: AVX store with half-to-even rounding and saturating packs; map blocks
  computed in registers. 1024, 1 thread: 8.2 ms → 4.1 ms (OpenCvSharp 4.0 ms).
- LevMarq normal equations: vectorised `JᵀJ` and `Jᵀb` in the OpenCV element order; 4-, 6- and
  9-parameter normal matrices kept in registers. `EstimateAffine2D` RANSAC, 5,000 points:
  1,120 µs → 452 µs (OpenCvSharp 691 µs).

### Benchmarks

- New suites `EstimationBenchmarks`, `UndistortBenchmarks`, `AlignmentBenchmarks` and their
  OpenCvSharp twins.

Port ÷ OpenCvSharp 4.13 after optimisation. "BDN" is BenchmarkDotNet `--fast` on a busy machine
(three agents built and tested during the run); "A/B" is an isolated Stopwatch harness (minimum
of 30 batches, port and OpenCvSharp alternating in one process).

| Case | BDN 1 thread | BDN 16 threads | A/B 1 thread |
|---|---|---|---|
| EstimateAffine2D RANSAC, 5,000 points | 0.43 | 0.80 | 0.65 |
| EstimateAffinePartial2D RANSAC, 5,000 points | 0.29 | 1.55 | 0.72 |
| FindHomography RANSAC, 5,000 points | 0.49 | 0.66 | 0.46 |
| FindHomography LMEDS, 5,000 points | 0.71 | 1.22 | 0.80 |
| FindHomography RHO, 5,000 points | 0.95 | 1.86 | 1.36 |
| InitUndistortRectifyMap 16SC2, 1024 | 1.64 | 0.60 | 1.03 |
| InitUndistortRectifyMap 32FC1, 1024 | 2.69 | 0.69 | 1.06 |
| Undistort U8C3, 1024 | 3.05 | 0.33 | 1.15 |
| UndistortPoints, 10,000 points | 1.18–1.46 | 0.62–0.68 | – |
| PhaseCorrelate F32, 1024 | 0.93 | 0.72 | – |
| MatchTemplate CCoeffNormed T64, 1024 | 1.65 | 1.04 | – |
| FindTransformEcc Euclidean, 256 | 0.93 | 1.09 | – |

Geometric mean (BDN), 1 thread / 16 threads: estimation 0.58 / 1.33, undistortion 1.78 / 0.95,
alignment 1.74 / 1.37. Results page: https://claude.ai/artifact/D7rGGiR4dJnBS18FUKpqKP (groups
"Geometric estimation", "Undistortion", "Alignment").

## Open items

- Undistortion maps at 256 × 256 stay 1.5–3.9× slower than OpenCvSharp. Each returned map is a
  new array on the large-object heap; a destination overload would remove this.
- `FindHomography` RHO is about 1.3× slower. Its float refinement runs sequential sums that
  define the result.
- Not ported: USAC, `solvePnPRansac` and the other PnP methods, `findFundamentalMat`,
  `findEssentialMat`, `initInverseRectificationMap`, fisheye.
