# PnP solver port notes

Notes for the port of the OpenCV 5.0 perspective-n-point solvers (`Calib/PnP`). File references
are against the OpenCV 5.0.0 submodule at `src/OpenCV`. The `SOLVEPNP_ITERATIVE` method is
described in `UndistortPortNotes.md`.

## Sources

| Port | OpenCV function | OpenCV source |
|---|---|---|
| `PnP.SolvePnP`, `PnP.SolvePnPGeneric`, `CameraModel.SolvePnP(…, method, …)` | `solvePnP`, `solvePnPGeneric` | `modules/geometry/src/solvepnp.cpp` |
| `PnP.SolveP3P` | `solveP3P` | `solvepnp.cpp` |
| `PnP.SolvePnPRansac`, `Internal/PnpRansac` | `solvePnPRansac` (non-USAC), `PnPRansacCallback`, `RANSACPointSetRegistrator::run` | `solvepnp.cpp`, `ptsetreg.cpp` |
| `PnP.SolvePnPRefineLM` | `solvePnPRefineLM` | `solvepnp.cpp`, `levmarq.cpp` |
| `PnP.SolvePnPRefineVVS`, `Internal/PnpRefineVvs` | `solvePnPRefineVVS`, `computeInteractionMatrixAndResiduals`, `exponentialMapToSE3Inv` | `solvepnp.cpp` |
| `Internal/Epnp` | `epnp` | `epnp.cpp`, `epnp.h` |
| `Internal/P3p` | `p3p` (Ding et al. 2023) | `p3p.cpp`, `p3p.h` |
| `Internal/Ap3p` | `ap3p` | `ap3p.cpp`, `ap3p.h` |
| `Internal/PnpPolynomials` | `solve_deg2`, `solve_deg3`, `solve_deg4` (used by AP3P) | `polynom_solver.cpp` |
| `Internal/Sqpnp` | `sqpnp::PoseSolver` | `sqpnp.cpp`, `sqpnp.hpp` |
| `Internal/Ippe` | `IPPE::PoseSolver::solveGeneric`, `solveSquare`, `HomographyHO` | `ippe.cpp`, `ippe.hpp` |

## Public API

```csharp
enum SolvePnPMethod { Iterative = 0, Epnp = 1, P3p = 2, Ap3p = 3, Ippe = 4, IppeSquare = 5, Sqpnp = 6 }
record struct PnPPose(Point3d RotationVector, Point3d TranslationVector)
record struct PnPSolution(Point3d RotationVector, Point3d TranslationVector, double ReprojectionError)
record PnPRansacResult(bool Success, Point3d RotationVector, Point3d TranslationVector, int[] Inliers)

bool PnP.SolvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, SolvePnPMethod method, out Point3d rvec, out Point3d tvec)
PnPSolution[] PnP.SolvePnPGeneric(objectPoints, imagePoints, cameraMatrix, distCoeffs, method = Iterative, PnPPose? extrinsicGuess = null)
PnPPose[] PnP.SolveP3P(objectPoints, imagePoints, cameraMatrix, distCoeffs, SolvePnPMethod method)
PnPRansacResult PnP.SolvePnPRansac(ReadOnlySpan<Point3f> | ReadOnlySpan<Point3d> objectPoints, Point2f | Point2d imagePoints, cameraMatrix, distCoeffs,
    method = Iterative, int iterationsCount = 100, float reprojectionError = 8, double confidence = 0.99, PnPPose? extrinsicGuess = null)
void PnP.SolvePnPRefineLM(objectPoints, imagePoints, cameraMatrix, distCoeffs, ref Point3d rvec, ref Point3d tvec, TermCriteria? criteria = null)
void PnP.SolvePnPRefineVVS(objectPoints, imagePoints, cameraMatrix, distCoeffs, ref Point3d rvec, ref Point3d tvec, TermCriteria? criteria = null, double vvsLambda = 1)
bool CameraModel.SolvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, SolvePnPMethod method, out Point3d rvec, out Point3d tvec)
```

`SolvePnP`, `SolvePnPGeneric`, `SolveP3P` and the refinements take `Point3d`/`Point2d` (OpenCV
`CV_64F` input). The existing `CameraModel.SolvePnP` and `CameraModel.SolvePnPRefine` overloads
are unchanged.

## OpenCV 5.0 behaviour

- The enum values are the OpenCV 5.0 values. OpenCV 5.0 removed `SOLVEPNP_DLS` and
  `SOLVEPNP_UPNP`; the values differ from OpenCV 4.x (`IPPE` 6 → 4, `SQPNP` 8 → 6).
- `solvePnP` returns the first solution of `solvePnPGeneric`. Point counts: at least 4; 3 for
  `ITERATIVE` with a guess and for `SQPNP`. P3P and AP3P in `solvePnP` need exactly 4 points
  (`solveP3P` accepts 3 or 4). `IPPE_SQUARE` needs 4 points. The coplanarity and square-layout
  checks are debug-only in OpenCV and are not ported.
- EPnP, P3P, AP3P, IPPE and SQPnP undistort the image points to normalized coordinates with
  `undistortPoints` (default criteria: 5 iterations). The pose is therefore not exact for
  distorted data, also in cv2.
- AP3P receives the camera matrix and the normalized points, as in OpenCV.
- `solveP3P` sorts the solutions by `‖image − projected‖²` (stable insertion sort).
- IPPE returns 2 solutions; the solution with the smaller float reprojection error is first.
  An OpenCV exception inside IPPE gives 0 solutions (OpenCV catches it).
- `solvePnPGeneric` reprojection error: `norm(projected, image, NORM_L2) / sqrt(2N)`.
- `solvePnPRansac`:
  - The points are converted to `float` (`CV_32F`), also when the input is `double`.
  - Minimal solver: EPnP with 5 points, P3P/AP3P with 4 points; P3P when there are exactly 4
    points. When the point count equals the minimal sample size, the minimal solver runs once on
    all points and every point is an inlier.
  - The minimal solver receives `float` points, so the undistorted points are rounded to
    `float`.
  - `computeError` is `projectPoints` with `float` output and `(float)‖Δ‖²` in `float`
    arithmetic. A point is an inlier when the error ≤ `(float)(reprojectionError²)`.
  - The RANSAC loop is `RANSACPointSetRegistrator::run` with `RNG((uint64)-1)` (`CvRng`).
    `PnPRansacCallback` has no `checkSubset`, so the first sample of each iteration is used.
  - The final pose comes from `method` on the inliers (the inliers widened to `double`); P3P
    and AP3P use EPnP. `ITERATIVE` without a guess starts from the RANSAC model.
  - USAC method values are not ported.
- `solvePnPRefineLM`: OpenCV 5.0 `LevMarq` with geodesic acceleration, `maxCount` iterations,
  step norm tolerance `epsilon` and small energy tolerance `epsilon²`. `CameraLevMarq` has
  optional tolerance parameters for this; the defaults keep the `ITERATIVE` path unchanged.
- `solvePnPRefineVVS`: the products are `MatExpr` products, evaluated as `gemm` calls:
  `R·Xᵀ + t` is `gemm(R, X, 1, t, 1, GEMM_2_T)`, `−λ·L⁺·e` is `gemm(L⁺, e, −λ)`,
  `−R₁·dt` is `gemm(R₁, dt, −1)`. `L⁺` is `invert(L, DECOMP_SVD)`.

## Bit-exactness

The golden battery (3114 cases) matches cv2 5.0.0 bit for bit, except `solvePnPRefineVVS` with
13 or more points (see below). The rules:

- All solvers are scalar `double` code in the C++ expression order. SVD, eigen, solve,
  invert, gemm, `mulTransposed` and `determinant` calls use `Core/LinearAlgebra` and
  `LapackDecomp`, which reproduce the OpenCV internal implementations.
- The `solvePnPGeneric` reprojection error uses `NormDiffL2_SIMD<double, double>` with 2-lane
  SSE2 vectors: four accumulators over blocks of 8 values, the lanes reduced as
  `((r0 + r1) + r2) + r3`, then lane 0 + lane 1, then the scalar tail. `Reductions.Norm(a, b)`
  does not use this order for 2-channel `double` data.
- The RANSAC error is `(0f + dx·dx) + dy·dy` in `float` (`normL2Sqr<float, float>`).

## Tolerances

- `solvePnPRefineVVS` with `2N ≥ 25` rows: cv2 routes `invert(L, DECOMP_SVD)` to LAPACK
  (`hal_internal.cpp`, larger dimension ≥ 25). The port uses the Jacobi SVD. The golden test
  accepts a relative difference of 1e-12; the measured maximum is below 1e-14.
- No other tolerance is necessary on the battery. EPnP SVDs are at most 12 × 12 and SQPnP and IPPE
  matrices are 9 × 9 or smaller, so cv2 uses the internal Jacobi SVD for them.

## Deliberate differences and limits

- OpenCV errors are .NET exceptions: `ArgumentException` for too few points, different counts,
  P3P/AP3P with more than 4 points, IPPE_SQUARE without 4 points and a `SolveP3P` method other
  than P3P/AP3P; `ArgumentOutOfRangeException` for an unknown method and for a RANSAC
  confidence outside (0, 1).
- A failed `SolvePnPRansac` returns `Success = false`. The pose is zero when RANSAC finds no
  model, and the RANSAC model when the final solver fails. OpenCV leaves the output vectors
  uninitialized in the first case.
- Not ported: USAC (`solvePnPRansac` with `UsacParams` or USAC flags), `float` camera matrices,
  `float` input to `SolvePnP` and `SolvePnPGeneric` (the reprojection error is then `double`).

## Golden constraints

- Generator: `Geometrikal.OpenCV.Managed.Tests/tools/generate_pnp_golden.py`, output
  `TestData/pnp_golden.json`. The generator stops when cv2 is not 5.0.0 or when AVX2/AVX-512
  dispatch is active. It disables IPP, OpenCL and threads, and runs every case twice.
- Scenes: ground-control-point layouts within ±60 m, flat and with ±4 m relief; a DJI FC6310
  camera (5472 × 3648) with an off-centre principal point; nadir, oblique and low poses at
  60-120 m; distortion none, k5 and k8; noise 0 or 0.5 px; 4 to 100 points; RANSAC with 4 to
  300 points and 10 %, 30 % and 45 % outliers, `float` and `double` inputs, two
  threshold/confidence settings and an extrinsic guess; IPPE_SQUARE markers.
- Runners: `Tests/Calib/PnpGoldenTests` (all ops) and `Tests/Calib/PnpBasicTests` (pose
  recovery, RANSAC outlier rejection, overload equivalence, validation, no input mutation).

## Performance

- The solvers allocate small `Mat<double>` objects for the SVD and gemm calls; the cost is
  dominated by `undistortPoints`, the SVDs and (for RANSAC) the per-iteration projection.
- RANSAC allocates per iteration (the minimal solver result and the projected points). The
  iteration count is small (adaptive, at most `iterationsCount`).
- Benchmarks: `PnpBenchmarks` and `OpenCvSharpPnpBenchmarks` (SolvePnP EPnP/SQPnP/IPPE/ITERATIVE
  with 100 points, SolvePnPRansac EPnP with 500 points and 30 % outliers). OpenCvSharp 4.13 uses
  the OpenCV 4.x enum values; its twin uses `OpenCvSharp.SolvePnPMethod` names.

## Pooled temporaries (option 1)

- `PnpRansac` projects the inlier test into one rented `Point2d` buffer per call
  (`CameraModel.ProjectPointsInto`). Before, every iteration allocated the widened object points,
  the projection and the narrowed projection.
- `Epnp.ComputePose` rents the 12 × 2N matrix M.
- The `SOLVEPNP_ITERATIVE` reprojection callback rents the projection buffer and the full
  2N × (10 + k) projection Jacobian on every call.

Allocated bytes per call, BenchmarkDotNet `--fast --inProcess`, 1 thread.

| Case | Size | Before | After |
|---|---|---|---|
| SolvePnP EPNP | 100 | 57.7 KB | 38.8 KB |
| SolvePnP ITERATIVE | 100 | 159.6 KB | 103.2 KB |
| SolvePnPRansac EPNP | 500 | 1203.8 KB | 645.7 KB |

The remaining bytes of `solvePnPRansac` come from the per-sample EPnP solve: the `List` result
of `PnP.Generic`, the undistorted sample points and the `LinearAlgebra` outputs (`MulTransposed`,
`SVDecomp`, `Solve`) in `Core`.
