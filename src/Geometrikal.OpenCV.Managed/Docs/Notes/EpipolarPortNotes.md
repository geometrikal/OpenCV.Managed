# Two-View Epipolar Geometry Port Notes

## Scope

The port covers the classic (non-USAC) two-view functions of the OpenCV 5.0 `geometry` module:

| Managed API | OpenCV function | OpenCV source |
|---|---|---|
| `EpipolarGeometry.FindFundamentalMat` | `cv::findFundamentalMat` (method form, `maxIters`) | `fundam.cpp` |
| `EpipolarGeometry.FindEssentialMat` | `cv::findEssentialMat` (camera matrix and focal/pp forms) | `five-point.cpp` |
| `EpipolarGeometry.DecomposeEssentialMat` | `cv::decomposeEssentialMat` | `five-point.cpp` |
| `EpipolarGeometry.RecoverPose` | `cv::recoverPose` (E form, with and without `distanceThresh`/`triangulatedPoints`; focal/pp form) | `five-point.cpp` |
| `EpipolarGeometry.TriangulatePoints` | `cv::triangulatePoints` | `triangulate.cpp` |
| `EpipolarGeometry.ComputeCorrespondEpilines` | `cv::computeCorrespondEpilines` | `fundam.cpp` |
| `EpipolarGeometry.ConvertPointsToHomogeneous`, `ConvertPointsFromHomogeneous` | `cv::convertPointsToHomogeneous`, `cv::convertPointsFromHomogeneous` (2-D ↔ 3-D) | `fundam.cpp` |

New public types: enum `FundamentalMatMethod` (`FM_7POINT`, `FM_8POINT`, `FM_LMEDS`, `FM_RANSAC`),
records `EpipolarEstimationResult` and `RecoverPoseResult`.

Internals (`Calib/Internal/`):

| File | Content |
|---|---|
| `EssentialRegistrators.cs` | `IMultiModelKernel<TPoint>`; `MultiModelRegistrator` (RANSAC and LMEDS registrators for kernels with several models per sample and any point type) |
| `EssentialFivePoint.cs` | `EMEstimatorCallback`: five-point solver (`getCoeffMat`, `solvePoly`, `SVD::solveZ`), Sampson error |
| `FundamentalKernels.cs` | `FMEstimatorCallback`: `run7Point`, `run8Point`, symmetric epipolar distance |
| `TriangulationKernels.cs` | DLT triangulation (`hal::SVD64f` 4 × 4), `decomposeEssentialMat`, the `recoverPose` chirality test, the `(x - c)/f` normalization |

The existing `RansacRegistrator` and `LmedsRegistrator` accept one model per sample and `Point2f`
only, so the port adds `MultiModelRegistrator`. It uses the same `getSubset`, `RANSACUpdateNumIters`,
inlier test and LMEDS median rule. The existing registrators are not changed.

## OpenCV 5.0 dispatch

- `findFundamentalMat` converts the points to `CV_32F`. Fewer than 7 pairs give an empty result.
  Exactly 7 pairs or `FM_8POINT` call the kernel once on all pairs and set the mask to ones. All
  other cases use RANSAC when `(method & ~3) == FM_RANSAC` and there are at least 15 pairs, and
  LMEDS otherwise. Thus `FM_RANSAC` with 8 to 14 pairs and `FM_7POINT` with more than 7 pairs run
  LMEDS. The minimal sample has 7 pairs; each sample gives 1 to 3 models. A threshold ≤ 0 becomes
  3; a confidence outside (DBL_EPSILON, 1 - DBL_EPSILON) becomes 0.99.
- `findEssentialMat` converts the points to `CV_64F`, normalizes them with the camera matrix and
  divides the threshold by `(fx + fy)/2`. `method == RANSAC` uses RANSAC; every other non-USAC
  value uses LMEDS. The return value of the registrator is ignored: LMEDS returns its best model
  also when fewer than 5 pairs are inliers. Exactly 5 pairs return all solutions (up to 10,
  stacked as a 3k × 3 matrix).
- `recoverPose` without `distanceThresh` uses 50. The output mask is the candidate mask (255 for a
  pass) AND the input mask. Without an input mask the values are 255; with the mask of
  `findEssentialMat` the values are 1.
- USAC methods throw `NotSupportedException`.

## Bit-exactness

These rules give exact results:

- The 7-point and five-point solvers take the null space from `SVD::compute` with `FULL_UV`
  (`Svd<double>`, Jacobi SVD with the `0x12345678` generator for the completed basis). Both
  matrices have 9 columns, below the LAPACK limit of 25.
- `run8Point` accumulates `A += r·rᵀ` as `A + (0 + rᵢ·rⱼ)` and uses `cv::eigen` (Jacobi). The
  rank-2 projection and the de-normalization use the `Matx` product loop (`s = 0; s += a·b`).
  `run7Point` de-normalizes with two `gemm` calls and scales with `convertTo` (`x·s + 0`).
- `EMEstimatorCallback`: `A.colRange(0, 10).inv() * A.colRange(10, 20)` is evaluated by
  `MatOp_Invert::matmul` as `solve(A0, A1, DECOMP_LU)`, not as an inverse and a product.
  Copies through `* 1.0` add `+ 0`. `Evec` follows the `MatExpr` order:
  `addWeighted(c0, x, c1, y)` (`(c0·x + c1·y) + 0`), then `scaleAdd` (`c2·z + s`), then `+ c3`;
  the norm is `NormL2_SIMD<double, double>` (`LevMarq.NormL2Sqr`).
- The Sampson error and the symmetric epipolar distance follow the `Matx` and scalar C++
  expression orders. `std::max` returns its first argument for NaN.
- `findEssentialMat` and `recoverPose` normalize with `convertTo(alpha = 1/f, beta = (-c)·(1/f))`.
- `recoverPose` computes `P·Q` with `gemm`, divides the rows of `Q` by row 3 in row order and keeps
  the C++ comparison order. `-t` is `0 - t`. `decomposeEssentialMat` negates with `x·(-1) + 0`
  and computes `U·Wᵀ` with `GEMM_2_T`.
- `triangulatePoints` builds the transposed 4 × 4 DLT matrix and calls `JacobiSVD` with
  `m = n = n1 = 4`, as `hal::SVD64f`. `float` points give a `float` result.

## Tolerance

`solvePoly` uses .NET `Math.Pow`, `Math.Cos` and `Math.Sin` for the initial root circle (see
`LinearAlgebraPortNotes.md`). The roots of the degree-10 polynomial can then differ from cv2 in
the last bits. In the golden battery 41 of the 45 `findEssentialMat` cases are bit-exact; 4 cases
differ by at most 1.2e-13 in an element of the unit-norm matrix. All masks are identical. The
golden test uses an absolute tolerance of 1e-10 for essential matrices only. All other outputs
are compared bit for bit.

## Behavioural differences

- OpenCV errors are .NET exceptions: `ArgumentException` for different point counts, a camera or
  essential matrix that is not 3 × 3, a projection matrix that is not 3 × 4 and a mask of the wrong
  length; `ArgumentOutOfRangeException` for a confidence outside (0, 1) in `findEssentialMat` and
  for `whichImage` other than 1 or 2; `NotSupportedException` for USAC.
- `decomposeEssentialMat` and `recoverPose` accept only one 3 × 3 matrix. OpenCV also rejects a
  stacked 3k × 3 matrix.
- Not ported: USAC and the `UsacParams` overloads, the overloads with distortion coefficients
  (`findEssentialMat` and `recoverPose` with `undistortPoints`), `correctMatches`,
  `sampsonDistance`, 3-D input to `findFundamentalMat`, integer and 4-D input to the homogeneous
  conversions.

## Golden constraints

- Generator: `Geometrikal.OpenCV.Managed.Tests/tools/generate_epipolar_golden.py`, output
  `TestData/epipolar_golden.json` (264 cases: 45 essential, 108 recoverPose, 81 fundamental,
  18 epilines, 6 decompositions, 4 triangulations, 2 homogeneous conversions). The generator stops
  when cv2 is not 5.0.0, when AVX2/AVX-512 dispatch is active, or when cv2 does not use the internal
  LAPACK. It disables IPP, OpenCL and threads.
- Scenes: nadir (100 m, ±10 m relief, 10 m baseline), oblique (20-200 m depth, 15° view change)
  and planar (degenerate for the fundamental matrix); 0.5 px noise; 0%, 10%, 30% and 50%
  outliers; 5 to 2000 pairs; `float` and `double` input; thresholds, iteration limits,
  out-of-range confidence and too few points.
- The suspect list is empty.

## Performance

- The registrators do not allocate in the iteration loop except inside the kernels. The masks and
  error buffers come from `ArrayPool<T>.Shared`.
- The five-point kernel allocates small matrices for each sample (`Svd<double>` of 5 × 9, the
  10 × 10 `solve`, one `SVD::solveZ` per real root). This follows the OpenCV kernel, which also
  allocates `Mat` objects per sample.
- `recoverPose` triangulates every pair four times (one 4 × 4 Jacobi SVD per pair and pose).
  `TriangulationKernels.JacobiSvd4LastVector` is a 4 × 4 `double` copy of `JacobiSVDImpl_` with
  the same operation order. It records the row permutation of the sort and returns only the last
  row of `Vᵀ`. It does not compute the left singular vectors after the sort, because
  `triangulatePoints` does not read them. It does not allocate.
- `LapackDecomp.JacobiSvd` keeps the squared column norms in a stack buffer when n ≤ 16.
- `FundamentalKernel.ComputeError` processes 4 pairs per step with AVX `Vector256<double>`. Each
  lane uses the scalar operation order. The JIT does not contract the operations into FMA. After
  the unpack, the lanes hold pairs 0, 2, 1, 3; a shuffle restores the order before the `float`
  store. A scalar loop processes the remaining pairs, and all pairs when AVX is not supported.
- Benchmarks: `EpipolarBenchmarks` and `OpenCvSharpEpipolarBenchmarks` (500 and 2000 pairs, 30%
  outliers).

### Results

BenchmarkDotNet `--fast --inProcess`. The ratio is port time ÷ OpenCvSharp 4.13 time in the same
run. Another process used the machine. The ratios have an uncertainty of approximately ±0.15.
"Before" is the 1-thread measurement given at the start of the work.

| Case | Pairs | Before (1 thread) | After (1 thread) | After (16 threads) |
|---|---|---|---|---|
| FindFundamentalMat_Ransac | 500 | 1.09 | 0.49–0.68 | 0.49 |
| FindFundamentalMat_Ransac | 2000 | 1.86 | 0.39–0.40 | 0.39 |
| RecoverPose | 500 | 1.29 | 0.24–0.25 | 0.23 |
| RecoverPose | 2000 | 1.63 | 0.25–0.26 | 0.21 |
| FindEssentialMat_Ransac | 500 | 1.16 | 0.95–1.06 | 0.93 |
| FindEssentialMat_Ransac | 2000 | 1.10 | 0.84–0.92 | 0.81 |

Absolute 1-thread times at 2000 pairs, measured in this work: `FindFundamentalMat_Ransac`
6.6 ms → 1.3 ms; `RecoverPose` 8.4 ms → 5.9–6.5 ms. The OpenCvSharp `RecoverPose` measured
23–28 ms in the same runs. `FindEssentialMat_Ransac` did not change; its ratio change is noise.

### Not changed

- The five-point kernel (`EssentialFivePointKernel`) allocates its small matrices for each
  sample. The `findEssentialMat` time does not depend on the number of pairs (9.4 ms at 500 and at
  2000 pairs). The kernel, not the error loop, sets the time. The time is at parity with
  OpenCvSharp.
- The seven-point kernel allocates `Svd<double>` and `Mat<double>` objects for each sample
  (approximately 300 KB for each call). The fundamental matrix estimation is already 2× faster
  than OpenCvSharp.

## Pooled temporaries (option 1)

- `findEssentialMat` and `recoverPose` normalize the points into rented `Point2d` arrays.
- `recoverPose` triangulates each of the four candidates into a rented 4 × N matrix and computes
  `P·Q` with `GemmKernels.Run` into a rented 3 × N matrix. The triangulated output, when requested,
  is still a new copy.
- `findFundamentalMat` with `Point2d` input converts to rented `Point2f` arrays.
- The five-point kernel uses a stack buffer for the 10 × 20 coefficient matrix and rents A0 and A1.

Allocated bytes per call, BenchmarkDotNet `--fast --inProcess`, 1 thread.

| Case | Size | Before | After |
|---|---|---|---|
| FindFundamentalMat RANSAC | 500 | 384.8 KB | 376.9 KB |
| FindFundamentalMat RANSAC | 2000 | 309.2 KB | 277.9 KB |
| FindEssentialMat RANSAC | 500 | 597.1 KB | 449.3 KB |
| FindEssentialMat RANSAC | 2000 | 632.6 KB | 439.1 KB |
| RecoverPose | 500 | 131.4 KB | 6.1 KB |
| RecoverPose | 2000 | 513.5 KB | 12.7 KB |

The remaining bytes of the estimators come from the per-sample kernels: `Svd<double>`, `Eigen`,
`Solve`, `SolvePoly` and `Gemm` return new matrices (`Core`).
