# Robust Geometric Estimation Port Notes

## Scope

The port covers the 2-D robust estimators of the OpenCV 5.0 `geometry` module:

| Managed API | OpenCV function | OpenCV source |
|---|---|---|
| `GeometricEstimation.EstimateAffine2D` | `cv::estimateAffine2D` (method form) | `ptsetreg.cpp` |
| `GeometricEstimation.EstimateAffinePartial2D` | `cv::estimateAffinePartial2D` | `ptsetreg.cpp` |
| `GeometricEstimation.EstimateTranslation2D` | `cv::estimateTranslation2D` | `ptsetreg.cpp` |
| `GeometricEstimation.FindHomography` | `cv::findHomography` (method form) | `fundam.cpp`, `rho.cpp` |

Every function has a `Point2f` and a `Point2d` overload. The `Point2d` overload rounds the points
to `float` first, as OpenCV converts all inputs to `CV_32FC2`. The functions do not change the
input points. The result records are `GeometricEstimationResult` (a 2 × 3 or 3 × 3 `Mat<double>`
model, or null, and an inlier mask) and `TranslationEstimationResult` (a `Point2d` and an inlier
mask).

Internals (`Calib/Internal/`):

| File | Content |
|---|---|
| `PointSetRegistrator.cs` | `IEstimatorKernel`, `RANSACUpdateNumIters`, `getSubset`, `compressElems`, `haveCollinearPoints`, `scaleFor` |
| `RansacRegistrator.cs` | `RANSACPointSetRegistrator::run` |
| `LmedsRegistrator.cs` | `LMeDSPointSetRegistrator::run` |
| `AffineKernels.cs` | `Affine2DEstimatorCallback`, `AffinePartial2DEstimatorCallback`, `Translation2DEstimatorCallback`; vector error loops |
| `HomographyKernels.cs` | `HomographyEstimatorCallback` (normalized DLT, Jacobi eigen) |
| `LevMarq.cs` | OpenCV 5.0 `LevMarq` with the dense backend and geodesic acceleration |
| `RefineProblems.cs` | The refinement callbacks of the four functions |
| `RhoEstimator.cs` | `RHO_HEST_REFC` (PROSAC, SPRT, non-randomness criterion, float LM) |

## OpenCV 5.0 dispatch

- `findHomography` and `estimateAffine2D` call `usac::findHomography` and
  `usac::estimateAffine2D` only for the method values `USAC_DEFAULT` (32) to `USAC_MAGSAC` (38).
  `RANSAC` (8) and `LMEDS` (4) use the classic `RANSACPointSetRegistrator` and
  `LMeDSPointSetRegistrator`. `RHO` (16) uses `rho.cpp`. Method 0 solves all points.
- `estimateAffinePartial2D` and `estimateTranslation2D` have no USAC route. OpenCV raises
  `StsBadArg` for any method other than `RANSAC` and `LMEDS`.
- The port implements every non-USAC path. A USAC method value throws
  `NotSupportedException`. The `UsacParams` overloads are not ported.

## Algorithms

- **RANSAC.** `RNG rng((uint64)-1)` is created in every call (`CvRng(ulong.MaxValue)`). Each
  iteration draws `modelPoints` distinct indices with `rng.uniform(0, count)`, checks the sample
  (`checkSubset`, at most 10000 attempts), computes the model and counts the points with a
  squared error ≤ `(float)(threshold²)`. A model replaces the best model when its inlier count
  is above `max(best, modelPoints - 1)`; the iteration bound is then updated with
  `RANSACUpdateNumIters`.
- **LMEDS.** The iteration count is `max(RANSACUpdateNumIters(confidence, 0.45, …), 3)`. The
  median is `std::nth_element` over the float bit patterns read as `int`. The final threshold is
  `max(2.5 · 1.4826 · (1 + 5/(count - modelPoints)) · sqrt(minMedian), 0.001)`.
- **Refinement.** `estimateAffine2D` (more than 3 pairs), `estimateAffinePartial2D` (more than 2
  pairs) and `estimateTranslation2D` (with `refineIters > 0`) move the inliers to the front and
  run `LevMarq` with `refineIters` iterations and geodesic acceleration. `estimateTranslation2D`
  with `refineIters = 0` returns the mean displacement of the inliers. `findHomography` (more than
  4 pairs, not RHO) recomputes the DLT model on the inliers for RANSAC and LMEDS, runs `LevMarq`
  with 10 iterations, scales the model by `1/H₂₂` and recomputes the mask over all points with
  the threshold.
- **RHO.** PROSAC sampling with the XORshift128+ generator seeded with `~0`, SPRT evaluation,
  the non-randomness stopping criterion (beta 0.35), at least 100 iterations, and a final float
  Levenberg-Marquardt refinement with a damped 8 × 8 Cholesky factorization.

## Bit-exactness

The golden data matches the port bit for bit: all inlier masks and all model coefficients. These
rules give the exact results:

- The error loops use `float` arithmetic in the C++ order. The vector loops hold four
  interleaved points per `Vector256<float>`; the y lanes add the two products in swapped order,
  which gives the same value because float addition is commutative.
- `haveCollinearPoints` subtracts the coordinates in `float` and compares in `double`.
- The homography DLT uses `cv::eigen` (Jacobi, `LapackDecomp.Jacobi`) of the 9 × 9 `LᵀL`, the
  `gemm` small-matrix order for the two 3 × 3 products (`t·1 + 0`), and `convertTo` with a scale
  (`x·s + 0`, or a copy when the scale is 1).
- `LevMarq` follows the `MatExpr` evaluation of each expression: `-x` is `0 - x`; `x·h` is
  `x·h + 0`; `calcJtrvv` is `addWeighted(scaleAdd(jtb, h-1, jtbv), α, multiply(lmDiag, v, h), α)`
  with `α = 1/(h·h)`; `x += xgeo·0.5` adds `xgeo·0.5 + 0`. `JᵀJ` uses the `MulTransposedR` order (one sequential sum per element, computed in one pass over the rows), `Jᵀb`
  uses `gemm` with `GEMM_1_T`, the linear system uses `solve` with `DECOMP_SVD`.
- `norm(NORM_L2SQR)` of a `double` vector uses four 2-lane accumulators
  (`NormL2_SIMD<double, double>`), `cv::sum` adds groups of four, `Mat::dot` uses `dotProd_`.
- RHO: `(int)` of a float in the sample test is the x86 `cvttss2si` result (`int.MinValue` for
  NaN and out-of-range values); `(unsigned)` of a double is the MSVC x64 64-bit truncation.
  `sacTRInv8x8` is called with the same buffer as source and destination; the port inverts in
  place in the same order.

## Reproduced quirks

- `findHomography` with exactly 4 pairs ignores the method and returns the direct solution with
  a mask of ones.
- `findHomography` with method 0 and more than 4 pairs refines the least-squares model and then
  sets the mask from the threshold. Points with a large error get 0.
- `findHomography` replaces a threshold ≤ 0 with 3. The affine and translation functions use the
  threshold as given.
- The SPRT evaluation of RHO stops at the first rejection. The inlier count and the mask of that
  model then cover only the tested points, and the model can still become the best model. The
  port keeps this behaviour and the stale mask bytes.
- The RHO sample test truncates the cross products to `int` before it compares the signs. Small
  cross products of opposite sign are not rejected.

## Behavioural differences

- OpenCV errors are .NET exceptions: `ArgumentException` for too few pairs, different lengths,
  an empty translation input and an unknown homography method; `ArgumentOutOfRangeException`
  for a confidence outside (0, 1) with RANSAC or LMEDS; `NotSupportedException` for USAC methods
  and for methods other than RANSAC and LMEDS in the affine and translation functions.
- Not ported: USAC, the `UsacParams` overloads, homogeneous (3-D) point input to
  `findHomography`, `estimateAffine3D`, `estimateTranslation3D`, `findFundamentalMat`,
  `findEssentialMat`, the RHO extrinsic guess.

## Golden constraints

- Generator: `Geometrikal.OpenCV.Managed.Tests/tools/generate_estimation_golden.py`, output
  `TestData/estimation_golden.json` (361 cases). The generator stops when cv2 is not 5.0.0, when
  AVX2/AVX-512 dispatch is active, or when cv2 does not use the internal LAPACK. It disables
  IPP, OpenCL and threads.
- The generator runs every case twice and asserts equal outputs: RANSAC and LMEDS create
  their generator in each call, and RHO creates a new context in each call.
- The battery: exact, noisy and outlier sets (10%, 30%, 45%, 60%) for each function and method;
  `float` and `double` inputs; minimal counts; refinement iterations 0, 1, 3, 30; thresholds,
  iteration limits and confidence levels; collinear, equal and duplicated points; integer grids;
  near-similarity "drone" sets with 300 and 1000 pairs; RHO with sorted (PROSAC) sets and small
  sets; error cases.
- The suspect list is empty.

## Performance

- The RANSAC and LMEDS loops do not allocate: the minimal samples, models and index buffers are
  on the stack, and the masks and the LMEDS error buffer come from `ArrayPool<T>.Shared`. The
  public functions allocate the copies of the inputs, the mask and the model.
- With AVX, the error and inlier loop of each kernel processes four points per instruction.
- `LevMarq` allocates its matrices (the Jacobian is `2n × p` doubles). It runs once per call.
- Benchmarks: `EstimationBenchmarks` and `OpenCvSharpEstimationBenchmarks` (500 and 5000 pairs,
  30% outliers).

### Measured optimisation (October 2026)

Profile: `EstimateAffine2D` with RANSAC on 5000 pairs took 47 µs in RANSAC and about 1070 µs
in the `LevMarq` refinement. Most of the refinement time was in `gemm(J, b, GEMM_1_T)` (two
calls per iteration, reference BLAS path) and in `MulTransposedR`.

Changes. All changes keep the summation order of each element, so the results are
bit-identical:

- `MulAtB` replaces `Gemm` for `Jᵀb` from `GemmKernels.BlasRowThreshold` rows. That gemm path
  sums each element sequentially over the rows, starting from zero. The new loop keeps this
  order and vectorises across the parameters. Below the threshold, the port calls `Gemm`.
- `MulTransposedR` keeps `JᵀJ` in `Vector256` registers for 4, 6 and 9 parameters. The
  9-parameter case uses three passes of three rows. Other parameter counts use a vectorised
  row loop.
- The residual callbacks of the affine, partial affine and homography problems read the
  parameters once and slice the spans once per call.
- Local accumulators in the RHO `sacCalcJacobianErrors` gave no measurable gain. This change
  is not kept.

Results: Stopwatch harness, minimum of 30 batches. The baseline and the new build ran back to
back, each with OpenCvSharp 4.13 in the same process. Times in µs.

| Case | Threads | Before | After | OpenCvSharp |
|---|---|---|---|---|
| EstimateAffine2D_Ransac 500 | 1 | 79 | 32 | 79 |
| EstimateAffine2D_Ransac 5000 | 1 | 1120 | 452 | 691 |
| EstimateAffinePartial2D_Ransac 500 | 1 | 46 | 22 | 62 |
| EstimateAffinePartial2D_Ransac 5000 | 1 | 467 | 244 | 339 |
| FindHomography_Ransac 5000 | 1 | 1297 | 977 | 2114 |
| FindHomography_Lmeds 5000 | 1 | 4812 | 4604 | 5748 |
| FindHomography_Rho 5000 | 1 | 2456 | 2562 | 1890 |
| EstimateAffine2D_Ransac 5000 | 16 | 1561 | 479 | 702 |
| EstimateAffinePartial2D_Ransac 5000 | 16 | 535 | 276 | 388 |

`FindHomography_Rho` did not change. About 60% of its time is the `float` refinement (up to 100
iterations over all inliers). The sequential sums of this refinement define the result.

## Pooled temporaries (option 1)

- `LevMarq.Optimize` rents the Jacobian, the residual vector and the geodesic residual vector
  (`PooledMat<double>`). The Jacobian has `ParamCount × ErrorCount` elements; this was the
  largest temporary of `findHomography` and of the bundle adjusters.
- `GeometricEstimation` passes the input spans to the registrators without a copy. The inlier
  compression of the refinement step writes to rented arrays (`CompressInliers`). The `Point2d`
  overloads convert to rented `Point2f` arrays.
- `RhoEstimator` rents its point copies, the two inlier masks and the non-randomness table.

Allocated bytes per call, BenchmarkDotNet `--fast --inProcess`, 1 thread.

| Case | Size | Before | After |
|---|---|---|---|
| FindHomography RANSAC | 500 | 90.8 KB | 13.5 KB |
| FindHomography RANSAC | 5000 | 765.8 KB | 11.9 KB |
| FindHomography LMEDS | 5000 | 766.5 KB | 12.4 KB |
| FindHomography RHO | 5000 | 192.3 KB | 6.7 KB |
| EstimateAffine2D RANSAC | 5000 | 528.6 KB | 15.7 KB |
| EstimateAffinePartial2D RANSAC | 5000 | 414.0 KB | 9.9 KB |
| EstimateAffinePartial2D LMEDS | 5000 | 414.1 KB | 9.9 KB |

The remaining bytes are the result mask, the model and small per-call objects.
