# Linear Algebra Port Notes

## Scope

The port covers the OpenCV 5.0 dense linear algebra of the core module:

| Managed API | OpenCV function | OpenCV source |
|---|---|---|
| `LinearAlgebra.Gemm` | `cv::gemm` | `matmul.dispatch.cpp`, `matmul.simd.hpp` (`gemmImpl`), `hal_internal.cpp` (`lapack_gemm`), `3rdparty/clapack/src/sgemm.c`, `dgemm.c` |
| `LinearAlgebra.Determinant` | `cv::determinant` | `lapack.cpp`, `matrix_decomp.cpp` (`LUImpl`) |
| `LinearAlgebra.Invert` | `cv::invert` | `lapack.cpp` |
| `LinearAlgebra.Solve` | `cv::solve` | `lapack.cpp`, `matrix_decomp.cpp` (`LUImpl`, `CholImpl`, `QRImpl`) |
| `LinearAlgebra.Eigen` | `cv::eigen` (no Eigen library) | `lapack.cpp` (`JacobiImpl_`) |
| `Svd<T>`, `LinearAlgebra.SVDecomp`, `LinearAlgebra.SVBackSubst` | `cv::SVD`, `SVD::compute`, `SVD::backSubst`, `SVD::solveZ` | `lapack.cpp` (`JacobiSVDImpl_`, `SVBkSbImpl_`), `operations.hpp` |
| `LinearAlgebra.SolveCubic`, `LinearAlgebra.SolvePoly` | `cv::solveCubic`, `cv::solvePoly` | `mathfuncs.cpp` |
| `LinearAlgebra.Transform` | `cv::transform` (float and double) | `matmul.simd.hpp` (`transform_`, `transform_32f`, `transform_64f`, `diagtransform_`) |
| `LinearAlgebra.PerspectiveTransform` | `cv::perspectiveTransform` | `matmul.simd.hpp` (`perspectiveTransform_`) |
| `LinearAlgebra.MulTransposed` | `cv::mulTransposed` (no delta) | `matmul.dispatch.cpp`, `matmul.simd.hpp` (`MulTransposedR`, `MulTransposedL`) |
| `LinearAlgebra.SetIdentity`, `Trace`, `CompleteSymm` | `cv::setIdentity`, `cv::trace`, `cv::completeSymm` | `matrix_operations.cpp` |

New types: `Point2d`, `Point3f`, `Point3d`, enums `DecompTypes`, `GemmFlags`, `SvdFlags`.
Internals: `Core/Internal/LapackScalar` (conversions, constants, `RNG::next`, the lapack.cpp
`hypot`), `LapackDecomp` (LU, Cholesky, QR, Jacobi eigen, Jacobi SVD, SVD back substitution),
`GemmKernels`, `PointTransformKernels`, `PolySolvers`.

`ImgProc/Geometry/Internal/MatrixSolvers` is a separate double-only copy for the warp helpers.
The new code does not use it.

## Reference build

The golden data comes from cv2 5.0.0 built with MSVC `/fp:precise`, baseline SSE3, and the
OpenCV internal LAPACK (`Lapack: YES (Built-In libclapack)`). The generator disables the AVX,
AVX2, FMA3 and AVX-512 dispatch. The internal LAPACK changes the code path of some functions,
because `hal_internal.cpp` routes these calls to LAPACK:

| Function | LAPACK when | Managed port |
|---|---|---|
| `gemm` | rows of A ≥ 100 | Reference BLAS element order (bit-exact) |
| LU (`solve`, `invert`, `determinant`) | n ≥ 100 | Always `LUImpl` |
| Cholesky | n ≥ 100 | Always `CholImpl` |
| SVD (`SVD::compute`, `solve` with SVD, `invert` with SVD) | larger dimension ≥ 25 | Always `JacobiSVDImpl_` |
| QR (`solve` with QR) | rows ≥ 30 | Always `QRImpl` |

## Bit-exactness

The kernels are generic over `float` and `double` and follow the C++ promotion rules of each
template. The JIT does not contract multiply-add, and MSVC `/fp:precise` at SSE3 does not contract
either. The following rules make the results bit-exact with the reference build:

- `gemm` below 100 rows of A follows `gemmImpl`:
  - The small path (no flags, inner size 2 to 4, inner size equal to a result dimension) sums in
    `T`, then computes `t·alpha + c·beta` in `double`. An absent C contributes `0·beta`.
  - An inner size of 1 computes `(a·alpha)·b`.
  - A transposed B (or a continuous single-column B, which OpenCV reads as transposed) uses four
    interleaved `double` partial sums. When OpenCV takes the block path with k-blocks smaller
    than the inner size, each k-block uses two interleaved partial sums with a carry.
  - All other cases use one sequential `double` sum per element. The port runs it as an axpy over
    the row with `Vector<double>` lanes (widened from `float`).
  - For `float`, `alpha` and `beta` are rounded to `float` first, as `hal::gemm32f` does.
- `gemm` from 100 rows of A follows the reference `xGEMM` loops that `cblas_?gemm(CblasRowMajor)`
  calls. Each element is rounded to `T` after each term, in increasing k order. The managed kernel
  keeps this per-element order and uses `Vector<T>` lanes across the output columns, 4 × 2-vector
  register tiles, 256 × 512 cache panels and parallel row stripes. The result does not depend on
  the thread count. A transposed B is packed (transposed) first, so that the dot-product forms also
  run in lanes.
- `invert` and `solve` for n ≤ 3 use the explicit formulas. `invert` 2 × 2 follows the
  `CV_SIMD128` code: `float` multiplies by `(float)(1/det)`; `double` writes `0 - s·d` for the
  off-diagonal elements (positive zero for a zero element). `solve` 3 × 3 `float` has one `float`
  product (`b1·s22`) in the second unknown, as in the C++ expression.
- `JacobiSVDImpl_` uses the lapack.cpp `hypot` (scaled square root), not the C library `hypot`.
  The zero-singular-value branch uses the OpenCV `RNG` with seed `0x12345678`.
- `transform` reproduces the 128-bit dispatch: `float` 3 → 3 uses `(v0·m0 + v1·m1) + (v2·m2 + m3)`
  (`v_matmuladd`) for every point except the last point of a run; `double` 3 → 3 uses
  `v0·m0 + (v1·m1 + (v2·m2 + m3))` for point pairs; `float` 4 → 4 uses the `v_reduce_sum4` order.
  A run is the whole matrix when the source is continuous and one row otherwise, as with
  `NAryMatIterator`.
- `mulTransposed` uses `MulTransposedR`/`MulTransposedL` with `double` sums, and `gemm` when both
  dimensions of the source are ≥ 100.

## Behavioural differences and quirks

- **Reproduced quirks.**
  - `invert` with SVD and fewer rows than columns returns `vt[n-1-nm] / w[0]`, because OpenCV
    reads `w[n-1]` past the singular values into the `Vᵀ` buffer.
  - `perspectiveTransform` for channel combinations other than 2 → 2, 3 → 3 and 3 → 2 multiplies
    the projected coordinates by W instead of dividing by W.
  - `invert` and `solve` with SVD or EIG return `w[n-1]/w[0]` computed in `T`.
- **Deliberate differences.**
  - `Solve` with `DecompTypes.Eig` and without `DecompTypes.Normal` throws for a non-square
    matrix. OpenCV reads outside the eigenvector buffer in this case.
  - `SolvePoly` fills the surplus roots of a polynomial with zero leading coefficients with zero.
    OpenCV copies them from a buffer slot that holds leftover coefficient data.
  - `Svd<T>` never modifies the source. `SvdFlags.ModifyA` is accepted and has no effect.
- **Not supported.** Two-channel (complex) `gemm`; `transform` with one source and one
  destination channel (a `convertTo` in OpenCV); integer `transform`; `mulTransposed` with
  delta or another destination type; `eigenNonSymmetric`.
- **Tolerances.** The port keeps the OpenCV algorithms where cv2 calls LAPACK, so these results
  agree only within a tolerance: SVD with a larger dimension ≥ 25 (singular values agree to about
  1e-12 relative for `double`; singular vectors can differ in sign), and QR with ≥ 30 rows.
  `solveCubic` and `solvePoly` use .NET `Math.Acos`, `Cos`, `Cbrt`, `Pow`; the golden tests use a
  relative tolerance of 1e-13 (`double`) and 1e-9 (`solvePoly`).

## Adjudicated golden cases

- `solve_f32_qr_50_nb1`, `solve_f64_qr_50_nb1`: cv2 solves a square `DECOMP_QR` system with 30 or
  more rows through `lapack_QR` and returns a wrong solution (residual about 40 for a 50 × 50
  system; `numpy.linalg.lstsq` and the managed `QRImpl` agree to 1e-15). The cases are suspect.
  `LinearAlgebraBasicTests.Solve_Qr_LargeSquare_MatchesLu` pins the port behaviour.

## Golden constraints

- Generator: `Geometrikal.OpenCV.Managed.Tests/tools/generate_linalg_golden.py`, output
  `TestData/linalg_golden.json` (915 cases). The generator stops when cv2 is not 5.0.0, when
  AVX2/AVX-512 dispatch is active, or when cv2 does not use the internal LAPACK.
- `solve` with LU or Cholesky, n ≤ 3 and one right-hand side does not write the destination on
  failure in OpenCV. The runner does not compare the solution in that case.
- `lapackSvd` cases compare only the singular values; the runner checks the managed vectors by
  reconstruction.

## Performance

Measured on the development machine (steady state, 1 thread unless stated):

| Case | Time |
|---|---|
| `Gemm` 256 × 256 `float` | 1.0 ms (0.5 ms, 16 threads) |
| `Gemm` 1024 × 1024 `float` | 94 ms (17 ms, 16 threads) |
| `Solve` 3 × 3 LU `double` | 41 ns |
| `Solve` 9 × 9 LU `double` | 0.5 µs |
| `Svd<double>` 8 × 9 `FullUV` | 8.8 µs |
| `PerspectiveTransform` 100 000 `Point2f` | 0.17 ms (1.0 ms before AVX) |

Small matrices (≤ 16 × 16) use stack buffers for LU and Cholesky; the only heap allocation is the
result. `Svd<T>` allocates its work buffers. `Transform` and the `PerspectiveTransform` layouts
other than 2D `float` are scalar loops in OpenCV order. Benchmarks: `LinearAlgebraBenchmarks` and
`OpenCvSharpLinearAlgebraBenchmarks`.

### `PerspectiveTransform` for 2D `float` points

OpenCV `perspectiveTransform_` is a scalar loop in `double` precision. The port computes four
points per iteration with `Vector256<double>` (AVX):

- One 256-bit load holds four points. `vcvtps2pd` converts each half to `double`.
- `vunpcklpd` and `vunpckhpd` separate x and y. The lane order is points 0, 2, 1, 3.
- The arithmetic is the scalar order: `w = (x·m6 + y·m7) + m8`, `w = 1/w`,
  `((x·m0 + y·m1) + m2)·w`. The JIT does not contract the products into FMA instructions.
- A `|w| > FLT_EPSILON` mask sets the result to `+0` where OpenCV writes 0. A NaN `w` gives 0,
  as in the scalar code.
- `vunpcklpd` and `vunpckhpd` restore the point order. `vcvtpd2ps` rounds to `float` (round to
  nearest, as the scalar cast).

The point-array overloads allocate the result with `GC.AllocateUninitializedArray`, because the
kernel writes every element.

BenchmarkDotNet `--fast --inProcess`, 100 000 points, OpenCvSharp 4.13 in the same run:

| Threads | Port before | Port after | OpenCvSharp | Ratio before | Ratio after |
|---|---|---|---|---|---|
| 1 | 1.28 ms | 0.17 ms | 0.32 ms | 3.9 | 0.53 |
| 16 | 1.20 ms | 0.17 ms | 0.31 ms | 3.3 | 0.55 |

The port and OpenCvSharp run `perspectiveTransform` on one thread.
`LinearAlgebraBasicTests.PerspectiveTransform_ManyPoints_MatchesScalarFormula` compares 1003
points (including `w = 0` and NaN) with the scalar formula bit for bit.
