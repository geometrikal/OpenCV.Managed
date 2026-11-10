# Priority 2: linear algebra, pyramids and keypoint types

**Date:** 2026-10-16 00:50:00

## Summary

The session ported the foundations for feature matching and stitching: public linear algebra,
image pyramids and the keypoint data types. Outputs are bit-exact to cv2 5.0.0, except where
cv2 calls its internal LAPACK; those cases use documented tolerances. The full test suite
passes (25,040 tests). An optimisation pass made pyrDown and perspectiveTransform faster with
identical output.

## Changes

### Linear algebra (`Core/LinearAlgebra`, `Core/Svd`, new)

- `Gemm`, `Determinant`, `Invert` (LU, Cholesky, SVD), `Solve` (LU, Cholesky, QR, SVD, EIG,
  Normal), `Eigen` (Jacobi), `SVDecomp`, `SVBackSubst`, `Svd<T>` (compute, backSubst, solveZ),
  `SolveCubic`, `SolvePoly`, `Transform`, `PerspectiveTransform`, `MulTransposed`,
  `SetIdentity`, `Trace`, `CompleteSymm`.
- Types: `Point2d`, `Point3f`, `Point3d`; enums `DecompTypes`, `GemmFlags`, `SvdFlags`.
- Large gemm follows the reference `sgemm`/`dgemm` element order with `Vector<T>` tiles and
  parallel rows; the result does not depend on the thread count.
- cv2 routes SVD (25 or more rows) and QR (30 or more rows) to LAPACK. The port keeps the
  OpenCV Jacobi and QR algorithms; these cases use tolerances (SVD 1e-12 relative for double).
- Adjudicated suspect: cv2 `lapack_QR` returns a wrong solution for square systems with 30 or
  more rows. The port gives the correct solution.

### Image pyramids (`ImgProc/Pyramids/Pyramids`, new)

- `PyrDown`, `PyrUp`, `BuildPyramid`; 8U/16U/16S/32F/64F, any channel count; OpenCV 5.0 size
  and border rules.
- Float sums use the OpenCV SSE3 vector order or scalar order per element, as cv2 does.

### Keypoint types (new)

- `Core/KeyPoint` (`Convert`, `Overlap`), `Core/DMatch`, `Features/KeyPointsFilter`
  (retainBest, runByImageBorder, runByKeypointSize, runByPixelsMask, removeDuplicated,
  removeDuplicatedSorted).
- `RetainBest` returns the OpenCV set sorted by decreasing response; OpenCV's order comes from
  `std::nth_element` and is implementation-defined.

### Performance

- pyrDown 8-bit 1 channel, float 1 channel and 8-bit 3 channels: fused horizontal 5-tap and
  decimation passes (byte shuffles and `Vector<ushort>` loads at byte offsets).
- 2-D float `PerspectiveTransform`: four points per `Vector256<double>` in the scalar
  operation order.

### Benchmarks

- New suites `PyramidBenchmarks`, `LinearAlgebraBenchmarks` and their OpenCvSharp twins.
- Geometric mean of port time ÷ OpenCvSharp 4.13 time, 1 thread / 16 threads: linear algebra
  0.21 / 0.16, pyramids 0.82 / 1.32 (before optimisation: 1.67 / 1.57).
- Examples at 1024, 1 thread: gemm 0.15×, PyrDown U8C3 4.0× → 0.76×, BuildPyramid U8C3
  4.4× → 0.75×, PerspectiveTransform 100k points 2.1× → 0.47×.
- Other agents used the machine during the runs; treat single-case ratios below 2× as noise.
- Results page: https://claude.ai/artifact/D7rGGiR4dJnBS18FUKpqKP (groups "Image pyramids",
  "Linear algebra").

## Open items

- pyrDown with 16 threads scales poorly on 1024 images (U8 2.2× OpenCvSharp); the cause is
  not found. Small images run on one thread because the work estimate is below two stripes.
- 4-channel 8-bit pyrDown and the 3-channel pyrUp interleave use scalar per-pixel paths.
- Not supported: complex gemm, integer `Transform`, `MulTransposed` with delta,
  `EigenNonSymmetric`.
