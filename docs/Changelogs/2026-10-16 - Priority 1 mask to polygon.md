# Priority 1: mask-to-polygon ports

**Date:** 2026-10-16 00:20:00

## Summary

The session ported the OpenCV 5.0 functions that convert a contour into a filtered polygon:
shape descriptors, shape fitting and flood fill. All outputs are bit-exact to cv2 5.0.0
(4,509 golden cases, no suspect cases). The full test suite passes (22,314 tests). Two
optimisations made 32F image moments and robust line fitting faster with identical output.

## Changes

### Shape descriptors (`ImgProc/Shapes/ShapeDescriptors`, new)

- `ApproxPolyDP`, `ArcLength`, `ContourArea`, `BoundingRect` (point sets and 8-bit masks),
  `Moments` (contours; images 8U/16U/16S/32F/64F, binary mode), `HuMoments`, `MatchShapes`
  (I1/I2/I3; contours, images and `Moments` pairs).
- Types: `Moments` (record struct), `ShapeMatchModes`.
- Image moments keep the OpenCV 32×32 tile order. Tiles run in parallel; tile sums are added
  on one thread, so the result does not depend on the thread count.
- Source: `modules/geometry/src/approx.cpp`, `shapedescr.cpp`, `moments.cpp`,
  `matchcontours.cpp`.

### Shape fitting (`ImgProc/Shapes/ShapeFitting`, new)

- `ConvexHull`, `ConvexHullIndices`, `IsContourConvex`, `ConvexityDefects`, `MinAreaRect`,
  `BoxPoints`, `MinEnclosingCircle`, `FitEllipse`, `FitEllipseAms`, `FitEllipseDirect`,
  `FitLine` (2-D; L2, L1, L12, Fair, Welsch, Huber), `PointPolygonTest`.
- Types: `Core/RotatedRect`, `Core/Size2f`, `ConvexityDefect`, `EnclosingCircle`, `FittedLine`.
- `Core/Internal/CvRng`: exact `cv::RNG` and per-thread `theRNG`. `minEnclosingCircle`,
  `fitLine` and the ellipse fits consume random numbers in the OpenCV order.
- `fitEllipse` with 25 or more points: the cv2 wheel uses LAPACK SVD; the port uses the Jacobi
  SVD. The golden cases match exactly; the tests allow a relative tolerance of 1e-5.

### Flood fill (`ImgProc/Segmentation/FloodFilling`, new)

- `FloodFill` without and with mask; 8U/32S/32F, 1 or 3 channels; floating and fixed range;
  4 and 8 connectivity; mask-only mode and mask fill value.
- OpenCV 5.0 always runs the gradient fill. The port does the same, and adds an equality fast
  path with identical output for zero differences.
- Images wider or higher than 65,535 pixels throw `ArgumentException`. OpenCV truncates the
  16-bit coordinates.

### Performance

- 32F/64F image moments: each `Vector256<double>` lane holds one row's sums in the OpenCV
  scalar order. At 1024×1024 the case went from 3.6 ms to 0.46–0.95 ms (OpenCvSharp, which uses
  IPP: 0.27–0.34 ms).
- Robust `FitLine`: the distance, weight and weight-sum passes run in one loop. Each double sum
  keeps its OpenCV order. Huber at 1024 went from 278 ms to 169–222 ms.

### Benchmarks

- New suites `ShapeDescriptorBenchmarks`, `ShapeFittingBenchmarks`, `FloodFillBenchmarks` and
  their OpenCvSharp twins. Matching cases were added to `tools/bench_cv2.py`.
- Geometric mean of port time ÷ OpenCvSharp 4.13 time, 1 thread / 16 threads: shape
  descriptors 0.53 / 0.48, shape fitting 0.38 / 0.25, flood fill 0.84 / 0.99.
- Other agents used the machine during the runs. Single cases differed by up to 5× between
  runs. Treat single-case ratios below 2× as noise for this set.
- Results page: https://claude.ai/artifact/D7rGGiR4dJnBS18FUKpqKP (groups "Shape descriptors",
  "Shape fitting", "Flood fill"; OpenCvSharp reference only).

### Documentation

- Port notes: `ShapeDescriptorsPortNotes.md`, `ShapeFittingPortNotes.md`,
  `FloodFillPortNotes.md`. `CLAUDE.md` index, `NOTICE` and the golden tool README are updated.

## Open items

- 32F image moments stay 1.2×–2.8× slower than OpenCvSharp (IPP). The OpenCV scalar order
  limits the port to 3 multiplies and 4 adds per pixel without FMA.
- Not ported: 3-D `fitLine`, `getClosestEllipsePoints`, a public `setRNGSeed`.
