# Shape-fitting port notes

Semantics of `ShapeFitting` (OpenCV `convexHull`, `isContourConvex`, `convexityDefects`,
`minAreaRect`, `boxPoints`, `minEnclosingCircle`, `fitEllipse`, `fitEllipseAMS`,
`fitEllipseDirect`, `fitLine`, `pointPolygonTest`), the OpenCV 5.0 conventions, the
build-dependent behaviour of cv2 5.0.0 and the golden constraints. Read this before you change
a shape-fitting function, `FitMatrix` or `CvRng`.

## Sources

- Port source: OpenCV 5.0. In 5.0 these functions moved from `imgproc` to the `geometry` module.
  - `modules/geometry/src/convhull.cpp`
  - `modules/geometry/src/rotcalipers.cpp`
  - `modules/geometry/src/shapedescr.cpp` (enclosing circle and the ellipse fits)
  - `modules/geometry/src/linefit.cpp`
  - `modules/geometry/src/geometry.cpp` (`pointPolygonTest`)
  - `modules/core/src/lapack.cpp`, `matrix_decomp.cpp`, `lda.cpp`, `matrix_operations.cpp`,
    `matmul.simd.hpp` (SVD, LU determinant, non-symmetric eigen decomposition, `sortIdx`,
    `mulTransposed`)
  - `modules/core/include/opencv2/core/operations.hpp`, `modules/core/src/rand.cpp` (`RNG`)
  - `modules/core/src/types.cpp` (`RotatedRect::points`)
- Golden reference: cv2 5.0.0 (opencv-contrib-python-headless 5.0.0.93, Windows x64, MSVC,
  baseline SSE3, built-in LAPACK).
- Files:

| File | Content |
|---|---|
| `ImgProc/Shapes/ShapeFitting.cs` | Public API |
| `ImgProc/Shapes/ConvexityDefect.cs` | Defect record (OpenCV `Vec4i`) |
| `ImgProc/Shapes/EnclosingCircle.cs` | Circle record |
| `ImgProc/Shapes/FittedLine.cs` | Line record (vx, vy, x0, y0) |
| `ImgProc/Shapes/Internal/HullSklansky.cs` | Sklansky hull, convexity test, convexity defects |
| `ImgProc/Shapes/Internal/CalipersMinAreaRect.cs` | Rotating calipers, `minAreaRect` |
| `ImgProc/Shapes/Internal/FitEnclosingCircle.cs` | Welzl circle with the seeded shuffle |
| `ImgProc/Shapes/Internal/FitEllipseCore.cs` | General, AMS and Direct ellipse fits |
| `ImgProc/Shapes/Internal/FitLine2D.cs` | 2-D line fit with M-estimators |
| `ImgProc/Shapes/Internal/FitPointPolygonTest.cs` | `pointPolygonTest` |
| `ImgProc/Shapes/Internal/FitMatrix.cs` | Jacobi SVD with right vectors, SVD back substitution, SVD solve, LU determinant, JAMA eigen decomposition |
| `Core/RotatedRect.cs`, `Core/Size2f.cs` | `cv::RotatedRect`, `cv::Size2f` |
| `Core/Internal/CvRng.cs` | `cv::RNG` and the per-thread `theRNG()` |

## Input types

- Every function has a `ReadOnlySpan<Point>` overload (OpenCV `CV_32SC2`) and a
  `ReadOnlySpan<Point2f>` overload (`CV_32FC2`). The functions do not change the input.
- `minAreaRect`, the ellipse fits and `fitLine` convert integer points to float first, as
  OpenCV does. `convexityDefects` accepts integer contours only, as OpenCV does.
- The hull, circle and point-test internals read interleaved `int` or `float` coordinates
  through `MemoryMarshal.Cast`. Generic code over `INumber<T>` gives the integer and the float
  arithmetic of the OpenCV templates.

## Bit-exactness

All functions match cv2 5.0.0 exactly on the golden battery. The port keeps the OpenCV
operation order, the `float` and `double` precision of each expression and the order of the
random draws. Rules that the battery depends on:

- `float` expressions stay `float`. Examples: the calipers, `findCircle3pts`, the `fitLine`
  sums of products (`w * x * x` is a `float` product added to a `double` sum), the
  `fitEllipse` centroid. C# and the baseline SSE3 MSVC build both round each `float`
  operation; neither contracts multiply-add.
- `cv::norm(Point2f)` and `normalize(Vec2f)` compute the length in `double`.
- `fitLine` uses `MathF.Cos`, `MathF.Sin` and `MathF.Exp` where OpenCV calls the `float`
  overloads (`std::cos(float)`).
- `fitEllipse` sums `fabs(p.x) + fabs(p.y)` in `float` (the MSVC `fabs(float)` overload).
- `mulTransposed` sums each element sequentially (`MatrixSolvers.MulTransposed`); the OpenCV
  two-lane SIMD body has the same per-element order.

## convexHull

- The points are sorted by x, then y, then input index. The index stands for the pointer
  comparison of `CHullCmpPoints`. The port sorts 64-bit keys and then orders runs of equal keys
  by index. `-0` and `+0` get the same key, because the float comparator finds them equal.
- Outside the Sklansky scan, OpenCV compares float points through `Point*` (as integers).
  The "all points equal" test, the collinear test of the lower chain and the duplicate-index
  search therefore treat `-0` and `+0` as different. The port does the same (`SameBits`).
  The golden cases `deg_negzero_dup*_f` pin this.
- `ConvexHullIndices` (`returnPoints = false`) runs the OpenCV duplicate-index search, which
  replaces an index by an equal point with a higher index when this keeps the index sequence
  monotonous. `ConvexHull` (`returnPoints = true`) does not run it. The two results can
  therefore refer to different duplicates.
- When all points are equal, the result is index 0.
- An empty input gives an empty result. OpenCV releases the output.

## convexityDefects

- The result is empty when the contour has 3 points or fewer, or the hull has fewer than 3
  indices. OpenCV releases the output in both cases.
- Non-monotonous hull indices (a self-intersecting contour) throw `ArgumentException`. OpenCV
  raises `StsBadArg`. An empty hull also throws `ArgumentException`; an index outside the
  contour throws `ArgumentOutOfRangeException`.
- `FixedPointDepth` is `cvRound(depth * 256)` (ties to even).

## minAreaRect and boxPoints

- OpenCV 5.0 convention: the angle is in [-90, 0). An axis-aligned rectangle and a single
  point give -90.
- The hull is the counter-clockwise (`clockwise = false`) Sklansky hull, converted to float.
- Two hull points give a zero-width rectangle with the 5.0 case analysis on `dx` and `dy`.
- An empty input gives a zero rectangle with angle -90.
- `BoxPoints` and `RotatedRect.Points` port `RotatedRect::points`: the angle is converted to
  radians in `double`, and `cos` and `sin` are rounded to `float` before the corner arithmetic.

## minEnclosingCircle

- OpenCV 5.0 shuffles inputs with more than 10 points before the Welzl iteration. The seed is
  `count ^ x0 ^ (y0 << 8) ^ (xn << 16) ^ (yn << 24)` over the `float` bit patterns of the first
  and last points, and the generator is `cv::RNG(seed)`. The result is deterministic.
- The radius includes the margin `EPS = 1e-4`. One point gives radius `1e-4`; an empty input
  gives a zero circle.
- `findSecondPoint` and `findThirdPoint` compare a `double` distance with the `float` radius;
  the main loop compares a `float` distance. The port keeps both comparisons.

## Ellipse fits

- `FitEllipse` uses the Direct fit for exactly 5 points and the general conic fit
  (`fitEllipseNoDirect`) for more points. All three fits require at least 5 points and throw
  `ArgumentException` otherwise. OpenCV raises `StsBadSize`.
- `fitEllipseNoDirect` sets the angle only when it swaps width and height. In the other case the
  angle is 0 (the `RotatedRect` default). The port keeps this OpenCV 5.0 behaviour.
- `fitEllipseAMS` and `fitEllipseDirect` draw one offset from `theRNG()` for every point in
  every attempt, also when the offset range is 0. The draws do not change the result for a
  regular input, but they advance the per-thread generator. The port draws the same values
  from `CvRng.TheRng`, so a later degenerate fit on the same thread uses the same offsets as
  OpenCV.
- `getOfs` builds `Point2f(rng.uniform(-eps, eps), rng.uniform(-eps, eps))`. The MSVC build
  evaluates the constructor arguments right to left: the y offset is drawn first. The port
  does the same. A build that evaluates left to right (for example GCC on some targets) gives
  different offsets for degenerate inputs only. The golden cases `deg_collinear_*` pin the MSVC
  order.
- `eigenNonSymmetric` sorts eigenvalues with `sortIdx`: `std::sort` ascending, then reversed.
  For fewer than 32 elements the MSVC `std::sort` is a stable insertion sort; the port uses the
  same sort, so equal eigenvalues (complex pairs) keep the OpenCV order.
- `determinant` of the 5 × 5 AMS matrix uses `LUImpl` with `DBL_EPSILON * 100`.

### SVD and LAPACK

- `fitEllipseNoDirect` uses `SVDecomp`, `SVBackSubst` and `solve(DECOMP_SVD)`. The port
  implements these with the OpenCV Jacobi SVD (`JacobiSVDImpl_`) and `SVBkSbImpl_`.
- cv2 5.0.0 wheels are built with LAPACK. The LAPACK HAL replaces the Jacobi SVD when the matrix
  has 25 or more rows (`HAL_SVD_SMALL_MATRIX_THRESH`). `fitEllipse` with 25 or more points
  therefore uses `dgesdd` in cv2.
- The golden runner compares those cases (`"lapack": true`) with a relative tolerance of 1e-5.
  On the current battery all 58 such cases match exactly, because the final `float` rounding
  hides the `double` differences. Inputs that are close to degenerate can differ in the last
  bits.
- OpenCV builds without LAPACK use the Jacobi SVD for all sizes and match the port exactly.

## fitLine

- `DistanceTypes.L2` is the closed-form weighted least-squares fit.
- `L1`, `L12`, `Fair`, `Welsch` and `Huber` run 20 random starts of 30 reweighting iterations.
  Each start selects 10 points with `cv::RNG(-1)` (state `0xFFFFFFFFFFFFFFFF`), so the result is
  deterministic.
- `param = 0` selects the OpenCV default constant of each weight function. `reps = 0` selects
  1 and `aeps = 0` selects 0.01.
- `DistanceTypes.C` and `DistanceTypes.User` throw `ArgumentException`. An empty point set
  throws `ArgumentException`. 3-D input is not ported.

## pointPolygonTest

- Integer contours with an integer test point and `measureDist = false` use the OpenCV
  integer branch with 64-bit cross products. All other cases use the float branch.
- An empty contour gives -1, or `-double.MaxValue` when `measureDist` is true.

## CvRng

- `Core/Internal/CvRng` is the OpenCV multiply-with-carry generator (coefficient 4164903690)
  with `next`, `uniform(int, int)`, `uniform(float, float)`, `uniform(double, double)` and the
  `float`/`double` conversions. Reuse it for any port that needs `cv::RNG` sequences.
- `CvRng.TheRng` is the per-thread `theRNG()` with the initial state `0xffffffff`.
  `CvRng.SetTheRngSeed` is `cv::setRNGSeed`. Both are internal; the tests use them to align the
  generator with cv2.

## Golden constraints

- Generator: `Geometrikal.OpenCV.Managed.Tests/tools/generate_shapefit_golden.py`. It asserts
  `cv2.__version__ == "5.0.0"` and disables IPP, OpenCL and threads.
- The battery: contours from `findContours` of analytic masks (rotated rectangles at 0, 30, 45
  and 90 degrees, circles, a star), random masks, thin lines; float ellipse samples with 5, 6,
  12 and 40 points; noisy lines; clouds; degenerate sets (1, 2, 3 points, equal points,
  collinear points, duplicates, signed zeros); 257- and 300-point sets for the shuffle.
- Each ellipse case calls `cv2.setRNGSeed(12345)` first; the runner calls
  `CvRng.SetTheRngSeed(12345)`.
- cv2 raises an error for `convexityDefects` on self-intersecting contours. Those cases store
  `"error": true` and the runner expects `ArgumentException`.
- The suspect list is empty: no case differs between cv2 5.0.0 and the port.
- `TestData/shapefit_golden.json` is not yet in the test csproj copy list. The runner reads it
  from the output folder when it is there, and otherwise from the source tree.

## Performance

- The hull sorts 64-bit keys with an index permutation instead of a comparison sort over
  pointers. Temporaries (sort keys, stacks, hull buffers, ellipse design matrices, SVD
  matrices) come from `ArrayPool<T>.Shared`.
- The functions run on one thread per call. A caller with many contours can run the calls in
  parallel: the only shared state is the per-thread `theRNG()`.
- Benchmarks: `ShapeFittingBenchmarks` and `OpenCvSharpShapeFittingBenchmarks` process every
  contour of `Inputs.Binary(Size)` per call.
- `fitLine` with a robust distance fuses OpenCV passes that are independent. `calcDist2D`, the
  weight function and the weight sum run in one loop. The weight normalization runs in the
  `fitLine2D_wods` loop. Each `double` sum keeps its OpenCV order, so the sums are bit-exact.
  The serial sums overlap, which hides the add latency. The weight function is a struct generic
  parameter, not a per-element `switch`.

Measured gaps, 1 thread, BenchmarkDotNet `--fast --inProcess`, busy machine. Ratio is port
time / OpenCvSharp 4.13 time.

| Case | Size | Port before (us) | Port after (us) | OpenCvSharp (us) | Ratio before | Ratio after |
|---|---|---|---|---|---|---|
| FitLineHuber | 256 | 16070 | 11100 | 12360 | 1.48x | 0.90x |
| FitLineHuber | 1024 | 278030 | 169380 | 177570 | 1.43x | 0.95x |
| MinAreaRect | 256 | 121 | 121 | 247 | 0.49x | 0.49x |
| MinAreaRect | 1024 | 2080 | 2080 | 3466 | 0.60x | 0.60x |
| ConvexHull | 1024 | 1964 | 1964 | 3271 | 0.60x | 0.60x |
| ConvexHullIndices | 1024 | 2043 | 2043 | 3732 | 0.55x | 0.55x |
| PointPolygonTest | 1024 | 128 | 128 | 340 | 0.38x | 0.38x |
| PointPolygonTestDistance | 1024 | 316 | 316 | 531 | 0.60x | 0.60x |

- The earlier gaps for `MinAreaRect`, `ConvexHullIndices` and `PointPolygonTest` did not
  reproduce in back-to-back runs: the port is faster than OpenCvSharp in every case. These
  functions are not changed.
- `ConvexHullIndices` and `ConvexHull` take the same time (within 4 %). The duplicate-index
  search (`KeepMonotonousIndices`) walks only the run of equal points after a hull vertex and
  is linear in the point count.

## Polygon intersections

`ImgProc/Shapes/ShapeIntersection` ports `intersectConvexConvex` (O'Rourke) and
`rotatedRectangleIntersection` from the OpenCV 5.0 geometry module (`geometry.cpp`,
`intersection.cpp`); enum `RectanglesIntersectType`.

- `IntersectConvexConvex`: each input is reversed when its first non-zero turn is negative; the
  result is reversed back when both were. Segment intersections subtract in double
  (PR #26974). The internal buffer starts with a FLT_MAX sentinel; the final pass shifts it out,
  computes the area in double and removes a closing duplicate. A buffer longer than `n + m + 1`
  returns area −1 (non-convex input, OpenCV 6623c62f56). With `handleNested`, a polygon whose
  vertices all lie inside or on the other is the result, with its `contourArea`.
- `RotatedRectangleIntersection`: both rectangles are shifted by their mean centre (PR #23690),
  equal rectangles return `Full` with the first corners, edge intersections and contained corners
  are collected in float, points closer than an adaptive tolerance are merged, more than 8 points
  are reduced by the OpenCV distance table, and the points are ordered by the cross-product swap.

Golden data: `tools/generate_intersection_golden.py` → `TestData/intersection_golden.json` (201
cases), `ImgProc/ShapeIntersectionGoldenTests`; 181 match bit for bit. 20 adjudicated cases are
pinned in `ImgProc/ShapeIntersectionOpenCv5DifferenceTests`:

- The cv2 4.5.5 wheel returns the unprocessed buffer with the FLT_MAX sentinel for touching and
  disjoint squares (6 cases). The port returns the 5.0 source result: the shared edge, or no
  vertices.
- PR #26974 and PR #23690 change float rounding: vertices within 1e-4 and areas within 1e-5
  relative (14 cases).
