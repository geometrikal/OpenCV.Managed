# Shape descriptors port notes

Semantics of `ShapeDescriptors` (OpenCV `approxPolyDP`, `arcLength`, `contourArea`,
`boundingRect`, `moments`, `HuMoments`, `matchShapes`), the OpenCV 5.0 decisions, the deliberate
differences and the golden constraints. Read this before you change a shape-descriptor kernel or
the shape golden generator.

## Sources

- Port source: OpenCV 5.0. In 5.0 these functions are in the `geometry` module, not in `imgproc`:
  - `modules/geometry/src/approx.cpp` (`approxPolyDP_`)
  - `modules/geometry/src/geometry.cpp` (`contourArea`, `arcLength`, `maskBoundingRect`,
    `pointSetBoundingRect`, `boundingRect`)
  - `modules/geometry/src/moments.cpp` (`contourMoments`, `momentsInTile`, `moments`,
    `completeMomentState`, `cv::Moments::Moments`, `HuMoments`)
  - `modules/geometry/src/matchcontours.cpp` (`matchShapes`)
- Golden reference: cv2 5.0.0 (opencv-contrib-python-headless), IPP off.
- Files:

| File | Content |
|---|---|
| `ImgProc/Shapes/ShapeDescriptors.cs` | Public API |
| `ImgProc/Shapes/Moments.cs` | `Moments` record struct (OpenCV `cv::Moments`) |
| `ImgProc/Shapes/ShapeMatchModes.cs` | `matchShapes` methods (OpenCV numeric values) |
| `ImgProc/Shapes/Internal/DescrPointOps.cs` | `Point_<int>` and `Point_<float>` arithmetic in the coordinate type |
| `ImgProc/Shapes/Internal/DescrApproxPoly.cs` | Douglas-Peucker with the OpenCV slice stack and clean-up pass |
| `ImgProc/Shapes/Internal/DescrPolygon.cs` | Perimeter, area, contour moments, point-set bounding rectangle |
| `ImgProc/Shapes/Internal/DescrMaskBounds.cs` | Mask bounding rectangle (`Vector<byte>` scans, row stripes) |
| `ImgProc/Shapes/Internal/DescrImageMoments.cs` | Image moments over 32 x 32 tiles |
| `ImgProc/Shapes/Internal/DescrHuMatch.cs` | Hu invariants and shape distances |

## Public API

```csharp
Point[]   ApproxPolyDP(ReadOnlySpan<Point> curve, double epsilon, bool closed)
Point2f[] ApproxPolyDP(ReadOnlySpan<Point2f> curve, double epsilon, bool closed)
double    ArcLength(ReadOnlySpan<Point> curve, bool closed)            // and Point2f
double    ContourArea(ReadOnlySpan<Point> contour, bool oriented = false) // and Point2f
Rect      BoundingRect(ReadOnlySpan<Point> points)                     // and Point2f
Rect      BoundingRect(Mat<byte> mask)
Moments   Moments(ReadOnlySpan<Point> contour)                         // and Point2f
Moments   Moments<T>(Mat<T> image, bool binaryImage = false)           // byte, ushort, short, float, double
double[]  HuMoments(in Moments moments)
void      HuMoments(in Moments moments, Span<double> destination)
double    MatchShapes(ReadOnlySpan<Point> c1, ReadOnlySpan<Point> c2, ShapeMatchModes method) // and Point2f
double    MatchShapes<T>(Mat<T> image1, Mat<T> image2, ShapeMatchModes method)
double    MatchShapes(in Moments m1, in Moments m2, ShapeMatchModes method)
```

`Moments` property names are the OpenCV field names in Pascal case: `M00` to `M03`, `Mu20` to
`Mu03`, `Nu20` to `Nu03`. `new Moments(m00, ..., m03)` computes the central and normalized
moments as the OpenCV constructor does. `default(Moments)` has all values 0.

## Semantics

### approxPolyDP

- `epsilon` must be in [0, 1e30). The port throws `ArgumentOutOfRangeException` otherwise (OpenCV
  `StsOutOfRange`).
- An empty curve gives an empty array. cv2 returns `None`.
- `eps` is squared once. Distances are compared as `dist² * segmentLength²` against
  `eps² * segmentLength²`, as OpenCV does; a point whose projection falls outside the segment uses
  the distance to the nearest end point.
- Closed curves: three farthest-point searches start at point 0. The last search start and its
  farthest point split the curve into two slices. An open curve whose first and last points are
  equal is processed as a closed curve with one search, and its last point is then appended.
- The slice stack is LIFO. The right slice is pushed first, so the left slice is processed first.
- A final pass removes points on almost straight lines:
  `dist² <= 0.5 * eps² * |end - start|²`, `dx != 0`, `dy != 0` and a non-negative inner product.
  The test `dx != 0 && dy != 0` keeps points between horizontal or vertical neighbours.
- The coordinate arithmetic runs in the coordinate type as in the C++ template: `int` differences
  and squared lengths wrap (OpenCV has undefined behaviour there), `float` differences round to
  single precision. `IDescrPointOps<TP>` holds these expressions.

### arcLength and contourArea

- Both functions convert coordinates to `float` first, also for `Point`. `int` coordinates above
  2^24 therefore lose precision, as in OpenCV.
- `arcLength`: each segment length is `sqrtf(dx² + dy²)` in `float`; the sum is `double`. 0 or 1
  point gives 0.
- `contourArea`: `0.5 * sum((double)x[i-1] * y[i] - (double)y[i-1] * x[i])`. The area is positive
  for a contour that is clockwise on screen (X right, Y down).

### boundingRect

- Point sets: `Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)`. `float` limits round down with
  `cvFloor` on all four sides; a fractional maximum can lie outside the rectangle. An empty set
  gives `Rect(0, 0, 0, 0)`.
- The point-set scan uses `Vector<int>`/`Vector<float>` min/max on interleaved x, y lanes, seeded
  with the first point and finished with an overlapping last vector, as the OpenCV SIMD loop.
  `floor(min) == min(floor)`, so the result equals the scalar code. NaN coordinates are not
  supported (the vector min/max NaN rules are platform-specific in OpenCV).
- Masks: the exact bounding box of the nonzero pixels; no nonzero pixel gives `Rect(0, 0, 0, 0)`.
  Each row scans only the columns outside the current column range (left part, right part, then the
  middle part only when the outer parts are zero), as OpenCV `maskBoundingRect`. The scans use
  `Vector<byte>`. Row stripes run in parallel; the stripe ranges merge with min/max, so the result
  does not depend on the thread count. Only `Mat<byte>` is supported (OpenCV also accepts `CV_8S`).

### moments

- Contours: Green's theorem over the polygon in `double` (`int` coordinates convert to `double`
  directly, not through `float`). The sign of the area is removed by the sign of the `1/2`, `1/6`,
  ... factors. When `|a00| <= FLT_EPSILON` all 24 values are 0; the central moments are not
  computed.
- Images: OpenCV computes the ten spatial moments of each 32 x 32 tile in local coordinates and
  adds the tiles in row-major order with the shift formulas in `double`. The port computes the tile
  moments in parallel (one tile row per work item) into a pooled buffer and adds them on one thread
  in the OpenCV order. The result is bit-exact and does not depend on the thread count.
- 8-bit, 16-bit and binary tiles: OpenCV accumulates `int` or `int64` sums that do not overflow
  (the largest 8-bit tile moment is 2.01e9 < 2^31). These sums are exact, so the port can use a
  different order: per tile row it sums each column with the weights `1, y, y², y³`
  (`Vector<int>` lanes for 8-bit and binary, `long` for 16-bit), then reduces each tile with the
  weights `x^j`. The tile moments are identical integers.
- 32F and 64F tiles accumulate in `double` in the OpenCV order (per row along x, then per row of
  the tile). With AVX, one `Vector256<double>` lane holds the four row sums of one row, so a
  vector computes 4 rows at a time. Each lane runs the scalar operations of its row in the scalar
  order. A 4 x 4 transpose loads the lanes. The per-row sums go to a 32-row buffer, and the tile
  sums add them in row order. Without AVX, and for the last `th mod 8` rows, the loop is scalar.
- Binary mode: `p = (v != 0) ? 255 : 0` (OpenCV `compare(src, 0, CMP_NE)`; NaN maps to 255, -0
  maps to 0), 8-bit tile moments, then each tile moment is multiplied by `1.0 / 255`.
- `completeMomentState` and the `cv::Moments` constructor are the same arithmetic; both are the
  `Moments` constructor. `|m00| <= DBL_EPSILON` gives a mass center (0, 0) and normalized moments
  0.

### HuMoments and matchShapes

- `HuMoments` is the OpenCV expression sequence without change.
- `matchShapes`: `m = sign(h) * log10(|h|)`; an invariant pair contributes only when both
  `|h| > 1e-5`. I1 sums `|1/mB - 1/mA|`, I2 sums `|mB - mA|`, I3 takes `max(|(mA - mB) / mA|)`.
  When exactly one shape has all invariants 0, the result is `double.MaxValue` (OpenCV `DBL_MAX`).
  Image inputs use non-binary image moments, as OpenCV `moments(image)`.
- An undefined `ShapeMatchModes` value throws `ArgumentOutOfRangeException` (OpenCV `StsBadArg`).

## Deliberate differences

- OpenCV `moments` treats a continuous single-channel `CV_32F` or `CV_32S` matrix with two columns
  (or a two-channel row or column) as a point vector. `Moments<T>(Mat<T>)` always treats its input
  as an image; use the point overloads for contours.
- `MatchShapes(in Moments, in Moments, ...)` is an addition. It compares a contour with an image,
  which OpenCV does with two `InputArray` arguments of different kinds.
- The HAL hooks (`cv_hal_imageMoments`, `cv_hal_polygonMoments`) and the IPP path of `moments` are
  not ported. The results match OpenCV without IPP and without a HAL replacement.

## Performance

`ShapeDescriptorBenchmarks.MomentsImage_F32` against `OpenCvSharpShapeDescriptorBenchmarks`
(OpenCV 4.13), 1 thread, BenchmarkDotNet `--fast --inProcess`, busy machine:

| Size | Port before (us) | Port after (us) | OpenCvSharp (us) | Ratio before | Ratio after |
|---|---|---|---|---|---|
| 256 | 234 | 64 (median) | 29 | 8.1x | 2.2x |
| 1024 | 3640 | 458 | 268 | 13.6x | 1.7x |

- The row-lane form is limited by the double multiply and add throughput: 3 multiplies and 4
  adds per pixel, in the OpenCV order and without FMA.
- A `Vector512<double>` form (8 rows per vector) gave no measurable gain on an i9-11900, which
  has one 512-bit FMA unit. The port keeps the `Vector256` form.
- OpenCvSharp runs the IPP path (`CV_IPP_RUN(!binary, ipp_moments(...))`) for non-binary
  images. The port must keep the OpenCV 5.0 scalar order for bit-exact results, so the
  remaining gap is not chased.

## Golden data

- Generator: `Geometrikal.OpenCV.Managed.Tests/tools/generate_shapes_golden.py`, output
  `TestData/shapes_golden.json`, runner `ImgProc/ShapeDescriptorsGoldenTests.cs`.
- The generator requires cv2 5.0.0. It sets `OPENCV_CPU_DISABLE`, disables IPP, OpenCL and
  threads. The shape functions use baseline code only, so the dispatch setting has no effect.
- Battery: contours from `findContours` (RETR_LIST, CHAIN_APPROX_NONE and SIMPLE) of seeded random
  masks and analytic masks (circle with hole, rotated ellipse, star, thin lines); float point sets
  (scaled contours, random sets, a noisy circle, negative coordinates); degenerate sets (1 and 2
  points, two equal points, collinear points, a back-tracking line, duplicate points, all-equal
  points, an open curve with equal end points, coordinates above 2^24); six epsilons and both
  `closed` values for every set; image moments for 8U, 16U, 16S, 32F, 64F with widths and heights
  that are not multiples of 32, constant and sparse images, signed values, `-0.0`, ROIs, and both
  `binary` values; mask bounding rectangles (empty, full, single pixel, corners, sparse, ROI, empty
  ROI); `matchShapes` for all three methods with contour, float and image pairs, including the
  `DBL_MAX` case.
- Every double is compared by bits. There is no tolerance and there are no suspect cases: all 1245
  cases match cv2 5.0.0.
- Float images never have width 2 (see the deliberate differences). NaN coordinates are not in the
  battery.

## Tests

- `ShapeDescriptorsGoldenTests`: 1245 golden cases plus a file check.
- `ShapeDescriptorsBasicTests`: analytic square, empty inputs, validation, ROI with stride, source
  not modified, the vector paths against scalar references at widths that are not multiples of the
  vector size, and a large-image test that compares 1 and 8 threads for 8-bit, float and binary
  image moments and the mask bounding rectangle.
