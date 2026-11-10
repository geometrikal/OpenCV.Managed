# Lens distortion and camera model port notes

Notes for the port of the OpenCV 5.0 lens distortion functions (`Calib/Undistortion`) and the
pinhole camera model functions (`Calib/CameraModel`). File references are against the OpenCV
5.0.0 submodule at `src/OpenCV`.

## Sources

| Port | OpenCV function | OpenCV source |
|---|---|---|
| `Undistortion.InitUndistortRectifyMap*`, `Internal/UndistortMapKernel` | `initUndistortRectifyMap` | `modules/imgproc/src/undistort.dispatch.cpp`, `undistort.simd.hpp` (`initUndistortRectifyMapComputer`) |
| `Undistortion.Undistort` | `undistort` | `undistort.dispatch.cpp` |
| `Undistortion.UndistortPoints`, `UndistortImagePoints`, `Internal/UndistortPointsKernel` | `undistortPoints`, `undistortImagePoints` | `modules/geometry/src/pinhole.cpp` (`undistortPointsInternal`) |
| `Undistortion.GetDefaultNewCameraMatrix` | `getDefaultNewCameraMatrix` | `pinhole.cpp` |
| `Undistortion.GetOptimalNewCameraMatrix` | `getOptimalNewCameraMatrix`, `getUndistortRectangles` | `modules/geometry/src/calibration_base.cpp` |
| `CameraModel.Rodrigues*`, `Internal/CameraRodrigues` | `Rodrigues` | `calibration_base.cpp` |
| `CameraModel.ProjectPoints`, `Internal/CameraProjection` | `projectPoints` (both overloads, jacobian without `dpdo`) | `calibration_base.cpp` |
| `Internal/CameraDistortion` | `computeTiltProjectionMatrix`, `Matx` products | `modules/geometry/src/distortion_model.hpp`, `modules/core/include/opencv2/core/matx.inl.hpp` |
| `CameraModel.SolvePnP`, `SolvePnPRefine`, `Internal/CameraPnP` | `solvePnP` with `SOLVEPNP_ITERATIVE`, `findExtrinsicCameraParams2`, `findHomography` method 0 | `modules/geometry/src/solvepnp.cpp`, `calibration_base.cpp`, `fundam.cpp` (`HomographyEstimatorCallback::runKernel`, refinement) |
| `Internal/CameraLevMarq` | `LevMarq` (dense backend, long callback, no mask, SVD solver) | `modules/geometry/src/levmarq.cpp` |

`Core/TermCriteria` (`cv::TermCriteria`) is shared with the geometric estimation port.

## Public API

```csharp
(Mat<float> MapX, Mat<float> MapY) Undistortion.InitUndistortRectifyMap(Mat<double> cameraMatrix, ReadOnlySpan<double> distCoeffs, Mat<double>? r, Mat<double>? newCameraMatrix, Size size)
Mat<float> Undistortion.InitUndistortRectifyMapInterleaved(...)            // CV_32FC2
(Mat<short> Xy, Mat<ushort> Fractions) Undistortion.InitUndistortRectifyMapFixedPoint(...) // CV_16SC2 + CV_16UC1
Mat<T> Undistortion.Undistort<T>(Mat<T> source, Mat<double> cameraMatrix, ReadOnlySpan<double> distCoeffs, Mat<double>? newCameraMatrix = null)
void / Point2f[] / Point2d[] Undistortion.UndistortPoints(points, [destination,] cameraMatrix, distCoeffs, r = null, p = null, TermCriteria? criteria = null)
Point2f[] / Point2d[] Undistortion.UndistortImagePoints(points, cameraMatrix, distCoeffs, TermCriteria? criteria = null)
Mat<double> Undistortion.GetDefaultNewCameraMatrix(Mat<double> cameraMatrix, Size imageSize = default, bool centerPrincipalPoint = false)
Mat<double> Undistortion.GetOptimalNewCameraMatrix(cameraMatrix, distCoeffs, Size imageSize, double alpha, Size newImageSize, out Rect validPixRoi, bool centerPrincipalPoint = false)
Mat<T> CameraModel.Rodrigues<T>(Mat<T> source[, out Mat<T> jacobian])   // T = float or double
Mat<double> CameraModel.Rodrigues(Point3d rotationVector)
Point3d CameraModel.RodriguesVector(Mat<double> rotationMatrix)
Point2d[] / Point2f[] CameraModel.ProjectPoints(objectPoints, Point3d rvec | Mat<double> rotationMatrix, Point3d tvec, cameraMatrix, distCoeffs, [out Mat<double> jacobian,] double aspectRatio = 0)
bool CameraModel.SolvePnP(ReadOnlySpan<Point3d> objectPoints, ReadOnlySpan<Point2d> imagePoints, cameraMatrix, distCoeffs, out Point3d rvec, out Point3d tvec)
bool CameraModel.SolvePnPRefine(objectPoints, imagePoints, cameraMatrix, distCoeffs, ref Point3d rvec, ref Point3d tvec) // useExtrinsicGuess = true
```

Camera matrices, R and P are `Mat<double>`. Distortion coefficients are a span of 0, 4, 5, 8, 12
or 14 values `(k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, τx, τy)`.

## OpenCV 5.0 behaviour

- `initUndistortRectifyMap` without a new camera matrix uses `getDefaultNewCameraMatrix(A, size,
  true)`: the principal point moves to the image centre. `undistort` without a new camera matrix
  uses A unchanged.
- `iR = (Ar·R)⁻¹` uses `gemm` and `invert(DECOMP_LU)`. The port calls `LinearAlgebra.Gemm` and
  `LinearAlgebra.Invert`, which reproduce these bit-exactly.
- `undistort` processes stripes of `min(max(1, 4096 / cols), rows)` rows. For each stripe it sets
  `Ar(1, 2) = v0 − y`, builds `CV_16SC2` maps for the stripe and calls `remap` with
  `INTER_LINEAR` and `BORDER_CONSTANT` (zero). Because `iR` is computed per stripe, the result is
  not always equal to one full-image map followed by one remap. The port keeps the stripe
  structure; independent stripe groups run in parallel through `ParallelRows.ForEachSerial`.
- `undistortPoints`: the default criteria are `MAX_ITER`, 5 iterations (`epsilon` 0.01 unused).
  With `EPS` the error is the reprojection distance in pixels; when the error increases, the
  damping factor is halved and the step is rejected. A negative `icdist` stops the iteration
  and returns the normalized point without distortion correction. An `EPS`-only criterion can iterate without
  end in OpenCV when the error does not fall below epsilon; the port has the same behaviour.
- `undistortImagePoints` is `undistortPoints` with no R and P = camera matrix.
- `getOptimalNewCameraMatrix` undistorts the 32 border points of a 9 × 9 grid with the default
  criteria. The ROI conversion from `Rect_<double>` uses `cvRound` (non-centred case) or
  `cvCeil`/`cvFloor` (centred case), then OpenCV 5.0 `Rect &=` (overflow-safe intersection).
- `Rodrigues` (matrix to vector) returns zero (and a zero jacobian) when an element is outside
  [−100, 100) or NaN (`checkRange`). The matrix is replaced by `U·Vᵀ` of its Jacobi SVD
  (`Svd<double>`; cv2 uses the Jacobi SVD for 3 × 3 matrices).
- `projectPoints` uses the `double` loop for `float` and `double` points; the HAL entries
  `cv_hal_project_points_pinhole32f/64f` are not implemented in OpenCV 5.0. `float` results are
  the `double` results rounded to `float`. A 3 × 3 rotation is used unchanged; its `∂R/∂r` comes
  from `Rodrigues(Rodrigues(R))`. The jacobian of the Python-style overload has
  `10 + n` columns; an empty coefficient list counts as 5 zero coefficients.

## solvePnP (SOLVEPNP_ITERATIVE)

- The image points are undistorted with `undistortPoints` (default criteria, P = I).
- Without a guess, the object points are centred and the SVD of their scatter matrix decides the
  initialization: `W[2]/W[1] < 1e-3` is a planar object. The planar case maps the points to the
  plane with `V` (or I), runs `findHomography` with method 0 on `float` copies of the points
  (normalized DLT from the eigenvector of the smallest eigenvalue, then 10 Levenberg-Marquardt
  iterations with geodesic acceleration when there are more than 4 points), and builds R and t
  from the normalized homography columns. The non-planar case needs at least 6 points and uses
  the DLT of `findExtrinsicCameraParams2`.
- The refinement is `LevMarq` with 20 iterations and geodesic acceleration; the callback is
  `projectPoints` with `∂p/∂r` and `∂p/∂t`.
- The result equals cv2 bit for bit on the golden battery (48 cases). This needs the OpenCV
  evaluation order of the `MatExpr` expressions in `levmarq.cpp`:
  - `(jtbv + jtb·(h − 1) + lmDiag.mul(x, h)) / h²` is `addWeighted(m1, s, m2, s, 0)` with
    `s = 1/h²`, `m1 = addWeighted(jtbv, 1, jtb, h − 1, 0)` and `m2 = (lmDiag·h)·x`.
  - `x.mul(jtb − lmDiag.mul(x))` with `cv::sum`, and `norm(·, NORM_L2SQR)` through
    `Reductions`.
  - `norm(_RR, NORM_L2)` of the non-continuous 3 × 3 DLT block sums one `NormL2_SIMD` call per
    row; `norm(matR, NORM_L2)` of a continuous 3 × 3 matrix uses the 2-lane SSE2 block of 8.
- `solvePnPRansac` is not ported. It needs the EPnP kernel (`epnp.cpp`), P3P for 4 points and
  the RANSAC point-set registrator of the geometric estimation port.

## Bit-exactness

The reference build is cv2 5.0.0 with baseline SSE3 and the AVX/AVX2/FMA3/AVX-512 dispatch
disabled. The map computer then runs the 128-bit `v_float64` body (2 lanes, 4 pixels per
iteration, no FMA):

- Blocks of four pixels: the block base `_x, _y, _w` starts at `i·ir[1] + ir[2]` (and so on) and
  advances by `4·ir[0]` per block; lane `l` uses `_x + ir[0]·l`. The arithmetic follows the
  `v_muladd` order: `kr = (((k3·r2 + k2)·r2 + k1)·r2 + 1) / (((k6·r2 + k5)·r2 + k4)·r2 + 1)`,
  `xd = ((2x²+r2)·p2 + x·kr) + p1·(x·y·2)`, `xd = s1·r2 + (s2·r2² + xd)`, the tilt rows
  `t0·xd + (t1·yd + t2)`, and `u = (fx·iz)·tx + u0`.
- The last `width mod 4` pixels of a row continue the base recurrence with steps of `ir[0]` and
  use the scalar order: `x·kr + p1·2xy + p2·(r2 + 2x²) + s1·r2 + s2·r2·r2`, and `Matx33d · Vec3d`
  products that start from 0.
- `CV_16SC2`: `cvRound(c·32)`. In blocks, `v_pack` saturates the integer parts to 16 bits; in the
  tail, `(short)` truncates them. The fraction index is the same in both.
- The managed kernel fills the block coordinates per row, runs the block arithmetic over
  `Vector<double>` lanes (each lane in the OpenCV order; the JIT does not contract multiply-add),
  narrows to `float` with `Vector.Narrow`, and runs the tail with scalar code. Rows run in
  parallel; no state crosses rows, so the output does not depend on the thread count.
- `computeTiltProjectionMatrix` uses `Math.Cos`/`Math.Sin`, which call the same CRT functions as
  the MSVC build of cv2.
- `undistortPoints`, `getOptimalNewCameraMatrix`, `Rodrigues` and `projectPoints` are scalar
  `double` code in the OpenCV expression order. The `EPS` error uses `Math.Pow(d, 2)` as OpenCV
  `std::pow`. `P·R` uses `LinearAlgebra.Gemm`.

## Deliberate differences and limits

- Only `double` camera, R and P matrices are accepted (OpenCV converts other types to `double`).
- `ProjectPoints` takes `double` rotation and translation vectors; OpenCV also accepts `float`
  vectors and converts them to `double` (the same values).
- Not ported: `initInverseRectificationMap`, `initWideAngleProjMap`, the fisheye model,
  `projectPoints` with `dpdo`, `solvePnP` methods other than `SOLVEPNP_ITERATIVE`,
  `solvePnPRansac`, `solvePnPGeneric` with the reprojection error output.

## Golden constraints

- Generator: `Geometrikal.OpenCV.Managed.Tests/tools/generate_undistort_golden.py`, output
  `TestData/undistort_golden.json` (975 cases, no suspect cases). The generator stops when cv2 is
  not 5.0.0 or when AVX2/AVX-512 dispatch is active.
- Camera: DJI FC6310 (5472 × 3648, 8.8 mm focal length on a 13.2 mm sensor) scaled to
  37 × 23, 101 × 64, 257 × 171 and 684 × 456, with an off-centre principal point.
- Coefficient sets: none, 4, 5, 8 (rational), 12 (thin prism), 14 (tilt), and an extreme 5-value
  set whose coordinates leave the 16-bit range (saturation and wrap rules).
- Maps and images above 4096 elements are compared by SHA-256; large inputs are stored once in
  `shared`.
- Runners: `Tests/Calib/UndistortGoldenTests` (maps, undistort, points, new camera matrices) and
  `Tests/Calib/CameraModelGoldenTests` (Rodrigues with jacobians, projectPoints with jacobians,
  solvePnP planar and non-planar, with and without distortion, noise and an extrinsic guess).
  `Tests/Calib/UndistortBasicTests` covers ROIs, invalid inputs, round trips and thread-count
  independence. Replacing the block path by the scalar path fails 10 golden map cases.

## Performance

- Map computation: one pass per row; per pixel one `double` division for `1/w`, one for `kr`
  and one for `1/z` in `Vector<double>` lanes; parallel rows (`ParallelRows.For`).
- `Undistort`: stripe groups in parallel; each group allocates one stripe of maps and calls the
  existing fixed-point `Geometry.Remap` per stripe.
- `UndistortPoints`: points run in parallel stripes for large inputs (the input is copied into a
  pooled buffer when it runs in parallel).
- Benchmarks: `UndistortBenchmarks` and `OpenCvSharpUndistortBenchmarks` (map 32FC1 and 16SC2,
  `Undistort` U8C3, `UndistortPoints` 10 000 points; sizes 256 and 1024).

### Measured optimisation (October 2026)

Changes:

- The fixed-point store (16SC2) converts four pixels per instruction with AVX `cvtpd2dq`. This
  instruction rounds half to even and returns `int.MinValue` out of range, as `CvRound`.
  `packs_epi32` saturates the integer parts as the OpenCV block code does. Before the change,
  the scalar store took about 70% of the 16SC2 time.
- With `Vector256`, the complete blocks of a row are computed in registers. Lane `l` reads
  `x0 + s[l]`, as the scalar block setup does. This path does not use the `bx`/`by`/`bw` row
  buffers.

Results: Stopwatch harness, minimum of 30 batches. The `p4-integration` baseline and the new
build ran back to back, each with OpenCvSharp 4.13 in the same process. Times in µs.

| Case | Threads | Before | After | OpenCvSharp |
|---|---|---|---|---|
| InitUndistortRectifyMap_32FC1 256 | 1 | 298 | 268 | 182 |
| InitUndistortRectifyMap_32FC1 1024 | 1 | 4460 | 4116 | 3871 |
| InitUndistortRectifyMap_16SC2 256 | 1 | 590 | 378 | 224 |
| InitUndistortRectifyMap_16SC2 1024 | 1 | 8165 | 4103 | 3980 |
| Undistort_U8C3 256 | 1 | 1073 | 741 | 574 |
| Undistort_U8C3 1024 | 1 | 13965 | 10272 | 8942 |
| InitUndistortRectifyMap_32FC1 256 | 16 | 124 | 156 | 58 |
| InitUndistortRectifyMap_32FC1 1024 | 16 | 1330 | 1164 | 1571 |
| InitUndistortRectifyMap_16SC2 256 | 16 | 244 | 211 | 53 |
| InitUndistortRectifyMap_16SC2 1024 | 16 | 2040 | 1177 | 1412 |
| Undistort_U8C3 256 | 16 | 308 | 284 | 918 |
| Undistort_U8C3 1024 | 16 | 4052 | 3150 | 16139 |

Remaining gap at 256 × 256: each output map is a new array of 128 KiB or more. These arrays go
to the large object heap. BenchmarkDotNet records a gen 2 collection on about one call in
seven. A smaller work estimate (`8·width` in place of `48·width`, so fewer stripes) did not
change the 16-thread time, so stripe overhead is not the cause. The maps are public managed
outputs, so the port cannot pool them.

## Pooled temporaries (option 1)

- `undistort` rents the fixed-point map pair of each parallel chunk. The number of chunks is
  `4 · threads`, so the saving grows with the thread count.

Allocated bytes per call, BenchmarkDotNet `--fast --inProcess`, 1 thread.

| Case | Size | Before | After |
|---|---|---|---|
| Undistort U8C3 | 256 | 319.5 KB | 223.2 KB |
| Undistort U8C3 | 1024 | 3646.3 KB | 3550.3 KB |

The result image is 3072 KB at 1024. The remaining temporaries are the per-stripe
`UndistortMapParams` (gemm, invert and small arrays, one set per stripe of `4096 / width` rows)
and the per-stripe remap objects. `initUndistortRectifyMap` and `undistortPoints` allocate only
their results.
