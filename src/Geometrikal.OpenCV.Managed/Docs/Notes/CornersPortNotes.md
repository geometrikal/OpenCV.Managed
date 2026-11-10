# Corner detector port notes

Notes for the port of `cornerMinEigenVal`, `cornerHarris`, `cornerEigenValsAndVecs`,
`goodFeaturesToTrack` and `cornerSubPix` (`ImgProc/Corners/`). File references are against the
OpenCV 5.0.0 submodule at `src/OpenCV`.

## Sources

| Port | OpenCV source |
|---|---|
| `Corners.CornerMinEigenVal`, `CornerHarris`, `CornerEigenValsAndVecs`; `Internal/CornerEngine` | `modules/imgproc/src/corner.cpp` (`cornerEigenValsVecs`, `calcMinEigenVal`, `calcHarris`, `eigen2x2`, `calcEigenValsVecs`) |
| `Corners.GoodFeaturesToTrack`, `GoodFeaturesToTrackWithQuality`; `Internal/GoodFeatures` | `modules/features/src/featureselect.cpp` (`goodFeaturesToTrack`, `greaterThanPtr`). In OpenCV 5.0 the function is in the `features` module; `gftt.cpp` holds only the `GFTTDetector` wrapper. |
| `Corners.CornerSubPix`; `Internal/SubPixRefiner` | `modules/imgproc/src/cornersubpix.cpp` |
| `Internal/RectSubPix` | `modules/imgproc/src/samplers.cpp` (`adjustRect`, `getRectSubPix_Cn_`, `getRectSubPix_8u32f`) |
| `Core/TermCriteria`, `TermCriteriaType` | `modules/core/include/opencv2/core/types.hpp` |

## Public API

```csharp
Mat<float> Corners.CornerMinEigenVal<T>(Mat<T> source, int blockSize, int ksize = 3, BorderType borderType = BorderType.Reflect101)
Mat<float> Corners.CornerHarris<T>(Mat<T> source, int blockSize, int ksize, double k, BorderType borderType = BorderType.Reflect101)
Mat<float> Corners.CornerEigenValsAndVecs<T>(Mat<T> source, int blockSize, int ksize, BorderType borderType = BorderType.Reflect101)
Point2f[] Corners.GoodFeaturesToTrack<T>(Mat<T> image, int maxCorners, double qualityLevel, double minDistance,
    Mat<byte>? mask = null, int blockSize = 3, int gradientSize = 3, bool useHarrisDetector = false, double k = 0.04)
(Point2f[] Corners, float[] Quality) Corners.GoodFeaturesToTrackWithQuality<T>(...same arguments...)
void Corners.CornerSubPix<T>(Mat<T> image, Span<Point2f> corners, Size winSize, Size zeroZone, TermCriteria criteria)
```

`T` is `byte` or `float`, one channel. `ksize` is 1, 3, 5, 7 (Sobel) or −1 (Scharr).
`CornerEigenValsAndVecs` returns 6 channels `(λ1, λ2, x1, y1, x2, y2)`.

## Arithmetic and bit-exactness

The port follows the OpenCV 5.0 SSE3 baseline with IPP disabled. The AVX line functions of
`corner.avx.cpp` use the same formulas as the 128-bit loops, so they do not change the result.

- Gradients: `Derivatives.Sobel`/`Scharr` to float with
  `scale = 1 / (2^(ksize−1) · blockSize [· 2 for Scharr] [· 255 for byte])`, computed in double
  as OpenCV, with the user border type.
- Covariance: `(dx·dx, dx·dy, dy·dy)` in float, then `Filters.BoxFilter` (3 channels,
  `normalize = false`, the user border type). The float box filter uses double running sums, as
  OpenCV.
- `calcMinEigenVal`: `(a + c) − sqrt((a − c)² + b²)` with `a`, `c` halved, all float. The 4-lane
  vector loop and the scalar loop give the same values (`b·b + t·t` equals `t·t + b·b`).
- `calcHarris`: the vector loop computes `(a·c − b·b) − (k·(a + c))·(a + c)` in float with
  `(float)k`. The scalar loop computes `k·(a + c)·(a + c)` in double and subtracts it from the
  float `a·c − b·b`. OpenCV treats the continuous output as one row of `width·height` elements,
  so only the last `(width·height) mod 4` elements use the scalar formula. The port reproduces
  this split. The split does not depend on AVX dispatch: the AVX loop covers multiples of 8 and
  the 128-bit loop then covers one more block of 4.
- `eigen2x2`: double arithmetic, `DBL_EPSILON` and `FLT_EPSILON` as in OpenCV.
- `goodFeaturesToTrack`: `Reductions.MinMaxLoc` with the mask, `Thresholding.Threshold`
  (`ToZero`, `maxVal·qualityLevel`), `Morphology.Dilate` with the default 3×3 kernel. The
  candidate order is the `greaterThanPtr` order: larger value first, and for equal values the
  larger address first (later in row-major order). This is a total order, so the result does not
  depend on the sort algorithm. The port sorts 64-bit keys (sortable value bits, inverted index)
  with `Array.Sort`. The grid test compares the float `dx² + dy²` with the double
  `minDistance²`; the cell size is `cvRound(minDistance)`.
- `cornerSubPix`: the window weights use `MathF.Exp` (the C runtime `expf`, as MSVC
  `std::exp(float)`). The gradient sums and the 2×2 solve are in double; `err` is the float
  squared step. Patch sampling uses `getRectSubPix` to float: `getRectSubPix_8u32f` (running
  `prev` term with the double factor `(1 − a)/a`) for byte images inside the image, and
  `getRectSubPix_Cn_` with border replication otherwise. `Rect::contains(Point2f)` compares in
  float (OpenCV 4.11 and later do not round the point).

## Deliberate differences

- `BORDER_ISOLATED` does not exist in the port; a region of interest is always an isolated
  image.
- Only the allocating forms exist. OpenCV writes `cornerHarris` into a caller destination; with a
  non-continuous destination the Harris scalar tail is per row. The port output is always
  continuous.
- `preCornerDetect`: five float Sobel derivatives, then `Dx²·Dyy + Dy²·Dxx − 2·Dx·Dy·Dxy` in
  float. Whole 4-element blocks of each row multiply by `(float)factor`; the remaining elements
  multiply the float sum by the `double` factor and round to float. The factor is
  `1/(2^(ksize−1) [·255])³`, so the split changes 8-bit results only. Golden data:
  `tools/generate_precorner_golden.py` → `TestData/precorner_golden.json` (48 cases, all bit-exact);
  float sources are integer-valued because of the small-kernel float Sobel association
  (`FilteringPortNotes.md`, item 2).
- The OpenCL paths and the IPP `cornerHarris` path are not ported. Default
  cv2 builds with IPP enabled can give different Harris values for `ksize` 3 and 5 with
  `Constant` or `Replicate` borders.
- `getRectSubPix` is internal; only the float-patch, one-channel paths that `cornerSubPix` uses
  are ported.

## Performance structure

- The Sobel derivatives and the interleaved covariance use pooled temporaries
  (`PooledMat<float>`); the box filter output is the only allocation before the response
  matrix.
- `goodFeaturesToTrack` applies the `ToZero` quality threshold in place on its response
  matrix.
- The covariance and response loops run in row stripes (`ParallelRows.For`). The Harris split
  point is a global element index, so stripes do not change it.
- `CornersBasicTests.Output_DoesNotDependOnThreadCount` compares 1 and 8 threads for every
  function on a 517×301 image.

## Golden data (`tools/generate_corners_golden.py`)

- cv2 5.0.0, `OPENCV_CPU_DISABLE` set, IPP and OpenCL off, one thread.
- Synthetic images: rectangles and triangles, checkerboards, noise, smoothed noise; byte and
  float (scaled plus small noise). Sizes include `width·height mod 4 ≠ 0` for the Harris tail.
- Responses: block sizes 2, 3, 5; `ksize` 1, 3, 5, 7, −1; borders `Reflect101`, `Replicate`,
  `Reflect`, `Constant`; Harris `k` 0.04 and 0.12. Outputs above 4096 elements are SHA-256.
- `goodFeaturesToTrack`: eight parameter sets per image (limits, quality levels, distances 0 to
  10, masks, Harris, Scharr, block 7), plus a 517×301 case.
- `cornerSubPix`: detected corners plus random offsets, image corners and the centre; four
  window, zero-zone and criteria sets.
- 1004 cases; all match exactly. No case is marked suspect.

## Pooled temporaries (option 1)

- `CornerEngine.Covariance` box-filters the covariance into a pooled 3-channel matrix
  (`BoxFilterEngine.ApplyInto`). The derivative and product matrices were already pooled.
  `cornerMinEigenVal`, `cornerHarris` and `cornerEigenValsAndVecs` allocate only the result.
- `goodFeaturesToTrack` computes the response (`Corners.CornerResponseInto`) and the 3×3
  dilation (`Morphology.DilateInto`) into pooled matrices. The sort keys are a pooled array.
  The distance grid is a pooled linked list (cell heads, next indices, kept corners) instead of
  a `List` per cell. Only the returned corner and quality arrays are allocated.
- `cornerSubPix` rents the window weights and the sample buffer.
- Allocated bytes per call, 1 thread, `CornerBenchmarks`:

| Case | Size | Before | After |
|---|---|---|---|
| CornerHarris_U8 | 256 | 1 MB | 258 KB |
| CornerHarris_U8 | 1024 | 16 MB | 4,100 KB |
| CornerMinEigenVal_U8 | 256 | 1 MB | 258 KB |
| CornerMinEigenVal_U8 | 1024 | 16 MB | 4,100 KB |
| GoodFeaturesToTrack_U8 | 256 | 1.43 MB | 14 KB |
| GoodFeaturesToTrack_U8 | 1024 | 22 MB | 44 KB |
