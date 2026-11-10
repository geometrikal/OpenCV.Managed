# Sparse optical flow port notes

Notes for the port of `buildOpticalFlowPyramid` and `calcOpticalFlowPyrLK` (`Video/`). File
references are against the OpenCV 5.0.0 submodule at `src/OpenCV`.

## Sources

| Port | OpenCV source |
|---|---|
| `OpticalFlow.BuildOpticalFlowPyramid`; `Internal/LkPyramidBuilder` | `modules/video/src/lkpyramid.cpp` (`buildOpticalFlowPyramid`) |
| `Internal/LkScharrDeriv` | `modules/video/src/lkpyramid.cpp` (`calcScharrDeriv`), `lkpyramid.simd.hpp` (`ScharrDerivInvoker_SIMD`) |
| `OpticalFlow.CalcOpticalFlowPyrLK`; `Internal/LkTracker`, `Internal/LkSse` | `modules/video/src/lkpyramid.cpp` (`SparsePyrLKOpticalFlowImpl::calc`, `LKTrackerInvoker::operator()`), `lkpyramid.hpp` |
| `Internal/LkBorders` | `modules/core/src/copy.cpp` (`copyMakeBorder` submatrix rule), `modules/core/src/matrix.cpp` (`Mat::locateROI`) |
| Vector semantics | `modules/core/include/opencv2/core/hal/intrin_sse.hpp` (`v_dotprod`, `v_zip`, `v_pack`, `v_interleave_pairs`, `v_recombine`, `v_reduce_sum`) |

## Public API

```csharp
OpticalFlowPyramid OpticalFlow.BuildOpticalFlowPyramid(Mat<byte> image, Size winSize, int maxLevel,
    bool withDerivatives = true, BorderType pyrBorder = BorderType.Reflect101,
    BorderType derivBorder = BorderType.Constant, bool tryReuseInputImage = true)
void OpticalFlow.CalcOpticalFlowPyrLK(Mat<byte> prevImg, Mat<byte> nextImg, Point2f[] prevPts,
    Point2f[] nextPts, byte[] status, float[]? err, Size winSize, int maxLevel, TermCriteria criteria,
    OpticalFlowFlags flags = None, double minEigThreshold = 1e-4)
void OpticalFlow.CalcOpticalFlowPyrLK(OpticalFlowPyramid prevPyramid, OpticalFlowPyramid nextPyramid, ...same...)
OpticalFlowResult OpticalFlow.CalcOpticalFlowPyrLK(Mat<byte> prevImg, Mat<byte> nextImg, Point2f[] prevPts,
    Point2f[]? initialNextPts = null, OpticalFlowFlags flags = None, double minEigThreshold = 1e-4)
OpticalFlowResult OpticalFlow.CalcOpticalFlowPyrLK(Mat<byte> prevImg, Mat<byte> nextImg, Point2f[] prevPts,
    Point2f[]? initialNextPts, Size winSize, int maxLevel, TermCriteria criteria, OpticalFlowFlags flags = None,
    double minEigThreshold = 1e-4)
```

- `OpticalFlowFlags`: `UseInitialFlow` (4), `LkGetMinEigenvals` (8), OpenCV values.
- `OpticalFlowPyramid`: `Images` (`Mat<byte>[]`), `Derivatives` (`Mat<short>[]` with 2·cn
  channels, or null), `WinSize`, `MaxLevel`. OpenCV returns one `vector<Mat>` with images and
  derivatives interleaved; the port keeps two arrays.
- `OpticalFlowResult`: `NextPoints`, `Status`, `Error`.
- `OpticalFlow.DefaultCriteria`: `(MaxIter | Eps, 30, 0.01)`.

## Pyramid layout

- Every level is a view into a buffer padded by `winSize` on each side. The tracker reads the
  padding: a window can start at `-winSize` and the bilinear neighbour can be one pixel further.
- Level 0 is the input itself when `tryReuseInputImage` is true and the input is a submatrix
  with at least `winSize` pixels of its array around it. Otherwise level 0 is a padded copy.
- Level 0 copy: `copyMakeBorder` without `BORDER_ISOLATED`. When the input is a submatrix, the
  pixels of its array replace border pixels where they exist (`LkBorders.CopyMakeBorder`).
- Levels above 0: `Pyramids.PyrDown` from the previous level view (the source is isolated;
  OpenCV pyrDown does not read outside the view), then the padding is filled with
  `pyrBorder | BORDER_ISOLATED`.
- Derivatives: `calcScharrDeriv` on the level view, then the padding is filled with
  `derivBorder | BORDER_ISOLATED`. `calcOpticalFlowPyrLK` with images computes the derivatives
  per level with a constant 0 border.
- The pyramid stops after level `L` when `((size + 1) / 2)` is not larger than the window in
  either direction. `MaxLevel` is then `L` (the OpenCV return value).
- `BorderType.Transparent` leaves the padding unchanged. The port allocates cleared buffers, so
  the padding is 0. OpenCV leaves it uninitialised.
- `LkBorders.FillBorder` supports any channel count. `Borders.CopyMakeBorder` supports at most 4
  channels and cannot fill the 6- and 8-channel derivative buffers.

## Submatrix rule

`Mat<T>` has no parent matrix. `LkBorders.LocateRoi` applies the OpenCV `Mat::locateROI`
formula with the start and end of the backing array as `datastart` and `dataend`. A matrix is a
submatrix when the formula gives a non-zero offset or a size larger than the matrix. A matrix that
wraps an array with row padding (stride larger than the row) is therefore a submatrix, and the
row padding counts as parent pixels. OpenCV sets `SUBMATRIX_FLAG` only for views made from a
`Mat`; this is a deliberate difference for wrapped arrays with row padding.

## Scharr derivatives

- Vertical pass: `t0 = (s0 + s2)·3 + s1·10`, `t1 = s2 − s0` per element as 16-bit values. Row −1
  is row 1 and row `rows` is row `rows − 2` (reflect-101; one row uses itself).
- Horizontal pass: `dx = t0[x + cn] − t0[x − cn]`, `dy = (t1[x + cn] + t1[x − cn])·3 + t1[x]·10`,
  interleaved as `(dx, dy)` per element. Column −1 is column 1, column `cols` is column
  `cols − 2`.
- The OpenCV vector and scalar loops give the same values (16-bit wrapping and an int result
  cast to `short` agree). The port vectorises the vertical pass with `Vector<short>` and the
  horizontal pass with SSE2 (8 elements, interleaved by `UnpackLow`/`UnpackHigh`); a scalar loop
  does the tail. Rows run in parallel stripes.

## Tracker arithmetic and bit-exactness

The port reproduces the OpenCV `CV_SIMD128 && !CV_NEON` path compiled for the SSE baseline
(no FMA). `LkSse` uses the same SSE2 instructions. `LkTracker` also contains a scalar emulation of
the same lanes (`ForceScalar`, used on processors without SSE2 and by a test).

- Weights: `cvRound((1 − a)(1 − b)·2^14)` and the other products in float, round half to even;
  `iw11 = 2^14 − iw00 − iw01 − iw10`.
- Window values: `(Σ w·I + 2^8) >> 9` (5 fractional bits); derivatives `(Σ w·dI + 2^13) >> 14`.
  The integer sums are exact. The vector part saturates to `short` (`v_pack`); the scalar part
  casts. The values agree for derivatives from `calcScharrDeriv`.
- Window rows split into the vector part `[0, 8·⌊n/8⌋)` and a scalar tail, with `n = width·cn`.
  - Gradient matrix: lane `k` of the 4-lane accumulators receives the elements with
    `x mod 4 = k` of the vector part, in row order. The product is `(float)ix · (float)iy`. The
    tail adds `(float)(ix·iy)` to a scalar sum. Final value: `tail + ((l0 + l2) + (l1 + l3))`
    (`v_reduce_sum` order).
  - Mismatch vector: `v_dotprod` adds the int products of elements `k` and `k + 4` of each block
    before the conversion to float. Accumulator `qb0` holds offsets 0 and 1, `qb1` offsets 2 and
    3, each as (Ix, Iy) lane pairs. Final value: `tail + ((qb0[0] + qb1[0]) + (qb0[2] + qb1[2]))`
    for `b1` and the odd lanes for `b2`.
- `A = sum · 2^−20`; `D = A11·A22 − A12·A12`; the minimum eigenvalue is
  `(A22 + A11 − sqrt((A11 − A22)² + 4·A12·A12)) / (2·w·h)`, all float.
- Step: `δ = ((A12·b2 − A22·b1)·D⁻¹, (A12·b1 − A11·b2)·D⁻¹)` in float. Stop when
  `(double)δx² + (double)δy² ≤ ε²`, or when the step reverses (`|δ + δprev| < 0.01` in both
  directions); the reversal subtracts `δ/2`.
- Error: `Σ |diff|` as a float running sum in row order, then `· 1 / (32·w·cn·h)`.
- Criteria: the count is clamped to [0, 100] (30 without `MaxIter`), epsilon to [0, 10] (0.01
  without `Eps`), then squared.

## Status and error semantics

- `status` is 1 at the start. Only level 0 sets it to 0: point window outside the image,
  minimum eigenvalue below the threshold or `D < FLT_EPSILON`, iterate outside the image, or the
  final error window outside the image.
- At a level above 0 a rejected point keeps its scaled estimate and continues at the next level.
- `err` with `LkGetMinEigenvals`: the minimum eigenvalue of the last level, or 0 when the point
  window is outside the image at level 0. Without the flag: the error for tracked points, 0 for a
  point window outside the image, and unchanged for other rejected points. OpenCV leaves these
  entries uninitialised; the allocating overloads return 0.

## Threads

Points run in parallel stripes (`ParallelRows.For`). Each point is independent, so the output
does not depend on the thread count. Each stripe allocates its own window buffers.

## Deliberate differences

- With pyramid inputs, the port checks the padding of every level and derivative against
  `winSize`. OpenCV checks only level 1. Valid OpenCV inputs pass both checks.
- The port does not implement the OpenCL path or the `cv_hal_LKOpticalFlowLevel` and
  `cv_hal_ScharrDeriv` HAL hooks (no HAL in the cv2 reference).
- The NEON path (`acctype` int64 on ARM without NEON, and the NEON loops) is not ported. On
  ARM the port gives the x86 results.
- `SparsePyrLKOpticalFlow` (the algorithm object) is not ported; the static functions cover the
  same computation.

## Golden constraints

- Generator: `Geometrikal.OpenCV.Managed.Tests/tools/generate_lk_golden.py`, cv2 5.0.0 with
  `OPENCV_CPU_DISABLE` (no AVX/AVX2/FMA3/AVX-512), IPP and OpenCL off, one thread. Output:
  `TestData/lk_golden.json` (566 cases: 198 pyramid, 368 tracking; 32 738 points, 30 521
  tracked). No case is suspect.
- Inputs: textures (blurred noise plus sinusoids, 1, 2, 3 and 4 channels) and the same texture
  moved by a sub-pixel shift, a rotation, or a large shift with noise; a flat image. Points from
  `goodFeaturesToTrack`, a grid, border and outside positions. All flags, thresholds, window sizes
  3 to 21 (non-square included), levels 0 to 5, and criteria with clamping.
- Python cannot return the pyramid padding, so the pyramid cases compare the level views. The
  tracking cases cover the padding indirectly through border points.
- `err` is compared where cv2 writes it (see above). Points are compared as float bit patterns.
- `OpticalFlowGoldenTests.OpticalFlow_ScalarLaneEmulation_MatchesOpenCvGolden` replays the
  tracking cases with the scalar lane emulation.
- cv2 cannot produce a submatrix input (NumPy arrays have no parent), so the submatrix border
  rule and `tryReuseInputImage` are covered by `OpticalFlowBasicTests`.

## Performance

### Changes

- Tracker: `LkSse.Patch` and `LkSse.Diff` process the whole window in one call. The float
  accumulators stay in registers, the weight vectors are built once per call, and the scalar
  tail `[nv, n)` reads through `ref` offsets without bounds checks. The tail sums and the lane
  sums are kept separate and added in the previous order, so the results do not change.
  `LkTracker.PatchScalar` and `LkTracker.DiffScalar` keep the scalar lane emulation.
- Tracker: `Range`, `ErrorSum`, `Patch` and `Diff` use `MethodImplOptions.AggressiveOptimization`.
  The first calls do not run tier-0 or OSR code.
- Scharr derivatives: the horizontal pass uses SSE2 (8 elements per step, interleaved with
  `UnpackLow`/`UnpackHigh`). The pass was scalar with bounds checks.
- `calcOpticalFlowPyrLK` with image inputs rents the temporary pyramid levels and derivatives
  from `ArrayPool` (`PooledMat`) and returns them at the end of the call. Allocation per call at
  1024 × 1024 decreased from 9.2 MB to 73 KB. A rented buffer has a larger backing array, so
  `LkBorders.LocateRoi` is not used on it.

### Results

BenchmarkDotNet `--fast --inProcess`, window 21 × 21, 3 levels, 500 grid points. Values in µs.
"Before" is the measurement before the changes on a quiet machine. "After" is the final run. The
port and the OpenCvSharp 4.13 twin ran back to back in separate processes, because OpenCV worker
threads that run before the port in the same process slow the port down at 16 threads.

| Case | Threads | Size | Before | After | OpenCvSharp |
|---|---|---|---|---|---|
| CalcOpticalFlowPyrLK_500 | 1 | 256 | 19 053 | 8 293 | 6 351 |
| CalcOpticalFlowPyrLK_500 | 1 | 1024 | 20 636 | 8 780 | 8 431 |
| CalcOpticalFlowPyrLK_500 | 16 | 256 | 6 517 | 3 667 | 1 425 |
| CalcOpticalFlowPyrLK_500 | 16 | 1024 | 10 661 | 3 643 | 3 711 |
| BuildOpticalFlowPyramid_U8 | 1 | 256 | 427 | 284 | 98 |
| BuildOpticalFlowPyramid_U8 | 1 | 1024 | 5 977 | 1 664 | 1 992 |
| BuildOpticalFlowPyramid_U8 | 16 | 256 | 426 | 555 | 125 |
| BuildOpticalFlowPyramid_U8 | 16 | 1024 | 4 274 | 1 876 | 1 849 |

The machine was shared during the final runs. The 16-thread 256 build value varied between
runs by more than its difference to the "before" value.

### Remaining gaps

- `BuildOpticalFlowPyramid` at 256 × 256: each call allocates 667 KB of new padded levels and
  derivatives on the large object heap. The returned pyramid owns these buffers, so they cannot
  be pooled. A Stopwatch profile shows that the first write to new pages costs more than the
  pixel work: a padded copy of a 256 × 256 image takes about 34 µs, and one level of derivatives
  takes about 85 µs. OpenCV reuses freed heap memory and does not pay this cost.
- `calcOpticalFlowPyrLK` at 256 × 256 with 16 threads: the fixed cost (pyramids and
  derivatives) is about 0.3 ms. The rest is the tracker, which scales about 2.2 times on this
  input. Smaller chunks of points with dynamic scheduling did not change the time and were not
  kept.

## Pooled temporaries (option 1)

Temporary buffers come from `ArrayPool<T>.Shared`. The returned pyramid levels, derivatives,
points, status and error arrays stay new allocations.

- `LkTracker`: the window buffers (`iwin`, `dwin`) and the scaled points are rented once per
  range of points. `RangeCore` keeps `AggressiveOptimization`; the rental and its
  `try`/`finally` are in the caller, so the hot method has no exception region.
- `LkScharrDeriv`: the two row buffers are rented once per stripe.
- `LkBorders.FillBorder`: the column map is rented once per call.
- `calcOpticalFlowPyrLK` already rented its temporary pyramids and derivatives
  (`LkPyramidBuilder` with a rental list). No temporary pyramid level is a new allocation.

Allocated bytes per call (BenchmarkDotNet `--fast --inProcess`, 1 thread):

| Case | Size | Before | After |
|---|---|---|---|
| `CalcOpticalFlowPyrLK_500` | 256 | 50.8 KB | 14.7 KB |
| `CalcOpticalFlowPyrLK_500` | 1024 | 73.2 KB | 14.7 KB |
| `BuildOpticalFlowPyramid_U8` | 256 | 668.1 KB | 660.9 KB |
| `BuildOpticalFlowPyramid_U8` | 1024 | 7653.9 KB | 7631.5 KB |

`CalcOpticalFlowPyrLK_500` returns 6.5 KB of arrays; the remainder is the rental list, the
pyramid objects, the level records and the delegates. `BuildOpticalFlowPyramid` returns the
padded levels and derivatives, which are almost all of its allocation.
