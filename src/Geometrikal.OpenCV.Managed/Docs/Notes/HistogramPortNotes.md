# Histogram and statistics port notes

Semantics of `Histograms` (OpenCV `calcHist`, `calcBackProject`, `compareHist`, `equalizeHist`),
`Clahe` (`createCLAHE`, `CLAHE::apply`), `IntegralImages` (`integral`) and
`Conversions.Normalize` (`normalize`), the OpenCV 5.0 decisions, the deliberate differences and
the golden constraints. Read this before you change a histogram kernel or the histogram golden
generator.

## Sources

- Port source: OpenCV 5.0:
  - `modules/imgproc/src/histogram.cpp`
  - `modules/imgproc/src/clahe.cpp`
  - `modules/imgproc/src/sumpixels.dispatch.cpp`, `sumpixels.simd.hpp`
  - `modules/core/src/norm.dispatch.cpp` (`normalize`)
- Golden reference: cv2 4.5.5.
- Files:

| File | Content |
|---|---|
| `ImgProc/Histograms/Histograms.cs` | Public API: `CalcHist`, `CalcBackProject`, `CompareHist`, `EqualizeHist` |
| `ImgProc/Histograms/DenseHistogram.cs` | N-dimensional `float` histogram |
| `ImgProc/Histograms/HistCompMethods.cs` | Comparison methods (OpenCV numeric values) |
| `ImgProc/Histograms/Clahe.cs` | Public CLAHE operator |
| `ImgProc/Histograms/Internal/HistLayout.cs` | Plane selection, uniform coefficients, 8U lookup tables (`histPrepareImages`, `calcHistLookupTables_8u`) |
| `ImgProc/Histograms/Internal/HistBins.cs` | `ushort` and `float` bin offsets per row, scalar and `Vector<double>` |
| `ImgProc/Histograms/Internal/BinSelect.cs` | Branch-free bin selection |
| `ImgProc/Histograms/Internal/CalcHistEngine.cs` | Counting with sub-histograms and row stripes |
| `ImgProc/Histograms/Internal/TableBinsVbmi.cs` | 8U two- and three-dimensional bin offsets, 64 pixels per step (AVX-512 VBMI) |
| `ImgProc/Histograms/Internal/BackProjectEngine.cs` | Back projection with per-bin values |
| `ImgProc/Histograms/Internal/CompareHistKernels.cs` | `compareHist` accumulation in `Vector128<double>` |
| `ImgProc/Histograms/Internal/ClaheEngine.cs` | Tile tables and interpolation |
| `Core/IntegralImages.cs`, `Core/Internal/IntegralKernels.cs` | Integral images |
| `Core/Internal/ByteLut.cs` | 256-entry byte table lookup (AVX-512 VBMI or scalar) |
| `Core/Conversions.cs` | `Normalize` |
| `Core/NormType.cs` | `NormType.MinMax` (OpenCV `NORM_MINMAX = 32`) |

## Histogram type

- `Mat<T>` has two dimensions. A three-dimensional colour histogram does not fit. The port
  therefore uses `DenseHistogram`: an `int[]` of sizes and a row-major `float[]` of bins.
- `DenseHistogram` with one dimension corresponds to the OpenCV `N x 1` matrix.
- `ToMat` converts one- and two-dimensional histograms. `FromMat` creates a two-dimensional
  histogram of `Height x Width` bins.
- The histogram has one channel. OpenCV interprets a multi-channel histogram `Mat` as one more
  dimension; the port has no equivalent.

## calcHist

### Planes and ranges

- `channels[i]` is a global channel index over the channels of all images, in image order.
- An empty `channels` requires one single-channel image per dimension.
- All images must have the same size. The element type is `byte`, `ushort` or `float`.
- Null `ranges` give `[0, 256)` per dimension. This is valid for `byte` images only.
- Uniform coefficients, in double: `a = size / ((double)hi - (double)lo)`, `b = -a * lo`.
  `lo` and `hi` are the `float` range bounds widened to double.
- Non-uniform ranges hold `size + 1` strictly increasing boundaries per dimension.
- The mask is a single-channel `byte` matrix. A nonzero element selects the pixel. It replaces
  OpenCV `CV_8U` and OpenCV 5.0 `CV_Bool` masks.

### Bin rules

| Source | Uniform | Non-uniform |
|---|---|---|
| `byte` | Table per value: if `lo <= v < hi`, then `clamp(cvFloor(v * a + b), 0, size - 1)`; else out of range | Table per value: bin `k` covers `ceil(R[k]) <= v < ceil(R[k + 1])` |
| `ushort`, `float` | `idx = cvFloor(v * a + b)` in double; drop `v < lo` or `v >= hi`; then clamp to `[0, size - 1]` | Float scan: `idx = -1; while (v >= R[idx + 1] && ++idx < size);` |

- The upper bound is exclusive. A value in `[lo, hi)` whose index rounds to `size` counts in bin
  `size - 1`.
- NaN passes the uniform range test (both comparisons are false). `cvFloor(NaN)` is
  `int.MinValue` on x86, and the clamp gives bin 0. The port reproduces this.
- With non-uniform bins, NaN is dropped.
- The port uses a binary search for non-uniform `ushort` and `float` bins. The boundaries
  increase strictly, so the search gives the bin of the linear scan.

### Accumulation and result

- Counting is in `int`.
- `accumulate: true` first rounds every existing bin with `cvRound` (half to even). Fractional
  values are lost: `[0.5, 1.5, 2.5, -0.5]` becomes `[0, 2, 2, 0]`. `accumulate: false` clears the
  histogram.
- The result converts each count to `float`. Counts above 2^24 lose precision, as OpenCV.

## calcBackProject

- The output has the size of the images, one channel and the source element type.
- Each pixel is `saturate_cast<T>(hist[bin] * (float)scale)`, or 0 outside the histogram.
  `byte` and `ushort` round half to even and saturate. `float` keeps the product.
- `byte` sources use the calcHist table rule (range test on the value, clamp on the index).
- Uniform `ushort` and `float` sources do not test the range and do not clamp. The pixel belongs to
  the histogram only if `0 <= cvFloor(v * a + b) < size`. Consequences:
  - A value in `[lo, hi)` whose index rounds to `size` gives 0, but calcHist counts it.
  - NaN gives 0.
- OpenCV 5.0 treats a two-dimensional histogram with one row or one column as one-dimensional.
  OpenCV 4.5.5 accepts only `N x 1`. The port follows OpenCV 5.0.

## compareHist

- Both histograms must have the same sizes. The bins are processed as one flat array.
- The `Mat<float>` overload compares all channels. Continuous matrices are one plane. Otherwise
  each row is one plane, as the OpenCV plane iterator. `CORREL` divides by the pixel count
  (OpenCV `H1.total()`), not by the element count.
- Accumulation order (OpenCV 5.0, 128-bit vector build):
  1. Blocks of four `float` bins. Each block widens to two double pairs (bins 0-1, then 2-3).
     Each pair adds into two double lanes.
  2. `lane0 + lane1` adds to the scalar accumulator.
  3. A scalar tail processes the last `Length mod 4` bins.
- The port uses `Vector128<double>`. Its width is fixed, so the result does not depend on the
  hardware vector width. `Vector<double>` is not used for this reason.

| Method | Block term | Tail term | Final |
|---|---|---|---|
| `Correl` | `s12 = a*b + s12`, `s11 = a*a + s11`, `s22 = b*b + s22`, `s1 += a`, `s2 += b` | same order, double | `num / sqrt(denom2)`, or 1 if `|denom2| <= DBL_EPSILON` |
| `Chisqr` | `(d*d) * (1/a)` with `d = a - b` in double; skipped if `|a| <= DBL_EPSILON` | `d = (float)(a - b)`, `d*d / a` | sum |
| `ChisqrAlt` | as `Chisqr` with `a + b` | `a + b` in float | `2 * sum` |
| `Intersect` | `m = a < b ? a : b` (SSE `minps`); `res += m0 + m2`, `m1 + m3` | `b < a ? b : a` (`std::min`) | sum |
| `Bhattacharyya` | `s1 += a`, `s2 += b`, `res += sqrt(a*b)` | same | `sqrt(max(1 - res / sqrt(s1*s2), 0))`; `FLT_EPSILON` guard |
| `KLDiv` | scalar only: skip `|p| <= DBL_EPSILON`; `q = 1e-10` if `|q| <= DBL_EPSILON` | — | `sum p * log(p / q)` |

- `Chisqr` and `ChisqrAlt` are OpenCV 5.0 terms. OpenCV 4.5.5 sums `a*a/b` in a scalar loop. The
  results differ in the last 5 to 9 significant digits.

## equalizeHist

- Source: single-channel `byte`. The destination may be the source.
- With `i` the smallest present value: if every pixel equals `i`, the destination is filled with
  `i`.
- Otherwise `scale = 255f / (total - hist[i])` in float. `lut[i] = 0`.
  `lut[k] = saturate_cast<byte>((float)sum * scale)` with `sum` the integer count of
  `hist[i+1..k]`.
- The histogram is complete before the table pass starts, so the in-place call is safe.

## CLAHE

- Source: single-channel `byte` or `ushort`. The destination may be the source.
- `histSize = (256 or 65536) >> BitShift`.
- Tiles:
  - If the width and the height are divisible by the grid, the tiles read the source directly.
  - Else the source is padded at the bottom and right with `Reflect101` to the next multiple of
    the grid. In this branch a divisible dimension is also padded by one whole tile (OpenCV
    computes `tiles - size % tiles`). Example: 64 x 67 with an 8 x 8 grid gives tiles of 9 x 9.
  - `Borders.CopyMakeBorderPooled` handles pads larger than the image.
- `lutScale = (float)(histSize - 1) / tileArea`.
- `clip = max((int)(ClipLimit * tileArea / histSize), 1)` in double, if `ClipLimit > 0`; else no
  clipping.
- Clipping: one pass. The excess is distributed as `excess / histSize` to every bin, then the
  remainder one count at a time with step `max(histSize / remainder, 1)` from bin 0.
- Table: `lut[i] = saturate_cast<T>((float)cumsum * lutScale)`.
- Interpolation per pixel, in float, in this order:
  `res = (L1[tx1] * xa1 + L1[tx2] * xa) * ya1 + (L2[tx1] * xa1 + L2[tx2] * xa) * ya`, then
  `saturate_cast<T>(res) << BitShift`. `xa`, `xa1` per column and `ya`, `ya1` per row use the
  unclamped tile index, as OpenCV.
- `BitShift` is new in OpenCV 5.0. The value 0 gives the OpenCV 4.x result.
- The class holds no buffers. OpenCV `collectGarbage` has no equivalent.

## integral

### Types

| Source | Sum | Squared sum |
|---|---|---|
| `byte` | `int` | `double`, `float`, `int` |
| `byte` | `float` | `double`, `float` |
| `byte` | `double` | `double` |
| `ushort`, `short` | `double` | `double` |
| `float` | `float` | `double`, `float` |
| `float` | `double` | `double` |
| `double` | `double` | `double` |

- The overloads without a squared sum validate the pair with a `double` squared sum.
- Other combinations throw `NotSupportedException`.

### Destinations

- Each allocating overload has a destination overload: `Integral(source, sum)`,
  `Integral(source, sum, sqsum)` and `Integral(source, sum, sqsum, tilted)`.
- A destination must have `(width + 1) x (height + 1)` pixels and the source channel count.
  Otherwise the call throws `ArgumentException`.
- A destination can be an ROI. The kernels write every element of the destination, including row 0
  and column 0, and no element outside it.
- A destination must not share an array with the source or with another destination. The call
  throws `ArgumentException` if two matrices have the same array.
- The allocating overloads create their outputs with `clear: false`, because every element is
  written.
- Reason: an `int` sum of 257 x 257 is 264 KB. Every allocation of 85,000 bytes or more goes on the
  large object heap and adds to the generation 2 budget. Single-thread BenchmarkDotNet results:

| Case | Allocating | Destination | cv2 baseline, destination |
|---|---|---|---|
| `Integral_U8S32`, 256 | 18.5 µs | 9.1 µs | 14.4 µs |
| `IntegralSqsum_U8S32F64`, 256 | 56.2 µs | 26.1 µs | 71.6 µs |
| `Integral_U8S32`, 1024 | 622 µs | 134 µs | 206 µs |
| `IntegralSqsum_U8S32F64`, 1024 | 1567 µs | 437 µs | 1125 µs |

- The allocating times vary between runs with the garbage collector state (33 µs to 111 µs for
  `Integral_U8S32` at 256 in earlier runs).

### Order

- OpenCV scalar loop: `s += src[x]; sum[y+1][x+1] = sum[y][x+1] + s`.
  `sq += (TSq)src * src; sqsum[y+1][x+1] = sqsum[y][x+1] + sq`.
- The port computes the row prefix `s` in one pass and the vertical addition in a second pass.
  Each output element sees the same additions as in OpenCV.
- An integer row prefix is computed exactly (`int` with SSE4.1, or `long`) and converted once.
  This equals the sequential sum when every partial sum is exact in the output type:
  - `int`: always. `int` sums wrap, as OpenCV.
  - `double`: below 2^53.
  - `float`: below 2^24.
- Other prefixes use the sequential loop in `float` or `double`.
- The tilted sum is a direct transcription of the OpenCV loop (`buf` row, special cases for one
  column). Float tilted sums keep the OpenCV operation order.

### Parallel strategy

- Row prefixes are independent. Vertical additions are independent per column.
- If a prefix uses the sequential float loop and the image has at least 2^20 elements, the
  prefix pass runs in row stripes and the vertical pass in column chunks of 512 elements, with
  more than one thread.
- Otherwise the passes are fused per row on one thread. For `byte` sources the SIMD prefix also
  adds the row above in the same pass. With AVX-512 the prefix processes 16 pixels per step: four
  `valignd` shifted additions form the prefix in the register, and `vpermd` broadcasts the last
  lane to the next step. Without AVX-512, SSE4.1 processes 4 pixels per step. Integer addition
  is exact, so both paths give the same result. A two-pass parallel run of the byte path was slower than
  the fused single-thread run (memory-bound vertical pass): 0.98 ms against 0.62 ms at 1024 x 1024.
- The tilted sum runs on one thread.

## normalize

- `NormType.MinMax`: `dmin = min(alpha, beta)`, `dmax = max(alpha, beta)`.
  `scale = (dmax - dmin) * (smax - smin > DBL_EPSILON ? 1 / (smax - smin) : 0)`.
  - `float` destination: `scale = (float)scale`, `shift = (float)dmin - (float)(smin * scale)`.
  - Other destinations: `shift = dmin - smin * scale`.
  - `smin` and `smax` come from `Reductions.MinMaxLoc` on the selected elements. NaN is skipped.
  - A multi-channel source uses all channels. A mask with a multi-channel source throws.
- `NormType.L1`, `L2`, `Inf`: `scale = norm > DBL_EPSILON ? alpha / norm : 0`, `shift = 0`.
- Other norm types throw `ArgumentOutOfRangeException`. The norm functions reject
  `NormType.MinMax`.
- The conversion is `ConvertTo` with `scale` and `shift`. With a mask, unselected destination
  elements keep their values. The overload that returns a new matrix fills them with 0.

## Performance

- 8U one-dimensional counting: four interleaved sub-histograms with byte loads. A sub-histogram
  per lane removes the store-to-load dependency of repeated values. Reading a `ulong` and shifting
  out the bytes was about 25% slower than byte loads. A continuous image without a mask is counted
  as one row of `width * rows` pixels, as OpenCV `histPrepareImages`.
- 8U two- and three-dimensional counting with AVX-512 VBMI (`TableBinsVbmi`), for planes of 1 to
  4 channels and fewer than 2^14 bins:
  1. Byte permutes (`vpermi2b`) deinterleave 64 pixels of each plane.
  2. Each table entry is split into a low-byte and a high-byte table. Two lookups per plane give
     16-bit offsets; an out-of-range entry becomes 2^14.
  3. The offsets of the planes are added in 16-bit lanes, masked pixels are set to `0xFFFF`, and
     `min(offset, total)` selects the overflow bin.
  4. Scalar increments count the 64 offsets into four sub-histograms.
  The unpack instructions change the lane order. Counting does not depend on the order. This
  path made `CalcHist3D_U8C3` (8 x 8 x 8 bins, 1024 x 1024) about 1.6 times faster than the
  scalar table loop.
- 8U two- and three-dimensional counting and back projection: the tables hold element offsets.
  An out-of-range value adds 2^28, and `BinSelect.Min` maps the sum to an overflow bin without a
  branch. A branch on the out-of-range test mispredicts on random data; removing it made the
  2-D back projection about 1.7 times faster.
- `ushort` and `float` bins: `HistBins` computes the offsets of one row with `Vector<double>`
  (widen, multiply, add, floor, compare, clamp). Each lane performs the scalar double
  operations, so the result does not depend on the vector width. The integer part of a clamped
  bin is extracted with `+ 2^52` and a narrowing conversion; AVX2 has no double-to-long
  conversion.
- Row stripes count into local histograms. The stripes add their counts to the shared histogram
  under a lock. Integer addition is exact in any order. Stripes are used only when the image has
  at least 32 pixels per bin.
- Back projection computes `saturate_cast<T>(hist[bin] * scale)` once per bin.
- `ByteLut` maps 64 bytes per step with two AVX-512 VBMI permutes (`vpermi2b`) and a blend. It is
  used by `equalizeHist`, the one-dimensional 8U back projection and the CLAHE table reads. The
  final vector overlaps the previous one and is computed before the first store, so an in-place
  call reads only source values.
- CLAHE phase 1 runs tiles in parallel. For each tile it:
  1. counts the tile in place, without an ROI object (four sub-histograms for `byte`);
  2. adds the sub-histograms and clips them in one `Vector<int>` pass;
  3. distributes the clip remainder, then forms the cumulative sum in one scalar pass that also
     adds the uniform redistribution;
  4. converts the table with `Vector<float>`: `ConvertToSingle`, multiply, `Round`,
     `ConvertToInt32`, saturate.
  The vector rounding omits the out-of-range handling of `cvRound`. The cumulative count is at most
  the tile area, so the product is finite and in the `int` range.
- CLAHE phase 2 runs row stripes. It reads the four table values per pixel (VBMI lookups per run of
  columns with the same tile pair, or scalar reads), then blends in the scalar operation order.
  For `byte` with `Vector512` acceleration, the blend processes 64 pixels per step with
  `Vector512<float>`, and the final step overlaps the previous one. Otherwise it uses
  `Vector<float>`. The blend result is in `[0, max]` up to rounding, so the blend also omits the
  out-of-range handling of `cvRound`.
- CLAHE column data (`ClaheColumns`) uses rented arrays.
- CLAHE phase 2 with row stripes first writes one element per page of a new destination on the
  calling thread (`PageTouch`; not when the destination is the source). Page faults taken by
  16 threads at once on a new 1 MB large-object-heap array cost more than on one thread. Probe at
  1024 × 1024 with 16 threads: 562 µs before, 510 µs after (cv2 default 748 µs).
- CLAHE work estimates per pixel, in `ParallelRows` units: 4 for phase 1 and 8 for phase 2. A sweep
  at 256 x 256 with 16 threads gave about 40 µs for these values. The previous estimates (1 and 4)
  kept phase 1 on one thread and gave about 100 µs. Higher estimates (more stripes) did not
  reduce the time further.
- CLAHE result, BenchmarkDotNet, 256 x 256: 141 µs on one thread (cv2 baseline 211 µs; before this
  change 240 µs) and 58 µs with 16 threads (cv2 default 63 µs; before this change 189 µs).
- `compareHist` uses `Vector128<double>` with the fixed OpenCV lane order.
- Output does not depend on the thread count. `CvParallelTests` covers every parallel path;
  memory-bound paths are in `LargeImage_Output_DoesNotDependOnThreadCount`.

## Deliberate differences from OpenCV 5.0

| Area | OpenCV 5.0 | Port | Reason |
|---|---|---|---|
| integral `byte` to `float`, row sums above 2^24 | Vector block order without IPP; rounding depends on the CPU dispatch width | Scalar order | Output must not depend on the hardware. cv2 4.5.5 with IPP also gives the scalar result. |
| Multi-channel histogram in `calcBackProject` | Extra dimension | Not supported (`DenseHistogram` has one channel) | No equivalent type. |
| Sparse histograms | Supported | Not ported | Out of scope. |
| IPP, HAL, OpenVX, OpenCL paths | Used when available | Not ported | Managed port. |
| `CLAHE::collectGarbage` | Releases buffers | No equivalent | The class holds no buffers. |
| Invalid CLAHE grid or bit shift | Undefined (division by zero, empty histogram) | `ArgumentOutOfRangeException` or `ArgumentException` | Defined failure. |

## Golden constraints

- Generator: `Geometrikal.OpenCV.Managed.Tests/tools/generate_histogram_golden.py`.
- Environment: `OPENCV_CPU_DISABLE=AVX,AVX2,AVX512F,FMA3` before `import cv2`; IPP, OpenCL and
  threads disabled.
- The battery also avoids the cv2 4.5.5 IPP paths that differ from the generic code when IPP is
  on:
  - accumulate cases pass a mask (IPP adds `float` counts and keeps fractional priors);
  - the `float` to `float` sum-only integral comes from `integral2` (IPP uses another order).
- Integral `byte` to `float` cases keep row sums below 2^24.
- cv2 Python converts a three-dimensional `ndarray` to a two-dimensional multi-channel `Mat`:
  - `compareHist` of three-dimensional histograms uses the flat `1 x n` view;
  - cv2 `calcBackProject` with a three-dimensional histogram uses channel 0 of the first two
    dimensions only. Three-dimensional back projections therefore come from a scalar model.
- Model-derived cases (`"model": true`) come from scalar models of the OpenCV 5.0 source in the
  generator: non-uniform histograms and back projections (the Python overload fixes
  `uniform=true`), CLAHE `BitShift`, three-dimensional back projections. The generator checks
  the uniform back projection model against cv2 on one- and two-dimensional cases.
- Suspect cases: `compareHist_*_m1` and `compareHist_*_m4` with at least four bins (12 cases).
  cv2 4.5.5 uses the scalar CHISQR loop. `values5` holds the OpenCV 5.0 model value.
  `HistogramOpenCv5DifferenceTests` requires the port to equal `values5` exactly and to stay
  within `1e-6` relative of the 4.5.5 value.
- `ClaheAndEqualizeHist_ScalarLutPath_MatchesDefaultPath`, `ByteLut_ScalarAndVbmiPaths_AgreeForAllLengths`,
  `CalcHist8u_VbmiTablePath_MatchesScalarPath`, `Clahe_Vector512BlendPath_MatchesVectorPath` and
  `Integral_Avx512PrefixPath_MatchesSsePath` cover the fallbacks on AVX-512 and VBMI hardware.
  The switches are `ByteLut.UseVbmi`, `ClaheEngine.UseVector512` and `IntegralKernels.UseAvx512`.
