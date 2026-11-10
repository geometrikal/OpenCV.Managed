# Advanced filtering port notes

Semantics, bit-exactness rules, parallel design, deliberate differences and golden constraints of
the advanced filters: the `filter2D` DFT path, `sqrBoxFilter`, `bilateralFilter`, the ximgproc
`guidedFilter` and `stackBlur`. Read this before you change one of these engines or the
generator `tools/generate_advfilter_golden.py`.

## Sources

- Port source: OpenCV 5.0.0 (`src/OpenCV`):
  - `modules/imgproc/src/filter.dispatch.cpp` (`dftFilter2D`), `templmatch.cpp` (`crossCorr`).
  - `modules/imgproc/src/box_filter.dispatch.cpp`, `box_filter.simd.hpp` (`sqrBoxFilter`,
    `SqrRowSum`, `ColumnSum`, `BlockSum`).
  - `modules/imgproc/src/bilateral_filter.dispatch.cpp`, `bilateral_filter.simd.hpp`.
  - `modules/core/include/opencv2/core/hal/intrin_math.hpp` (`v_exp_default_32f`).
  - `modules/imgproc/src/stackblur.cpp`.
- opencv_contrib 5.x (not in this repository): `modules/ximgproc/src/guided_filter.cpp` and
  `edgeaware_filters_common.cpp`.
- Golden reference: cv2 5.0.0 with contrib (`opencv-contrib-python-headless==5.0.0.93`), SSE2
  dispatch, IPP off, one thread.
- Files:
  - `ImgProc/Filtering/Filters.cs`: `Filter2D`, `SqrBoxFilter`, `BilateralFilter`, `StackBlur`.
  - `ImgProc/Filtering/GuidedFilter.cs`: reusable `GuidedFilter` object.
  - `ImgProc/Filtering/EdgePreservingFilters.cs`: `GuidedFilter` one-shot calls.
  - `ImgProc/Filtering/Internal/CrossCorrEngine.cs`, `BoxFilterEngine.cs`, `BilateralEngine.cs`,
    `GuidedFilterEngine.cs`, `StackBlurEngine.cs`.

## filter2D DFT path

- Selection (`dftFilter2D`): the kernel area is at least 130 for `byte→byte`, `byte→short` and
  `float→float`, and at least 50 for every other pair (`byte→ushort/float/double`,
  `ushort→ushort/float/double`, `short→short/float/double`, `double→double`). These are the values
  for SSE3 hardware. Every matrix of the port is an isolated whole image, so the ROI condition of
  OpenCV always holds.
- `BorderType.Wrap` is accepted on the DFT path and rejected on the direct path, as in cv2.
- Spectra (`maxDepth` in `crossCorr`): float32 for byte sources, except `byte→double`; float64
  for 16-bit, float and double sources and for `byte→double`.
- Kernel: `crossCorr` converts the kernel to `max(CV_32F, depth)`. It is float32 for every source
  except double; double sources keep a float64 kernel (the internal `double[]` overload of
  `Filter2DEngine.Apply`).
- Block size: `bw = min(max(cvRound(4.5·kw), 257 − kw), W)`, `dw = max(getOptimalDFTSize(bw + kw − 1), 2)`,
  then `bw = min(dw − kw + 1, W)`. The same rule applies to the height.
- Template spectrum: the kernel in the top-left corner of a zero `dh×dw` buffer, forward real 2D
  DFT with `nonzeroRows = kh`. It is computed once per call and shared by all tiles.
- Each tile and channel:
  1. Clear the `dh×dw` buffer. Copy the clipped source window.
  2. If the window is smaller than `dsz = bsz + (kw − 1, kh − 1)`, extrapolate the border from the
     window, not from the image. This differs from image-border semantics when a tile is narrower
     than the border it must reflect (a 302-wide image, 3×60 kernel, anchor (0, 1)).
  3. Forward DFT with `nonzeroRows = dsz.h`; `mulSpectrums(conjB = true)` in place; inverse DFT
     with `Scale` and `nonzeroRows = bsz.h`.
  4. Store `bsz` with `convertTo(cdepth, 1, delta)`: float spectra add `(float)delta` in float,
     double spectra add `delta` in double, then `cvRound` and saturation for integer outputs.
     Exception: double spectra to 16-bit outputs with a non-zero delta use `cvt_32f`. Rows of at
     least 8 elements convert each value to float and add `(float)delta` in float; shorter rows
     (narrow images, a narrow last tile) compute `v·1f + (float)delta` in double.
- With several channels and a non-zero delta, the correlation goes into the destination for
  float and double outputs and into a float temporary otherwise; the delta is added afterwards in
  the type of that buffer (`dftFilter2D`).
- Bit-exactness: the DFT, `mulSpectrums` and the conversions are the bit-exact ports of
  `SpectralTransforms`. OpenCV uses IPP for some 32F row DFTs when IPP is on; the goldens are
  made with IPP off.
- Parallelism: OpenCV processes the tiles serially. The port processes each (tile, channel)
  item on its own thread (`ParallelRows.ForEachSerial`); the DFTs inside an item stay on that
  thread. A single item uses the parallel DFT. Every item has the same operations for any thread
  count.

## Box filter column scaling and sqrBoxFilter

- `ColumnSum<int, T>` for `T` = byte, short, int and float multiplies by a float32 scale in its
  vector loops and by a double scale in its scalar tail. The vector loops cover the first
  `8·⌊n/8⌋` row elements for byte and short outputs and `4·⌊n/4⌋` elements for int and float
  outputs, for every SIMD width (the wide loop is followed by a 128-bit loop). The port applies
  this split for normalized `boxFilter` (`byte→byte` with area > 256, `byte→short`, `byte→float`)
  and for `sqrBoxFilter` (`byte→float`, `byte→int`). Before this module the port used the double
  scale for all elements. Results of these paths change where the float and double products
  differ: the last bit of float outputs, and exact halves of integer outputs for even kernel
  areas.
- `sqrBoxFilter` sums squares in int32 for byte sources and in double for float sources. The row
  pass is the running `SqrRowSum` (`s += v1·v1 − v0·v0`) for every kernel width.
- `sqrBoxFilter` passes the border type, including `BORDER_ISOLATED`, to the filter engine
  unchanged, and cv2 rejects `BORDER_ISOLATED` for it. The kernel shrink of single-row and
  single-column images therefore applies only to non-constant borders.
- `BlockSum` (OpenCV 5): same-type float box filters with both kernel sides at most 5 compute
  direct row sums for every kernel width; widths 1, 2 and 4 start from 0. The column recurrence
  equals `ColumnSum`. The running `RowSum` of other widths differs for NaN and infinite values
  and for values whose differences are not exact in double. The port uses the `BlockSum` row
  rule for `boxFilter<float, float>` and for the guided filter means.
- Parallelism: int32 column sums seed each row stripe from the rows above it (exact). Double
  column sums split the columns into stripes of whole 32-element blocks, so every column keeps one
  sequential history.

## bilateralFilter

- Setup (`bilateral_filter.dispatch.cpp`): a sigma at most 1e-6 returns a copy (OpenCV 5.0).
  `radius = d ≤ 0 ? cvRound(1.5·sigmaSpace) : d/2`, at least 1. The source is padded with
  `copyMakeBorder` (every border type except `Transparent`, including `Wrap`).
- Space weights: `(float)exp(r·r·gsc)` over the disc `r = √(i² + j²) ≤ radius` in row-major
  order, with `r·r` the square of the rounded root. The 8-bit filter uses
  `gsc = (float)(−0.5/σ²)`; the float filter uses the double value and skips the centre.
- 8-bit colour weights: `v_exp((i·i)·gcc)` in 4-lane blocks while `i < 256·cn − 4`, then
  `expf((float)(i·i)·gcc)` for the last four entries. `v_exp_default_32f` is ported as scalar
  float code with each `v_fma` as a multiply and an add (`BilateralEngine.VExp`). `MathF.Exp`
  calls the UCRT `expf` on Windows, as cv2 does; a unit test pins entries of the table.
- Float colour table: `len = (float)(max − min)·cn`, `N = 4096·cn`, `scale = N/len`;
  `lut[i] = (float)exp(v·v·gcc)` with `v = (float)i/scale`, and zero after the first zero entry.
  The range comes from `MinMaxLoc` (NaN skipped). A range below `FLT_EPSILON` returns a copy.
- Row formulas: OpenCV computes the first `L` columns with its vector formula and the others
  with its scalar formula. `L = 16·⌊W/16⌋` for 8-bit images and `8·⌊W/8⌋` for float images
  (SSE2 block sizes).
  - 8-bit, 1 channel: both formulas are `w = sw·cw[|v − c|]`, `wsum += w`, `sum = v·w + sum`,
    `cvRound(sum/wsum)`. For 13 taps (radius 2) the vector columns accumulate in the order
    0, 12, 1, 2, 3, 9, 10, 11, 4, 5, 6, 7, 8.
  - 8-bit, 3 channels: identical formulas in both parts; the output is `cvRound((1/wsum)·sum)`.
  - Float: the vector formula masks NaN, uses `idx = min(trunc(alpha), N)` and
    `w = sw·(lut[idx+1]·a + lut[idx]·(1 − a))`; the scalar formula clamps `alpha` to N, uses
    `cvFloor` and `w = sw·(lut[idx] + a·(lut[idx+1] − lut[idx]))` and skips NaN neighbours.
- Vectorization: the vector columns run in AVX2 blocks (32 or 16 bytes, 16 or 8 floats) with
  table loads. The loads are scalar loads inserted into vectors: `VPGATHERDD` was about 2× slower
  on the development processor (gather data sampling mitigation). The float table is also stored
  as `(lut[i], lut[i+1])` pairs, so one 64-bit load fetches both interpolation entries. Without
  AVX2 the same columns use a scalar emulation of the vector formula
  (`BilateralEngine.VectorBodyEnabled` selects it in a test). Float 1-channel rows use 512-bit
  blocks of 32 columns when AVX-512F is present; each lane has the same operations. No path uses
  fused multiply-add.
- Padding: the padded source is built in parallel, one plane per channel.
- Parallelism: row stripes; rows are independent.

## guidedFilter (ximgproc)

- API: `GuidedFilter.Create(guide, radius, eps, scale)` and `Filter<TSrc, TDst>(src)`, plus
  `EdgePreservingFilters.GuidedFilter`. The guide is byte, ushort or float with 1 to 3 channels;
  the source is byte or float with any channel count; the output type is any of byte, ushort,
  short, float, double (`convertTo` with `cvRound` and saturation).
- Planes are float32. Means are `boxFilter(CV_32F, (2r+1)², normalize, BORDER_REFLECT)` without
  `BORDER_ISOLATED`, so OpenCV 5 does not shrink the kernel of single-row planes. Kernels up to
  5×5 use the `BlockSum` row rule.
- Order of operations per element, as `GuidedFilterImpl`:
  1. Guide covariances `C_ab = box(I_a·I_b) − m_a·m_b` for the walk pattern; diagonal entries use
     `sub_mad` with `−(float)eps`.
  2. Inverse: 1 channel `inv_self`; 2 channels `div_det_2x2`; 3 channels cofactors, the
     determinant `C00·I00 + C10·I10 + C20·I20`, the clamp `|det| < 1e-6 → 1` for `eps < 0.01`,
     and division of every entry.
  3. Per source plane: `cov_k = box(p·I_k) − pm·m_k`, `alpha_g = Σ_k inv(g, k)·cov_k`,
     `beta = pm − Σ_g alpha_g·m_g`, `q = box(beta) + Σ_g box(alpha_g)·I_g`.
- SSE split: OpenCV computes `sub_mad` and `inv_self` with SSE on the first `4·⌊w/4⌋` columns
  and with scalar code on the others. `sub_mad` lanes compute `(d − c0) − s2·s1`; the scalar code
  computes `d − (s1·s2 + c0)`. `inv_self` lanes use `RCPPS`; the scalar code uses `1/x`. The other
  kernels give the same values in both parts. Without SSE every column uses the scalar formula.
- `scale < 1`: the float guide and the source planes (in their own depth) are resized with
  `INTER_LINEAR` to `(int)(W·scale) × (int)(H·scale)`; `alpha` and `beta` means are resized back.
- Box filters: the running row sums widen each padded row to double and form the differences
  `x[i + k] − x[i]` (or `x[i+k]² − x[i]²`) with vector operations; only the running additions are
  sequential, four rows at a time. Without row stripes (one thread or a small image) the row sums
  live in a ring of `k + 3` rows.
- Box filter threads: the two-pass parallel form (full-size row-sum image, then column blocks)
  runs only when the row pass splits into at least 4 stripes. With 2 stripes it was slower than
  the fused single-threaded form; at 256×256 the guided filter with 16 threads took longer than
  with one thread.
- Independent box filters run together (`BoxFilterEngine.FloatBoxes`): the guide means, the guide
  covariances, the `p·I_k` and `p` means, and the `beta` and `alpha_g` means. When one box filter
  would split into fewer than 4 stripes, whole filters run in parallel, each in the fused
  single-threaded form. Otherwise the filters run one after another, each with row stripes. The
  results are the same in both cases.
- Effect, 256×256, 16 threads (probe): gray guide 735 → 649 µs (one thread 755 µs), color guide
  1923 → 1063 µs, prepared gray 429 → 359 µs (one thread 513 µs).
- Memory: the one-shot `EdgePreservingFilters.GuidedFilter` takes every plane from the array pool
  and returns it; the 3-channel guide covariances are always pooled temporaries. A reusable
  `GuidedFilter` object owns its guide planes.
- Fusion: the products `p·I_k` and `I_a·I_b` are computed inside the box row pass. The
  covariance correction, `alpha` and `beta` are one row pass. The final sum and conversion are
  one row pass.
- Hardware dependence: `RCPPS` is an approximation whose bits depend on the processor. The port
  calls `Sse.Reciprocal`, so a 1-channel guide gives OpenCV's result on the same processor. The
  golden file stores the `RCPPS` results of a probe vector; the runner compares the 1-channel
  guide cases exactly only when the probe matches, and with a tolerance otherwise.

## stackBlur

- Supported types: byte and float, any channel count. 16-bit types are not ported (the OpenCV
  vector code overflows for them).
- `ksize` components are odd and positive. A component larger than the image becomes the image
  size, or the size minus 1 when the size is even. A width of 1 copies the source; then only the
  column pass runs.
- Row pass, kernel ≤ 9 and image wider than the kernel: direct weighted sums with clamped
  indices at both ends. Byte middle elements in `[r·cn, r·cn + 16·⌊(W·cn − 2r·cn)/16⌋)` use
  `(s·mul) >> shr` with a wrapping 32-bit multiply and saturating packs; other byte elements use
  the truncation of `s·(1/(r+1)²)`. Float 3-tap middle elements in the vector range
  (`4·⌊(W·cn − 2·cn)/4⌋` elements) use `((x1 + x1) + x0) + x2`, the others `x1·2 + (x2 + x0)`.
- Row pass, other kernels: running sums with a difference buffer; byte outputs round.
  - Byte rows: the int32 sums are exact. For radius ≤ 32 and `W ≥ 2r + 3` the port computes them
    as two vectorized box sums of width r + 1 over the replicate-padded row (a triangular sum).
    OpenCV's difference buffer stops following the replicate border near the right end, so the
    last `r + 3` pixels repeat OpenCV's recurrence from the exact sums of pixel `W − r − 3`.
    A test compares this path with the running-sum path for many widths and radii.
  - Float rows keep the running sums (their rounding depends on the order). Single-channel rows
    run four at a time; 3-channel rows interleave the channels.
- Column pass: running `sum`, `sumIn`, `sumOut` and a stack per element, in place. Byte elements
  below `16·⌊W·cn/16⌋` use `(s·mul) >> shr` (wrapping; a negative product gives 255); the others
  truncate. Radii above 254 have no table entry and use truncation for all elements.
- Byte rows with radius ≤ 15: the box sums run in uint16 lanes. The triangular sums are at most
  `255·16² = 65280`, so they are exact and equal the int32 sums.
- Parallelism, one thread or a small image: the column pass runs in place, in column blocks (64
  bytes or 16 floats, one cache line). The vector/scalar split uses the global element index, so
  the block width does not change the result.
- Parallelism, byte images that split into row stripes: the row pass writes into a pooled image,
  and the column pass reads it and writes the destination in tiles of 64 columns × a row band.
  The int32 column state at row `i` is exact: `sum = Σ (r + 1 − |j|)·v(i + j)`,
  `sumOut = Σ_{j ≤ 0} v(i + j)`, `sumIn = Σ_{j > 0} v(i + j)` over the clamped rows, and stack
  slot `(i + j + r) mod ksize` holds `v(i + j)`. Each band computes this state from its first row,
  so the bands give the same values as one column stripe. The number of bands is chosen so that
  tiles ≥ 2 × threads. Float columns keep whole-column stripes, because the float running sums
  depend on their history.
- Effect, 256×256, 16 threads (probe, cv2 5.0.0 default in brackets): U8 86 → 37–47 µs (46 µs),
  F32 94 → 76–84 µs (58 µs), U8C3 143 → 110–123 µs (96 µs). The U8C3 and F32 outputs are
  large-object-heap arrays; see the performance note below.

## Performance notes

- filter2D DFT tiles: 15×15 and 31×31 kernels on 256×256 and 1024×1024 images both use 256×256
  DFT tiles (`bw = max(round(4.5·kw), 256 − kw + 1)`, then `getOptimalDFTSize(bw + kw − 1)`), so
  the two kernels cost the same: probe at 1024×1024, one thread, 9.6 ms and 9.8 ms. An earlier run
  that measured 16.3 ms for 15×15 was run-to-run variation. The block size is OpenCV's and affects
  the result, so it stays.
- Outputs of 85 KB or more (256×256 float, int16 or 3-channel byte images) are large-object-heap
  arrays. One gen2 collection follows every 7 to 17 such calls, and a new array is written to
  pages that are not in the cache. This adds about 10–20 µs per call at 256×256 on one thread
  and more with 16 threads, because a collection suspends every thread. `sqrBoxFilter` 5×5 F32
  (512 KB double output) is the largest case: about 95 µs with `System.GC.LOHThreshold` raised
  to 8 MB and 116 µs without it (cv2 baseline 109 µs). The threshold is an application setting.

## Deliberate differences

- stackBlur: OpenCV splits the column pass into one stripe per thread, and each stripe has its
  own vector/scalar split, so cv2 results depend on `setNumThreads`. The port always reproduces
  one stripe (cv2 with one thread).
- bilateralFilter: OpenCV builds with AVX2 or AVX-512 use fused multiply-add and other block
  sizes, so their results depend on the processor. The port reproduces the SSE2 dispatch.
- bilateralFilter float range: the port takes the minimum and maximum with NaN skipped
  (`MinMaxLoc`, see `CorePortNotes.md`). OpenCV 5.0 `minMaxLoc` resets a vector lane at a NaN and
  starts from the first element, so with scattered NaN its colour table scale can differ, and a
  NaN first element makes its result NaN. The four `bilat_nanrandom_*` goldens are suspects for
  this reason.
- bilateralFilter infinite float values: OpenCV reads outside its table. The port maps an index
  outside `[0, N]` to N. The result is not specified.
- guidedFilter 1-channel guide: processor-dependent `RCPPS`, inherited from OpenCV (see above).
- filter2D DFT path: the tiles run in parallel (OpenCV: serially); the values are the same.

## OpenCV 4.5.5 differences (port notes only)

- bilateralFilter 4.5.5: a non-positive sigma becomes 1; 8-bit tables use double arithmetic;
  vector columns cover `4·⌊W/4⌋` (1 channel) or `16·⌊W/16⌋` (3 channels) and accumulate in tap
  order; tail columns group four taps with `v_reduce_sum`. Float outputs are equal to 5.0 when
  `W % 8 == 0` and the sigmas are above 1e-6.
- boxFilter 4.5.5 has no `BlockSum`; same-type float filters with kernel width 1, 2 or 4 use
  running row sums.
- guidedFilter 4.5.5 has no `scale` parameter and shrinks the box kernel of single-row planes.
- stackBlur does not exist in 4.5.5.
- filter2D DFT outputs of 4.5.5 and 5.0 are equal.

## Golden constraints

- Generator: `Geometrikal.OpenCV.Managed.Tests/tools/generate_advfilter_golden.py`; data
  `TestData/advfilter_golden.json`; runner `ImgProc/AdvancedFilterGoldenTests.cs`. Comparison is
  byte-exact; large outputs are compared by SHA-256.
- Required environment: cv2 5.0.0 with contrib (`pip install --no-deps --target <dir>
  opencv-contrib-python-headless==5.0.0.93`, `PYTHONPATH=<dir>`) and
  `OPENCV_CPU_DISABLE=AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL` before cv2 is
  imported. The script stops when cv2 is not 5.0.0 or when AVX2 is still enabled.
- `filter2D` and `boxFilter` cases pass `BORDER_ISOLATED`; `sqrBoxFilter` cases do not.
- NaN bilateral cases keep the NaNs in the upper rows and the extremes in the last row, so the
  OpenCV 5.0 and NaN-skipping ranges agree.
- A large 8-bit radius-2 bilateral case is needed to detect the 13-tap accumulation order: the
  order changes only a few pixels per million.
- Suspects: `bilat_nanrandom_c1_w37`, `bilat_nanrandom_c1_w40`, `bilat_nanrandom_c3_w37`,
  `bilat_nanrandom_c3_w40` (NaN range rule, see Deliberate differences).

## blendLinear

`ImgProc/Blend/Blending.BlendLinear` ports `blendLinear` (`imgproc/src/blend.cpp`): per element
`(s1·w1 + s2·w2) / (w1 + w2 + 1e-5f)` in float, rounded half to even and saturated for byte. The
OpenCV SSE blocks and the scalar tail use the same float operations, so the port is scalar and
row-parallel. Golden data: `tools/generate_blend_golden.py` (24 cases: byte and float, 1-4
channels, block and tail widths, zero and equal weights), all bit-exact.
