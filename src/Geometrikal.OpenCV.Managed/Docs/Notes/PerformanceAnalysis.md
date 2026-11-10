# Performance analysis: managed port vs OpenCV

Date: 2026-10-10. Machine: Intel Core i9-11900 (AVX-512 capable), .NET 10.0.12, cv2 4.5.5.
Source benchmark: `Geometrikal.OpenCV.Managed.Benchmark` (see its README).

## Summary

The slow operations are slow for four reasons, in decreasing order of impact:

1. **No SIMD.** OpenCV processes 16–64 pixels per instruction through universal intrinsics.
   The managed port processes one element per iteration.
2. **Per-element overhead in inner loops.** Row spans are created per element per tap, kernels
   are read through tuple/array indirection, and generic `bool` flags select min/max per element.
3. **Whole-image intermediates.** Every filter allocates a padded copy of the source and a
   full-size row-pass buffer. OpenCV's `FilterEngine` streams rows through a small ring buffer
   (`filterengine.hpp`, `ringBuf`) that stays in cache, and pads only the row buffer.
4. **Large-object-heap allocation.** Every buffer of 85 KB or more is a LOH allocation and is
   collected only in gen2. 50 calls of `Erode` (1024²) caused 19 gen2 collections.

Contours, drawing, and MedianBlur ksize ≥ 7 are within 0.6–1.5× of cv2. These are scalar in
OpenCV too (or algorithmically equivalent), so the managed port already matches them.

## Measured causes (1024×1024, U8, single thread)

Each experiment re-implements one library path in two ways. All variants produce output identical
to the library. The scratch harness is not committed; the numbers below are its output.

| Operation | Library | Scalar, overhead removed | Vector256 | cv2 baseline (SSE, 1T) | cv2 1T (AVX2+IPP) |
|---|---|---|---|---|---|
| MedianBlur 3×3 | 88.1 ms | 43.0 ms | 2.19 ms | 0.72 ms | 0.60 ms |
| Erode rect 3×3 | 13.2 ms | 9.7 ms | 1.55 ms | 0.39 ms | 0.36 ms |
| Dilate ellipse 5×5 | 75.5 ms | 30.1 ms | 1.55 ms | 0.50 ms | 0.41 ms |
| GaussianBlur 5×5 | 11.4 ms | 4.5 ms | 2.20 ms | 0.80 ms | 0.70 ms |
| Sobel dx 3×3 → S16 | 10.8 ms | 2.9 ms | not measured | 0.73 ms | 0.71 ms |
| CopyMakeBorder 16 px, C3 | 2.62 ms | 0.42 ms (span copy) | n/a | 0.76 ms | 0.92 ms |

The remaining gap between the Vector256 variants and cv2 comes mostly from item 3: the
experiments still call the library's `CopyMakeBorder` and allocate full-size intermediates.

## Findings per operation

### MedianBlur ksize 3 and 5 (`MedianEngine.SortNet`)

- OpenCV: `medianBlur_SortNet` (`median_blur.simd.hpp`) applies the sorting network with `v_min`/`v_max`
  to whole vectors of pixels (`MinMax8u`). It computes 32 medians per network pass with AVX2.
- Port: one pixel per pass. The network is a `(int A, int B)[]` read per compare-exchange. The
  window is a heap `T[]` with bounds checks. The edge clamps `j0`/`j2` are computed for every pixel.
- Ratio: SIMD alone gives about 20× (43 ms → 2.2 ms). Overhead removal gives 2× (88 ms → 43 ms).

### Morphology (`MorphEngine`)

- OpenCV: `MorphRowVec`, `MorphColumnVec` and `MorphVec` (`morph.simd.hpp`) apply `v_min`/`v_max`
  to four vectors per iteration. Non-rectangular kernels fold the vector loads of every kernel
  point for each output row.
- Port, rectangular path: per-element `MinMax(…, erode)` with a runtime flag; full-size `rowPass`
  intermediate.
- Port, arbitrary path (`ApplyArbitrary`): `padded.GetReadOnlyRowSpan(y + coordY[k])` runs inside
  the per-pixel, per-channel, per-tap loop. This is about 21 span constructions (validation,
  `checked` arithmetic) per output element for a 5×5 ellipse. The loop order is pixel-major, so
  the code does not stream rows.
- `MorphologyEx` composes `Erode`/`Dilate`. Each call pads again and allocates again. `Gradient`
  also allocates for `Subtract`.

### Separable and fixed-point filters (`SepFilterEngine`, `GaussianFixedPoint`)

- `ColumnPassIntToByte`/`ColumnPassIntToShort` call `rows.Row(y + k)` per element per tap. This
  path serves Sobel, Scharr and Laplacian U8→S16, and SepFilter2D in bit-exact mode. Hoisting the
  row spans out of the element loop gives 3.7× on Sobel without SIMD.
- `GaussianFixedPoint.Apply` allocates one `ushort[]` per padded row and reads `rows[y + k][i]` through a
  jagged array in the inner loop. The tap loop is generic. OpenCV uses `hlineSmooth5N`,
  `hlineSmooth5N14641` and `vlineSmooth` (`smooth.simd.hpp`). These are specialised per kernel
  size and symmetry, vectorised on `v_uint16`, and run on a ring buffer.
- Float paths (`RowPassFloat`, `ColumnPassFloat`) keep row spans hoisted and are only 2.4–4.4×
  slower than cv2. The remaining gap is SIMD and intermediates.

### Box filter (`BoxFilterEngine`)

- The row pass recomputes the full `kw` sum per element. OpenCV's `RowSum` uses a running sum
  (add the incoming pixel, subtract the outgoing pixel) and vectorises the column pass.
- `CastSum` converts every output through `double` and `Math.Round(MidpointRounding.ToEven)`.

### CopyMakeBorder (`Borders.CopyMakeBorder`)

- Every destination element goes through `colMap[x]` and a per-channel loop, including the
  interior. OpenCV copies the interior with `memcpy` and fills only the border columns.
- Every padded filter calls this function, so its cost is part of every filter result above.

### Mat row access (`Mat<T>.GetRowSpan`)

- Each call runs `ValidateRow`, `checked` offset arithmetic, and `AsSpan` range validation. The cost
  is acceptable once per row. Inside element loops it dominates (see Morphology and SepFilter).

## Status after steps 1 and 2 (2026-10-10)

Step 1 (remove per-element overhead) and step 2 (vectorize hot loops) are implemented for median, morphology, `CopyMakeBorder`, the separable
engine, the fixed-point Gaussian, `filter2D`, and the box filter. Internal temporaries
(padded copies, row buffers) now come from `PooledMat<T>`. Outputs are byte-identical to
the scalar implementation; `Tests/Regression` verifies this on 924 wide-image cases.

Full benchmark run, 1024×1024, managed vs cv2 baseline (1 thread, SSE, no IPP):

| Operation | Before | After | cv2 baseline | Ratio after |
|---|---|---|---|---|
| MedianBlur 3×3 | 99.8 ms | 1.44 ms | 0.72 ms | 2.01× |
| MedianBlur 5×5 | 510 ms | 4.35 ms | 3.63 ms | 1.20× |
| Erode rect 3×3 | 12.9 ms | 0.21 ms | 0.39 ms | 0.54× |
| Dilate ellipse 5×5 | 63.6 ms | 0.31 ms | 0.50 ms | 0.62× |
| Open ellipse 5×5 | 129 ms | 0.65 ms | 0.79 ms | 0.83× |
| GaussianBlur 5×5 U8 | 10.6 ms | 0.72 ms | 0.80 ms | 0.91× |
| GaussianBlur 5×5 U8C3 | 29.8 ms | 2.34 ms | 2.26 ms | 1.04× |
| Blur 5×5 U8 | 6.90 ms | 1.62 ms | 1.28 ms | 1.26× |
| Sobel dx U8→S16 | 9.53 ms | 0.70 ms | 0.73 ms | 0.97× |
| Laplacian 3 U8→S16 | 9.02 ms | 0.90 ms | 1.02 ms | 0.88× |
| Filter2D 3×3 U8 | 12.9 ms | 0.81 ms | 0.93 ms | 0.87× |
| CopyMakeBorder 16 px C3 | 2.22 ms | 0.32 ms | 0.76 ms | 0.42× |

Large kernels, added after steps 1 and 2. 1024×1024, managed vs cv2:

| Operation | Managed | cv2 baseline | Ratio | cv2 1T (AVX2+IPP) | Ratio |
|---|---|---|---|---|---|
| Sobel dx ksize 7 U8→S16 | 1.26 ms | 3.08 ms | 0.41× | 2.91 ms | 0.43× |
| GaussianBlur 15×15 U8 | 2.03 ms | 2.39 ms | 0.85× | 1.74 ms | 1.17× |
| GaussianBlur 15×15 F32 | 3.06 ms | 3.01 ms | 1.02× | 2.12 ms | 1.44× |
| Filter2D 11×11 U8 | 13.2 ms | 12.6 ms | 1.04× | 8.57 ms | 1.54× |
| Erode rect 15×15 | 0.56 ms | 0.49 ms | 1.14× | 0.44 ms | 1.28× |
| Dilate ellipse 15×15 | 2.74 ms | 2.35 ms | 1.17× | 1.54 ms | 1.78× |
| Blur 15×15 U8 | 1.60 ms | 0.83 ms | 1.93× | 0.75 ms | 2.12× |
| MedianBlur 15 U8 | 64.1 ms | 29.0 ms | 2.21× | 28.5 ms | 2.25× |

At 256×256, `Blur` 15×15 is 3.8× cv2 and `MedianBlur` 15 is 1.8× cv2.

## Status after steps A–D and G (2026-10-11)

Steps A (box filter fixed-point divide), B (constant-time median histogram), C (median
row tails and Paeth's 3×3 method), D (register-blocked float and `filter2D` loops) and G
(row-stripe multithreading) are implemented. Outputs are unchanged except for one
correction: U8 box filters with an even kernel area ≤ 256 now round exact halves up, as
OpenCV's `ColumnSum<ushort, uchar>` does (see `FilteringPortNotes.md`).

Selected results, BenchmarkDotNet, 100 ms iterations. "1T" compares the single-threaded port
with cv2 baseline (1 thread, SSE, no IPP). "MT" compares the port with all 16 logical cores
(`CvParallel.MaxDegreeOfParallelism = 16`) with cv2 default (16 threads, AVX2, IPP).

| Operation, 1024×1024 | Port 1T | Ratio 1T | Port MT | Ratio MT |
|---|---|---|---|---|
| MedianBlur 3×3 | 0.24 ms | 0.33× | 0.33 ms | 0.52× |
| MedianBlur 5×5 | 2.11 ms | 0.58× | 0.60 ms | 0.20× |
| MedianBlur 15 | 26.1 ms | 0.90× | 4.63 ms | 0.16× |
| Blur 5×5 U8 | 0.40 ms | 0.31× | 0.37 ms | 0.30× |
| Blur 15×15 U8 | 0.86 ms | 1.03× | 0.56 ms | 0.72× |
| Filter2D 11×11 U8 | 5.51 ms | 0.44× | 1.58 ms | 0.17× |
| Sobel ksize 7 U8→S16 | 1.33 ms | 0.43× | 0.83 ms | 0.27× |
| GaussianBlur 15×15 U8 | 2.21 ms | 0.92× | 0.66 ms | 1.08× |
| GaussianBlur 15×15 F32 | 4.15 ms | 1.38× | 1.43 ms | 0.64× |
| Erode rect 3×3 | 0.26 ms | 0.66× | 0.26 ms | 0.73× |
| Dilate ellipse 15×15 | 2.89 ms | 1.23× | 0.82 ms | 0.52× |

Measurement note: each run has a low coefficient of variation (1–6%), but some cases differ
by up to 2× between runs of the same code (MedianBlur 15 at 256×256: 1.94 ms and 3.71 ms;
GaussianBlur 15×15 F32 at 1024×1024: 2.79 ms and 4.15 ms). The cause is not yet identified.
Alignment of 32-byte vectors in pooled buffers is a hypothesis. Treat single-thread
differences below about 30% between runs as noise.

## Core operations and colour conversion (2026-10-11)

Conditions:
- Single thread, 1024×1024, `byte` unless stated.
- Port numbers come from a Stopwatch probe with tiered compilation disabled, so that the first
  calls run fully optimized code.
- cv2 runs with 1 thread, SSE baseline and IPP off.

A short probe with tiered compilation enabled measures unoptimized code (`Sum`: 763 µs instead
of 56 µs). Use BenchmarkDotNet or disable tiering.

| Operation | Port (µs) | cv2 (µs) | Ratio |
|---|---|---|---|
| Add | 161 | 228 | 0.71 |
| AbsDiff | 180 | 222 | 0.81 |
| AddWeighted | 549 | 431 | 1.27 |
| Multiply, scale 0.5 | 432 | 354 | 1.22 |
| Compare | 132 | 223 | 0.59 |
| InRange, 3 channels | 835 | 943 | 0.88 |
| ConvertTo byte to float | 600 | 973 | 0.62 |
| ConvertTo float to byte | 392 | 296 | 1.32 |
| Sum | 50 | 36 | 1.41 |
| MeanStdDev | 74 | 68 | 1.09 |
| MinMaxLoc | 124 | 91 | 1.35 |
| Norm L2, float | 227 | 356 | 0.64 |
| Split, 3 channels | 1494 | 1437 | 1.04 |
| BGR to gray | 280 | 426 | 0.66 |
| BGR to RGB | 660 | 649 | 1.02 |
| BGR to HSV | 2574 | 2153 | 1.20 |
| HSV to BGR | 3826 | 3096 | 1.24 |
| BGR to Lab | 4430 | 4338 | 1.02 |
| Lab to BGR | 6109 | 6141 | 0.99 |
| BGR to gray, float | 1469 | 1226 | 1.20 |
| BGR to Lab, float | 10230 | 9546 | 1.07 |
| BGR to YCrCb | 1390 | 1045 | 1.33 |

Techniques (see `ImgProc/Color/Internal/ColorVectors.cs`, `ColorVectorKernels.cs`,
`LabVectorKernels.cs`):

- 3- and 4-channel rows are deinterleaved and interleaved with byte shuffles (`pshufb` on
  SSSE3, `tbl` on ARM64). Each kernel computes on channel planes in `Vector128` lanes, in the
  operation order of the scalar kernel for the same pixel.
- Byte gray and the Y of byte YCrCb use 16-bit multiply-add (`pmaddwd`) on (channel 0,
  channel 1) and (channel 2, 1) pairs, with the rounding constant in the second pair. The sums
  equal the 32-bit sums exactly.
- The 8U HSV division tables are computed per lane with float division. A start-up check
  confirms that float division reproduces every table entry.
- sRGB float to Lab: the 8 corner values and 8 weights of the trilinear table are contiguous
  `short`, so each channel is one `pmaddwd` and two horizontal adds per 4 pixels.
- Float splines (Lab cube root and gamma) load the 4 coefficients per lane and transpose them.
- Byte sums use `psadbw`; byte sums of squares use `pmaddwd`, flushed to 64 bits every 4096
  blocks.
- Byte `Multiply` and `AddWeighted` skip the `cvRound` range check when the scale bounds the
  result inside the `int` range.
- HSV forward was 7× slower than cv2 because of data-dependent branches. Branch-free
  arithmetic, then vectors with a 16-bit hue numerator, reduced it to 1.2×.

## Gap closure for filters, edges, CLAHE and advanced filters (2026-10-11)

Changes (details in the port notes of each area):

- `RowStream` streams the separable engine, the 8-bit Gaussian and small integer `filter2D`
  through a row buffer (step E). The padded copy and the full-size row-pass image are gone.
- Exact int16 lanes for 8U→16S Sobel/Scharr and for integer `filter2D` kernels (Laplacian
  3×3); `Vector512` lanes for the 8-bit Gaussian.
- `PageTouch` commits the pages of a new destination on one thread before a parallel run.
- stackBlur: uint16 row box sums; byte column pass in tiles of columns × row bands.
- guidedFilter: independent box filters run together; the two-pass parallel box filter runs only
  with at least 4 stripes.
- spatialGradient: two-step row kernel (column sums, then shifted differences).
- copyMakeBorder: destination overload; VBMI permutation for byte borders.

BenchmarkDotNet, in-process, 15 × 100 ms. "1T" is the port on one thread against cv2 baseline;
"MT" is the port on 16 threads against cv2 default (cv2 5.0.0 for stackBlur, contrib 4.5.5 for
guidedFilter, 4.5.5 otherwise).

| Case | Size | Before 1T | After 1T | Ratio | Before MT | After MT | Ratio |
|---|---|---|---|---|---|---|---|
| GuidedFilter8_Gray | 256 | 0.76 ms | 0.85 ms | 0.58 | 1.52 ms | 0.68 ms | 0.32 |
| GuidedFilter8_GrayPrepared | 256 | 0.51 ms | 0.50–0.79 ms | 0.86–1.36 | 1.46 ms | 0.43 ms | 0.38 |
| StackBlur15x15_U8 | 256 | – | 112 µs | 0.73 | 86 µs | 51 µs | 1.10 |
| Laplacian3_U8S16 | 1024 | 0.73 ms | 0.43 ms | 0.45 | 0.61 ms | 0.31 ms | 0.59 |
| Sobel3Dx_U8S16 | 1024 | 0.68 ms | 0.54 ms | 0.85 | 0.45 ms | 0.30 ms | 0.51 |
| GaussianBlur5x5_U8C3 | 1024 | 2.23 ms | 1.51 ms | 0.75 | 1.19 ms | 0.82 ms | 0.90 |
| Clahe_U8 | 1024 | 1.39 ms | 1.30 ms | 0.43 | 0.85 ms | 0.56 ms | 0.75 |
| CopyMakeBorderInto16_U8C3 | 256 | – | 7.4 µs | 0.48 | – | 8.2 µs | 0.53 |

The "before" values come from the request (BenchmarkDotNet) or from the probe at the start of the
work. The prepared guided filter varied from 0.50 ms (probe, repeated) to 0.79 ms (one
BenchmarkDotNet run, minimum 0.62 ms).

## Remaining gaps

| Gap | Ratio | Cause | Status |
|---|---|---|---|
| Output allocation at 256×256 (float, int16, double and three-channel outputs of 85 KB or more) | 1.1–2.5× | In an in-process BenchmarkDotNet run the large-object heap grows between gen2 collections, so every new output lands on newly committed pages that are not in the cache (64 page faults for 256 KB). In the probe, gen2 collections every 7–17 calls cost 10–20 µs per call instead. cv2 reuses a freed heap block. Examples, one thread: Sobel3Dxy_F32 64 µs against 33 µs (24 µs into an existing destination), SpatialGradient_U8 38 µs against 24 µs (17 µs kernel), CopyMakeBorder16_U8C3 35 µs against 16 µs (7 µs into a destination). With 16 threads a collection suspends every worker, which also limits Laplacian3, Sobel, Scharr, GaussianBlur5x5_U8C3 and stackBlur F32/U8C3 at 256×256. | Application setting (`System.GC.LOHThreshold`); destination overloads where they exist. |
| Output allocation at 256×256, three channels (before this pass) | 2.7–5.2× | Outputs of 85 KB or more go on the LOH; LOH allocations trigger gen2 collections. With `DOTNET_GCLOHThreshold=0x200000`, `Erode` U8C3 drops from 85 µs to 33 µs. | Application setting (`System.GC.LOHThreshold`). |
| Float separable filters, 1024×1024 | 0.8–1.4× (1T) | Memory traffic: padded copy, full-size row-pass image, and a 4 MB output written to fresh pages (about 0.5 ms). | Step E. |
| Rectangular erode/dilate, large kernel | 1.35× (1T) | Separable min/max over 15 taps per pass. | Not planned (see below). |
| Run-to-run variation | up to 2× | Not identified. | Open. |

## Next steps

Apply in this order. Each step keeps the cv2 golden tests and `Tests/Regression` green.
Steps A to D change no public API.

### A. Box filter U8: port the fixed-point divide — done

Result: the fixed-point rule differs from `cvRound(s/area)` for even
areas and equals cv2. The port now matches cv2; 5 even-kernel golden cases were added.

1. Port `divScale` and `divDelta` from `ColumnSum<ushort, uchar>` (`box_filter.simd.hpp:438-457`).
2. Replace the lookup table with `(s + divDelta) * divScale >> 23`. Compute it on
   `Vector<uint>` after a widen from `Vector<ushort>`, then narrow to bytes.
3. Verify that the fixed-point result equals the current `CastSum` result for every sum in
   `[0, area * 255]` and every kernel size used in the tests. The golden tests pass with
   `CastSum`. If the two methods differ for some sum, the fixed-point method is the OpenCV
   5.0.0 behaviour, and the golden data decides.
4. Remove the per-call table allocation.

Expected result: `Blur` 15×15 at or below cv2 at both sizes. The same change applies to
`Blur` 5×5 (1.26× now).

### B. MedianBlur ksize ≥ 7: vectorize the histogram method — done

Result: constant-time column histograms (`medianBlur_8u_O1`), processed in 256-column
stripes, with a branch-free SIMD search for the median bin. ksize 15 at 1024×1024: 64 ms →
26 ms.

1. Store the coarse and fine histograms as `ushort[16]` blocks.
2. Add and subtract one column histogram with one `Vector128<ushort>` or two
   `Vector<ushort>` operations per 16-bin block, as `median_blur.simd.hpp` does.
3. Keep the scalar search for the median bin. OpenCV keeps this step scalar too.

Expected result: 2.2× → about 1×.

### C. MedianBlur 3×3: profile, then fix — done

Result: the cause was the scalar tail of each row, not the network. An overlapping final
vector replaces it (3×3 and 5×5), and U8 3×3 uses Paeth's column-sort method.

1. Profile with the dotTrace or BenchmarkDotNet disassembly diagnoser.
2. Check for register spills in `Network9` and for redundant loads. Adjacent output
   vectors share six of nine input vectors. Reuse them.

### D. Float filters: wider and unrolled vector loops — done

Result: `filter2D` and the float separable passes keep four accumulators in registers across
taps. `Filter2D` 11×11 improved 2.4×. Float Gaussian 15×15 did not improve; it is
memory-bound (step E). Wider vectors were not measured.

1. Process two to four `Vector<float>` accumulators per iteration in `RowFloat`,
   `ColumnPassFloat` and `AccumulateFloats` to hide multiply and add latency.
2. Measure `Vector512<float>` on AVX-512 hardware (`DOTNET_PreferredVectorBitWidth=512`).
3. Do not use FMA. Keep the lane operation order of the scalar code.

Expected result: GaussianBlur F32 and Filter2D at about 1× cv2 with AVX2.

### E. Row ring buffer (former step 3, large)

Replace the padded-source copy and the full-size row-pass image with a `kh`-row pooled
ring buffer, as OpenCV `FilterEngine` does. Pad only the current row. This keeps the
working set in cache at 1024×1024 and above, and removes one full-image copy per call.

### F. Destination overloads (former step 4, API change, on demand)

Add overloads that write into a caller-supplied `dst`, as OpenCV `OutputArray` and
OpenCvSharp `Cv2.Erode(src, dst, kernel)` do. This removes allocation only when the caller
reuses `dst` across calls, for example in a per-frame pipeline. It does not change the
allocate-per-call benchmark, and it is not required to close the 256×256 gap: the
`System.GC.LOHThreshold` measurement shows that the gap is gen2 collection cost, which the
application can configure.

Proposed contract, to agree before implementation:

- `dst` must have the output width, height and channel count. Otherwise the call throws
  `ArgumentException`. `Mat<T>` cannot reallocate as OpenCV `create` does.
- `dst` can be `src` or overlap `src`. The result equals the result with a separate `dst`.
- Overlap is detected by memory range, not by start address. The engines copy the source
  to a pooled buffer today, so overlap is safe now. Step E must keep this guarantee.

Pooled results with explicit disposal are not planned. They add a use-after-dispose risk
to `Mat<T>`, which does not require disposal today.

### G. Row parallelism — done

Result: `CvParallel.MaxDegreeOfParallelism` (public) and `ParallelRows` (internal). Stripe
bodies are static methods; morphology work estimates are divided by `Vector<T>.Count`. The
float-source box filter stays single-threaded.

`Parallel.For` over row stripes after step E. This matches the cv2 default
configuration. The GaussianBlur 15×15 U8 case shows the effect: cv2 default is 0.61 ms,
cv2 single thread is 1.74 ms.

### Not planned

- Rectangular morphology with the van Herk/Gil-Werman algorithm. It costs O(1) per pixel
  for any kernel size and is exact for min and max. OpenCV does not use it. The port is
  already 1.14× cv2 at 15×15. Reconsider only if large rectangular kernels are a
  production requirement.
- DFT-based `Filter2D` for kernel areas ≥ 130. The port uses direct convolution, which
  is exact. Port the DFT path only if a production kernel exceeds that area.
