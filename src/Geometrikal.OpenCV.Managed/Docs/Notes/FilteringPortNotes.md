# Filtering port — bit-exactness study notes

Working notes from the OpenCV 5.0 source study for the filtering port
(`filter2D`, `sepFilter2D`, `blur`/`boxFilter`, `GaussianBlur`, `medianBlur`,
`Sobel`, `Scharr`, `Laplacian`). They record *why* the managed implementation
is arithmetic-equivalent to OpenCV and which constraints the golden-data
generator must respect. File/line references are against the OpenCV 5.0.0
submodule at `src/OpenCV`.

## Architecture decision

The first port replaced OpenCV's streaming `FilterEngine` (ring buffer of
border-extended rows) by a padded working image (`Borders.CopyMakeBorder`),
as the morphology port did. The engine's border assembly per row equals
padding: constant borders use value 0 for all filters here (no public
`borderValue`), and virtual rows/columns contribute identical values.
`BORDER_WRAP` is rejected (`FilterEngine::init`, filter.dispatch.cpp:130).
ROIs are always treated as isolated (documented project-wide difference).

The separable engine (`SepFilterEngine`), the 8-bit Gaussian
(`GaussianFixedPoint`) and the small integer `filter2D` path
(`SmallIntFilter2D`) now stream rows through `RowStream`, as `FilterEngine`
does:

- `RowStream` builds one border-extended source line at a time (the same
  values as a padded image row), applies the row stage into a buffer of
  `block + kh − 1` rows (about 32 K elements), then applies the column stage
  to `block` output rows. The next block keeps the last `kh − 1` buffer rows.
- Every output row receives the same row-stage results as with the padded
  image, so the output does not depend on the block size or the stripe split.
  Each stripe recomputes the `kh − 1` buffer rows above its first row.
- When the destination is the source array, the source is copied first.
- Removing the padded copy and the full-size row-pass image keeps the working
  set in the cache. Effect at 1024×1024, one thread: Sobel 3×3 U8→S16
  684 → 366 µs, GaussianBlur 5×5 F32 1.11 → 0.94 ms.

16-bit lanes for exact integer filters:

- 8U→16S separable filters (Sobel, Scharr) with integer kernels use int16
  lanes when `255·Σ|kx|·Σ|ky| + |delta| ≤ 32767`. Every partial sum is then
  inside the int16 range, so the int16 sums equal OpenCV's int32 sums and no
  saturation occurs. Zero taps are dropped; kernels with up to 3 or 5 taps use
  unrolled loops with the coefficients in registers.
- `filter2D` with an 8-bit source, a byte or short destination, integer
  coefficients, an integer delta and `255·Σ|k| + |delta| ≤ 32767` (for
  example Laplacian ksize 1 and 3, sharpening kernels) uses int16 lanes. Every
  partial sum of OpenCV's float accumulation is an exact integer below 2^24,
  so the float sums equal the int16 sums and the final cast is exact.
- The 8-bit Gaussian computes its Q8.8 row sums and Q16.16 column sums in
  `Vector512` lanes when AVX-512 is available. The arithmetic is exact, so the
  lane width does not change the result.

Multithreading for streamed filters: the work estimate is the number of vector
operations per output row times 8 `ParallelRows` units (4 and 16 were slower at 256×256 with 16 threads). When the filter splits
into stripes, `PageTouch` first writes one element per page of a new
destination of 64 KB or more on the calling thread. Page faults taken by
several threads at once on a new large-object-heap array cost more than the
same faults on one thread: Laplacian 3×3 at 1024×1024 with 16 threads took
617 µs without and 188 µs with the pre-touch.

## copyMakeBorder

- `Borders.CopyMakeBorder(source, destination, top, bottom, left, right,
  borderType, value)` writes into a caller-supplied destination of the
  bordered size and the source channel count; other sizes throw
  `ArgumentException`. A destination that shares the source array gets the
  source copied first. The result equals the allocating overload.
- Border elements come from a per-image element map (source element or
  constant channel). For byte images with AVX-512 VBMI, a source row of at
  least 64 bytes and borders of at most 64 bytes, each border is one `vpermb`
  of the first or last 64 source bytes blended with the constant colour. The
  permutation also reproduces the interior bytes that the 64-byte stores
  cover, so the stores and the interior copy can overlap.
- Effect, 256×256 three channels, 16-pixel Reflect101 border, one thread:
  38.5 µs → 19.6 µs with allocation (cv2 baseline 15.8 µs) and 7.5 µs into a
  reused destination (cv2 with `dst` 15.3 µs). The allocating overload
  allocates a 248 KB result on the large object heap; one gen2 collection
  follows about every 17 calls.

## Separable filter core (filter.simd.hpp)

Path selection ported from `createSeparableLinearFilter`
(filter.dispatch.cpp:313):

| Condition | Buffer | bits | Row/column arithmetic |
|---|---|---|---|
| 8U src, 8U dst, both kernels SMOOTH+SYMMETRICAL, int32-quantizable | int32 | 16 | int kernels ×256; final `saturate_u8((s + 0x8000) >> 16)` |
| 8U src, 16S dst, kernels (A)SYMMETRICAL + INTEGER | int32 | 0 | integer kernels; final `saturate_i16(s)` |
| otherwise (byte/float scope) | float32 | – | float kernels; `cvRound`+saturate for integer dst |

`getKernelType` (filter.dispatch.cpp:233) and `createBitExactKernel_32S`
(:296, quantization `cvRound(k*2^bits)` validated within `10·FLT_EPS·2^bits`)
are ported 1:1. Integer-buffer paths are exact integer arithmetic — OpenCV's
symmetric special-case kernels equal the generic loop there, so the managed
int paths use one generic loop. Float paths replicate OpenCV's *scalar*
branch structure exactly (`RowFilter`, `SymmRowSmallFilter` ksize≤5 with the
`{1,±2,1}` / `{-2,0,1}` / `{0,±1}` special associations, `ColumnFilter`,
`SymmColumnFilter`, `SymmColumnSmallFilter` ksize 3).

Known OpenCV-internal divergences (scalar tail vs SIMD lanes) that the port
resolves in favour of scalar semantics, and the generator must avoid:

1. **8U→8U bit-exact column** (`SymmColumnVec_32s8u`, filter.simd.hpp:1295):
   SIMD converts back to float and rounds half-to-even; scalar shifts with
   round-half-up. Equal except when a pixel's exact result sits on a half
   boundary (`isum ≡ 2^15 mod 2^16`). Generator asserts no golden case hits a
   half boundary.
2. **Float `[1,±2,1]`-style kernels** (`SymmRowSmallVec_32f` etc.): SIMD uses
   a different addition association than scalar. Exact when image values are
   integer-valued floats (all sums < 2^24 representable) — generator uses
   integer-valued float images for such kernels (Sobel/Laplacian-shaped).
3. **`SymmColumnSmallVec_32s16s`** adds `delta` after int16 saturation in
   SIMD; scalar saturates the full sum. Only differs at int16 extremes with
   nonzero delta — unreachable for 8U-derivative data.
4. **`SymmColumnSmallVec_32f` generic symmetric branch** groups delta as
   `(S0+S2)·f1 + (S1·f0 + δ)` while the scalar tail computes
   `((S0+S2)·f1 + S1·f0) + δ` — divergent for δ ≠ 0 on non-exact data
   (found empirically via a golden failure). Generator: no nonzero delta on
   ksize-3 float→float symmetric columns; ksize-5 columns group identically
   and carry the delta coverage.

`filter2D` (non-separable): float accumulation over the nonzero-coefficient
list in row-major order starting from `(float)delta` (`Filter2D` template,
filter.simd.hpp:3391; `preprocess2DKernel` keeps a single zero tap when the
kernel is all-zero). Kernels with an area of at least 130 (8U→8U/16S and
32F→32F) or 50 (8U→32F) use the DFT path (`dftFilter2D`/`crossCorr`,
dispatch.cpp:1331) exactly as OpenCV does on SSE3 hardware; the port treats
every matrix as an isolated whole image, so the ROI exclusion never applies.
The DFT path accepts `BORDER_WRAP`; the direct path rejects it. Its rules are
in `AdvancedFilteringPortNotes.md`. The generator of this file keeps filter2D
kernels under area 50; `generate_advfilter_golden.py` covers the DFT path.

## Box filter (box_filter.simd.hpp)

Sum-type ladder (`createBoxFilter`:1634): 8U→8U with area≤256 → **ushort**
sums; other 8U dst combos → int32 (while `!normalize || area ≤ 2^23`);
32F → **double** sums. `RowSum`: direct 3/5-term sums for ksize 3/5,
running sums otherwise (order matters for double). `ColumnSum`: running
vertical `SUM` initialized by summing the first `ksize-1` rows in order;
per row `s = SUM + Sp; out = saturate(s*scale); SUM = s - Sm`.
**8U→8U with area ≤ 256** uses `ColumnSum<ushort, uchar>`, which divides
with a fixed-point rule in both SIMD and scalar code:
`out = (s + divDelta) * divScale >> 23` (box_filter.simd.hpp:438-457). This
rounds exact halves up and is not always equal to `cvRound(s/area)`: for a
2×1 kernel a zero sum gives 1. cv2 output equals this rule with IPP on and
off; the port implements it, and the goldens include even kernels for this
path. The paragraph below applies to the other sum types.
For int32 sums (`ColumnSum<int, T>`), OpenCV scales with a float32 factor
in its vector loops and with a double factor in its scalar tail. The vector
loops cover the first `8·⌊n/8⌋` row elements for byte and short outputs and
`4·⌊n/4⌋` for int and float outputs, for every SIMD width. For even kernel
areas exact halves occur (e.g. 810/20 = 40.5): the f32 product lands just
above the half and rounds up, the double product rounds to even. For 8U→32F
the two products can differ by 1 ulp for any area. The port applies the same
split (since the advanced filtering module; earlier versions used the double
factor for every element). The cv2 5.0.0 goldens of
`generate_advfilter_golden.py` cover even and odd areas for 8U→8U, 8U→16S and
8U→32F. Double sums (`ColumnSum<double, T>`) use the double factor everywhere.
OpenCV 5's `blockSum` path (same-type 32F filters with both kernel sides ≤ 5;
box_filter.dispatch.cpp:408) has the `ColumnSum` column recurrence but
direct row sums for every kernel width; widths 1, 2 and 4 start from 0
instead of the running `RowSum`. The two row rules agree for finite values
whose differences are exact in double, which is why cv2 4.5.5 goldens pass
with either; NaN, infinity and widely spread magnitudes expose the
difference. The port uses the `blockSum` rule.
1-pixel-dimension ksize shrink: OpenCV's condition compares
`borderType != BORDER_CONSTANT` *without stripping BORDER_ISOLATED*, so for
isolated matrices (this port's model) the shrink applies to every border
type including constant, whenever `normalize` — verified against cv2 with
`BORDER_CONSTANT|BORDER_ISOLATED` on a 1-wide image (divides by 3, not 9).
GaussianBlur's analogous shrink *does* strip the bit and genuinely skips
constant borders.

## Pooled temporaries (option 1): box filter and morphology

- `BoxFilterEngine.ApplyInto` writes the box filter into a given destination; `Apply` allocates
  the destination and calls it. The corner functions use it with a pooled destination.
- `MorphEngine.ApplyInto` writes one erosion or dilation into a given destination. The engine
  copies the source into a padded buffer first, so the destination can be the source.
  Internal `Morphology.ErodeInto` and `Morphology.DilateInto` expose it; iterations run in
  place in the destination.
- `morphologyEx` reuses the result matrix for the second operation of open, close, top-hat
  and black-hat; gradient rents the eroded image. `MorphEngine.SubtractInto` subtracts in
  place. The hit-or-miss transform erodes into the result and rents the complement.
- Allocated bytes per call, 1 thread, `MorphologyBenchmarks` (the other cases allocate only the
  result before and after; `Blur*` in `FilteringBenchmarks` did not change):

| Case | Size | Before | After |
|---|---|---|---|
| OpenEllipse5_U8 | 256 | 129.3 KB | 65.3 KB |
| OpenEllipse5_U8 | 1024 | 2,051 KB | 1,027 KB |
| GradientRect3_U8 | 256 | 193.4 KB | 65.3 KB |
| GradientRect3_U8 | 1024 | 3,076 KB | 1,027 KB |

## GaussianBlur (smooth.dispatch.cpp, smooth.simd.hpp, fixedpoint.inl.hpp)

* Kernel sizes from sigma: `cvRound(sigma·(depth==8U?3:4)·2+1) | 1`; the
  1-pixel-dimension collapse happens only when border ≠ CONSTANT (:540).
* `getGaussianKernelBitExact` runs in softdouble. All softdouble ops are
  correctly-rounded IEEE binary64 = C# `double` ops; `mulAdd` ⇒
  `Math.FusedMultiplyAdd`; `cv::exp(softdouble)` (`f64_exp`,
  core/src/softfloat.cpp:3535) is transcribed onto C# doubles verbatim
  (64-entry table + polynomial; only ±,×,÷ and round-to-int are used).
* Fixed-point quantization (`getGaussianKernelFixedPoint_ED`): error
  diffusion with `cvRound`, centre coefficient forced to `256 − Σsides` so
  raw kernels sum to exactly 256 (Q8.8).
* 8U path: horizontal pass is **exact** in Q8.8 (`Σ kraw·pixel ≤ 65280`, no
  saturation), vertical pass exact in Q16.16 (`≤ 16,711,680 < 2^32`), single
  final rounding `saturate_u8((sum + 0x8000) >> 16)`. All hline/vline
  size-specializations and the invoker's border logic are arithmetic-equal
  to one generic padded two-pass — verified by exactness, so the port uses
  the generic form.
* Float path: `sepFilter2D` with float32 kernels cast from softdouble.

## medianBlur (median_blur.simd.hpp)

`useSortNet = ksize==3 || ksize==5` on SIMD builds (any depth/channels).
Sort networks are ported with OpenCV's exact `std::min/max` comparison
semantics (`b < a ? b : a` — NOT `Math.Min`, which differs on NaN), including
the 1-row/1-column 1-D variants and clamped (replicate) border indices.
ksize ≥ 7 requires 8U and cn ∈ {1,3,4}; medians are value-determined, so the
managed histogram implementation (two-tier 16×16, Huang sliding window over
a replicate-padded image) is output-identical to OpenCV's Om/O1 pair by
construction rather than by transcription.

## Derivatives (deriv.cpp)

`getSobelKernels` (:86, Pascal-triangle ints, `ksize odd ≤ 31`, `ksize >
order`; ksize 1 ⇒ 3 for the differentiated axis) and `getScharrKernels`
(:55, `[3,10,3]`/`[-1,0,1]`) produce float32 kernels; `scale` is folded into
the smoothing kernel (kx if dx==0, else ky). Sobel/Scharr are `sepFilter2D`
calls — 8U→16S rides the bit-exact integer path automatically because the
kernels classify INTEGER+(A)SYMMETRICAL. Laplacian ksize 1/3 uses the fixed
3×3 kernel via `filter2D`; ksize ≥ 5 computes d²x and d²y separably in the
work depth (16S for 8U ksize≤5 — per-pass int16 saturation, saturating add —
else 32F), then `convertTo(dst, ddepth, scale, delta)` with OpenCV's
float-arithmetic `cvtScale` semantics.

### Element types (16U, 16S and 64F)

Sobel, Scharr, Laplacian, sepFilter2D and filter2D accept `byte`, `ushort`, `short`,
`float` and `double` sources and destinations where OpenCV does:

| Function | Unsupported (as in OpenCV) |
|---|---|
| Sobel, Scharr, sepFilter2D | `double → float` (`getLinearColumnFilter` has no 64F→32F filter) |
| filter2D, Laplacian ksize 1/3 | destination depth below the source depth; `float → double` |
| Laplacian ksize ≥ 5 | none |

- Kernel type: `ktype = max(CV_32F, sdepth, ddepth)`. The kernel is double when the source
  or the destination is double, otherwise float. Sobel/Scharr multiply the smoothing kernel by
  `scale` in the kernel type.
- Buffer type: `bdepth = max(CV_32F, sdepth, ddepth)` outside the 8-bit bit-exact modes.
- 16U/16S sources with a float buffer use the generic `RowFilter` (OpenCV's `RowVec_16s32f`
  has the same per-element order). The row is widened to float, then the generic float row
  loop runs.
- A double buffer uses the generic `RowFilter<ST, double>` and `ColumnFilter` /
  `SymmColumnFilter` with `Cast<double, DT>`; OpenCV has no vector or small-kernel variant for
  double. The port's vector lanes keep the scalar order.
- Float column filters for `ushort` destinations: generic or symmetric column filter, then
  `saturate_cast<ushort>(cvRound(sum))`.
- Laplacian ksize ≥ 5: work type `short` (8-bit source, ksize ≤ 5), `double` (64F source) or
  `float`; `d²x + d²y` with `cv::add` semantics; then `Conversions.ConvertTo` (OpenCV
  `convertTo(dst, ddepth, scale, delta)`).
- OpenCV 5.0 change (commit `1d5f28a44f`): the Laplacian ksize 1/3 kernel is double for
  double images and is multiplied by `scale` in double. cv2 4.5.5 multiplies a float kernel.
  The port follows 5.0. The golden cases `lap_f64_k3_scale01` and `lap_u8f64_k1_scale01` are
  suspects; `FiltersOpenCv5DifferenceTests` pins both the 5.0 result and the 4.5.5 golden.
- 16-bit and double data are not integer-valued floats in general. The new paths keep OpenCV's
  scalar order in every lane, so no data constraint is necessary except for float sources with
  small symmetric kernels (rule 2 above, integer-valued float images).

## Golden generator requirements (cv2 4.5.5)

```python
cv2.ipp.setUseIPP(False)          # IPP hijacks GaussianBlur/box/median/Laplacian rows
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
# process env: OPENCV_CPU_DISABLE=AVX,AVX2,AVX512F (set before import) — kills FMA paths
```

plus per-case constraints: `BORDER_ISOLATED` OR'd into every borderType
(matches the port's always-isolated ROI semantics); no exact-half outputs in
8U→8U sepFilter cases (checked in-generator with int64 arithmetic);
integer-valued float images for reassociating special kernels; filter2D
kernel area < 50; normalized box goldens use odd×odd kernels, except
8U→8U with area ≤ 256 (fixed-point path); 8U→32F
box goldens are unnormalized. The 16U/16S/64F derivative inputs come from a separate RNG
(`depth_images()`), so the older cases keep their bytes when the battery grows. Derivative
cases used by Canny (aperture 5 and 7 with `scale = 1/16`, Scharr and `ksize = -1` with
Replicate, unscaled 7×7 with int16 saturation) are part of the battery. These restrictions predate the column-scale
split; the cv2 5.0.0 goldens of the advanced filtering module cover the
normalized int32 cases and the filter2D DFT path.

## Pooled temporaries (option 1)

- Internal `Filters.GaussianBlurInto(Mat<float>, Mat<float>, ...)` writes the float Gaussian
  into a caller-supplied destination (source size, separate array). The public float
  `GaussianBlur` and `GaussianBlurInto` share `GaussianBlurCore`. SIFT uses it for pooled
  pyramid levels.
- `SepFilterEngine.Apply` with float kernels takes an optional destination.
- The row ring buffer, line buffer and column map of `RowStream` were already pooled per
  stripe. A Gaussian call allocates only the result and about 1.5 KB of kernel arrays and
  job objects. `FilteringBenchmarks.Gaussian*` allocations are unchanged (for example
  257.3 KB for `GaussianBlur5x5_F32` at 256, 4099 KB at 1024; both equal the result size).
