# Edge detection port notes

Semantics of `EdgeDetection` (OpenCV `Canny` and `spatialGradient`), the bit-exactness rules,
the parallel strategy, the deliberate difference from OpenCV 5.0, and the golden constraints.
Read this before you change the Canny engine, the fused gradient kernel or the edge golden
generator.

## Sources

- Port source: OpenCV 5.0, `modules/imgproc/src/canny.cpp` and `spatialgradient.cpp`.
- Golden reference: cv2 4.5.5 native path (IPP off).
- Files:
  - `ImgProc/EdgeDetection/EdgeDetection.cs`: public API, validation, threshold conversion.
  - `ImgProc/EdgeDetection/Internal/CannyEngine.cs`: stripes, magnitude, non-maximum
    suppression, hysteresis, final pass.
  - `ImgProc/EdgeDetection/Internal/SobelGradient3x3.cs`: fused 3×3 Sobel gradient (dx and dy
    in one pass).
- `git diff 4.5.5 5.0.0` of `canny.cpp` and `spatialgradient.cpp` changes only the SIMD API,
  removes the disabled OpenVX path and the C API, and adds a HAL hook. No output change.

## API

| Method | OpenCV equivalent | Result |
|---|---|---|
| `Canny(Mat<byte> image, threshold1, threshold2, apertureSize = 3, l2Gradient = false)` | `Canny(image, edges, ...)` | New `Mat<byte>`, one channel, values 0 and 255 |
| `Canny(Mat<short> dx, Mat<short> dy, threshold1, threshold2, l2Gradient = false)` | `Canny(dx, dy, edges, ...)` | New `Mat<byte>`, one channel, values 0 and 255 |
| `SpatialGradient(Mat<byte> source, out Mat<short> dx, out Mat<short> dy, ksize = 3, borderType = Reflect101)` | `spatialGradient` | Two new `Mat<short>` |

- The image overload accepts any channel count. For more than one channel, each pixel takes
  magnitude, dx and dy of the first channel with the largest magnitude (strict `>` scan).
- The dx/dy overload accepts 1 or 3 channels; dy must have the size and channel count of dx.
- All inputs can be ROIs. A ROI is processed as an isolated image. OpenCV's `Canny` also drops
  the ROI parent (`Mat src(src0.size(), src0.type(), src0.data, src0.step)`).

## Validation

| Condition | Exception |
|---|---|
| Null argument | `ArgumentNullException` |
| Aperture not 3, 5, 7 or -1 after the legacy-flag step | `ArgumentOutOfRangeException` |
| dx/dy channel count not 1 or 3; dy size or channels differ from dx | `ArgumentException` |
| spatialGradient: source not single-channel; border not Reflect101 or Replicate | `ArgumentException` |
| spatialGradient: ksize not 3 | `ArgumentOutOfRangeException` |

Legacy flag (canny.cpp:778-785): when `l2Gradient` is false and bit 31 of `apertureSize` is
set, the bit is cleared and the L2 norm is used. Then the range check runs. Consequence:
aperture -1 (Scharr) is accepted only with `l2Gradient = true`. With false, -1 becomes
`0x7FFFFFFF` and fails the check. The port keeps this order.

## Threshold conversion

Image overload, in this order (canny.cpp:788-817):

1. Aperture 7: both thresholds are divided by 16 (double).
2. When `threshold1 > threshold2`, the thresholds are swapped. NaN does not swap.
3. L2: each threshold is clamped with `std::min(32767.0, t)` (NaN gives 32767). A positive
   threshold is squared in double. A negative threshold is not squared.
4. `low = cvFloor(t1)`, `high = cvFloor(t2)` with x86 semantics
   (`ScalarConverter.CvFloor`): values at or above 2^31 give `int.MinValue`; values at or below
   `-2^31 - 1` wrap to `int.MaxValue`.

The dx/dy overload does steps 2 to 4.

A pixel is a candidate when `magnitude > low` and an edge seed when `magnitude > high`. Both
comparisons are strict, so a fractional threshold `t` acts as `magnitude >= floor(t) + 1`.
With L1 and `threshold2 >= 2^31`, `high = int.MinValue` and every local maximum is an edge.

## Derivatives

- Image overload: `Sobel(image, CV_16S, 1, 0)` and `(0, 1)` with the aperture and
  `BORDER_REPLICATE` on the isolated image.
- Aperture 3: `SobelGradient3x3` computes dx and dy of each row inside the stripe that needs
  it. The arithmetic is exact in 16 bits (|d| ≤ 1020).
- Apertures 5, 7 and -1: `Derivatives.SobelInto` computes the whole image into pooled
  matrices first. Aperture 7 uses `scale = 1/16`; the float sums are exact multiples of 1/16,
  so the result is `cvRound(S / 16)` with round half to even. Aperture -1 is Scharr.
- OpenCV computes the derivatives per stripe from `src.rowRange(rowStart, rowEnd)` without
  `BORDER_ISOLATED`, so a stripe reads real neighbour rows. The whole-image computation gives
  the same values.

## Magnitude

- L1: `|dx| + |dy|` as int32.
- L2: `dx·dx + dy·dy` as int32 with wrap. Only `dx = dy = -32768` wraps (to `int.MinValue`),
  and that pixel is never a candidate.
- Vector body: `Vector.Widen` of `short` to `int`, then `Vector.Abs` and add, or multiply and
  add. The lanes compute the same value as the scalar code.

## Non-maximum suppression

Per pixel with `m > low` (canny.cpp:594-638):

```
x = |dx|, y = |dy| << 15, tg22x = x * 13573          (TG22 = round(tan 22.5° · 2^15))
y < tg22x                 horizontal: m > left && m >= right
y > tg22x + (x << 16)     vertical:   m > up   && m >= down     (int32 wrap)
otherwise                 diagonal:   s = (dx ^ dy) < 0 ? -1 : 1;
                                      m > up[j - s] && m > down[j + s]
```

- The angle tests are strict. `y == tg22x` and `y == tg67x` are diagonal.
- `tg67x = x · 79109` overflows int32 for `|dx| >= 27146`. The negative value selects the
  vertical branch. This is reachable only through the dx/dy overload. The port uses
  `unchecked` arithmetic and reproduces it.
- Magnitudes outside the image are 0: a three-row ring buffer with one zero element on each
  side, and zero rows above row 0 and below the last row. Edges therefore occur on the outer
  rows and columns.
- Vector body: blocks of 16 magnitudes are compared with `low` (`Vector256` or `Vector128`)
  into a 16-bit mask. The block's 16 map bytes are set to 1 with one store; only the set bits
  run the scalar test. Most pixels of a real image are below `low`.

## Map and hysteresis

- The map is a pooled `byte[]` of `(rows + 2) · (cols + 2)`. Pixel `(i, j)` is at
  `(i + 1) · stride + j + 1`. The ring is 1. Values: 0 weak candidate, 1 not an edge,
  2 edge.
- Rows are split into stripes: at most `CvParallel.MaxDegreeOfParallelism`, at least 4 rows
  and 16384 elements (pixels × channels) per stripe. Each stripe computes magnitudes for its rows plus one halo row
  on each side, runs non-maximum suppression and pushes every edge seed on a pooled `int[]`
  stack.
- Local hysteresis per stripe (canny.cpp:642-676): a popped pixel inside the stripe interior
  pushes all 8 neighbours with value 0. A pixel on the first or last row of a stripe (except
  the image edges) is recorded as a border peak and pushes only its 2 horizontal neighbours and
  the 3 neighbours toward the stripe interior. Each stripe writes only its own rows.
- Global pass: one thread expands all border peaks with 8 neighbours.
- Final pass: `dst = map == 2 ? 255 : 0` (`Vector.Equals`), parallel over rows.
- Determinism: the final edge set is the union of the 8-connected components of
  `{map != 1}` that contain an edge seed. The stripe split and the push order do not change
  this set. Tests compare 1 and 8 threads (`CvParallelTests`) and the golden battery passes at
  1 thread.

## spatialGradient

- `dx = (v22 − v00) + (v02 − v20) + 2(v12 − v10)`,
  `dy = (v22 − v00) + (v20 − v02) + 2(v21 − v01)`; equal to `Sobel(src, CV_16S, 1, 0, 3)` and
  `(0, 1, 3)` with the same border (verified on 30 cases).
- Borders: `BORDER_DEFAULT` (Reflect101) or `BORDER_REPLICATE`, exact equality
  (spatialgradient.cpp:105). A one-pixel axis repeats the pixel.
- Each row reads three source rows and writes both outputs in two vector steps. Step 1 forms the
  column sums `S = a + 2b + c` and differences `D = c − a` in `short` lanes into a thread-local
  row buffer and copies the border columns (`BorderInterpolate`) to both ends. Step 2 computes
  `dx = S[x + 1] − S[x − 1]` and `dy = D[x − 1] + 2D[x] + D[x + 1]` from shifted loads. The int16
  values equal the direct formula. Step 1 widens each source byte once; the former single step
  widened eight loads per output block and was limited by the shuffle port. Rows shorter than
  `Vector<byte>.Count` use the scalar code. Rows run through `ParallelRows`; a split run first
  touches the pages of new outputs (`PageTouch`).
- Effect, one thread, into existing outputs: 256×256 31 → 17 µs, 1024×1024 432 → 212 µs. Canny
  aperture 3 uses the same row function.

## Deliberate difference from OpenCV

- L1 magnitude of -32768 (dx/dy overload only). OpenCV's vector loop computes
  `v_reinterpret_as_s16(v_abs(v_int16))`, which maps -32768 to -32768; its scalar tail gives
  32768. The vector body covers row elements `[0, ⌊cols·cn / L⌋ · L)` with `L = 8` (SSE) or
  16 (AVX2), so the result depends on the column and the build. The port uses 32768 for every
  element. `EdgeDifferenceTests` pins this. The image overload cannot produce -32768
  (|d| ≤ 12240).

## Golden constraints (`tools/generate_edge_golden.py`)

- `cv2.ipp.setUseIPP(False)`, `cv2.ocl.setUseOpenCL(False)`, `cv2.setNumThreads(1)`,
  `OPENCV_CPU_DISABLE=AVX,AVX2,AVX512F,FMA3`.
- IPP hazard: `ipp_Canny` runs for 8UC1, 1 thread, aperture 3 or 5 (and the 16SC1 dx/dy
  overload). It receives the thresholds before the L2 clamp and `cvFloor`, and it differs from
  the native path for out-of-range thresholds (L1 `1e10`, L2 `2000/50000`). With IPP on, the
  golden would be wrong for those cases.
- L1 dx/dy cases contain no -32768 (the generator replaces it with -32767 and asserts).
- Battery: 730 cases. Full threshold sweep (10 pairs, including swapped, fractional, negative,
  equal, aperture-7 fractional, L2 clamp, `1e10` and `-1e10`) for 7 aperture/norm
  configurations on blurred, structured, structured-with-noise and 3-channel images; other
  images (random, constant, 4-channel, 2-channel) with 4 pairs; sizes from 1×1 to 70 columns;
  stride ROIs; the legacy aperture flag; dx/dy cases from Sobel 3/5 (1 and 3 channels) and from
  random full-range int16 (tg67 overflow and L2 wrap); spatialGradient on 15 sizes, both
  borders and ROIs. No suspects.
- Wide images (up to 1031×257, 1 and 3 channels, 8 threads) were also compared with cv2
  outside the test suite: 232 cases, 0 differences.

## Performance

`EdgeBenchmarks` (benchmark project) against `tools/bench_cv2.py` cases `Edge.*`, 1024×1024,
indicative values (other processes were running). Ratio = managed / cv2; below 1 is faster.

| Case | Managed 1 thread | cv2 baseline | Ratio | Managed all threads | cv2 default | Ratio |
|---|---|---|---|---|---|---|
| Canny3_L1_U8 | 13.8 ms | 17.2 ms | 0.80 | 2.78 ms | 5.49 ms | 0.51 |
| Canny3_L2_U8 | 11.2 ms | 14.2 ms | 0.79 | 2.26 ms | 4.34 ms | 0.52 |
| Canny5_L1_U8 | 13.6 ms | 16.6 ms | 0.82 | 3.25 ms | 5.99 ms | 0.54 |
| Canny5_L2_U8 | 11.8 ms | 15.3 ms | 0.77 | 2.70 ms | 4.57 ms | 0.59 |
| Canny3_L1_U8C3 | 32.1 ms | 34.3 ms | 0.94 | 6.23 ms | 8.62 ms | 0.72 |
| Canny5_L2_U8C3 | 33.1 ms | 39.0 ms | 0.85 | 8.56 ms | 9.38 ms | 0.91 |
| Canny3_L1_U8Sparse | 1.76 ms | 4.97 ms | 0.35 | 0.78 ms | 2.42 ms | 0.32 |
| CannyDeriv_L2_S16 | 11.0 ms | 12.7 ms | 0.87 | 2.41 ms | 4.53 ms | 0.53 |
| SpatialGradient_U8 | 1.01 ms | 1.03 ms | 0.97 | 0.45 ms | 1.60 ms | 0.28 |
| Sobel3Dx_U16F32 | 1.50 ms | 2.30 ms | 0.65 | 1.13 ms | 2.97 ms | 0.38 |
| Sobel3Dx_F64 | 3.22 ms | 3.12 ms | 1.03 | 2.84 ms | 3.83 ms | 0.74 |

- The `Gray` benchmark input has a dense edge map; hysteresis and the stack dominate. The
  sparse case (`Binary` input) shows the candidate mask: most 16-pixel blocks cost one compare.
- `SpatialGradient_U8` at 256×256 allocates two 128 KB large-object-heap outputs per call; one
  gen2 collection follows about every 15 calls. With the two-step kernel (17 µs) the call takes
  about 30 µs; the rest is allocation and collection (cv2 baseline 24 µs).
- Sobel F64 and the aperture-5/7 Canny derivatives use the streamed separable engine (see
  `FilteringPortNotes.md`): Sobel 3×3 F64 at 256×256 119 → 89 µs (cv2 baseline 112 µs).
