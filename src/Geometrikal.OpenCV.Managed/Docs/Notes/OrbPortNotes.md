# FAST and ORB port notes

Notes for the port of the FAST detector (`Features/Fast`, `Features/FastFeatureDetector`) and the
ORB detector and descriptor (`Features/Orb`). File and line references are against the OpenCV
5.0.0 submodule at `src/OpenCV`.

## Sources

| Port | OpenCV source |
|---|---|
| `Fast.Detect`, `FastFeatureDetector`, `Internal/FastEngine` | `modules/features/src/fast.cpp` (`FAST_t`, `FAST`, `FastFeatureDetector_Impl::detect`) |
| `Internal/FastEngine.MakeOffsets`, `CornerScore` | `modules/features/src/fast_score.cpp` (`makeOffsets`, `cornerScore<8/12/16>`) |
| `Orb`, `Internal/OrbEngine` | `modules/features/src/orb.cpp` (`ORB_Impl::detectAndCompute`, `computeKeyPoints`, `HarrisResponses`, `ICAngles`, `computeOrbDescriptors`) |
| `Internal/OrbEngine.FastAtan2` | `modules/core/src/mathfuncs_core.simd.hpp` (`atan_f32`) |
| `Internal/OrbPatterns` | `orb.cpp` (`bit_pattern_31_`, `makeRandomPattern`, `initializeOrbPattern`) |
| `Internal/OrbRng` | `modules/core/include/opencv2/core/operations.hpp` (`RNG::next`, `RNG::uniform`) |
| `Orb.Detect`, `Orb.Compute` | `modules/features/src/feature2d.cpp` (`Feature2D::detect`, `Feature2D::compute`) |

## Public API

```csharp
KeyPoint[] Fast.Detect(Mat<byte> image, int threshold, bool nonmaxSuppression = true, FastDetectorType type = FastDetectorType.Type9_16)
new FastFeatureDetector(int threshold = 10, bool nonmaxSuppression = true, FastDetectorType type = FastDetectorType.Type9_16)
KeyPoint[] FastFeatureDetector.Detect(Mat<byte> image, Mat<byte>? mask = null)

new Orb(int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 8, int edgeThreshold = 31, int firstLevel = 0,
        int wtaK = 2, OrbScoreType scoreType = OrbScoreType.Harris, int patchSize = 31, int fastThreshold = 20)
KeyPoint[] Orb.Detect(Mat<byte> image, Mat<byte>? mask = null)
(KeyPoint[] Keypoints, Mat<byte>? Descriptors) Orb.Compute(Mat<byte> image, IReadOnlyList<KeyPoint> keypoints)
(KeyPoint[] Keypoints, Mat<byte>? Descriptors) Orb.DetectAndCompute(Mat<byte> image, Mat<byte>? mask = null)
```

Enums: `FastDetectorType` (`Type5_8`, `Type7_12`, `Type9_16`), `OrbScoreType` (`Harris`, `Fast`).
Descriptors are an `N × 32` `Mat<byte>`; the value is null when no keypoint remains (OpenCV
releases the output matrix). `Orb.ScaleFactor` is a `double` because OpenCV stores a `double`;
`ORB::create` and the cv2 binding pass a `float`, so the constructor takes a `float`.

## FAST

### Candidate test and arc test

- OpenCV first applies a paired test: the 8 pairs `(0, 8), (2, 10), (4, 12), (6, 14), (1, 9),
  (3, 11), (5, 13), (7, 15)` of circle indexes. Each pair must have a pixel darker than
  `v − t` (bit 1) or brighter than `v + t` (bit 2), and the bits are combined with AND.
  The arc test then needs `K + 1` contiguous circle pixels on one side, with `K = patternSize/2`.
- The offset table repeats the circle after `patternSize` entries. For `Type5_8` the pairs are
  `(k, k)`, so the paired test needs all 8 neighbours on the same side. For `Type7_12` the pair
  `(0, 8)` is 4 positions apart on a 12-pixel circle, so the arc `1..7` fails the paired test.
  These two types therefore detect fewer corners than the arc test alone. The port applies the
  same paired test in the vector and the scalar code. For `Type9_16` the paired test is a
  necessary condition of the arc test, so the OpenCV 128-bit candidate test (4 quarter pixels)
  and the paired test give the same corner set.
- The OpenCV SIMD code exists for `Type9_16` only and uses saturating `v ± t`. The port uses
  `min(v, 255 − t) + t` and `max(v, t) − t` with unsigned compares, which are equal.

### Score

- cv2 builds with `CV_SIMD128`, so `cornerScore` uses the vector form: the maximum over all
  arcs of `K + 1` circle pixels of `max(min d, −max d)`, minus 1, with `d = centre − pixel`.
  The vector form does not use the threshold. The scalar fallback of OpenCV (no SIMD) starts
  from the threshold. The NMS score in the `FAST_t` 16-pixel vector loop is the same
  expression.
- The port computes the 16 arc minima and maxima in one `Vector256<short>` pass (lane `s` is
  the arc that starts at circle pixel `s`; lanes past the circle size repeat lanes from the
  start), then takes the horizontal maximum. Without `Vector256` acceleration it uses the
  scalar loop. Both give the same score.
- Without non-maximum suppression the score row is not written, so the response is 0.

### Order and threshold

- Keypoints are emitted row by row, one row late (after the 3x3 NMS of the previous row), in
  ascending x. Rows `3 .. h − 4` and columns `3 .. w − 4` are searched. Images smaller than 7
  pixels in a dimension give no keypoints.
- The threshold is clamped to `[0, 255]`. OpenCV sets the vector threshold
  (`v_setall_u8((char)threshold)`) before the clamp, so `Type9_16` results for a threshold outside
  `[0, 255]` depend on the OpenCV SIMD width. The port does not reproduce this.

### Vector and parallel structure

- `FastEngine.VectorRow` processes `Vector<byte>.Count` pixels per block: the paired test (early
  exit after 4 pairs), then the arc count in byte lanes (`c = (c − m) & m`, running maximum).
  The last block overlaps the previous block and skips the lanes already processed. The corner
  lanes are read from the lane bit mask (`ExtractMostSignificantBits`). Each corner gets a
  score from `CornerScore`.
- OpenCV `FAST` is serial. The port splits the output rows into `ParallelRows` stripes. Each
  stripe recomputes the two score rows above its first output row, so the output does not
  depend on the thread count. `OrbBasicTests.Fast_Output_DoesNotDependOnThreadCount` compares
  1 and 8 threads on 1031×517.

## ORB

### Pyramid

- `border = max(edgeThreshold, ceil(halfPatch·√2), 4) + 1`. Level `l` has the scale
  `(float)pow(scaleFactor, l − firstLevel)` and the size `cvRound(cols · (1/scale))` in single
  precision. Levels are placed left to right in one buffer of width
  `alignSize(level0 width + 2·border, 16)`; a level that does not fit starts a new band.
- Level `firstLevel` is the input with a `Reflect101` border. Other levels are
  `resize(prev, INTER_LINEAR_EXACT)` with a `Reflect101` border, where `prev` is the input for
  levels below `firstLevel` and the previous level above it.
- The mask is `threshold(mask, 0, 255, BINARY)`, resized with `INTER_LINEAR_EXACT` per level,
  and above `firstLevel` reduced with `threshold(254, TOZERO)` (only 255 stays). The mask border
  is constant 0.
- `Geometry.Resize` rejects a destination that shares the source array, so the port resizes into
  a pooled temporary and `copyMakeBorder` copies the temporary with its border into the level.

### Keypoints

- Features per level: `ndesired = nfeatures·(1 − f)/(1 − (float)pow(f, nlevels))` with
  `f = (float)(1/scaleFactor)`, all in single precision, `cvRound` per level, and the remainder
  on the last level. For the defaults: 109, 90, 75, 63, 52, 44, 36, 31.
- Per level: FAST 9/16 with NMS and `fastThreshold`, `runByPixelsMask`, `runByImageBorder`
  (`edgeThreshold`), `retainBest(2n)` (Harris) or `retainBest(n)` (FAST score).
- `runByImageBorder` removes every keypoint within `edgeThreshold` pixels of the level edge.
  When `edgeThreshold > 4` and the level is larger than `2·edgeThreshold`, the port runs FAST
  on the level without its outer `edgeThreshold − 4` pixels. A kept keypoint is then at least 4
  pixels inside the searched region, so its corner test, its score and the scores of its 8
  neighbours are the same as in the full level. The keypoint list after the border filter is
  the same.
- Harris: integer gradients `(p[1] − p[−1])·2 + …` on a 7x7 block, then
  `(a·b − c·c − k·(a + b)²)·scale⁴` in single precision with `scale = 1/(4·7·255)`. Then
  `retainBest(n)` per level.
- Angle: intensity centroid on the circular patch (`umax`), `fastAtan2(m01, m10)`. The port of
  `atan_f32` uses the same float operation order; cv2 runs the SSE baseline without FMA.
- Positions are multiplied by the level scale in single precision; `size = patchSize · scale`.

### Order of keypoints

- OpenCV `retainBest` uses `std::nth_element` and `std::partition`, so the order inside a level
  is implementation-defined (MSVC for cv2 on Windows). The set is defined: the top `n` and every
  keypoint with the boundary response. The Harris path applies `retainBest` twice; both sets
  are defined, so the final set is defined.
- The port uses `KeyPointsFilter.RetainBest` (decreasing response, ties in input order). A level
  that is not reduced keeps the order of the previous step. The golden tests sort both sides by
  `(octave, y, x)` and compare every member bit for bit, and each descriptor with the
  descriptor of its keypoint.

### Descriptors

- OpenCV `GaussianBlur` uses the fixed-point path only when the source is not a submatrix or
  the border has `BORDER_ISOLATED`. A pyramid level is a submatrix, so OpenCV uses
  `sepFilter2D` with the float kernel `getGaussianKernel(7, 2, CV_32F)`. The port calls
  `Filters.SepFilter2D<byte, byte>` with the bit-exact kernel rounded to float. The fixed-point
  `Filters.GaussianBlur` fails 61 of the 63 ORB golden cases.
- OpenCV blurs only the level rectangle; the border keeps the unblurred `Reflect101` copy. Pattern
  points that reach the border (small `edgeThreshold`) read unblurred pixels. The port keeps the
  same buffer layout and reproduces this.
- The level blur is in place on a non-isolated region in OpenCV. The neighbours outside the
  region are the `Reflect101` border of the unblurred level, so the result equals the isolated
  blur of the port.
- Sample point: `x = px·cos − py·sin`, `y = px·sin + py·cos` in single precision with
  `cos = cosf(angle·(float)(π/180))` (MSVC calls the float overloads), rounded with `cvRound`.
  The port uses `MathF.Cos`/`MathF.Sin` (the same C runtime functions on Windows). The port
  rotates 8 pattern points per `Vector256<float>` step with the same multiply and subtract
  order, rounds to even, and then reads the pixel values.
- WTA_K 2: one bit per point pair. WTA_K 3 and 4: 2 bits per tuple (index of the largest
  value; ties keep the lower index as OpenCV). The tuple pattern is `initializeOrbPattern` with
  `RNG(0x12345678)`; patch sizes other than 31 use `makeRandomPattern` with `RNG(0x34985739)`.

### compute with given keypoints

- `runByImageBorder` on the input image size with `edgeThreshold`; the number of levels is the
  largest octave plus 1. When the octaves are not ascending, the keypoints are grouped by octave
  (stable). Descriptor rows follow the output keypoint order, which is exact.

### Parallel structure

- OpenCV ORB is serial on the CPU path. The port runs FAST in row stripes and the Harris,
  angle and descriptor loops in keypoint ranges (`ParallelRows.For`). These loops write only
  their own keypoints, so the output does not depend on the thread count
  (`OrbBasicTests.Orb_Output_DoesNotDependOnThreadCount`).
- Possible next steps: a level blur into the pyramid without the temporary result; a vector
  Harris block sum.

## Deliberate differences

- Regions of interest are isolated images. OpenCV builds the border of level `firstLevel` with
  a non-isolated `copyMakeBorder`, which reads the parent matrix of a region of interest.
- `scaleFactor ≤ 1` throws `ArgumentException`. OpenCV divides by zero for 1 and continues with
  undefined level counts.
- An empty pyramid level (image too small for the scale) throws `ArgumentException`. OpenCV
  `resize` fails with an assertion.
- Compute with a keypoint whose pattern reaches outside the pyramid buffer throws
  `IndexOutOfRangeException`. OpenCV reads outside the buffer.
- FAST thresholds outside `[0, 255]` (see above).

## Performance

BenchmarkDotNet `--fast --inProcess` on the 16-core development machine, `Inputs.Gray`, with
other work on the machine. The OpenCvSharp 4.13 twin runs in the same process run. Values in µs.
Errors are up to ±15 % for the ORB cases.

| Case | Size | Port before, 1 thread | Port after, 1 thread | OpenCvSharp, 1 thread | Port after, 16 threads | OpenCvSharp, 16 threads |
|---|---|---:|---:|---:|---:|---:|
| `Fast9_16_Nms` | 256 | 2,878 | 282 | 271 | 341 | 381 |
| `Fast9_16_Nms` | 1024 | 66,933 | 4,788 | 5,942 | 1,722 | 4,319 |
| `Fast9_16_All` | 256 | 384 | 260 | 212 | 199 | 203 |
| `Fast9_16_All` | 1024 | 5,920 | 3,154 | 4,357 | 1,925 | 3,478 |
| `Fast7_12_Nms` | 256 | 2,263 | 265 | 1,657 | 243 | 1,215 |
| `Fast7_12_Nms` | 1024 | 39,617 | 6,050 | 21,219 | 1,692 | 20,029 |
| `OrbDetect` | 256 | 15,831 | 3,236 | 2,355 | 1,272 | 1,633 |
| `OrbDetect` | 1024 | 240,027 | 18,263 | 20,816 | 12,721 | 20,243 |
| `OrbDetectAndCompute` | 256 | 16,758 | 2,890 | 2,721 | 2,254 | 2,269 |
| `OrbDetectAndCompute` | 1024 | 220,505 | 29,917 | 20,449 | 14,789 | 25,423 |

The minimum of a Stopwatch loop on one thread is a more stable comparison for ORB: 1,488 µs
(256) and 18,955 µs (1024) for `OrbDetect`, 2,343 µs and 24,316 µs for `OrbDetectAndCompute`.

Changes, each with unchanged output:

- FAST score. The scalar `cornerScore` had a zero-initialised `stackalloc` per call and 144
  scalar min/max operations. One `Vector256<short>` pass replaces it. This change removes
  most of the gap of every NMS case: the score ran once per corner, and the noise input gives
  about 75,000 corners at 1024×1024.
- FAST corner lanes come from the lane bit mask, not from a byte-per-lane test.
- `KeyPointsFilter.RetainBest` finds the boundary response with a quickselect and sorts only
  the kept keypoints (48,000 keypoints to 200: 6.7 ms to 0.9 ms). A NaN response keeps the
  full sort.
- ORB runs FAST only where the border filter can keep keypoints (see Keypoints).
- ORB resizes each level into a pooled temporary and copies it with its border once. Before,
  the resize allocated a result, copied it into the level, and `CopyMakeBorder` cloned the
  level because the source shared the pyramid array.
- `ResizeLinearExact` has a single-channel 8-bit horizontal pass (1024 to 853: 2.0 ms to 1.2 ms).
- ORB descriptors rotate 8 pattern points per vector step.

Not changed:

- The level blur (`SepFilter2D` into a new matrix, then a copy into the pyramid) takes about
  6 ms of `OrbDetectAndCompute` at 1024×1024 on one thread. It is the largest remaining part
  of the descriptor stage.
- FAST without NMS allocates the keypoint list by growth and copies it to an array (9 MB at
  1024×1024 for 75,000 keypoints).

## Pooled temporaries (option 1)

Temporary buffers come from `ArrayPool<T>.Shared` or `PooledMat<T>`. The returned keypoint
array and descriptor matrix stay new allocations.

- FAST: each stripe writes its keypoints to `FastEngine.KeyPointBuffer`, a growable pooled
  array. `Fast.Detect` and `FastFeatureDetector.Detect` without a mask copy the keypoints once
  into the result array. The score and corner-position rows were already pooled.
- ORB: the pyramid buffer, the mask pyramid and the binary mask are `PooledMat<byte>`. The
  pixels between the bordered levels are not initialized, as in OpenCV; no step reads them.
- ORB: each level is blurred into a pooled matrix through `SepFilterEngine.Apply` with a
  destination, then copied back into the level. The kernel is the float kernel converted to
  `double`, so the filter path and the output do not change.
- ORB: the FAST keypoints of a level stay in one pooled buffer. The mask, border and
  `RetainBest` filters run on that span (`KeyPointsFilter.*Span`). Only the kept keypoints go to
  the output list. The Harris pass reduces each level in place.
- ORB: the descriptor pattern coordinates, offsets and sample values are pooled per stripe.

Allocated bytes per call (BenchmarkDotNet `--fast --inProcess`, 1 thread):

| Case | Size | Before | After |
|---|---|---|---|
| `Fast9_16_Nms` | 256 | 303.4 KB | 78.9 KB |
| `Fast9_16_Nms` | 1024 | 4918.4 KB | 1328.5 KB |
| `Fast9_16_All` | 256 | 570.7 KB | 121.8 KB |
| `Fast9_16_All` | 1024 | 9231.8 KB | 2057.8 KB |
| `Fast7_12_Nms` | 256 | 309.8 KB | 85.3 KB |
| `Fast7_12_Nms` | 1024 | 5011.8 KB | 1422.7 KB |
| `OrbDetect` | 256 | 727.6 KB | 64.5 KB |
| `OrbDetect` | 1024 | 8481.5 KB | 130.3 KB |
| `OrbDetectAndCompute` | 256 | 952.7 KB | 83.2 KB |
| `OrbDetectAndCompute` | 1024 | 11686.7 KB | 153.4 KB |

The FAST values after the change are the result arrays (28 bytes per keypoint). The ORB
remainder is the output keypoint list, the result array and small per-call arrays.

## Golden data (`tools/generate_orb_golden.py`)

- cv2 5.0.0, `OPENCV_CPU_DISABLE` for AVX/AVX2/FMA3/AVX-512, IPP, OpenCL and threads off. 228
  cases: 165 FAST cases (3 types; thresholds 0, 1, 8, 10, 12, 15, 20, 25, 255; NMS on and off;
  masks; 7×7, 40×9 and 9×40 images), 52 `detectAndCompute` cases (every `ORB::create`
  parameter, masks) and 11 `compute` cases (sorted and unsorted octaves, hand-made keypoints
  with half-pixel positions and multiples of 90°).
- Images and masks are recipes (hashed noise with an integer box mean, rectangles, circles,
  checkerboards, hashed jitter) that `OrbGolden.BuildImage` rebuilds with integer arithmetic.
  FAST lists of more than 3000 keypoints are stored as SHA-256.
- Every case matches bit for bit. No suspect cases.
