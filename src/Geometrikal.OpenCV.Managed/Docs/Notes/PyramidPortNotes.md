# Pyramid and keypoint port notes

Notes for the port of `pyrDown`, `pyrUp`, `buildPyramid` (`ImgProc/Pyramids/`), the feature
data types `KeyPoint` and `DMatch` (`Core/`) and `KeyPointsFilter` (`Features/`). File and line
references are against the OpenCV 5.0.0 submodule at `src/OpenCV`.

## Sources

| Port | OpenCV source |
|---|---|
| `Pyramids.PyrDown`, `Internal/PyrDownEngine` | `modules/imgproc/src/pyramids.cpp` (`pyrDown`, `pyrDown_`, `PyrDownInvoker`, `PyrDownVecH`, `PyrDownVecV`) |
| `Pyramids.PyrUp`, `Internal/PyrUpEngine` | `pyramids.cpp` (`pyrUp`, `pyrUp_`, `PyrUpVecV`, `PyrUpVecVOneRow`) |
| `Pyramids.BuildPyramid` | `pyramids.cpp` (`buildPyramid`) |
| `Internal/PyrRows`, `Internal/PyrColumns` | `pyramids.cpp` (`FixPtCast`, `FltCast`, row and column loops) |
| `Core/KeyPoint` | `modules/core/include/opencv2/core/types.hpp`, `modules/core/src/types.cpp` |
| `Core/DMatch` | `modules/core/include/opencv2/core/types.hpp` |
| `Features/KeyPointsFilter` | `modules/features/src/keypoint.cpp` |

## Public API

```csharp
Mat<T> Pyramids.PyrDown<T>(Mat<T> source, Size dstSize = default, BorderType borderType = BorderType.Reflect101)
void Pyramids.PyrDown<T>(Mat<T> source, Mat<T> destination, BorderType borderType = BorderType.Reflect101)
Mat<T> Pyramids.PyrUp<T>(Mat<T> source, Size dstSize = default, BorderType borderType = BorderType.Reflect101)
void Pyramids.PyrUp<T>(Mat<T> source, Mat<T> destination, BorderType borderType = BorderType.Reflect101)
Mat<T>[] Pyramids.BuildPyramid<T>(Mat<T> source, int maxLevel, BorderType borderType = BorderType.Reflect101)
```

`T` is `byte`, `ushort`, `short`, `float` or `double`. Any channel count is accepted; OpenCV
has special code for 1 to 4 channels and a table path for more channels, with equal results.

## Validation (OpenCV 5.0 rules)

- `pyrDown`: `borderType != BORDER_CONSTANT` (assertion). `borderInterpolate` rejects
  `BORDER_TRANSPARENT` and `BORDER_ISOLATED` combinations, and the first output row always
  reaches the border. Accepted: `Reflect101` (default), `Reflect`, `Replicate`, `Wrap`.
- `pyrDown` size: `|2·dw − sw| ≤ 2` and `|2·dh − sh| ≤ 2`.
- `pyrUp`: `borderType == BORDER_DEFAULT` only. Size: `|dw − 2·sw| == dw mod 2`, so
  `dw ∈ {2sw − 1, 2sw, 2sw + 1}`; the same for the height.
- A `dstSize` with a component of zero or less is OpenCV `Size::empty()` and selects the
  default size: `((w + 1)/2, (h + 1)/2)` for `pyrDown`, `(2w, 2h)` for `pyrUp`.
- `buildPyramid`: `maxLevel ≥ 0`; level 0 is the source instance (OpenCV assigns the header,
  not a copy). The other levels are `pyrDown` with the default size.
- ROIs are isolated images. OpenCV without a HAL ignores the parent image as well, because
  `pyrDown_` uses `borderInterpolate` on the view size.

## Arithmetic and bit-exactness

### Integer types

- `pyrDown`: horizontal 5-tap sum `[1 4 6 4 1]`, vertical 5-tap sum, then
  `FixPtCast<T, 8>`: `(s + 128) >> 8` with a plain cast (no saturation; the value is always in
  range).
- `pyrUp`: horizontal sums `t0 = s[x−1] + 6s[x] + s[x+1]`, `t1 = 4(s[x] + s[x+1])`, vertical
  `r0 + 6r1 + r2` and `4(r1 + r2)`, then `FixPtCast<T, 6>`: `(s + 32) >> 6`.
- All sums are exact. The OpenCV vector code packs to 16-bit lanes for 8-bit data
  (`PyrDownVecV<int, uchar>` adds in `v_uint16`); the largest 8-bit sums are 65 280 (down) and
  16 320 (up), so the 16-bit sums are exact and equal the scalar sums. The port therefore uses
  `ushort` work rows for `byte` and `int` work rows for `ushort` and `short`, in any operation
  order.

### Floating-point types

OpenCV groups the floating-point sums differently in the 128-bit vector loops and in the scalar
loops. cv2 5.0.0 compiles `pyramids.cpp` for the SSE3 baseline only (it is not a dispatched
file), so the vector loops use 4 float lanes or 2 double lanes and no FMA (`v_muladd` is a
multiply then an add). Each output element uses the order of the loop that computes it in OpenCV:

| Stage | Vector order (OpenCV `*Vec*`) | Scalar order |
|---|---|---|
| pyrDown row | `r2·6 + ((r1 + r3)·4 + (r0 + r4))` | `((r2·6 + (r1 + r3)·4) + r0) + r4` |
| pyrDown column | `(((r1 + r3) + r2)·4 + ((r0 + r4) + (r2 + r2)))·(1/256)` | `(((r2·6 + (r1 + r3)·4) + r0) + r4)·(1/256)` |
| pyrUp row | none (scalar only) | `(s−1 + s0·6) + s+1`, `(s0 + s+1)·4` |
| pyrUp even row | `((6·r1 + r0) + r2)·(1/64)` (equal to scalar) | `((r0 + r1·6) + r2)·(1/64)` |
| pyrUp odd row | `(r1 + r2)·(1/16)` | `((r1 + r2)·4)·(1/64)` |

Elements covered by the OpenCV vector loops (`W` = interior element count of a row,
`n` = destination row length in elements):

| Loop | float | double |
|---|---|---|
| pyrDown row, 1 channel | `4·⌊W/4⌋` | `2·⌊W/2⌋` |
| pyrDown row, 2 channels | `8·⌊W/8⌋` | 0 |
| pyrDown row, 3 channels | `3·(⌊(W − 4)/3⌋ + 1)` for `W ≥ 4` | 0 |
| pyrDown row, 4 channels | `W` | 0 |
| pyrDown column | `4·⌊n/4⌋` | 0 |
| pyrUp columns | `4·⌊n/4⌋` | 0 |

The interior of a `pyrDown` row is the elements `[cn, width0·cn)` with
`width0 = min((sw − 3)/2 + 1, dw)` (C integer division). The port computes both orders with
`Vector<T>` lanes; only the element ranges come from the OpenCV 128-bit layout. The golden
battery checks this: forcing every row element to the vector order fails 232 cases, and a
column boundary of `8·⌊n/8⌋` fails 105 cases.

`FltCast` multiplies by `(T)(1./(1 << shift))`, so the scale is exact for both types.

## Border and size quirks reproduced

- `pyrDown` right border: OpenCV fills the columns from `width0` with a table
  `tabR[k] = borderInterpolate(k + 2·width0 − 2)`; output column `width0 + j` reads
  `tabR[j .. j + 4]`. For `j = 1` (even source width `2m` and destination width `m + 1`) the
  column reads source columns `2m − 3 .. 2m + 1` instead of `2m − 2 .. 2m + 2`. The port uses
  the same table.
- `pyrDown` with a destination width of 1 uses only the left table, as OpenCV
  (`if (x == dsize.width) break`).
- `pyrUp` row borders: the left column reflects (`t0 = 6s0 + 2s1`), the right column replicates
  (`t0 = s[n−2] + 7s[n−1]`, `t1 = 8s[n−1]`). The source row index is
  `borderInterpolate(2·sy, 2·sh, BORDER_REFLECT_101)/2`.
- `pyrUp` with `dw = 2sw + 1` copies column `2sw − 1` to column `2sw`; with `dh = 2sh + 1` it
  copies row `2sh − 2` to row `2sh` after the main loop.

## Deliberate differences

- `pyrUp` with a one-column source and a three-column destination: OpenCV returns from the
  row loop before it writes column 2, so column 2 holds uninitialized `AutoBuffer` memory.
  The port repeats column 1. The golden battery excludes this size.

## Performance structure

- `pyrDown`: each `ParallelRows` stripe keeps a ring of five work rows (`ArrayPool`). The
  horizontal stage has four forms:
  - 8-bit, 1 channel: three `Vector<ushort>` loads at byte offsets `2x - 2`, `2x` and `2x + 2`.
    The low byte of each lane is an even source element and the high byte is an odd source
    element. The stage does not widen the row and does not decimate.
  - 8-bit, 3 channels (AVX2): an SSSE3 byte shuffle splits each row into even and odd pixels
    (12 source bytes per shuffle). The 5-tap sum then uses `vpmovzxbw` loads at element offsets
    `0` and `±3` in the two halves.
  - `float`, 1 channel (AVX2): `vshufps` and `vpermpd` deinterleave even and odd elements at
    offsets `2x - 2`, `2x` and `2x + 2`, as OpenCV `PyrDownVecH<float, float, 1>`. The sum
    order is that of the OpenCV vector code.
  - Other layouts: the source row is widened to the work type, the 5-tap sum is computed at full
    resolution with `Vector<T>` loads at offsets `±cn`, `±2cn` (overlapping final vector), and
    the sums are decimated. One-channel decimation narrows `uint`/`ulong` pairs; two-channel
    `ushort` decimation narrows `ulong` pairs; the other layouts move one pixel per unaligned
    load and store of a fixed-size struct (2 to 32 bytes).

  The column stage is a `Vector<T>` loop with an overlapping final vector for integer types.
- `pyrUp`: each stripe of source rows keeps a ring of three upsampled rows. The two row sums are
  computed in `Vector<T>` lanes and interleaved by pixel (one-channel rows use widen, shift and
  OR; other layouts use the fixed-size pixel moves). OpenCV runs `pyrUp_` on one thread; the port splits the source rows into stripes.
- Each stripe recomputes the ring rows above its first row, so the output does not depend on
  the stripe split. `Tests/Core/PyramidParallelTests` compares 1 and 8 threads; seeding a
  stripe one source row late fails all 32 cases.
- `buildPyramid` calls `pyrDown` once per level. Level 0 is the source instance (no copy).
- Possible next steps: byte shuffles for the 4-channel 8-bit `pyrDown` pass and for the
  3-channel `pyrUp` interleave.

## Performance

A/B harness: the port and OpenCvSharp 4.13 (IPP, AVX2 dispatch) run alternately in one
process; each value is the minimum of 25 timed batches. Other processes used the machine during
the measurement. "Before" is commit `8ec2765`.

1 thread:

| Case | Size | Port before (µs) | Port after (µs) | OpenCvSharp (µs) | Ratio before | Ratio after |
|---|---|---|---|---|---|---|
| `PyrDown_U8` | 256 | 28.1 | 19.5 | 20.5 | 1.47 | 0.95 |
| `PyrDown_U8C3` | 256 | 73.0 | 47.9 | 77.0 | 0.95 | 0.62 |
| `PyrDown_F32` | 256 | 34.9 | 22.4 | 22.5 | 1.34 | 0.99 |
| `PyrUp_U8` | 256 | 109.4 | 100.7 | 99.3 | 0.93 | 1.01 |
| `PyrUp_U8C3` | 256 | 349.4 | 321.6 | 467.0 | 0.67 | 0.69 |
| `PyrUp_F32` | 256 | 127.0 | 103.2 | 265.6 | 0.46 | 0.39 |
| `BuildPyramid4_U8` | 256 | 33.2 | 27.9 | 27.6 | 1.19 | 1.01 |
| `BuildPyramid4_U8C3` | 256 | 96.6 | 67.3 | 98.7 | 0.90 | 0.68 |
| `PyrDown_U8` | 1024 | 285.7 | 125.9 | 185.7 | 1.28 | 0.68 |
| `PyrDown_U8C3` | 1024 | 1249.9 | 486.7 | 1142.5 | 0.94 | 0.43 |
| `PyrDown_F32` | 1024 | 476.2 | 275.6 | 433.5 | 0.87 | 0.64 |
| `PyrUp_U8` | 1024 | 864.1 | 709.5 | 2276.8 | 0.32 | 0.31 |
| `PyrUp_U8C3` | 1024 | 4278.4 | 3526.6 | 7039.1 | 0.56 | 0.50 |
| `PyrUp_F32` | 1024 | 2835.1 | 2819.1 | 5124.8 | 0.55 | 0.55 |
| `BuildPyramid4_U8` | 1024 | 391.2 | 209.6 | 323.6 | 0.94 | 0.65 |
| `BuildPyramid4_U8C3` | 1024 | 2231.9 | 820.8 | 1615.4 | 0.88 | 0.51 |

16 threads:

| Case | Size | Port before (µs) | Port after (µs) | OpenCvSharp (µs) | Ratio before | Ratio after |
|---|---|---|---|---|---|---|
| `PyrDown_U8` | 256 | 29.5 | 17.6 | 12.0 | 2.52 | 1.47 |
| `PyrDown_U8C3` | 256 | 27.7 | 34.7 to 80.6 | 27.0 | 1.02 | 1.28 to 2.84 |
| `PyrDown_F32` | 256 | 39.1 | 26.2 | 14.2 | 2.68 | 1.84 |
| `PyrUp_U8` | 256 | 122.9 | 115.8 | 111.7 | 1.06 | 1.04 |
| `PyrUp_U8C3` | 256 | 287.1 | 323.8 | 525.5 | 0.56 | 0.62 |
| `PyrUp_F32` | 256 | 79.2 | 335.4 | 329.0 | 0.28 | 1.02 |
| `BuildPyramid4_U8` | 256 | 34.4 | 29.7 | 39.7 | 0.91 | 0.75 |
| `BuildPyramid4_U8C3` | 256 | 79.3 | 67.0 | 61.1 | 1.35 | 1.10 |
| `PyrDown_U8` | 1024 | 132.2 | 101.3 | 68.9 | 1.93 | 1.47 |
| `PyrDown_U8C3` | 1024 | 304.5 | 258.9 | 618.9 | 0.60 | 0.42 |
| `PyrDown_F32` | 1024 | 213.6 | 219.3 | 390.9 | 0.49 | 0.56 |
| `PyrUp_U8` | 1024 | 356.5 | 386.2 | 2238.8 | 0.16 | 0.17 |
| `PyrUp_U8C3` | 1024 | 1427.8 | 1468.5 | 6588.1 | 0.22 | 0.22 |
| `PyrUp_F32` | 1024 | 2523.7 | 2586.1 | 4716.1 | 0.36 | 0.55 |
| `BuildPyramid4_U8` | 1024 | 179.0 | 134.9 | 104.7 | 1.71 | 1.29 |
| `BuildPyramid4_U8C3` | 1024 | 490.2 | 320.2 | 478.4 | 1.04 | 0.67 |

Findings:

- The `pyrUp` code did not change. Differences in the `pyrUp` rows are measurement noise.
- Small images with 16 threads have a large spread between runs (for example
  `PyrDown_U8C3` 256: 35 µs to 81 µs for the same code). The spread comes from the .NET thread
  pool dispatch while other processes use the machine.
- `pyrDown` 8-bit and `float` 256 × 256 run on one thread in the port: the work estimate
  (`3·src + dst` elements per row) is below two stripes of `MinWorkPerStripe`. OpenCV splits
  these images across its own thread pool. A work estimate four times larger gave no repeatable
  gain in this environment. The estimate is unchanged.
- `PyrDown_U8` 1024 with 16 threads scales from 126 µs to 101 µs. OpenCvSharp scales from
  186 µs to 69 µs. The single-thread port is faster than OpenCvSharp. The cause of the weak
  scaling was not found; a quiet machine is necessary to examine it.
- IPP does not cause the gap that remains at 16 threads: the single-thread port is faster than
  OpenCvSharp for every `pyrDown` case at 1024 × 1024.

## Keypoints

- `KeyPoint` is a mutable `record struct` (`Pt`, `Size`, `Angle`, `Response`, `Octave`,
  `ClassId`). The parameterless constructor gives the OpenCV defaults (angle −1, class −1).
- `KeyPoint.Overlap` uses single precision as OpenCV; the centre distance is
  `(float)sqrt((double)dx² + (double)dy²)` (`cv::norm` of `Point2f`). `acos` and `sin` are
  `MathF.Acos` and `MathF.Sin` (the MSVC build of cv2 calls `acosf`/`sinf`). 309 golden pairs
  match bit for bit on the generating machine.
- `DMatch` ordering compares the distance only (`operator<`). `CompareTo` uses
  `float.CompareTo`, so sorts with NaN distances are defined.
- `KeyPointsFilter.RetainBest`: OpenCV calls `std::nth_element` and `std::partition`. The set of
  kept keypoints is defined (the top `n` plus every keypoint with the boundary response), but the
  order is implementation-defined. The port returns the kept keypoints sorted by decreasing
  response, with ties in input order. This order meets every OpenCV postcondition
  (element `n − 1` has the boundary response, earlier elements are not weaker, later elements
  are ties) and is the same on every platform. NaN responses rank after all numbers.
- `RunByImageBorder` rounds positions with `cvRound` (half to even), as OpenCV
  `Point_<int>(Point2f)`, and keeps positions in `[b, w − b) × [b, h − b)`.
- `RunByPixelsMask` reads the mask at `((int)(y + 0.5f), (int)(x + 0.5f))`. OpenCV reads outside
  the matrix for positions outside the mask; the port throws `ArgumentOutOfRangeException`.
  A null mask is the OpenCV empty mask (no change).
- `RemoveDuplicated` sorts indexes with OpenCV `KeyPoint_LessThan` (index tie-break, so the
  result is defined) and keeps input order. `RemoveDuplicatedSorted` uses `std::sort` in OpenCV;
  the port sorts stably. Keypoints that compare equal differ at most in the sign of zero members,
  so only the sign of zero can differ from an OpenCV build.
- NaN members make the OpenCV comparators invalid as strict weak orders. The port uses
  `float.CompareTo` (NaN first) in the duplicate sorts.

## Pooled temporaries (option 1)

- `PyrDownEngine` and `PyrUpEngine` already rent their row rings and row buffers per stripe.
  `pyrDown`, `pyrUp` and `buildPyramid` allocate only the returned levels, plus less than 1 KB of
  small per-call objects (row tables, the ring array, the delegate). The benchmark values were
  not changed by option 1: for example `PyrDown_U8` at 1024 allocates 256.7 KB, of which 256 KB
  is the result.
- `KeyPointsFilter` has internal span forms (`RetainBestSpan`, `RunByImageBorderSpan`,
  `RunByPixelsMaskSpan`) that compact the kept keypoints to the start of the span and return
  their count. The public `List<KeyPoint>` methods call them and set the list count. The kept
  set and order do not change. `RetainBest` rents its selection and result buffers instead of
  allocating them.

## Golden data (`tools/generate_pyramid_golden.py`)

- cv2 5.0.0 with IPP, OpenCL and threads off; 1712 image cases plus keypoint data.
- Float inputs are random normal values, so the two sum orders differ on many elements.
- Covered: all five depths with 1 to 4 channels; sizes 1×1, 2×2, 3×3, 1×5, 5×1 and odd and
  even sizes to 67 columns; float widths 16 to 29 (down) and 5 to 11 (up) for every channel
  count; custom `pyrDown` sizes (including the shifted right table) with `Reflect101` and
  `Replicate`; `pyrUp` sizes `2n ± 1`; `Reflect`, `Replicate` and `Wrap` borders; integer
  extremes; ROI views; `buildPyramid` with 3, 4 and 6 levels; large hashed inputs
  (1031×517, 640×481, 1024×1024) stored as SHA-256.
- cv2 has no `buildPyramid` binding. The generator applies `pyrDown` per level, which is what
  OpenCV `buildPyramid` does without IPP.
- cv2 has no `KeyPointsFilter` binding. `Features/KeyPointTests.cs` tests the filters
  analytically.
- No suspect cases.
