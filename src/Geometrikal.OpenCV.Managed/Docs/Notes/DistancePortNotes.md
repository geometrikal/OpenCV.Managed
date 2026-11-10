# Distance transform and watershed port notes

Semantics of `DistanceTransforms` (OpenCV `distanceTransform`, both overloads) and
`Segmentation.Watershed` (OpenCV `watershed`), the OpenCV 5.0 choices, the vectorization rules
that keep the output bit-exact, and the golden constraints. Read this before you change a
distance kernel, the watershed queue, or the distance golden generator.

## Sources

- Port source: OpenCV 5.0, `modules/imgproc/src/distransform.cpp` and
  `modules/imgproc/src/segmentation.cpp` (lines 49-325).
- Golden reference: cv2 4.5.5 with IPP off.
- OpenCV tests ported: `modules/imgproc/test/test_distancetransform.cpp`.
- Files:
  - `ImgProc/Distance/DistanceTransforms.cs`: public API, routing and validation.
  - `ImgProc/Distance/DistanceTypes.cs`, `DistanceTransformMasks.cs`,
    `DistanceTransformLabelTypes.cs`: enums with the OpenCV numeric values.
  - `ImgProc/Distance/DistanceTransformResult.cs`: result record of the labelled transform.
  - `ImgProc/Distance/Internal/ChamferDistance.cs`: 3x3, 5x5 and labelled 5x5 chamfer
    kernels, weights, label initialization, exact `uint` to `float` row store.
  - `ImgProc/Distance/Internal/PreciseDistance.cs`: `trueDistTrans` (column pass, row pass).
  - `ImgProc/Distance/Internal/DistanceL1Byte.cs`: `distanceATS_L1_8u`.
  - `ImgProc/Segmentation/Segmentation.cs`: public `Watershed`.
  - `ImgProc/Segmentation/Internal/WatershedFlood.cs`: initialization and priority flooding.

## API

| Method | OpenCV equivalent | Output |
|---|---|---|
| `DistanceTransform(source, distanceType, maskSize)` | `distanceTransform(src, dst, type, mask, CV_32F)` | New `Mat<float>` |
| `DistanceTransform(source, Mat<float> destination, distanceType, maskSize)` | same | Writes `destination` |
| `DistanceTransform(source, Mat<byte> destination, DistanceTypes.L1, maskSize)` | `distanceTransform(src, dst, DIST_L1, mask, CV_8U)` | Writes `destination`, saturated at 255 |
| `DistanceTransformWithLabels(source, distanceType, maskSize, labelType)` | `distanceTransform(src, dst, labels, type, mask, labelType)` | `DistanceTransformResult` |
| `DistanceTransformWithLabels(source, destination, labels, distanceType, maskSize, labelType)` | same | Writes `destination` and `labels` |
| `Segmentation.Watershed(image, markers)` | `watershed` | Writes `markers` in place |

- `source` is `Mat<byte>` with one channel. Every pixel gets the distance to the nearest zero
  pixel. The source is not modified.
- Destinations and `labels` must have the source size and one channel. They can be ROIs.
- The `byte` destination can be the source (in-place transform).
- `image` of `Watershed` is `Mat<byte>` with three channels. `markers` is `Mat<int>` with one
  channel and the image size. Both can be ROIs.
- The default `labelType` is `ConnectedComponent` (`DIST_LABEL_CCOMP`), as the C++ default.

## Routing

The port reproduces the OpenCV 5.0 routing (D5:1026-1199):

1. `byte` destination: the L1 8-bit transform. `maskSize` is not read. OpenCV selects this path
   for `DIST_L1` with `dstType == CV_8U` only.
2. Float destination without labels:
   1. `maskSize` must be 0, 3 or 5. Other values throw before any other check (OpenCV
      `StsBadSize`).
   2. L1 and C use the 3x3 mask for every `maskSize`, including 5 and precise.
   3. `maskSize == Precise`: the precise transform. The distance type is not checked; every
      value, including `L12`, `Fair`, `User` and undefined values, gives the Euclidean result.
   4. Otherwise the distance type must be L1, L2 or C.
3. Labelled transform: the label type must be `ConnectedComponent` or `Pixel`; the 5x5 mask is
   always used. `maskSize` is not read and not validated (OpenCV replaces it with 5 before the
   check). The distance type must be L1, L2 or C.

Reachable kernels: L1 3x3, C 3x3, L2 3x3, L2 5x5, precise; with labels C 5x5, L1 5x5, L2 5x5.

## Validation

| Condition | Exception |
|---|---|
| Null source, destination, labels, image or markers | `ArgumentNullException` |
| Multi-channel source, destination or labels; size mismatch | `ArgumentException` |
| `byte` destination with a type other than L1 | `ArgumentException` |
| Mask size other than 0, 3, 5 (no labels, float destination) | `ArgumentOutOfRangeException` (`maskSize`) |
| Chamfer mask with a type other than L1, L2, C | `ArgumentOutOfRangeException` (`distanceType`) |
| Label type other than 0 or 1 | `ArgumentOutOfRangeException` |
| Watershed image without three channels; markers not single-channel; size mismatch | `ArgumentException` |

## Chamfer kernels

- Weights are `cvRound(metric * 65536)` of the float metrics:

  | Type, mask | HV | DIAG | LONG |
  |---|---|---|---|
  | C 3x3 | 65536 | 65536 | - |
  | L1 3x3 | 65536 | 131072 | - |
  | L2 3x3 | 62587 | 89738 | - |
  | C 5x5 | 65536 | 65536 | 131072 |
  | L1 5x5 | 65536 | 131072 | 196608 |
  | L2 5x5 | 65536 | 91750 | 143976 |

- The temporary buffer is `uint`, padded by 1 (3x3) or 2 (5x5) pixels on each side, and pooled.
  The border value is `DIST_MAX = UINT_MAX - DIAG` (3x3) or `UINT_MAX - LONG` (5x5, labels).
  A border value plus a weight never overflows and never wins a minimum.
- Forward pass: `T = 0` for a zero pixel, else the minimum of the causal neighbours plus their
  weights, clamped to `DIST_MAX`. Backward pass: the minimum with the anti-causal neighbours.
- Output: `dst = (float)T * 2^-16`, with the exact (round-to-nearest-even) `uint` to `float`
  conversion. OpenCV 5.0 test `max_distance_*` requires this conversion for values above 2^31.

### Vectorization of the unlabelled kernels

- Each row runs in two stages. Stage 1 computes, for all pixels with `Vector<uint>`, the minimum
  of the neighbours in the adjacent rows (rows above in the forward pass, rows below in the
  backward pass). Stage 2 runs the horizontal recursion `left = min(T[j], left + HV)` in scalar
  code.
- Forward stage 1 writes 0 for zero pixels. Stage 2 then needs no source test: `min(0, x) = 0`.
- OpenCV skips the backward update when `t0 <= HV`. Every candidate is at least HV (HV is the
  smallest weight), so the unconditional minimum gives the same value.
- Stage 1 has no running state and the minimum is idempotent, so the final vector of a row
  overlaps the previous vector instead of a scalar tail.
- Row store: `hi + lo * 2^-16` with `hi = (float)(T >> 16)` and `lo = (float)(T & 0xFFFF)`.
  Both terms are exact, so the addition rounds once and equals `(float)T * 2^-16`.
- Rows narrower than `Vector<byte>.Count` use the scalar stage 1.

### Labelled kernel

- Label initialization: `ConnectedComponent` labels the 8-connected components of the zero
  pixels with `ConnectedComponentLabeling` (`Wu`, raster order of the first pixel of each
  component). `Pixel` numbers the zero pixels from 1 in raster order.
- Neighbours are tested in the OpenCV order with a strict comparison. The first neighbour that
  gives the minimum supplies the label. In the backward pass the current label wins ties. The
  order is load-bearing.
- Zero pixels keep their initial label. A pixel that no zero pixel reaches keeps distance
  `DIST_MAX` and label 0.
- The label buffer has the padded layout of the distance buffer. Border labels are never read
  into a result, because the matching border distance never wins.
- Vectorization: the seven neighbours in the adjacent rows are tested per lane in the OpenCV
  order (`Vector.GreaterThan` and `ConditionalSelect`). The horizontal neighbour is the last
  neighbour in both passes, so the scalar recursion tests it after the vector stage. The OpenCV
  `t0 > HV` gate is redundant with the strict comparison, for the reason given above.

## Precise transform

- Stage 1, columns: `d[j]` is the distance to the nearest zero pixel below, then
  `dist = min(dist + 1, d[j])` from the top; `dst = (float)sqr(dist)`, where `sqr(dist)` is
  `dist * dist` for `dist < min(rows, 65536)` and `UINT_MAX` otherwise.
- Stage 2, rows: the lower envelope of parabolas in float32, with the OpenCV operation order
  `(f[q] - f[p] + (float)(sqr[q] - sqr[p])) * inv[q - p]`, `inv[i] = (float)(0.5 / i)`, and
  `(float)(q + p) * (q - p)` for `q >= 65536`. Output `sqrt((float)sqr[|q - p|] + f[p])`.
- Distances of 65536 pixels or more give 65536. An image without zero pixels gives 65536.
- Stage 1 processes `Vector<byte>.Count` adjacent columns per block, one column per lane, so
  every source and destination access is a contiguous row segment. `(float)sqr(dist)` is the
  float product `(float)dist * (float)dist`: both operands are exact and the product rounds
  once, as the conversion of the exact integer square. The last block overlaps the previous
  block. Narrow images use a scalar column loop.
- Stage 2 is scalar (the envelope has data-dependent control flow).
- Threads: the column blocks and the rows run through `ParallelRows.For`, as the OpenCV
  `parallel_for_` over columns and rows. Columns and rows are independent, so the output does
  not depend on the thread count. The column pass splits into stripes only on large images; it
  is in `LargeImage_Output_DoesNotDependOnThreadCount`.

## L1 8-bit transform

- OpenCV `distanceATS_L1_8u` runs two raster passes over the 4-neighbourhood with
  `lut[x] = min(x + 1, 255)`. `lut` is monotone, so `lut[min(a, b)] = min(lut[a], lut[b])`, and
  the result is `min(city-block distance, 255)`. The brute-force test
  `DistanceTransform_L1AndC_MatchBruteForce` and the goldens confirm this.
- The port computes the same values with the separable form of the city-block distance:
  - Pass 1, rows: `d = s == 0 ? 0 : 255`, then `a = min(a + 1, d[x])` from the left and from the
    right. Rows are independent and run in parallel stripes.
  - Pass 2, columns: `d[y] = min(min(d[y-1], 254) + 1, d[y])` downwards, then the same upwards,
    on `Vector<byte>` lanes. Column chunks are independent and run in parallel stripes. The
    stripe with the last chunk covers the remaining columns with a final vector that overlaps
    its own last chunk (the update is idempotent).
- Pass 1 reads each source element before it writes the same destination element and reads no
  other source row, so the source can be the destination.
- The output does not depend on the thread count. The column pass splits into stripes only on
  large images; it is in `LargeImage_Output_DoesNotDependOnThreadCount`.

## Watershed

- Initialization: rows 0 and `h - 1` become -1 (WSHED). For each interior row, columns 0 and
  `w - 1` become -1 before the row scan. Negative interior values become 0. A 0 pixel with a
  positive 4-neighbour is queued at the minimum colour difference to its positive neighbours
  (order left, right, top, bottom) and marked -2 (IN_QUEUE).
- Colour difference: the maximum absolute channel difference, 0 to 255.
- Flooding: pop the head of the lowest non-empty queue. The label is the first positive
  neighbour label, or -1 when two positive neighbour labels differ. A labelled pixel queues its
  0 neighbours at their colour difference; the active queue can decrease.
- Seeds on the outer frame are overwritten before they are read and have no effect. Interior 0
  pixels that no seed reaches stay 0.
- The flood runs on packed rows: one pixel index `p` addresses marker `p` and image bytes `3p`.
  ROI inputs are copied to pooled packed buffers, and the markers are copied back. Continuous
  inputs are used in place.
- Queues: every pixel enters a queue at most once, so the queues are linked through one pooled
  `int` array indexed by pixel (`link[p + 1]` is the next node, 0 ends a queue). The 256 queue
  heads and tails follow the pixel links in the same array, so the flood loop needs one base
  reference for all queue state.
- The seed phase skips blocks of `Vector<int>.Count` markers in which no marker is negative and
  no 0 marker has a positive 4-neighbour. The seed phase writes only 0 and -2, never a positive
  value, so the test on the current markers equals the sequential test.
- Register pressure decides the speed of the flood loop. The loop is a separate method with few
  live values: it reads the neighbour markers again after it stores the label, as OpenCV, and
  uses `nint` indices. A version with the initialization and the flood in one method spilled
  the colour differences to the stack.
- The colour difference uses a branch-free absolute value and a branch-free maximum, and the
  active-level update a branch-free minimum. The JIT compiled `Math.Max`/`Math.Min` here to
  branches that the random differences mispredict; the branch-free forms made the flood about
  25% faster. (In the chamfer and L1 recursions the opposite holds: the branches predict well
  and branch-free minimums were up to 60% slower.)
- Rejected variants (measured slower or equal): 12-byte nodes with separate marker and image
  offsets and a free list (the OpenCV layout), 8-byte nodes with a free list, branch-free queue
  appends to a discard queue, precomputed horizontal and vertical difference planes, software
  prefetch of the next queued pixel.
- Single-threaded, as OpenCV.

## Performance

`DistanceBenchmarks` (input `Inputs.SparseZeros`, 0.1% zero pixels; watershed on
`Inputs.Color` with 200 seeds). Time ratio managed / cv2 4.5.5, measured in one session while
other benchmarks ran on the machine (indicative only):

| Case | 256, 1 thread / baseline | 1024, 1 thread / baseline | 256, all threads / default | 1024, all threads / default |
|---|---|---|---|---|
| L2Mask3 | 0.58 | 0.64 | 0.51 | 0.42 |
| L2Mask5 | 0.49 | 0.32 | 0.56 (cv2 IPP) | 0.60 (cv2 IPP) |
| L2Precise | 0.82 | 0.42 | 0.54 | 0.53 |
| L1Mask3 | 0.77 | 0.38 | 0.45 | 0.36 |
| L1Byte | 0.39 | 0.30 | 0.30 (cv2 IPP) | 0.19 (cv2 IPP) |
| LabelsCcomp | 0.42 | 0.35 | 0.26 | 0.38 |
| LabelsPixel | 0.47 | 0.43 | 0.43 | 0.52 |
| Watershed | 0.96 | 0.94 | 0.90 | 0.95 |

- cv2 baseline: one thread, no IPP, SSE dispatch only. cv2 default: all threads, IPP, AVX2.
- The float chamfer kernels and watershed are sequential on both sides.

## OpenCV 5.0 choices

| Input | cv2 4.5.5 | OpenCV 5.0 (port) |
|---|---|---|
| Chamfer distance of 8192 pixels or more | Clamped to 8192 | Fixed-point sum, clamped to `DIST_MAX` |
| No zero pixel, L2 3x3 | 8192 | 65534.62890625 |
| No zero pixel, L2 5x5 or labelled L2 | 8192 | 65533.8046875 |
| No zero pixel, L1 3x3 / C 3x3 | 8192 | 65534 / 65535 |
| No zero pixel, labelled L1 / C | 8192 | 65533 / 65534 |
| No zero pixel, precise | 31622776 | 65536 |
| Precise distance of 65536 pixels or more | sqrt of float squares | 65536 |
| Precise squares above 2^24 (sides above about 2896) | Float squares, rounding errors | Exact integer squares |

- Sources: commit `d752bac43f` (PR 24234, 4.9.0) for the chamfer border value; commit
  `ddcbd2cc26` (PR 22798, 4.7.0) and `PRECISE_DIST_MAX` for the precise transform.
- The L1 8-bit transform and watershed did not change.
- `DistanceOpenCv5DifferenceTests` pins these values.

## Deliberate differences from OpenCV 5.0

- IPP is not ported. With IPP on, cv2 uses IPP for the 5x5 chamfer and the precise transform
  (the precise selection depends on the thread count and the image size, with differences up
  to 6.5e-5), and for the L1 8-bit transform (identical results on all tested images).
- The `byte` destination overload throws for types other than L1. OpenCV ignores `dstType` for
  those types and returns a float image, which a typed `byte` destination cannot hold.
- Error types are .NET exceptions (see [Validation](#validation)).
- `Mat<T>` cannot be empty; the empty-input cases of OpenCV do not apply.

## Golden constraints (`tools/generate_distance_golden.py`)

- Environment: one thread, IPP and OpenCL off.
- Battery: analytic images (single pixel, all zero, one zero column, frame, diagonal, equidistant
  zeros, tall image with distances above 255) and random images with 0.5%, 2%, 20% and 50% zero
  pixels at sizes from 1x7 to 70x131, every distance type with masks 3, 5 and 0, labelled
  transforms with both label types, the L1 8-bit transform, and precise transforms with L12,
  Fair and User types. Three large images store SHA-256 hashes. Watershed: 36 random cases
  (smooth, noisy, quantized images with many equal differences; seeds including 100000;
  negative interior values; dense seeds; sides of 3) and analytic cases (no markers, frame
  markers only, adjacent markers, negative interior, sides of 1 and 2, a 257x301 image with 200
  seeds).
- Every image with a zero pixel has distances below 8192 pixels and sides below 2896, so cv2
  4.5.5 and OpenCV 5.0 agree.
- Suspect cases: the 60 chamfer and precise cases on images without zero pixels
  (`no_zero_12x9_*`, `no_zero_3x40_*`, `single_fg_*`, excluding `dist8u`). The reason is the
  4.5.5 versus 5.0 unreachable value above.
- A scratch model of the OpenCV 5.0 kernels reproduced every non-suspect golden case and
  differed from every suspect case.
- The runner also replays each stored case with an ROI source and ROI destinations, and the L1
  8-bit cases in place.
