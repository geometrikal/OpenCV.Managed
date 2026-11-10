# Brute-force matcher port notes

Notes for the port of `cv::BFMatcher` (`Features/BFMatcher.cs`) and `cv::batchDistance`
(`Features/Internal/`). File references are against the OpenCV 5.0.0 submodule at `src/OpenCV`.

## Sources

| Port | OpenCV source |
|---|---|
| `Features/BFMatcher` | `modules/features/src/matchers.cpp` (`DescriptorMatcher::match`, `knnMatch`, `radiusMatch`, `checkMasks`, `convertMatches`, `BFMatcher::knnMatchImpl`, `BFMatcher::radiusMatchImpl`) |
| `Internal/BatchDistance` | `modules/core/src/batch_distance.cpp` (`batchDistance`, `BatchDistInvoker`, cross-check) |
| `Internal/BatchDistanceKernels` | `batch_distance.cpp` (`batchDistL1_`, `batchDistL2Sqr_`, `batchDistL2_`, `batchDistHamming`, `batchDistHamming2`); `modules/core/src/norm.dispatch.cpp` (`hal::normL2Sqr_`, `hal::normHamming` with `cellSize`, `popCountTable2`); `modules/core/src/stat.simd.hpp` (`hal::normHamming`); `modules/core/include/opencv2/core/base.hpp` (`normL1`, `normL2Sqr` templates); `intrin_sse.hpp` (`v_reduce_sum`) |
| `Core/Internal/MsvcSort` | MSVC STL `<algorithm>` (`std::sort`) |
| `Core/NormType.Hamming`, `Hamming2` | `modules/core/include/opencv2/core/base.hpp` (`NORM_HAMMING` = 6, `NORM_HAMMING2` = 7) |

## Public API

```csharp
new BFMatcher(NormType normType = NormType.L2, bool crossCheck = false)
DMatch[]   Match<T>(Mat<T> query, Mat<T> train, Mat<byte>? mask = null)
DMatch[][] KnnMatch<T>(Mat<T> query, Mat<T> train, int k, Mat<byte>? mask = null, bool compactResult = false)
DMatch[][] RadiusMatch<T>(Mat<T> query, Mat<T> train, float maxDistance, Mat<byte>? mask = null, bool compactResult = false)
void Add<T>(Mat<T> descriptors); void Clear(); int TrainSetCount; bool IsEmpty
DMatch[]   Match<T>(Mat<T> query, IReadOnlyList<Mat<byte>?>? masks = null)
DMatch[][] KnnMatch<T>(Mat<T> query, int k, IReadOnlyList<Mat<byte>?>? masks = null, bool compactResult = false)
DMatch[][] RadiusMatch<T>(Mat<T> query, float maxDistance, IReadOnlyList<Mat<byte>?>? masks = null, bool compactResult = false)
```

`T` is `byte` or `float`. One descriptor is one row of a single-channel matrix. Regions of
interest with a stride are accepted. The overloads with a `train` argument ignore the train
collection, as OpenCV (`clone(true)` plus `add`). The collection stores references; it does not
copy the matrices.

## Supported combinations

| Element | Norm | OpenCV function | Distance type |
|---|---|---|---|
| byte | `Hamming` | `batchDistHamming` | int |
| byte | `Hamming2` | `batchDistHamming2` | int |
| byte | `L1` | `batchDistL1_8u32s` | int |
| byte | `L2Sqr` | `batchDistL2Sqr_8u32f` | float |
| byte | `L2` | `batchDistL2_8u32f` | float |
| float | `L1` | `batchDistL1_32f` | float |
| float | `L2Sqr` | `batchDistL2Sqr_32f` | float |
| float | `L2` | `batchDistL2_32f` | float |

Other combinations throw `NotSupportedException`, as the OpenCV `StsUnsupportedFormat` error.
Int distances are converted to float for `DMatch.Distance` (`Mat::convertTo(CV_32F)`).

## Arithmetic and bit-exactness

- Hamming: `popcount(a ^ b)` over 64-bit words, then over the tail bytes. `Hamming2` counts the
  non-zero 2-bit cells: `popcount((x | x >> 1) & 0x55..55)`. This equals `popCountTable2`.
  Integer sums do not depend on the order.
- Byte L1: `SumAbsoluteDifferences` (SSE2) or a scalar sum. The int result is exact.
- Byte L2Sqr and L2: OpenCV `normL2Sqr<uchar, float>` adds `v·v` sequentially in float. For a
  descriptor length up to 258, every partial sum is an integer below 2^24, so the port uses an
  int sum and converts it once. Longer descriptors use the sequential float sum.
- Float L1: OpenCV `normL1<float, float>` (template, not the HAL function) is one sequential sum
  of `|a − b|`. The port keeps that order and runs four train rows together for
  instruction-level parallelism.
- Float L2Sqr and L2: OpenCV `hal::normL2Sqr_` on the SSE3 baseline uses four 4-lane
  accumulators over 16-element blocks. Element `j` goes to partial sum `j mod 16`. The lane sum
  is `((d0 + d1) + d2) + d3`, the reduction is `(s0 + s2) + (s1 + s3)` (SSE `v_reduce_sum`), and
  the tail adds sequentially. Multiply and add round separately (no FMA on the baseline). The
  port uses two `Vector256` accumulators (same partial sums) or four `Vector128` accumulators.
  Two train rows run together on `Vector256` hardware.
- `L2` is `MathF.Sqrt` of the squared distance (`std::sqrt(float)`).

## Order and tie rules

- k-nearest: `BatchDistInvoker` inserts each train distance with `d < dist[K−1]` and shifts while
  `dist[k] > d`. A later train descriptor with an equal distance never displaces an earlier one,
  so equal distances keep increasing train index. The comparison uses the raw 32-bit values as
  `int`; positive float bits order as integers.
- Masked pairs get `INT_MAX` or the bits of `FLT_MAX`. They never enter the k-nearest list. In
  `radiusMatch` they pass only when `maxDistance` is at least that value.
- Train collection: `update` adds `2^18` per train set; `trainIdx = idx & (2^18 − 1)`,
  `imgIdx = idx >> 18`. The k-nearest lists of later sets merge into the lists of earlier sets.
- Cross-check (`batchDistance` with `crosscheck`): needs `K = 1`, no mask and `update = 0`. The
  port throws `NotSupportedException` for `k > 1` (after `K = min(k, train rows)`), a mask, or a
  second train set, where OpenCV asserts. A query is kept only when it is the nearest query of
  its nearest train descriptor; among queries with equal distance to a train descriptor the first
  query wins.
- Radius match: the per-query lists are sorted with `std::sort` on `DMatch::operator<`. This sort
  is not stable. The cv2 Windows wheels are built with MSVC, so the port reproduces the MSVC
  `std::sort` (insertion sort up to 32 elements, median-of-three or Tukey ninther partition,
  heap sort after `1.5·log2(n)` unbalanced levels) in `Core/Internal/MsvcSort`. A libstdc++
  (Linux) build of OpenCV gives a different order for equal distances in lists longer than 16
  elements.
- `compactResult` in `radiusMatch`: OpenCV moves the non-empty lists to the front and does not
  remove the empty lists. The result has one list per query, with the empty lists last. The port
  keeps this behaviour. `knnMatch` with `compactResult` removes the empty lists.

## Deliberate differences

- A later train set with fewer rows than `k` and fewer rows than the first set makes OpenCV
  reallocate the distance matrix without initialization (undefined result). The port throws
  `NotSupportedException`.
- Cross-check with a float distance that is NaN or infinity leaves no nearest query in OpenCV,
  which then reads `dist[-1]`. The port skips such train descriptors.
- `DescriptorMatcher::create`, `read`/`write`, `clone` and the OpenCL paths are not ported.

## Performance structure

- `BatchDistance.Knn` and `Full` run query rows in parallel stripes (`ParallelRows.For`); each
  stripe rents one distance row. Output order is fixed by the query index, so the result does not
  depend on the thread count (`BFMatcherBasicTests.KnnMatch_Output_DoesNotDependOnThreadCount`).
- 32-byte Hamming descriptors load the query as four `ulong` values once per query.

## Golden data (`tools/generate_matcher_golden.py`)

- cv2 5.0.0, `OPENCV_CPU_DISABLE` set, IPP and OpenCL off, one thread.
- Byte and float descriptors with random values and with few distinct values plus duplicated
  train rows (many equal distances). Lengths 1 to 300 (byte) and 1 to 130 (float) cover the
  16-element block and the tails.
- `match` with and without mask and cross-check; `knnMatch` with k 1, 2, 7 (k above the train
  count included), masks and `compactResult`; `radiusMatch` at two radii with masks and
  `compactResult`; `sort_*` cases with 33 to 400 equal distances check the MSVC sort order;
  `coll_*` cases use three train sets with per-set masks.
- 2550 cases; all match exactly. No case is marked suspect.
