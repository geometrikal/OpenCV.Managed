# Seam finder port notes

Notes for the port of the OpenCV stitching seam finders (`Stitching/`): `NoSeamFinder`,
`VoronoiSeamFinder`, `DpSeamFinder`, `GraphCutSeamFinder` and the `GCGraph<float>` max-flow
graph. File references are against the OpenCV 5.0.0 submodule at `src/OpenCV`.

## Sources

| Port | OpenCV source |
|---|---|
| `Stitching/SeamFinders.cs` (`SeamFinder`, `NoSeamFinder`, `PairwiseSeamFinder`, `VoronoiSeamFinder`, `DpSeamFinder`, `GraphCutSeamFinder`, enums) | `modules/stitching/src/seam_finders.cpp`, `modules/stitching/include/opencv2/stitching/detail/seam_finders.hpp` |
| `Internal/SeamVoronoi` | `seam_finders.cpp` (`VoronoiSeamFinder::findInPair`) |
| `Internal/SeamDp` | `seam_finders.cpp` (`DpSeamFinder::process`, `findComponents`, `findEdges`, `resolveConflicts`, `computeGradients`, `hasOnlyOneNeighbor`, `closeToContour`, `getSeamTips`, `computeCosts`, `estimateSeam`, `updateLabelsUsingSeam`); `modules/core/include/opencv2/core/operations.hpp` (`partition`) |
| `Internal/SeamPairSort` | `DpSeamFinder::find` (`std::sort` with `ImagePairLess`); MSVC STL `<algorithm>` (`_Sort_unchecked` and helpers) |
| `Internal/SeamGraphCut` | `seam_finders.cpp` (`GraphCutSeamFinder::Impl`) |
| `Internal/GcGraph` | `modules/geometry/include/opencv2/geometry/detail/gcgraph.hpp` (`GCGraph<TWeight>`) |
| `Internal/SeamRoi` | `modules/stitching/src/util.cpp` (`overlapRoi`); `util_inl.hpp` (`normL2`, `sqr`) |

## Public API

```csharp
namespace Geometrikal.OpenCV.Managed.Stitching;

enum SeamFinderType { No, VoronoiSeam, DpSeam }
enum DpSeamCostFunction { Color, ColorGrad }
enum GraphCutCostType { Color, ColorGrad }

abstract class SeamFinder
{
    static SeamFinder CreateDefault(SeamFinderType type);
    abstract void Find<T>(IReadOnlyList<Mat<T>> src, IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> masks);
}
sealed class NoSeamFinder : SeamFinder
abstract class PairwiseSeamFinder : SeamFinder
sealed class VoronoiSeamFinder : PairwiseSeamFinder
{
    void Find(IReadOnlyList<Size> sizes, IReadOnlyList<Point> corners, IReadOnlyList<Mat<byte>> masks);
}
sealed class DpSeamFinder : SeamFinder
{
    DpSeamFinder(DpSeamCostFunction costFunction = Color);
    DpSeamCostFunction CostFunction { get; set; }
}
sealed class GraphCutSeamFinder : SeamFinder
{
    GraphCutSeamFinder(GraphCutCostType costType = ColorGrad, float terminalCost = 10000f, float badRegionPenalty = 1000f);
}
```

- `corners[i]` is the top-left corner of `src[i]` in panorama coordinates.
- `masks[i]` is a one-channel `byte` matrix with the size of `src[i]`. Non-zero marks a valid
  pixel. `Find` changes the masks in place. Images and masks can be regions of interest.
- The class hierarchy has private protected constructors. Code outside the library cannot add
  seam finders. `GraphCutSeamFinder` derives from `SeamFinder`, as in OpenCV (the OpenCV
  `Impl` is the pairwise class).

| Finder | Element type | Channels |
|---|---|---|
| `NoSeamFinder` | any | any |
| `VoronoiSeamFinder` | any (pixels are not read) | any |
| `DpSeamFinder` | `byte`, `float` | 3 or 4 (all images equal) |
| `GraphCutSeamFinder` | `float` | 3 |

## Semantics

### Pairwise order (Voronoi, graph cut)

`PairwiseSeamFinder::run` visits pairs (i, j), i < j, in index order. A pair runs only when
`overlapRoi` of the two image rectangles is not empty. The masks that a pair changes are the
inputs of the next pairs.

### Voronoi

- The window is the overlap rectangle plus a gap of 10 pixels on each side. Mask pixels
  outside an image are 0.
- `unique1 = submask1` with 0 where both masks are non-zero; the same for `unique2`.
- `dist1 = distanceTransform(unique1 == 0, DIST_L1, 3)`; the same for `dist2`.
- In the overlap rectangle: `dist1 < dist2` sets `mask2` to 0; otherwise `mask1` is set to 0.
  The write happens also where that mask is already 0 or where the other mask is 0.

### DP (`DpSeamFinder`)

- Pair order: all pairs (i, j), i < j, sorted by `std::sort` with the squared distance between
  the image centres (`corner + (cols / 2, rows / 2)`, integer), then reversed. See
  [Pair order](#pair-order-msvc-stdsort).
- Per pair: the union rectangle of the two images gets the two masks, the contour masks
  (4-neighbour border pixels of each mask), and labels: 4-connected components of
  "both masks", "first only" and "second only" in raster order of their first pixel.
- `findEdges`: the ordered pairs (ci, cj) of components with a 4-adjacency, in a sorted set.
- `resolveConflicts`: the loop takes the first edge (in set order) whose first component is an
  intersection and whose remaining state differs from the neighbour. With one neighbour, the
  component merges into the neighbour. Otherwise `getSeamTips` and `estimateSeam` cut it,
  `updateLabelsUsingSeam` moves the cut parts to the neighbour, and the state becomes
  `INTERS_FIRST` or `INTERS_SECOND`. The bounding box and contour of both components are
  recomputed inside their old bounding boxes only (OpenCV behaviour, kept).
- Final masks: `mask2` gets 0 where the label state has `FIRST` and `mask1` is non-zero;
  then `mask1` gets 0 where the label state has `SECOND` and the updated `mask2` is non-zero.
- Costs: `diffL2Square3`/`diffL2Square4` (first three channels). `byte`: integer squares, sum
  as `int`, then `float`. `float`: `float` squares summed left to right. `costColor = (d1 + d2) / 2`.
  `ColorGrad`: `costColor / (|g1a| + |g1b| + |g2a| + |g2b| + 1)` with 3x3 `CV_32F` Sobel
  derivatives of the BGR (or BGRA) to gray conversion. The bad-region cost is the squared norm
  `normL2(Point3f(255, 255, 255), Point3f(0, 0, 0)) = 195075`.
- `getSeamTips`: contour points of the intersection that are within 2 pixels (Chebyshev) of both
  mask contours and 4-adjacent to the neighbour. `cv::partition` with `ClosePoints(10)`
  (squared distance < 100). The two clusters with the largest distance between their
  `cvRound` centres (first maximum wins); in each, the first point nearest to the centre.
- `estimateSeam`: horizontal when `|dx| > |dy|`. The step minimum uses `std::pair<float, int>`
  order (cost, then step id 1, 2, 3; the first minimum wins).
- `updateLabelsUsingSeam`: `std::map` keys exist only after an access. Key 0 never reaches the
  label update, because a 0 mask value is skipped there.

### Graph cut (`GraphCutSeamFinder`)

- Per image: `dx = normL2(Sobel(src, CV_32F, 1, 0))` per pixel, a squared norm of the three
  channel derivatives; the same for `dy`. The port computes them only for `ColorGrad`; the
  colour cost does not read them.
- Window: overlap rectangle plus a gap of 10. Pixels outside an image have colour (0, 0, 0),
  mask 0 and gradient 0.
- Vertices in raster order; terminal weights `(mask1 ? terminal : 0, mask2 ? terminal : 0)`.
  Edges per pixel: right neighbour first, then the lower neighbour.
- `Color`: `w = normL2(c1(p) - c2(p)) + normL2(c1(q) - c2(q)) + 1`.
  `ColorGrad`: `w = (d(p) + d(q)) / (g1(p) + g1(q) + g2(p) + g2(q) + 1) + 1`.
  Add `badRegionPenalty` when one of the four mask values is 0.
- After `maxFlow`: a source-side pixel removes the pixel from `mask2` when `mask1` is
  non-zero; a sink-side pixel removes it from `mask1` when `mask2` is non-zero.

### `GcGraph` (`GCGraph<float>`)

Exact port of the OpenCV Boykov–Kolmogorov implementation: the active queue with the stub
vertex, the `ei ^ 1` reverse-edge trick, the two dummy edges at index 0 and 1, the `MIN` macro
order, the orphan stack (last in, first out) and the timestamp distance heuristic. Vertices
and edges are structs in pooled arrays. `next` is a vertex index; -1 is the null pointer and
index `vertexCount` is the stub.

## Bit-exactness

- All float operations keep the OpenCV operand order. The JIT does not contract multiply-add.
  cv2 5.0.0 on its SSE3 baseline does not contract either.
- `cvRound` uses `ScalarConverter.CvRound` (round half to even).
- `floodFill` on the label image (OpenCV 5.0 gradient fill with zero differences) fills the
  4-connected region of equal values. The port uses a scanline fill; the region is identical,
  and the fill order has no effect on the labels.
- The port computes the DP gradients once per image and reuses them for every pair. OpenCV
  recomputes them per pair with the same result.

### Pair order (MSVC `std::sort`)

`std::sort` is not stable, and grid layouts give many equal centre distances. The order of
equal pairs changes the masks (the dense grid golden scenes fail with a stable sort).
`SeamPairSort` reproduces the MSVC STL `std::sort` of the cv2 Windows wheels: insertion sort
for ranges of up to 32 elements, `_Partition_by_median_guess_unchecked` with median of three
(Tukey ninther above 40 elements) and a fat pivot, `ideal = ideal/2 + ideal/4`, and the heap
sort fallback. A cv2 build with libstdc++ (Linux) can order equal pairs differently for more
than 16 pairs.

## Deliberate differences

| Case | OpenCV 5.0 | Port |
|---|---|---|
| Invalid inputs (count mismatch, mask size, element type, channel count) | Assertion, or no error until a cost is computed (DP), or undefined reads (graph cut on `CV_8UC3`) | Checked before any mask changes: `ArgumentException` / `NotSupportedException` |
| `NoSeamFinder` | No checks | No checks |
| DP final mask update reads `mask1`/`mask2` outside the image | Unchecked `at<uchar>` | Treated as 0 (not reachable for valid labels) |
| DP `isAdjComp` index above the component count | Out-of-range write | Ignored (not reachable with fewer than 255 cut components) |
| Graph cut gradients with `COST_COLOR` | Computed, not read | Not computed |
| `DpSeamFinder(String)`, `GraphCutSeamFinder(String)` | String cost names | Enums |
| UMat / OpenCL | Supported | Not applicable |

## Performance structure

- Voronoi: pooled windows (`PooledMat`), one pass builds both distance-transform sources, row
  spans for the mask writes. About 4x faster than cv2 5.0.0 on the SSE3 baseline.
- DP: union-size buffers are reused across pairs; components use a scanline fill that also
  returns the bounding box; contours come from one raster pass; contour masks use
  `Vector<byte>` with an overlapping final vector; gradients are cached per image.
- Graph cut: one colour-difference value per window pixel (each edge reads two), pooled
  windows, and `GcGraph` with struct vertices and edges read through `ref` locals (OpenCV
  memory layout). The max-flow itself is sequential and dominates the run time.
- No step is parallel; the output does not depend on the thread count.

Stopwatch measurements (one thread, `SeamBenchmarks` inputs, cv2 5.0.0 with
`OPENCV_CPU_DISABLE` and IPP off; not BenchmarkDotNet results):

| Case | Port | cv2 |
|---|---|---|
| Voronoi 256 / 1024 | 1.0 ms / 11.7 ms | 4.0 ms / 41.7 ms |
| DP `COLOR_GRAD` 256 / 1024 | 17-22 ms / 260-320 ms | 28 ms / 627 ms |
| Graph cut `COST_COLOR` 256 / 1024 | 95 ms / 12.7 s | 95 ms / 12.5 s |

The port masks equal the cv2 masks on these inputs (mask pixel counts compared for all six
cases).

## Golden data (`tools/generate_seam_golden.py`)

- `TestData/seam_golden.json`: 19 scenes (2 to 25 tiles cut from seeded synthetic images, with
  per-tile gain and noise; full, polygon and holed masks; equal and different sizes; disjoint,
  contained and dense grid layouts) and 119 cases over Voronoi, DP (`COLOR`/`COLOR_GRAD`; float
  and byte; 3 and 4 channels) and graph cut (both costs; default and custom terminal cost and
  penalty). Every case matches cv2 5.0.0 byte for byte; no case is suspect.
- cv2 `detail_VoronoiSeamFinder()` has no constructor binding; `find` on it fails with an
  unknown C++ exception. The generator uses `detail.SeamFinder_createDefault(VORONOI_SEAM)`.
- Byte cases convert the float tiles with `np.clip(np.rint(x), 0, 255)`; 4-channel cases append
  alpha 255. The runner repeats these conversions.
- The dense grid scenes check the MSVC pair order (see above). They are valid for the Windows
  cv2 wheel.

## Pooled temporaries (option 1)

- `SeamDp` is disposable. It rents the gray image and the two Sobel derivatives of each image
  (gray conversion with `GrayFloatRow`/`GrayIntRow`, `Derivatives.SobelInto`), the union masks,
  the contour masks and the label map, and returns them at the end of `find`.
- `EstimateSeam` rents the cost, control and reachability arrays; `UpdateLabelsUsingSeam` rents
  its component mask.

Allocated bytes per call, BenchmarkDotNet `--fast --inProcess`, 1 thread.

| Case | Size | Before | After |
|---|---|---|---|
| DpSeamFinder COLOR_GRAD F32C3 | 256 | 6103.0 KB | 305.8 KB |
| DpSeamFinder COLOR_GRAD F32C3 | 1024 | 93803.9 KB | 1114.0 KB |

The Voronoi and graph-cut seam finders allocated less than 30 KB per call before the change.
