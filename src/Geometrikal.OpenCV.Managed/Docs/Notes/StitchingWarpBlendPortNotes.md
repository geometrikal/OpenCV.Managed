# Stitching warpers, blenders and exposure compensators: port notes

Semantics of the `Stitching` port of the OpenCV 5.0 `stitching` module: rotation warpers,
blenders, exposure compensators and the region utilities.

## Scope

| Managed API | OpenCV | Source files |
|---|---|---|
| `PlaneWarper`, `AffineWarper`, `CylindricalWarper`, `SphericalWarper` (base `RotationWarper`) | `detail::PlaneWarper`, `AffineWarper`, `CylindricalWarper`, `SphericalWarper`, `RotationWarperBase<P>`, `ProjectorBase` | `warpers.cpp`, `warpers.hpp`, `warpers_inl.hpp` |
| `Blender`, `FeatherBlender`, `MultiBandBlender`, enum `BlendWeightType` | `detail::Blender`, `FeatherBlender`, `MultiBandBlender` (CPU path) | `blenders.cpp`, `blenders.hpp` |
| `Blender.CreateWeightMap`, `NormalizeUsingWeightMap`, `CreateLaplacePyr`, `RestoreImageFromLaplacePyr` | `detail::createWeightMap`, `normalizeUsingWeightMap`, `createLaplacePyr`, `restoreImageFromLaplacePyr` | `blenders.cpp` |
| `ExposureCompensator`, `NoExposureCompensator`, `GainCompensator`, `ChannelsCompensator`, `BlocksCompensator`, `BlocksGainCompensator`, `BlocksChannelsCompensator`, enum `ExposureCompensatorType` | `detail::ExposureCompensator` and subclasses | `exposure_compensate.cpp`, `exposure_compensate.hpp` |
| `StitchingUtils` (`OverlapRoi`, `ResultRoi`, `ResultRoiIntersection`, `ResultTl`) | `detail::overlapRoi`, `resultRoi`, `resultRoiIntersection`, `resultTl` | `util.cpp` |

Internals: `Stitching/Internal/WarpProjectors.cs` (`ProjectorParams`, projector structs),
`WarpMaps.cs` (result regions, map building), `BlendKernels.cs` (accumulation and
normalisation loops), `BlendPyramids.cs` (Laplacian pyramids), `ExposureKernels.cs` (overlap
sums, similarity mask, gain-map multiply).

Not ported: the other warpers (fisheye, stereographic, compressed rectilinear, Panini, Mercator,
transverse Mercator, portrait variants), the OpenCL and CUDA paths, `GraphCutSeamFinder` and the
other seam finders (separate port), motion estimators, matchers and `Stitcher`.

## Warpers

- `ProjectorParams.SetCameraParams` follows `ProjectorBase::setCameraParams`: `K⁻¹` with
  `LinearAlgebra.Invert` (LU, closed form for 3 × 3), `R·K⁻¹` and `K·Rᵀ` with `LinearAlgebra.Gemm`
  in `float`. The results are bit-exact with OpenCV.
- `AffineWarper` splits `H` into `R` and `T` as `getRTfromHomogeneous` (`T = −Rᵀ·t` through
  `gemm` with `alpha = −1`). A pure translation `(tx, ty)` moves the warped region by
  `(−tx, −ty)`, as OpenCV.
- `AffineWarper.WarpBackward` is not overridden in OpenCV. It uses `H` as the rotation and a zero
  translation. The port keeps this behaviour.
- Result regions: the plane warper projects the four source corners; the cylindrical and
  spherical warpers project every border pixel (`detectResultRoiByBorder`); the spherical warper
  then extends the region to a pole when the pole projects inside the source image. The bounds use
  `std::min`/`std::max` semantics (a NaN keeps the old bound). The float-to-int conversion gives
  `INT_MIN` for out-of-range values, as C++ on x64.
- `BuildMaps` returns `Rect(dst_tl, dst_br)`: the width and height are one smaller than the map
  size, as OpenCV. `WarpRoi` returns the map size.
- `Warp` is `BuildMaps` followed by `Geometry.Remap` with the constant border value 0.

### Bit-exactness and vector paths

- All projections compute in `float` with the C++ operation order. The JIT does not contract
  multiply-add, and MSVC `/fp:precise` does not contract either.
- The trigonometric functions use `MathF.Sin`, `Cos`, `Atan2`, `Acos`, `Sqrt`. On the reference
  machine they give the same bits as the MSVC C runtime functions that cv2 calls; all point and
  map cases of the golden data match bit for bit.
- Plane maps run in `Vector<float>` lanes. The lane order is the scalar order:
  `((k0·u + k1·v) + k2·(1 − t2)) / z` with `u = col / scale − t0`.
- Spherical and cylindrical maps compute `sin(u)` and `cos(u)` once per column and the row
  terms (`sin(π − v)`, `cos(π − v)` or `v`) once per row. The function arguments are the same as
  in the per-pixel C++ code, so the values are identical. The remaining products and quotients run
  in lanes with the scalar order; `z ≤ 0` (or NaN) selects −1.
- Map rows run in `ParallelRows` stripes. Each element depends only on its own coordinates, so the
  result does not depend on the thread count.

### Remap difference

`Geometry.Remap` follows the OpenCV 4.x table path for bilinear and bicubic interpolation (see
`GeometryPortNotes.md`). cv2 5.0 remaps float maps with new float kernels. Consequences for
`Warp` and `WarpBackward`:

| Interpolation | Comparison with cv2 5.0.0 `warp` |
|---|---|
| Nearest | Bit-exact |
| Linear | Bit-exact with cv2 `remap` through `convertMaps` (`CV_16SC2`, the table path); the direct cv2 5.0 `warp` differs by at most 1 (non-constant borders) or 4 (constant border, pixels at the source edge) |
| Cubic | The direct cv2 5.0 `warp` differs by at most 1 |

## Blenders

- `Blender` (OpenCV `Blender::NO`), `FeatherBlender` and `MultiBandBlender` use `CV_16SC3`
  (`Mat<short>`, 3 channels) panoramas and 1-channel `byte` masks. `MultiBandBlender.Feed` also
  accepts 3-channel `byte` images (the OpenCV `CV_8UC3` path with 8-bit Gaussian levels).
- `Blend` returns the panorama with zero pixels outside the mask and releases the internal
  buffers; call `Prepare` again for the next panorama.
- Feather weights: `min(1, sharpness · distanceTransform(mask, L1, 3))`; the multiply runs in
  `double` (`Arithmetic.Multiply` with a scalar), as OpenCV.
- Accumulation: `dst += (short)(v · w)` for `float` weights and `dst += (short)((v · w) >> 8)` for
  `short` weights. The float-to-short conversion truncates to 32 bits and keeps the low 16 bits;
  the short additions wrap, as the C++ code.
- Normalisation: `(short)(v / (w + 1e-5f))` (`float`) or `(v · 256) / (w + 1)` (`short`).
  The destination mask is `w > 1e-5f` (`float`) or `w > 0` (`short`).
- `MultiBandBlender.Prepare` limits the bands to `ceil(log2(max(width, height)))` and pads the
  panorama to a multiple of `2^bands`. `Feed` pads each image with a reflected border of
  `3 · 2^bands` pixels, aligned to the band grid, and adds each band to the band ROI of the
  panorama pyramid. The pyramids use `Pyramids.PyrDown`/`PyrUp` (bit-exact with OpenCV).
- The `float` accumulation and normalisation loops run in `Vector<T>` lanes over the interleaved
  channels with a per-row expanded weight buffer (`ArrayPool`). The lane conversion equals the
  C++ conversion because the values stay below 2³¹ in magnitude (see the remarks in
  `BlendKernels`). The `short` loops are scalar. All loops run in `ParallelRows` stripes with
  independent rows.

All blender golden cases (feather and multi-band with 3 and 4 tiles, 0 to 5 bands, `float` and
`short` weights, `byte` and `short` sources) are bit-exact.

## Exposure compensators

- `GainCompensator` follows `singleFeed`: per overlapping pair, the count of pixels where both
  masks have their level (and the similarity mask is set), and the sums of the pixel intensities
  (1 channel) or L2 norms (3 channels) in `double`, row-major. The linear system (alpha 0.01,
  beta 100) is solved with `LinearAlgebra.Solve` (LU, `double`). cv2 5.0.0 wheels are built
  without Eigen, so OpenCV also uses LU; the gains are bit-exact.
- `Apply` multiplies by the gain with `Arithmetic.Multiply` and a scalar (`double` work type,
  as OpenCV `multiply` with a `double` scalar).
- Similarity masks (`SimilarityThreshold < 1`) follow `buildSimilarityMask`: the float difference
  scaled by `1/255f`, the norm in `double`, then a 3 × 3 erode and dilate. The masks are built on
  the first feed and kept by the instance, as OpenCV.
- `ChannelsCompensator` runs one `GainCompensator` per channel with the similarity masks of the
  3-channel images.
- `BlocksCompensator` divides each image into blocks, runs a gain or channels compensator on the
  blocks, smooths each gain map with `Filters.SepFilter2D` ([0.25 0.5 0.25], reflect-101) and, in
  `Apply`, resizes the map to the image size (`Geometry.Resize`, bilinear) and computes
  `saturate_cast<uchar>((float)v · gain)` (the OpenCV `CV_32F` work type of a mixed `multiply`).

### Deliberate difference

With `NrFeeds > 1`, OpenCV multiplies the fed images (and, for the block compensators, the
caller's images through the block views) by the intermediate gains. The port works on copies and
never modifies the fed images. The gains are the same.

## Golden constraints (`tools/generate_stitch_warp_golden.py`)

- cv2 5.0.0, IPP and OpenCL off, one thread, `OPENCV_CPU_DISABLE` removes the AVX/AVX2/FMA and
  AVX-512 dispatch. Output: `TestData/stitch_warp_golden.json` (294 cases).
- Inputs: synthetic 8-bit colour tiles (sine pattern plus seeded noise), elliptic masks with a
  ragged edge, known `K`/`R` (Rodrigues rotations) and 3 × 3 affine matrices.
- Float point results are stored as IEEE-754 bit patterns; maps above 4096 elements as SHA-256.
- `warp` outputs are stored raw so that the runner can compare them with the tolerances above.
  The Linear cases also store the cv2 table-path result (`convertMaps` + `remap`), which the
  port matches bit for bit. `warpBackward` cases store the Linear result (tolerance) and the
  Nearest result (bit-exact).
- Suspect cases (`BlenderGoldenTests.GoldenFile_SuspectCases_MatchAdjudicatedList`):
  `createLaplacePyr_i16_l1`, `createLaplacePyr_i16_l3`. OpenCV `createLaplacePyr` with a `CV_16S`
  source writes level 0 into the `UMat` that wraps the input array. Through the Python binding,
  the returned level 0 is the unmodified input and level 1 is not `pyrDown(input)`. The blender
  path (a `UMat` source) is correct and is covered by the multi-band `s16` cases.
  `BlenderBasicTests.CreateLaplacePyr_Short_LevelsRestoreSource` pins the port behaviour.

## Tests

- `Stitching/WarperGoldenTests`, `BlenderGoldenTests`, `ExposureGoldenTests`: golden replay.
- `Stitching/WarperBasicTests`: map equals the point-wise backward mapping (vector paths), warp
  size equals `WarpRoi`, thread-count independence on 1024 × 768 sources, argument checks.
- `Stitching/BlenderBasicTests`: Laplacian round trip, sequence errors, `Blender` overwrite rule,
  band cropping, thread-count independence of both blenders.
- `Stitching/ExposureBasicTests`: factory, gains without overlap, gain direction, argument checks.

## Benchmarks

`StitchingBenchmarks` (tile sizes 256 and 1024): `PlaneWarper.Warp` (`byte`, 3 channels),
`SphericalWarper.BuildMaps`, `FeatherBlender` and `MultiBandBlender` (5 bands) feed and blend of
four tiles that overlap by 30 %, `GainCompensator.Feed` of four tiles. OpenCvSharp does not
expose the `detail` warpers, blenders or compensators, so there is no OpenCvSharp twin.

## Performance

### Changes

- `GainCompensator`: `ExposureKernels.OverlapSums` computes the norms of four overlap pixels
  with one `Vector256` square root (correctly rounded, as `Math.Sqrt`) and adds them in pixel
  order. The order of the `double` sums does not change. An overlap of an image with itself
  computes one sum and uses it for both images. The overlap pairs run in parallel
  (`ParallelRows.ForEachSerial`). The gain system is built from the pair results in pair order.
- `Blender.CreateWeightMap`: `BlendKernels.ScaleAndTruncate` replaces `multiply` and
  `threshold(THRESH_TRUNC)` with one in-place pass. The product is computed in `double` and
  rounded to `float`, as the `multiply` of the port.
- `BlendKernels.WeightMask` uses vector compares. `BlendKernels.ZeroOutsideMask` skips blocks of
  32 pixels whose mask bytes are all 255. Both loops run in `ParallelRows` stripes.
- `MultiBandBlender.Feed`: the bordered source, the Laplacian levels and the weight pyramid are
  pooled matrices (`PooledMat<T>`). The feed returns them to the pool before it exits.
  `BlendPyramids.Restore` pools the up-sampled level. The 8-bit level difference and the
  per-channel weight expansion (`vpermps`) are vector loops.
- `RotationWarper.Warp`: the backward maps are pooled matrices.
- `Geometry.Remap`: 3-channel 8-bit bilinear inliers use `pmaddwd`/`phaddd` (see
  `GeometryPortNotes.md`).

### Results

BenchmarkDotNet `--fast --inProcess`, `StitchingBenchmarks`, µs, before → after. Other jobs ran
on the machine; differences below about 20 % are noise. Single-thread values are medians;
16-thread values are means.

| Case | 1 thread 256 | 1 thread 1024 | 16 threads 256 | 16 threads 1024 |
|---|---|---|---|---|
| `PlaneWarper_Warp_U8C3` | 1292 → 644 | 13426 → 10238 | 1211 → 1101 | 3534 → 2537 |
| `SphericalWarper_BuildMaps` | 134 → 98 | 1569 → 1558 | 141 → 93 | 765 → 1071 |
| `FeatherBlender_FeedBlend_4Tiles` | 5221 → 2247 | 59126 → 45771 | 4064 → 2185 | 34274 → 17399 |
| `MultiBandBlender5_FeedBlend_4Tiles` | 7837 → 6559 | 82799 → 71635 | 10188 → 6694 | 50776 → 41051 |
| `GainCompensator_Feed_4Tiles` | 1970 → 665 | 18566 → 9907 | 1469 → 184 | 24515 → 2643 |

Allocated bytes per call at 1024: feather 84.7 MB → 51.9 MB, multi-band 212 MB → 80.5 MB,
plane warp 12.7 MB → 3.5 MB.

### Remaining cost

- Multi-band blending: `pyrDown`/`pyrUp` of the feed regions and of the pyramid collapse take
  about two thirds of the time. For 8-bit 3-channel images the port pyramids are as fast as
  OpenCvSharp. The zeroed panorama pyramids of `Prepare` are part of the OpenCV algorithm.
- Feather blending: the four L1 distance transforms take about 8 ms at 1024.
- Gain compensation: the `double` sums of the norms are sequential, as in OpenCV. The add
  latency limits one overlap pair to about one pixel per nanosecond.
- `SphericalWarper.BuildMaps`: no change. The map write is memory-bound; the 16-thread values
  vary with the machine load.

## Pooled temporaries (option 1)

- `FeatherBlender.Feed` writes the weight map into a rented matrix (`DistanceTransform` with a
  destination).
- `MultiBandBlender.Feed` converts the mask to weights in a rented matrix (`ConvertScaleFloatOp`)
  and rents the `short` weight map.
- `FeatherBlender` and `MultiBandBlender` do not allocate the unused base-class panorama mask;
  `Blend` sets the result mask.
- `MultiBandBlender.Blend` returns level 0 without a copy when the panorama needs no padding.
- Object state: the feather weight sum, the pyramid levels 1..n and the band weights stay after
  `Blend` and are cleared and reused by the next `Prepare` of the same size.
- `RotationWarper.WarpBackward` rents its maps. The spherical and cylindrical map builders rent the
  `sin(u)` and `cos(u)` tables.

Allocated bytes per call, BenchmarkDotNet `--fast --inProcess`, 1 thread.

| Case | Size | Before | After |
|---|---|---|---|
| FeatherBlender feed/blend, 4 tiles | 256 | 3246.6 KB | 2036.6 KB |
| FeatherBlender feed/blend, 4 tiles | 1024 | 51876.9 KB | 32531.1 KB |
| MultiBandBlender 5 bands, 4 tiles | 256 | 5187.9 KB | 3965.4 KB |
| MultiBandBlender 5 bands, 4 tiles | 1024 | 80534.5 KB | 61126.2 KB |
| SphericalWarper buildMaps | 1024 | 7053.1 KB | 7045.3 KB |
| PlaneWarper warp U8C3 | 1024 | 3456.3 KB | 3456.5 KB |

The benchmark creates a new blender per call, so the reuse across `Prepare` calls does not show
in it. The remaining bytes are the panorama, the result mask, the pyramid and weight state and,
for `MultiBandBlender` with padding, the copy of the unpadded result region. OpenCV returns that
region as a view; the port returns a packed copy. The warper results are the maps and the image.
