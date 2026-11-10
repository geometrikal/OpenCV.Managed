# Geometric transforms port notes

Semantics of `Geometry` (OpenCV `resize`, `warpAffine`, `warpPerspective`, `remap`,
`convertMaps` and the transform-matrix helpers) and `MatrixTransforms` (OpenCV `flip`,
`transpose`, `rotate`), the bit-exactness rules, the deliberate differences from OpenCV 5.0 and
the golden constraints. Read this before you change a resize or warp kernel or the geometry
golden generator.

## Sources

- Port source: OpenCV 5.0, `modules/imgproc/src/resize.cpp`, `imgwarp.cpp`,
  `fixedpoint.inl.hpp`, `modules/geometry/src/geometry.cpp`, `modules/core/src/matrix_transform.cpp`,
  `lapack.cpp`, `matrix_decomp.cpp`.
- The bicubic warp kernel and its table come from OpenCV 4.5.5 `imgwarp.cpp`
  (`remapBicubic`, `BicubicTab`). OpenCV 5.0 removed them.
- Golden reference: cv2 4.5.5.
- Files:
  - `ImgProc/Geometry/Geometry.Resize.cs`, `Geometry.Warp.cs`, `Geometry.Remap.cs`,
    `Geometry.Matrices.cs`: public API and dispatch.
  - `ImgProc/Geometry/Internal/ResizeCoefficients.cs`, `ResizeGeneric.cs`, `ResizeRows8u.cs`, `ResizeLinearGather.cs`:
    separable resize (linear, cubic, Lanczos4, area enlargement).
  - `ImgProc/Geometry/Internal/ResizeArea.cs`, `ResizeNearest.cs`, `ResizeLinearExact.cs`.
  - `ImgProc/Geometry/Internal/InterTables.cs`, `WarpCoords.cs`, `RemapKernels.cs`,
    `WarpEngine.cs`: warp and remap path.
  - `ImgProc/Geometry/Internal/MatrixSolvers.cs`: LU solve, `mulTransposed`, Jacobi SVD.
  - `Core/MatrixTransforms.cs`, `Core/Internal/TransformKernels.cs`.

## API

| Method | OpenCV equivalent | Element types |
|---|---|---|
| `Resize<T>(src, dsize, fx, fy, interpolation)` | `resize` | see the resize table |
| `WarpAffine<T>(src, M, dsize, interpolation, flags, border, value)` | `warpAffine` | all `Saturate` types; `sbyte`/`int` nearest only |
| `WarpPerspective<T>(...)` | `warpPerspective` | as `WarpAffine` |
| `Remap<T>(src, mapX, mapY, ...)` | `remap` with `CV_32FC1` pair or `CV_32FC2` | as `WarpAffine` |
| `Remap<T>(src, xy, fractions, ...)` | `remap` with `CV_16SC2` (+ `CV_16UC1`) | as `WarpAffine` |
| `ConvertMapsToFixedPoint`, `ConvertMapsToFloat`, `ConvertMapsToFloatInterleaved` | `convertMaps` | `float`, `short`, `ushort` maps |
| `GetRotationMatrix2D`, `GetAffineTransform`, `GetPerspectiveTransform`, `InvertAffineTransform` | same names | `double` (and `float` for `InvertAffineTransform`) |
| `MatrixTransforms.Flip`, `Transpose`, `Rotate` | `flip`, `transpose`, `rotate` | all types; `Transpose`/`Rotate` with pixels of at most 32 bytes |

- `InterpolationFlags`, `WarpFlags` and `RotateFlags` use the OpenCV numeric values.
  `WarpFlags` holds `WARP_INVERSE_MAP` and `WARP_RELATIVE_MAP` as a separate parameter.
- `BorderType.Transparent` (`BORDER_TRANSPARENT`) is accepted by the warps and remap only.
  `CopyMakeBorder` and `BorderInterpolate` reject it.
- Overloads without a destination return a new matrix. With `BorderType.Transparent` the new
  matrix starts as zeros.
- Destination overloads of the warps, remap, flip, transpose and rotate accept a destination that
  shares its array with the source. The source is then copied to a pooled matrix first.
  `Resize(src, dst)` rejects a shared array.
- A 2 × 3 matrix is a `Mat<T>` of width 3 and height 2.

## Resize

### Dispatch (`hal::resize`)

| Condition | Path |
|---|---|
| Destination size equals source size | copy |
| `LinearExact`, `float`/`double` | `Linear` |
| `LinearExact`, integer scale 2 × 2, channels ≠ 2 | area, integer factor |
| `LinearExact`, other | bit-exact fixed point |
| `Nearest`, `NearestExact` | index tables, pixel gather |
| `Linear`, integer scale 2 × 2 | area, integer factor |
| `Area`, both scales ≥ 1, integer scales | area, integer factor |
| `Area`, both scales ≥ 1, non-integer | area decimation (at most 4 channels) |
| other | separable path (`Area` uses area coordinates with linear weights) |

- With `dsize`, the scale is `dsize.Width / width`. With `fx`, the scale is `fx` and the size is
  `cvRound(width · fx)`. The two forms give different pixels for the same size.
- "Integer scale" means `|1/inv − cvRound(1/inv)| < DBL_EPSILON` on both axes.

| Path | `byte` | `sbyte` | `ushort` | `short` | `int` | `float` | `double` |
|---|---|---|---|---|---|---|---|
| Nearest, NearestExact | yes | yes | yes | yes | yes | yes | yes |
| Linear, Cubic, Lanczos4, Area | yes | no | yes | yes | no | yes | yes |
| LinearExact | yes | yes | yes | yes | yes | Linear | Linear |

### Separable path

- Coefficients: `fx = (float)((dx + 0.5)·scale − 0.5)`, `sx = cvFloor(fx)`, `fx −= sx`. Area
  coordinates: `sx = cvFloor(dx·scale)`, `fx = (float)((dx + 1) − (sx + 1)/scale)`.
- `interpolateCubic` (A = −0.75) and `interpolateLanczos4` run in single precision; Lanczos4 adds
  3 to `x` in single precision before the double expression.
- 8-bit weights are `saturate_cast<short>(w · 2048)`; the work type is `int`. Other depths use
  `float` weights and a `float` work type (`double` for `double`).
- Horizontal taps outside the source repeat the first or last pixel of the same channel. Float
  border elements accumulate from 0 (`v = 0; v += …`).
- 8-bit vertical pass:
  - Linear: `((b0·(S0 >> 4)) >> 16) + ((b1·(S1 >> 4)) >> 16) + 2) >> 2`.
  - Cubic: the first `width / 8 · 8` elements of each row use single precision,
    `f(S0)·b0' + (f(S1)·b1' + (f(S2)·b2' + f(S3)·b3'))` with `bk' = bk / 2048²`, rounded with
    `cvRound`. The remaining elements use `(Σ Sk·bk + 2^21) >> 22`.
  - Lanczos4: `(Σ Sk·bk + 2^21) >> 22`, 32-bit wrapping sums.
- Float vertical pass (`width = dsize.Width · channels`):

  | Depth | Cubic | Lanczos4 |
  |---|---|---|
  | `ushort` | right to left below `width / 8 · 8` | left to right |
  | `short` | right to left below `width / 8 · 8` | right to left below `width / 8 · 8` |
  | `float` | right to left below `width / 4 · 4` | right to left below `width / 4 · 4` |
  | `double` | left to right | left to right |

  Elements above the block limit sum left to right. This is the vector-body and scalar-tail
  split of the 128-bit OpenCV build (see `ColorPortNotes.md`). The block sizes do not depend on
  the managed vector width.

### Other resize paths

- Nearest: `min(cvFloor(d · (1/inv)), size − 1)`.
- NearestExact (OpenCV 5.0): `f = (src/dst)·0.5`, index `min(floor(f), src − 1)`, then
  `f += src/dst`, in double precision.
- Area, integer factor: 2 × 2 cells of `byte`/`ushort`/`short` with 1, 3 or 4 channels use
  `(a + b + c + d + 2) >> 2`. 2 × 2 cells of `float` with 4 channels, and the first `w / 4 · 4`
  elements with 1 channel, use `((a + b) + (c + d))·0.25f`. Other cells sum in groups of four and
  multiply by `1/area` (single precision). Cells cut by the image edge use
  `(float)sum / count`, also for `double`.
- Area decimation: table `computeResizeAreaTab`; per source row a weighted row in `float`
  (`double` for `double`), then `sum = beta·buf` for the first source row of a destination row
  and `sum += beta·buf` for the others. The first destination row of the image starts from a
  zeroed sum (`0 + beta·buf`), as a single-range OpenCV run does; this keeps −0 results
  independent of the stripe split.
- LinearExact: OpenCV fixed-point types per depth (`ufixedpoint16` rows for `byte`,
  `fixedpoint32` for `sbyte` and `short`, `ufixedpoint32` for `ushort`, `fixedpoint64` for
  `int`), including the saturating adds and the rounded 64 × 64-bit product of `fixedpoint64`.

## Warps and remap

All depths and interpolations use the table-driven OpenCV 4.x path:

1. Coordinates on a 1/32-pixel grid:
   - `warpAffine`: `adelta[x] = cvRound(M0·x·1024)`, per row
     `X0 = cvRound((M1·y + M2)·1024) + 16` (512 for nearest), `X = (X0 + adelta[x]) >> 5`, with
     32-bit wrapping additions.
   - `warpPerspective`: per destination block of `bw0 = min(1024 / min(16, height), width)`
     columns, `X0 = M0·xb + M1·y + M2`; per pixel `W = W0 + M6·x1`, `W = W ≠ 0 ? 32/W : 0`,
     `X = cvRound(clamp((X0 + M0·x1)·W))`. A pixel with `W = 0` samples source pixel (0, 0).
   - Float maps: `cvRound(m · 32)`. Nearest: `cvRound(m)`.
   - Integer part `saturate_cast<short>(X >> 5)`, table index `(Y & 31)·32 + (X & 31)`.
2. Kernels with the 32 × 32 weight tables (`InterTables`): 16-bit weights for `byte`, cast
   `(sum + 16384) >> 15`; `float` weights otherwise.

- `saturate_cast<int>(double)` is `cvRound`: values outside the `int` range give `int.MinValue`.
- Without `WarpFlags.InverseMap`, `warpAffine` inverts the matrix with the OpenCV 2 × 3 formula
  and `warpPerspective` with the 3 × 3 cofactor formula. A singular matrix gives the zero matrix.
- Lanczos4 inlier sums start from 0; bicubic inlier sums start from the first row sum.
- Border paths follow `remapBilinear`, `remapBicubic` (4.5.5) and `remapLanczos4`. Constant
  borders write the value without blending when the whole neighbourhood is outside.

### Inherited OpenCV quirks

The port reproduces these OpenCV behaviours:

| Quirk | Effect |
|---|---|
| Table sum correction reads the flat table array | Linear entry 0 is `{32767, 0, 0, 1}`; 1 linear, 657 bicubic and 856 Lanczos4 entries are corrected |
| `NNDeltaTab_i` | Nearest remap with fixed-point maps adds 1 to a coordinate whose fraction is below 1/2 |
| Lanczos4 warp table | No 1e-6 guard; `x < FLT_EPSILON` gives the unit impulse at tap 3 |
| Perspective block start | The block start column enters the double expression |
| `invertAffineTransform` for `CV_32F` | Determinant and linear terms in single precision, translation in double precision |
| `getPerspectiveTransform` | The products `−x·u` are computed in single precision |
| Area integer factor, edge cells | Float division `(float)sum / count` for every depth |

## OpenCV 5.0 choices and deliberate differences

| Area | OpenCV 5.0 | This port |
|---|---|---|
| Nearest and bilinear warps/remap of 8U, 16U, 32F with 1, 3, 4 channels | New float-coordinate kernels, FMA and lane-width dependent | Table path for all types (4.x results) |
| Bicubic warps/remap | `genericWarp` float kernel | 4.5.5 table-driven `remapBicubic` |
| `AlgorithmHint` | Selects approximate 8U kernels | Not exposed; results are the accurate table path |
| `INTER_NEAREST_EXACT` | Double-precision accumulation | Same as 5.0 |
| `getPerspectiveTransform` degenerate input | SVD of `AᵀA` when LU fails or the residual is ≥ 1e-8 | Same as 5.0 (Jacobi SVD port); `DECOMP_SVD` as solve method is not ported |
| Bilinear remap with `BORDER_TRANSPARENT` | Renormalised weights for pixels whose base tap is inside | Same as 5.0 |
| `WARP_RELATIVE_MAP` | Supported by remap | Same as 5.0 |
| IPP | Used by cv2 for some resize cases | Not emulated |
| `warpPolar` | Exists | Ported (see below) |
| `linearPolar`, `logPolar` | Removed | Not ported |
| Fraction map of type `CV_16SC1` | Accepted | Only `ushort` fraction maps |
| `CV_16SC2` map without fractions, interpolating | Undefined result | Rejected |

The 5.0 float warp kernels give results that depend on the CPU (AVX2/FMA, vector width). A
deterministic port cannot match them on every machine; the table path gives the same results on
every machine and equals cv2 4.x.

## Polar remap

`Geometry.WarpPolar` ports `cv::warpPolar` (`imgwarp.cpp`) on top of the port's `Remap`:

- Forward maps are computed in `double` (`std::exp`, `std::cos`, `std::sin` as `Math`) and stored as
  `float`; the radius table is `float`.
- Inverse maps: one wrapped row (`BORDER_WRAP`) above and below the polar source; per row,
  `cartToPolar` on a `1 × width` matrix (blocks of 1024, the port's `MathKernels.CartToPolar32f`),
  `log(r + 1)` in `float` for the semi-log mode, then `(float)(ρ / Kmag)` and
  `(float)(φ / Kangle) + 1`.
- Without `WARP_FILL_OUTLIERS` the remap uses `BORDER_TRANSPARENT`; the port's new destination is
  zero, OpenCV's is uninitialised.
- `linearPolar` and `logPolar` are removed in OpenCV 5.0 and are not ported.

Golden data: `tools/generate_polar_golden.py` → `TestData/polar_golden.json` (280 cases: 8U one and
three channels, 32F, nearest/linear/cubic, linear and log, forward and inverse, default sizes, a
1100-column case for the 1024 block; all match bit for bit), `ImgProc/WarpPolarGoldenTests`.

## Flip, transpose, rotate

- `rotate` is `transpose` with reversed source rows (clockwise) or reversed source columns
  (counterclockwise), in one pass; 180 degrees is `flip(−1)`.
- Transpose and rotate move 16-byte rows of `n × n` pixel blocks (`n = 16 / pixel bytes` for
  pixels of 1, 2, 4 or 8 bytes) with interleave instructions (SSE2 or AdvSimd). Other pixel
  sizes use 16 × 16 pixel blocks of scalar copies.
- Horizontal flip copies the row and reverses it (`Span.Reverse`) for pixels of 1, 2, 4 and
  8 bytes; 3-byte pixels use a byte shuffle (SSSE3).

## Performance structure

- Every path runs destination rows through `ParallelRows.For`. Each stripe keeps its own row
  buffers and resizes or converts the source rows it needs, so results do not depend on the
  thread count.
- 8-bit horizontal resize: each destination element is an integer sum over precomputed taps.
  With SSSE3, four elements are gathered from one or more 16-byte loads by byte shuffles and
  summed with `pmaddwd`/`phaddd`.
- `float` horizontal linear and bicubic resize: with AVX2, groups of 8 destination elements take
  their taps from two 8-element windows with `vpermps`; the products and sums keep the scalar
  order. Groups whose taps do not fit a window use the scalar loop.
- Vertical resize passes, area accumulation, map conversion and warp coordinates are vector
  loops. Perspective coordinates use 4-lane double vectors (AVX) with `minpd`/`maxpd`
  operand order as `std::min`/`std::max` and `cvtpd2dq` rounding as `cvRound`.
- 8-bit single-channel bilinear inliers are processed four at a time with `pmaddwd`.
- 8-bit 3-channel bilinear inliers are processed one pixel at a time: one byte shuffle puts the
  2 × 2 neighbourhood of the three channels next to the weights, and `pmaddwd`/`phaddd` give the
  three integer sums. The sums equal the scalar sums. This path is about 12 % faster than the
  scalar loop (identity map, 1024 × 1024, single thread).

## Golden constraints (`tools/generate_geometry_golden.py`)

- IPP is off, `OPENCV_CPU_DISABLE` removes the AVX/AVX2/FMA dispatch, one thread.
- Suspect cases (version differences, listed in `GeometryGoldenTests` and pinned in
  `GeometryOpenCv5DifferenceTests`):
  - 9 `INTER_NEAREST_EXACT` cases where cv2 4.5.5 selects other source indices. Other
    `INTER_NEAREST_EXACT` cases that differ are not generated.
  - `perspectiveTransform_collinear`.
- Transparent bilinear warps and remaps avoid pixels where 5.0 renormalises: matrices and maps
  are chosen so that no base tap lies on the last source row or column.
- `WARP_RELATIVE_MAP` exists only in 5.0; analytic tests cover it.
- Wide resize cases build their source from a formula (`synth` in the generator,
  `GeometryGoldenTests.Synthetic`) and store the SHA-256 of the expected output.
