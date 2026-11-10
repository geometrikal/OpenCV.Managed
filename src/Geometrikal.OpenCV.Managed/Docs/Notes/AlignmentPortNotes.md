# Alignment port notes

Semantics of `TemplateMatching` (OpenCV `matchTemplate`), `PhaseCorrelation` (OpenCV
`phaseCorrelate`, `phaseCorrelateIterative`, `createHanningWindow`) and `EccAlignment` (OpenCV
`findTransformECC`, `findTransformECCWithMask`, `computeECC`), the bit-exactness rules, the
deliberate differences from OpenCV 5.0, and the golden constraints. Read this before you change
an alignment kernel or the alignment golden generator.

## Sources

- Port source: OpenCV 5.0.
  - `modules/imgproc/src/templmatch.cpp`: `matchTemplate`, `crossCorr`, `matchTemplateMask`,
    `common_matchTemplate`.
  - `modules/imgproc/src/phasecorr.cpp`: `phaseCorrelate`, `magSpectrums`, `fftShift`,
    `weightedCentroid`, `createHanningWindow`.
  - `modules/imgproc/src/phasecorr_iterative.cpp`: `phaseCorrelateIterative`.
  - `modules/core/src/dxt.cpp`: `divSpectrums`.
  - `modules/video/src/ecc.cpp`: `findTransformECCWithMask`, `computeECC`, Jacobian and update
    helpers.
  - `modules/imgproc/src/warp_kernels.simd.hpp`, `warp_common.scalar.hpp`,
    `warp_common.vector.hpp`: the float-coordinate warp kernels that ECC calls.
  - `modules/core/src/sum.simd.hpp`, `mean.dispatch.cpp`, `mean.simd.hpp`, `norm.simd.hpp`,
    `matmul.simd.hpp`, `matrix_expressions.cpp`, `arithm.simd.hpp`: summation orders and
    matrix-expression evaluation that the results depend on.
- Golden reference: cv2 5.0.0, SSE3 baseline, `OPENCV_CPU_DISABLE` for AVX and later, IPP and
  OpenCL off, one thread.
- Files (`ImgProc/Alignment/`):
  - `TemplateMatching.cs`, `TemplateMatchModes.cs`: public API and validation.
  - `PhaseCorrelation.cs`: public API.
  - `EccAlignment.cs`, `MotionTypes.cs`: public API and validation. `Core/TermCriteria.cs`
    holds `TermCriteria` and `TermCriteriaType`.
  - `Internal/TemplateCrossCorr.cs`: `crossCorr` with per-channel template spectra and the
    channel sum.
  - `Internal/TemplateNormalization.cs`: `common_matchTemplate`.
  - `Internal/TemplateMatchMask.cs`: `matchTemplateMask`.
  - `Internal/FloatReductions.cs`: OpenCV-order `sum`, `mean`, `meanStdDev`, `norm(L2SQR)` and
    `Mat::dot` for float matrices.
  - `Internal/PhaseCorrKernels.cs`: `magSpectrums`, `divSpectrums`, `fftShift`,
    `weightedCentroid`, the `v_cos` polynomial.
  - `Internal/PhaseCorrIterative.cs`: `phaseCorrelateIterative`.
  - `Internal/EccEngine.cs`: `findTransformECCWithMask` and `computeECC`.
  - `Internal/EccWarp.cs`: the OpenCV 5.0 float-coordinate warps for ECC.

## API

| Method | OpenCV equivalent |
|---|---|
| `TemplateMatching.MatchTemplate<T>(image, templ, method, Mat<byte>? mask = null)` | `matchTemplate(image, templ, result, method, mask)` with a `CV_8U` mask |
| `TemplateMatching.MatchTemplate<T>(image, templ, method, Mat<float> mask)` | `matchTemplate` with a `CV_32F` mask |
| `PhaseCorrelation.PhaseCorrelate<T>(src1, src2, window, out response)` | `phaseCorrelate(src1, src2, window, &response)` |
| `PhaseCorrelation.PhaseCorrelateIterative<T>(src1, src2, l2Size = 7, maxIters = 10)` | `phaseCorrelateIterative` (new in 5.0) |
| `PhaseCorrelation.CreateHanningWindow<T>(winSize)` | `createHanningWindow(dst, winSize, type)` |
| `EccAlignment.ComputeEcc<T>(templ, input, inputMask = null)` | `computeECC` |
| `EccAlignment.FindTransformEcc<T>(templ, input, warp, motion, criteria, inputMask, gaussFiltSize)` | `findTransformECC` |
| `EccAlignment.FindTransformEccWithMask<T>(templ, input, templMask, inputMask, warp, motion, criteria, gaussFiltSize)` | `findTransformECCWithMask` |

- `matchTemplate`: `byte` or `float`, 1 to 4 channels; the result is a new 1-channel float matrix.
- Phase correlation: 1-channel `float` or `double`.
- ECC: `byte`, `ushort`, `float` or `double`, 1 or 3 channels. The warp matrix is a `float`
  matrix of width 3 and height 2 (height 3 for a homography). The function updates it in place.
  A null `criteria` selects `EccAlignment.DefaultCriteria` (50 iterations, epsilon 0.001).
  Non-convergence throws `InvalidOperationException` (OpenCV `StsNoConv`).

## matchTemplate

### Path selection

- The OpenCV CPU path has no naive loop. `useNaive` and the naive `CCORR`/`SQDIFF` kernels
  exist only in the OpenCL path. Every CPU call uses the DFT block correlation `crossCorr`,
  for every template size. The port does the same.
- When both template dimensions are greater than or equal to the image dimensions, the inputs
  are swapped (no mask only). A template that is larger in one dimension and smaller in the
  other is rejected.

### Cross-correlation (`TemplateCrossCorr`)

- Block size, DFT size and tile grid are the `crossCorr` rules of `CrossCorrEngine`
  (block scale 4.5, minimum block 256, `getOptimalDFTSize`).
- Byte images use float32 spectra; float images use float64 spectra (`maxDepth`).
- Each image channel has its own template spectrum. For a 1-channel result, the channel
  correlations are converted to float and added in channel order (`add(src, cdst, cdst)`).
- Every tile window lies inside the image (anchor (0, 0), result size `W - w + 1`), so no
  border is extrapolated.
- `CrossCorrEngine` is not changed. Its tiles × channels parallel scheme cannot keep the channel
  sum order, so `TemplateCrossCorr` runs the tiles in parallel and the channels of one tile in
  order.

### Normalisation (`common_matchTemplate`)

- Window sums come from `IntegralImages` with float64 sum and squared sum.
- `mean` and `meanStdDev` of a float template use `FloatReductions`. OpenCV `sum_` for a
  1-channel float source adds groups of four elements in float before it adds the group to the
  double sum (`WT = float`), after a 2-lane double vector body over blocks of 12 elements.
  `Reductions.Mean` sums per row in double, which changes the last bits of the mean and the
  result of `CCOEFF` by up to 1e-5 relative.
- The rounding guards are ported: `diff2 <= min(0.5, 10·FLT_EPSILON·wndSum2)` gives 0, and a
  ratio within 1.125 of the bound is clamped to ±1. A template with `templNorm < DBL_EPSILON`
  gives a result of 1 for `CCOEFF_NORMED`.

### Mask path (`matchTemplateMask`, OpenCV 5.0)

- Byte image and template are converted to float. A byte mask is thresholded to 0/1; a float
  mask is used as a weight without thresholding. A 1-channel mask is merged to the template
  channel count.
- Each matrix expression is evaluated as `MatOp_AddEx`/`MatOp_Bin` evaluates it:
  - `-2·result + temp_result + s` is an in-place `addWeighted` (`dst == src1`). The 128-bit
    vector body computes `a·w1 + (b·w2 + d)` on blocks of 4 elements; the in-place call computes
    the remaining elements with the scalar form `(a·w1 + b·w2) + d`. Out-of-place calls would use
    an overlapping last vector. The port implements the in-place form.
  - `result /= norm_imgx * norm_templx` first evaluates the scaled product with `convertTo`
    (float scale), then divides.
  - `temp_res2 - 2·img_mask2_corr` is `scaleAdd(img_mask2_corr, -2, temp_res2)`.
  - The multi-channel `reduce(REDUCE_SUM)` sums the channels in float in channel order.
- `norm(…, NORM_L2SQR)` and `sum` of float matrices use `FloatReductions` (OpenCV 2-lane double
  accumulators, multiply then add).

## phaseCorrelate and createHanningWindow

- The spectra come from `SpectralTransforms` (CCS layout). `magSpectrums` computes float
  magnitudes in double; `divSpectrums` adds `FLT_EPSILON` or `DBL_EPSILON` to the denominators
  and computes float values in double, as dxt.cpp.
- `fftShift` (phasecorr.cpp) moves the quadrant that contains the centre of an odd size to the
  lower right. A row vector swaps two halves.
- `weightedCentroid` sums `x·v`, `y·v` and `v` in double in row order over the 5×5 window,
  clipped to the image. The response is the sum divided by the padded area.
- `createHanningWindow` computes the column factors of the even-length prefix with the OpenCV
  vector cosine `v_cos` for `v_float64` (Cephes polynomial, multiply then add in the SSE3
  baseline build). The last column of an odd width and all row factors use the C library
  cosine. The window is the float or double square root of the product.
- `phaseCorrelateIterative` uses the window, complex spectra, the cross-power spectrum in the
  element type, the quadrant swap of phasecorr_iterative.cpp (an odd last row or column stays in
  place), bilinear `resize` to 223×223, and image moments for the centroids.

## findTransformECC and computeECC

- Smoothing, gradients, mask blur, `meanStdDev`, Jacobians, projections, `invert` (LU) and
  `gemm` follow ecc.cpp in order. Euclidean Jacobian expressions (`-(X·h1) - (Y·h0)` and
  `(X·h0) - (Y·h1)`) select `add`, `subtract`, `scaleAdd` or `addWeighted` by the weights, as
  `MatOp_AddEx::assign`.
- The homography `hatX += h6` adds `Scalar(h6)`: with 3 channels only channel 0 changes, as in
  OpenCV.
- `Mat::dot` of float matrices is `dotProd_32f`: blocks of 8192 elements with four 4-lane float
  accumulators (multiply, then add), reduced to double per block, and a double tail in groups of
  four. Non-continuous blocks (Jacobian column ranges) are one plane per row.
- `norm(block)` squared with `pow(·, 2)` gives the Hessian diagonal; the row sums are the
  `NormL2_SIMD<float, double>` order.
- The Euclidean update calls `asin` with a float argument. The float overload (`asinf`) is
  selected, so the port uses `MathF.Asin`.
- Warps: OpenCV 5.0 `warpAffine`/`warpPerspective` for 32F (1 and 3 channels, bilinear) and 8UC1
  (nearest) use new kernels with single-precision source coordinates. `EccWarp` ports them:
  blocks of 8 pixels compute `M0·x + (y·M1 + M2)`; the remaining pixels of a row compute
  `(x·M0 + y·M1) + M2`; perspective coordinates are divided by `w` in float. `Geometry` warps use
  the 4.x table path (1/32 pixel), which moves the ECC result by about 1e-2 pixel.
- `computeECC`:
  - A 1-channel `byte`/`ushort` image is converted to a signed type, and the mean is rounded to
    an integer (`cvRound`) before the subtraction (the scalar becomes `CV_32S` work type).
  - A 3-channel `byte`/`ushort` image is not converted: OpenCV tests `type == CV_8U`, which is
    false for `CV_8UC3`. The differences saturate to the unsigned range (negative values become
    0). The port reproduces this.
  - Float images subtract `(float)mean`; double images subtract the double mean.

## Exactness

| Function | Result against cv2 5.0.0 |
|---|---|
| `matchTemplate`, all methods, 8U/32F, 1–4 channels, byte/float masks, 1 and several tiles | Bit-exact |
| `phaseCorrelate` (shift and response), with and without window, f32/f64 | Bit-exact |
| `createHanningWindow` f32/f64 | Bit-exact |
| `phaseCorrelateIterative` f32/f64 | Bit-exact |
| `computeECC` 8U/32F, 1/3 channels, mask | Bit-exact |
| `findTransformECC`/`WithMask`, all motion types, 8U/32F, 1/3 channels, masks, `gaussFiltSize` 1/5 | Bit-exact (warp and correlation) |

The ECC result depends on the CPU dispatch of cv2: with AVX2/FMA the 5.0 warp kernels and
`dotProd_32f` use fused multiply-add and 8 lanes, so the last bits can differ. The
port follows the SSE baseline.

## Deliberate differences from OpenCV 5.0

| Area | OpenCV 5.0 | This port |
|---|---|---|
| `phaseCorrelate` with a window and no padding | Multiplies the window into the caller's source arrays | Multiplies into copies; the sources do not change |
| `phaseCorrelate` with one column | Fails (ROI assertion in `fftShift`) | Throws `NotSupportedException` |
| `matchTemplate` with IPP | IPP `crossCorr`/`sqrDistance` when IPP is on | Not emulated (CPU path only) |
| `matchTemplate` mask of a different depth from 8U/32F | Rejected | Only `Mat<byte>` and `Mat<float>` overloads |
| ECC empty warp matrix | Initialised to identity | The warp matrix is required |
| ECC with AVX2/FMA dispatch | FMA-dependent last bits | SSE baseline results |

## Performance

- `TemplateCrossCorr` runs tiles in parallel (`ParallelRows.ForEachSerial`); each tile uses
  pooled buffers. `common_matchTemplate` runs result rows through `ParallelRows.For`.
- `EccWarp.Linear` runs destination rows in parallel. Each pixel has the same operation sequence
  for any thread count.
- The reductions (`FloatReductions`) and the mask path are serial, because their summation order
  defines the result.
- Results do not depend on the thread count (`AlignmentBasicTests` compares 1 and 8 threads).

### Measured status (October 2026)

No code change. Stopwatch harness, minimum of 30 batches, with the port and OpenCvSharp 4.13
in the same process. Times in µs.

| Case | Threads | Port | OpenCvSharp |
|---|---|---|---|
| PhaseCorrelate_F32 256 | 1 | 1489 | 1558 |
| PhaseCorrelate_F32 1024 | 1 | 19534 | 20171 |
| PhaseCorrelate_F32 256 | 16 | 947 | 1023 |
| MatchTemplate_CCoeffNormed_U8_T16 256 | 1 | 891 | 913 |
| MatchTemplate_CCoeffNormed_U8_T16 256 | 16 | 784 | 646 |
| MatchTemplate_CCoeffNormed_U8_T64 256 | 16 | 749 | 542 |

The earlier BenchmarkDotNet ratios for `PhaseCorrelate_F32` at 256 (3.8) and for matchTemplate
at 16 threads (4.7) did not reproduce on a quieter machine. At 256 × 256, neither library
scales well to 16 threads.

## Golden data (`tools/generate_alignment_golden.py`)

- Inputs are smooth textures (blurred seeded noise plus sinusoids), stored in the JSON.
- Templates are small and large; large images use several DFT tiles (8×8 template on 300×280,
  64×64 template on 400×380). Outputs above 4096 elements are stored as SHA-256.
- Masks: byte masks with values 0, 60, 120, 180 (binarised by OpenCV), float weights in
  [0, 1.5) with zeros; 1 channel and the template channel count.
- `phaseCorrelate` receives copies: OpenCV changes unpadded sources in place when a window is
  given. The 1×32 column case is not in the battery (OpenCV fails).
- Float results are stored as IEEE-754 bit patterns. No case is suspect.

## Pooled temporaries (option 1)

- `matchTemplate`: `TemplateNormalization` writes the integral images into pooled matrices
  (`IntegralImages.Integral` destination overloads). `TemplateCrossCorr` rents the padded
  template and the template spectra; the job returns the spectra after the last tile. The masked
  path rents the squared image, the correlation maps, the channel products and the
  denominators, and converts a byte image into a pooled float matrix. The score map is the only
  image-size allocation.
- `phaseCorrelate` rents the padded images and the two CCS spectra and reuses them for the
  product, the magnitude, the quotient, the inverse transform and the `fftShift` scratch copy.
  `PhaseCorrKernels.MagSpectrums`, `DivSpectrums` and `FftShift` write into given matrices.
  `phaseCorrelateIterative` rents the windowed images and the complex spectra; the cross-power
  spectrum and the inverse transform reuse them.
- `findTransformECC` rents the coordinate grids, the Jacobian, the warped images and mask, the
  zero-mean template, the error image and the Euclidean and homography Jacobian intermediates.
  The two Gaussian blurs write into pooled matrices (`Filters.GaussianBlurInto`), and the two
  gradients write into pooled matrices (internal `Filters.Filter2DInto`, which passes a
  destination to `Filter2DEngine`). A float input goes to the Gaussian blur without a copy
  (the blur reads a region of interest as an isolated image); a byte, ushort or double input
  converts into a pooled float matrix (`Conversions.ConvertInto`, internal).
  `computeECC` rents its zero-mean images.
- `FloatReductions.ConvertScale` is the float `convertTo` with a scale (copy when the scale is
  1 within `DBL_EPSILON`, else `x·(float)alpha + 0f`).
- Remaining temporaries: the blurred masks and the zero masks of `findTransformECC` with
  masks, the small per-iteration matrices of ECC (Hessian inverse, projections), the Hanning
  window of `phaseCorrelateIterative`, and template-size matrices of the masked
  `matchTemplate`.
- Allocated bytes per call, 1 thread, `AlignmentBenchmarks`:

| Case | Size | Before | After |
|---|---|---|---|
| MatchTemplate_CCoeffNormed_U8_T16 | 256 | 1.74 MB | 232 KB |
| MatchTemplate_CCoeffNormed_U8_T16 | 1024 | 20.48 MB | 4,042 KB |
| MatchTemplate_CCoeffNormed_U8_T64 | 256 | 1.66 MB | 152 KB |
| MatchTemplate_CCoeffNormed_U8_T64 | 1024 | 20.59 MB | 3,655 KB |
| PhaseCorrelate_F32 | 256 | 2.26 MB | 9.9 KB |
| PhaseCorrelate_F32 | 1024 | 36.04 MB | 30.5 KB |
| FindTransformEcc_Euclidean_F32 | 256 | 18.76 MB | 40.6 KB |
| FindTransformEcc_Euclidean_F32 | 1024 | 299.09 MB | 70.6 KB |

Before the pooled blurs and gradients, ECC allocated 1,065 KB (256) and 16,455 KB (1024).
