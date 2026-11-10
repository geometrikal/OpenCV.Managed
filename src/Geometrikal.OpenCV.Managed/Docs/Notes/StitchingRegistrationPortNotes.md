# Stitching Registration and Pipeline Port Notes

## Scope

The port covers the registration part of the OpenCV 5.0 `stitching` module and the high-level
`Stitcher`. Namespace: `Geometrikal.OpenCV.Managed.Stitching`.

| Managed API | OpenCV | OpenCV source |
|---|---|---|
| `ImageFeatures` (`Compute` for ORB and SIFT) | `detail::ImageFeatures`, `computeImageFeatures` | `matchers.hpp`, `matchers.cpp` |
| `MatchesInfo` | `detail::MatchesInfo` | `matchers.hpp`, `matchers.cpp` |
| `CameraParams` | `detail::CameraParams` | `camera.hpp`, `camera.cpp` |
| `FeaturesMatcher` (`Match` for one pair and for all pairs with a mask) | `detail::FeaturesMatcher`, `MatchPairsBody`, `CpuMatcher` | `matchers.cpp` |
| `BestOf2NearestMatcher` | `detail::BestOf2NearestMatcher` | `matchers.cpp` |
| `BestOf2NearestRangeMatcher` | `detail::BestOf2NearestRangeMatcher` | `matchers.cpp` |
| `AffineBestOf2NearestMatcher` | `detail::AffineBestOf2NearestMatcher` | `matchers.cpp` |
| `Estimator`, `HomographyBasedEstimator`, `AffineBasedEstimator` | `detail::Estimator` and subclasses | `motion_estimators.cpp` |
| `BundleAdjusterBase`, `BundleAdjusterReproj`, `BundleAdjusterRay`, `BundleAdjusterAffine`, `BundleAdjusterAffinePartial`, `NoBundleAdjuster` | `detail::BundleAdjuster*` | `motion_estimators.cpp` |
| `WaveCorrection.WaveCorrect`, `WaveCorrection.AutoDetectWaveCorrectKind`, enum `WaveCorrectKind` | `waveCorrect`, `autoDetectWaveCorrectKind` | `motion_estimators.cpp` |
| `MatchesGraph.LeaveBiggestComponent`, `MatchesGraph.MatchesGraphAsString` | `leaveBiggestComponent`, `matchesGraphAsString` | `motion_estimators.cpp` |
| `Autocalibration.FocalsFromHomography`, `EstimateFocal`, `CalibrateRotatingCamera` | `focalsFromHomography`, `estimateFocal`, `calibrateRotatingCamera` | `autocalib.cpp` |
| `Stitcher`, enums `StitcherMode`, `StitcherStatus` | `cv::Stitcher` | `stitching.hpp`, `stitcher.cpp` |
| `IFeaturesFinder`, `OrbFeaturesFinder`, `SiftFeaturesFinder`, `WarperCreator` and the four creators | `Feature2D` argument, `WarperCreator` | `stitching/warpers.hpp`, `matchers.cpp` |

Internals:

| File | Content |
|---|---|
| `Stitching/Internal/StitchGraph.cs` | `DisjointSets`, `GraphEdge`, `Graph` (`walkBreadthFirst`) from `util.cpp` and `util_inl.hpp` |
| `Stitching/Internal/SmallMat.cs` | `MatExpr` products and inverses as sequential `gemm` and `invert(DECOMP_LU)`; `float` conversions |
| `Calib/Internal/Rodrigues.cs` | `cv::Rodrigues` (vector to matrix and matrix to vector, no Jacobian) from `geometry/src/calibration_base.cpp` |
| `Calib/Internal/LevMarq.cs` | Extended: `stepNormTolerance` and `smallEnergyTolerance` parameters (`LevMarq::Settings`) |

The warpers, blenders, exposure compensators and seam finders are separate ports; see
`StitchingWarpBlendPortNotes.md` and `SeamFinderPortNotes.md`.

## Algorithms

- **Pairwise matching.** `FeaturesMatcher.Match(features, mask)` lists the pairs (i, j), i < j,
  where both images have keypoints and `mask(i, j)` is not 0. The range matcher limits j to
  `i + rangeWidth - 1`. Element `i·n + j` holds the result. Element `j·n + i` holds a copy with
  swapped query and train indices and `H⁻¹` (`invert(DECOMP_LU)`). The pairs run in parallel;
  each pair is independent because RANSAC creates its own generator.
- **Match filter.** A 1→2 match is kept when `d₁ < (1 - match_conf)·d₂` in `float`. A 2→1 match
  is kept with the same test when the 1→2 matches do not contain the same pair.
- **Homography matchers.** The points are relative to the image centres
  (`p - size·0.5f` in `float`). `findHomography(RANSAC, 3)` gives the mask. The pair is rejected
  when `|det H| < DBL_EPSILON`. `confidence = inliers / (8 + 0.3·matches)`, set to 0 when it is
  above `matches_confindece_thresh`. With at least `num_matches_thresh2` inliers, `H` is
  estimated again from the inliers.
- **Affine matcher.** Image coordinates, `estimateAffine2D` or `estimateAffinePartial2D` with
  the defaults, `H` extended with the row 0 0 1. The confidence has no upper threshold.
- **Estimators.** `findMaxSpanningTree` sorts the edges with a transform by inlier count
  (MSVC `std::sort` with `std::greater`, through `MsvcSort`), runs Kruskal and selects the
  centres by breadth-first distances from the leaves. `HomographyBasedEstimator` chains
  `R_to = R_from · K_from⁻¹ · H⁻¹ · K_to`; `AffineBasedEstimator` chains `R_to = R_from · H`.
  `estimateFocal` takes the median of `sqrt(f0·f1)` over all valid pairs, or the mean of
  `width + height` when fewer than n - 1 values exist.
- **Bundle adjusters.** Edges are the pairs i < j with confidence above `ConfThresh`. The
  residuals and the central-difference Jacobian follow `calcError`/`calcJacobian` (steps 1e-4,
  1e-3 for Ray). The solver is `LevMarq` with geodesic acceleration, `maxIterations =
  TermCriteria.MaxCount`, `stepNormTolerance = ε`, `smallEnergyTolerance = ε²`. The initial
  rotations are orthogonalized with a `float` SVD (`u·vt`, sign fix by the determinant) and
  converted with `Rodrigues` to a `float` vector. After the optimization every rotation is
  multiplied by the inverse rotation of the spanning tree centre (`float`).
- **Wave correction.** `float` arithmetic as OpenCV: outer products by `gemm`, `cv::eigen`
  (Jacobi), `Mat::cross` in `float`, the norm in `double`, the scaling by `convertTo` with a
  `float` factor, the dot products in `double`.
- **Stitcher.** `matchImages` → `leaveBiggestComponent` → estimator → `R` to `float` → bundle
  adjuster → median focal (`float`) → wave correction. `composePanorama` warps the seam images
  and masks, feeds and applies the exposure compensator, finds seams on `float` images, then
  warps each image at the compositing scale, applies the exposure compensation, dilates and
  resizes the seam mask (`INTER_LINEAR_EXACT`), and feeds the blender with `short` images.

## Bit-exactness

Golden data: `TestData/stitch_reg_golden.json` from `tools/generate_stitch_reg_golden.py`
(cv2 5.0.0, internal LAPACK, AVX/AVX2/FMA3/AVX-512 dispatch disabled, IPP, OpenCL and threads
off). The tile sets are synthetic crops and warps of a textured plane: affine (SCANS-like),
planar homography, yaw and pitch rotations, and a set with an unrelated tile. Each tile stores
its ORB keypoints and descriptors, so the managed side runs on identical features.

| Component | Result |
|---|---|
| Matchers against a Python reference of `matchers.cpp` with `cv2.BFMatcher` (matches, distances, masks, inlier counts, `H`, confidence, dual pairs) | Bit-exact, 17 cases |
| `HomographyBasedEstimator`, `AffineBasedEstimator` | Bit-exact |
| `BundleAdjusterRay`, `BundleAdjusterReproj` (3 images, full and partial refinement masks), `BundleAdjusterAffinePartial`, `NoBundleAdjuster`, `BundleAdjusterAffine` (4 images) | Bit-exact |
| `WaveCorrect` (horizontal, vertical, auto) | Bit-exact, 25 runs |
| `FocalsFromHomography`, `LeaveBiggestComponent`, `MatchesGraphAsString`, `CalibrateRotatingCamera` (failure path) | Exact |
| `BundleAdjusterAffine` with 5 images (30 parameters) | Relative tolerance 1e-5; measured maximum 3.4e-7 |

Rules that give the exact results:

- Every `MatExpr` product is a separate `gemm` with `alpha = 1`; every `inv()` is evaluated
  first by `invert(DECOMP_LU)` (`SmallMat`).
- `Rodrigues` computes in `double`; a `float` rotation is converted first, a `float` vector is
  rounded at the end.
- The adjusters keep the `CV_32F` rounding of `R` on input and output.
- `LevMarq` uses `MulTransposedR` for `JᵀJ` (fewer than 100 parameters), `gemm` for `Jᵀb` and
  `solve(DECOMP_SVD)`.

## Deviations and limits

- **FLANN.** OpenCV `CpuMatcher` uses `FlannBasedMatcher`: LSH for `CV_8U` descriptors and
  randomized k-d trees for `CV_32F` descriptors. FLANN is not ported. The port uses an
  exhaustive `BFMatcher` (Hamming for `byte`, L2 for `float`). The exhaustive search finds the
  true two nearest neighbours; LSH often does not. On the golden tile sets only 2 of 137 matched
  pairs have the same match list in cv2 and in the port; LSH misses close second neighbours, so
  cv2 passes more matches through the ratio test (for example 48 against 39). The cv2 matcher
  is also not repeatable: LSH builds random hash tables, and two cv2 runs differ in all 17
  cases. The port is deterministic. Downstream results (estimators, adjusters, wave correction)
  are exact when they receive the same matches.
- **SIFT byte descriptors.** OpenCV sends any `CV_8U` descriptor to LSH (Hamming). The port
  uses Hamming for all `byte` descriptors, which matches that choice.
- **LAPACK SVD.** cv2 routes `solve(DECOMP_SVD)` to LAPACK when a dimension is 25 or more. The
  port always uses the Jacobi SVD. Bundle adjustment results are bit-exact only when
  cameras × parameters per camera is below 25. Above that, well-conditioned problems agree to
  about 1e-7 relative. An ill-conditioned `BundleAdjusterReproj` problem with 5 images
  (35 parameters) converged to a different solution in cv2 (focal lengths differed by 1% to
  20%); this configuration is not in the golden battery.
- **100 or more parameters.** `JᵀJ` uses the `MulTransposedR` order. OpenCV uses `gemm` from
  100 parameters; results then differ in the last bits.
- **No residuals.** When no pair passes `ConfThresh`, the adjuster skips the solver and keeps
  the initial parameters (OpenCV runs `LevMarq` with zero residuals).
- **Errors.** OpenCV assertions are .NET exceptions: different descriptor types
  (`ArgumentException`), a disconnected spanning tree with more than two centres
  (`InvalidOperationException`), a wrong camera count (`ArgumentException`).
- **`CameraParams.R`** is always `Mat<double>`. The values equal the OpenCV values; the
  `CV_32F` stages round them to `float`.
- **`Stitcher.ComposePanorama`** keeps the seam images, so it can run more than once (OpenCV
  releases them).
- **Not ported:** GPU matchers, `LightGlueFeaturesMatcher`, `Timelapser`, the `UMat` paths.
- **Not tested against cv2:** `Stitcher` end to end (the tests check the geometry against the
  synthetic scene), `Stitcher.SetTransform`, the `CalibrateRotatingCamera` success path.

## Golden constraints

- The generator stops when cv2 is not 5.0.0, when AVX2/AVX-512 dispatch is active, or when cv2
  does not use the internal LAPACK. Set `CV2_500_DIR` to the cv2 directory.
- Every case except the cv2 FLANN matcher runs twice and must give equal outputs.
- The cv2-matcher comparison runs only on pairs where the cv2 and reference match lists agree.
- `cv2.detail.focalsFromHomography` does not return the values; the golden data uses a Python
  reference of `autocalib.cpp`. `leaveBiggestComponent` in cv2 returns only the indices.
- Some homographies give an infinite focal estimate; the focal values are stored as base64
  `double` values.

## Tests

- `Tests/Stitching/RegistrationGoldenTests.cs`: the golden cases above.
- `Tests/Stitching/StitcherBasicTests.cs`: SCANS mode on 4 tiles of a synthetic scene
  (panorama size within 6 pixels, mean absolute difference to the scene below 8 levels overall
  and in the 4-tile overlap), SCANS camera transforms, PANORAMA mode on 3 views of a rotating
  camera (focal within 10%, panorama size), and the error statuses.

## Performance

- The matchers run the pairs in parallel (`ParallelRows.ForEachSerial`); `BFMatcher` does the
  distance work.
- The bundle adjusters evaluate the residuals 2·p + 1 times per Jacobian. They allocate small
  `Mat` objects per edge and per evaluation; the cost is dominated by the residual loops.
- Benchmarks: `StitchRegistrationBenchmarks` (6 tiles of ORB features, sizes 256 and 1024:
  `BestOf2NearestMatcher_Match`, `HomographyEstimator_BundleAdjusterRay`,
  `AffineMatcher_BundleAdjusterAffinePartial`) and `OpenCvSharpStitchRegistrationBenchmarks`
  (the two matchers; OpenCvSharp does not expose the estimators or adjusters). OpenCV uses FLANN,
  so the matcher timings compare different search methods.

## Pooled temporaries (option 1)

- The bundle adjusters compute the per-edge 3 × 3 products in five scratch matrices of the
  adjuster: `SmallMat.MulInto` (`GemmKernels.Run`), `SmallMat.Inv3Into` (the explicit 3 × 3 formula
  of `invert(DECOMP_LU)`), `SmallMat.SetK` and the internal span overload of
  `Geometry.InvertAffineTransform`. The LevMarq callback rents the parameter backup.
- `LevMarq` rents the Jacobian (see `EstimationPortNotes.md`).
- `BestOf2NearestMatcher` and `AffineBestOf2NearestMatcher` rent the matched point arrays.

Allocated bytes per call, BenchmarkDotNet `--fast --inProcess`, 1 thread.

| Case | Size | Before | After |
|---|---|---|---|
| BestOf2NearestMatcher match | 256 | 1.66 MB | 1.53 MB |
| HomographyEstimator + BundleAdjusterRay | 256 | 7.92 MB | 0.91 MB |
| HomographyEstimator + BundleAdjusterRay | 1024 | 3.17 MB | 0.21 MB |
| AffineMatcher + BundleAdjusterAffinePartial | 256 | 1.93 MB | 1.27 MB |
| AffineMatcher + BundleAdjusterAffinePartial | 1024 | 2.80 MB | 1.56 MB |

The remaining bytes of the matchers come from `BFMatcher.KnnMatch` results (`Features`), the
`MatchesInfo` results and the per-pair estimation results.
