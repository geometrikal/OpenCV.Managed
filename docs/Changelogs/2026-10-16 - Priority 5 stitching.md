# Priority 5: stitching pipeline

**Date:** 2026-10-16 01:50:00

## Summary

The session ported the OpenCV 5.0 stitching module: rotation warpers, blenders, exposure
compensators, seam finders, stitching matchers, motion estimators, bundle adjusters, wave
correction and the `Stitcher` pipeline in PANORAMA and SCANS modes. The full test suite passes
(31,445 tests). An optimisation pass made blending and gain compensation faster and reduced
allocations, with identical output.

## Changes

### Warpers, blenders and exposure compensation (`Stitching/`, new)

- Warpers: `PlaneWarper`, `AffineWarper`, `CylindricalWarper`, `SphericalWarper` (base
  `RotationWarper`: warpPoint, warpPointBackward, buildMaps, warp, warpBackward, warpRoi).
- Blenders: `Blender`, `FeatherBlender`, `MultiBandBlender` (float or short weights), with the
  static helpers createWeightMap, normalizeUsingWeightMap, createLaplacePyr and
  restoreImageFromLaplacePyr.
- Exposure compensators: `GainCompensator`, `ChannelsCompensator`, `BlocksGainCompensator`,
  `BlocksChannelsCompensator`, `NoExposureCompensator`.
- `StitchingUtils`: overlapRoi, resultRoi, resultRoiIntersection, resultTl.
- Exactness: warper maps, blenders and compensators are bit-exact to cv2 5.0.0 (294 cases).
  Warped images with linear or cubic interpolation differ from cv2 5.0 `warp` by at most 1
  (4 at constant-border edges): `Geometry.Remap` follows the OpenCV 4.x table path, which is
  bit-exact to cv2 `convertMaps` + `remap`.

### Seam finders (`Stitching/SeamFinders`, new)

- `NoSeamFinder`, `VoronoiSeamFinder`, `DpSeamFinder` (COLOR, COLOR_GRAD), `GraphCutSeamFinder`
  (COST_COLOR, COST_COLOR_GRAD).
- `GcGraph`: exact port of the OpenCV Boykov–Kolmogorov max-flow (`GCGraph<float>`).
- `SeamPairSort` reproduces the MSVC `std::sort` order of image pairs. Grid layouts with equal
  pair distances depend on this order.
- All 119 golden cases are byte-exact. Single thread, the port is 3–4× faster than cv2 for
  Voronoi and DP, and level for graph cut.

### Registration and pipeline (`Stitching/`, new)

- `ImageFeatures`, `MatchesInfo`, `CameraParams`; matchers `BestOf2NearestMatcher`,
  `BestOf2NearestRangeMatcher`, `AffineBestOf2NearestMatcher`; estimators
  `HomographyBasedEstimator`, `AffineBasedEstimator`; bundle adjusters Reproj, Ray, Affine,
  AffinePartial and No; `WaveCorrection`; `MatchesGraph`; `Autocalibration`; `Stitcher`
  (PANORAMA and SCANS) with `IFeaturesFinder` (ORB, SIFT) and `WarperCreator`.
- Estimators, bundle adjusters and wave correction are bit-exact to cv2 5.0.0, except
  `BundleAdjusterAffine` with 5 images (relative tolerance 1e-5, measured 3.4e-7).
- **Deviation: FLANN is not ported.** OpenCV `CpuMatcher` uses FLANN (LSH for binary, k-d tree
  for float descriptors); LSH is random and gives different matches on each run. The port uses an
  exhaustive, deterministic `BFMatcher`. On real data the match lists differ from cv2.

### Performance

- `GainCompensator`: overlap sums with `Vector256.Sqrt`, one sum for self-overlap, pairs in
  parallel. 1024, 16 threads: 24.5 ms → 2.6 ms.
- `FeatherBlender`: fused weight scale and truncation; vector mask loops. 1024, 16 threads:
  34 ms → 17 ms; allocation 85 MB → 52 MB per call.
- `MultiBandBlender`: pooled pyramid temporaries; vector level difference and weight expansion.
  Allocation 212 MB → 81 MB per call.
- Rotation warper: pooled maps. `Geometry.Remap` 3-channel 8-bit inliers use `pmaddwd`
  (about 12 % faster).

### Benchmarks

- New suites `StitchingBenchmarks`, `SeamBenchmarks`, `StitchRegistrationBenchmarks`.
  OpenCvSharp does not expose these classes, so there is no OpenCvSharp reference, except the
  registration matcher twin, which crashes natively in its setup.

Port times after optimisation, BenchmarkDotNet `--fast`, milliseconds (no agents ran during the
16-thread part of this run; the 1-thread part overlapped with agent work):

| Case | 256, 1 thread | 1024, 1 thread | 256, 16 threads | 1024, 16 threads |
|---|---|---|---|---|
| PlaneWarper warp U8C3 | 1.07 | 21.4 | 0.49 | 9.7 |
| SphericalWarper buildMaps | 0.10 | 0.91 | 0.08 | 1.33 |
| FeatherBlender, 4 tiles | 3.2 | 25.8 | 5.3 | 48.5 |
| MultiBandBlender 5 bands, 4 tiles | 37.4 | 59.7 | 14.5 | 102.4 |
| GainCompensator, 4 tiles | 0.78 | 10.8 | 0.26 | 2.6 |
| Voronoi seam, 4 tiles | 0.76 | 29.6 | 0.70 | 29.2 |
| DP seam COLOR_GRAD, 4 tiles | 20.3 | 193 | 17.6 | 187 |
| Graph-cut seam COST_COLOR, 4 tiles | 88 | 9,230 | 72 | 8,986 |
| BestOf2NearestMatcher, 6 tiles | 28.9 | 55.5 | 6.5 | 5.8 |
| Homography estimator + BundleAdjusterRay | 28.4 | 10.5 | 21.2 | 9.1 |
| Affine matcher + BundleAdjusterAffinePartial | 23.6 | 41.0 | 7.6 | 7.8 |

The agent's back-to-back measurements before and after optimisation are in
`StitchingWarpBlendPortNotes.md`. Run-to-run variation on this machine is large (for example
MultiBandBlender at 1024: 41–102 ms with 16 threads). Results page:
https://claude.ai/artifact/D7rGGiR4dJnBS18FUKpqKP (groups "Stitching: …").

## Open items

- FLANN (LSH and k-d tree) is not ported.
- The 3-channel remap processes one pixel at a time on the outlier path (Reflect border).
- `MultiBandBlender` time is about two thirds pyrDown/pyrUp.
- Not ported: GPU paths, `LightGlueFeaturesMatcher`, `Timelapser`, the other warper types.
- `Stitcher` end-to-end output is tested against the synthetic scene, not against cv2.
