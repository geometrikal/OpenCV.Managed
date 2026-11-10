# Priority 6: optical flow, PnP, two-view geometry and anti-aliased drawing

**Date:** 2026-10-16 02:30:00

## Summary

The session ported the lower-priority functions for orthomosaic work: sparse optical flow,
the PnP solvers with `solvePnPRansac`, two-view epipolar geometry and anti-aliased drawing.
Outputs are bit-exact to cv2 5.0.0, with the documented exceptions below (4,348 golden cases).
The full test suite passes (35,873 tests). `putText` is not ported (see Open items).

## Changes

### Sparse optical flow (`Video/OpticalFlow`, new)

- `BuildOpticalFlowPyramid` (padded levels, Scharr derivatives, `tryReuseInputImage`, the
  OpenCV submatrix border rule) and `CalcOpticalFlowPyrLK` (image or pyramid inputs, all flags,
  `minEigThreshold`).
- The tracker reproduces the OpenCV SSE2 baseline lane order (`LkSse`). All 566 golden cases
  are bit-exact.

### PnP (`Calib/PnP`, new)

- `SolvePnP` with ITERATIVE, EPNP, P3P, AP3P, IPPE, IPPE_SQUARE and SQPNP; `SolvePnPGeneric`;
  `SolveP3P`; `SolvePnPRansac` (classic RANSAC); `SolvePnPRefineLM`; `SolvePnPRefineVVS`.
- `SolvePnPRansac` gives the cv2 inlier sets and poses bit for bit.
- `SolvePnPRefineVVS` with 13 or more points: cv2 sends `invert(DECOMP_SVD)` to LAPACK; the
  test tolerance is 1e-12 relative (measured below 1e-14).
- SQPnP is BSD 3-Clause (George Terzakis); `NOTICE` is updated.

### Two-view geometry (`Calib/EpipolarGeometry`, new)

- `FindFundamentalMat` (7-point, 8-point, RANSAC, LMEDS), `FindEssentialMat` (five-point
  RANSAC and LMEDS), `DecomposeEssentialMat`, `RecoverPose`, `TriangulatePoints`,
  `ComputeCorrespondEpilines`, `ConvertPointsToHomogeneous`/`FromHomogeneous`.
- `MultiModelRegistrator`: RANSAC and LMEDS for kernels with several models per sample.
- `FindEssentialMat`: all masks match; 41 of 45 cases are bit-exact, and in 4 cases one element
  of E differs by at most 1.2e-13. The likely cause is `SolvePoly`, which uses .NET `Math`
  functions; it is not confirmed.

### Anti-aliased drawing (`ImgProc/Drawing`)

- `LineType.AntiAliased` for every `Drawing` primitive (8-bit; other depths use 8-connected
  lines, as OpenCV does), `Drawing.ArrowedLine`, `Drawing.DrawMarker`, enum `HersheyFonts`,
  `TextMetrics.GetFontScaleFromHeight`. All 304 golden cases are byte-exact.
- `DrawingBasicTests`: three tests that required `NotSupportedException` for `LINE_AA` now
  require that `LINE_AA` is accepted.

### Performance

- LK tracker: whole-window SSE2 kernels with register accumulators; the sums keep the original
  order. Horizontal Scharr pass in SSE2. Pooled temporary pyramids (9.2 MB → 73 KB per call at
  1024).
- Fundamental matrix RANSAC error loop: 4 pairs per `Vector256<double>` step in the scalar
  order. `triangulatePoints`: allocation-free 4×4 Jacobi SVD that returns only the needed row.
- `LineAA` specialised per channel count with one base reference; stack buffers for ellipse
  points; short 3-channel runs written per pixel. 64 AA circles: 2.28 MB → 2.3 KB allocated.

### Benchmarks

New suites `OpticalFlowBenchmarks`, `PnpBenchmarks`, `EpipolarBenchmarks`,
`DrawingAaBenchmarks` and their OpenCvSharp twins. Final run on a quiet machine, port ÷
OpenCvSharp 4.13 (BenchmarkDotNet `--fast`):

| Case | Small, 1 thread | Large, 1 thread | Small, 16 threads | Large, 16 threads | Before optimisation (large, 1 thread) |
|---|---|---|---|---|---|
| CalcOpticalFlowPyrLK, 500 points (256/1024) | 1.20 | 1.08 | 2.33 | 1.70 | 2.38 |
| BuildOpticalFlowPyramid (256/1024) | 3.40 | 1.17 | 5.50 | 1.91 | 2.90 |
| FindFundamentalMat RANSAC (500/2000 points) | 0.56 | 0.41 | 0.45 | 0.42 | 1.86 |
| FindEssentialMat RANSAC (500/2000 points) | 1.10 | 1.18 | 1.09 | 1.18 | 1.10 |
| RecoverPose (500/2000 points) | 0.25 | 0.26 | 0.26 | 0.26 | 1.63 |
| SolvePnPRansac EPnP, 500 points | – | 0.79 | – | 0.82 | 1.24 |
| SolvePnP SQPnP / IPPE / EPnP, 100 points | – | 0.77 / 1.40 / 1.78 | – | 0.76 / 0.45 / 1.16 | 0.77 / 0.45 / 0.97 |
| LineAA 1 px, 64 lines (256/1024) | 0.93 | 1.51 | 0.98 | 0.99 | 1.41 |
| CircleAA outline, 64 circles | 0.78 | 0.84 | 1.20 | 0.65 | 1.13 |
| FillPolyAA | 1.30 | 1.39 | 1.03 | 1.16 | 1.40 |

Geometric mean, 1 thread / 16 threads: two-view geometry 0.52 / 0.50, PnP 1.20 / 0.72, AA
drawing 1.18 / 0.97, optical flow 1.50 / 2.54. OpenCV drawing and PnP are single-threaded, so
their 1-thread and 16-thread columns measure the same code; the differences between them (for
example EPnP 1.78 and 1.16) show the run-to-run variation. Results page:
https://claude.ai/artifact/D7rGGiR4dJnBS18FUKpqKP.

## Open items

- **`putText` and `getTextSize` are not ported.** OpenCV 5.0 does not use Hershey vector fonts.
  It maps each font identifier to the bundled Rubik TrueType font and renders it with a modified
  `stb_truetype` (about 5,000 lines of C: inflate, variable-font loader, rasteriser, layout). A
  port of the OpenCV 4.x Hershey tables is possible but does not match OpenCV 5.0 output. This
  needs a decision.
- Not ported: USAC variants, `correctMatches`, `sampsonDistance`, the distortion overloads of
  `findEssentialMat`/`recoverPose`, the `SparsePyrLKOpticalFlow` object.
- Optical flow with 16 threads on 256 × 256 images (2.3×) and `BuildOpticalFlowPyramid` at 256:
  each call writes about 667 KB of new large-object-heap buffers that belong to the returned
  pyramid. A reusable pyramid API would remove this.
- Image I/O is not ported; it waits for the ImageSharp licence decision.
