# Overview

The purpose of this project is to port some opencv functions to C#

## Porting instructions

Port the requested OpenCV function into the managed .NET library at:

`src/Geometrikal.OpenCV.Managed/Geometrikal.OpenCV.Managed.csproj`

The OpenCV source is available at:

`src/OpenCV`

This is a true managed C# port, not a P/Invoke wrapper. Use the existing `Mat<T>` abstraction (`Core/Mat.cs`), managed arrays, `Span<T>` and `ReadOnlySpan<T>`. Support stride and non-contiguous ROIs, avoid unnecessary allocations, and do not require disposal for normal matrices.

Ported so far (reuse these — do not recreate equivalents):

- `Core/`: `Mat<T>`, `Point`, `Rect`, `Size`, `Scalar`, `BorderType`,
  `Borders` (`BorderInterpolate`, `CopyMakeBorder` allocating and destination
  overloads; internal `BorderPermutation` for VBMI byte borders), and
  `Core/Internal/ScalarConverter` (OpenCV `scalarToRawData`/`cvRound`
  saturation semantics).
- `Core/` operations: `Arithmetic`, `Bitwise`, `Comparison`, `Conversions`
  (convertTo/convertScaleAbs/split/merge/LUT), `Reductions`
  (sum/mean/meanStdDev/minMaxLoc/countNonZero/norm/reduce), `Mat.CopyTo(dst,
  mask)`/`Mat.SetTo`. Internals: `Core/Internal/Saturate` (`saturate_cast` for
  all depths), `Elementwise` (row driver with masks, scalar rows and
  `ParallelRows`), `ArithmKernels`, `ConvertKernels`, `CompareKernels`,
  `ReductionKernels`. Notes: `Docs/Notes/CorePortNotes.md`.
- `ImgProc/Color/`: `ColorConversions.CvtColor` (reorder/alpha, gray, HSV,
  Lab, XYZ, YCrCb), enum `ColorConversionCode`. Internals: `LabTables`
  (softfloat-exact tables), `ColorKernels` (`IColorRow<T>` per conversion),
  `ColorVectors` (3/4-channel deinterleave and interleave with byte shuffles;
  reuse it for other interleaved-pixel loops), `ColorVectorKernels`,
  `LabVectorKernels`.
  Notes: `Docs/Notes/ColorPortNotes.md`.
- `ImgProc/Color/`: `Demosaicing` (Bayer bilinear BGR/BGRA/gray, VNG with the OpenCV 5.0
  reflection padding and SSE scale split, edge-aware; 8U/16U), enums `BayerPattern`/
  `DemosaicMethod`. Notes: `Docs/Notes/DemosaicingPortNotes.md`.
- `ImgProc/Threshold/`: `Thresholding` (threshold with Otsu/triangle, mask and
  dry-run; adaptiveThreshold), enums `ThresholdType`/`AdaptiveThresholdType`.
  `Core/Internal/ScalarConverter` also provides `CvFloor`/`CvCeil` (x86
  semantics). Notes: `Docs/Notes/ThresholdPortNotes.md`.
- `ImgProc/Geometry/`: `Geometry` (Resize: nearest, nearest-exact (OpenCV
  5.0), linear, linear-exact, cubic, area, Lanczos4; WarpAffine,
  WarpPerspective, Remap with float and fixed-point maps and
  `WarpFlags.RelativeMap`; ConvertMapsTo*; GetRotationMatrix2D,
  GetAffineTransform, GetPerspectiveTransform, InvertAffineTransform, WarpPolar
  linear/log), enums `InterpolationFlags`/`WarpFlags`/`WarpPolarMode`. Internals: `InterTables` (warp weight
  tables), `WarpCoords`, `RemapKernels`, `ResizeGeneric`/`ResizeRows8u`/
  `ResizeLinearGather`, `ResizeArea`, `ResizeLinearExact`, `MatrixSolvers` (LU,
  Jacobi SVD). `Core/MatrixTransforms` (Flip/Transpose/Rotate, enum
  `RotateFlags`), `Core/Point2f`, `BorderType.Transparent` (warps and remap
  only). Notes: `Docs/Notes/GeometryPortNotes.md`.
- `ImgProc/ConnectedComponents/`: `ConnectedComponentLabeling`
  (connectedComponents, connectedComponentsWithStats; int/ushort labels; all
  `CCL_*` numbering orders; OpenCV 5.0 ushort overflow rule; parallel
  stripes), enums `ConnectedComponentsAlgorithm`/`ConnectedComponentsStat`,
  records `ConnectedComponentsResult<TLabel>`/`ConnectedComponentsStatsResult<TLabel>`.
  Notes: `Docs/Notes/ConnectedComponentsPortNotes.md`.
- `Core/` spectral transforms: `SpectralTransforms` (dft/idft with CCS,
  complex output/input, rows, nonzeroRows; dct/idct; mulSpectrums; divSpectrums;
  getOptimalDFTSize), enums `DftFlags`/`DctFlags`. Internals: `DftPlan`
  (factorization, permutation, twiddles, plan cache), `DftKernels` (dxt.cpp
  kernels over scalar or `Vector<T>` lanes), `DftLanes`, `DftEngine` (row and
  column lane batches), `LaneTranspose` (row-to-lane transposes and complex
  de-interleave; reuse for other batch-of-lines algorithms),
  `SpectrumKernels`. Notes: `Docs/Notes/DftPortNotes.md`.
- `ImgProc/EdgeDetection/`: `EdgeDetection` (Canny image and dx/dy
  overloads, spatialGradient). Internals: `CannyEngine` (stripes, magnitude
  ring, NMS candidate mask, stripe-local and global hysteresis),
  `SobelGradient3x3` (fused 3x3 Sobel dx/dy; reuse it for other 3x3 gradient
  consumers). Notes: `Docs/Notes/EdgePortNotes.md`.
- `ImgProc/Distance/`: `DistanceTransforms` (distanceTransform float and
  byte destinations, distanceTransformWithLabels CCOMP/PIXEL; OpenCV 5.0
  routing quirks and unreachable values), enums `DistanceTypes`/
  `DistanceTransformMasks`/`DistanceTransformLabelTypes`, record
  `DistanceTransformResult`. Internals: `ChamferDistance` (vector neighbour
  stage plus scalar row step, exact uint to float store), `PreciseDistance`
  (column blocks in lanes, parallel rows), `DistanceL1Byte`.
  `ImgProc/Segmentation/`: `Segmentation.Watershed` (internal
  `WatershedFlood`). Notes: `Docs/Notes/DistancePortNotes.md`.
- `ImgProc/Blend/`: `Blending.BlendLinear`. Notes: `Docs/Notes/AdvancedFilteringPortNotes.md`.
- `ImgProc/Filtering/` advanced filters: `Filters.SqrBoxFilter`,
  `Filters.BilateralFilter`, `Filters.StackBlur`, the `Filters.Filter2D` DFT
  path (`Internal/CrossCorrEngine`, OpenCV `crossCorr`; reuse it for
  matchTemplate), `GuidedFilter` (ximgproc, reusable object) and
  `EdgePreservingFilters.GuidedFilter`. `BoxFilterEngine.FloatBox` is the fast
  float box mean; `BoxFilterEngine.FloatBoxes` runs several of them.
  `ParallelRows.ForEachSerial` runs independent work items in
  parallel with serial inner loops. Notes:
  `Docs/Notes/AdvancedFilteringPortNotes.md`.
- `ImgProc/Histograms/`: `Histograms` (calcHist 1-32 dims, uniform and
  non-uniform, byte/ushort/float, mask, accumulate; calcBackProject;
  compareHist; equalizeHist), `DenseHistogram` (N-d float histogram;
  `ToMat`/`FromMat` for 1-D/2-D), enum `HistCompMethods`, `Clahe`
  (byte/ushort, OpenCV 5.0 `BitShift`). `Core/IntegralImages` (integral: sum,
  sqsum, tilted, all OpenCV type combinations; allocating and destination overloads), `Conversions.Normalize`
  (`NormType.MinMax`, L1, L2, Inf, mask). Internals: `HistLayout`, `HistBins`
  (vector bin offsets), `BinSelect` (branch-free bin selection),
  `CalcHistEngine`, `TableBinsVbmi` (AVX-512 VBMI 2-D/3-D 8U counting),
  `BackProjectEngine`, `CompareHistKernels`, `ClaheEngine`,
  `Core/Internal/IntegralKernels`, `Core/Internal/ByteLut` (256-entry byte LUT
  with AVX-512 VBMI; reuse it for byte table lookups). Notes:
  `Docs/Notes/HistogramPortNotes.md`.
- `Core/` linear algebra: `LinearAlgebra` (gemm, determinant, invert, solve with
  LU/Cholesky/QR/SVD/EIG/NORMAL, eigen (Jacobi), SVDecomp/SVBackSubst, solveCubic,
  solvePoly, transform, perspectiveTransform, mulTransposed, setIdentity, trace,
  completeSymm), `Svd<T>` (compute, backSubst, solveZ), `Point2d`/`Point3f`/`Point3d`,
  enums `DecompTypes`/`GemmFlags`/`SvdFlags`. Internals: `LapackDecomp` (LU, Cholesky, QR,
  Jacobi, Jacobi SVD, SVBkSb; reuse for findHomography/fitEllipse), `GemmKernels`,
  `PointTransformKernels`, `PolySolvers`. Notes: `Docs/Notes/LinearAlgebraPortNotes.md`.
- `ImgProc/Pyramids/`: `Pyramids` (pyrDown, pyrUp, buildPyramid; 8U/16U/16S/32F/64F,
  any channel count; OpenCV 5.0 size and border rules and right-border quirks). Internals:
  `PyrDownEngine`/`PyrUpEngine` (row rings per `ParallelRows` stripe), `PyrRows`
  (5-tap sums, decimation, pixel interleave), `PyrColumns` (fixed-point casts; float sums in
  OpenCV SSE3 vector/scalar order per element). `Core/KeyPoint`, `Core/DMatch`,
  `Features/KeyPointsFilter`. Notes: `Docs/Notes/PyramidPortNotes.md`.
- `Features/` FAST and ORB: `Fast.Detect`, `FastFeatureDetector` (types 5_8/7_12/9_16,
  NMS, mask), `Orb` (Detect/Compute/DetectAndCompute; all `ORB::create` parameters, WTA_K
  2/3/4, Harris/FAST score, mask), enums `FastDetectorType`/`OrbScoreType`. Internals:
  `FastEngine` (Vector<byte> paired and arc tests, row stripes), `OrbEngine` (pyramid buffer,
  Harris, ICAngle, `FastAtan2` port of `atan_f32`, descriptors), `OrbPatterns`, `OrbRng`
  (minimal `cv::RNG`). Notes: `Docs/Notes/OrbPortNotes.md`.
- `Features/`: `Sift` (detect, compute, detectAndCompute with mask; nfeatures,
  nOctaveLayers, contrast/edge thresholds, sigma, precise upscale, float or byte
  descriptors), enum `SiftDescriptorType`. Internals: `SiftPyramid`, `SiftExtrema`,
  `SiftDescriptor`. `Core/Internal/FastMath` (`cv::fastAtan2`, `hal::fastAtan32f`,
  `hal::magnitude32f`, MSVC `hal::exp32f`; reuse it for other OpenCV float math).
  Notes: `Docs/Notes/SiftPortNotes.md`.
- `Features/`: `BFMatcher` (match/knnMatch/radiusMatch, masks, crossCheck, train
  collection; L1/L2/L2Sqr for byte and float, Hamming/Hamming2 for byte; OpenCV tie order),
  `NormType.Hamming`/`Hamming2`. Internals: `BatchDistance`, `BatchDistanceKernels` (popcount
  Hamming, SSE3-order float L2Sqr). `Core/Internal/MsvcSort` (MSVC `std::sort` element order;
  reuse it where OpenCV sorts with a non-stable comparator). Notes: `Docs/Notes/MatcherPortNotes.md`.
- `ImgProc/Corners/`: `Corners` (cornerMinEigenVal, cornerHarris with the vector/scalar
  split, cornerEigenValsAndVecs, preCornerDetect, goodFeaturesToTrack with quality, cornerSubPix),
  `Core/TermCriteria`. Internals: `CornerEngine`, `GoodFeatures`, `SubPixRefiner`,
  `RectSubPix` (getRectSubPix float paths). Notes: `Docs/Notes/CornersPortNotes.md`.
- `Calib/`: `GeometricEstimation` (estimateAffine2D, estimateAffinePartial2D,
  estimateTranslation2D, findHomography; methods 0/RANSAC/LMEDS/RHO; OpenCV 5.0 LevMarq
  refinement; Point2f and Point2d inputs; bit-exact masks and models), enum
  `RobustEstimationMethod` (USAC values unsupported), records `GeometricEstimationResult`/
  `TranslationEstimationResult`. Internals: `IEstimatorKernel` kernels, `RansacRegistrator`,
  `LmedsRegistrator`, `LevMarq` (reuse it for other OpenCV 5.0 LM refinements), `RhoEstimator`.
  Notes: `Docs/Notes/EstimationPortNotes.md`.
- `Calib/`: `Undistortion` (initUndistortRectifyMap 32FC1/32FC2/16SC2 with the SSE2 block
  order, undistort with OpenCV stripes, undistortPoints/undistortImagePoints with
  `TermCriteria`, getDefaultNewCameraMatrix, getOptimalNewCameraMatrix) and `CameraModel`
  (Rodrigues with jacobians, projectPoints with jacobian, solvePnP/SolvePnPRefine
  SOLVEPNP_ITERATIVE). Internals: `UndistortMapKernel`, `UndistortPointsKernel`,
  `CameraDistortion`, `CameraRodrigues`, `CameraProjection`, `CameraLevMarq`, `CameraPnP`.
  Notes: `Docs/Notes/UndistortPortNotes.md`.
- `ImgProc/Alignment/`: `TemplateMatching` (matchTemplate, all methods, 8U/32F, 1-4 channels,
  byte/float masks; DFT `crossCorr` path only, as the OpenCV CPU path), `PhaseCorrelation`
  (phaseCorrelate with window and response, phaseCorrelateIterative, createHanningWindow),
  `EccAlignment` (findTransformECC, findTransformECCWithMask, computeECC), enums
  `TemplateMatchModes`/`MotionTypes`. Internals: `TemplateCrossCorr`, `TemplateNormalization`,
  `TemplateMatchMask`, `FloatReductions` (OpenCV-order float sum/mean/meanStdDev/norm/dot;
  reuse it where cv2 float reductions must match exactly), `PhaseCorrKernels`,
  `PhaseCorrIterative`, `EccEngine`, `EccWarp` (OpenCV 5.0 float-coordinate warp kernels).
  Notes: `Docs/Notes/AlignmentPortNotes.md`.
- `Stitching/`: rotation warpers `PlaneWarper`, `AffineWarper`, `CylindricalWarper`,
  `SphericalWarper` (base `RotationWarper`; bit-exact maps), blenders `Blender`/
  `FeatherBlender`/`MultiBandBlender` (CV_16SC3, float or short weights), exposure
  compensators (`GainCompensator`, `ChannelsCompensator`, `BlocksGainCompensator`,
  `BlocksChannelsCompensator`), `StitchingUtils` (overlapRoi, resultRoi, resultTl). Internals:
  `WarpProjectors`, `WarpMaps`, `BlendKernels`, `BlendPyramids`, `ExposureKernels`.
  Notes: `Docs/Notes/StitchingWarpBlendPortNotes.md`.
- `Stitching/`: `SeamFinders` (`NoSeamFinder`, `VoronoiSeamFinder`, `DpSeamFinder`
  COLOR/COLOR_GRAD, `GraphCutSeamFinder` COST_COLOR/COST_COLOR_GRAD on 32FC3). Internals:
  `GcGraph` (exact `GCGraph<float>` Boykov–Kolmogorov max-flow; reuse it for grabCut),
  `SeamDp`, `SeamVoronoi`, `SeamGraphCut`, `SeamPairSort` (MSVC `std::sort` order), `SeamRoi`.
  Notes: `Docs/Notes/SeamFinderPortNotes.md`.
- `Stitching/` registration and pipeline: `ImageFeatures`, `MatchesInfo`, `CameraParams`,
  matchers `BestOf2NearestMatcher`/`BestOf2NearestRangeMatcher`/`AffineBestOf2NearestMatcher`
  (exhaustive `BFMatcher` instead of FLANN), estimators `HomographyBasedEstimator`/
  `AffineBasedEstimator`, bundle adjusters (Reproj, Ray, Affine, AffinePartial, No),
  `WaveCorrection`, `MatchesGraph`, `Autocalibration`, `Stitcher` (PANORAMA/SCANS) with
  `IFeaturesFinder` and `WarperCreator`. Internals: `StitchGraph`, `SmallMat` (MatExpr order),
  `Calib/Internal/Rodrigues`. Notes: `Docs/Notes/StitchingRegistrationPortNotes.md`.
- `Video/`: `OpticalFlow` (buildOpticalFlowPyramid with padded levels and the submatrix border
  rule; calcOpticalFlowPyrLK with image or pyramid inputs, all flags, minEigThreshold; bit-exact
  SSE-baseline lane order), `OpticalFlowPyramid`, record `OpticalFlowResult`, enum
  `OpticalFlowFlags`. Internals: `LkTracker`, `LkSse`, `LkScharrDeriv`, `LkPyramidBuilder`,
  `LkBorders`. Notes: `Docs/Notes/OpticalFlowPortNotes.md`.
- `Calib/`: `PnP` (solvePnP ITERATIVE/EPNP/P3P/AP3P/IPPE/IPPE_SQUARE/SQPNP, solvePnPGeneric,
  solveP3P, solvePnPRansac classic RANSAC with exact inlier sets, solvePnPRefineLM/VVS), enum
  `SolvePnPMethod` (OpenCV 5.0 values), records `PnPPose`/`PnPSolution`/`PnPRansacResult`.
  Internals: `Epnp`, `P3p`, `Ap3p`, `PnpPolynomials`, `Sqpnp`, `Ippe`, `PnpRansac`,
  `PnpRefineVvs`. Notes: `Docs/Notes/PnpPortNotes.md`.
- `Calib/`: `EpipolarGeometry` (findFundamentalMat 7POINT/8POINT/RANSAC/LMEDS, findEssentialMat
  five-point RANSAC/LMEDS, decomposeEssentialMat, recoverPose, triangulatePoints,
  computeCorrespondEpilines, convertPointsTo/FromHomogeneous). Internals: `MultiModelRegistrator`
  (RANSAC/LMEDS for several models per sample and any point type), `EssentialFivePointKernel`,
  `FundamentalKernel`, `TriangulationKernels`. Notes: `Docs/Notes/EpipolarPortNotes.md`.
- `ImgProc/Drawing/` anti-aliased drawing: `LineType.AntiAliased` for all `Drawing` primitives
  (8-bit; LINE_8 for other depths, as OpenCV), `Drawing.ArrowedLine`, `Drawing.DrawMarker`
  (enum `MarkerType`), enum `HersheyFonts`, `TextMetrics.GetFontScaleFromHeight`. `putText` and
  `getTextSize` are not ported: OpenCV 5.0 renders a TrueType font through stb_truetype.
  Notes: `Docs/Notes/DrawingAaPortNotes.md`.
- `Core/`: `MathFunctions` (magnitude, phase, cartToPolar, polarToCart, exp, log, pow, sqrt; OpenCV
  5.0 vector/scalar split for polarToCart and pow(−0.5)). Internals: `MathKernels` (Cephes
  `v_sincos` per lane, sin table, `rsqrtps` estimate, integer powers). Notes:
  `Docs/Notes/MathFunctionsPortNotes.md`.
- `ImgProc/Hough/`: `HoughTransforms` (HoughLines standard and multi-scale with `useEdgeValue`,
  HoughLinesP with the `cv::RNG` point order, HoughLinesPointSet, HoughCircles HOUGH_GRADIENT),
  records `HoughLine`, `LineSegment`, `HoughPointSetLine`, `HoughCircle`. Notes: `Docs/Notes/HoughPortNotes.md`.
- `ImgProc/Motion/`: `Accumulation` (accumulate, accumulateSquare, accumulateProduct,
  accumulateWeighted with the SSE vector/scalar β split). Notes: `Docs/Notes/AccumulationPortNotes.md`.
- `Photo/`: `Inpainting` (inpaint Telea and Navier-Stokes; 8U 1/3 channels, 16U, 32F), enum
  `InpaintMethod`. Notes: `Docs/Notes/InpaintPortNotes.md`.
  `Denoising` (fastNlMeansDenoising L2/L1 8U/16U 1-4 channels, fastNlMeansDenoisingColored).
  Notes: `Docs/Notes/DenoisingPortNotes.md`.
- `ImgProc/Contours/`: `Contours.FindContours` (full Suzuki–Abe port).
- `ImgProc/Shapes/`: `ShapeDescriptors` (approxPolyDP int/float with the OpenCV
  start point, stack order and clean-up; arcLength; contourArea; boundingRect
  for point sets and 8-bit masks; contour and image moments 8U/16U/16S/32F/64F
  with binary mode; HuMoments; matchShapes I1/I2/I3), record struct `Moments`,
  enum `ShapeMatchModes`. Internals: `Descr*` (tile-ordered image moments,
  parallel tiles with a serial sum). Notes: `Docs/Notes/ShapeDescriptorsPortNotes.md`.
- `ImgProc/Shapes/`: `ShapeFitting` (convexHull points/indices, isContourConvex,
  convexityDefects, minAreaRect (5.0 angle in [-90, 0)), boxPoints, minEnclosingCircle
  (5.0 seeded shuffle), fitEllipse/AMS/Direct, fitLine 2-D, pointPolygonTest), records
  `ConvexityDefect`/`EnclosingCircle`/`FittedLine`; `ShapeIntersection` (intersectConvexConvex,
  rotatedRectangleIntersection; enum `RectanglesIntersectType`); `Core/RotatedRect`, `Core/Size2f`,
  `Core/Internal/CvRng` (`cv::RNG` and per-thread `theRNG`; reuse it for RNG-dependent ports).
  Internals: `HullSklansky`, `CalipersMinAreaRect`, `FitEnclosingCircle`, `FitEllipseCore`,
  `FitLine2D`, `FitPointPolygonTest`, `FitMatrix` (Jacobi SVD with Vt, SVD solve, LU
  determinant, JAMA eigenNonSymmetric). Notes: `Docs/Notes/ShapeFittingPortNotes.md`.
- `ImgProc/Segmentation/`: `FloodFilling.FloodFill` (floodFill without and with
  mask; 8U/32S/32F, 1 or 3 channels; floating and fixed range, 4/8 connectivity,
  mask-only, mask fill value; OpenCV 5.0 always-gradient dispatch and mask frame
  rewrite), enum `FloodFillFlags`, record `FloodFillResult`. Internals:
  `FloodFillScan` (16-bit segment stack, gradient fill, equality fast path),
  `FloodFillDiffs`. Notes: `Docs/Notes/FloodFillPortNotes.md`.
- `ImgProc/Drawing/`: `Drawing` (non-AA line/rectangle/circle/ellipse/
  fillPoly/polylines/drawContours).
- `ImgProc/Morphology/`: `Morphology` (erode/dilate/morphologyEx/
  getStructuringElement).
- `ImgProc/Filtering/`: `Filters` (GaussianBlur, blur/boxFilter, medianBlur,
  filter2D, sepFilter2D, getGaussianKernel) and `Derivatives` (Sobel, Scharr,
  Laplacian, getDerivKernels). Sobel, Scharr, Laplacian, sepFilter2D and
  filter2D support 8U/16U/16S/32F/64F as in OpenCV (no 64F to 32F separable;
  filter2D needs ddepth >= sdepth). Internals: `RowStream` (row-ring
  streaming of a row stage and a column stage, as OpenCV `FilterEngine`;
  reuse it for new separable or small 2-D filters), `SmallIntFilter2D`
  (exact int16 filter2D for 8-bit sources and integer kernels).
- Performance helpers: `Core/Internal/MinMaxOps` (`MinOp`/`MaxOp` with
  `std::min`/`std::max` semantics for scalars and `Vector<T>`),
  `Core/Internal/PooledMat<T>` and `Borders.CopyMakeBorderPooled` (pooled
  temporaries), `ImgProc/Filtering/Internal/FilterVectors` (widening,
  `cvRound` and saturation for `Vector<T>`), `Core/CvParallel` (public
  thread limit, as `cv::setNumThreads`), `Core/Internal/ParallelRows`
  (row-stripe `Parallel.For`) and `Core/Internal/PageTouch` (commits the
  pages of a new large destination on one thread before a parallel run).

Performance conventions (see `Docs/Notes/PerformanceAnalysis.md` and the
tutorial `Docs/Tutorials/MemoryAndSimd.md`):
vectorize inner loops with `Vector<T>` plus a scalar tail; never create row
spans inside element loops; pad and buffer temporaries through `PooledMat<T>`;
keep each lane's float operation order identical to the scalar code (the JIT
does not contract multiply-add, so lane results stay bit-exact). Prefer an
overlapping final vector to a scalar tail when the loop has no running state.
Run row loops through `ParallelRows.For` and put the stripe body in a static
method (a hot loop inside the lambda reads captured variables from the closure);
divide work estimates by `Vector<T>.Count` for memory-bound SIMD loops; output
must not depend on the thread count (seed running state per stripe, or stay on one thread for
`double` running sums). `Tests/Regression` hashes outputs on wide images and
`Tests/Core/CvParallelTests` compares 1 and 8 threads; both must stay green.
Memory-bound vector loops split into stripes only on large images: add such
loops to `LargeImage_Output_DoesNotDependOnThreadCount`.

Conventions: public API file per operation with internals under an
`Internal/` subfolder; attribution headers citing the OpenCV source files;
XML docs stating mutation, clipping/border semantics, and unsupported modes.

Tests use **xUnit** in the sibling `Geometrikal.OpenCV.Managed.Tests` project.
Ports are validated byte-for-byte against golden data generated from
`opencv-python` by scripts in `Geometrikal.OpenCV.Managed.Tests/tools/`; known
cv2-version differences are handled by the suspect/adjudication mechanism
described in the porting instructions. There is no image-I/O dependency
(ImageSharp was removed pending a licence decision).

Before implementing, locate the canonical OpenCV implementation, its helpers, public declaration, and relevant tests. Port the required dependency chain, add equivalent managed tests, then build and run the project tests.

Follow the full porting instructions here:

[Porting instructions](./Docs/Instructions/Porting.md)

## Docs index

| Document | Content |
|---|---|
| [Docs/Instructions/Porting.md](./Docs/Instructions/Porting.md) | Porting process, API and test conventions, golden-data suspect/adjudication |
| [Docs/Notes/CorePortNotes.md](./Docs/Notes/CorePortNotes.md) | Core operation semantics, OpenCV 5.0 choices, golden constraints |
| [Docs/Notes/ColorPortNotes.md](./Docs/Notes/ColorPortNotes.md) | cvtColor bit-exactness, vector/scalar row split, Lab tables, golden constraints |
| [Docs/Notes/FilteringPortNotes.md](./Docs/Notes/FilteringPortNotes.md) | Filter bit-exactness rules and golden-generator constraints |
| [Docs/Notes/ThresholdPortNotes.md](./Docs/Notes/ThresholdPortNotes.md) | Threshold semantics, OpenCV 5.0 mask and dry-run, deliberate differences, golden constraints |
| [Docs/Notes/GeometryPortNotes.md](./Docs/Notes/GeometryPortNotes.md) | Resize, warp and warpPolar bit-exactness, table-driven warps, OpenCV 5.0 choices, golden constraints |
| [Docs/Notes/ConnectedComponentsPortNotes.md](./Docs/Notes/ConnectedComponentsPortNotes.md) | Label numbering per algorithm, scans, parallel strategy, ushort overflow rule, golden constraints |
| [Docs/Notes/DftPortNotes.md](./Docs/Notes/DftPortNotes.md) | DFT/DCT bit-exactness, CCS layout, lane batches, golden constraints |
| [Docs/Notes/EdgePortNotes.md](./Docs/Notes/EdgePortNotes.md) | Canny and spatialGradient semantics, threshold quirks, stripe hysteresis, deliberate L1 difference, golden constraints |
| [Docs/Notes/DistancePortNotes.md](./Docs/Notes/DistancePortNotes.md) | Distance transform routing, chamfer/precise bit-exactness, label tie order, watershed queues, golden constraints |
| [Docs/Notes/AdvancedFilteringPortNotes.md](./Docs/Notes/AdvancedFilteringPortNotes.md) | DFT filter2D, sqrBoxFilter, bilateral, guided filter, stackBlur: bit-exactness, deliberate differences, golden constraints |
| [Docs/Notes/HistogramPortNotes.md](./Docs/Notes/HistogramPortNotes.md) | calcHist/backproject bin rules, compareHist lane order, CLAHE, integral exactness, normalize, golden constraints |
| [Docs/Notes/ShapeDescriptorsPortNotes.md](./Docs/Notes/ShapeDescriptorsPortNotes.md) | approxPolyDP, arcLength, contourArea, boundingRect, moments, Hu moments, matchShapes: bit-exactness, tile order, deliberate differences, golden constraints |
| [Docs/Notes/ShapeFittingPortNotes.md](./Docs/Notes/ShapeFittingPortNotes.md) | Hull, calipers, enclosing circle, ellipse and line fits, point test, polygon and rotated-rectangle intersections: bit-exactness, cv2 5.0.0 build quirks (LAPACK SVD, RNG order, float hull equality), golden constraints |
| [Docs/Notes/FloodFillPortNotes.md](./Docs/Notes/FloodFillPortNotes.md) | floodFill dispatch (simple fill unreachable in 5.0), pixel tests, scan order, equality fast path, deliberate differences, golden constraints |
| [Docs/Notes/LinearAlgebraPortNotes.md](./Docs/Notes/LinearAlgebraPortNotes.md) | gemm/LU/Cholesky/QR/Jacobi SVD bit-exactness, internal-LAPACK routing and tolerances, transform lane orders, reproduced quirks, golden constraints |
| [Docs/Notes/PyramidPortNotes.md](./Docs/Notes/PyramidPortNotes.md) | pyrDown/pyrUp fixed-point and float sum order per element, size and border quirks, KeyPoint/DMatch/KeyPointsFilter semantics, golden constraints |
| [Docs/Notes/OrbPortNotes.md](./Docs/Notes/OrbPortNotes.md) | FAST candidate and score quirks, ORB pyramid layout, submatrix blur path, nth_element order, golden constraints |
| [Docs/Notes/SiftPortNotes.md](./Docs/Notes/SiftPortNotes.md) | SIFT pipeline, bit-exactness (4-lane orders, CRT expf), keypoint order and threads, golden constraints |
| [Docs/Notes/MatcherPortNotes.md](./Docs/Notes/MatcherPortNotes.md) | BFMatcher/batchDistance distance order, k-nearest tie rule, cross-check, MSVC std::sort radius order, golden constraints |
| [Docs/Notes/CornersPortNotes.md](./Docs/Notes/CornersPortNotes.md) | Corner response bit-exactness, Harris vector/scalar split, goodFeaturesToTrack order, cornerSubPix sampling, golden constraints |
| [Docs/Notes/EstimationPortNotes.md](./Docs/Notes/EstimationPortNotes.md) | estimateAffine2D/Partial2D/Translation2D and findHomography: OpenCV 5.0 dispatch (classic RANSAC, USAC gap), RNG and LevMarq bit-exactness, RHO quirks, golden constraints |
| [Docs/Notes/UndistortPortNotes.md](./Docs/Notes/UndistortPortNotes.md) | Undistortion maps (SSE2 block and tail order, fixed-point saturation), undistort stripes, undistortPoints, new camera matrices, Rodrigues, projectPoints, solvePnP ITERATIVE, golden constraints |
| [Docs/Notes/AlignmentPortNotes.md](./Docs/Notes/AlignmentPortNotes.md) | matchTemplate crossCorr/normalisation/mask expression order, phase correlation and Hanning `v_cos`, ECC with 5.0 float-coordinate warps, bit-exactness, deliberate differences, golden constraints |
| [Docs/Notes/StitchingWarpBlendPortNotes.md](./Docs/Notes/StitchingWarpBlendPortNotes.md) | Rotation warpers, feather/multi-band blenders, exposure compensators: bit-exactness, remap table-path tolerances, vector map paths, deliberate differences, golden constraints |
| [Docs/Notes/SeamFinderPortNotes.md](./Docs/Notes/SeamFinderPortNotes.md) | Voronoi, DP and graph-cut seam finders: pair order (MSVC std::sort), DP component/seam semantics, GCGraph max-flow port, cv2 Voronoi binding bug, golden constraints |
| [Docs/Notes/StitchingRegistrationPortNotes.md](./Docs/Notes/StitchingRegistrationPortNotes.md) | Stitching matchers (FLANN deviation), estimators, bundle adjusters (LevMarq, LAPACK SVD limit at 25 parameters), wave correction, Stitcher pipeline, golden constraints |
| [Docs/Notes/OpticalFlowPortNotes.md](./Docs/Notes/OpticalFlowPortNotes.md) | buildOpticalFlowPyramid padding and submatrix rule, Scharr derivatives, LK tracker SSE lane order and fixed-point weights, status/err semantics, golden constraints |
| [Docs/Notes/PnpPortNotes.md](./Docs/Notes/PnpPortNotes.md) | PnP solvers (EPnP, P3P, AP3P, SQPnP, IPPE), solvePnPGeneric error lane order, solvePnPRansac float path and RNG exactness, refine LM/VVS, VVS LAPACK tolerance, golden constraints |
| [Docs/Notes/EpipolarPortNotes.md](./Docs/Notes/EpipolarPortNotes.md) | findFundamentalMat/findEssentialMat/recoverPose/triangulatePoints: OpenCV 5.0 dispatch quirks, multi-model registrator, MatExpr order (inv·B as solve), solvePoly tolerance, golden constraints |
| [Docs/Notes/DrawingAaPortNotes.md](./Docs/Notes/DrawingAaPortNotes.md) | LINE_AA lines and fills, thick AA lines, arrowedLine, drawMarker, OpenCV 5.0 TrueType text (putText not ported), golden constraints |
| [Docs/Notes/MathFunctionsPortNotes.md](./Docs/Notes/MathFunctionsPortNotes.md) | magnitude/phase/cartToPolar/polarToCart/exp/log/pow: block structure, vector/scalar kernels, 64F through float, cv2 4.5.5 differences |
| [Docs/Notes/HoughPortNotes.md](./Docs/Notes/HoughPortNotes.md) | HoughLines/HoughLinesP/HoughLinesPointSet: accumulator layout, vote rounding, RNG order, multi-scale fallback, cv2 4.5.5 differences (ρ bias, angle count) |
| [Docs/Notes/AccumulationPortNotes.md](./Docs/Notes/AccumulationPortNotes.md) | accumulate/Square/Product/Weighted: depth pairs, weighted β per SSE block, deliberate −0 difference, golden constraints |
| [Docs/Notes/DemosaicingPortNotes.md](./Docs/Notes/DemosaicingPortNotes.md) | Bayer bilinear/gray/VNG/EA: integer formulas, VNG padding and scale split, cv2 4.5.5 differences (gray SIMD, VNG shift, small-image pattern) |
| [Docs/Notes/InpaintPortNotes.md](./Docs/Notes/InpaintPortNotes.md) | inpaint Telea/NS: frame layout, queue order, float/double steps, cv2 4.5.5 differences (border reads, float +0.5) |
| [Docs/Notes/DenoisingPortNotes.md](./Docs/Notes/DenoisingPortNotes.md) | fastNlMeansDenoising: fixed-point weights, exact patch sums per offset, coloured variant, golden constraints |
| [Docs/Notes/PerformanceAnalysis.md](./Docs/Notes/PerformanceAnalysis.md) | Measured causes, benchmark status, remaining gaps and next steps |
| [Docs/Tutorials/MemoryAndSimd.md](./Docs/Tutorials/MemoryAndSimd.md) | Pooling, SIMD, scalar tails, register blocking, multithreading; checklist for new loops |
| [../Geometrikal.OpenCV.Managed.Benchmark/CLAUDE.md](../Geometrikal.OpenCV.Managed.Benchmark/CLAUDE.md) | Adding and running benchmark cases, comparing with cv2 |
| [../../docs/Changelogs/](../../docs/Changelogs/) | Session changelogs |

