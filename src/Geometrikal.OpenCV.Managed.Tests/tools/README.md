# Golden contour data

Regenerate `TestData/contours_golden.json` (reference outputs from OpenCV
`cv2` 4.5.5 + numpy) by running, from the `Geometrikal.OpenCV.Managed.Tests`
directory:

```
python tools/generate_golden.py
```

The script builds a battery of analytic shapes and seeded random binaries,
captures `cv2.findContours` contours + hierarchy for the mode/method matrix in
the spec, runs sanity assertions, and writes the JSON that
`ImgProc/ContoursGoldenTests.cs` replays against the managed port.

Regenerate `TestData/drawing_golden.json` (reference pixel buffers from `cv2`
4.5.5 for line/rectangle/circle/ellipse/fill/polylines/drawContours) with
`python tools/generate_drawing_golden.py`; `ImgProc/DrawingGoldenTests.cs`
replays it, and version-risk cases are named `*_oob_*` / `*_ellipsex_*`.

Morphology goldens: regenerate `TestData/morphology_golden.json` with
`python tools/generate_morphology_golden.py` (covers erode/dilate/morphologyEx,
getStructuringElement, copyMakeBorder, borderInterpolate; diamond elements are
OpenCV-5-only and covered analytically in MorphologyBasicTests instead).

Filtering goldens: regenerate `TestData/filters_golden.json` with
`python tools/generate_filters_golden.py` (GaussianBlur, blur/boxFilter,
medianBlur, filter2D, sepFilter2D, Sobel, Scharr, Laplacian,
getGaussianKernel). The generator disables IPP/OpenCL and the AVX/FMA CPU
dispatch and constrains the battery so cv2's lane-dependent rounding corners
are avoided — read the generator header and
`Geometrikal.OpenCV.Managed/Docs/Notes/FilteringPortNotes.md` before adding
cases.

Core goldens: regenerate `TestData/core_golden.json` with
`python tools/generate_core_golden.py` (arithmetic, bitwise, compare, inRange,
convertTo, convertScaleAbs, LUT, merge, copyTo, sum, mean, meanStdDev,
minMaxLoc, countNonZero, norm, reduce). Input and output arrays are stored
once in a shared `arrays` list that cases reference by index. Read the
generator header and `Geometrikal.OpenCV.Managed/Docs/Notes/CorePortNotes.md`
before adding cases.

Colour goldens: regenerate `TestData/color_golden.json` with
`python tools/generate_color_golden.py` (cvtColor for reorder, gray, HSV, Lab,
XYZ and YCrCb). The generator calls `cv2.setUseOptimized(False)`. "cube" cases
store only the SHA-256 of the output for all 2^24 8-bit triples; the runner
regenerates the input. Read
`Geometrikal.OpenCV.Managed/Docs/Notes/ColorPortNotes.md` before adding cases.

Threshold goldens: regenerate `TestData/threshold_golden.json` with
`python tools/generate_threshold_golden.py` (threshold for 8U/16S/16U/32F/64F
with fixed, Otsu and triangle thresholds; adaptiveThreshold with mean and
Gaussian methods). Arrays are pooled as in the core goldens. The large Otsu case
stores no pixels: the generator and `ImgProc/ThresholdGoldenTests.cs` build the
image from the same formula. Read the generator header and
`Geometrikal.OpenCV.Managed/Docs/Notes/ThresholdPortNotes.md` before adding cases.

Geometry goldens: regenerate `TestData/geometry_golden.json` with
`python tools/generate_geometry_golden.py` (resize for all interpolations and
depths, warpAffine, warpPerspective, remap, convertMaps, transform-matrix
helpers, flip/rotate/transpose). IPP is off and the AVX/FMA dispatch is
disabled. Arrays are pooled; wide resize cases build their source from a
formula and store the SHA-256 of the output. Suspects: INTER_NEAREST_EXACT and
collinear getPerspectiveTransform. Read
`Geometrikal.OpenCV.Managed/Docs/Notes/GeometryPortNotes.md` before adding cases.

Connected components goldens: regenerate `TestData/cc_golden.json` with
`python tools/generate_cc_golden.py`. 4-connectivity outputs come from
`CCL_WU` (the cv2 4.5.5 Spaghetti 4C defect). Random images are rebuilt from a
hash; large cases store SHA-256 hashes. Arrays are pooled as in the core
goldens.

DFT goldens: regenerate `TestData/dft_golden.json` with
`python tools/generate_dft_golden.py` (dft in all six modes, rows,
nonzeroRows, column ROI, dct/idct, mulSpectrums, getOptimalDFTSize). IPP must
stay disabled; read the generator header and
`Geometrikal.OpenCV.Managed/Docs/Notes/DftPortNotes.md` before adding cases.

Edge goldens: regenerate `TestData/edge_golden.json` with
`python tools/generate_edge_golden.py` (Canny image and dx/dy overloads,
spatialGradient). IPP must stay disabled: `ipp_Canny` differs for
out-of-range thresholds. L1 dx/dy cases contain no -32768. Arrays are pooled
as in the core goldens. Read `Geometrikal.OpenCV.Managed/Docs/Notes/EdgePortNotes.md`
before you add cases. The filtering goldens include 16U/16S/64F derivative
cases from a separate RNG. Suspects: `lap_f64_k3_scale01` and
`lap_u8f64_k1_scale01` (OpenCV 5.0 double Laplacian kernel).

Distance goldens: regenerate `TestData/distance_golden.json` with
`python tools/generate_distance_golden.py` (distanceTransform for all types
and masks, 8-bit L1, labelled transforms, watershed). IPP must stay off.
Suspects: chamfer and precise cases on images without zero pixels (OpenCV 5.0
unreachable values). Arrays are pooled; large cases store SHA-256 hashes.

Advanced filtering goldens: regenerate `TestData/advfilter_golden.json` with
`python tools/generate_advfilter_golden.py`. The script requires cv2 5.0.0
with contrib: `pip install --no-deps --target <dir>
opencv-contrib-python-headless==5.0.0.93`, then run with `PYTHONPATH=<dir>`
(numpy 1.22 works). `OPENCV_CPU_DISABLE=AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL`
must be set before cv2 is imported; the script sets it when it is absent. The
script stops if cv2 is not 5.0.0, if ximgproc is missing, or if AVX2 is still
enabled. Large cases rebuild their input from a hash and store SHA-256
outputs. The 1-channel-guide guided cases depend on the `rcpProbe` RCPPS
values. Read `Geometrikal.OpenCV.Managed/Docs/Notes/AdvancedFilteringPortNotes.md`
before you add cases.

Histogram goldens: regenerate `TestData/histogram_golden.json` with
`python tools/generate_histogram_golden.py` (calcHist, calcBackProject,
compareHist, equalizeHist, CLAHE, integral, normalize). IPP and the AVX/FMA
dispatch are off. Arrays are pooled; ROI cases store the parent and apply
"roi". Cases with "model": true come from scalar models of the OpenCV 5.0
source (non-uniform bins, CLAHE bitShift, 3-D back projection), because cv2
Python passes 3-D histograms as multi-channel 2-D Mats. Suspects: compareHist
CHISQR and CHISQR_ALT with at least 4 bins ("values5" holds the 5.0 value).
Read `Geometrikal.OpenCV.Managed/Docs/Notes/HistogramPortNotes.md` before you add
cases.

# Regression snapshots

`TestData/regression_hashes.json` holds a SHA-256 prefix of the output of every
case in `Regression/RegressionCases.cs`. The cases use wide, odd-sized random
images, so they exercise the vectorized loops and their scalar tails, which
the small golden images do not reach. `Regression/RegressionSnapshotTests.cs`
requires the current build to reproduce every hash.

The hashes must come from a build that passes the cv2 golden tests. The
current file was generated from commit `f613774` (the scalar implementation).
To add cases, generate the hashes from a known-good commit, not from the
working tree under change:

```
git worktree add --detach <dir> <known-good-commit>
dotnet run -c Release --project tools/RegressionSnapshotGen -p:OpenCvProject=<dir>/src/Geometrikal.OpenCV.Managed/Geometrikal.OpenCV.Managed.csproj -- TestData/regression_hashes.json
git worktree remove <dir>
```

The known-good commit must contain the public API used by new cases.

Shape descriptor, shape fitting and flood fill goldens: regenerate
`TestData/shapes_golden.json`, `TestData/shapefit_golden.json` and
`TestData/floodfill_golden.json` with `python tools/generate_shapes_golden.py`,
`python tools/generate_shapefit_golden.py` and
`python tools/generate_floodfill_golden.py`. The scripts require cv2 5.0.0
(install as for the advanced filtering goldens and run with `PYTHONPATH=<dir>`);
each script stops if the version is different. Read the port notes
(`ShapeDescriptorsPortNotes.md`, `ShapeFittingPortNotes.md`,
`FloodFillPortNotes.md`) before you add cases.

Linear algebra and pyramid goldens: regenerate `TestData/linalg_golden.json` and
`TestData/pyramid_golden.json` with `python tools/generate_linalg_golden.py` and
`python tools/generate_pyramid_golden.py` (cv2 5.0.0, IPP off, AVX/FMA dispatch off).
The linear algebra cases that cv2 routes to its internal LAPACK use tolerances; read
`LinearAlgebraPortNotes.md` and `PyramidPortNotes.md` before you add cases.

Feature goldens: regenerate `TestData/orb_golden.json`, `TestData/sift_golden.json`,
`TestData/matcher_golden.json` and `TestData/corners_golden.json` with
`python tools/generate_orb_golden.py`, `generate_sift_golden.py`,
`generate_matcher_golden.py` and `generate_corners_golden.py` (cv2 5.0.0 Windows wheel,
IPP off, AVX/FMA dispatch off). SIFT exactness depends on the MSVC C runtime `expf`;
radius-match tie order depends on the MSVC `std::sort`. Read `OrbPortNotes.md`,
`SiftPortNotes.md`, `MatcherPortNotes.md` and `CornersPortNotes.md` before you add cases.

Estimation, undistortion and alignment goldens: regenerate
`TestData/estimation_golden.json`, `TestData/undistort_golden.json` and
`TestData/alignment_golden.json` with `python tools/generate_estimation_golden.py`,
`generate_undistort_golden.py` and `generate_alignment_golden.py` (cv2 5.0.0, IPP off,
AVX/FMA dispatch off). RANSAC and LMEDS use a fixed `cv::RNG` seed per call. Read
`EstimationPortNotes.md`, `UndistortPortNotes.md` and `AlignmentPortNotes.md` before you add
cases.

Stitching goldens: regenerate `TestData/stitch_warp_golden.json`,
`TestData/seam_golden.json` and `TestData/stitch_reg_golden.json` with
`python tools/generate_stitch_warp_golden.py`, `generate_seam_golden.py` and
`generate_stitch_reg_golden.py` (cv2 5.0.0; `generate_stitch_reg_golden.py` reads the cv2
directory from `CV2_500_DIR`). cv2 stitching matchers use FLANN, which is random and not
ported; the registration goldens store ORB features and cv2 matcher output. Read the three
stitching port notes before you add cases.

Optical flow, PnP, two-view geometry and anti-aliased drawing goldens: regenerate
`TestData/lk_golden.json`, `TestData/pnp_golden.json`, `TestData/epipolar_golden.json` and
`TestData/drawing_aa_golden.json` with `python tools/generate_lk_golden.py`,
`generate_pnp_golden.py`, `generate_epipolar_golden.py` and `generate_drawing_aa_golden.py`
(cv2 5.0.0, IPP off, AVX/FMA dispatch off, one thread). Read `OpticalFlowPortNotes.md`,
`PnpPortNotes.md`, `EpipolarPortNotes.md` and `DrawingAaPortNotes.md` before you add cases.
