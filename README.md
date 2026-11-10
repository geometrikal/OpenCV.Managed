# Geometrikal.OpenCV.Managed

Managed C# ports of OpenCV algorithms.

This repository contains true managed ports — not P/Invoke wrappers — of
selected OpenCV functions, built on a stride-aware `Mat<T>` matrix backed by
ordinary managed arrays. No native binaries, no external package
dependencies, no `IDisposable` matrices.

## Installation

```bash
dotnet add package Geometrikal.OpenCV.Managed
```

The package targets .NET 10.

## Status

| Component | Description |
| --- | --- |
| `Mat<T>` | Stride-aware matrix over managed `T[]`; non-copying ROI views, row spans, `CopyTo` and `SetTo` with masks, no disposal required |
| `Arithmetic`, `Bitwise`, `Comparison` | Per-element `Add`, `Subtract`, `AbsDiff`, `Multiply`, `Divide`, `AddWeighted`, `ScaleAdd`, `Min`, `Max`, bitwise `And`/`Or`/`Xor`/`Not`, `Compare`, `InRange` — all OpenCV depths (`byte` to `double`), array and `Scalar` operands, masks, OpenCV saturation and rounding |
| `Conversions` | `ConvertTo` (with OpenCV's float/double work-type table), `ConvertScaleAbs`, `Split`, `Merge`, `Lut` |
| `Reductions` | `Sum`, `Mean`, `MeanStdDev`, `MinMaxLoc`, `CountNonZero`, `Norm` (Inf/L1/L2/L2Sqr, masks, difference and relative norms), `Reduce` (sum/avg/max/min/sum2) |
| `Contours.FindContours` | Full port of OpenCV 5's Suzuki–Abe contour detection (`contours_new.cpp`), including hierarchy, all four retrieval modes (`External`, `List`, `ConnectedComponents`, `Tree`) and all four approximation methods (`None`, `Simple`, `Tc89L1`, `Tc89KCos`) |
| `Drawing` | Non-anti-aliased drawing (`drawing.cpp`): `Line`, `Rectangle`, `Circle`, `Ellipse`, `FillConvexPoly`, `FillPoly`, `Polylines`, `DrawContours` — generic element types, 1–4 channels, sub-pixel `shift`, OpenCV clipping semantics |
| `Morphology` | `Erode`, `Dilate`, `MorphologyEx` (open/close/gradient/top-hat/black-hat/hit-or-miss) and `GetStructuringElement` (rect/cross/ellipse/diamond) with OpenCV's anchor, iteration-folding, and default-border semantics |
| `Borders` | `BorderInterpolate` and `CopyMakeBorder` (constant/replicate/reflect/wrap/reflect101) — the shared border foundation for filter ports |
| `Filters` | `GaussianBlur` (including OpenCV's bit-exact fixed-point 8-bit path and softdouble kernel computation), `Blur`/`BoxFilter`, `MedianBlur`, `Filter2D`, `SepFilter2D` (with OpenCV's bit-exact integer paths), `GetGaussianKernel` — byte and float32 sources, 1–4 channels |
| `Derivatives` | `Sobel`, `Scharr`, `Laplacian`, `GetDerivKernels` with OpenCV's kernel generation, scale folding and work-depth arithmetic |
| `Thresholding` | `Threshold` (binary, binary-inverted, truncate, to-zero, to-zero-inverted; Otsu and triangle automatic thresholds; OpenCV 5.0 mask and dry-run) for `byte`, `short`, `ushort`, `float` and `double`, and `AdaptiveThreshold` (mean and Gaussian) |
| `ColorConversions` | `CvtColor` for BGR/RGB reordering and alpha, gray, HSV (including `_FULL`), CIE L\*a\*b\* (sRGB and linear), CIE XYZ and YCrCb — `byte`, `ushort` and `float`, bit-exact to OpenCV built for 128-bit vectors (Lab tables reproduced from OpenCV's softfloat code) |
| `Geometry`, `MatrixTransforms` | `Resize` (nearest, nearest-exact, linear, linear-exact, cubic, area, Lanczos4; all depths), `WarpAffine`, `WarpPerspective`, `Remap` (float and fixed-point maps, all borders including transparent), `ConvertMaps*`, rotation/affine/perspective matrix helpers, `Flip`, `Transpose`, `Rotate` — bit-exact to cv2 4.x (table-driven warps) |
| `ConnectedComponentLabeling` | `ConnectedComponents`, `ConnectedComponentsWithStats` — 4/8 connectivity, `int`/`ushort` labels, OpenCV's label numbering for every `CCL_*` algorithm, stats and centroids, parallel stripes with thread-independent output |
| `SpectralTransforms` | `Dft`/`Idft` (float, double; real to CCS, complex output and input, rows, nonzeroRows, scaling), `Dct`/`Idct` (even sizes), `MulSpectrums`, `GetOptimalDftSize` — bit-exact to OpenCV's mixed-radix `dxt.cpp` without IPP; rows and columns run as `Vector<T>` lane batches in parallel |
| `EdgeDetection` | `Canny` (apertures 3/5/7 and Scharr, L1/L2, any channel count, dx/dy overload), `SpatialGradient`; derivatives (`Sobel`, `Scharr`, `Laplacian`, `SepFilter2D`, `Filter2D`) for 8U/16U/16S/32F/64F — bit-exact to OpenCV without IPP; parallel stripes with thread-independent output |
| `DistanceTransforms`, `Segmentation` | `DistanceTransform` (L1/L2/C chamfer 3x3/5x5, precise Euclidean, 8-bit L1), `DistanceTransformWithLabels` (component and pixel labels), `Watershed` — bit-exact to OpenCV 5.0 without IPP; precise transform parallel |
| `Filters` (advanced), `GuidedFilter`, `EdgePreservingFilters` | `Filter2D` DFT path for large kernels, `SqrBoxFilter`, `BilateralFilter` (8U/32F, OpenCV 5.0 SSE2 arithmetic), `GuidedFilter` (ximgproc), `StackBlur` (8U/32F) — bit-exact to cv2 5.0.0 without IPP |
| `Histograms`, `Clahe`, `IntegralImages`, `Conversions.Normalize` | `CalcHist` (1-32 dims, uniform and non-uniform, mask, accumulate), `CalcBackProject`, `CompareHist` (6 methods), `EqualizeHist`, CLAHE (8U/16U), `Integral` (sum, sqsum, tilted), `Normalize` — bit-exact to OpenCV without IPP; 8U histograms with sub-histograms and parallel stripes |
| `CvParallel` | Process-wide thread limit (`MaxDegreeOfParallelism`, as `cv::setNumThreads`). Filters, morphology, core operations, colour conversions and thresholding split rows across threads; output does not depend on the thread count |

Filtering and morphology are vectorized with `Vector<T>` and use pooled temporaries. Single-threaded,
most cases run at 0.2× to 1.4× the time of cv2 4.5.5 on one thread; see
[`PerformanceAnalysis.md`](src/Geometrikal.OpenCV.Managed/Docs/Notes/PerformanceAnalysis.md).

Not supported (by design, validated and rejected rather than approximated):
contours — `RETR_FLOODFILL`, `LINK_RUNS`, chain-code output; drawing —
anti-aliasing (`LINE_AA`), `putText`, `drawMarker`, `arrowedLine`;
morphology and filtering — `BORDER_WRAP` (rejected by OpenCV's own filter
engine too); filtering — 16-bit/64-bit image depths, `sqrBoxFilter`,
`bilateralFilter`, and OpenCV's DFT-based `filter2D` path for large kernels
(area ≥ 130 for 8-bit and float32 on SSE3 hardware, otherwise ≥ 50; the port
always convolves directly, which is exact).

## Repository layout

```text
Geometrikal.OpenCV.Managed.slnx            The solution
src/OpenCV/                         OpenCV 5.0.0 source tree (git submodule) — the porting reference
src/Geometrikal.OpenCV.Managed/            Class library (net10.0), published to NuGet
src/Geometrikal.OpenCV.Managed.Tests/      xUnit test suite
src/Geometrikal.OpenCV.Managed.Benchmark/  BenchmarkDotNet suites and the matching cv2 script
docs/Changelogs/                    Session changelogs
```

## Getting started

Requires the .NET 10 SDK. Clone with the OpenCV submodule. The build does not
need it, but it is the porting reference and the attribution headers cite it:

```bash
git clone --recurse-submodules https://github.com/geometrikal/OpenCV.Managed.git
```

Build and test:

```bash
dotnet build Geometrikal.OpenCV.Managed.slnx
dotnet test  Geometrikal.OpenCV.Managed.slnx
```

## Usage

```csharp
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

var image = new Mat<byte>(width: 64, height: 64);
// ... set foreground pixels to any non-zero value ...

ContourResult result = Contours.FindContours(
    image,
    RetrievalMode.Tree,
    ContourApproximationMode.Simple);

for (int i = 0; i < result.Contours.Count; i++)
{
    Point[] contour = result.Contours[i];
    ContourHierarchy links = result.Hierarchy[i]; // Next / Previous / FirstChild / Parent
}
```

The source matrix is never modified, and inputs may be non-contiguous ROI
views with arbitrary stride.

## Porting new functions

The porting methodology (source discovery, span/stride rules, staged
optimization, testing requirements) is documented in
[`src/Geometrikal.OpenCV.Managed/Docs/Instructions/Porting.md`](src/Geometrikal.OpenCV.Managed/Docs/Instructions/Porting.md).

Ports are validated against golden data generated from OpenCV via
`opencv-python`; see
`src/Geometrikal.OpenCV.Managed.Tests/tools/generate_golden.py`
for regeneration instructions.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).

Portions of this software are C# ports of OpenCV
(https://github.com/opencv/opencv), which is also licensed under the Apache
License 2.0. The [NOTICE](NOTICE) file gives the attribution details, and the
ported source files carry headers that identify their original OpenCV sources.
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) lists the other licences that
apply to ported code, including the BSD 3-Clause licence of SQPnP.

## Releasing

The `Publish to NuGet` workflow packs and publishes the library when a tag of
the form `v<version>` is pushed. The tag must match the `<Version>` in
`src/Geometrikal.OpenCV.Managed/Geometrikal.OpenCV.Managed.csproj`:

```bash
git tag v0.1.0
git push origin v0.1.0
```
