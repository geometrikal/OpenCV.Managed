# Instructions for Porting an OpenCV Function to `Geometrikal.OpenCV.Managed`

## Objective

Port a selected OpenCV function from the OpenCV C++ source tree into the managed .NET class library located at:

```text
src/Geometrikal.OpenCV.Managed/Geometrikal.OpenCV.Managed.csproj
```

The OpenCV source tree is located at:

```text
src/OpenCV
```

The implementation must be written in managed C# and should use the matrix and image abstractions already defined in `Geometrikal.OpenCV.Managed`.

Do not create a P/Invoke wrapper around the native OpenCV implementation.

The goal is to port the algorithm itself into idiomatic, efficient C# while preserving OpenCV behaviour as closely as practical.

---

## Current Project State

The library has **no external package dependencies**. (An earlier `SixLabors.ImageSharp`
reference was removed because version 4.x enforces a paid build-time licence key —
see [Image I/O and ImageSharp](#image-io-and-imagesharp).)

Existing support types and subsystems — reuse these, do not recreate equivalents:

* `Core/`: `Mat<T>` (stride-aware managed matrix), `Point`, `Rect`, `Size`,
  `Scalar`, `BorderType`, `Borders.BorderInterpolate` and
  `Borders.CopyMakeBorder`, plus `Core/Internal/ScalarConverter`
  (OpenCV `scalarToRawData`/`cvRound` saturation semantics).
* `Core/` operations: `Arithmetic`, `Bitwise`, `Comparison`, `Conversions`,
  `Reductions`, masked `Mat.CopyTo`/`Mat.SetTo`. Reuse
  `Core/Internal/Saturate` for element conversions and
  `Core/Internal/Elementwise` (with an `IBinaryRowOp`/`IUnaryRowOp` struct)
  for new per-element operations. See `Docs/Notes/CorePortNotes.md`.
* `ImgProc/Color/`: cvtColor port (public entry `ColorConversions`, enum
  `ColorConversionCode`). See `Docs/Notes/ColorPortNotes.md`, including the
  rule for kernels whose OpenCV vector body and scalar tail differ.
* `ImgProc/Threshold/`: thresholding port (public entry `Thresholding`, enums
  `ThresholdType`/`AdaptiveThresholdType`). Use `ScalarConverter.CvFloor` and
  `ScalarConverter.CvCeil` for OpenCV `cvFloor`/`cvCeil`; they reproduce the x86
  results for NaN and out-of-range values. See `Docs/Notes/ThresholdPortNotes.md`.
* `ImgProc/Geometry/` and `Core/MatrixTransforms`: geometric transform port
  (public entries `Geometry`, `MatrixTransforms`). Use `TransformKernels` for
  whole-pixel moves and `InterTables`/`RemapKernels` for new warp-like
  operations. See `Docs/Notes/GeometryPortNotes.md`.
* `ImgProc/ConnectedComponents/`: connected components port (public entry
  `ConnectedComponentLabeling`). Label numbering depends on the algorithm; read
  `Docs/Notes/ConnectedComponentsPortNotes.md` before changing a scan.
* `Core/SpectralTransforms`: DFT/DCT/mulSpectrums port (enums
  `DftFlags`/`DctFlags`), bit-exact to OpenCV without IPP. Use
  `Core/Internal/LaneTranspose` with `Vector<T>` lanes for algorithms that run
  one independent line per lane. See `Docs/Notes/DftPortNotes.md`.
* `ImgProc/EdgeDetection/`: Canny and spatialGradient port (public entry
  `EdgeDetection`). Thresholds use x86 `cvFloor`. Read
  `Docs/Notes/EdgePortNotes.md` before you change the engine. Golden
  generation must disable IPP.
* `ImgProc/Distance/` and `ImgProc/Segmentation/`: distance transform and
  watershed ports (public entries `DistanceTransforms`, `Segmentation`).
  Neighbour order decides label ties; see `Docs/Notes/DistancePortNotes.md`.
* `ImgProc/Filtering/` advanced filters: filter2D DFT path
  (`CrossCorrEngine`), sqrBoxFilter, bilateralFilter, guidedFilter, stackBlur.
  Goldens come from cv2 5.0.0. On processors with the gather data sampling
  mitigation, `Avx2.GatherVector256` is slower than scalar loads; measure
  before you use it. After a mutation check that restores files, rebuild with
  `--no-incremental`. See `Docs/Notes/AdvancedFilteringPortNotes.md`.
* `ImgProc/Histograms/`, `Core/IntegralImages`, `Conversions.Normalize`:
  histogram, CLAHE, integral and normalize ports. Use `Core/Internal/ByteLut`
  for 256-entry byte table lookups and `BinSelect` for branch-free bin
  selection. See `Docs/Notes/HistogramPortNotes.md`.
* `ImgProc/Contours/`: `findContours` port (public entry `Contours`).
* `ImgProc/Drawing/`: non-anti-aliased drawing port (public entry `Drawing`,
  enum `LineType`).
* `ImgProc/Morphology/`: morphology port (public entry `Morphology`, enums
  `MorphShape`/`MorphOperation`).
* `ImgProc/Filtering/`: smoothing/linear/derivative filter port (public
  entries `Filters` and `Derivatives`): GaussianBlur, blur/boxFilter,
  medianBlur, filter2D, sepFilter2D, Sobel, Scharr, Laplacian, plus the
  bit-exact Gaussian kernel machinery. Bit-exactness constraints and the
  golden-generator requirements are documented in
  `Docs/Notes/FilteringPortNotes.md` — read it before extending a filter.
* Performance infrastructure (see `Docs/Tutorials/MemoryAndSimd.md`):
  `Core/Internal/PooledMat<T>` and `Borders.CopyMakeBorderPooled` (pooled
  temporaries), `Core/Internal/MinMaxOps` (`MinOp`/`MaxOp` for scalars and
  `Vector<T>`), `ImgProc/Filtering/Internal/FilterVectors` (widening, `cvRound`,
  saturation), `Core/CvParallel` (public thread limit) and
  `Core/Internal/ParallelRows` (row-stripe parallelism).
* Test project `Geometrikal.OpenCV.Managed.Tests` (xUnit) with golden-data
  infrastructure (see [OpenCV Comparison Tests](#opencv-comparison-tests)),
  `Regression/` (output hashes of wide, odd-sized images) and
  `Core/CvParallelTests` (output must not depend on the thread count).
* Benchmark project `Geometrikal.OpenCV.Managed.Benchmark` (BenchmarkDotNet and a
  matching cv2 script); see its `CLAUDE.md`.

Established layout convention: one public static class per subsystem under
`ImgProc/<Area>/`, internal algorithm files under `ImgProc/<Area>/Internal/`
(namespace `Geometrikal.OpenCV.Managed.ImgProc.Internal`), attribution headers on
every ported file.

Where these instructions say "use the existing type" or "prefer existing project
types", check the list above and the source tree first — do not create a second
`Point` when one already exists. Types still missing are created by the port
that first needs them and reused thereafter.

---

## Core Requirements

1. Locate the canonical OpenCV implementation in `src/OpenCV`.
2. Identify all helper functions, constants, enums, lookup tables, and data structures used by the implementation.
3. Port the minimum required dependency graph into `Geometrikal.OpenCV.Managed`.
4. Preserve the observable behaviour of the OpenCV function.
5. Use managed arrays and spans by default.
6. Do not introduce native allocations unless there is a clear, documented requirement.
7. Do not require callers to dispose ordinary matrices or result objects.
8. Support matrix stride and non-contiguous regions of interest.
9. Add tests comparing the managed implementation against OpenCV where practical.
10. Document any deliberate behavioural differences.

---

## Repository Layout

Use these paths as the starting point.

### Managed implementation

```text
src/Geometrikal.OpenCV.Managed/
```

Project file:

```text
src/Geometrikal.OpenCV.Managed/Geometrikal.OpenCV.Managed.csproj
```

Solution file:

```text
Geometrikal.OpenCV.Managed.slnx
```

### OpenCV source

```text
src/OpenCV/
```

OpenCV modules are generally located under:

```text
src/OpenCV/modules/
```

Common locations include:

```text
src/OpenCV/modules/core/src/
src/OpenCV/modules/imgproc/src/
src/OpenCV/modules/imgcodecs/src/
```

Tests may be located under:

```text
src/OpenCV/modules/<module>/test/
```

Public declarations may be located under:

```text
src/OpenCV/modules/<module>/include/
```

---

## Before Writing Code

First inspect the managed project.

Determine:

* the namespace conventions;
* the existing `Mat<T>` implementation (`Core/Mat.cs`);
* which point, rectangle, size, scalar, or other support types already exist (see [Current Project State](#current-project-state) — most do not yet);
* existing validation helpers, if any;
* existing exception conventions;
* the test project location, if one has been created;
* whether the target function or related helpers already exist.

Do not create duplicate types when an equivalent type already exists.

Then inspect the OpenCV implementation.

Search for:

* the public function declaration;
* the primary implementation;
* overloads;
* internal implementation functions;
* legacy C API implementations;
* optimized CPU paths;
* SIMD implementations;
* parallel implementations;
* OpenCL or IPP implementations;
* tests that define edge-case behaviour.

Prefer the normal portable CPU implementation as the behavioural reference.

Do not port OpenCL, CUDA, IPP, HAL, or architecture-specific SIMD code during the first implementation unless explicitly requested.

---

## Source Discovery Procedure

For a target function such as:

```text
threshold
```

search the OpenCV tree for:

```text
threshold(
cv::threshold
CV_IMPL
```

Also search for related internal names (implementation helpers, per-type
functions, and legacy C entry points).

Trace the call chain until the core algorithm is understood.

Record:

* the public entry point;
* the actual algorithm implementation;
* required internal types;
* required constants;
* required preprocessing;
* required postprocessing;
* mutation of the source image;
* coordinate offsets;
* border assumptions;
* output ordering;
* error conditions.

Do not port only the public wrapper while overlooking the real implementation beneath it.

---

## Managed Matrix Model

Use the existing managed matrix abstraction: `Mat<T>` in `Core/Mat.cs`
(namespace `Geometrikal.OpenCV.Managed.Core`). Read that file rather than assuming an
API; the summary below is a guide, not a substitute.

`Mat<T>` is `where T : unmanaged`, is backed by a managed `T[]`, and may either own
a whole allocation or be a non-copying view (ROI) into another matrix's buffer.

Key members:

```csharp
int Width
int Height
int Channels
int Stride        // in ELEMENTS, not bytes; includes channels
int RowLength     // Width * Channels
bool IsContinuous // Stride == RowLength

Span<T> GetRowSpan(int y)
ReadOnlySpan<T> GetReadOnlyRowSpan(int y)

Span<T> GetSpan()               // continuous matrices only
ReadOnlySpan<T> GetReadOnlySpan()

ref T At(int y, int x)          // single-channel pixel (row, col)
ref T At(int y, int x, int channel)
ref T AtElement(int y, int elementX)

Mat<T> Roi(int x, int y, int width, int height)  // non-copying view
Mat<T> Clone()                                   // packed copy
void CopyTo(Mat<T> destination)
void Fill(T value)
void Clear()
T[] ToArray()
```

The backing storage is managed. No disposal is required for ordinary matrices.

Note that `Stride` is measured in elements and includes channels, and that an ROI
typically has `Stride > RowLength` (`IsContinuous == false`).

---

## Span Usage Rules

Use spans as the default processing abstraction.

For a source row:

```csharp
ReadOnlySpan<byte> sourceRow = source.GetReadOnlyRowSpan(y);
```

For a writable row:

```csharp
Span<byte> destinationRow = destination.GetRowSpan(y);
```

For an in-place operation:

```csharp
Span<byte> row = image.GetRowSpan(y);
```

Do not assume a matrix or ROI is continuous.

This is incorrect for general matrix code:

```csharp
Span<byte> pixels = image.GetSpan();
```

unless continuity has already been checked.

Use a whole-image fast path only when appropriate:

```csharp
if (source.IsContinuous && destination.IsContinuous)
{
    ReadOnlySpan<byte> src = source.GetReadOnlySpan();
    Span<byte> dst = destination.GetSpan();

    // Packed processing.
}
else
{
    for (int y = 0; y < source.Height; y++)
    {
        ReadOnlySpan<byte> src = source.GetReadOnlyRowSpan(y);
        Span<byte> dst = destination.GetRowSpan(y);

        // Row processing.
    }
}
```

For most initial ports, prefer a clear row-based implementation over an early continuous-buffer optimization.

---

## Stride and ROI Requirements

All implementations must work with matrices whose:

```text
Stride > Width × Channels
```

This commonly occurs for:

* regions of interest;
* externally wrapped buffers;
* padded image rows;
* submatrices.

Do not calculate a pixel location as:

```csharp
buffer[(y * width) + x]
```

unless operating on a known packed local array.

For a matrix, use row spans:

```csharp
Span<byte> row = image.GetRowSpan(y);
byte value = row[x];
```

For multi-channel data:

```csharp
int index = x * image.Channels;
```

or use an existing typed pixel abstraction where available.

---

## Input Validation

Match OpenCV's accepted input types and shapes as closely as practical.

Validate:

* null arguments;
* empty images;
* channel count;
* element type;
* matching dimensions;
* allowed enum values;
* output shape requirements;
* unsupported modes;
* integer overflow when calculating sizes.

Use standard .NET exceptions.

Typical choices include:

```csharp
ArgumentNullException
ArgumentException
ArgumentOutOfRangeException
InvalidOperationException
NotSupportedException
```

Do not silently reinterpret unsupported matrix formats.

Example:

```csharp
ArgumentNullException.ThrowIfNull(image);

if (image.Channels != 1)
{
    throw new ArgumentException(
        "The input image must be single-channel.",
        nameof(image));
}
```

---

## OpenCV Type Mapping

Translate common OpenCV types approximately as follows.

| OpenCV               | Managed C#                                       |
| -------------------- | ------------------------------------------------ |
| `uchar`              | `byte`                                           |
| `schar`              | `sbyte`                                          |
| `ushort`             | `ushort`                                         |
| `short`              | `short`                                          |
| `int`                | `int`                                            |
| `float`              | `float`                                          |
| `double`             | `double`                                         |
| `cv::Point`          | managed `Point` / `Point2i` (create if absent)   |
| `cv::Point2f`        | managed `Point2f` (create if absent)             |
| `cv::Size`           | managed `Size` (create if absent)                |
| `cv::Rect`           | managed `Rect` (create if absent)                |
| `cv::Scalar`         | managed scalar type (create if absent)           |
| `cv::Mat`            | `Mat<T>`                                          |
| `std::vector<T>`     | `List<T>` or `T[]`                               |
| `InputArray`         | explicit strongly typed method parameters        |
| `OutputArray`        | return values or explicit destination parameters |

Prefer existing project types over creating new equivalents. Where a type does not
exist yet, create it once and reuse it (see [Current Project State](#current-project-state)).

---

## API Design

The managed public API should feel natural in .NET while remaining recognizably equivalent to OpenCV.

For example, OpenCV may expose:

```cpp
double threshold(
    InputArray src,
    OutputArray dst,
    double thresh,
    double maxval,
    int type);
```

A managed API might be:

```csharp
public static double Threshold(
    Mat<byte> source,
    Mat<byte> destination,
    double thresh,
    double maxValue,
    ThresholdType type);
```

or, where returning a new matrix is more natural in .NET:

```csharp
public static Mat<byte> Threshold(
    Mat<byte> source,
    double thresh,
    double maxValue,
    ThresholdType type);
```

Do not blindly reproduce C++ output-parameter conventions when a clear managed return type is better.

Preserve function names and enum meanings where practical.

---

## Mutation Semantics

Determine whether the OpenCV implementation modifies the source image.

Some OpenCV functions accept the same array as both input and output, or mutate a
working image internally. The managed implementation should make this behaviour
explicit.

Prefer one of these approaches:

### Non-mutating public API

```csharp
public static Mat<byte> Threshold(
    Mat<byte> source,
    double thresh,
    double maxValue,
    ThresholdType type)
{
    var destination = new Mat<byte>(source.Width, source.Height, source.Channels);
    ThresholdInto(source, destination, thresh, maxValue, type);
    return destination;
}
```

### Explicit in-place API

```csharp
public static void ThresholdInPlace(
    Mat<byte> image,
    double thresh,
    double maxValue,
    ThresholdType type)
```

Do not unexpectedly alter the caller's source matrix unless the API explicitly states that it is in-place.

---

## Border Handling

Many OpenCV image algorithms rely on implicit or explicit border conditions.

For any neighbourhood operation (filters, morphology, edge detection), carefully
determine what OpenCV assumes at the image edges: replicated pixels, reflected
pixels, a constant value, or a required zero border.

If a padded border is required, create it explicitly:

```csharp
Mat<byte> bordered = CopyMakeBorder(
    source,
    top: 1,
    bottom: 1,
    left: 1,
    right: 1,
    value: 0);
```

Use `Borders.CopyMakeBorder` for results and `Borders.CopyMakeBorderPooled` for
internal temporaries.

When returning coordinates from an algorithm that ran on a bordered working image,
account for the added border:

```text
returnedX = internalX - borderLeft + offset.X
returnedY = internalY - borderTop + offset.Y
```

Verify the border mode and any coordinate offset against OpenCV tests.

---

## Algorithm Porting Strategy

Port in stages.

### Stage 1: Behaviourally correct scalar implementation

Write a simple scalar C# implementation using:

```csharp
Span<T>
ReadOnlySpan<T>
List<T>
```

Prioritize:

* correctness;
* clear control flow;
* matching OpenCV behaviour;
* support for stride and ROI;
* testability.

Do not begin with SIMD or unsafe code.

### Stage 2: Reduce unnecessary allocations

After correctness is established:

* reuse small temporary arrays;
* pre-size lists where useful;
* avoid per-pixel allocations;
* use value types for small coordinate structures;
* avoid LINQ inside pixel loops;
* avoid delegates inside hot loops.

### Stage 3: Optimize hot loops

Only after profiling:

* use continuous-matrix fast paths;
* use `System.Numerics.Vector<T>`;
* use `System.Runtime.Intrinsics`;
* use `ArrayPool<T>` for large temporary buffers;
* consider parallel processing if the algorithm is safely parallelizable.

Do not alter output ordering or numeric behaviour during optimization.

Follow `Docs/Tutorials/MemoryAndSimd.md` (section 10 is a checklist). Rules that
earlier work established:

* Run row loops through `ParallelRows.For`. Put the stripe body in a static method
  with parameters. A loop written inside the lambda reads captured variables from
  the closure object on every iteration; this made the separable filters 25–45%
  slower.
* Give `ParallelRows.For` a realistic work estimate. Divide element counts by
  `Vector<T>.Count` for memory-bound SIMD loops; otherwise cheap operations split
  into stripes that cost more than they save.
* Output must not depend on the thread count. Seed running state (running sums,
  histograms) per stripe, or keep the loop on one thread when the state is a
  floating-point accumulation.
* Replace a scalar row tail with an overlapping final vector when the loop has no
  running state. A generic scalar tail can cost more than the vector body.
* Do not assume OpenCV rounds with `cvRound`. Some paths use fixed-point division
  that rounds exact halves up (`ColumnSum<ushort, uchar>`). Generate golden cases
  that produce exact halves (even kernel areas) for such paths.

---

## C++ Translation Guidance

Translate pointer-heavy C++ into row-span-based C# where possible.

### OpenCV-style C++

```cpp
for (int y = 0; y < image.rows; y++)
{
    uchar* row = image.ptr<uchar>(y);

    for (int x = 0; x < image.cols; x++)
    {
        row[x] = row[x] > threshold ? 255 : 0;
    }
}
```

### Managed C#

```csharp
for (int y = 0; y < image.Height; y++)
{
    Span<byte> row = image.GetRowSpan(y);

    for (int x = 0; x < image.Width; x++)
    {
        row[x] = row[x] > threshold
            ? byte.MaxValue
            : (byte)0;
    }
}
```

For neighbourhood access:

```csharp
for (int y = 1; y < image.Height - 1; y++)
{
    ReadOnlySpan<byte> previous = image.GetReadOnlyRowSpan(y - 1);
    Span<byte> current = image.GetRowSpan(y);
    ReadOnlySpan<byte> next = image.GetReadOnlyRowSpan(y + 1);

    for (int x = 1; x < image.Width - 1; x++)
    {
        byte north = previous[x];
        byte east = current[x + 1];
        byte south = next[x];
        byte west = current[x - 1];

        // Ported algorithm logic.
    }
}
```

---

## Avoid C++-Style Pointer Emulation

Do not reproduce pointer arithmetic through convoluted indexes merely to make the C# code visually resemble C++.

Avoid code such as:

```csharp
int position = baseOffset + y * stride + x;
```

when this is clearer:

```csharp
Span<byte> row = image.GetRowSpan(y);
ref byte pixel = ref row[x];
```

Some algorithms may use neighbour-direction tables. These can still be expressed clearly:

```csharp
private static readonly Point[] Neighbours8 =
[
    new(1, 0),
    new(1, 1),
    new(0, 1),
    new(-1, 1),
    new(-1, 0),
    new(-1, -1),
    new(0, -1),
    new(1, -1)
];
```

However, avoid allocating arrays inside processing methods.

---

## No Per-Pixel Allocations

Never allocate objects inside a pixel loop.

Incorrect:

```csharp
for (int x = 0; x < row.Length; x++)
{
    var point = new SomeReferenceType(x, y);
}
```

Prefer value types:

```csharp
var point = new Point(x, y);
```

or accumulate primitive coordinates directly.

Avoid LINQ in hot paths:

```csharp
row.Where(...)
row.Select(...)
pixels.Any(...)
```

Use explicit loops instead.

---

## Integer Behaviour

OpenCV algorithms often depend on precise integer behaviour.

Check carefully for:

* signed versus unsigned comparison;
* integer promotion rules;
* overflow;
* truncation;
* rounding;
* saturation;
* bit shifts;
* negative marker values;
* sentinel values.

A C++ implementation using `schar`, `char`, or `uchar` may behave differently depending on signedness.

Use explicit C# types.

Example:

```csharp
int value = row[x];
```

If negative marker values are required, a `byte` working image may be unsuitable. Consider:

```csharp
Mat<int>
```

or:

```csharp
Mat<sbyte>
```

for the internal representation, while still accepting `Mat<byte>` publicly.

Do not force negative algorithm markers into a `byte` without confirming the intended encoding.

---

## Image I/O and ImageSharp

The library currently has **no image-I/O dependency**. `SixLabors.ImageSharp` was
removed because version 4.x enforces a paid build-time licence key, which blocked the
build. Until an image library and licence are agreed, do **not** add ImageSharp (or a
similar package) as a dependency of the core library.

Consequences for ports:

* Ported algorithms operate purely on `Mat<T>`. Do not take a dependency on any
  external imaging type.
* Tests construct `Mat<T>` inputs in code and assert against precomputed expected
  values (see [OpenCV Comparison Tests](#opencv-comparison-tests)). Do not rely on
  loading image files.

If image I/O is needed later (for example to load real fixtures), it should be added
deliberately — as a separately licensed dependency, ideally isolated in an optional
interop project rather than the core library — with a small `Mat<T>` ↔ image
conversion utility that copies row by row (`GetRowSpan` / `GetReadOnlyRowSpan`) and
does **not** assume either side is contiguous.

---

## File Organization

Prefer one public function file per operation, with separate internal algorithm
files where an algorithm is large.

Example layout:

```text
Geometrikal.OpenCV.Managed/
  Core/
    Mat.cs
  ImgProc/
    Threshold.cs
    ThresholdType.cs
```

Group functions by their OpenCV module (`Core`, `ImgProc`, `ImgCodecs`, ...) once
more than one function exists in a module.

Avoid placing an entire port into one extremely large source file unless the
original algorithm is small.

There is currently no established folder convention beyond `Core/`; the layout above
is the recommended starting point.

---

## Documentation

Every public API should include XML documentation.

Example:

```csharp
/// <summary>
/// Applies a fixed-level threshold to a single-channel image.
/// </summary>
/// <param name="source">
/// Single-channel source image.
/// </param>
/// <param name="thresh">
/// Threshold value.
/// </param>
/// <param name="maxValue">
/// Value assigned to pixels that satisfy the threshold condition.
/// </param>
/// <param name="type">
/// Determines how the threshold is applied.
/// </param>
/// <returns>
/// A new thresholded image.
/// </returns>
public static Mat<byte> Threshold(
    Mat<byte> source,
    double thresh,
    double maxValue,
    ThresholdType type)
```

Document:

* accepted matrix type;
* channel requirements;
* source mutation behaviour;
* coordinate system;
* supported modes;
* differences from OpenCV.

---

## Testing Requirements

Tests use **xUnit**.

If no test project exists yet, create one:

```text
src/Geometrikal.OpenCV.Managed.Tests/Geometrikal.OpenCV.Managed.Tests.csproj
```

The test project must:

* use xUnit;
* reference `Geometrikal.OpenCV.Managed`;
* be added to `Geometrikal.OpenCV.Managed.slnx`.

Create tests for both normal and pathological inputs. Tailor them to the ported
function, but at minimum cover these general cases:

1. Empty or all-zero image.
2. Smallest meaningful input (for example a single pixel).
3. A typical valid image.
4. Inputs exercising each supported mode or overload.
5. A region of interest with a larger stride than its row length.
6. Invalid inputs that must throw (null, wrong channel count, unsupported mode).
7. Source image remains unchanged for non-mutating APIs.
8. Repeated calls return deterministic results.
9. Widths that are not a multiple of the vector size (add cases to
   `Regression/RegressionCases.cs`; see `tools/README.md` for the hash workflow).
10. For parallel paths: a `Core/CvParallelTests` case on an image large enough to
    split into several stripes. A memory-bound vector loop splits only above
    several million elements, because the work estimate is divided by
    `Vector<T>.Count`; the 517 × 301 images of `Output_DoesNotDependOnThreadCount`
    stay on one stripe for such loops. Add those loops to
    `LargeImage_Output_DoesNotDependOnThreadCount` and confirm with a stripe
    mutation.

A new test must be able to fail. After writing a test for an optimized path, break
the code under test on purpose (for example, seed a stripe from row 0) and confirm
that the test fails.

For an ROI test:

```csharp
var parent = new Mat<byte>(20, 20);
var roi = parent.Roi(3, 4, 10, 8);

// Fill and process the ROI.
// Verify results are identical to an equivalent packed 10×8 matrix.
```

This test is mandatory because it verifies that the port does not assume tightly packed rows.

---

## OpenCV Comparison Tests

There is currently **no** OpenCV .NET binding (such as OpenCvSharp or Emgu.CV)
referenced by the project, and no live native OpenCV test harness.

Compare against **precomputed golden data** generated from `opencv-python`.
This is the established convention, proven across the contours, drawing, and
morphology ports — follow it for new ports:

1. Write a deterministic generator script in
   `Geometrikal.OpenCV.Managed.Tests/tools/generate_<area>_golden.py` (seeded
   inputs, no timestamps) that runs the cv2 reference over a case battery and
   writes `TestData/<area>_golden.json` (inputs and expected outputs as
   base64 raw buffers plus the call arguments). Include a few in-script
   sanity assertions. Add the JSON to the test csproj with
   `CopyToOutputDirectory`.
2. Write a data-driven xUnit runner (`<Area>GoldenTests`) that replays every
   case through the managed API and asserts raw output buffers byte-for-byte,
   with a readable diff on failure.
3. **Version conflicts (suspect/adjudication mechanism).** The local cv2 may
   be older than the OpenCV 5 submodule the port follows. Design the battery
   to avoid known-changed areas where possible; when a golden case still
   fails, adjudicate it — determine (e.g. with an independent transcription
   of the OpenCV 5 C++, or by locating the upstream commit) whether the port
   or the golden is wrong. Genuine version differences are marked
   `"suspect": true` (with a reason) in the generator so the runner skips
   them, a visibility fact asserts the suspect set matches the adjudicated
   list exactly, and the OpenCV 5 behaviour is pinned in a dedicated
   `<Area>OpenCv5DifferenceTests` so the cases stay covered. Never silently
   drop or weaken a failing case.
4. Golden JSON stores NaN as `"NaN"`. Pass it to the port as the positive quiet NaN
   (`BitConverter.Int64BitsToDouble(0x7FF8000000000000)`). `double.NaN` in .NET
   has the sign bit set, and kernels that keep NaN bits then differ from cv2.
5. Behaviour that the local cv2 cannot generate (features new in OpenCV 5,
   hardware-dependent code paths) is covered by analytic tests with
   hand-derived expectations instead.

Construct inputs as `Mat<T>` in code or from the golden JSON rather than
loading image files (the library has no image-I/O dependency — see
[Image I/O and ImageSharp](#image-io-and-imagesharp)).

If a binding is later added to the test project, prefer direct output comparison.

Where you compare against OpenCV, compare the properties that matter for the ported
function, for example:

* output pixel values;
* result counts and orderings;
* coordinate offsets;
* any structured results the function returns.

Do not normalize away genuine differences (winding direction, omitted points,
ordering) unless OpenCV itself permits them. If you must tolerate a permitted
difference, write an explicit normalization helper and document why.

---

## Test Naming

Use descriptive names of the form `Method_Scenario_ExpectedResult`.

Examples:

```csharp
Threshold_EmptyImage_ReturnsEmptyResult
Threshold_BinaryType_ProducesExpectedPixels
Threshold_RoiWithStride_MatchesPackedImage
Threshold_DoesNotModifySource
Threshold_UnsupportedType_Throws
```

---

## Performance Expectations

The first implementation should avoid obvious performance problems but should prioritize correctness.

Required:

* no per-pixel object allocation;
* no LINQ in hot loops;
* no copying each row unless algorithmically required;
* no repeated property lookup where local variables improve clarity;
* no unnecessary source clones beyond the required working image;
* no use of `GetSpan()` on non-contiguous matrices.

Optional after profiling:

* list capacity estimation;
* pooled temporary buffers;
* SIMD;
* vectorized processing;
* continuous-memory fast paths;
* specialized paths for common modes.

---

## Unsafe Code

Do not use unsafe code for the initial port unless the algorithm cannot reasonably be expressed with spans.

Before adding unsafe code:

1. produce a correct span-based implementation;
2. add tests;
3. benchmark it;
4. identify a specific bottleneck;
5. isolate unsafe code in a small internal method;
6. retain a readable scalar reference implementation where practical.

Do not make the public API pointer-based.

---

## Disposal and Lifetime

Ordinary `Mat<T>` objects are backed by managed arrays.

They must not require:

```csharp
using
Dispose()
IDisposable
```

`Span<T>` and `Memory<T>` also do not require disposal.

If `ArrayPool<T>` is introduced for temporary buffers, always return rented arrays:

```csharp
T[] buffer = ArrayPool<T>.Shared.Rent(requiredLength);

try
{
    Span<T> working = buffer.AsSpan(0, requiredLength);

    // Processing.
}
finally
{
    ArrayPool<T>.Shared.Return(buffer);
}
```

Do not expose a pooled buffer as a long-lived result.

---

## Build and Validation

Build the managed project with:

```bash
dotnet build src/Geometrikal.OpenCV.Managed/Geometrikal.OpenCV.Managed.csproj
```

Or build the whole solution:

```bash
dotnet build Geometrikal.OpenCV.Managed.slnx
```

Run the tests:

```bash
dotnet test Geometrikal.OpenCV.Managed.slnx
```

If no test project exists yet, create one as described in
[Testing Requirements](#testing-requirements) before reporting completion.

Do not leave:

* compiler warnings;
* failing tests;
* dead code;
* unused copied OpenCV helpers;
* placeholder exceptions in supported code paths;
* unsupported modes that silently produce incorrect results.

---

## Copyright and Attribution

OpenCV is distributed under the **Apache License 2.0** (`src/OpenCV/LICENSE`);
a managed port is a derivative work and carries attribution obligations.

This repository is itself licensed under the Apache License 2.0 (root
`LICENSE` file). OpenCV attribution is recorded in the root `NOTICE` file —
when a port introduces newly ported OpenCV source files, add them to the
ported-sources list in `NOTICE`.

When porting substantial algorithmic code:

* preserve required copyright notices;
* add an attribution comment;
* identify the original OpenCV source file;
* identify the OpenCV version or commit where practical;
* include the relevant license text if required by the repository's licensing policy.

Example file header:

```csharp
// This implementation is based on an OpenCV algorithm.
//
// Original source:
//   modules/imgproc/src/thresh.cpp
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/imgproc/src/thresh.cpp
//
// OpenCV is licensed under the Apache License 2.0.
// See the OpenCV license and this repository's licensing documentation.
```

Do not copy comments claiming C++ constructs or memory behaviour that no longer apply to the managed implementation.

---

## Required Output From Claude

For each requested port, complete all of the following:

1. Identify the OpenCV source files used.
2. Briefly explain the algorithm and dependency chain.
3. State which OpenCV overload and modes are being implemented.
4. Add the managed implementation to `Geometrikal.OpenCV.Managed`.
5. Add or update required managed support types.
6. Add comprehensive tests.
7. Build the project.
8. Run the relevant tests.
9. Report any unsupported functionality.
10. Report any deliberate differences from OpenCV.
11. Report the files created or modified.

Do not stop after merely explaining the approach.

Make the code changes.

---

## Expected Final Report Format

Use this structure in the final response:

```markdown
## Ported Function

`FunctionName`

## OpenCV Sources

- `src/OpenCV/...`
- `src/OpenCV/...`

## Managed Files Changed

- `src/Geometrikal.OpenCV.Managed/...`
- `src/Geometrikal.OpenCV.Managed.Tests/...`

## Supported Behaviour

- ...
- ...

## Unsupported Behaviour

- None

or:

- `<mode>`
- `<overload>`

## Behavioural Differences

- None

or describe each difference precisely.

## Validation

- `dotnet build ...` — passed
- `dotnet test ...` — passed
- Number of tests added: N
```

---

## Function-Specific Prompt Template

Use the following prompt when assigning a function to port:

```markdown
Port the OpenCV function `<FUNCTION_NAME>` into the managed class library:

`src/Geometrikal.OpenCV.Managed/Geometrikal.OpenCV.Managed.csproj`

The OpenCV source is available under:

`src/OpenCV`

Follow the repository instructions in this document.

Requirements specific to this port:

- Implement: `<OVERLOADS_OR_MODES>`
- Input type: `<INPUT_TYPE>`
- Output type: `<OUTPUT_TYPE>`
- Source must remain unmodified: `<YES_OR_NO>`
- Required compatibility: `<DETAILS>`
- Explicitly excluded functionality: `<DETAILS>`

Inspect the OpenCV implementation and its tests before writing code.

Use managed arrays and `Span<T>`/`ReadOnlySpan<T>`.

Support stride and non-contiguous ROIs.

Do not use P/Invoke.

Do not add unsafe code unless clearly justified by profiling.

Add xUnit tests and run the build and test commands before reporting completion.
```
