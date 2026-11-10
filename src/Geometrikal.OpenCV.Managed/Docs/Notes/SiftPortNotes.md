# SIFT port notes

Notes for the port of OpenCV `cv::SIFT` (`Features/Sift.cs`, internals in
`Features/Internal/Sift*.cs`) and the math helpers in `Core/Internal/FastMath.cs`. File
references are against the OpenCV 5.0.0 submodule at `src/OpenCV`.

## Sources

| Port | OpenCV source |
|---|---|
| `Sift` (constructor, `Detect`, `Compute`, `DetectAndCompute`) | `modules/features/src/sift.dispatch.cpp` (`SIFT::create`, `SIFT_Impl::detectAndCompute`), `modules/features/src/feature2d.cpp` (`Feature2D::detect`, `Feature2D::compute`) |
| `SiftPyramid.CreateInitialImage` | `sift.dispatch.cpp` (`createInitialImage`) |
| `SiftPyramid.BuildGaussianPyramid`, `BuildDoGPyramid` | `sift.dispatch.cpp` (`buildGaussianPyramid`, `buildDoGPyramidComputer`) |
| `SiftExtrema` | `sift.dispatch.cpp` (`findScaleSpaceExtrema`), `sift.simd.hpp` (`findScaleSpaceExtremaT`, `adjustLocalExtrema`, `calcOrientationHist`), `modules/core/include/opencv2/core/operations.hpp` (`Matx_FastSolveOp<_Tp, 3, 3, 1>`), `matx.inl.hpp` (`Matx_DetOp<_Tp, 3>`) |
| `SiftDescriptor` | `sift.dispatch.cpp` (`unpackOctave`, `calcDescriptorsComputer`), `sift.simd.hpp` (`calcSIFTDescriptor`) |
| `FastMath` | `modules/core/src/mathfuncs_core.simd.hpp` (`atan_f32`, `v_atan_f32`, `fastAtan32f`, `magnitude32f`, `exp32f`) |

## Public API

```csharp
enum SiftDescriptorType { Float32 = 5, Byte = 0 }

sealed class Sift
{
    Sift(int nFeatures = 0, int nOctaveLayers = 3, double contrastThreshold = 0.04,
         double edgeThreshold = 10, double sigma = 1.6,
         SiftDescriptorType descriptorType = SiftDescriptorType.Float32,
         bool enablePreciseUpscale = true);
    const int DescriptorSize = 128;
    int NFeatures; int NOctaveLayers; double ContrastThreshold; double EdgeThreshold;
    double Sigma; SiftDescriptorType DescriptorType { get; } bool EnablePreciseUpscale;

    List<KeyPoint> Detect(Mat<byte> image, Mat<byte>? mask = null);
    List<KeyPoint> DetectAndCompute(Mat<byte> image, Mat<byte>? mask, out Mat<float>? descriptors);
    List<KeyPoint> DetectAndCompute(Mat<byte> image, Mat<byte>? mask, out Mat<byte>? descriptors);
    void Compute(Mat<byte> image, IReadOnlyList<KeyPoint> keypoints, out Mat<float>? descriptors);
    void Compute(Mat<byte> image, IReadOnlyList<KeyPoint> keypoints, out Mat<byte>? descriptors);
}
```

- Input: 8-bit images with 1, 3 or 4 channels. 3 and 4 channels convert with
  `ColorConversions.CvtColor` (`Bgr2Gray`, `Bgra2Gray`); OpenCV calls `COLOR_BGR2GRAY` for
  both, which gives the same values.
- The descriptor overload must match `DescriptorType`; a mismatch throws
  `InvalidOperationException`. OpenCV selects the type in `SIFT::create` only.
- A result without keypoints returns `null` descriptors. `Mat<T>` cannot have zero rows;
  OpenCV returns an empty matrix.
- `Compute` builds the pyramid from the keypoint octaves (`useProvidedKeypoints = true`): it
  upscales only when a keypoint has octave −1. Octaves below −1 or layers above
  `nOctaveLayers + 2` throw `ArgumentException` (OpenCV `CV_Assert`).

## Pipeline and reuse

1. `createInitialImage`: gray, `Conversions.ConvertTo<byte, float>`, then for octave −1 either
   `Geometry.WarpAffine` (matrix `[[0.5, 0, 0], [0, 0.5, 0]]`, `InterpolationFlags.Linear`,
   `WarpFlags.InverseMap`, `BorderType.Reflect`) or `Geometry.Resize` with
   `InterpolationFlags.Linear`, then `Filters.GaussianBlur` with
   `sig_diff = sqrtf(max(σ² − 4·0.5², 0.01))` in single precision.
2. `buildGaussianPyramid`: incremental sigmas in double precision (`pow`, `sqrt`), octave bases
   by `Geometry.Resize(..., InterpolationFlags.Nearest)` to `(w / 2, h / 2)`, other levels by
   `Filters.GaussianBlur`.
3. `buildDoGPyramid`: `Arithmetic.Subtract` per level, levels in parallel
   (`ParallelRows.ForEachSerial`).
4. `findScaleSpaceExtrema`, `KeyPointsFilter.RemoveDuplicatedSorted`,
   `KeyPointsFilter.RetainBest` (when `nFeatures > 0`), octave −1 rescale,
   `KeyPointsFilter.RunByPixelsMask`.
5. `calcDescriptors`: one descriptor per keypoint, keypoints in parallel.

The 2× upscales are exact for 8-bit sources: every bilinear weight is 0, 0.25, 0.5 or 0.75 and
every intermediate value is an exact multiple of 1/16 below 2^24. The OpenCV 5.0 float warp
kernel (`p00 + α·(p01 − p00)`) and the port table path therefore give the same values, although
`GeometryPortNotes.md` lists the 5.0 float warp kernel as not ported.

## Bit-exactness

The target is cv2 5.0.0 with IPP off and AVX, AVX2, FMA3 and AVX-512 disabled
(`OPENCV_CPU_DISABLE`). The SIFT dispatch then runs 128-bit vector code (4 float lanes, no
FMA). Achieved: keypoints (position, size, angle, response, octave, class) and descriptors
equal cv2 bit for bit in all 22 golden cases, including a 1024 × 768 image with 5290 keypoints.

Operation-order rules:

| Step | Rule |
|---|---|
| `calcOrientationHist` smoothing | OpenCV vector form `(t[−2] + t[2])·1/16 + ((t[−1] + t[1])·4/16 + t[0]·6/16)` for all 36 bins (36 is a multiple of 4, so the scalar tail never runs). The scalar form groups differently. |
| `calcSIFTDescriptor` first norm | 4 lane accumulators of `r·r` (mul then add), reduced as `(a0 + a2) + (a1 + a3)` (SSE `v_reduce_sum`). |
| `calcSIFTDescriptor` second norm | Sequential scalar sum after the `min(r, thr)` clamp. |
| Vector and scalar loops of the histogram update, `atan_f32`, `magnitude32f`, the extrema scan and the 0..255 conversion | Same per-element operations without FMA; the port uses `Vector<T>` lanes with the scalar order. |
| `adjustLocalExtrema` solve | `Matx_FastSolveOp<float, 3, 3, 1>`: Cramer's rule in single precision with `d = 1 / det`; zero result for `det == 0`. |
| `Matx::dot` | `((0 + a0·b0) + a1·b1) + a2·b2`. |
| `kpt.octave` | `cvRound((xi + 0.5)·255)` in double precision. |
| `kpt.size` | `σ·powf(2, (layer + xi) / nOctaveLayers)·(1 << o)·2` in single precision. |

Transcendental functions:

- `hal::exp32f`: the MSVC build of OpenCV (the Windows cv2 wheel) compiles the
  `std::exp(float)` loop, not the table and polynomial code of other compilers. With IPP off,
  cv2 `exp` equals the C runtime `expf` (checked on 20 000 values). `MathF.Exp` calls the same
  function. With IPP on, cv2 uses `ippsExp_32f_A21`, which differs in about 20% of values.
- `cosf`, `sinf`, `powf`: C runtime functions through `MathF.Cos`, `MathF.Sin`, `MathF.Pow`.
- `fastAtan32f`, `magnitude32f`: polynomial and `sqrt` code, ported exactly.
- `FastMath_MatchesOpenCvGolden` compares the three helpers with cv2 `exp`, `phase` (degrees)
  and `magnitude` on 3001 values.

A Linux or macOS cv2 build uses the table `exp32f` and other C runtime `cosf`/`sinf`/`powf`
implementations. Results on those platforms can differ in the last bits of angles and in
descriptor values; keypoint positions, sizes and responses do not use these functions.

## Keypoint order and threads

- OpenCV collects keypoints in thread-local vectors (`TLSDataAccumulator`) and concatenates
  them in an implementation-defined thread order. `removeDuplicatedSorted` then sorts the list,
  so the OpenCV result order is deterministic except for keypoints that compare equal.
- The port scans one work item per DoG row (octave, layer, row) in parallel, stores the
  keypoints of each row in its own list and concatenates the lists in serial scan order.
  The order before the sort is therefore the order of a one-thread OpenCV run, for every
  thread count. `KeyPointsFilter.RemoveDuplicatedSorted` keeps the input order of equal keys.
- `retainBest` order is implementation-defined in OpenCV (`nth_element`, `partition`). The
  port returns the kept keypoints by decreasing response. The golden runner compares those
  cases after a canonical sort.
- Descriptors are computed per keypoint without shared state. All outputs are identical for 1
  and 8 threads (`SiftBasicTests.Output_DoesNotDependOnThreadCount`).

## Performance structure

- Gaussian pyramid: `Filters.GaussianBlurInto` into pooled levels (separable float engine, parallel rows).
- DoG: `Arithmetic.Subtract` into pooled levels (vectorized), levels in parallel.
- Extrema scan: one `Vector<float>` block of columns at a time, as the OpenCV vector loop:
  threshold test, then the 8-neighbour maxima and minima of the current, previous and next
  levels. Only set lanes go to `adjustLocalExtrema`. Rows run in parallel.
- Descriptor: per row of the sampling square, the column range is first narrowed to a
  superset of the rotated 4 × 4 window (two-step margin); the exact OpenCV test then selects
  the samples. Sample coordinates, gradients and weights are computed in `Vector<float>`
  lanes. The trilinear weights and bin indices are computed in lanes; the histogram update
  stays in sample order. `fastAtan32f` and `magnitude32f` are vector loops; `expf` is scalar.
- Measured on a 1024 × 1024 synthetic image (6924 keypoints, no benchmark harness, busy
  machine): about 360 ms on 1 thread and 100 ms on 16 threads. The descriptor stage is the
  largest part (about 50% on 1 thread).

## Pooled temporaries (option 1)

- `Sift.Run` rents every pyramid level from `ArrayPool<float>.Shared` as a `PooledMat<float>`
  and returns all levels in a `finally` block after the descriptor stage: the float base image
  and its 2× upscale, the Gaussian pyramid and the DoG pyramid. The levels are written with
  `Filters.GaussianBlurInto` (internal), `Geometry.Resize`/`Geometry.WarpAffine` with a
  destination, and `Arithmetic.Subtract` with a destination. The `byte` → `float` conversion
  of the input is a local loop into a pooled matrix.
- `SiftExtrema.Find` maps a scan unit to (octave, layer, row) through a prefix array instead
  of a list of tuples. Each `ParallelRows` stripe collects its keypoints in one list (not one
  list per row); the first stripe appends to the result list directly. The per-stripe list
  array is pooled.
- The sample buffers of `calcOrientationHist` and `calcSIFTDescriptor` (per stripe) are
  pooled arrays; they grow by renting a larger array.
- `KeyPointsFilter.RemoveDuplicatedSorted` rents the keypoint copy and the sort index array.
  The index tie-break makes the sort order total, so the kept set and order do not change.
- The keypoint list and the descriptor matrix stay new allocations.
- Remaining allocations: the result list growth, about 1.5 KB per pyramid level (kernel
  arrays, filter job, closures), and `CvtColor` for 3- and 4-channel input.

Allocated bytes per call, `SiftBenchmarks` (benchmark gray input, 1 thread):

| Case | Size | Before | After |
|------|------|-------:|------:|
| DetectAndCompute_U8 | 256 | 17.06 MB | 408 KB (descriptors 266 KB) |
| DetectAndCompute_U8 | 1024 | 262.76 MB | 5.61 MB (descriptors 4.61 MB) |

Before the pooled `RemoveDuplicatedSorted` arrays, the values were 425 KB (256) and 5.90 MB
(1024).

`Detect` alone after the first change: 149 KB (256) and 1.28 MB (1024).

## Golden constraints (`tools/generate_sift_golden.py`)

- cv2 5.0.0, IPP and OpenCL off, one thread, `OPENCV_CPU_DISABLE` as in the other
  generators.
- Images are built with an integer hash formula (bilinear value noise, small noise, 6
  rectangles); `SiftTestImages` in the test project rebuilds them and the runner checks the
  SHA-256 of each image.
- Cases: sizes 24 × 20 to 1024 × 768, odd sizes, BGR and BGRA input, mask, `nfeatures` 50
  and 200, byte descriptors, `enable_precise_upscale = false`, `nOctaveLayers` 2, 4 and 5,
  other sigma, contrast and edge thresholds, `detect` only, `compute` with keypoints of
  another detector (with and without octave −1 keypoints).
- Descriptors of cases with more than 1500 keypoints are stored as SHA-256.
- No suspect cases.
