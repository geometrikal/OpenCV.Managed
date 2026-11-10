# Geometrikal.OpenCV.Managed.Benchmark

Performance comparison of the managed OpenCV port against cv2 (Python) and OpenCvSharp (native
OpenCV 4.13).

## Scope

The benchmark covers the ported subsystems at image sizes 256x256 and 1024x1024:

| Class | Cases |
|---|---|
| `FilteringBenchmarks` | GaussianBlur (U8, U8C3, F32), Blur, MedianBlur (3, 5, 7), Filter2D, SepFilter2D, Sobel, Scharr, Laplacian, CopyMakeBorder (allocating and into a reused destination); large kernels: GaussianBlur 15×15 (U8, F32), Blur 15×15, MedianBlur 15, Filter2D 11×11, Sobel ksize 7 |
| `MorphologyBenchmarks` | Erode, Dilate, MorphologyEx (Open, Gradient); large kernels: Erode rect 15×15, Dilate ellipse 15×15 |
| `ContourBenchmarks` | FindContours (Tree/None, External/Simple, List/Tc89L1) |
| `DrawingBenchmarks` | Line, Rectangle, Circle, Ellipse (64 shapes per call), FillPoly, Polylines, DrawContours |

Each port class has an OpenCvSharp twin; see [OpenCvSharp suites](#opencvsharp-suites).

`tools/bench_cv2.py` implements the same cases with the same names. `Inputs.cs` and
`bench_cv2.py` generate identical inputs from one integer hash. The `--checksums`
option of both programs prints input sums to confirm this.

## cv2 configurations

The managed benchmarks run single-threaded by default (`CvParallel.MaxDegreeOfParallelism = 1`).
cv2 runs in three configurations:

| Mode | Threads | SIMD | IPP |
|---|---|---|---|
| `baseline` | 1 | SSE baseline (`OPENCV_CPU_DISABLE=AVX,AVX2,AVX512F,FMA3`) | off |
| `single` | 1 | runtime dispatch (AVX2) | on |
| `default` | all cores | runtime dispatch (AVX2) | on |

`baseline` is the closest comparison to the single-threaded managed port. `default` is the
typical production cv2 configuration. Compare it with the multithreaded managed run.

## OpenCvSharp suites

Each port suite `<Suite>Benchmarks` has a twin `OpenCvSharp<Suite>Benchmarks` (for example
`OpenCvSharpFilteringBenchmarks`). The twin runs the same cases through OpenCvSharp, which calls
native OpenCV in-process without the Python call cost.

- Native build: `OpenCvSharp4.runtime.win` 4.13.0.20260627, OpenCV 4.13 with the contrib modules,
  IPP 2022.2 and runtime dispatch to AVX2 and AVX-512.
- Method names and `[Params]` are the same as in the port class. The calls use the same
  arguments (kernel sizes, flags, thresholds, interpolation, border types, depths, channel counts,
  shapes).
- Inputs are native copies of the port inputs (`OcvInputs.ToOcv`), so both sides process
  identical pixels.
- Each call allocates its result in a new `Mat` and disposes it, as the port allocates its result.
  The `Into` cases and the in-place cases write to a preallocated native `Mat`, as the port cases do.
- The `CV_THREADS` variable (`--threads N|all`) sets `Cv2.SetNumThreads`. The default is 1.
- Results go to the same artifacts folders as the port results (`BenchmarkDotNet.Artifacts/`,
  `BenchmarkDotNet.Artifacts-mt/`, `BenchmarkDotNet.Artifacts-tN/`).
- `compare.py` shows the OpenCvSharp rows in separate sections named `OpenCvSharp<Suite>`. In
  those sections the `Managed` column holds the OpenCvSharp time.
- OpenCvSharp does not expose `stackBlur`. `OpenCvSharpAdvancedFilterBenchmarks` has no StackBlur cases.
- `FillPoly`, `Polylines` and `DrawContours` take managed point arrays, as the port does.
  OpenCvSharp marshals these arrays on each call.

Run only the OpenCvSharp suites:

```
dotnet run -c Release -- --filter *OpenCvSharp*
```

OpenCV 4.13 is not bit-exact with the port (OpenCV 5.0.0 source). The outputs are identical for
most cases. The floating-point DFT, DCT, resize, Gaussian blur, bilateral filter, color conversion
and L2 distance transform cases differ by small rounding amounts. The 8-bit bicubic resize and
bilateral filter differ by at most 1.

## Run

Run from this directory:

```
dotnet run -c Release -- --filter *
dotnet run -c Release -- --threads all --filter *Filtering* *Morphology*
python tools/bench_cv2.py
python tools/compare.py
```

1. BenchmarkDotNet writes single-threaded results to `BenchmarkDotNet.Artifacts/results/`.
2. `--threads N|all` sets the thread limit and writes to `BenchmarkDotNet.Artifacts-mt/` (all)
   or `BenchmarkDotNet.Artifacts-tN/` (N).
   Contours and drawing are single-threaded, so the multithreaded run omits them.
3. `bench_cv2.py` writes `results/cv2-<mode>.json`.
4. `compare.py` merges all of them into `results/comparison.md`. The columns `Managed MT`
   and `Ratio MT vs cv2 default` appear when the multithreaded results exist.

A filtered BenchmarkDotNet run replaces the result files of the classes it runs. Run the
full filter before `compare.py` to get every case.

Use `--filter *Gaussian*` (BenchmarkDotNet) and `--filter Gaussian` (`bench_cv2.py`)
to run a subset. Add `--quick` to run BenchmarkDotNet's 3-iteration short job instead
of the full job (3 warmup and 15 measured iterations of at least 100 ms each); use it
for spot checks only.

## Measurement notes

- Managed and cv2 times are medians per call.
- The benchmark processes run with `DOTNET_TC_QuickJitForLoops=0` (job environment variable
  and `TieredCompilationQuickJitForLoops` in the csproj for `--inProcess`). The JIT compiles
  methods with loops fully optimised at once. With the default setting, the 3 warmup iterations
  can end before tier-up, and the first cases measure tier-0 or OSR code: `OrbDetect` at 256
  measured 3.2–4.3 ms, against 1.4–1.5 ms with this setting. Results from before 2026-10-16 that
  were taken with `--fast` have this error in some 256 cases.
- cv2 drawing times include the Python loop over 64 shapes and the Python-to-C++ call cost.
  This cost is approximately 1 us per call.
- cv2 filter and morphology calls allocate the output array, as the managed port does.
- `Filter2D11x11_U8` uses a kernel area of 121. cv2 switches to its DFT path at area 130 for
  U8→U8 on SSE3 hardware, so both sides use direct convolution.
- The local cv2 is 4.5.5. The ported source is OpenCV 5.0.0.
