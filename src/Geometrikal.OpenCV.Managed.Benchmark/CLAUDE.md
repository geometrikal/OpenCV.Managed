# Geometrikal.OpenCV.Managed.Benchmark

BenchmarkDotNet suites for the managed OpenCV port, OpenCvSharp twins of these suites and a
matching cv2 script. `README.md` describes the cv2 modes, the OpenCvSharp suites, the run commands
and the measurement notes.

## Add a benchmark case

1. Add a `[Benchmark]` method to the class for the subsystem (`CoreBenchmarks`,
   `ColorBenchmarks`, `ThresholdBenchmarks`, `FilteringBenchmarks`, `MorphologyBenchmarks`,
   `ContourBenchmarks`, `DrawingBenchmarks`, `GeometryBenchmarks`, `ConnectedComponentsBenchmarks`, `DftBenchmarks`, `EdgeBenchmarks`, `DistanceBenchmarks`, `AdvancedFilterBenchmarks`, `HistogramBenchmarks`, `ShapeDescriptorBenchmarks`, `ShapeFittingBenchmarks`, `FloodFillBenchmarks`, `PyramidBenchmarks`, `LinearAlgebraBenchmarks`, `FeatureBenchmarks`, `SiftBenchmarks`, `MatcherBenchmarks`, `CornerBenchmarks`, `EstimationBenchmarks`, `UndistortBenchmarks`, `AlignmentBenchmarks`, `StitchingBenchmarks`, `SeamBenchmarks`, `StitchRegistrationBenchmarks`, `OpticalFlowBenchmarks`, `PnpBenchmarks`, `EpipolarBenchmarks`, `DrawingAaBenchmarks`). OpenCvSharp does not expose the stitching warpers, blenders, compensators, seam finders, estimators or bundle adjusters; those suites have no twin.
  cv2 4.5.5 lacks `stackBlur` and the default install lacks `cv2.ximgproc`; `bench_cv2.py` skips
  those cases when they are missing. Install opencv-contrib for guidedFilter and use cv2 >= 4.7
  for stackBlur.
  cv2 uses IPP for the 5x5 distance transform, the precise transform and the 8-bit L1 transform
  in the default and single modes; compare those cases with the baseline mode. The 8-bit L1
  distance transform is parallel in the port. Build kernels and
   other fixed arguments in `[GlobalSetup]`, not in the method.
2. Add the same case to `build_cases` in `tools/bench_cv2.py`. The key is
   `"<Class without 'Benchmarks'>.<Method>"`, for example `"Filtering.Blur15x15_U8"`.
   `compare.py` joins the two sides by this name.
3. Use identical arguments on both sides. Use kernel coefficients that are exact in
   float32 (for example `1/128`), so that both sides get the same kernel.
4. Check that both sides take the same code path. Example: cv2 `filter2D` uses a DFT for
   kernel areas ≥ 130 (U8→U8 on SSE3 hardware); the port always convolves directly.
5. Inputs come from `Inputs.cs` and `bench_cv2.py`, which share one integer hash. Run
   `dotnet run -c Release -- --checksums` and `python tools/bench_cv2.py --checksums`
   after changing an input generator; the sums must be equal.
6. Classes whose operations are multithreaded call `Threads.Apply()` in `[GlobalSetup]`.
7. Add the same method name to the OpenCvSharp twin `OpenCvSharp<Suite>Benchmarks` (for example
   `OpenCvSharpFilteringBenchmarks`). Use the same `[Params]` and the same arguments.
   - Build native inputs with `OcvInputs.ToOcv` from the port inputs; do not regenerate them.
   - Call `OcvInputs.ApplyThreads()` in `[GlobalSetup]` (sets `Cv2.SetNumThreads` from `CV_THREADS`).
   - Allocate the result in `using var dst = new Mat();` and return `OcvInputs.Consume(dst)`. For
     an `Into` or in-place port case, write to a preallocated native `Mat` instead.
   - Dispose every native object in `[GlobalCleanup]`.
   - Omit the case when OpenCvSharp does not expose the function (for example `stackBlur`).
   - Pass a `Mat` directly to `InputArray`, `OutputArray` and `InputOutputArray` parameters. In
     OpenCvSharp 4.13 a stored `InputOutputArray` fails with "Invalid OutputArray", and
     `InputArray.Create(IEnumerable<Mat>)` passed to `FillPoly` crashes natively.
   - Use the `Point[][]` overloads of `FillPoly`, `Polylines` and `DrawContours`. The
     `DrawContours` overload with `IEnumerable<Mat>` contours fails with "Unknown/unsupported array type".

The OpenCvSharp build is OpenCV 4.13 with IPP 2022.2 and AVX2/AVX-512 dispatch. Its outputs differ
from the port in rounding for some floating-point cases; see `README.md`.

## Run

- In deep paths (for example agent worktrees), BenchmarkDotNet's generated project fails with
  MSB3030 (260-character path limit). Run from a short path or add `--inProcess`.

- Build first: `dotnet build -c Release`. A build error makes `dotnet run` exit at once and
  a background run produces no results.
- A filtered run (`--filter *Gaussian*`) replaces the result files of the classes it runs.
  Run `--filter *` before `tools/compare.py` to get every case.
- `bench_cv2.py --filter "Core.,Color."` takes a comma-separated list. A filtered run
  merges its cases into the existing `results/cv2-<mode>.json`; an unfiltered run replaces it.
- `--threads all` runs with all cores and writes to `BenchmarkDotNet.Artifacts-mt/`; `--threads N`
  writes to `BenchmarkDotNet.Artifacts-tN/`. Compare the all-cores run with the cv2 `default`
  mode.
- `DOTNET_MaxVectorTBitWidth=512` before `dotnet run` widens `Vector<T>` to 512 bits on AVX-512
  hardware. The benchmark processes inherit the variable, and the results go to folders with a
  `-v512` suffix (for example `BenchmarkDotNet.Artifacts-mt-v512/`).
- A full single-threaded run takes about 10 minutes. `--quick` runs 3 iterations for spot
  checks. `--fast` runs 10 iterations of at least 50 ms. `--inProcess` runs the job in the
  `dotnet run` process (no generated project; required in deep paths).
- Do not build, test or run other CPU work during a run. Compare runs only when the machine
  is quiet; some cases differ by up to 2× between runs.

## Results page

The results page is the claude.ai artifact
https://claude.ai/artifact/D7rGGiR4dJnBS18FUKpqKP. It embeds the data as `const DATA` with
one row per case and size: `managed`, `mt`, `alloc`, `previous`, `before` (scalar port) and
`cv2.{baseline,single,default}`, all in nanoseconds. To update it, read the published page,
replace the row values from `BenchmarkDotNet.Artifacts*/results/*-report-full-compressed.json`
and `results/cv2-*.json`, update the findings text, and publish to the same URL.
- `OpenCvSharpStitchRegistrationBenchmarks.Setup` crashes natively (access violation in
  `CvDetail.ComputeImageFeatures`), and a filter such as `*StitchRegistrationBenchmarks*` also
  selects it. Run the port class with `Geometrikal.OpenCV.Managed.Benchmark.StitchRegistrationBenchmarks*`.
