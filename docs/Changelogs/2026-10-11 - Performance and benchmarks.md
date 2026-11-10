# Performance work and benchmarks for the managed OpenCV port

**Date:** 2026-10-11 00:58:00

## Summary

The session added a benchmark project that compares the managed port with cv2. It then made
filtering and morphology 2× to 200× faster with byte-identical output. It also added
multithreading with a public thread-limit setting. Two correctness defects were found and
fixed during the work. Output stays bit-exact to the scalar port and to cv2: golden tests,
924 regression hashes and thread-count tests pass (1891 tests).

## Changes

### Benchmark project (`PerceptionAg.CV.OpenCV.Benchmark`, new)

- BenchmarkDotNet suites for filtering, morphology, contours and drawing at 256×256 and
  1024×1024. Large-kernel cases: GaussianBlur 15×15 (U8, F32), Blur 15×15, MedianBlur 15,
  Filter2D 11×11, Sobel ksize 7, Erode rect 15×15, Dilate ellipse 15×15.
- `tools/bench_cv2.py` runs the same cases in cv2 in three modes: `baseline` (1 thread, SSE,
  no IPP), `single` (1 thread, AVX2, IPP) and `default` (all threads).
- `tools/compare.py` merges the results into `results/comparison.md`.
- Program options: `--quick`, `--checksums` (input parity with Python), and `--threads N|all`
  (writes to `BenchmarkDotNet.Artifacts-mt`). The full job uses 15 iterations of at least 100 ms.

### Performance (no public API change)

- `PooledMat<T>` and `Borders.CopyMakeBorderPooled`: internal temporaries come from `ArrayPool`.
- `CopyMakeBorder` copies the interior with span copies; only border columns use the index map.
- Per-element overhead removed: row references hoisted out of element loops, min/max selected
  at compile time (`MinMaxOps`: `MinOp`/`MaxOp` with `std::min`/`std::max` semantics).
- SIMD with `Vector<T>` in median, morphology, separable filters, fixed-point Gaussian,
  `filter2D` and the box filter. `FilterVectors` provides widening, `cvRound` and saturation.
- Box filter U8 (kernel area ≤ 256): OpenCV's fixed-point divide replaces a per-call lookup table.
- MedianBlur ksize ≥ 7: constant-time column histograms (as `medianBlur_8u_O1`), processed in
  column stripes, with a branch-free SIMD bin search.
- MedianBlur 3×3 and 5×5: an overlapping final vector replaces the scalar row tail. U8 3×3 uses
  Paeth's column-sort method.
- `filter2D` and the float separable passes keep accumulators in registers across taps.
  `filter2D` converts U8 sources to float once.

### Multithreading (public API addition)

- `Core/CvParallel.MaxDegreeOfParallelism`: process-wide thread limit, as `cv::setNumThreads`.
  Default `Environment.ProcessorCount`; 1 disables multithreading.
- `Core/Internal/ParallelRows`: splits rows into stripes. Used by the separable engine, fixed-point
  Gaussian, `filter2D`, box filter (integer paths), all median paths and morphology. The box
  filter for float sources stays single-threaded because its `double` running sums depend on
  the accumulation history.

### Correctness fixes

- `PixelOps.HLine`: `FillPoly` on multi-channel images threw on empty scanline runs. Empty runs
  are now skipped, as `ICV_HLINE_X` does.
- Box filter U8→U8 with an even kernel area ≤ 256 rounded exact halves to even. OpenCV's
  `ColumnSum<ushort, uchar>` rounds them up through its fixed-point divide (for example, cv2
  `blur(zeros, (2,1))` is 1). The port now matches cv2. The test that pinned the old values is
  corrected, and the golden generator now emits even-kernel cases for this path.

### Tests

- `Tests/Regression`: 924 hash cases on wide, odd-sized images; hashes generated from the
  scalar commit `f613774` by `tools/RegressionSnapshotGen`.
- `Tests/Core/CvParallelTests`: every parallel path on a 517×301 image with 1 and 8 threads must
  give identical bytes.
- `FiltersBasicTests`: wide-image median brute-force comparison (stripe boundaries), even-kernel
  blur tests, multi-channel `FillPoly` test.
- `filters_golden.json`: 5 even-kernel blur cases added.

### Documentation

- `Docs/Notes/PerformanceAnalysis.md`: causes, measurements, remaining gaps and next steps (A–G).
- `Docs/Tutorials/MemoryAndSimd.md`: tutorial on pooling, references, compile-time dispatch,
  SIMD, scalar tails, register blocking, multithreading and bit-exactness.
- `Docs/Notes/FilteringPortNotes.md`: fixed-point box filter rule.
- `CLAUDE.md` and the benchmark README updated.

## Notes

- Results page: https://claude.ai/artifact/D7rGGiR4dJnBS18FUKpqKP (private).
- Selected results at 1024×1024, single thread, vs cv2 1 thread/SSE/no IPP: MedianBlur 3×3
  0.31×, MedianBlur 5×5 0.52×, Blur 5×5 0.30×, Filter2D 11×11 0.42×, Sobel ksize 7 0.50×,
  Erode rect 3×3 0.58×. With all cores vs cv2 default: MedianBlur 0.14×–0.50×, Filter2D 11×11
  0.18×.
- Open items:
  - The 256×256 three-channel cases are dominated by large-object-heap allocation of the
    result. The application can raise `System.GC.LOHThreshold`.
  - Float filters are memory-bound; a row ring buffer (step E) is the next structural change.
  - The first multithreaded run showed two defects, both fixed and confirmed by a rerun:
    closure-captured locals in the separable engine's stripe lambdas (now static methods;
    SepFilter2D 1.59 ms → 1.18 ms) and over-splitting of cheap morphology passes (work
    estimate divided by vector lanes; Erode 3×3 with 16 threads 338 µs → 255 µs).
  - Some cases differ by up to 2× between benchmark runs of the same code while each run has
    a low coefficient of variation. The cause is not identified.
  - Documentation updated: `Porting.md` (current project state, optimization rules, testing
    requirements), `CLAUDE.md` (docs index), `PerformanceAnalysis.md` (status after A–D and G),
    the tutorial checklist, a new benchmark `CLAUDE.md`, and the root README.
- cv2 reference is 4.5.5; the ported source is OpenCV 5.0.0.
