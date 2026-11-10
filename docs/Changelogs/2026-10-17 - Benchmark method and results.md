# Benchmark method and results

**Date:** 2026-10-17 01:03:00

## Summary

The benchmark job now runs with `DOTNET_TC_QuickJitForLoops=0`. Before this change, the 3
warmup iterations of the `--fast` job could end before tier-up, so the first cases of a class
measured tier-0 or OSR code. All suites were run again with the fix (two full passes) and the
results page is at version 26. The agent worktrees and integration branches of the
2026-10-16 fleet are removed.

## Changes

- **Benchmark job (f713b56):**
  - `Program.cs` sets the job environment variable `DOTNET_TC_QuickJitForLoops=0` for
    out-of-process runs.
  - The benchmark csproj sets `TieredCompilationQuickJitForLoops=false` for `--inProcess` runs.
  - The benchmark `README.md` explains the setting in the measurement notes.
- **Warm-up options compared** (`FeatureBenchmarks`, two alternating rounds, `OrbDetect` 256):

  | Setting | Round 1 | Round 2 |
  |---|---|---|
  | 3 warmups (old) | 4.34 ms | 3.23 ms |
  | 10 warmups | 2.33 ms (CV 17 %) | 1.74 ms (CV 24 %) |
  | QuickJitForLoops=0 | 1.43 ms | 1.36 ms |

- **Steady-state effect of QuickJitForLoops=0:** an A/B run on 52 cases (Pyramid, Color, Core,
  ShapeDescriptor, Threshold, ORB, MatchTemplate; two rounds, minimum per case) gives new ÷
  default = 0.95 at 1 thread and 0.90 at 16 threads. No case is slower outside the noise. The
  loss of Dynamic PGO for methods with loops does not slow these cases.
- **Full rerun:** all suites, port and OpenCvSharp, 1 and 16 threads, `--fast --inProcess`,
  each class in its own process. The OpenCvSharp `StitchRegistration` twin is excluded (native
  crash). Two passes, the second in reverse class order. Each case uses the lowest median of the
  two passes and, where present, of the two A/B rounds.
- **Corrected cases** (256, 1 thread, old → new): `OrbDetect` 4.46 → 1.13 ms,
  MatchTemplate CCoeffNormed T16 5.90 → 1.15 ms, `LineAa1px` 0.84 → 0.12 ms,
  `CornerHarris` 1.15 → 0.40 ms, `InitUndistortRectifyMap` 32FC1 1.33 → 0.35 ms.
- **Geomean new ÷ old:** port 0.92 (1 thread), 0.86–0.92 (16 threads); OpenCvSharp
  0.89–0.92.
- **Results page version 26** (https://claude.ai/artifact/D7rGGiR4dJnBS18FUKpqKP):
  - New port times (`managed`, `mt`, `peer`) and OpenCvSharp times (`ocvs`, `ocvs_mt`) for all
    rows; old values are kept in `pre_qjl`.
  - New rows: `Filtering.CopyMakeBorderInto16_U8C3` at 256 and 1024.
  - Fixed size keys: cases without parameters map to 1024; `Points=2000` maps to 1024.
  - Updated Method text and a new Findings bullet.
- **Repository cleanup:** 30 merged `worktree-agent-*` worktrees and branches, and the merged
  branches `opt1-integration` and `p3`–`p6-integration`, are deleted.
- **Push:** `main` is pushed to origin at f713b56.

## Notes

- The machine was not quiet: Rider was running (20–35 % CPU before the run). Pass 2 was on
  average 1.16× pass 1.
- About 10 memory-bound cases per thread count are 1.5–1.9× their old values (`Split`,
  `Compare`, `InRange`, `Trunc` F32, `BgrToBgra`, `LabToBgr` F32, `ResizeDownCubic` F32). Treat
  these as load error until a run on a quiet machine.
- The 8 MB LOH and 512-bit views of the page still use runs from before the fix.
- Open: a rerun on a quiet machine with the default, 8 MB LOH (alternated per class) and
  512-bit Morphology runs. The script is `C:/tmp/q/run.sh`.
- Untracked folders `BenchmarkDotNet.Artifacts-loh8` and `-mt-loh8` remain in the benchmark
  project.
