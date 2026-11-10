# Project setup, porting docs, and managed ports of findContours, drawing, and morphology

**Date:** 2026-07-28 09:30:00

## Summary

Established the PerceptionAg.CV.OpenCV project foundations (docs, licensing,
test infrastructure) and ported three OpenCV subsystems to pure managed C#:
contour detection (`findContours`), the non-anti-aliased drawing functions,
and the morphological operations with their border-handling foundation.
All ports are validated byte-for-byte against cv2-generated golden data
(733 tests, all green, zero build warnings).

## Changes

### Project setup and documentation
- Reviewed and fixed the porting instructions: repaired the broken
  `CLAUDE.md` link, removed all findContours-specific content from
  `Docs/Instructions/Porting.md`, corrected greenfield-state assumptions,
  standardized tests on xUnit, and documented build/test commands.
- Created the xUnit test project (`PerceptionAg.CV.OpenCV.Tests`) and added
  it to `PerceptionAg.CV.slnx`.
- Removed the `SixLabors.ImageSharp` 4.0.0 dependency (v4 enforces a paid
  build-time licence key and the project had never built); the library now
  has zero external dependencies. Docs updated to match.
- Added root `README.md`, Apache-2.0 `LICENSE`, and `NOTICE` with OpenCV
  attribution (the OpenCV 5.0.0 source tree is a git submodule at
  `src/OpenCV`, pinned at the release tag).

### findContours port (`ImgProc/Contours/`)
- Full Suzuki–Abe port of OpenCV 5's `contours_new.cpp`: retrieval modes
  External/List/ConnectedComponents/Tree, approximation None/Simple/
  TC89_L1/TC89_KCos, hierarchy output, coordinate offsets, arbitrary
  stride/ROI inputs, non-mutating.
- New `Core` types: `Point`, `Rect`.
- 332 cv2 golden cases plus ports of OpenCV's own analytic tests. Two
  adjudicated cv2-4.5.5-vs-OpenCV-5 TC89_L1 differences pinned with
  documented root cause (OpenCV 5 `calc_new_start` push_back path).

### Drawing port (`ImgProc/Drawing/`)
- Non-anti-aliased subset of `drawing.cpp`: `Line`, `Rectangle` (both
  overloads), `Circle`, `Ellipse`, `FillConvexPoly`, `FillPoly`,
  `Polylines`, `DrawContours`; generic over byte/sbyte/ushort/short/int/
  float/double, 1–4 channels, sub-pixel `shift`, OpenCV clipping semantics.
  LINE_AA, putText, drawMarker, and arrowedLine deliberately unsupported.
- New `Core` types: `Scalar`, `Size`; colour saturation reproduces OpenCV's
  x86-64 `cvRound` sentinel semantics (verified against cv2).
- 83 golden cases; five adjudicated version differences (ThickLine pre-clip
  block and the LINE_4/LINE_8 thin-line condition swap, both upstream
  OpenCV 5 changes) pinned in `DrawingOpenCv5DifferenceTests`.

### Morphology port (`ImgProc/Morphology/` + `Core/Borders.cs`)
- `Erode`, `Dilate`, `MorphologyEx` (Open/Close/Gradient/TopHat/BlackHat/
  HitMiss), `GetStructuringElement` (Rect/Cross/Ellipse/Diamond) with
  OpenCV's anchor, iteration-folding, and default-border-value semantics.
- Public border foundation: `BorderType`, `Borders.BorderInterpolate`,
  `Borders.CopyMakeBorder` (reusable by future filter ports).
- `ScalarConverter` moved to `Core/Internal` so drawing and borders share
  the same saturation semantics.
- 181 golden cases, all matching cv2 byte-for-byte on the first run;
  BORDER_WRAP rejected for morphology exactly as OpenCV's FilterEngine does.

## Notes

- Both pushed commits: `30ec47b` (setup + contours + drawing) and `30f53d6`
  (morphology + borders).
- Golden data comes from cv2 4.5.5; regeneration scripts live in
  `PerceptionAg.CV.OpenCV.Tests/tools/`. The suspect/adjudication mechanism
  documents every known cv2-vs-OpenCV-5 behavioural difference; the managed
  ports follow OpenCV 5 by design.
- Deliberate API differences from OpenCV are documented in the XML docs:
  non-mutating return-new morphology, isolated-ROI border semantics,
  `Mat<sbyte>` structuring elements, strict argument validation.
- The org's monthly agent-usage credits ran out mid-session; the drawing
  review was completed inline and the morphology port was done fully inline
  as a result.

---

# Filtering subsystem port (GaussianBlur, box, median, filter2D, sepFilter2D, derivatives)

**Date:** 2026-07-28 22:09:51

## Summary

Ported OpenCV's image filtering subsystem to pure managed C#: `GaussianBlur`
(including the bit-exact fixed-point 8-bit path and the softdouble kernel
computation), `blur`/`boxFilter`, `medianBlur`, `filter2D`, `sepFilter2D`,
`getGaussianKernel`, and the derivative filters `Sobel`, `Scharr`,
`Laplacian`, `getDerivKernels`. All ports are validated byte-for-byte
against cv2 4.5.5 golden data with an empty suspect set (914 tests total,
all green, zero build warnings).

## Changes

### Filtering port (`ImgProc/Filtering/`)
- Public API: `Filters` (GaussianBlur, Blur, BoxFilter, MedianBlur,
  Filter2D, SepFilter2D, GetGaussianKernel) and `Derivatives` (Sobel,
  Scharr, Laplacian, GetDerivKernels). Sources are `byte`/`float32` with
  1–4 channels; linear-filter destinations follow OpenCV's ddepth rules as
  `byte→byte/short/float` and `float→float` type parameters.
- OpenCV's streaming `FilterEngine` replaced by padded working images
  (morphology precedent), with OpenCV's exact scalar arithmetic:
  - `SepFilterEngine`: replicates `createSeparableLinearFilter`'s bit-exact
    mode selection (8U→8U smooth-symmetric → int32/Q8.8·Q8.8 with
    round-half-up shift; 8U→16S integer (anti)symmetric → pure int32) and
    the float paths' branch structure (`SymmRowSmallFilter` /
    `SymmColumnSmallFilter` special-case associations) operation-for-operation.
  - `GaussianFixedPoint`: the bit-exact 8-bit Gaussian — exact Q8.8/Q16.16
    two-pass with a single final rounding; kernels from an exact managed
    transcription of OpenCV's softdouble `exp` (`SoftDoubleMath`, verified
    bit-for-bit against `cv2.getGaussianKernel` across sigmas) plus the
    error-diffusion fixed-point quantization.
  - `BoxFilterEngine`: RowSum/ColumnSum running sums with OpenCV's sum-type
    ladder (ushort ≤ area 256, int32, double for float sources); OpenCV 5's
    new `blockSum` path was proven arithmetic-identical, so one
    implementation matches both 4.5.5 and 5.0.
  - `MedianEngine`: OpenCV's 9- and 25-element sorting networks with exact
    `std::min/max` comparison semantics (NaN-sensitive for float), plus a
    two-tier-histogram sliding window for 8-bit ksize ≥ 7 (medians are
    value-determined, so any correct histogram is output-identical).
  - `Filter2DEngine`: direct convolution over the nonzero-coefficient list;
    OpenCV's DFT path for kernel areas ≥ 50 is deliberately not used
    (documented — direct convolution is exact).
  - Kernel builders: `getKernelType`, `createBitExactKernel_32S`,
    `getSobelKernels`/`getScharrKernels`, `getGaussianKernel` (softdouble).

### Tests (181 new)
- 159 cv2 golden cases (`generate_filters_golden.py` →
  `FiltersGoldenTests`), all passing byte-for-byte with an empty suspect
  set. The generator disables IPP/OpenCL and the AVX/FMA CPU dispatch and
  constrains the battery to avoid corners where OpenCV itself is
  lane-position-dependent (documented in
  `Docs/Notes/FilteringPortNotes.md`): exact-half outputs in the 8U→8U
  bit-exact separable path, even-area normalized box kernels,
  integer-special float kernels on non-integer data, and nonzero delta on
  ksize-3 float symmetric columns.
- `FiltersBasicTests`: ROI/stride equivalence for every entry point,
  argument validation (including BORDER_WRAP rejection and unsupported type
  combinations), non-mutation, determinism, and analytic pins for the
  documented corners (even-kernel normalized blur rounds half-to-even in
  double precision; single-pixel Gaussian border quirk; 1-D median clamping).

### Documentation
- `Docs/Notes/FilteringPortNotes.md`: the full bit-exactness study — path
  selection tables, the four OpenCV-internal scalar-vs-SIMD divergences
  found (including two discovered empirically via golden failures), and the
  golden-generator environment requirements.
- README status table, NOTICE ported-sources list, Porting.md project
  state, and the test tools README updated.

## Notes

- Discovered during validation: `boxFilter`'s 1-pixel-dimension kernel
  collapse compares the border type *without* stripping `BORDER_ISOLATED`,
  so for isolated matrices it applies to constant borders too (GaussianBlur
  strips the bit and genuinely skips constant) — the port and its tests
  follow the verified cv2 behaviour.
- Deliberately unsupported (validated and rejected): `BORDER_WRAP`,
  16-bit/64-bit image depths, `sqrBoxFilter`, `bilateralFilter`, median
  apertures > 5 for float or 2-channel images, and the DFT `filter2D` path.
- The port follows OpenCV's *scalar* reference arithmetic wherever OpenCV's
  own SIMD lanes and scalar tail disagree (all such cases are 1-LSB corner
  divergences inside OpenCV itself; each is documented in the notes file).
