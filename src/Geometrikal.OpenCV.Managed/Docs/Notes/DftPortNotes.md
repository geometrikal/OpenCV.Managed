# DFT and DCT port notes

Semantics of `SpectralTransforms` (OpenCV `dft`, `idft`, `dct`, `idct`, `mulSpectrums`,
`getOptimalDFTSize`), the bit-exactness rules, the parallel batch design, and the golden
constraints. Read this before you change a spectral kernel, the batch driver, or the DFT golden
generator.

## Sources

- Port source: OpenCV 5.0, `modules/core/src/dxt.cpp`.
- Golden reference: cv2 4.5.5 with IPP disabled. The non-IPP CPU code of `dxt.cpp` is the same
  in 4.5.5 and 5.0.
- Files:
  - `Core/SpectralTransforms.cs`: public API, validation, output type.
  - `Core/DftFlags.cs`: `DftFlags` and `DctFlags` with the OpenCV values.
  - `Core/Internal/DftPlan.cs`: `DFTFactorize`, `DFTInit`, `DCTInit`, the copied tables
    (`bitrevTab`, `DFTTab`, `DctScale`), and the plan caches.
  - `Core/Internal/DftKernels.cs`: `DFT` (radix 4, 2, 3, 5 and generic odd radix),
    `RealDFT`, `CCSIDFT`, `ExpandCCS`, `DCT`, `IDCT`.
  - `Core/Internal/DftLanes.cs`: scalar and `Vector<T>` element arithmetic of the kernels.
  - `Core/Internal/DftEngine.cs`: `OcvDftImpl` and `OcvDctImpl` drivers (stages, row pass,
    column pass, `complementComplex`), lane batches and parallel stripes.
  - `Core/Internal/LaneTranspose.cs`: AVX transposes between matrix rows and lane-major buffers.
  - `Core/Internal/SpectrumKernels.cs`: `mulSpectrums`.

## API

| Method | OpenCV equivalent | Element types |
|---|---|---|
| `Dft<T>(src, flags, nonzeroRows)` | `dft` | float, double; 1 or 2 channels |
| `Idft<T>(src, flags, nonzeroRows)` | `idft` | as `Dft` |
| `Dct<T>(src, flags)`, `Idct<T>` | `dct`, `idct` | float, double; 1 channel |
| `MulSpectrums<T>(a, b, flags, conjB)` | `mulSpectrums` | float, double; 1 or 2 channels |
| `GetOptimalDftSize(size)` | `getOptimalDFTSize` | int |

- Every method returns a new matrix. Sources are not modified.
- Output channels of `Dft`: forward with 1-channel input and `ComplexOutput` gives 2 channels.
  Inverse with 2-channel input and `RealOutput` gives 1 channel. Otherwise the output has the
  source channel count. `ComplexInput` only requires a 2-channel source.
- Inverse transforms are not scaled unless `Scale` is set.
- `MulSpectrums` uses only `DftFlags.Rows`.
- `divSpectrums` (`SpectralTransforms.DivSpectrums`): `a·conj(b) / (|b|² + ε)` (and `a·b / ...`
  with `conjB`, as OpenCV), complex pairs in double, real CCS elements `a / (b + ε)` in the element
  type; ε is `FLT_EPSILON` or `DBL_EPSILON`; same 1D rules as `mulSpectrums`. Golden data:
  `tools/generate_divspectrums_golden.py` (96 cases). The 48 double cases match directly. The 48
  float cases are suspects: commit e1dba2c6d2 (2024) computes the float row pairs in double, while
  4.5.5 used float products. An internal 4.5.5 mode reproduces all of them
  (`Core/DivSpectrumsGoldenTests`).
- `magnitude`, `phase`, `cartToPolar` and `polarToCart` are in `Core/MathFunctions` (see `MathFunctionsPortNotes.md`). Their 4.5.5 and 5.0
  implementations differ (fused 5.0 `cartToPolar` computes the 64F magnitude in float; 5.0
  `polarToCart` uses a different sine polynomial), and the SIMD bodies use FMA.

## Layouts

- CCS (complex-conjugate-symmetric) 1D row of n reals: `Re0, Re1, Im1, ..., Re(n/2)` for even
  n; `Re0, Re1, Im1, ..., Re((n-1)/2), Im((n-1)/2)` for odd n.
- CCS 2D (H×W): each row holds its 1D CCS. Column 0 and, for even W, column W-1 hold the 1D CCS
  of the column. Columns 2k-1 and 2k hold the real and imaginary parts of complex column
  spectra over all H rows.
- `ComplexOutput` writes the full spectrum. The upper half comes from conjugate symmetry
  (`complementComplex`).

## Dimensions and stages

- The DFT is 1D when `Rows` is set, when the source has one row, or when it has one column and
  source and destination are continuous. A non-continuous single column (an ROI) uses the
  column algorithm (`OneDimColWise`). Its forward CCS result is computed with a complex DFT of
  `(x, 0)` and differs in the last bits from the continuous case, as in OpenCV.
- 2D forward and 2D inverse complex-to-complex: rows first, then columns. 2D inverse with real
  output: columns first, then rows.
- The column pass always uses the complex kernel. Only the row pass of a real transform uses
  `RealDFT` or `CCSIDFT`.
- Scaling applies in the last stage only: `1/(len·count)`, or `1/len` with `Rows`. Odd-length
  `RealDFT` scales the input; even-length `RealDFT` scales during post-processing.
- `nonzeroRows`: the row pass transforms the first `nonzeroRows` rows and sets the other rows to
  zero. A single-column source with positive `nonzeroRows` throws `NotSupportedException`.
- DCT: 1D when `Rows` is set, when the source has one row, or when it has one column and is
  continuous; otherwise rows, then columns. Every transformed length must be even or 1.

## Bit-exactness rules

- The kernels are the scalar `dxt.cpp` code. The SSE3 float kernels of OpenCV have the same
  per-element IEEE operation sequence, so one implementation serves float and double.
- Keep the operand order and association of every expression. Never use `FusedMultiplyAdd`.
  The JIT does not contract `a*b + c`.
- Twiddles: power-of-2 lengths start from the copied `DFTTab`; other lengths use one
  `Math.Sin(-2π/n)` and one `Math.Sqrt`. The recurrence runs in double; values are rounded to
  T when they are stored. For n ≤ 5 (except 4) OpenCV does not compute the twiddles; the kernels
  do not read them.
- DCT twiddles start the recurrence from the scale rounded to T.
- Unary minus is a sign flip. `VectorLanes.Neg` uses XOR with −0.0, so zero signs match C++.
- The permutation is always applied as an out-of-place gather, and `CCSIDFT` always uses its
  out-of-place scatter form. OpenCV uses the in-place forms only when the first and last factor
  are equal; the permutation is then an involution, and both forms give the same values.
- `mulSpectrums` multiplies complex pairs in double and rounds to T. The AVX path computes
  `addsub(ar·b, ai·swap(b))`; each lane has the scalar rounding sequence. Real CCS elements are
  multiplied in T.
- On Windows `Math.Sin` calls the UCRT `sin`, which is the library of cv2. On other platforms
  the C library `sin` can differ in the last bit for rare arguments; the golden tests detect
  this.

## Performance design

- Lane batches: `DftKernels<T, TV, TL>` runs with `TV = T` (one transform) or
  `TV = Vector<T>` (8 float or 4 double transforms in the lanes). The lanes hold independent
  rows or columns of the same length, so every lane executes the scalar operation sequence,
  including the special cases at j = 0, the generic odd radix and the post-processing. The
  vector path and the scalar path give identical bits.
- Row pass: `LaneTranspose.RowsToLanes` transposes 8×8 float or 4×4 double blocks into a
  lane-major buffer; `LanesToRows` writes the results back.
- Column pass: lanes are adjacent columns. Complex columns are split into real and imaginary
  lanes with AVX2 shuffles. One visit of a row reads or writes about 256 contiguous bytes
  (`ColumnGroup`). Column 0 and the last column of a real transform form a separate batch.
- A single line (1D row, continuous column vector) uses the scalar kernels.
- `ParallelRows.For` runs batch groups in parallel. Batches are fixed by index; the stripe split
  does not change any element. `complementComplex` and `mulSpectrums` rows also run in parallel
  on large matrices.
- Plans (factors, permutation, twiddles) and DCT twiddles are cached per length in
  `ConcurrentDictionary` instances. A cache is cleared when it holds 128 (DFT) or 64 (DCT)
  entries.
- Temporaries come from `ArrayPool<TV>`. Output matrices are allocated without clearing,
  except for the zero rows of `Rows | ComplexOutput` with `nonzeroRows`.
- A prime length n costs O(n²), as in OpenCV (no Bluestein algorithm).

## Deliberate differences

- `Rows | ComplexOutput` with `nonzeroRows`: rows after `nonzeroRows` are zero in the port.
  OpenCV leaves the upper half of those rows uninitialized.
- In-place calls do not exist: the API always returns a new matrix. OpenCV in-place calls give
  the same values.

## Golden constraints

- Generator: `Geometrikal.OpenCV.Managed.Tests/tools/generate_dft_golden.py`, data
  `TestData/dft_golden.json`, runner `Core/DftGoldenTests.cs`. Comparison is byte-exact.
- IPP must be disabled. With IPP, 32F 1D DFTs with `n·count ≥ 64`, 2D DFTs with `W·H > 64`,
  64F 1D DFTs and the 32F DCT use IPP algorithms with other results.
- `OPENCV_CPU_DISABLE` has no effect on `dxt.cpp`; the generator sets it for consistency. One
  thread.
- The battery covers radix 2, 4, 3, 5, generic odd factors (7, 11, 13, 17 and products such as
  77, 121, 169, 289, 1155; prime 257), lengths above 256 (32-bit bit reversal), all six DFT
  modes with and without scale, `nonzeroRows`, `Rows`, continuous and non-continuous columns,
  DCT and IDCT in 1D, rows and 2D, `mulSpectrums` (CCS and complex, rows, `conjB`) and
  `getOptimalDFTSize` edges.
- `Rows | ComplexOutput` with `nonzeroRows` is not generated (uninitialized cv2 output).
- The suspect set is empty.
