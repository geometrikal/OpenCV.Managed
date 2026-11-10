# Math functions port notes

Port of OpenCV 5.0 `magnitude`, `phase`, `cartToPolar`, `polarToCart`, `exp`, `log`, `pow` and
`sqrt` (`modules/core/src/mathfuncs.cpp`, `mathfuncs_core.simd.hpp`, `hal/intrin_math.hpp`).
Public API: `Core/MathFunctions`. Kernels: `Core/Internal/MathKernels`, `Core/Internal/FastMath`.

## Structure

- Planes follow `NAryMatIterator`: one plane when all matrices are continuous, else one per row.
- `cartToPolar` and `polarToCart` split each plane into blocks of `((1024 + cn − 1)/cn)·cn`
  elements. The 64F variants convert 128-element sub-blocks to `float`, run the 32F kernel and
  convert back, so 64F results have `float` precision (OpenCV 5.0; cv2 4.5.5 computed some 64F
  outputs in `double`).
- Vector blocks are two SSE registers (8 floats), as the SSE-baseline build that the golden data
  uses. A span of at least 8 elements is covered by vector blocks, the last one overlapping the
  previous; a shorter span uses the scalar code only.

## Kernels and bit-exactness

| Function | Vector form | Scalar form | Port |
|---|---|---|---|
| `magnitude` 32F/64F | `sqrt(x·x + y·y)` | same | identical; bit-exact |
| `phase`, `cartToPolar` angle | `v_atan_f32` | `atan_f32` | identical without FMA; bit-exact |
| `polarToCart` 32F | Cephes `v_sincos` on `angle·(float)π/180`, float magnitude product | 64-entry table, cubic correction, double magnitude product | both ported; differ in the last bits |
| `exp`, `log` | — | C runtime per element (MSVC build) | `MathF`/`Math` |
| `pow` integer power | `iPow_SIMD` | `iPow_i`/`iPow_f` | identical (wrapping working type, saturation); bit-exact |
| `pow(x, −0.5)` 32F | `rsqrtps` + one Newton step | `1/sqrt(x)` | `Sse.ReciprocalSqrtScalar` on x86; matches on the same CPU |
| `pow` real power | — | `exp(log(x)·p)`, then 0 → 0 or +∞, x < 0 → NaN `0x7fffffff` | ported |

`v_sincos` is a per-lane function, so `MathKernels.SinCosCephes` evaluates it per element with
the SSE semantics: `v_trunc` is `cvttps2dq` (`int.MinValue` for NaN and overflow) and every
`v_fma` is a multiply and an add.

`saturate_cast<T>(unsigned)` is `min(v, max)`; the integer power of unsigned elements saturates
with that rule, not through `cvRound` (which overflows above 2³¹).

## Deliberate differences and limits

- `pow` of integer elements with a non-integer power throws `NotSupportedException` (OpenCV
  reinterprets the buffer as floating point).
- `pow(x, 2)` uses `Arithmetic.Multiply`; 16U products of 2³¹ and more follow its documented
  behaviour.
- Results of `polarToCart` and `pow(x, −0.5)` depend on the block structure (vector or scalar
  form), as in OpenCV; an ROI (row planes) can differ from a packed copy in the last bits.

## Golden data

`tools/generate_mathfuncs_golden.py` → `TestData/mathfuncs_golden.json` (288 cases),
`Core/MathFunctionsGoldenTests`. 55 cases are adjudicated cv2 4.5.5 versus OpenCV 5.0
differences, pinned in `Core/MathFunctionsOpenCv5DifferenceTests`:

- 64F `cartToPolar` and `polarToCart`: 4.5.5 computes in `double`; 5.0 through `float`.
- 64F `polarToCart` without magnitudes: 4.5.5 returns uninitialised memory.
- 32F `polarToCart`: 4.5.5 uses the table for every element and multiplies the magnitude in
  `float` (a Python model of this reproduces all 4.5.5 outputs); 5.0 uses `v_sincos` for vector
  blocks and a `double` product.
- `exp`, `log`, real `pow`: the cv2 4.5.5 wheel uses an approximate vector routine (its values
  are not correctly rounded); 5.0 for MSVC calls the C runtime per element. Differences are a
  few ulps.

The 16U `pow(x, 2)` battery keeps products below 2³¹ (see above).
