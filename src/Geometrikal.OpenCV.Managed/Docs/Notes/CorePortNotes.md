# Core operations port notes

Semantics of the `Core` operations (`Arithmetic`, `Bitwise`, `Comparison`, `Conversions`,
`Reductions`, masked `Mat.CopyTo`/`Mat.SetTo`), the OpenCV 5.0 decisions, and the golden
constraints. Read this before you change a core kernel or the core golden generator.

## Source version policy

- The port follows the OpenCV 5.0 source (`src/OpenCV`).
- Some OpenCV 5.0 kernels give results that depend on the element position: the vector body
  and the scalar tail use different formulas. In that case the port uses the vector-body
  formula for all elements.
- The golden reference is cv2 4.5.5. Where 4.5.5 and 5.0 differ, the golden battery avoids the
  input range, or the case is a suspect case. `CoreOpenCv5DifferenceTests` pins the 5.0 behaviour.

## Element conversion

- `Core/Internal/Saturate` implements `saturate_cast` for all depths.
- Float-to-integer conversion uses `cvRound`: round half to even. NaN and values outside the
  `int` range become `int.MinValue`, then clamp. Example: 3e9 or NaN to `byte` gives 0, not 255.

## Arithmetic (OpenCV 5.0 formulas)

| Operation | 8/16-bit | `int` | `float` | `double` |
|---|---|---|---|---|
| add, subtract | saturate | wrap | float | double |
| absdiff | saturate `\|a-b\|` | `abs` of wrapped difference (`int.MinValue` stays) | float | double |
| multiply | float `(a*fs)*b` | double `(a*b)*s` | float `(a*fs)*b` | double `(a*s)*b` |
| divide | float `(a*fs)/b`, 0 if `b == 0` | double, 0 if `b == 0` | float, IEEE | double, IEEE |
| reciprocal | float `fs/b`, 0 if `b == 0` | double, 0 if `b == 0` | float | double |
| addWeighted | float `a*w1 + (b*w2 + d)` | double `(a*w1 + b*w2) + d` | float, as 8/16-bit | double, as 8/16-bit |

- `fs` is `(float)scale`. The JIT does not contract multiply-add, so these orders are exact.
- 16U multiply with a product above `int.MaxValue` gives 0, as the 5.0 vector code
  (`v_round` then `v_pack_u`).
- Scalar operands:
  - add, subtract and absdiff on 8/16-bit types round the scalar to `int` first
    (`add(x, 2.5)` adds 2).
  - If the rounded scalar is outside the element range, the operation runs in 32-bit `int`
    with wrap, then saturates (`IntWorkScalarOp`).
  - multiply and divide with a scalar compute in double for every depth. Division by a zero
    scalar is not special-cased.
- Masks: the result is computed for the full row, then stored where the mask is nonzero.

## Comparison

- `Compare(a, value, op)` takes one value. OpenCV uses only `s[0]` for all channels.
- Integer sources: an out-of-range value gives a constant result. A fractional value uses
  `ceil` for `Lt`/`Ge` and `floor` for `Le`/`Gt`.
- `InRange` with scalar bounds rounds the bounds half to even. An empty or out-of-range
  8/16-bit interval selects nothing.

## Conversion

- `ConvertTo` uses OpenCV's work-type table:
  - double work for any source to `double`, for `int` to `int`, and for `double` to `int` or
    `float`;
  - float work for all other pairs.
- A `double` source with float work converts each element to float first (the vector-body form).
- `ConvertScaleAbs` always computes in float.

## Reductions

- Integer sources accumulate exactly.
- Floating sources accumulate in double per row; the rows are added in row order. The result
  does not depend on the thread count. The last bits can differ from OpenCV, whose vector code
  uses a different summation order.
- `Norm` of a `float` difference without a mask squares the float `|a - b|` (5.0 vector body).
  With a mask it subtracts in double (5.0 masked loop).
- `Norm` for `int`:
  - `Inf` reads `abs(int.MinValue)` as 2147483648.
  - The difference wraps in 32 bits for `Inf` and `L1`.
- `MinMaxLoc` skips NaN and keeps the first occurrence. OpenCV 5.0 starts from the first element,
  so a leading NaN propagates there, and its float vector path depends on the lane. The port
  keeps the 4.5.5 rule, which does not depend on the element position.
- `Reduce`:
  - 16-bit sources to `float` sum exactly, then convert once (5.0).
  - `float` to `float` sums in float in index order.
  - `Sum2` is new in 5.0.

## Golden constraints (`tools/generate_core_golden.py`)

- Environment: `OPENCV_CPU_DISABLE=AVX,AVX2,AVX512F,FMA3`, IPP, OpenCL and threads disabled.
- Pass scalars as 4-tuples. A bare Python number changes channel 0 only.
- Avoid these 4.5.5 versus 5.0 differences:
  - `int` multiply overflow;
  - `int` divide operands above 2^12;
  - `int` absdiff above `int.MaxValue`;
  - `int.MinValue` in norms;
  - 16-bit `Reduce` sums above 2^24.
- Avoid these 4.5.5 position-dependent results:
  - `AddWeighted` with weights that are not dyadic;
  - NaN in `Min`/`Max`;
  - 16U products at or above 2^31;
  - 64F sources of scaled conversions that are not representable in float.
- Suspect cases: `multiply_f32_*` with scale 0.1 and 1/3. 4.5.5 computes these in double;
  5.0 computes them in float.
- Floating reductions are compared with a relative tolerance.
