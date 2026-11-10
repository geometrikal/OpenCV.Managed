# Thresholding port notes

Semantics of `Thresholding.Threshold` (OpenCV `threshold` and `thresholdWithMask`) and
`Thresholding.AdaptiveThreshold` (OpenCV `adaptiveThreshold`), the deliberate differences from
OpenCV 5.0, and the golden constraints. Read this before you change a threshold kernel or the
threshold golden generator.

## Sources

- Port source: OpenCV 5.0, `modules/imgproc/src/thresh.cpp`.
- Golden reference: cv2 4.5.5.
- Files:
  - `ImgProc/Threshold/Thresholding.cs`: public API, flag decoding, integer conversion,
    out-of-range shortcut, adaptive threshold.
  - `ImgProc/Threshold/Internal/ThresholdKernels.cs`: vector row kernels, masked rows,
    adaptive comparison.
  - `ImgProc/Threshold/Internal/AutoThreshold.cs`: Otsu and triangle threshold selection.
  - `Core/Internal/ScalarConverter.cs`: `CvFloor` and `CvCeil` with x86 semantics.

## API

| Method | OpenCV equivalent | Result |
|---|---|---|
| `Threshold<T>(src, thresh, maxValue, type)` | `threshold` | New matrix |
| `Threshold<T>(src, thresh, maxValue, type, out usedThreshold)` | `threshold` | New matrix and applied threshold |
| `Threshold<T>(src, dst, thresh, maxValue, type, mask = null)` | `threshold`, `thresholdWithMask` | Writes `dst`; returns applied threshold |
| `AdaptiveThreshold(src, maxValue, method, type, blockSize, c)` | `adaptiveThreshold` | New matrix |

- `ThresholdType` and `AdaptiveThresholdType` use the OpenCV numeric values.
- `dst` may be the same matrix as `src`. Partially overlapping views are not supported.
- `dst` may be null only with `ThresholdType.DryRun`.
- The overloads that return a new matrix reject `DryRun`.

## Element rules

| Rule | Integer depths | Floating depths (all elements) |
|---|---|---|
| `Binary` | `src > t ? maxval : 0` | `(t < src) & maxval` |
| `BinaryInv` | `src <= t ? maxval : 0` | `(src <= t) & maxval` |
| `Trunc` | `min(src, t)` | `src < t ? src : t` |
| `ToZero` | `src > t ? src : 0` | `(t < src) & src` |
| `ToZeroInv` | `src <= t ? src : 0` | `(src <= t) & src` |

- `&` is a bitwise AND with an all-ones or all-zeros comparison mask.
- A false floating comparison gives +0.0. A true comparison keeps the exact bits of `src` or
  `maxval` (−0.0, NaN payload and infinity stay).
- A NaN source fails every comparison. `Trunc` gives `t` for a NaN source.
- A NaN threshold fails every comparison. `Trunc` gives NaN for every element.
- These are the OpenCV 5.0 vector-body rules. The port applies them to every element.
- The rule applies per element. Channels are not distinguished. One threshold applies to all
  channels.

## Parameter conversion

### Integer depths (`byte`, `short`, `ushort`)

1. `ithresh = cvFloor(thresh)`. The method returns `ithresh`.
2. `imaxval = saturate(cvRound(maxValue))`. For `Trunc`, `imaxval = saturate(ithresh)`; the
   value is not used.
3. If `ithresh < min` or `ithresh >= max` of the element type, the shortcut applies:

   | Rule | `ithresh < min` | `ithresh >= max` |
   |---|---|---|
   | `Binary` | fill `imaxval` | fill 0 |
   | `BinaryInv` | fill 0 | fill `imaxval` |
   | `Trunc` | fill 0 | copy `src` |
   | `ToZero` | copy `src` | fill 0 |
   | `ToZeroInv` | fill 0 | copy `src` |

4. Else the kernel runs with `(T)ithresh` and `(T)imaxval`.

`cvFloor`, `cvCeil` and `cvRound` use x86 semantics, not .NET conversions:

| Input | `cvFloor` | `cvCeil` | `cvRound` |
|---|---|---|---|
| NaN | `int.MinValue` | `int.MinValue` | `int.MinValue` |
| +inf, values ≥ 2^31 | `int.MinValue` | `int.MinValue + 1` | `int.MinValue` |
| −inf, values < −2^31 | `int.MaxValue` (wrap) | `int.MinValue` | `int.MinValue` |

Examples for `byte`: a NaN or +inf threshold gives `ithresh = int.MinValue`; `Binary` then fills
`imaxval`. A −inf threshold gives `int.MaxValue`; `Binary` then fills 0. A NaN `maxValue` gives 0.

### Floating depths (`float`, `double`)

- `float` uses `(float)thresh` and `(float)maxValue` (round to nearest; 1e300 becomes +inf).
- `double` uses the values unchanged.
- No shortcut applies. The method returns `thresh` unchanged.

## Automatic thresholds

- `Otsu`: single-channel `byte` or `ushort`. The histogram has 256 or 65536 bins.
- `Triangle`: single-channel `byte`.
- Other sources throw `ArgumentException`. `Otsu | Triangle` throws `ArgumentException`.
- The computed threshold replaces `thresh`. The integer conversion then applies.
- The Otsu loop is a literal transcription in `double`, in source order, without fused
  multiply-add. `mu1 *= q1` runs before the `FLT_EPSILON` test. For images above about 8.4 million
  pixels with rare low values, this changes the result. The golden case
  `otsu_u8_synthetic_3001x3002` covers this.
- A constant image gives Otsu threshold 0.
- Triangle results range from −1 to 256. Both extremes take the shortcut.
- With a mask, the histograms count only pixels where the mask is nonzero. An all-zero mask gives
  Otsu threshold 0 and triangle threshold 2.

## Mask (OpenCV 5.0 `thresholdWithMask`)

- The mask is a single-channel `byte` matrix with the size of `src`.
- Where the mask is zero, `dst` keeps its values.
- Masked floating `Trunc` uses `std::min(src, t)` = `t < src ? t : src`, as the OpenCV 5.0
  masked kernel. It differs from the unmasked rule for a NaN source (result NaN) and for zeros of
  opposite sign (result `src`).

## DryRun (OpenCV 5.0 `THRESH_DRYRUN`)

- The method computes the threshold (including Otsu and triangle) and returns it.
- The method does not write `dst`.
- Integer depths return `cvFloor` of the threshold.

## Adaptive threshold

1. `src` must be single-channel. `blockSize` must be odd and at least 3.
2. If `maxValue < 0`, the method returns an all-zero matrix. This check runs before the method
   and type checks, as OpenCV. −0.0 and NaN do not take this branch.
3. Mean image:
   - `MeanC`: `Filters.Blur<byte>` with `BorderType.Replicate`.
   - `GaussianC`: conversion to `float`, `Filters.GaussianBlur(Mat<float>)` with sigma 0 and
     `BorderType.Replicate`, conversion to `byte` (round half to even). The 8-bit fixed-point
     Gaussian is not used; it gives other results.
4. `imaxval = saturate_cast<byte>(maxValue)`.
5. `idelta = cvCeil(c)` for `Binary`, `cvFloor(c)` for `BinaryInv`. `k = -idelta` with wrap.
6. `Binary`: `dst = (src - mean > k) ? imaxval : 0`. `BinaryInv`:
   `dst = (src - mean <= k) ? imaxval : 0`.
- Other methods or types throw `ArgumentException`.
- The port computes the comparison directly. OpenCV uses an equivalent 768-entry table.

## Deliberate differences from OpenCV 5.0

| Area | OpenCV 5.0 | Port | Reason |
|---|---|---|---|
| Mask with out-of-range threshold | Fills or copies the whole `dst`; ignores the mask | Fills or copies only where the mask is nonzero | The 5.0 behaviour contradicts the mask contract. It is an upstream defect. |
| Floating `Trunc`, scalar tail | `std::min` in the last `width mod (2 × lanes)` elements | Vector rule for every element | Output must not depend on the element position. |
| Rule values 5, 6, 7 | Depth-dependent: zeros, error, or unwritten `dst` | `ArgumentException` | Undefined in OpenCV. |
| Unknown flag bits (for example 32) | Ignored | `ArgumentException` | The enum defines the valid bits. |
| `int`, `sbyte` and half-float depths | Error | `NotSupportedException` | Same outcome, .NET exception type. |
| Region-of-interest borders in adaptive threshold | `BORDER_ISOLATED` | Always isolated | Same outcome. |

## Performance

- The unmasked kernels use `Vector<T>` for all five depths. The rule is a generic struct
  (`IThresholdOp<T>`).
- The final partial vector overlaps the previous vector. The kernel computes it before the
  first store, so an in-place row reads only source values.
- Rows shorter than one vector use a stack buffer.
- Masked rows are computed into a pooled row, then stored with `Elementwise.StoreMasked`.
- Row loops run through `ParallelRows.For` with static stripe methods. The kernels have no running
  state, so output does not depend on the thread count.
- The work estimate divides the row length by `Vector<T>.Count`. Images split into stripes only
  above several million elements. `CvParallelTests.LargeImage_Output_DoesNotDependOnThreadCount`
  uses a 4099 × 2049 image for this reason.
- The Otsu and triangle histograms are single-threaded.

## Golden constraints

- Generator: `Geometrikal.OpenCV.Managed.Tests/tools/generate_threshold_golden.py`.
- Environment: `OPENCV_CPU_DISABLE=AVX,AVX2,AVX512F,FMA3` before `import cv2`; IPP, OpenCL and
  threads disabled.
  - The 32F Gaussian in adaptive threshold depends on FMA. Exact .5 means occur.
  - The IPP 32F `Trunc`, `ToZero` and `ToZeroInv` paths give other NaN results.
- `Trunc` cases with a zero threshold give every zero source the sign of the threshold. This
  avoids the 4.5.5 scalar-tail difference for signed zero.
- Floating parameters are written as JSON numbers or as `"NaN"`, `"Infinity"`, `"-Infinity"`.
  The runner must pass NaN as the positive quiet NaN (`0x7FF8000000000000`). `double.NaN` in .NET
  has the sign bit set, and the kernels keep NaN bits.
- The large Otsu case stores no pixels. The generator and the runner build the image from the same
  formula and compare only the threshold.
- The suspect set is empty. No cv2 4.5.5 versus OpenCV 5.0 difference is known for the features in
  the battery.
- cv2 4.5.5 has no `thresholdWithMask` and no `THRESH_DRYRUN`. `ThresholdBasicTests` covers them
  with analytic expectations:
  - masked result equals `mask ? threshold(src) : dst` (upstream `test_thresh.cpp` identity);
  - Otsu and triangle with a rectangular mask equal the unmasked result of the rectangle;
  - out-of-range shortcut with a mask keeps unmasked pixels;
  - masked floating `Trunc` uses `std::min`;
  - all-zero mask gives Otsu 0 and triangle 2;
  - `DryRun` returns the threshold and does not write `dst`.
