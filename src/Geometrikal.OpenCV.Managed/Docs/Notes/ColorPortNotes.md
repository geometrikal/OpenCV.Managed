# Colour conversion port notes

Semantics of `ColorConversions.CvtColor`, the bit-exactness rules and the golden constraints.
Read this before you change a colour kernel, a Lab table or the colour golden generator.

## Scope

| Family | Codes | Element types |
|---|---|---|
| Reorder, alpha | `Bgr2Bgra` … `Bgra2Rgba` | `byte`, `ushort`, `float` |
| Gray | `Bgr2Gray`, `Rgb2Gray`, `Bgra2Gray`, `Rgba2Gray`, `Gray2Bgr`, `Gray2Bgra` | `byte`, `ushort`, `float` |
| HSV | `Bgr2Hsv`, `Rgb2Hsv`, `*Full`, `Hsv2Bgr`, `Hsv2Rgb`, `Hsv2*Full` | `byte`, `float` |
| Lab | `Bgr2Lab`, `Rgb2Lab`, `Lbgr2Lab`, `Lrgb2Lab` and inverses | `byte`, `float` |
| XYZ | `Bgr2Xyz`, `Rgb2Xyz` and inverses | `byte`, `ushort`, `float` |
| YCrCb | `Bgr2YCrCb`, `Rgb2YCrCb` and inverses | `byte`, `ushort`, `float` |

The port does not include Luv, HLS, YUV 4:2:x or Bayer codes.

## Vector-body and scalar-tail split

- OpenCV processes each row with a vector body and a scalar tail. In some kernels the two parts
  use different float formulas.
- The port applies the same split per row. The first `width / B * B` pixels use the vector
  formula; the remaining pixels use the scalar formula.
- `B` is the block size of a 128-bit vector build (SSE baseline):

| Kernel | B | Difference |
|---|---|---|
| `HSV2RGB_b` (8U HSV to RGB) | 16 | vector body truncates `x * 255`; scalar tail rounds |
| Linear RGB to Lab, 32F | 8 | cube-root spline and `R*c0 + (G*c1 + B*c2)` against `cubeRoot` and `(R*c0 + G*c1) + B*c2` |
| Lab to RGB, 32F | 8 | reciprocal multiplies and `c0*x + (c1*y + c2*z)` against divisions and left-to-right sums |
| XYZ, 32F, both directions | 4 | `s0*c0 + (s1*c1 + s2*c2)` against `(s0*c0 + s1*c1) + s2*c2` |
| RGB to YCrCb, 32F | 4 | as XYZ, for Y |

- OpenCV builds with 256-bit vectors (AVX2) use larger blocks and fused multiply-add. For these
  kernels their results differ from this port.

## Vector bodies

- Each `IColorRow<T>` kernel first calls a `Vector128` body (`ColorVectorKernels`,
  `LabVectorKernels`) and then processes the remaining pixels with its scalar loop.
- A vector body uses, in each lane, the operations of the formula that applies to that pixel:
  the vector-body formula inside the split, otherwise the scalar formula. Kernels with a split
  pass the split point as the width of the vector body.
- `ColorVectors` deinterleaves and interleaves 3- and 4-channel data with byte shuffles. Without
  `pshufb` (SSSE3) or `tbl` (ARM64), every kernel uses its scalar loop.
- The exhaustive cube and width-sweep goldens test the vector bodies and the scalar loops.

## Fixed-point and table rules

- Gray 8U/16U: 15-bit weights 9798, 19235, 3735 (R, G, B): `(sum + 16384) >> 15`. Only YCrCb
  uses the 14-bit set (4899, 9617, 1868).
- HSV 8U: division tables `cvRound((255 << 12) / i)` and `cvRound((hrange << 12) / (6 i))`.
  The hue range is 180. For the `Full` codes it is 256 (forward) and 255 (inverse).
- Lab:
  - `LabTables` builds all tables once, with the float and double operations of OpenCV's
    softfloat code in the same order.
  - `SoftCbrt` is a port of softfloat `f32_cbrt`. It truncates the result mantissa. A
    correctly rounded cube root gives different tables.
  - `ApplyGamma` uses `Math.Pow`. The table hashes in `ColorTablesTests` confirm that the
    tables equal OpenCV's tables.
  - 8U Lab is pure fixed point. 32F sRGB to Lab uses the 33³ trilinear table on 14-bit
    quantized values. Its output is therefore quantized (L to 100/16384, a and b to 1/64).

## OpenCV 5.0 differences from cv2 4.5.5

- Float HSV to RGB with a negative hue:
  - 5.0 wraps the sector (`ComputeSectorAndClampedH`). 4.5.5 maps it to sector 0.
  - The port follows 5.0. `ColorBasicTests.HsvToBgr_NegativeFloatHue_WrapsSector` pins this.
- There is no other numeric difference for in-range inputs.

## Golden constraints (`tools/generate_color_golden.py`)

- Call `cv2.setUseOptimized(False)`. This disables IPP (which replaces 32F gray and 8U
  `HSV2BGR_FULL`) and the AVX2 dispatch.
- "cube" cases convert all 2^24 8-bit triples and store only the SHA-256 of the output.
- The width sweep (1 to 100) covers the blocks of 4, 8 and 16 and their tails.
- The battery excludes NaN, Inf and negative float hue.
