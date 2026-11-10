# Flood fill port notes

Semantics of `FloodFilling.FloodFill` (OpenCV `floodFill`), the OpenCV 5.0 dispatch, the
equality fast path, the deliberate differences from OpenCV 5.0, and the golden constraints.
Read this before you change a scan loop or the flood fill golden generator.

## Sources

- Port source: OpenCV 5.0, `modules/imgproc/src/floodfill.cpp`.
- Golden reference: cv2 5.0.0 (the version of the ported source).
- Files:
  - `ImgProc/Segmentation/FloodFill.cs`: public API, validation, value conversion, dispatch.
  - `ImgProc/Segmentation/FloodFillFlags.cs`: flags with the OpenCV bit layout.
  - `ImgProc/Segmentation/FloodFillResult.cs`: area and bounding rectangle.
  - `ImgProc/Segmentation/Internal/FloodFillScan.cs`: segment stack (`FFillSegment`,
    `ICV_PUSH`, `ICV_POP`), gradient fill (`floodFillGrad_CnIR`), equality fills
    (`floodFill_CnIR`).
  - `ImgProc/Segmentation/Internal/FloodFillDiffs.cs`: pixel tests (`Diff8uC1`, `Diff8uC3`,
    `DiffC1`, `DiffC3`) and the equality tests of the fast path.

## API

| Method | OpenCV equivalent |
|---|---|
| `FloodFill<T>(image, seedPoint, newValue, loDiff = 0, upDiff = 0, flags = Connectivity4)` | `floodFill(image, seedPoint, newVal, &rect, loDiff, upDiff, flags)` |
| `FloodFill<T>(image, mask, seedPoint, newValue, loDiff = 0, upDiff = 0, flags = Connectivity4)` | `floodFill(image, mask, seedPoint, newVal, &rect, loDiff, upDiff, flags)` |
| `MaskFillValue(byte)` | `flags \|= value << 8` |

- `T` is `byte`, `int` or `float`. The image has 1 or 3 channels.
- Both overloads modify the image in place, except with `FloodFillFlags.MaskOnly`.
- The mask overload modifies the mask in place. The mask is `(width + 2) x (height + 2)`.
- `FloodFillResult` holds `Area` (OpenCV return value) and `Rect` (OpenCV `rect` output).
- Flags: bits 0-7 connectivity (0 or 4, or 8), bits 8-15 mask fill value (0 selects 1),
  `FixedRange` = 1 << 16, `MaskOnly` = 1 << 17. OpenCV values are kept, so an OpenCV flag
  integer cast to `FloodFillFlags` has the same meaning.

## OpenCV 5.0 dispatch

- `cv::floodFill` creates a zero mask when the caller gives none. The test
  `is_simple = mask.empty() && ...` comes after this step, so `is_simple` is always false.
  OpenCV 5.0 always runs `floodFillGrad_CnIR`. `floodFill_CnIR` is unreachable.
- Consequence: when the seed pixel equals the new value, OpenCV still walks the region and
  returns its area. The image does not change.
- `cv::floodFill` sets the one-pixel frame of the mask to 1
  (`copyMakeBorder(..., BORDER_ISOLATED | BORDER_CONSTANT, 1)`). The caller's mask frame is
  overwritten. The port does the same.
- When the seed pixel is non-zero in the mask, the result is area 0 and rectangle (0, 0, 0, 0).
  The image does not change.

## Pixel tests

| Type | lo/up conversion | Test for pixel `a` against pixel `b` |
|---|---|---|
| 8U | `saturate_cast<uchar>(cvFloor(d))` | `(uint)(a - b + lo) <= lo + up` per channel |
| 32S | `cvFloor(d)` | `d = a - b` (wraps); `-lo <= d <= up` per channel; `-lo` wraps |
| 32F | `(float)d` | `d = a - b`; `-lo <= d <= up`; NaN fails; `inf - inf` is NaN |

- `newValue` is converted with `saturate_cast` (`ScalarConverter.SaturateCast`).
- Negative `loDiff` or `upDiff` throws only for the channels of the image (`i < cn`).

## Scan order

- `FloodSegment` keeps the 16-bit fields of `FFillSegment`. The stack is last-in first-out.
  The stack starts at `2 * max(width, height)` segments in a pooled array and doubles when it
  is full. OpenCV grows by 3/2. The capacity does not change the push or pop order.
- The three neighbour scans per popped segment use the OpenCV `data[3][3]` table.
- The gradient fill writes the new value to a segment after its neighbour scans. The pixel tests
  therefore read unfilled values, as in OpenCV.
- A neighbour row outside the image is skipped. OpenCV scans the mask frame row there and finds
  only non-zero values. The result is the same, and the port does not form references outside
  the image.
- The region is the set of pixels that the scans reach. The segment order does not change the
  region, the area or the rectangle. The port keeps the OpenCV order regardless.

## Equality fast path

- Conditions: no caller mask, no `MaskOnly`, `T` is `byte` or `int`, the converted lo/up
  values are 0 in every image channel, and the seed pixel differs from the new value.
- Under these conditions the gradient test is pixel equality. The port then runs the
  `floodFill_CnIR` scan without a mask. The region, area, rectangle and image are identical
  to the gradient fill. `FloodFillBasicTests` compares both paths.
- `float` is excluded: `inf - inf` is NaN, so the gradient fill does not join equal infinite
  pixels, but equality does. A new value of -0.0 against a seed of +0.0 also breaks the
  equality fill, because the filled pixels stay equal to the seed value.
- Single-channel images use `SimpleC1`: `IndexOf`, `IndexOfAnyExcept` and
  `LastIndexOfAnyExcept` find run starts and run ends, and `Span.Fill` writes the runs. The
  pushed segments and their order are those of `floodFill_CnIR`.
- The gradient fill uses a pooled, cleared mask when the caller gives none.

## Deliberate differences from OpenCV 5.0

- Image width or height above 65535 throws `ArgumentException`. OpenCV truncates the 16-bit
  segment coordinates and gives undefined results.
- Validation runs before any mutation. OpenCV sets the mask frame before it checks `loDiff`,
  `upDiff` and the seed point, so an invalid call in OpenCV can still change the mask frame.
- Unsupported element types throw `NotSupportedException`. Wrong channel counts, connectivity,
  mask size and negative differences throw `ArgumentException`. A seed outside the image
  throws `ArgumentOutOfRangeException`.

## Performance

- The scans are serial. No `Vector<T>` loop applies except the run searches of `SimpleC1`.
- Pixel and mask access uses references (`Unsafe.Add`). No span is created per element.
- The pixel test is a struct generic parameter (`IFloodDiff<T>`), so the JIT inlines it.
- Benchmarks: `FloodFillBenchmarks` and `OpenCvSharpFloodFillBenchmarks` (`HoleFill`,
  `HoleFillMask`, `GradientGray`, `FixedRangeGray`). OpenCvSharp runs the gradient fill with
  a cleared internal mask for `HoleFill`; the port runs the equality fast path.
- Measured, 1 thread, BenchmarkDotNet `--fast --inProcess`, busy machine (port us /
  OpenCvSharp 4.13 us): `HoleFillMask` 73 / 63 (256) and 1261 / 1296 (1024); `GradientGray`
  1627 / 1582 (256) and 28826 / 25511 (1024). The ratios are 0.97x to 1.16x, inside the
  run-to-run noise. The scan has no per-element span, allocation or interface dispatch, so the
  code is not changed.

## Golden constraints (`tools/generate_floodfill_golden.py`)

- Requires cv2 5.0.0. The script asserts the version. No case is suspect.
- Cases without a mask call `cv2.floodFill(img, None, ...)`.
- Each case stores the input, the seed, the new value, loDiff, upDiff, flags, area, rectangle
  and output image. Mask cases also store the input and output mask.
- The runner replays each case on packed matrices and on an ROI image and ROI mask in padded
  parents. The image parent padding repeats the seed pixel, so a fill outside the ROI changes
  it and fails the test.
- Battery: hole images with border seeds (all corners, edge middle, foreground seeds), rings
  with diagonal leaks (4 versus 8 connectivity), gray 8U with floating and fixed ranges
  (fractional and saturating differences), masks with obstacles, a masked seed, a mask frame
  that OpenCV rewrites, BGR 8U, 32S (with `int.MinValue`/`int.MaxValue` wrap cases, huge
  differences), 32S and 32F with three channels, 32F with NaN, infinity and signed zero,
  1x1, 1xN and Nx1 images, mask fill values 0, 2, 3, 7, 11, 42, 128, 200 and 255, and
  257x131 random images.
- The JSON is read from the output `TestData` folder. When the test project does not copy it,
  the runner reads it from the source `TestData` folder.
