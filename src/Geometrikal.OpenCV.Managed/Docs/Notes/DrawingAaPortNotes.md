# Anti-Aliased Drawing Port Notes

## Scope

Source: OpenCV 5.0.0 `modules/imgproc/src/drawing.cpp` and
`modules/imgproc/src/drawing_text.cpp`.

Ported:

* `LINE_AA` (`LineType.AntiAliased = 16`) for `Line`, `Rectangle` (both
  overloads), `Circle`, `Ellipse`, `FillConvexPoly`, `FillPoly`, `Polylines`
  and `DrawContours`.
* `ArrowedLine` (`arrowedLine`) and `DrawMarker` (`drawMarker`, enum
  `MarkerType`).
* `HersheyFonts` enum and `TextMetrics.GetFontScaleFromHeight`
  (`getFontScaleFromHeight`).

Not ported: `putText` and `getTextSize`. See [Text](#text).

## Files

* `ImgProc/Drawing/Internal/AntiAliasedLine.cs`: `LineAA` with
  `SlopeCorrTable`, `FilterTable`, the end-point correction table and the
  `ICV_PUT_POINT` double blend.
* `ImgProc/Drawing/Internal/PolygonPrimitives.cs`: `FillConvexPoly` AA span
  rounding (`delta1 = XY_ONE - 1`, `delta2 = 0`), the AA edge-advance guard
  (`y < ymax || y == ymin`) and the AA outline. `CollectPolyEdges` AA branch:
  the outline uses `LineAA` with the row shifted to `XY_SHIFT`, and the edge
  uses the unclipped end points.
* `ImgProc/Drawing/Internal/CurvePrimitives.cs`: `ThickLine` thin AA lines use
  `LineAA`. AA round caps use `EllipseEx` with axes `thickness << 15` and
  thickness `-1`.
* `ImgProc/Drawing/Drawing.cs`: validation accepts `LineType.AntiAliased`.

## Behaviour

* `LineAA` blends into the existing pixels. The result depends on the
  background. Each channel gets two blend steps:
  `v += ((c - v) * a + 127) >> 8`.
* The AA path applies to `Mat<byte>` with 1, 3 or 4 channels.
* For `Mat<T>` with `T` not `byte`, every public method replaces `LINE_AA`
  with `LINE_8` (OpenCV `img.depth() != CV_8U` rule). The output is identical
  to `LINE_8`.
* For `Mat<byte>` with 2 channels, `LineAA` draws an 8-connected line between
  the truncated (`>> XY_SHIFT`, not rounded) end points. The fill routines
  still use the AA span rounding. This is the OpenCV behaviour.
* `Circle` with `LINE_AA` always uses `EllipseEx` (the condition is
  `line_type != LINE_8`).
* The non-AA code paths are unchanged. The existing drawing goldens pass
  without change.

## Text

OpenCV 5.0 does not contain Hershey glyph tables. `putText` and `getTextSize`
call `hersheyToTruetype`, which maps the font identifier to the Rubik or
Rubik-Italic TrueType font (`modules/imgproc/fonts/*.ttf.gz`), a pixel size
`cvRound(100 * fontScale / sf)` and a font weight. The text engine
(`FontRenderEngine::putText_`) renders anti-aliased glyphs with a modified
`stb_truetype` that supports variable-font weights (`fvar`, `gvar`, `avar`,
`HVAR`). The `lineType` argument is ignored: cv2 5.0.0 output contains
intermediate grey values for `LINE_8`.

A byte-exact port needs: gzip inflate of the fonts, the `stb_truetype` glyph
loader with variation interpolation, the `stb_truetype` scanline rasteriser in
the same float operation order, the glyph cache, and the layout code in
`drawing_text.cpp`. This is approximately 5000 lines of C. The port of
Hershey tables from OpenCV 4.x does not match OpenCV 5.0 output, so it is not
done.

`GetFontScaleFromHeight` needs only the size mapping. It returns
`pixelHeight / cvRound(100 / sf)` and ignores `thickness`, as OpenCV 5.0 does.

## Tests

* `Tests/ImgProc/DrawingAntiAliasedGoldenTests.cs`: 304 cv2 5.0.0 cases from
  `TestData/drawing_aa_golden.json` (generator
  `tools/generate_drawing_aa_golden.py`). The cases cover all primitives,
  thickness 0..10, shift 0..4 and 16, clipping, 1/2/3/4 channels, `ushort` and
  `float` fallback, zero and pattern backgrounds, arrowed lines and all marker
  types. 256x256 canvases are stored as SHA-256. All cases are byte-exact.
* `Tests/ImgProc/TextGoldenTests.cs`: 64 `getFontScaleFromHeight` cases, exact.
* `Tests/ImgProc/DrawingAntiAliasedBasicTests.cs`: fallback equality, ROI
  write-through, off-image no-op.
* `Tests/ImgProc/DrawingBasicTests.cs`: the tests that required
  `NotSupportedException` for `LINE_AA` now require that `LINE_AA` is accepted.

## Benchmarks

`DrawingAaBenchmarks` and `OpenCvSharpDrawingAaBenchmarks`: 64 AA lines with
thickness 1 and 5, 64 AA circle outlines, `FillPoly` AA; canvas sizes 256 and
1024. No `PutText` case, because `putText` is not ported.

## Performance

### Changes

* `AntiAliasedLine` rasterises through a generic channel-count struct
  (`Nch1`, `Nch3`, `Nch4`). The JIT removes the channel branches from the blend.
* `AntiAliasedLine` addresses pixels with one `ref byte` base and the row
  stride. The rasteriser does not create a row span for each pixel. The
  coordinate checks do not change. A reference is made only for a pixel inside
  the image.
* `EllipseEx` and `Ellipse2Poly` use stack buffers of 76 points instead of
  `List<T>`. 360° in 5° steps gives at most 74 points. Thick AA lines and thick
  AA outlines call `EllipseEx` for each round cap. Before the change, each cap
  allocated two lists.
* `PixelOps.HLine` writes 3-channel runs shorter than 16 pixels one pixel at a
  time. Longer runs use the doubling copy. The written values do not change.

The rasterisation order, the blend arithmetic and the edge order do not change.
All golden tests, including the non-AA drawing goldens, stay byte-exact.

### Results

BenchmarkDotNet `--fast --inProcess`, 1 thread. The ratio is port time ÷
OpenCvSharp 4.13 time in the same run. Another process used the machine. The
ratios have an uncertainty of approximately ±0.15. "Before" is the measurement
given at the start of the work.

| Case | Size | Before | After |
|---|---|---|---|
| LineAa1px | 256 | 1.45 | 0.78–0.92 |
| LineAa1px | 1024 | 1.41 | 0.86–1.02 |
| LineAa5px | 256 | 1.20 | 0.98–1.21 |
| LineAa5px | 1024 | 1.83 | 0.84–1.01 |
| FillPolyAa | 256 | 1.47 | 0.96–1.01 |
| FillPolyAa | 1024 | 1.40 | 1.06–1.13 |
| CircleAaOutline | 256 | 1.30 | 0.56–0.63 |
| CircleAaOutline | 1024 | 1.13 | 0.64–1.00 |

Allocation for 64 thick lines decreased from 106 KB to 2 KB. Allocation for 64
AA circle outlines decreased from 2.28 MB to 2.3 KB. OpenCV drawing is
single-threaded. The 16-thread results are the same as the 1-thread results
within the noise.

### Not changed

* `FillPolyAa` at 1024 stays approximately 10% slower than OpenCvSharp. The
  time is in the scanline fill of large areas, which is limited by memory
  writes. A direct 48-pixel start for long 3-channel runs gave no measurable
  gain and is not used.
* `CollectPolyEdges` and `FillEdgeCollection` keep the `List<PolyEdge>` and the
  `PolyEdge[]` working array. The allocation is approximately 10 KB for each
  call and is not in a per-pixel loop.
