# Demosaicing port notes

Port of OpenCV 5.0 Bayer demosaicing (`modules/imgproc/src/demosaicing.cpp`): `demosaicing` and the
Bayer codes of `cvtColor`. Public API: `ImgProc/Color/Demosaicing` (`Demosaic`, `BayerToGray`),
enums `BayerPattern` (OpenCV names BG, GB, RG, GR) and `DemosaicMethod`.

## Behaviour

- Sources: one channel, 8U or 16U. VNG: 8U only. BGRA output: bilinear only; alpha is the type
  maximum.
- Bilinear (`Bayer2RGB_`) and gray (`Bayer2Gray_`): integer averages; gray uses the weights
  R 4899, G 9617, B 1868 with `CV_DESCALE` (shift 16 for two-pixel sums, 15 for the others). The
  first and last columns copy their neighbours; then the first and last rows copy theirs. Images
  with fewer than 3 rows give zero (also zero alpha); fewer than 3 columns give zero colour.
- The 8U SSE kernels of bilinear, gray and edge-aware compute the same integers as the scalar code
  (OpenCV 5.0 gray kernel: `v_dotprod` with the exact descale), so the port is scalar.
- Edge-aware (`Bayer2RGB_EdgeAware_T`): green from the direction with the smaller absolute
  difference; images with at most 2 rows or columns are zero.
- VNG (`Bayer2RGB_VNG_8u`), OpenCV 5.0 form: images smaller than 8 pixels in either direction use
  bilinear interpolation with their own pattern. Larger images are padded by 2 pixels with
  `BORDER_REFLECT_101`, so every output pixel is interpolated. Gradients and sums are integers. The
  final step multiplies a sum difference by a float scale and rounds with `cvRound`. The SSE blocks
  (8 pixels from the first non-green pixel of the padded row while `i <= N − 10`) use `0.5f / ng`;
  the scalar code uses a table whose `ng = 7` entry is `0.0714286f`. The port uses each value for the
  same pixels.

## Golden data

`tools/generate_demosaic_golden.py` → `TestData/demosaic_golden.json` (288 cases: 4 patterns ×
gray/BGR/BGRA/VNG/EA, 8U and 16U, 8 sizes from 1×1 to 64×48), `ImgProc/DemosaicingGoldenTests`.
258 cases match bit for bit. 30 cases are adjudicated cv2 4.5.5 versus OpenCV 5.0 differences,
pinned in `ImgProc/DemosaicingOpenCv5DifferenceTests`:

- PR #25968 (2024, issue #25823): the 4.5.5 8U SIMD gray kernel descaled wrongly (errors up to
  about 200). OpenCV 5.0 equals the scalar formula, which is also the 16U result (12 cases).
- PR #27226 (2025, issues #5089, #27225): 4.5.5 VNG wrote each interior pixel two rows above its
  source neighbourhood and copied the border pixels. The interior matches the port with that shift,
  except at most 0.1 % of values where the SSE block positions changed with the padding and a
  pixel has `ng = 7` (12 cases).
- PR #27226: VNG on images smaller than 8 pixels used the RG pattern for every code in 4.5.5
  (6 cases).
