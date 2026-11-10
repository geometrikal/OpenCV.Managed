# Inpaint port notes

Port of OpenCV 5.0 `inpaint` (`modules/photo/src/inpaint.cpp`). Public API: `Photo/Inpainting`
(`Inpaint`), enum `InpaintMethod` (`NavierStokes` = `INPAINT_NS`, `Telea` = `INPAINT_TELEA`).

## Behaviour

- Types: 8U with 1 or 3 channels, 16U and 32F with 1 channel. The mask is 8-bit, one channel.
- The radius is `cvRound(radius)` clamped to [1, 100].
- The flag, distance and band matrices have a one-pixel frame (`rows + 2`, `cols + 2`). The band
  is the cross dilation of the mask minus the mask. Telea first marches outwards over the
  `(2r+1)²` dilation ring and negates those distances.
- Priority queue: smallest distance first; equal distances in insertion order. The 4.5.5 linked
  list and the 5.0 `std::priority_queue` with an order counter give the same order.
- Arithmetic follows the MSVC build: `sqrt` and `fabs` of float arguments are float; the fast
  marching solution is double. Telea one-channel uses `1/(|r|²·sqrt_f(|r|²))` in float, three-channel
  uses a double square root. Integer Telea outputs are `saturate_cast(value + 0.5)` (double, half
  to even); float Telea outputs are the value. Navier-Stokes outputs are
  `saturate_cast((double)Ia / s)`.
- The port runs on one thread, as OpenCV.

## Golden data

`tools/generate_inpaint_golden.py` → `TestData/inpaint_golden.json` (72 cases: 8U gray and colour,
16U, 32F; disc, edge and scattered masks; radii 1, 3, 5.6; both methods),
`Photo/InpaintGoldenTests`. 21 cases match cv2 4.5.5 directly. 51 cases are suspects. An internal
4.5.5 mode (`Inpaint(..., legacy455: true)`) changes two expressions and reproduces every suspect
case exactly (`Photo/InpaintOpenCv5DifferenceTests`):

- Commit 2688248df7 (2024, C-API cleanup): OpenCV 5.0 reads a neighbour value at `(k − 1, l − 1)`;
  4.5.5 read `(km, lm)`, which is image row or column 1 for neighbours in row or column 0. The
  generator marks every case whose mask lies within the radius of the top or left edge.
- Commit dfb6069ad9 (2024): float Telea outputs no longer add 0.5; 4.5.5 rounded
  `(float)(value + 0.5f)`.
