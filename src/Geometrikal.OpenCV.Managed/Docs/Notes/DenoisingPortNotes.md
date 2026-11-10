# Denoising port notes

Port of OpenCV 5.0 `fastNlMeansDenoising` and `fastNlMeansDenoisingColored`
(`modules/photo/src/denoising.cpp`, `fast_nlmeans_denoising_invoker.hpp`,
`fast_nlmeans_denoising_invoker_commons.hpp`). Public API: `Photo/Denoising`.

## Behaviour

- Types: 8U with `NORM_L2` or `NORM_L1`, 16U with `NORM_L1`; 1 to 4 channels; one `h` or one `h`
  per channel (per-channel weights and weight sums).
- Window sizes are reduced to odd values (`2·(size/2) + 1`). The source is extended by
  `search/2 + template/2` pixels with `BORDER_REFLECT_101`.
- Fixed-point multiplier: `min(IT_max / (search² · sampleMax), INT_MAX)`, with `IT = int` (8U) or
  `int64` (16U). The weight table has an entry per shifted average distance:
  `d = k · 2^shift / template²`, weight `cvRound(mult · exp(−d / (h²·cn)))` (L2) or
  `exp(−d² / (h²·cn))` (L1), with `h²·cn` in float; weights below `0.001 · mult` are zero.
- The distance of a patch pair is the integer sum over the template window; its table index is
  `sum >> shift`. The output is `(Σ w·p + Σw/2) / Σw` in unsigned integer arithmetic.
- OpenCV computes the patch sums incrementally per row stripe. The sums are exact integers, so the
  port computes them per search offset with running box sums. The result does not depend on the
  stripe layout or the thread count.
- Coloured variant: `COLOR_LBGR2Lab`, L with `h`, the a and b channels as one 2-channel image with
  `hColor`, then `COLOR_Lab2LBGR` with the source channel count.
- OpenCL and the multi-frame variants (`fastNlMeansDenoisingMulti`) are not ported.

## Golden data

`tools/generate_nlmeans_golden.py` → `TestData/nlmeans_golden.json` (58 cases: L2 and L1, 8U with
1-4 channels and per-channel `h`, 16U, four window pairs including even sizes, a 96×80 image, the
coloured variant with 3 and 4 channels), `Photo/DenoisingGoldenTests`. All cases match cv2 4.5.5
bit for bit.
