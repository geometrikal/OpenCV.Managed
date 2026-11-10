# Accumulation port notes

Port of OpenCV 5.0 `accumulate`, `accumulateSquare`, `accumulateProduct` and `accumulateWeighted`
(`modules/imgproc/src/accum.cpp`, `accum.simd.hpp`). Public API: `ImgProc/Motion/Accumulation`.

## Behaviour

- Depth pairs: 8U, 16U, 32F to 32F or 64F; 64F to 64F. Any channel count. The mask is one
  channel, 8-bit; non-zero pixels are updated.
- Each source element converts to the accumulator type exactly. `accumulateSquare` and
  `accumulateProduct` round the product, then the sum (the SSE build has no FMA; the vector
  kernels convert to float before they multiply, so vector and scalar elements agree).
- `accumulateWeighted` computes `src·α + dst·β`. With a `float` accumulator OpenCV has two values of
  β: `(float)(1.0 − α)` in the vector kernels and `1 − (float)α` in the scalar tail. They differ
  for some α (for example 0.1794 and 0.289). The port uses the vector β for the elements of whole
  SSE blocks from the start of each plane:
  - no mask: blocks of 16 elements (8U source) or 8 elements (16U and 32F sources);
  - mask: blocks of 16 (8U) or 8 (16U) pixels, for 1 and 3 channels only; 32F has no masked vector
    kernel.
  A plane is the whole image when the source, accumulator and mask are continuous
  (`NAryMatIterator`), otherwise one row.
- With a `double` accumulator both β values are `1 − α` in `double`.

## Deliberate differences

- The masked vector kernels of `accumulate`, `accumulateSquare` and `accumulateProduct` add zero
  to masked-out elements. This changes an accumulator value of −0 to +0. The port leaves masked-out
  elements unchanged.
- IPP and OpenCL paths are not ported.

## Golden data

`tools/generate_accumulate_golden.py` → `TestData/accumulate_golden.json` (784 cases: all depth
pairs, 1-4 channels, two sizes, with and without a mask, four α values; all match bit for bit),
`ImgProc/AccumulationGoldenTests`.
