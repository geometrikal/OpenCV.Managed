# Hough transform port notes

Port of OpenCV 5.0 `HoughLines` (standard and multi-scale), `HoughLinesP`,
`HoughLinesPointSet` and `HoughCircles` with `HOUGH_GRADIENT` (`modules/imgproc/src/hough.cpp`).
Public API: `ImgProc/Hough/HoughTransforms`, records `HoughLine`, `LineSegment`,
`HoughPointSetLine`, `HoughCircle`. `HOUGH_GRADIENT_ALT` is not ported.

## Behaviour

- Standard transform: accumulator of `(numangle + 2) × (numrho + 2)` with a one-cell frame,
  `numrho = cvRound((2·(w + h) + 1)/ρ)`, `numangle = floor((max − min)/θ) + 1` (minus one when the
  last angle is about π from the first). Each pixel votes at `cvRound(x·cos + y·sin) + (numrho − 1)/2`
  with float products (`tabCos`/`tabSin` are `(float)(cos(angle)·(1/ρ))`; the table angle advances
  in `float`). Local maximums are sorted by decreasing votes, then by accumulator index (a total
  order, so any sort gives the OpenCV order). Output `ρ = (r − (numrho − 1)/2)·ρ` (integer offset),
  `θ = (float)min + n·θ`. `Votes` is the `CV_32FC3` third value. `useEdgeValue` (OpenCV 5.0) votes
  with the pixel value.
- Multi-scale transform (`srn`, `stn` non-zero): the coarse accumulator is a byte array with the
  OpenCV wrap-around increments; when more than 1 % of its cells pass the threshold it falls back to
  the standard transform. The sine table uses `float cos` (`std::cos(float)` in the MSVC build),
  confirmed by the golden data. The threshold is clamped to 255 and the angle range is ignored.
- Probabilistic transform: points are visited in the order of `cv::RNG((uint64)-1).uniform(0,
  count)` (`Core/Internal/CvRng`), with the OpenCV fixed-point walk (16-bit shift, `int64`
  coordinates) and accumulator removal for accepted segments. The result is deterministic.
- `HoughCircles` (gradient): 16S Sobel with replicated borders and the port's Canny dx/dy overload
  (thresholds `max(1, param1/2)` and `param1`); each edge point with a gradient magnitude of at
  least 1 votes along its gradient in both directions with 10-bit fixed-point steps
  (`cvRound(v·(1/dp)·1024/|v|)`). The accumulator has no frame offset in its writes but its
  centre search skips the outer cells, as OpenCV. Centres are sorted by votes and index; the radius
  of each centre comes from a histogram of 10 bins per `dp` with the OpenCV sliding window and the
  `curCount·rBest ≥ maxCount·rCur` rule; circles are sorted by `cmpAccum` (support, radius, x, y)
  and overlaps closer than `minDist` are removed in order. `maxRadius < 0` returns centres only.
  OpenCV's parallel loops give thread-independent results, so the port runs on one thread.
- IPP and OpenCL paths are not ported (the golden data disables IPP).

## Golden data

`tools/generate_hough_golden.py` → `TestData/hough_golden.json` (52 cases, 15 of them circles; all
circle cases match bit for bit),
`ImgProc/HoughGoldenTests`. 7 cases are adjudicated cv2 4.5.5 versus OpenCV 5.0 differences,
pinned in `ImgProc/HoughOpenCv5DifferenceTests`:

- PR #27992 (2025): 4.5.5 subtracted `(numrho − 1)·0.5` in the output, half a ρ step below the
  accumulator bin when `numrho` is even (6 cases with ρ = 2 or 0.5).
- Commit 1f0bfc8d83 (2022): 4.5.5 used `numangle = cvRound((max − min)/θ)`; with the range
  0.3 to 2.0 OpenCV 5.0 has one more angle (1 case).
