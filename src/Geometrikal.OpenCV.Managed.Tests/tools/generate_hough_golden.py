#!/usr/bin/env python3
"""Golden-data generator for the managed Hough transform port.

Runs cv2 4.5.5 HoughLines (standard and multi-scale), HoughLinesP,
HoughLinesPointSet and HoughCircles over a seeded battery and writes
``TestData/hough_golden.json``. ``ImgProc/HoughGoldenTests.cs`` replays each
case; lines and circles must match exactly (float32 values compared bit for
bit).

Regenerate with:
    python tools/generate_hough_golden.py
(run from the ``Geometrikal.OpenCV.Managed.Tests`` directory).

Environment: OPENCV_CPU_DISABLE removes AVX/AVX2/FMA dispatch; IPP, OpenCL
and threads are disabled (IPP has its own HoughLines/HoughLinesP).

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import json
import os

os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX,AVX2,AVX512F,FMA3")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

_BIAS = ("OpenCV 5.0 (PR #27992) outputs rho with the integer bin offset (numrho - 1)/2; 4.5.5 used "
         "(numrho - 1)*0.5, half a bin lower when numrho is even")
SUSPECT_CASES = {n: _BIAS for n in (
    "lines_a_r2_t0.0349_th10", "lines_b_r2_t0.0349_th10", "lines_c_r2_t0.0349_th10", "lines_sparse_r2_t0.0349_th10",
    "lines_b_r0.5_t0.0087_th25", "lines_c_r0.5_t0.0087_th25")}
SUSPECT_CASES["lines_c_range"] = ("OpenCV commit 1f0bfc8d83 computes numangle as floor(range/step) + 1 "
                                  "(98 angles for 0.3..2.0); 4.5.5 used cvRound(range/step) (97)")


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def img(a):
    return {"width": int(a.shape[1]), "height": int(a.shape[0]), "data": b64(a.astype(np.uint8))}


def lines_image(rng, w, h, nlines, noise):
    a = np.zeros((h, w), np.uint8)
    for _ in range(nlines):
        p1 = tuple(int(v) for v in rng.integers(0, (w, h)))
        p2 = tuple(int(v) for v in rng.integers(0, (w, h)))
        cv2.line(a, p1, p2, 255, 1)
    ys = rng.integers(0, h, noise)
    xs = rng.integers(0, w, noise)
    a[ys, xs] = 255
    return a


def main():
    rng = np.random.default_rng(20261023)
    cases = []

    def add(name, op, params, image, result):
        cases.append({"name": name, "op": op, "params": params, "image": image,
                      "result": b64(np.zeros(0, np.float32) if result is None else np.asarray(result)),
                      "count": 0 if result is None else int(np.asarray(result).shape[0]),
                      "suspect": name in SUSPECT_CASES})

    images = {
        "a": lines_image(rng, 64, 48, 4, 30),
        "b": lines_image(rng, 97, 61, 6, 80),
        "c": lines_image(rng, 160, 120, 10, 200),
        "sparse": lines_image(rng, 50, 50, 2, 5),
    }
    for key, a in images.items():
        for rho, theta, thr in ((1, np.pi / 180, 15), (2, np.pi / 90, 10), (0.5, np.pi / 360, 25)):
            res = cv2.HoughLines(a, rho, theta, thr)
            add(f"lines_{key}_r{rho}_t{round(theta, 4)}_th{thr}", "lines",
                {"rho": rho, "theta": theta, "threshold": thr}, img(a), None if res is None else res.reshape(-1, 2).astype(np.float32))
        res = cv2.HoughLines(a, 1, np.pi / 180, 12, min_theta=0.3, max_theta=2.0)
        add(f"lines_{key}_range", "lines", {"rho": 1, "theta": np.pi / 180, "threshold": 12, "minTheta": 0.3, "maxTheta": 2.0},
            img(a), None if res is None else res.reshape(-1, 2).astype(np.float32))
        for srn, stn in ((2, 2), (3, 4)):
            res = cv2.HoughLines(a, 1, np.pi / 180, 20, srn=srn, stn=stn)
            add(f"linesMS_{key}_s{srn}_{stn}", "lines", {"rho": 1, "theta": np.pi / 180, "threshold": 20, "srn": srn, "stn": stn},
                img(a), None if res is None else res.reshape(-1, 2).astype(np.float32))
        for thr, minlen, gap in ((10, 10, 3), (20, 25, 5), (5, 0, 0)):
            res = cv2.HoughLinesP(a, 1, np.pi / 180, thr, minLineLength=minlen, maxLineGap=gap)
            add(f"linesP_{key}_th{thr}_l{minlen}_g{gap}", "linesP", {"rho": 1, "theta": np.pi / 180, "threshold": thr,
                "minLineLength": minlen, "maxLineGap": gap}, img(a), None if res is None else res.reshape(-1, 4).astype(np.int32))

    pts = np.array([[0, 369], [10, 364], [20, 358], [30, 352], [40, 346], [50, 341], [60, 335], [70, 329], [80, 323],
                    [90, 318], [100, 312], [110, 306], [120, 300], [130, 295], [140, 289], [150, 284], [160, 277],
                    [170, 271], [180, 266], [190, 260]], np.float32)
    pts = np.concatenate([pts, rng.uniform(0, 400, (30, 2)).astype(np.float32)])
    res = cv2.HoughLinesPointSet(pts.reshape(-1, 1, 2), 20, 1, 0, 360, 1, 0, np.pi / 2, np.pi / 180)
    cases.append({"name": "pointSet", "op": "pointSet", "params": {}, "points": b64(pts),
                  "result": b64(res.reshape(-1, 3).astype(np.float64)), "count": int(res.shape[0]), "suspect": False})

    def circles_image(w, h, n, filled):
        a = np.full((h, w), 20, np.uint8)
        for _ in range(n):
            c = tuple(int(v) for v in rng.integers(10, (w - 10, h - 10)))
            r = int(rng.integers(5, 30))
            cv2.circle(a, c, r, int(rng.integers(120, 255)), -1 if filled else 2)
        noise = rng.normal(0, 6, (h, w))
        return cv2.GaussianBlur(np.clip(a + noise, 0, 255).astype(np.uint8), (5, 5), 1.5)

    circle_images = {"ca": circles_image(120, 90, 4, True), "cb": circles_image(160, 140, 7, False),
                     "cc": circles_image(200, 160, 10, True)}
    for key, a in circle_images.items():
        for dp, md, p1, p2, rmin, rmax in ((1, 15, 100, 20, 0, 0), (1.5, 10, 80, 15, 4, 35), (2, 20, 120, 12, 5, 40),
                                           (1, 15, 100, 18, 3, -1), (1, 8, 60, 25, 10, 12)):
            res = cv2.HoughCircles(a, cv2.HOUGH_GRADIENT, dp, md, param1=p1, param2=p2, minRadius=rmin, maxRadius=rmax)
            add(f"circles_{key}_dp{dp}_md{md}_p{p1}_{p2}_r{rmin}_{rmax}", "circles",
                {"dp": dp, "minDist": md, "param1": p1, "param2": p2, "minRadius": rmin, "maxRadius": rmax},
                img(a), None if res is None else res.reshape(-1, 3).astype(np.float32))

    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "hough_golden.json")
    with open(out, "w") as f:
        json.dump({"cases": cases}, f, separators=(",", ":"))
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
