#!/usr/bin/env python3
"""Golden-data generator for the managed polygon intersection port.

Runs cv2 4.5.5 intersectConvexConvex (float and int polygons, crossing,
nested, disjoint and touching cases, handleNested on and off) and
rotatedRectangleIntersection (random, equal, nested, edge-sharing and
disjoint rectangles), and writes ``TestData/intersection_golden.json``.
``ImgProc/ShapeIntersectionGoldenTests.cs`` replays each case; areas and
vertices must match bit for bit.

Regenerate with:
    python tools/generate_intersection_golden.py
(run from the ``Geometrikal.OpenCV.Managed.Tests`` directory).

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import json
import os

import cv2
import numpy as np

_BUFFER = ("cv2 4.5.5 wheel returns the unprocessed internal buffer with the FLT_MAX sentinel for touching and "
           "disjoint polygons; the OpenCV 5.0 source returns the deduplicated vertices or none")
_LINES = ("OpenCV PR #26974 (2025) computes the segment intersection differences in double; 4.5.5 subtracted in "
          "float (1-ulp vertex and area differences)")
_RECTS = ("OpenCV PR #23690 (2023) shifts both rectangles by their mean centre before the float computation; "
          "4.5.5 did not (1-ulp vertex differences)")
SUSPECT_CASES = {f"cc_{k}_{h}": _BUFFER for k in ("disjoint", "touch_edge", "touch_vertex") for h in "nx"}
SUSPECT_CASES.update({f"cc_14_{h}": _LINES for h in "nx"})
SUSPECT_CASES.update({f"rr_{i}": _RECTS for i in (11, 13, 16, 25, 27, 29, 43, 45, 47, 49, 50, 59)})


def pts(a):
    return [] if a is None else [[float(v) for v in p] for p in np.asarray(a, np.float32).reshape(-1, 2)]


def convex(rng, cx, cy, r, k, as_int=False, ccw=True):
    ang = np.sort(rng.random(k) * 2 * np.pi)
    p = np.stack([cx + r * np.cos(ang) * (0.6 + 0.4 * rng.random(k)), cy + r * np.sin(ang) * (0.6 + 0.4 * rng.random(k))], 1)
    hull = cv2.convexHull(p.astype(np.int32 if as_int else np.float32), clockwise=not ccw)
    return hull.reshape(-1, 2)


def main():
    rng = np.random.default_rng(20261023)
    cases = []
    for idx in range(60):
        as_int = idx % 3 == 0
        k1, k2 = int(rng.integers(3, 9)), int(rng.integers(3, 9))
        c = rng.random(4) * 40
        p1 = convex(rng, 50 + c[0], 50 + c[1], 30 * (0.4 + rng.random()), k1, as_int, ccw=idx % 2 == 0)
        p2 = convex(rng, 50 + c[2], 50 + c[3], 30 * (0.4 + rng.random()), k2, as_int, ccw=idx % 4 < 2)
        for nested in (True, False):
            area, inter = cv2.intersectConvexConvex(p1, p2, handleNested=nested)
            cases.append({"name": f"cc_{idx}_{'n' if nested else 'x'}", "op": "convex", "int": as_int, "p1": pts(p1), "p2": pts(p2),
                          "nested": nested, "area": float(area), "result": pts(inter), "suspect": f"cc_{idx}_{'n' if nested else 'x'}" in SUSPECT_CASES})
    # Special shapes: nested, disjoint, touching edge, shared vertex.
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32)
    specials = {
        "nested": (sq, sq * 0.5 + 2.5), "disjoint": (sq, sq + 20), "touch_edge": (sq, sq + [10, 0]),
        "touch_vertex": (sq, sq + [10, 10]), "same": (sq, sq.copy()), "overlap_half": (sq, sq + [5, 0]),
        "triangle_in_square": (sq, np.array([[2, 2], [8, 2], [5, 8]], np.float32)),
    }
    for name, (a, b) in specials.items():
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        for nested in (True, False):
            area, inter = cv2.intersectConvexConvex(a, b, handleNested=nested)
            cname = f"cc_{name}_{'n' if nested else 'x'}"
            cases.append({"name": cname, "op": "convex", "int": False, "p1": pts(a), "p2": pts(b),
                          "nested": nested, "area": float(area), "result": pts(inter), "suspect": cname in SUSPECT_CASES})

    def rr(c, s, a):
        return ((float(c[0]), float(c[1])), (float(s[0]), float(s[1])), float(a))

    rects = []
    for idx in range(60):
        r1 = rr(rng.random(2) * 40 + 30, rng.random(2) * 30 + 5, rng.random() * 180 - 90)
        r2 = rr(rng.random(2) * 40 + 30, rng.random(2) * 30 + 5, rng.random() * 180 - 90)
        rects.append((f"rr_{idx}", r1, r2))
    rects += [
        ("rr_same", rr((10, 10), (8, 4), 30), rr((10, 10), (8, 4), 30)),
        ("rr_nested", rr((10, 10), (8, 6), 15), rr((10, 10), (2, 2), 40)),
        ("rr_disjoint", rr((10, 10), (4, 4), 0), rr((30, 30), (4, 4), 10)),
        ("rr_shared_edge", rr((10, 10), (4, 4), 0), rr((14, 10), (4, 4), 0)),
        ("rr_axis_cross", rr((10, 10), (20, 2), 0), rr((10, 10), (2, 20), 0)),
        ("rr_far", rr((1e5 + 10, 2e5), (8, 4), 20), rr((1e5 + 12, 2e5 + 1), (6, 6), 65)),
        ("rr_empty", rr((10, 10), (0, 4), 0), rr((10, 10), (4, 4), 0)),
    ]
    for name, r1, r2 in rects:
        ret, region = cv2.rotatedRectangleIntersection(r1, r2)
        cases.append({"name": name, "op": "rect", "r1": [r1[0][0], r1[0][1], r1[1][0], r1[1][1], r1[2]],
                      "r2": [r2[0][0], r2[0][1], r2[1][0], r2[1][1], r2[2]], "type": int(ret), "result": pts(region),
                      "suspect": name in SUSPECT_CASES})

    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "intersection_golden.json")
    with open(out, "w") as f:
        json.dump({"cases": cases}, f, separators=(",", ":"))
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
