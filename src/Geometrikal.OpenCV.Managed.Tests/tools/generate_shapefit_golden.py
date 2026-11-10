#!/usr/bin/env python3
"""Golden-data generator for the managed shape-fitting port.

Runs cv2 5.0.0 convexHull, isContourConvex, convexityDefects, minAreaRect, boxPoints,
minEnclosingCircle, fitEllipse, fitEllipseAMS, fitEllipseDirect, fitLine and
pointPolygonTest over a seeded battery and writes ``TestData/shapefit_golden.json``.
``ImgProc/ShapeFittingGoldenTests.cs`` replays each case.

Regenerate with (from the ``Geometrikal.OpenCV.Managed.Tests`` directory):
    PYTHONPATH=<folder with cv2 5.0.0> python tools/generate_shapefit_golden.py

Environment and battery constraints:
 * The script requires cv2 5.0.0 (the geometry module of the port). IPP, OpenCL and
   threads are disabled.
 * The ellipse fits call cv2.setRNGSeed(SEED) first. fitEllipseAMS and fitEllipseDirect
   draw offsets from theRNG for every point, and degenerate inputs use those offsets.
 * cv2 5.0.0 wheels use LAPACK for SVDs with 25 or more rows. fitEllipse with 6 or more
   points uses that SVD; such cases have "lapack": true and the runner compares them with a
   relative tolerance. All other cases must match exactly.
 * Input points are stored as JSON numbers; float32 values round-trip exactly through the
   double representation.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import json
import math
import os

import cv2
import numpy as np

assert cv2.__version__ == "5.0.0", cv2.__version__
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

SEED = 12345
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TestData", "shapefit_golden.json")
DIST_TYPES = {"L1": cv2.DIST_L1, "L2": cv2.DIST_L2, "L12": cv2.DIST_L12, "Fair": cv2.DIST_FAIR,
              "Welsch": cv2.DIST_WELSCH, "Huber": cv2.DIST_HUBER}


def jnum(v):
    v = float(v)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    return v


class Gen:
    def __init__(self):
        self.inputs = []
        self.cases = []
        self.names = set()

    def add_input(self, name, pts):
        pts = np.ascontiguousarray(pts).reshape(-1, 2)
        kind = "i32" if pts.dtype == np.int32 else "f32"
        assert pts.dtype in (np.int32, np.float32), pts.dtype
        self.inputs.append({"name": name, "kind": kind, "points": [jnum(v) if kind == "f32" else int(v) for v in pts.reshape(-1)]})
        return len(self.inputs) - 1

    def add(self, name, op, inp, params, expected, lapack=False):
        assert name not in self.names, name
        self.names.add(name)
        self.cases.append({"name": name, "op": op, "input": inp, "params": params, "expected": expected,
                           "lapack": lapack, "suspect": False})


def call(fn):
    try:
        return fn(), None
    except cv2.error:
        return None, True


def rect5(r):
    (cx, cy), (w, h), a = r
    return [jnum(np.float32(v)) for v in (cx, cy, w, h, a)]


def cases_for(g, name, pts):
    """Adds every applicable operation for one point set."""
    pts = np.ascontiguousarray(pts)
    n = len(pts)
    inp = g.add_input(name, pts)
    p = pts.reshape(-1, 1, 2)
    is_int = pts.dtype == np.int32

    if n > 0:
        for cw in (False, True):
            hp = cv2.convexHull(p, clockwise=cw, returnPoints=True)
            g.add(f"{name}_hullPts_cw{int(cw)}", "convexHull", inp, {"clockwise": cw},
                  {"points": [jnum(v) if not is_int else int(v) for v in hp.reshape(-1)]})
            hi = cv2.convexHull(p, clockwise=cw, returnPoints=False)
            g.add(f"{name}_hullIdx_cw{int(cw)}", "convexHullIndices", inp, {"clockwise": cw},
                  {"indices": [int(v) for v in hi.reshape(-1)]})
        g.add(f"{name}_isConvex", "isContourConvex", inp, {}, {"value": bool(cv2.isContourConvex(p))})

    if is_int and n > 0:
        hi = cv2.convexHull(p, returnPoints=False)
        d, err = call(lambda: cv2.convexityDefects(p, hi))
        if err:
            g.add(f"{name}_defects", "convexityDefects", inp, {"hull": [int(v) for v in hi.reshape(-1)]}, {"error": True})
        else:
            vals = [] if d is None else [int(v) for v in d.reshape(-1)]
            g.add(f"{name}_defects", "convexityDefects", inp, {"hull": [int(v) for v in hi.reshape(-1)]}, {"defects": vals})

    if n > 0:
        r = cv2.minAreaRect(p)
        g.add(f"{name}_minAreaRect", "minAreaRect", inp, {}, {"rect": rect5(r)})
        bp = cv2.boxPoints(r)
        g.add(f"{name}_boxPoints", "boxPoints", inp, {"rect": rect5(r)}, {"points": [jnum(v) for v in bp.reshape(-1)]})
        (cx, cy), rad = cv2.minEnclosingCircle(p)
        g.add(f"{name}_minCircle", "minEnclosingCircle", inp, {}, {"circle": [jnum(np.float32(v)) for v in (cx, cy, rad)]})

    if n >= 5:
        for op, fn in (("fitEllipse", cv2.fitEllipse), ("fitEllipseAMS", cv2.fitEllipseAMS), ("fitEllipseDirect", cv2.fitEllipseDirect)):
            cv2.setRNGSeed(SEED)
            res, err = call(lambda: fn(p))
            exp = {"error": True} if err else {"rect": rect5(res)}
            g.add(f"{name}_{op}", op, inp, {"seed": SEED}, exp, lapack=(op == "fitEllipse" and n >= 25))

    if n > 0:
        for dname, dt in DIST_TYPES.items():
            ln = cv2.fitLine(p, dt, 0, 0.01, 0.01)
            g.add(f"{name}_fitLine_{dname}", "fitLine", inp, {"dist": dname, "param": 0.0, "reps": 0.01, "aeps": 0.01},
                  {"line": [jnum(v) for v in ln.reshape(-1)]})


def ppt_cases(g, name, pts, rng):
    inp = g.add_input(name + "_ppt", pts)
    p = np.ascontiguousarray(pts).reshape(-1, 1, 2)
    x0, y0 = pts.reshape(-1, 2).min(axis=0)
    x1, y1 = pts.reshape(-1, 2).max(axis=0)
    test = [(float(x0) - 1, float(y0) - 1), ((float(x0) + float(x1)) / 2, (float(y0) + float(y1)) / 2),
            (float(pts.reshape(-1, 2)[0][0]), float(pts.reshape(-1, 2)[0][1])), (float(x1) + 0.5, float(y1)),
            (float(x0), (float(y0) + float(y1)) / 2)]
    for _ in range(6):
        test.append((float(rng.integers(int(x0) - 2, int(x1) + 3)), float(rng.integers(int(y0) - 2, int(y1) + 3))))
        test.append((float(np.float32(rng.uniform(float(x0) - 2, float(x1) + 2))), float(np.float32(rng.uniform(float(y0) - 2, float(y1) + 2)))))
    test = [(float(np.float32(a)), float(np.float32(b))) for a, b in test]
    for md in (False, True):
        vals = [jnum(cv2.pointPolygonTest(p, t, md)) for t in test]
        g.add(f"{name}_ppt_md{int(md)}", "pointPolygonTest", inp, {"measureDist": md, "points": [v for t in test for v in t]},
              {"values": vals})


def contours_of(mask, method=cv2.CHAIN_APPROX_NONE):
    cs, _ = cv2.findContours(mask, cv2.RETR_LIST, method)
    return [c.reshape(-1, 2).astype(np.int32) for c in cs]


def main():
    g = Gen()
    rng = np.random.default_rng(20251015)

    # Analytic masks: rotated rectangles, circles, ellipses, star polygons.
    for ang in (0, 30, 45, 90):
        m = np.zeros((96, 96), np.uint8)
        box = cv2.boxPoints(((48.0, 48.0), (50.0, 24.0), float(ang)))
        cv2.fillPoly(m, [np.round(box).astype(np.int32)], 255)
        for k, c in enumerate(contours_of(m)):
            cases_for(g, f"rectmask{ang}_{k}", c)
        for k, c in enumerate(contours_of(m, cv2.CHAIN_APPROX_SIMPLE)):
            cases_for(g, f"rectmask{ang}_simple_{k}", c)
            ppt_cases(g, f"rectmask{ang}_simple_{k}", c, rng)
    m = np.zeros((96, 96), np.uint8)
    cv2.circle(m, (40, 50), 30, 255, -1)
    cv2.ellipse(m, (70, 30), (20, 8), 35, 0, 360, 0, -1)
    for k, c in enumerate(contours_of(m)):
        cases_for(g, f"circlemask_{k}", c)
    m = np.zeros((128, 128), np.uint8)
    star = [(64 + (50 if i % 2 == 0 else 20) * math.cos(i * math.pi / 5), 64 + (50 if i % 2 == 0 else 20) * math.sin(i * math.pi / 5)) for i in range(10)]
    cv2.fillPoly(m, [np.round(np.array(star)).astype(np.int32)], 255)
    for k, c in enumerate(contours_of(m)):
        cases_for(g, f"starmask_{k}", c)
        ppt_cases(g, f"starmask_{k}", c, rng)
    for k, c in enumerate(contours_of(m, cv2.CHAIN_APPROX_SIMPLE)):
        cases_for(g, f"starmask_simple_{k}", c)

    # Random masks: overlapping filled shapes.
    for s in range(6):
        m = np.zeros((128, 128), np.uint8)
        for _ in range(5):
            kind = rng.integers(0, 3)
            if kind == 0:
                cv2.circle(m, (int(rng.integers(10, 118)), int(rng.integers(10, 118))), int(rng.integers(3, 25)), 255, -1)
            elif kind == 1:
                cv2.ellipse(m, (int(rng.integers(10, 118)), int(rng.integers(10, 118))),
                            (int(rng.integers(3, 30)), int(rng.integers(3, 30))), float(rng.uniform(0, 180)), 0, 360, 255, -1)
            else:
                poly = rng.integers(0, 128, size=(int(rng.integers(3, 8)), 2)).astype(np.int32)
                cv2.fillPoly(m, [poly], 255)
        cs = contours_of(m)
        for k, c in enumerate(cs[:6]):
            cases_for(g, f"rand{s}_{k}", c)
            if k < 2:
                ppt_cases(g, f"rand{s}_{k}", c, rng)
        for k, c in enumerate(contours_of(m, cv2.CHAIN_APPROX_SIMPLE)[:3]):
            cases_for(g, f"rand{s}_simple_{k}", c)

    # Float point sets.
    for ang in (0.0, 45.0, 90.0, 17.5):
        box = cv2.boxPoints(((20.5, 30.25), (12.0, 7.0), ang)).astype(np.float32)
        cases_for(g, f"frect{int(ang * 10)}", box)
    for s in range(4):
        t = rng.uniform(0, 2 * math.pi, size=40)
        a, b, th = rng.uniform(5, 30), rng.uniform(3, 20), rng.uniform(0, math.pi)
        x = 50 + a * np.cos(t) * math.cos(th) - b * np.sin(t) * math.sin(th) + rng.normal(0, 0.3, size=40)
        y = 40 + a * np.cos(t) * math.sin(th) + b * np.sin(t) * math.cos(th) + rng.normal(0, 0.3, size=40)
        pts = np.stack([x, y], axis=1).astype(np.float32)
        cases_for(g, f"fellipse{s}", pts)
        cases_for(g, f"fellipse{s}_n12", pts[:12])
        cases_for(g, f"fellipse{s}_n6", pts[:6])
        cases_for(g, f"fellipse{s}_n5", pts[:5])
        fp = cv2.convexHull(pts.reshape(-1, 1, 2)).reshape(-1, 2)
        ppt_cases(g, f"fellipse{s}_hull", fp, rng)
    for s in range(3):
        n = 30
        x = rng.uniform(0, 100, size=n)
        y = 0.4 * x + 7 + rng.normal(0, 0.5, size=n)
        y[rng.integers(0, n, size=4)] += rng.uniform(-30, 30, size=4)
        cases_for(g, f"fline{s}", np.stack([x, y], axis=1).astype(np.float32))
        cases_for(g, f"fcloud{s}", rng.normal(50, 15, size=(25, 2)).astype(np.float32))
        cases_for(g, f"icloud{s}", rng.integers(-20, 60, size=(int(rng.integers(8, 40)), 2)).astype(np.int32))

    # Degenerate inputs.
    deg = {
        "one_i": np.array([[3, 4]], np.int32),
        "two_i": np.array([[3, 4], [10, 1]], np.int32),
        "two_vert_i": np.array([[3, 4], [3, 9]], np.int32),
        "two_up_i": np.array([[3, 4], [9, 9]], np.int32),
        "three_i": np.array([[0, 0], [10, 0], [3, 7]], np.int32),
        "same_i": np.array([[5, 5]] * 6, np.int32),
        "collinear_i": np.array([[i, 2 * i + 1] for i in range(8)], np.int32),
        "collinear_h_i": np.array([[i, 3] for i in (5, 1, 7, 2, 9, 0)], np.int32),
        "dups_i": np.array([[0, 0], [4, 0], [4, 0], [4, 4], [0, 4], [0, 4], [0, 0], [2, 2]], np.int32),
        "square_i": np.array([[0, 0], [0, 10], [10, 10], [10, 0]], np.int32),
        "one_f": np.array([[3.5, 4.25]], np.float32),
        "two_f": np.array([[3.5, 4.25], [10.0, 1.5]], np.float32),
        "three_f": np.array([[0, 0], [10.5, 0], [3, 7.25]], np.float32),
        "collinear_f": np.array([[i * 0.5, i * 1.5 + 1] for i in range(9)], np.float32),
        "same_f": np.array([[1.5, -2.5]] * 7, np.float32),
        "negzero_f": np.array([[0.0, 1.0], [-0.0, 2.0], [1.0, -0.0], [1.0, 0.0], [0.5, 0.5]], np.float32),
        "negzero_dup_f": np.array([[5.0, 0.0], [0.0, 0.0], [0.0, 5.0], [-0.0, 0.0], [5.0, 5.0], [0.0, -0.0]], np.float32),
        "negzero_dup2_f": np.array([[-0.0, 0.0], [5.0, 0.0], [0.0, 0.0], [0.0, 5.0], [5.0, 5.0]], np.float32),
    }
    for k, v in deg.items():
        cases_for(g, f"deg_{k}", v)

    # Repeated hull vertices: the index output keeps the index sequence monotonous.
    cases_for(g, "dupcorner_i", np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0], [10, 0], [10, 10]], np.int32))
    cases_for(g, "dupcorner_f", np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0], [10, 0], [10, 10]], np.float32))
    for s in range(3):
        cases_for(g, f"dupcloud{s}_i", rng.integers(0, 4, size=(30, 2)).astype(np.int32))
        cases_for(g, f"dupcloud{s}_f", (rng.integers(0, 4, size=(30, 2)) * 0.5).astype(np.float32))
    m = np.zeros((64, 64), np.uint8)
    cv2.line(m, (5, 5), (50, 30), 255, 1)
    cv2.polylines(m, [np.array([[10, 50], [30, 40], [55, 58], [60, 10]], np.int32)], False, 255, 1)
    m[20, 40] = 255
    for k, c in enumerate(contours_of(m)):
        cases_for(g, f"thin_{k}", c)
    for k, c in enumerate(contours_of(m, cv2.CHAIN_APPROX_SIMPLE)):
        cases_for(g, f"thin_simple_{k}", c)

    # Large sets for the shuffle of minEnclosingCircle.
    cases_for(g, "big_i", rng.integers(0, 500, size=(300, 2)).astype(np.int32))
    cases_for(g, "big_f", rng.normal(0, 100, size=(257, 2)).astype(np.float32))

    with open(OUT, "w", newline="\n") as f:
        json.dump({"cv2": cv2.__version__, "inputs": g.inputs, "cases": g.cases}, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases, {len(g.inputs)} inputs to {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
