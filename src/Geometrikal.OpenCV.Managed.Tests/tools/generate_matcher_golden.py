#!/usr/bin/env python3
"""Golden-data generator for the managed brute-force matcher port.

Runs cv2 5.0.0 ``BFMatcher`` ``match``, ``knnMatch`` and ``radiusMatch`` over a seeded battery
and writes ``TestData/matcher_golden.json``. ``Features/BFMatcherGoldenTests.cs`` replays the
cases and compares every ``DMatch`` exactly (indexes and distance bits).

Required environment (the script stops if cv2 is not 5.0.0):
 * cv2 5.0.0, for example
       pip install --no-deps --target <dir> opencv-contrib-python-headless==5.0.0.93
   and run with ``PYTHONPATH=<dir>``. numpy 1.22 or later.
 * OPENCV_CPU_DISABLE=AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL before cv2 is
   imported (the script sets it when it is not set). The distance functions are compiled for the
   SSE3 baseline; the setting keeps the environment equal to the other generators.
 * IPP, OpenCL and threads are disabled in the script.

Regenerate with (from the ``Geometrikal.OpenCV.Managed.Tests`` directory):
    PYTHONPATH=<cv2 5.0.0 dir> python tools/generate_matcher_golden.py

Battery notes:
 * Byte descriptors with few distinct values and duplicated train rows give many equal
   distances, so the k-nearest tie order and the radius sort order are checked.
 * Radius cases with more than 32 matches for one query check the MSVC ``std::sort`` order
   (the cv2 Windows wheels are built with MSVC; insertion sort runs only up to 32 elements).
 * ``coll_*`` cases use ``add`` with several train sets and per-set masks.
 * Each expected match is ``[queryIdx, trainIdx, imgIdx, distance float32 bits]``.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import json
import os
import platform
import sys

os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX,AVX2,FMA3,AVX512F,AVX512-COMMON,AVX512-SKX,AVX512-ICL")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

if cv2.__version__ != "5.0.0":
    sys.exit(f"generate_matcher_golden.py requires cv2 5.0.0 (found {cv2.__version__}); "
             "set PYTHONPATH to the opencv-contrib-python-headless==5.0.0.93 install")
if cv2.checkHardwareSupport(11) or cv2.checkHardwareSupport(13):
    sys.exit("AVX2/AVX-512 dispatch is enabled; set OPENCV_CPU_DISABLE before importing cv2")

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

NORMS = {"L1": cv2.NORM_L1, "L2": cv2.NORM_L2, "L2Sqr": cv2.NORM_L2SQR,
         "Hamming": cv2.NORM_HAMMING, "Hamming2": cv2.NORM_HAMMING2}
BYTE_NORMS = ["Hamming", "Hamming2", "L1", "L2", "L2Sqr"]
FLOAT_NORMS = ["L1", "L2", "L2Sqr"]
SUSPECT_CASES = {}


ARRAYS = []
ARRAY_INDEX = {}


def mat(a):
    """Index of the matrix in the shared ``arrays`` table."""
    a = np.ascontiguousarray(a)
    m = {"dtype": "u8" if a.dtype == np.uint8 else "f32", "rows": int(a.shape[0]), "cols": int(a.shape[1]),
         "data": base64.b64encode(a.tobytes()).decode("ascii")}
    key = (m["dtype"], m["rows"], m["cols"], m["data"])
    if key not in ARRAY_INDEX:
        ARRAY_INDEX[key] = len(ARRAYS)
        ARRAYS.append(m)
    return ARRAY_INDEX[key]


def dm(m):
    return [m.queryIdx, m.trainIdx, m.imgIdx, int(np.array(m.distance, np.float32).view(np.uint32))]


class Gen:
    def __init__(self):
        self.cases = []
        self.names = set()

    def add(self, name, case):
        assert name not in self.names, name
        self.names.add(name)
        case["name"] = name
        case["suspect"] = name in SUSPECT_CASES
        if case["suspect"]:
            case["suspectReason"] = SUSPECT_CASES[name]
        self.cases.append(case)


def descriptors(rng, dt, rows, cols, kind):
    if dt == "u8":
        if kind == "few":
            return rng.choice(np.array([0, 1, 3, 128, 255], np.uint8), size=(rows, cols))
        return rng.integers(0, 256, (rows, cols), dtype=np.uint8)
    if kind == "few":
        return rng.integers(-2, 3, (rows, cols)).astype(np.float32)
    return (rng.standard_normal((rows, cols)) * 10).astype(np.float32)


def with_duplicates(rng, train):
    t = train.copy()
    n = t.shape[0]
    for _ in range(max(1, n // 4)):
        t[rng.integers(0, n)] = t[rng.integers(0, n)]
    return t


def mask_for(rng, rows, cols):
    m = (rng.random((rows, cols)) < 0.7).astype(np.uint8) * rng.integers(1, 256, (rows, cols), dtype=np.uint8)
    m[m == 0] = 0
    if rows > 2:
        m[1, :] = 0
    return m


def pairs(rng, g):
    shapes_u8 = [(1, 1, 1), (3, 5, 2), (6, 9, 7), (10, 12, 8), (17, 23, 16), (20, 30, 32), (25, 18, 33),
                 (9, 40, 61), (12, 20, 64), (5, 7, 300)]
    shapes_f32 = [(1, 1, 1), (3, 5, 3), (6, 9, 15), (10, 12, 16), (17, 23, 17), (8, 30, 31), (25, 18, 32),
                  (9, 40, 33), (12, 20, 64), (14, 21, 128), (5, 7, 130)]
    for dt, norms, shapes in (("u8", BYTE_NORMS, shapes_u8), ("f32", FLOAT_NORMS, shapes_f32)):
        for norm in norms:
            for (nq, nt, ln) in shapes:
                for kind in ("rand", "few"):
                    q = descriptors(rng, dt, nq, ln, kind)
                    t = with_duplicates(rng, descriptors(rng, dt, nt, ln, kind))
                    if kind == "few" and nq > 2:
                        q[0] = t[nt // 2]
                    base = f"{dt}_{norm}_{nq}x{nt}x{ln}_{kind}"
                    mask = mask_for(rng, nq, nt)
                    single(g, base, norm, q, t, mask)


def run(norm, cross, op, q, trains, masks, single_train, k=None, radius=None, compact=False):
    m = cv2.BFMatcher(NORMS[norm], crossCheck=cross)
    if single_train:
        t = trains[0]
        mk = masks[0] if masks else None
        if op == "match":
            res = m.match(q, t) if mk is None else m.match(q, t, mask=mk)
            return [[dm(x) for x in res]]
        if op == "knn":
            res = m.knnMatch(q, t, k, mask=mk, compactResult=compact) if mk is not None else \
                m.knnMatch(q, t, k, compactResult=compact)
        else:
            res = m.radiusMatch(q, t, radius, mask=mk, compactResult=compact) if mk is not None else \
                m.radiusMatch(q, t, radius, compactResult=compact)
        return [[dm(x) for x in row] for row in res]
    m.add(trains)
    if op == "match":
        res = m.match(q, masks=masks) if masks else m.match(q)
        return [[dm(x) for x in res]]
    if op == "knn":
        res = m.knnMatch(q, k, masks=masks, compactResult=compact) if masks else m.knnMatch(q, k, compactResult=compact)
    else:
        res = m.radiusMatch(q, radius, masks=masks, compactResult=compact) if masks else \
            m.radiusMatch(q, radius, compactResult=compact)
    return [[dm(x) for x in row] for row in res]


def emit(g, name, norm, cross, op, q, trains, masks, single_train, k=None, radius=None, compact=False):
    out = run(norm, cross, op, q, trains, masks, single_train, k, radius, compact)
    case = {"op": op, "norm": norm, "crossCheck": cross, "single": single_train, "query": mat(q),
            "trains": [mat(t) for t in trains], "masks": [mat(x) for x in masks] if masks else None,
            "k": k, "radius": radius, "compact": compact, "expected": out}
    g.add(name, case)


def radii(q, t, norm):
    m = cv2.BFMatcher(NORMS[norm])
    d = np.array([[x.distance for x in row] for row in m.knnMatch(q, t, t.shape[0])], np.float64)
    vals = np.unique(d)
    picks = [vals[len(vals) // 6], vals[len(vals) // 3]]
    return [float(max(p, 1e-3)) for p in picks]


def single(g, base, norm, q, t, mask):
    emit(g, f"{base}_match", norm, False, "match", q, [t], None, True)
    emit(g, f"{base}_match_mask", norm, False, "match", q, [t], [mask], True)
    emit(g, f"{base}_match_cross", norm, True, "match", q, [t], None, True)
    for k in (1, 2, 7):
        emit(g, f"{base}_knn{k}", norm, False, "knn", q, [t], None, True, k=k)
        emit(g, f"{base}_knn{k}_mask{'_compact' if k == 2 else ''}", norm, False, "knn", q, [t], [mask], True, k=k,
             compact=k == 2)
    emit(g, f"{base}_knn1_cross", norm, True, "knn", q, [t], None, True, k=1)
    for i, r in enumerate(radii(q, t, norm)):
        emit(g, f"{base}_radius{i}", norm, False, "radius", q, [t], None, True, radius=r)
        emit(g, f"{base}_radius{i}_mask_compact", norm, False, "radius", q, [t], [mask], True, radius=r, compact=True)


def sort_order(rng, g):
    # One query, many train rows with few distinct distances: radius lists of 33 to 400 elements.
    for nt in (33, 41, 50, 64, 100, 257, 400):
        q = np.zeros((2, 4), np.uint8)
        t = rng.choice(np.array([0, 1, 3, 7], np.uint8), size=(nt, 4))
        t[:, 1:] = 0
        emit(g, f"sort_hamming_{nt}", "Hamming", False, "radius", q, [t], None, True, radius=2.0)
        tf = rng.integers(0, 3, (nt, 5)).astype(np.float32)
        emit(g, f"sort_l1f_{nt}", "L1", False, "radius", np.zeros((1, 5), np.float32), [tf], None, True, radius=4.0)


def collections(rng, g):
    for dt, norms in (("u8", BYTE_NORMS), ("f32", FLOAT_NORMS)):
        for norm in norms:
            for kind in ("rand", "few"):
                ln = 32 if dt == "u8" else 20
                q = descriptors(rng, dt, 11, ln, kind)
                trains = [descriptors(rng, dt, n, ln, kind) for n in (9, 14, 6)]
                masks = [mask_for(rng, 11, t.shape[0]) for t in trains]
                base = f"coll_{dt}_{norm}_{kind}"
                emit(g, f"{base}_match", norm, False, "match", q, trains, None, False)
                emit(g, f"{base}_match_masks", norm, False, "match", q, trains, masks, False)
                for k in (1, 2, 5):
                    emit(g, f"{base}_knn{k}", norm, False, "knn", q, trains, None, False, k=k)
                    emit(g, f"{base}_knn{k}_masks", norm, False, "knn", q, trains, masks, False, k=k)
                    emit(g, f"{base}_knn{k}_masks_compact", norm, False, "knn", q, trains, masks, False, k=k,
                         compact=True)
                r = radii(q, trains[0], norm)[1]
                emit(g, f"{base}_radius", norm, False, "radius", q, trains, None, False, radius=r)
                emit(g, f"{base}_radius_masks_compact", norm, False, "radius", q, trains, masks, False, radius=r,
                     compact=True)


def large(rng, g):
    q = rng.integers(0, 256, (300, 32), dtype=np.uint8)
    t = rng.integers(0, 256, (700, 32), dtype=np.uint8)
    emit(g, "large_u8_Hamming_knn2", "Hamming", False, "knn", q, [t], None, True, k=2)
    emit(g, "large_u8_Hamming_cross", "Hamming", True, "match", q, [t], None, True)
    qf = (rng.standard_normal((200, 128)) * 10).astype(np.float32)
    tf = (rng.standard_normal((400, 128)) * 10).astype(np.float32)
    emit(g, "large_f32_L2_knn2", "L2", False, "knn", qf, [tf], None, True, k=2)
    emit(g, "large_f32_L1_match", "L1", False, "match", qf, [tf], None, True)


def main():
    rng = np.random.default_rng(20261016)
    g = Gen()
    pairs(rng, g)
    sort_order(rng, g)
    collections(rng, g)
    large(rng, g)
    assert set(SUSPECT_CASES) <= g.names, set(SUSPECT_CASES) - g.names
    doc = {"generator": {"cv2": cv2.__version__, "numpy": np.__version__, "machine": platform.processor()},
           "arrays": ARRAYS, "cases": g.cases}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TestData", "matcher_golden.json")
    with open(out, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases to {out}")


if __name__ == "__main__":
    main()
