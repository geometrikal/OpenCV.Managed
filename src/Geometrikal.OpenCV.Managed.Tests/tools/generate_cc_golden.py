#!/usr/bin/env python3
"""Golden-data generator for the managed connected components port.

Runs cv2 4.5.5 ``connectedComponentsWithAlgorithm`` and
``connectedComponentsWithStatsWithAlgorithm`` over a seeded battery and writes
``TestData/cc_golden.json``. ``ImgProc/CcGoldenTests.cs`` replays each case.

Regenerate with:
    python tools/generate_cc_golden.py
(run from the ``Geometrikal.OpenCV.Managed.Tests`` directory).

Environment and battery constraints:
 * One thread, IPP and OpenCL off. The parallel variants give the same
   output, but the battery records the sequential reference.
 * 4.5.5 Spaghetti4C (CCL_DEFAULT/BOLELLI/SPAGHETTI with connectivity 4)
   leaves background labels uninitialised (OpenCV issue 21366, fixed in
   4.6.0). OpenCV 5.0 Spaghetti4C gives the SAUF 4-connectivity output, so
   every 4-connectivity case takes its expected output from CCL_WU. The
   field ``source`` records the cv2 algorithm that produced the output.
 * 4.5.5 has no CV_16U overflow check (added in 4.13.0) and wraps labels.
   CV_16U cases use images whose provisional label bound is below 65535.
   The overflow is covered by analytic tests in CcBasicTests.
 * Random images are not stored: the runner rebuilds them from
   ``synthetic`` (the hash of Inputs.cs in the benchmark project). Large
   cases store the SHA-256 of the output buffers instead of the buffers.
 * Arrays are stored once in a shared ``arrays`` list.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import hashlib
import json
import os

import cv2
import numpy as np

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(1)
cv2.ipp.setUseIPP(False)

SUSPECT_CASES = {}

ALGORITHMS = {"default": -1, "wu": 0, "grana": 1, "bolelli": 2, "sauf": 3, "bbdt": 4, "spaghetti": 5}
LTYPES = {"s32": cv2.CV_32S, "u16": cv2.CV_16U}
DTYPES = {"u8": np.uint8, "i8": np.int8, "s32": np.int32, "u16": np.uint16, "f64": np.float64}


def hash_u32(x, y):
    h = (np.asarray(x, dtype=np.uint32) * np.uint32(73856093)) ^ (np.asarray(y, dtype=np.uint32) * np.uint32(19349663))
    h ^= h >> np.uint32(13)
    h = h * np.uint32(0x5BD1E995)
    h ^= h >> np.uint32(15)
    return h


def synthetic(w, h, seed, permille):
    """Foreground where hash(x + 7919 * seed, y) % 1000 < permille; mirrors CcGoldenTests.Synthetic."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.uint32)
    hv = hash_u32(xs + np.uint32(7919 * seed), ys)
    return np.where(hv % np.uint32(1000) < permille, 255, 0).astype(np.uint8)


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def sha(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def dt_of(arr):
    for k, v in DTYPES.items():
        if arr.dtype == v:
            return k
    raise ValueError(arr.dtype)


class Gen:
    def __init__(self):
        self.cases = []
        self.names = set()
        self.arrays = []
        self.index = {}

    def ref(self, arr):
        arr = np.ascontiguousarray(arr)
        m = {"dtype": dt_of(arr), "width": int(arr.shape[1]), "height": int(arr.shape[0]), "data": b64(arr)}
        key = (m["dtype"], m["width"], m["height"], m["data"])
        if key not in self.index:
            self.index[key] = len(self.arrays)
            self.arrays.append(m)
        return self.index[key]

    def battery(self, tag, img, synth=None, hashed=False, ltypes=("s32", "u16"), algorithms=ALGORITHMS):
        """Adds every (connectivity, algorithm, label type, stats) case for one image."""
        h, w = img.shape
        for conn in (4, 8):
            for aname in algorithms:
                alg = ALGORITHMS[aname]
                source = ALGORITHMS["wu"] if conn == 4 else alg
                for lt in ltypes:
                    for with_stats in (False, True):
                        name = f"{tag}_c{conn}_{aname}_{lt}{'_stats' if with_stats else ''}"
                        assert name not in self.names, name
                        self.names.add(name)
                        if with_stats:
                            n, labels, stats, cents = cv2.connectedComponentsWithStatsWithAlgorithm(
                                img, conn, LTYPES[lt], source)
                        else:
                            n, labels = cv2.connectedComponentsWithAlgorithm(img, conn, LTYPES[lt], source)
                            stats = cents = None
                        case = {"name": name, "connectivity": conn, "algorithm": alg, "source": source,
                                "ltype": lt, "stats": with_stats, "n": int(n), "suspect": name in SUSPECT_CASES}
                        if case["suspect"]:
                            case["suspectReason"] = SUSPECT_CASES[name]
                        if synth is not None:
                            case["synthetic"] = synth
                        else:
                            case["input"] = self.ref(img)
                        outs = {"labels": labels, "statsData": stats, "centroids": cents}
                        for key, arr in outs.items():
                            if arr is None:
                                continue
                            if hashed:
                                case[key + "Sha"] = sha(arr)
                            else:
                                case[key] = self.ref(arr)
                        self.cases.append(case)


def structured():
    imgs = {}
    imgs["discriminating"] = np.array([[0, 0, 0, 0, 1], [1, 0, 0, 0, 0]], np.uint8) * 255
    imgs["single"] = np.array([[7]], np.uint8)
    imgs["single_bg"] = np.array([[0]], np.uint8)
    imgs["all_fg_3x4"] = np.full((3, 4), 1, np.uint8)
    imgs["all_bg_2x2"] = np.zeros((2, 2), np.uint8)
    imgs["all_bg_5x9"] = np.zeros((5, 9), np.uint8)
    yy, xx = np.mgrid[0:9, 0:11]
    imgs["checker_9x11"] = (((yy + xx) % 2) * 200).astype(np.uint8)
    imgs["checker_8x8"] = (((yy[:8, :8] + xx[:8, :8]) % 2) * 1).astype(np.uint8)
    u = np.zeros((12, 17), np.uint8)
    u[1:11, 2] = 255
    u[1:11, 8] = 255
    u[10, 2:9] = 255
    u[1:9, 12] = 255
    u[1:9, 15] = 255
    u[9, 13] = 255
    u[9, 14] = 255
    imgs["u_shapes"] = u
    v = np.zeros((9, 19), np.uint8)
    for i in range(9):
        v[i, i] = 255
        v[i, 18 - i] = 255
    v[0, 9] = 255
    imgs["v_shapes"] = v
    sp = np.zeros((21, 23), np.uint8)
    top, left, bottom, right = 0, 0, 20, 22
    while top <= bottom and left <= right:
        sp[top, left:right + 1] = 1
        sp[top:bottom + 1, right] = 1
        if top + 2 <= bottom:
            sp[bottom, left:right + 1] = 1
        if left + 2 <= right:
            sp[top + 2:bottom + 1, left] = 1
        top, left, bottom, right = top + 2, left + 2, bottom - 2, right - 2
    imgs["spiral"] = sp
    diag = np.zeros((7, 9), np.uint8)
    for i in range(7):
        diag[i, (i * 3) % 9] = 1
        diag[i, (i * 5 + 1) % 9] = 1
    imgs["diagonals"] = diag
    imgs["block_edges"] = np.array([[0, 1, 1, 0, 0, 1, 1],
                                    [1, 0, 0, 1, 1, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 1],
                                    [1, 1, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 1, 0, 1, 1]], np.uint8)
    imgs["row_1x9"] = np.array([[1, 1, 0, 1, 0, 0, 1, 1, 1]], np.uint8)
    imgs["col_9x1"] = np.array([[1, 1, 0, 1, 0, 0, 1, 1, 1]], np.uint8).T.copy()
    return imgs


def int8_images():
    return {
        "i8_negative": np.array([[0, -1, 0], [0, 0, 0], [1, 0, -128]], np.int8),
        "i8_mixed": np.array([[-5, -5, 0, 3], [0, 0, 0, 3], [127, 0, -1, -1]], np.int8),
    }


def sanity():
    img = np.array([[0, 0, 0, 0, 1], [1, 0, 0, 0, 0]], np.uint8)
    _, a = cv2.connectedComponentsWithAlgorithm(img, 8, cv2.CV_32S, cv2.CCL_WU)
    _, b = cv2.connectedComponentsWithAlgorithm(img, 8, cv2.CV_32S, cv2.CCL_SPAGHETTI)
    assert a.tolist() == [[0, 0, 0, 0, 1], [2, 0, 0, 0, 0]], a
    assert b.tolist() == [[0, 0, 0, 0, 2], [1, 0, 0, 0, 0]], b
    n, _, st, ce = cv2.connectedComponentsWithStatsWithAlgorithm(np.ones((3, 4), np.uint8), 8, cv2.CV_32S, cv2.CCL_WU)
    assert n == 2 and st[0].tolist() == [-1, 2147483647, 0, 0, 0] and np.isnan(ce[0]).all(), (st, ce)
    assert st[1].tolist() == [0, 0, 4, 3, 12] and ce[1].tolist() == [1.5, 1.0], (st, ce)
    for bad in (lambda: cv2.connectedComponents(np.zeros((3, 3, 3), np.uint8)),
                lambda: cv2.connectedComponents(np.zeros((3, 3), np.uint16)),
                lambda: cv2.connectedComponents(np.zeros((3, 3), np.uint8), connectivity=6),
                lambda: cv2.connectedComponents(np.zeros((3, 3), np.uint8), ltype=cv2.CV_8U)):
        try:
            bad()
        except cv2.error:
            continue
        raise AssertionError("expected cv2.error")


def main():
    sanity()
    g = Gen()
    for tag, img in structured().items():
        g.battery(tag, img)
    for tag, img in int8_images().items():
        g.battery(tag, img)

    sizes = [(1, 1), (1, 7), (7, 1), (2, 2), (3, 5), (17, 31), (64, 64)]
    densities = [50, 300, 500, 600, 900]
    seed = 1
    for (h, w) in sizes:
        for d in densities:
            img = synthetic(w, h, seed, d)
            g.battery(f"rand_{h}x{w}_d{d}", img, synth={"width": w, "height": h, "seed": seed, "permille": d})
            seed += 1

    g.battery("rand_101x203_d550", synthetic(203, 101, seed, 550),
              synth={"width": 203, "height": 101, "seed": seed, "permille": 550})
    seed += 1

    for (h, w), ds in (((101, 203), (100, 400, 700)), ((256, 257), (50, 300, 500, 600, 900))):
        for d in ds:
            g.battery(f"hash_{h}x{w}_d{d}", synthetic(w, h, seed, d),
                      synth={"width": w, "height": h, "seed": seed, "permille": d}, hashed=True)
            seed += 1

    for d in (300, 550):
        g.battery(f"hash_1023x1025_d{d}", synthetic(1025, 1023, seed, d),
                  synth={"width": 1025, "height": 1023, "seed": seed, "permille": d}, hashed=True,
                  ltypes=("s32",), algorithms=("default", "wu", "bbdt"))
        seed += 1

    assert set(SUSPECT_CASES) <= g.names, set(SUSPECT_CASES) - g.names
    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "cc_golden.json")
    with open(out, "w") as f:
        json.dump({"arrays": g.arrays, "cases": g.cases}, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases, {len(g.arrays)} arrays to {out}")


if __name__ == "__main__":
    main()
