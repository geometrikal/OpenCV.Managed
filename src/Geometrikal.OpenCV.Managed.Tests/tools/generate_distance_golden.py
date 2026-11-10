#!/usr/bin/env python3
"""Golden-data generator for the managed distance transform and watershed port.

Runs cv2 4.5.5 ``distanceTransform``, ``distanceTransformWithLabels`` and
``watershed`` over a seeded battery and writes ``TestData/distance_golden.json``.
``ImgProc/DistanceGoldenTests.cs`` replays each case.

Regenerate with:
    python tools/generate_distance_golden.py
(run from the ``Geometrikal.OpenCV.Managed.Tests`` directory).

Environment and battery constraints:
 * One thread, IPP and OpenCL off. With IPP on, cv2 uses IPP for the 5x5
   chamfer and for the precise transform (the precise selection depends on the
   thread count and the image size); those results differ from the C code by
   up to 6.5e-5. The port implements the C code only.
 * The port follows OpenCV 5.0. 4.5.5 clamps chamfer distances at 8192 and
   uses other "unreachable" values (images without zero pixels: chamfer 8192,
   precise 31622776; OpenCV 5.0: (UINT_MAX - w) / 65536 and 65536). Every
   image with a zero pixel has distances below 8192 pixels and sides below
   2896, so 4.5.5 and 5.0 agree. Images without zero pixels are listed in
   SUSPECT_CASES and pinned in DistanceOpenCv5DifferenceTests. The L1 CV_8U
   path did not change and keeps its no-zero cases.
 * watershed did not change between 4.5.5 and 5.0.
 * Arrays are stored once in a shared ``arrays`` list. Large cases store the
   SHA-256 of the output buffers instead of the buffers.

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

DIST_TYPES = {"l1": cv2.DIST_L1, "l2": cv2.DIST_L2, "c": cv2.DIST_C}
EXTRA_TYPES = {"l12": cv2.DIST_L12, "fair": cv2.DIST_FAIR, "user": cv2.DIST_USER}
LABEL_TYPES = {"ccomp": cv2.DIST_LABEL_CCOMP, "pixel": cv2.DIST_LABEL_PIXEL}
DTYPES = {"u8": np.uint8, "f32": np.float32, "s32": np.int32}
NO_ZERO_REASON = ("image without zero pixels: cv2 4.5.5 gives 8192 (chamfer) or 31622776 (precise); "
                  "OpenCV 5.0 gives (UINT_MAX - w) / 65536 or 65536")

SUSPECT_CASES = {}


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
        ch = 1 if arr.ndim == 2 else int(arr.shape[2])
        m = {"dtype": dt_of(arr), "width": int(arr.shape[1]), "height": int(arr.shape[0]), "channels": ch,
             "data": b64(arr)}
        key = (m["dtype"], m["width"], m["height"], ch, m["data"])
        if key not in self.index:
            self.index[key] = len(self.arrays)
            self.arrays.append(m)
        return self.index[key]

    def add(self, name, case, outputs, hashed):
        assert name not in self.names, name
        self.names.add(name)
        case = {"name": name, **case, "suspect": name in SUSPECT_CASES}
        if case["suspect"]:
            case["suspectReason"] = SUSPECT_CASES[name]
        for key, arr in outputs.items():
            if hashed:
                case[key + "Sha"] = sha(arr)
            else:
                case[key] = self.ref(arr)
        self.cases.append(case)

    def distance_battery(self, tag, img, hashed=False, extra_types=False):
        no_zero = not (img == 0).any()
        inp = self.ref(img)
        for tn, t in DIST_TYPES.items():
            for mask in (3, 5, 0):
                name = f"{tag}_dist_{tn}_m{mask}"
                if no_zero:
                    SUSPECT_CASES[name] = NO_ZERO_REASON
                out = cv2.distanceTransform(img, t, mask)
                assert out.dtype == np.float32
                self.add(name, {"op": "dist", "input": inp, "distanceType": t, "maskSize": mask},
                         {"dist": out}, hashed)
            for ln, lt in LABEL_TYPES.items():
                for mask in ((3, 0, 5) if tn == "l2" else (3,)):
                    name = f"{tag}_labels_{tn}_{ln}_m{mask}"
                    if no_zero:
                        SUSPECT_CASES[name] = NO_ZERO_REASON
                    d, lab = cv2.distanceTransformWithLabels(img, t, mask, labelType=lt)
                    assert d.dtype == np.float32 and lab.dtype == np.int32
                    self.add(name, {"op": "labels", "input": inp, "distanceType": t, "maskSize": mask,
                                    "labelType": lt}, {"dist": d, "labels": lab}, hashed)
        for mask in ((3, 5, 0) if extra_types else (3,)):
            out = cv2.distanceTransform(img, cv2.DIST_L1, mask, dstType=cv2.CV_8U)
            assert out.dtype == np.uint8
            self.add(f"{tag}_dist8u_l1_m{mask}", {"op": "dist8u", "input": inp, "distanceType": cv2.DIST_L1,
                                                  "maskSize": mask}, {"dist": out}, hashed)
        if extra_types:
            for tn, t in EXTRA_TYPES.items():
                name = f"{tag}_dist_{tn}_m0"
                if no_zero:
                    SUSPECT_CASES[name] = NO_ZERO_REASON
                out = cv2.distanceTransform(img, t, 0)
                self.add(name, {"op": "dist", "input": inp, "distanceType": t, "maskSize": 0},
                         {"dist": out}, hashed)

    def watershed_case(self, tag, img, markers, hashed=False):
        out = markers.copy()
        cv2.watershed(img, out)
        self.add(f"ws_{tag}", {"op": "watershed", "input": self.ref(img), "markers": self.ref(markers)},
                 {"labels": out}, hashed)


def random_mask(rng, h, w, zero_fraction, values="binary"):
    zero = rng.random((h, w)) < zero_fraction
    if values == "binary":
        fg = np.full((h, w), 255, np.uint8)
    else:
        fg = rng.integers(1, 256, (h, w), dtype=np.uint8)
    return np.where(zero, 0, fg).astype(np.uint8)


def structured():
    imgs = {}
    imgs["single_zero"] = np.zeros((1, 1), np.uint8)
    imgs["single_fg"] = np.full((1, 1), 9, np.uint8)
    imgs["all_zero_4x6"] = np.zeros((4, 6), np.uint8)
    imgs["no_zero_12x9"] = np.full((12, 9), 255, np.uint8)
    imgs["no_zero_3x40"] = np.full((3, 40), 7, np.uint8)
    r = np.full((1, 20), 1, np.uint8)
    r[0, 7] = 0
    imgs["row_1x20"] = r
    imgs["col_20x1"] = r.T.copy()
    c = np.full((300, 9), 1, np.uint8)
    c[0, 0] = 0
    imgs["tall_300x9_corner"] = c  # distances above 255 for CV_8U
    b = np.full((30, 41), 7, np.uint8)
    b[5, 3] = 0
    b[20, 3] = 0
    imgs["one_zero_column_30x41"] = b
    w = np.full((5, 200), 3, np.uint8)
    w[2, 0] = 0
    imgs["wide_5x200_left"] = w
    p = np.full((33, 35), 255, np.uint8)
    p[16, 17] = 0
    imgs["centre_point_33x35"] = p
    yy, xx = np.mgrid[0:24, 0:37]
    imgs["checker_24x37"] = (((yy // 3 + xx // 4) % 2) * 200).astype(np.uint8)
    ring = np.full((41, 43), 255, np.uint8)
    ring[:, 0] = 0
    ring[0, :] = 0
    ring[:, 42] = 0
    imgs["frame_41x43"] = ring
    d = np.full((29, 31), 1, np.uint8)
    for i in range(29):
        d[i, (i * 3) % 31] = 0
    imgs["diagonal_29x31"] = d
    tie = np.full((9, 9), 255, np.uint8)
    tie[0, 0] = tie[0, 8] = tie[8, 0] = tie[8, 8] = tie[4, 4] = 0
    imgs["equidistant_9x9"] = tie
    return imgs


def watershed_cases(g, rng):
    for t in range(36):
        h, w = (int(x) for x in rng.integers(3, 64, 2))
        if t % 9 == 0:
            h = 3
        if t % 9 == 4:
            w = 3
        img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        if t % 2:
            img = cv2.GaussianBlur(img, (0, 0), 2)
        if t % 5 == 0:
            img[:] = img[:, :, :1] // 64 * 64  # many equal differences
        mk = np.zeros((h, w), np.int32)
        for _ in range(int(rng.integers(0, 9))):
            mk[rng.integers(0, h), rng.integers(0, w)] = int(rng.choice([1, 2, 3, 7, 100000, -5, -1]))
        if t % 7 == 0:
            mk[rng.random((h, w)) < 0.05] = -3
        if t % 6 == 3:
            mk[rng.random((h, w)) < 0.2] = 5  # dense seeds
        g.watershed_case(f"random{t}_{h}x{w}", img, mk)

    z = np.zeros((10, 10, 3), np.uint8)
    g.watershed_case("no_markers", z, np.zeros((10, 10), np.int32))
    m = np.zeros((10, 10), np.int32)
    m[0, 5] = 4
    m[9, 2] = 6
    m[4, 0] = 8
    g.watershed_case("frame_markers_only", z, m)
    m = np.zeros((12, 15), np.int32)
    m[5, 6] = 1
    m[5, 7] = 2
    g.watershed_case("adjacent_markers", rng.integers(0, 256, (12, 15, 3), dtype=np.uint8), m)
    m = np.full((8, 11), -7, np.int32)
    m[3, 4] = 9
    g.watershed_case("negative_interior", rng.integers(0, 256, (8, 11, 3), dtype=np.uint8), m)
    for (h, w) in ((1, 9), (2, 9), (9, 1), (9, 2), (3, 3)):
        mk = np.full((h, w), 3, np.int32)
        g.watershed_case(f"tiny_{h}x{w}", np.zeros((h, w, 3), np.uint8), mk)
    yy, xx = np.mgrid[0:48, 0:64]
    grad = np.stack([(xx * 4) % 256, (yy * 5) % 256, ((xx + yy) * 2) % 256], axis=2).astype(np.uint8)
    m = np.zeros((48, 64), np.int32)
    m[10, 10] = 1
    m[40, 50] = 2
    m[24, 32] = 3
    g.watershed_case("gradient_three_seeds", grad, m)

    h, w = 257, 301
    img = cv2.GaussianBlur(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), (0, 0), 3)
    m = np.zeros((h, w), np.int32)
    for k in range(200):
        m[rng.integers(1, h - 1), rng.integers(1, w - 1)] = k + 1
    g.watershed_case("large_257x301_200_seeds", img, m, hashed=True)


def sanity():
    s = np.full((5, 5), 3, np.uint8)
    s[2, 2] = 0
    assert cv2.distanceTransform(s, cv2.DIST_L1, 3)[0, 0] == 4.0
    assert cv2.distanceTransform(s, cv2.DIST_C, 5)[0, 0] == 2.0
    assert cv2.distanceTransform(s, cv2.DIST_L2, 0)[0, 0] == np.float32(np.sqrt(8))
    assert cv2.distanceTransform(s, cv2.DIST_L2, 3, dstType=cv2.CV_8U).dtype == np.float32
    for bad in (lambda: cv2.distanceTransform(s, cv2.DIST_L2, 4),
                lambda: cv2.distanceTransform(s, cv2.DIST_L12, 3),
                lambda: cv2.distanceTransform(np.zeros((3, 3, 3), np.uint8), cv2.DIST_L2, 3),
                lambda: cv2.watershed(np.zeros((3, 3), np.uint8), np.zeros((3, 3), np.int32))):
        try:
            bad()
        except cv2.error:
            continue
        raise AssertionError("expected cv2.error")


def main():
    sanity()
    rng = np.random.default_rng(20261011)
    g = Gen()
    for tag, img in structured().items():
        g.distance_battery(tag, img, extra_types=tag in ("centre_point_33x35", "no_zero_12x9"))

    sizes = [(1, 7), (7, 1), (1, 45), (45, 1), (5, 9), (17, 31), (37, 53), (40, 31), (16, 16),
             (23, 64), (64, 67), (70, 131), (9, 130), (33, 32)]
    for (h, w) in sizes:
        for zf in (0.005, 0.02, 0.2, 0.5):
            img = random_mask(rng, h, w, zf, "binary" if zf < 0.1 else "random")
            if not (img == 0).any():
                img[rng.integers(0, h), rng.integers(0, w)] = 0
            g.distance_battery(f"rand_{h}x{w}_z{int(zf * 1000)}", img, extra_types=(h, w) == (37, 53))

    for (h, w, zf) in ((257, 301, 0.001), (301, 517, 0.02), (130, 1031, 0.0005)):
        img = random_mask(rng, h, w, zf, "random")
        g.distance_battery(f"hash_{h}x{w}_z{zf}", img, hashed=True)

    watershed_cases(g, rng)

    assert set(SUSPECT_CASES) <= g.names, set(SUSPECT_CASES) - g.names
    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "distance_golden.json")
    with open(out, "w") as f:
        json.dump({"arrays": g.arrays, "cases": g.cases}, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases ({len(SUSPECT_CASES)} suspect), {len(g.arrays)} arrays to {out}")


if __name__ == "__main__":
    main()
