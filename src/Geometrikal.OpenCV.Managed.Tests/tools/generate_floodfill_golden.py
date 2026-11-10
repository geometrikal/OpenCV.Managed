#!/usr/bin/env python3
"""Golden-data generator for the managed flood fill port.

Runs cv2 5.0.0 ``floodFill`` over a seeded battery and writes
``TestData/floodfill_golden.json``. ``ImgProc/FloodFillGoldenTests.cs`` replays each case.

Regenerate with (from the ``Geometrikal.OpenCV.Managed.Tests`` directory):
    PYTHONPATH=<cv2 5.0.0 build> python tools/generate_floodfill_golden.py

Battery and constraints:
 * cv2 must be 5.0.0, the version of the ported source. The script asserts the version.
 * Each case stores the input image, the input mask (mask cases only), the seed, the
   new value, loDiff, upDiff and the OpenCV flags, and the expected area, rectangle,
   output image and output mask (mask cases only).
 * Cases without a mask call the no-mask overload (``mask=None``).
 * Images cover binary masks with holes, gray and BGR 8U, 32S and 32F with one and
   three channels, NaN and infinite floats, and 32S values near the int range limits.
 * Arrays are stored once in a shared ``arrays`` list.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import json
import os

import cv2
import numpy as np

assert cv2.__version__ == "5.0.0", cv2.__version__
cv2.setNumThreads(1)

SUSPECT_CASES = {}

DTYPES = {"u8": np.uint8, "s32": np.int32, "f32": np.float32}
FIXED = cv2.FLOODFILL_FIXED_RANGE
MASK_ONLY = cv2.FLOODFILL_MASK_ONLY


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def dt_of(arr):
    for k, v in DTYPES.items():
        if arr.dtype == v:
            return k
    raise ValueError(arr.dtype)


def quad(v):
    v = list(v) if isinstance(v, (tuple, list)) else [v]
    return [float(x) for x in (v + [0, 0, 0, 0])[:4]]


class Gen:
    def __init__(self):
        self.cases = []
        self.names = set()
        self.arrays = []
        self.index = {}

    def ref(self, arr):
        arr = np.ascontiguousarray(arr)
        cn = 1 if arr.ndim == 2 else arr.shape[2]
        m = {"dtype": dt_of(arr), "width": int(arr.shape[1]), "height": int(arr.shape[0]), "channels": cn,
             "data": b64(arr)}
        key = (m["dtype"], m["width"], m["height"], cn, m["data"])
        if key not in self.index:
            self.index[key] = len(self.arrays)
            self.arrays.append(m)
        return self.index[key]

    def add(self, name, img, seed, new_val, lo=0, up=0, flags=4, mask=None, use_mask=False):
        assert name not in self.names, name
        self.names.add(name)
        work = img.copy()
        if use_mask:
            if mask is None:
                mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
            m_work = mask.copy()
        else:
            m_work = None
        area, out, m_out, rect = cv2.floodFill(work, m_work, seed, quad(new_val), quad(lo), quad(up), flags)
        case = {"name": name, "input": self.ref(img), "seed": list(seed), "newVal": quad(new_val),
                "loDiff": quad(lo), "upDiff": quad(up), "flags": int(flags), "useMask": use_mask,
                "area": int(area), "rect": [int(v) for v in rect], "output": self.ref(out),
                "suspect": name in SUSPECT_CASES}
        if case["suspect"]:
            case["suspectReason"] = SUSPECT_CASES[name]
        if use_mask:
            case["maskInput"] = self.ref(mask)
            case["maskOutput"] = self.ref(m_out)
        self.cases.append(case)
        return area


def hole_image(h, w, rng):
    """Random rectangular rings and blobs with holes on a zero background."""
    img = np.zeros((h, w), np.uint8)
    for _ in range(max(2, (h * w) // 120)):
        y0, x0 = int(rng.integers(0, h - 3)), int(rng.integers(0, w - 3))
        bh, bw = int(rng.integers(3, 9)), int(rng.integers(3, 9))
        img[y0:y0 + bh, x0:x0 + bw] = 255
        if bh > 3 and bw > 3 and rng.random() < 0.7:
            img[y0 + 1:y0 + bh - 1, x0 + 1:x0 + bw - 1] = 0
    return img


def smooth_gray(h, w, rng):
    ys, xs = np.mgrid[0:h, 0:w]
    base = (xs * 3 + ys * 2 + 40 * np.sin(xs / 5.0) + 30 * np.cos(ys / 4.0)).astype(np.int64)
    noise = rng.integers(0, 4, (h, w))
    return ((base + noise) % 256).astype(np.uint8)


def smooth_bgr(h, w, rng):
    return np.dstack([smooth_gray(h, w, rng), np.roll(smooth_gray(h, w, rng), 3, axis=1),
                      np.roll(smooth_gray(h, w, rng), 2, axis=0)])


def sanity():
    img = np.zeros((3, 4), np.uint8)
    m = np.full((5, 6), 7, np.uint8)
    m[1:4, 1:5] = 0
    area, _, m2, rect = cv2.floodFill(img, m, (0, 0), 9, 0, 0, 4 | (5 << 8))
    assert area == 12 and rect == (0, 0, 4, 3), (area, rect)
    assert (m2[0] == 1).all() and (m2[:, 0] == 1).all(), m2
    # The seed value equals the new value: OpenCV 5.0 still reports the component area.
    img = np.full((2, 3), 9, np.uint8)
    area, _, _, rect = cv2.floodFill(img, None, (1, 1), 9)
    assert area == 6 and rect == (0, 0, 3, 2), (area, rect)
    for bad in (lambda: cv2.floodFill(np.zeros((3, 3, 2), np.uint8), None, (0, 0), 1),
                lambda: cv2.floodFill(np.zeros((3, 3), np.uint16), None, (0, 0), 1),
                lambda: cv2.floodFill(np.zeros((3, 3), np.uint8), None, (3, 0), 1),
                lambda: cv2.floodFill(np.zeros((3, 3), np.uint8), None, (0, 0), 1, -1, 0),
                lambda: cv2.floodFill(np.zeros((3, 3), np.uint8), None, (0, 0), 1, 0, 0, 6)):
        try:
            bad()
        except cv2.error:
            continue
        raise AssertionError("expected cv2.error")


def battery(g):
    rng = np.random.default_rng(20261015)

    # Hole filling from a border seed: the primary use case.
    for (h, w) in ((24, 31), (40, 40), (63, 97)):
        img = hole_image(h, w, rng)
        img[0, 0] = 0
        img[h - 1, w - 1] = 0
        img[0, w - 1] = 0
        img[h - 1, 0] = 0
        for conn in (4, 8):
            t = f"holes_{h}x{w}_c{conn}"
            g.add(f"{t}_corner_255", img, (0, 0), 255, flags=conn)
            g.add(f"{t}_corner_128", img, (0, 0), 128, flags=conn)
            g.add(f"{t}_br_100", img, (w - 1, h - 1), 100, flags=conn)
            g.add(f"{t}_tr_mask", img, (w - 1, 0), 255, flags=conn, use_mask=True)
            g.add(f"{t}_bl_maskonly_255", img, (0, h - 1), 255, flags=conn | MASK_ONLY | (255 << 8), use_mask=True)
            g.add(f"{t}_corner_maskonly_nomask", img, (0, 0), 255, flags=conn | MASK_ONLY)
            fg = tuple(int(v) for v in np.argwhere(img == 255)[0][::-1])
            g.add(f"{t}_fg_seed_0", img, fg, 0, flags=conn)
            g.add(f"{t}_fg_seed_mask7", img, fg, 50, flags=conn | (7 << 8), use_mask=True)
            g.add(f"{t}_edge_mid", img, (w // 2, 0), 77, flags=conn)
            g.add(f"{t}_same_value", img, (0, 0), 0, flags=conn)

    # Rings with one missing corner pixel: the hole leaks with 8-connectivity only.
    leak = np.zeros((19, 23), np.uint8)
    for (y0, x0) in ((2, 2), (2, 12), (10, 5), (10, 15)):
        leak[y0:y0 + 6, x0:x0 + 6] = 255
        leak[y0 + 1:y0 + 5, x0 + 1:x0 + 5] = 0
        leak[y0 + (5 if x0 % 2 else 0), x0] = 0
    for conn in (4, 8):
        g.add(f"diag_leak_c{conn}", leak, (0, 0), 255, flags=conn)
        g.add(f"diag_leak_c{conn}_mask", leak, (0, 0), 255, flags=conn | MASK_ONLY | (255 << 8), use_mask=True)
        g.add(f"diag_leak_s32_c{conn}", leak.astype(np.int32), (0, 0), 255, flags=conn)

    # Gradient fills on gray 8U.
    for (h, w) in ((32, 45), (57, 64)):
        img = smooth_gray(h, w, rng)
        seeds = [(w // 2, h // 2), (0, 0), (w - 1, h // 3), (w // 3, h - 1)]
        for conn in (4, 8):
            for (lo, up) in ((0, 0), (2, 3), (5, 5), (0, 10), (12, 1), (2.7, 3.9), (300, 0)):
                for fixed in (False, True):
                    for si, s in enumerate(seeds[:2] if (lo, up) != (5, 5) else seeds):
                        fl = conn | (FIXED if fixed else 0)
                        name = f"gray_{h}x{w}_c{conn}_lo{lo}_up{up}{'_fixed' if fixed else ''}_s{si}"
                        g.add(name, img, s, 255 if si % 2 == 0 else 0, lo, up, fl)
            g.add(f"gray_{h}x{w}_c{conn}_mask_grad", img, seeds[0], 200, 4, 4, conn | (3 << 8), use_mask=True)
            g.add(f"gray_{h}x{w}_c{conn}_maskonly_grad", img, seeds[0], 200, 4, 4, conn | MASK_ONLY, use_mask=True)
            g.add(f"gray_{h}x{w}_c{conn}_maskonly_fixed", img, seeds[0], 200, 9, 9, conn | MASK_ONLY | FIXED | (128 << 8),
                  use_mask=True)

    # Masks with obstacles and a masked seed.
    img = smooth_gray(30, 37, rng)
    mask = np.zeros((32, 39), np.uint8)
    mask[10, 3:30] = 1
    mask[3:25, 20] = 9
    mask[5, 5] = 255
    for conn in (4, 8):
        g.add(f"obstacles_c{conn}", img, (2, 2), 255, 6, 6, conn, mask=mask, use_mask=True)
        g.add(f"obstacles_c{conn}_fixed", img, (30, 20), 255, 20, 20, conn | FIXED | (42 << 8), mask=mask,
              use_mask=True)
        g.add(f"obstacles_c{conn}_masked_seed", img, (4, 4), 255, 6, 6, conn, mask=mask, use_mask=True)
        g.add(f"obstacles_c{conn}_eq", img, (2, 2), 255, 0, 0, conn, mask=mask, use_mask=True)
    ring = np.zeros((12, 12), np.uint8)
    ring[0, :] = ring[-1, :] = ring[:, 0] = ring[:, -1] = 255
    mask = np.zeros((14, 14), np.uint8)
    mask[0, :] = 5  # Frame values are replaced with 1 by OpenCV.
    mask[:, 13] = 0
    g.add("frame_rewritten", ring, (5, 5), 99, 0, 0, 4, mask=mask, use_mask=True)

    # BGR 8U.
    for (h, w) in ((29, 33),):
        img = smooth_bgr(h, w, rng)
        for conn in (4, 8):
            for (lo, up) in (((0, 0, 0), (0, 0, 0)), ((3, 4, 5), (5, 4, 3)), ((8, 0, 8), (0, 8, 8)),
                             ((40, 40, 40), (40, 40, 40))):
                for fixed in (False, True):
                    fl = conn | (FIXED if fixed else 0)
                    name = f"bgr_{h}x{w}_c{conn}_lo{'-'.join(map(str, lo))}_up{'-'.join(map(str, up))}" \
                           f"{'_fixed' if fixed else ''}"
                    g.add(name, img, (w // 2, h // 2), (255, 0, 128), lo, up, fl)
            g.add(f"bgr_{h}x{w}_c{conn}_mask", img, (1, 1), (1, 2, 3), (6, 6, 6), (6, 6, 6), conn | (11 << 8),
                  use_mask=True)
        blocks = np.kron(rng.integers(0, 3, (6, 7, 3)), np.ones((4, 4, 1))).astype(np.uint8) * 100
        g.add("bgr_blocks_eq_c4", blocks, (0, 0), (7, 7, 7), flags=4)
        g.add("bgr_blocks_eq_c8", blocks, (5, 5), (7, 8, 9), flags=8)
        g.add("bgr_blocks_same", blocks, (0, 0), tuple(int(v) for v in blocks[0, 0]), flags=4)

    # 32S.
    base = smooth_gray(26, 35, rng).astype(np.int32) * 1000 - 100000
    lim = base.copy()
    lim[5:9, 5:20] = np.iinfo(np.int32).max
    lim[12:15, 2:30] = np.iinfo(np.int32).min
    lim[20, :] = -1
    for conn in (4, 8):
        for (lo, up) in ((0, 0), (2000, 3000), (5500, 0), (1e10, 1e10), (2.5, 2.5)):
            for fixed in (False, True):
                fl = conn | (FIXED if fixed else 0)
                g.add(f"s32_c{conn}_lo{lo}_up{up}{'_fixed' if fixed else ''}", base, (17, 13), 7, lo, up, fl)
        g.add(f"s32_limits_c{conn}_wrap", lim, (6, 6), -5, 2, 2, conn)
        g.add(f"s32_limits_c{conn}_wrap_fixed", lim, (6, 6), -5, 2, 2, conn | FIXED)
        g.add(f"s32_limits_c{conn}_eq", lim, (2, 13), 3e9, 0, 0, conn)
        g.add(f"s32_c{conn}_mask", base, (0, 0), 1, 4000, 4000, conn | (200 << 8), use_mask=True)
    base3 = np.dstack([base, np.roll(base, 4, axis=1), base // 3]).astype(np.int32)
    for conn in (4, 8):
        g.add(f"s32c3_c{conn}_eq", base3, (10, 10), (1, 2, 3), flags=conn)
        g.add(f"s32c3_c{conn}_grad", base3, (10, 10), (1, 2, 3), (3000, 2000, 1000), (3000, 4000, 1000), conn)
        g.add(f"s32c3_c{conn}_fixed", base3, (10, 10), (1, 2, 3), (9000, 9000, 9000), (9000, 9000, 9000),
              conn | FIXED)

    # 32F.
    fimg = smooth_gray(27, 31, rng).astype(np.float32) * np.float32(0.37)
    special = fimg.copy()
    special[4, 3:12] = np.nan
    special[8:10, 8:20] = np.inf
    special[15, :] = -np.inf
    special[20, 5:25] = -0.0
    special[21, 5:25] = 0.0
    for conn in (4, 8):
        for (lo, up) in ((0, 0), (0.75, 1.1), (1.5, 0), (0.37, 0.37), (1e30, 1e30)):
            for fixed in (False, True):
                fl = conn | (FIXED if fixed else 0)
                g.add(f"f32_c{conn}_lo{lo}_up{up}{'_fixed' if fixed else ''}", fimg, (15, 13), -1.25, lo, up, fl)
        g.add(f"f32_special_c{conn}_nan_seed", special, (5, 4), 3.5, 1, 1, conn)
        g.add(f"f32_special_c{conn}_inf_seed", special, (10, 8), 3.5, 0, 0, conn)
        g.add(f"f32_special_c{conn}_inf_wide", special, (10, 8), 3.5, 1e30, 1e30, conn)
        g.add(f"f32_special_c{conn}_zero", special, (10, 20), 9.0, 0, 0, conn)
        g.add(f"f32_special_c{conn}_grad", special, (0, 0), 9.0, 2, 2, conn)
        g.add(f"f32_special_c{conn}_mask", special, (0, 0), 9.0, 2, 2, conn | MASK_ONLY, use_mask=True)
    f3 = np.dstack([fimg, fimg[::-1].copy(), fimg[:, ::-1].copy()]).astype(np.float32)
    for conn in (4, 8):
        g.add(f"f32c3_c{conn}_eq", f3, (3, 3), (1.0, 2.0, 3.0), flags=conn)
        g.add(f"f32c3_c{conn}_grad", f3, (3, 3), (1.0, 2.0, 3.0), (1, 1, 1), (2, 2, 2), conn)
        g.add(f"f32c3_c{conn}_fixed", f3, (3, 3), (1.0, 2.0, 3.0), (5, 5, 5), (5, 5, 5), conn | FIXED)

    # Small and degenerate sizes.
    g.add("one_pixel", np.array([[3]], np.uint8), (0, 0), 9)
    g.add("one_pixel_mask", np.array([[3]], np.uint8), (0, 0), 9, flags=8, use_mask=True)
    g.add("one_pixel_f32_same", np.array([[2.5]], np.float32), (0, 0), 2.5)
    row = np.array([[1, 1, 2, 1, 1, 1, 3, 1]], np.uint8)
    g.add("row_eq", row, (4, 0), 7)
    g.add("row_grad", row, (4, 0), 7, 1, 1, 8)
    g.add("col_eq", row.T.copy(), (0, 4), 7)
    g.add("col_grad_fixed", row.T.copy(), (0, 0), 7, 1, 1, 4 | FIXED)
    g.add("conn0_means4", hole_image(20, 20, rng), (0, 0), 50, flags=0)

    # Large random regions (deep segment stack).
    big = (rng.random((131, 257)) < 0.42).astype(np.uint8) * 255
    big[0, 0] = 0
    for conn in (4, 8):
        g.add(f"big_c{conn}", big, (0, 0), 128, flags=conn)
        g.add(f"big_c{conn}_mask", big, (0, 0), 128, flags=conn | (2 << 8), use_mask=True)
    bigg = smooth_gray(131, 257, rng)
    for conn in (4, 8):
        g.add(f"biggray_c{conn}", bigg, (128, 65), 0, 3, 3, conn)
        g.add(f"biggray_c{conn}_fixed", bigg, (128, 65), 0, 40, 40, conn | FIXED)


def main():
    sanity()
    g = Gen()
    battery(g)
    assert set(SUSPECT_CASES) <= g.names, set(SUSPECT_CASES) - g.names
    assert sum(1 for c in g.cases if c["area"] > 100) > 40
    out = os.path.join(os.path.dirname(__file__), "..", "TestData", "floodfill_golden.json")
    with open(out, "w") as f:
        json.dump({"arrays": g.arrays, "cases": g.cases}, f, separators=(",", ":"))
    print(f"wrote {len(g.cases)} cases, {len(g.arrays)} arrays to {out}")


if __name__ == "__main__":
    main()
