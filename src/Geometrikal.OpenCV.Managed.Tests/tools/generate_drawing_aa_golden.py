#!/usr/bin/env python3
"""Golden-data generator for the anti-aliased drawing port.

Runs cv2 5.0.0 drawing primitives with LINE_AA (and the LINE_AA fallback paths)
and writes the cases to ``TestData/drawing_aa_golden.json``. The C# runner
``ImgProc/DrawingAntiAliasedGoldenTests.cs`` replays each case through the
managed ``Drawing`` API and compares the raw pixel buffers byte for byte.

Each case starts from a background: ``zero`` or ``pattern``. The pattern value
at (y, x, c) is ``(x*7 + y*13 + c*29 + 11) & 255``. The anti-aliased blend reads
the background, so the pattern cases test the blend arithmetic.

Buffers up to ``INLINE_LIMIT`` bytes are stored as base64. Larger buffers are
stored as the SHA-256 of the raw buffer.

The file also holds ``fontScale`` cases for ``getFontScaleFromHeight``.

Regenerate from the ``Geometrikal.OpenCV.Managed.Tests`` directory:
    PYTHONPATH=<cv2 5.0.0 build> python tools/generate_drawing_aa_golden.py

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import hashlib
import json
import os

import cv2
import numpy as np

assert cv2.__version__ == "5.0.0", cv2.__version__

AA = cv2.LINE_AA
INLINE_LIMIT = 4096
DTYPES = {"u8": np.uint8, "u16": np.uint16, "f32": np.float32}


def background(kind, w, h, ch, dtype):
    if kind == "zero":
        img = np.zeros((h, w, ch), DTYPES[dtype])
    else:
        y, x, c = np.meshgrid(np.arange(h), np.arange(w), np.arange(ch), indexing="ij")
        img = ((x * 7 + y * 13 + c * 29 + 11) & 255).astype(DTYPES[dtype])
    return img if ch > 1 else img[:, :, 0].copy()


def color_arg(color, ch):
    return tuple(float(v) for v in color[:ch]) + (0.0,) * (4 - ch)


def pts_np(p):
    return np.array(p, np.int32).reshape(-1, 1, 2)


def run(op, img, a, ch):
    col = color_arg(a["color"], ch)
    if op == "line":
        cv2.line(img, tuple(a["p1"]), tuple(a["p2"]), col, a["thickness"], a["lineType"], a["shift"])
    elif op == "rectangle":
        cv2.rectangle(img, tuple(a["p1"]), tuple(a["p2"]), col, a["thickness"], a["lineType"], a["shift"])
    elif op == "rectangle_rect":
        cv2.rectangle(img, tuple(a["rect"]), col, a["thickness"], a["lineType"], a["shift"])
    elif op == "circle":
        cv2.circle(img, tuple(a["center"]), a["radius"], col, a["thickness"], a["lineType"], a["shift"])
    elif op == "ellipse":
        cv2.ellipse(img, tuple(a["center"]), tuple(a["axes"]), a["angle"], a["start"], a["end"],
                    col, a["thickness"], a["lineType"], a["shift"])
    elif op == "fillConvexPoly":
        cv2.fillConvexPoly(img, pts_np(a["points"]), col, a["lineType"], a["shift"])
    elif op == "fillPoly":
        cv2.fillPoly(img, [pts_np(p) for p in a["polys"]], col, a["lineType"], a["shift"], tuple(a["offset"]))
    elif op == "polylines":
        cv2.polylines(img, [pts_np(p) for p in a["polys"]], a["closed"], col, a["thickness"],
                      a["lineType"], a["shift"])
    elif op == "drawContours":
        cv2.drawContours(img, [pts_np(p) for p in a["polys"]], a["index"], col, a["thickness"],
                         a["lineType"], None, 2147483647, tuple(a["offset"]))
    elif op == "arrowedLine":
        cv2.arrowedLine(img, tuple(a["p1"]), tuple(a["p2"]), col, a["thickness"], a["lineType"],
                        a["shift"], a["tip"])
    elif op == "drawMarker":
        cv2.drawMarker(img, tuple(a["position"]), col, a["markerType"], a["markerSize"],
                       a["thickness"], a["lineType"])
    else:
        raise ValueError(op)
    return img


class Battery:
    def __init__(self):
        self.cases = []
        self.names = set()

    def add(self, name, op, args, w=24, h=20, ch=1, dtype="u8", bg="pattern"):
        assert name not in self.names, name
        self.names.add(name)
        img = background(bg, w, h, ch, dtype)
        img = run(op, img, args, ch)
        raw = np.ascontiguousarray(img).tobytes()
        case = {"name": name, "op": op, "width": w, "height": h, "channels": ch, "dtype": dtype,
                "background": bg, "args": args}
        if len(raw) <= INLINE_LIMIT:
            case["expected"] = base64.b64encode(raw).decode("ascii")
        else:
            case["sha256"] = hashlib.sha256(raw).hexdigest()
        self.cases.append(case)


def build():
    b = Battery()
    rng = np.random.default_rng(20261016)
    white = [255, 200, 120, 60]
    chans = [1, 3, 4]

    # Lines: every octant, thickness 1..10, shift 0..4, clipping.
    dirs = [(17, 3), (3, 17), (-15, 6), (-4, -16), (12, -12), (0, 14), (14, 0), (-9, 9), (1, 1), (0, 0)]
    for i, (dx, dy) in enumerate(dirs):
        for ch in chans:
            for t in (1, 2, 3):
                p1 = [11, 10]
                b.add(f"line_dir{i}_c{ch}_t{t}", "line",
                      dict(p1=p1, p2=[p1[0] + dx, p1[1] + dy], color=white, thickness=t, lineType=AA, shift=0),
                      ch=ch)
    for t in range(1, 11):
        for bg in ("zero", "pattern"):
            b.add(f"line_thick_t{t}_{bg}", "line",
                  dict(p1=[4, 6], p2=[36, 25], color=[230, 90, 40, 255], thickness=t, lineType=AA, shift=0),
                  w=40, h=32, ch=3, bg=bg)
    for s in range(0, 5):
        for t in (1, 4):
            b.add(f"line_shift{s}_t{t}", "line",
                  dict(p1=[5 << s | (s and 1), 3 << s], p2=[(30 << s) + (1 << s) // 3, (21 << s) + s],
                       color=white, thickness=t, lineType=AA, shift=s), w=36, h=28)
    b.add("line_shift16", "line",
          dict(p1=[3 << 16 | 12345, 2 << 16 | 40000], p2=[20 << 16 | 777, 17 << 16 | 65000],
               color=white, thickness=1, lineType=AA, shift=16))
    for k, (p1, p2) in enumerate([([-10, 5], [40, 12]), ([5, -30], [12, 50]), ([-5, -5], [-1, 30]),
                                  ([30, 30], [50, 40]), ([-100, -100], [200, 150]), ([23, 0], [23, 19])]):
        for t in (1, 3, 8):
            b.add(f"line_clip{k}_t{t}", "line",
                  dict(p1=p1, p2=p2, color=white, thickness=t, lineType=AA, shift=0), ch=3)
    for k in range(40):
        p1 = rng.integers(-8, 40, 2).tolist()
        p2 = rng.integers(-8, 40, 2).tolist()
        t = int(rng.integers(1, 11))
        s = int(rng.integers(0, 3))
        ch = int(rng.choice(chans))
        col = rng.integers(0, 256, 4).tolist()
        b.add(f"line_rand{k}", "line",
              dict(p1=[v << s for v in p1], p2=[v << s for v in p2], color=col, thickness=t, lineType=AA, shift=s),
              w=32, h=30, ch=ch)

    # Fallbacks: 2-channel bytes and non-byte depths.
    for t in (1, 3):
        b.add(f"line_c2_t{t}", "line", dict(p1=[2, 3], p2=[21, 15], color=white, thickness=t, lineType=AA, shift=0),
              ch=2)
        b.add(f"circle_c2_t{t}", "circle", dict(center=[12, 10], radius=7, color=white, thickness=t, lineType=AA,
                                                 shift=0), ch=2)
    b.add("fillpoly_c2", "fillPoly", dict(polys=[[[2, 2], [20, 4], [12, 18]]], color=white, lineType=AA, shift=0,
                                          offset=[0, 0]), ch=2)
    for dt in ("u16", "f32"):
        b.add(f"line_{dt}_t1", "line", dict(p1=[2, 3], p2=[21, 15], color=[1000, 2, 3, 4], thickness=1, lineType=AA,
                                             shift=0), dtype=dt, bg="zero")
        b.add(f"line_{dt}_t4", "line", dict(p1=[2, 3], p2=[21, 15], color=[1000, 2, 3, 4], thickness=4, lineType=AA,
                                             shift=0), ch=3, dtype=dt, bg="zero")
        b.add(f"circle_{dt}", "circle", dict(center=[12, 10], radius=6, color=[500, 2, 3, 4], thickness=2,
                                             lineType=AA, shift=0), dtype=dt, bg="zero")

    # Circles.
    for r in (0, 1, 2, 3, 5, 9, 14, 30):
        for t in (1, 2, 5, -1):
            b.add(f"circle_r{r}_t{t}", "circle",
                  dict(center=[16, 15], radius=r, color=white, thickness=t, lineType=AA, shift=0), w=32, h=30, ch=3)
    for s in (1, 3):
        b.add(f"circle_shift{s}", "circle",
              dict(center=[(16 << s) + 1, (15 << s) + 1], radius=(9 << s) + 1, color=white, thickness=2,
                   lineType=AA, shift=s), w=32, h=30)
    for ch in chans:
        b.add(f"circle_clip_c{ch}", "circle",
              dict(center=[2, 28], radius=11, color=white, thickness=3, lineType=AA, shift=0), w=32, h=30, ch=ch)

    # Ellipses.
    for k, (axes, ang, st, en, t) in enumerate([([12, 6], 0, 0, 360, 1), ([12, 6], 30, 0, 360, 2),
                                               ([12, 6], 45, 20, 250, 1), ([12, 6], 100, -30, 120, 4),
                                               ([3, 9], 15, 0, 360, -1), ([12, 6], 60, 10, 200, -1),
                                               ([0, 0], 0, 0, 360, 1), ([20, 2], 170, 0, 360, 3)]):
        b.add(f"ellipse_{k}", "ellipse",
              dict(center=[16, 15], axes=axes, angle=ang, start=st, end=en, color=white, thickness=t, lineType=AA,
                   shift=0), w=34, h=30, ch=3)
    b.add("ellipse_shift2", "ellipse",
          dict(center=[65, 61], axes=[45, 23], angle=33, start=0, end=360, color=white, thickness=1, lineType=AA,
               shift=2), w=34, h=30)

    # Rectangles.
    for t in (1, 3, -1):
        b.add(f"rect_t{t}", "rectangle",
              dict(p1=[3, 4], p2=[20, 15], color=white, thickness=t, lineType=AA, shift=0), ch=4)
        b.add(f"rectrect_t{t}", "rectangle_rect",
              dict(rect=[-3, 5, 30, 10], color=white, thickness=t, lineType=AA, shift=0), ch=3)
    b.add("rect_shift2", "rectangle",
          dict(p1=[13, 17], p2=[81, 61], color=white, thickness=1, lineType=AA, shift=2))

    # Polygons.
    tri = [[3, 2], [21, 6], [9, 18]]
    quad = [[2, 3], [22, 1], [20, 17], [4, 16]]
    star = [[12, 1], [15, 8], [23, 8], [17, 13], [19, 19], [12, 15], [5, 19], [7, 13], [1, 8], [9, 8]]
    clip = [[-6, -4], [30, 3], [12, 26]]
    for nm, poly in (("tri", tri), ("quad", quad), ("clip", clip)):
        for ch in chans:
            b.add(f"fillconvex_{nm}_c{ch}", "fillConvexPoly",
                  dict(points=poly, color=white, lineType=AA, shift=0), ch=ch)
    b.add("fillconvex_shift3", "fillConvexPoly",
          dict(points=[[v * 8 + 3 for v in p] for p in quad], color=white, lineType=AA, shift=3))
    for nm, polys in (("tri", [tri]), ("star", [star]), ("clip", [clip]), ("nested", [quad, [[8, 6], [15, 7], [11, 12]]])):
        for ch in chans:
            b.add(f"fillpoly_{nm}_c{ch}", "fillPoly",
                  dict(polys=polys, color=white, lineType=AA, shift=0, offset=[0, 0]), ch=ch)
    b.add("fillpoly_shift2_offset", "fillPoly",
          dict(polys=[[[v * 4 + 1 for v in p] for p in star]], color=white, lineType=AA, shift=2, offset=[1, -1]))
    for t in (0, 1, 2, 6):
        for closed in (False, True):
            b.add(f"polylines_t{t}_{'closed' if closed else 'open'}", "polylines",
                  dict(polys=[star], closed=closed, color=white, thickness=t, lineType=AA, shift=0), ch=3)
    b.add("polylines_shift1", "polylines",
          dict(polys=[[[v * 2 + 1 for v in p] for p in star]], closed=True, color=white, thickness=2, lineType=AA,
               shift=1))
    for t in (1, 3, -1):
        b.add(f"contours_t{t}", "drawContours",
              dict(polys=[star, tri], index=-1, color=white, thickness=t, lineType=AA, offset=[1, 0]), ch=3)

    # Arrowed lines and markers.
    for k, (p1, p2) in enumerate([([2, 2], [28, 20]), ([28, 3], [4, 25]), ([15, 27], [15, 3])]):
        for lt in (cv2.LINE_8, AA):
            b.add(f"arrow{k}_lt{lt}", "arrowedLine",
                  dict(p1=p1, p2=p2, color=white, thickness=2, lineType=lt, shift=0, tip=0.3), w=32, h=30, ch=3)
    for m in range(8):
        for lt in (cv2.LINE_8, AA):
            b.add(f"marker{m}_lt{lt}", "drawMarker",
                  dict(position=[15, 14], color=white, markerType=m, markerSize=17, thickness=1 + m % 3, lineType=lt),
                  w=32, h=30, ch=3)

    # Large canvases stored as hashes.
    lines = []
    for k in range(64):
        lines.append((rng.integers(-20, 276, 2).tolist(), rng.integers(-20, 276, 2).tolist()))
    for t in (1, 5):
        b.add(f"large_lines_t{t}", "polylines",
              dict(polys=[[p1, p2] for p1, p2 in lines], closed=False, color=[40, 200, 255, 0], thickness=t,
                   lineType=AA, shift=0), w=256, h=256, ch=3)
    b.add("large_circle", "circle",
          dict(center=[130, 120], radius=100, color=[250, 10, 100, 0], thickness=3, lineType=AA, shift=0),
          w=256, h=256, ch=3)
    b.add("large_fillpoly", "fillPoly",
          dict(polys=[[[v * 10 + 5 for v in p] for p in star]], color=[30, 60, 90, 0], lineType=AA, shift=0,
               offset=[0, 0]), w=256, h=256, ch=4)
    return b.cases


def font_scale_cases():
    out = []
    for face in list(range(8)) + [16 + f for f in range(8)]:
        for height in (1, 10, 23, 100):
            for thick in (1, 2):
                out.append({"fontFace": face, "pixelHeight": height, "thickness": thick,
                            "expected": cv2.getFontScaleFromHeight(face, height, thick)})
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "TestData", "drawing_aa_golden.json")
    data = {"opencv": cv2.__version__, "cases": build(), "fontScale": font_scale_cases()}
    with open(path, "w", encoding="ascii") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"wrote {len(data['cases'])} cases to {os.path.normpath(path)}")


if __name__ == "__main__":
    main()
