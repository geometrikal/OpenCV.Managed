#!/usr/bin/env python3
"""Golden-data generator for the managed drawing (non-anti-aliased) port.

Runs the reference OpenCV (cv2 4.5.5) drawing primitives over a battery of
small images, then writes the inputs and the reference pixel buffers to
``TestData/drawing_golden.json``. The managed C# runner
(``ImgProc/DrawingGoldenTests.cs``) reads the SAME file, replays each case
through the managed ``Drawing`` API, and asserts the raw pixel buffers match
byte-for-byte.

Regenerate with:
    python tools/generate_drawing_golden.py
(run from the ``Geometrikal.OpenCV.Managed.Tests`` directory).

See ``tools/README.md`` for details.

KNOWN VERSION RISK (cv2 4.5.5 legacy vs the OpenCV 5 analytic port)
------------------------------------------------------------------
A few areas of OpenCV drawing changed between 4.5.5 and 5:
  * fillPoly / polylines / fillConvexPoly with vertices OUTSIDE the image
    (the clipped-endpoint correction in collectPolyEdges is newer than 4.5.5)
    -> such cases are named ``*_oob_*`` so they are trivial to mark suspect if
    they later conflict with the OpenCV 5 port.
  * Thick / LINE_4 / shifted circles and every ellipse take the EllipseEx path,
    whose rounding evolved -> those cases are named ``*_ellipsex_*``.
  * drawContours hierarchy / maxLevel walks changed semantics; goldens here use
    ONLY hierarchy=None (draw-all or single-index), which is version stable. The
    maxLevel semantics are covered by analytic C# tests (DrawContoursMaxLevelTests).
Cases listed in ``SUSPECT_CASES`` below are emitted with ``suspect: true`` and
excluded from the generic golden runner; each carries an adjudicated reason and
stays covered by pinned OpenCV-5 expectations in DrawingOpenCv5DifferenceTests.

OpenCV / cv2 is licensed under the Apache License 2.0.
"""

import base64
import json
import os
import sys

import cv2
import numpy as np

# Adjudicated cv2 4.5.5 vs OpenCV 5 conflicts (2026-07-28). A bit-exact Python
# transcription of OpenCV 5's drawing.cpp reproduced the managed port's output
# on all five cases and reproduced these 4.5.5 goldens after reverting exactly
# two upstream changes:
#   * commit 9fc556a83e (PR #27366): ThickLine pre-clips thick segments against
#     a margin-expanded rect (drawing.cpp:1651-1661) -> line_clip_thick_l8.
#   * commit a27a13921b (issue #26413): ThickLine's thin-line branch condition
#     changed from (line_type==1||line_type==4||shift==0) to
#     (line_type==1||line_type==8||shift==0) (drawing.cpp:1672) -> the four
#     EllipseEx-path cases below.
# The managed port matches OpenCV 5 by design; see DrawingOpenCv5DifferenceTests.
_THIN_LINE_SWAP = (
    "OpenCV 5 routes thin polygon segments differently after commit a27a13921b "
    "(LINE_4/LINE_8 swap fix, issue #26413); the port matches OpenCV 5. "
    "See DrawingOpenCv5DifferenceTests."
)
SUSPECT_CASES = {
    "line_clip_thick_l8":
        "OpenCV 5 pre-clips thick line segments against a margin-expanded rect "
        "(commit 9fc556a83e, PR #27366); cv2 4.5.5 rasterised from unclipped "
        "endpoints. The port matches OpenCV 5. See DrawingOpenCv5DifferenceTests.",
    "circle_ellipsex_line4_r7": _THIN_LINE_SWAP,
    "circle_ellipsex_shift2_r5": _THIN_LINE_SWAP,
    "ellipse_ellipsex_a0_t1": _THIN_LINE_SWAP,
    "ellipse_ellipsex_a30_t1": _THIN_LINE_SWAP,
}

# cv2 line-type / retrieval / approximation integer values match the managed
# C# enum values exactly (LineType.FourConnected=4, EightConnected=8;
# RetrievalMode External=0 List=1 CComp=2 Tree=3; ApproxMode None=1 Simple=2).
L4 = cv2.LINE_4     # 4
L8 = cv2.LINE_8     # 8
FILLED = cv2.FILLED  # -1


class Battery:
    def __init__(self):
        self.cases = []
        self._names = set()

    def _color_arg(self, color, channels):
        if channels == 1:
            return float(color[0])
        return tuple(float(v) for v in color[:channels])

    def _expected(self, img):
        # Raw row-major buffer, exactly how the managed Mat<T> lays out pixels.
        return base64.b64encode(np.ascontiguousarray(img).tobytes()).decode("ascii")

    def _add(self, name, img, op, args, channels, dtype):
        assert name not in self._names, f"duplicate case name: {name}"
        self._names.add(name)
        h, w = img.shape[0], img.shape[1]
        case = {
            "name": name,
            "width": int(w),
            "height": int(h),
            "channels": int(channels),
            "dtype": dtype,
            "op": op,
            "args": args,
            "background": 0,
            "expected": self._expected(img),
            "suspect": name in SUSPECT_CASES,
        }
        if name in SUSPECT_CASES:
            case["suspectReason"] = SUSPECT_CASES[name]
        self.cases.append(case)

    def _blank(self, w, h, channels, dtype):
        np_dtype = {"u8": np.uint8, "u16": np.uint16, "f32": np.float32}[dtype]
        if channels == 1:
            return np.zeros((h, w), np_dtype)
        return np.zeros((h, w, channels), np_dtype)

    # -- primitive adders ---------------------------------------------------

    def line(self, name, w, h, pt1, pt2, color=(255,), thickness=1,
             line_type=L8, shift=0, channels=1, dtype="u8"):
        img = self._blank(w, h, channels, dtype)
        cv2.line(img, tuple(pt1), tuple(pt2), self._color_arg(color, channels),
                 thickness, line_type, shift)
        self._add(name, img, "line", {
            "pt1": [int(pt1[0]), int(pt1[1])],
            "pt2": [int(pt2[0]), int(pt2[1])],
            "color": [float(v) for v in color],
            "thickness": int(thickness),
            "lineType": int(line_type),
            "shift": int(shift),
        }, channels, dtype)

    def rectangle(self, name, w, h, pt1, pt2, color=(255,), thickness=1,
                  line_type=L8, shift=0, channels=1, dtype="u8"):
        img = self._blank(w, h, channels, dtype)
        cv2.rectangle(img, tuple(pt1), tuple(pt2), self._color_arg(color, channels),
                      thickness, line_type, shift)
        self._add(name, img, "rectangle", {
            "pt1": [int(pt1[0]), int(pt1[1])],
            "pt2": [int(pt2[0]), int(pt2[1])],
            "color": [float(v) for v in color],
            "thickness": int(thickness),
            "lineType": int(line_type),
            "shift": int(shift),
        }, channels, dtype)

    def rect2(self, name, w, h, rect, color=(255,), thickness=1,
              line_type=L8, shift=0, channels=1, dtype="u8"):
        img = self._blank(w, h, channels, dtype)
        # cv2 supports the Rect-tuple overload directly (x, y, width, height).
        cv2.rectangle(img, tuple(rect), self._color_arg(color, channels),
                      thickness, line_type, shift)
        self._add(name, img, "rect2", {
            "rect": [int(v) for v in rect],
            "color": [float(v) for v in color],
            "thickness": int(thickness),
            "lineType": int(line_type),
            "shift": int(shift),
        }, channels, dtype)

    def circle(self, name, w, h, center, radius, color=(255,), thickness=1,
               line_type=L8, shift=0, channels=1, dtype="u8"):
        img = self._blank(w, h, channels, dtype)
        cv2.circle(img, tuple(center), radius, self._color_arg(color, channels),
                   thickness, line_type, shift)
        self._add(name, img, "circle", {
            "center": [int(center[0]), int(center[1])],
            "radius": int(radius),
            "color": [float(v) for v in color],
            "thickness": int(thickness),
            "lineType": int(line_type),
            "shift": int(shift),
        }, channels, dtype)

    def ellipse(self, name, w, h, center, axes, angle, start_angle, end_angle,
                color=(255,), thickness=1, line_type=L8, shift=0,
                channels=1, dtype="u8"):
        img = self._blank(w, h, channels, dtype)
        cv2.ellipse(img, tuple(center), tuple(axes), angle, start_angle, end_angle,
                    self._color_arg(color, channels), thickness, line_type, shift)
        self._add(name, img, "ellipse", {
            "center": [int(center[0]), int(center[1])],
            "axes": [int(axes[0]), int(axes[1])],
            "angle": float(angle),
            "startAngle": float(start_angle),
            "endAngle": float(end_angle),
            "color": [float(v) for v in color],
            "thickness": int(thickness),
            "lineType": int(line_type),
            "shift": int(shift),
        }, channels, dtype)

    def fill_convex_poly(self, name, w, h, points, color=(255,),
                         line_type=L8, shift=0, channels=1, dtype="u8"):
        img = self._blank(w, h, channels, dtype)
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillConvexPoly(img, pts, self._color_arg(color, channels),
                           line_type, shift)
        self._add(name, img, "fillConvexPoly", {
            "points": [[int(p[0]), int(p[1])] for p in points],
            "color": [float(v) for v in color],
            "lineType": int(line_type),
            "shift": int(shift),
        }, channels, dtype)

    def fill_poly(self, name, w, h, polygons, color=(255,), line_type=L8,
                  shift=0, offset=(0, 0), channels=1, dtype="u8"):
        img = self._blank(w, h, channels, dtype)
        pts = [np.array(poly, dtype=np.int32).reshape((-1, 1, 2)) for poly in polygons]
        cv2.fillPoly(img, pts, self._color_arg(color, channels), line_type,
                     shift, tuple(offset))
        self._add(name, img, "fillPoly", {
            "polygons": [[[int(p[0]), int(p[1])] for p in poly] for poly in polygons],
            "color": [float(v) for v in color],
            "lineType": int(line_type),
            "shift": int(shift),
            "offset": [int(offset[0]), int(offset[1])],
        }, channels, dtype)

    def polylines(self, name, w, h, lines, is_closed, color=(255,), thickness=1,
                  line_type=L8, shift=0, channels=1, dtype="u8"):
        img = self._blank(w, h, channels, dtype)
        pts = [np.array(line, dtype=np.int32).reshape((-1, 1, 2)) for line in lines]
        cv2.polylines(img, pts, is_closed, self._color_arg(color, channels),
                      thickness, line_type, shift)
        self._add(name, img, "polylines", {
            "polylines": [[[int(p[0]), int(p[1])] for p in line] for line in lines],
            "isClosed": bool(is_closed),
            "color": [float(v) for v in color],
            "thickness": int(thickness),
            "lineType": int(line_type),
            "shift": int(shift),
        }, channels, dtype)

    def draw_contours(self, name, mask, mode, method, contour_index,
                      color=(255,), thickness=1, line_type=L8, offset=(0, 0),
                      channels=1, dtype="u8"):
        """Draw contours found by cv2 on the mask. hierarchy is always None
        (version-stable draw-all / single-index selection). The C# runner
        re-runs OUR FindContours on the SAME mask to obtain the contours."""
        mask = np.ascontiguousarray(mask, dtype=np.uint8)
        rows = ["".join("1" if v else "0" for v in row) for row in mask]
        h, w = mask.shape
        src = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(src.copy(), mode, method)
        target = self._blank(w, h, channels, dtype)
        cv2.drawContours(target, contours, contour_index,
                         self._color_arg(color, channels), thickness, line_type,
                         None, 2147483647, tuple(offset))
        self._add(name, target, "drawContours", {
            "maskRows": rows,
            "mode": int(mode),
            "method": int(method),
            "contourIndex": int(contour_index),
            "color": [float(v) for v in color],
            "thickness": int(thickness),
            "lineType": int(line_type),
            "offset": [int(offset[0]), int(offset[1])],
        }, channels, dtype)


# ---------------------------------------------------------------------------
# Battery construction
# ---------------------------------------------------------------------------

def build():
    b = Battery()
    C3 = (10.0, 200.0, 32.0)

    # -- lines --------------------------------------------------------------
    b.line("line_horiz_t1_l8", 20, 20, (2, 10), (17, 10))
    b.line("line_vert_t1_l8", 20, 20, (10, 2), (10, 17))
    b.line("line_diag_t1_l8", 20, 20, (2, 2), (17, 17))
    b.line("line_diag_t1_l4", 20, 20, (2, 2), (17, 17), line_type=L4)
    b.line("line_steep_t1_l8", 20, 20, (5, 2), (9, 18))
    b.line("line_shallow_t1_l4", 20, 20, (2, 5), (18, 9), line_type=L4)
    b.line("line_horiz_t2_l8", 20, 20, (2, 10), (17, 10), thickness=2)
    b.line("line_diag_t3_l8", 20, 20, (2, 2), (17, 17), thickness=3)
    b.line("line_diag_t3_l4", 20, 20, (2, 2), (17, 17), thickness=3, line_type=L4)
    b.line("line_steep_t5_l8", 24, 24, (6, 3), (12, 20), thickness=5)
    b.line("line_clip_horiz_l8", 20, 20, (-6, 10), (26, 10))
    b.line("line_clip_diag_l8", 20, 20, (-4, -4), (24, 24))
    b.line("line_clip_thick_l8", 20, 20, (-4, 6), (25, 15), thickness=3)
    b.line("line_shift2_horiz", 20, 20, (8, 40), (68, 40), shift=2)
    b.line("line_shift2_diag", 20, 20, (8, 8), (68, 68), shift=2)
    b.line("line_degenerate_t1", 20, 20, (10, 10), (10, 10))
    b.line("line_degenerate_t3", 20, 20, (10, 10), (10, 10), thickness=3)

    # -- rectangles ---------------------------------------------------------
    b.rectangle("rect_outline_t1_l8", 20, 20, (3, 3), (15, 12))
    b.rectangle("rect_outline_t3_l8", 20, 20, (3, 3), (15, 12), thickness=3)
    b.rectangle("rect_filled_l8", 20, 20, (3, 3), (15, 12), thickness=FILLED)
    b.rectangle("rect_partial_outline_t1", 20, 20, (10, 10), (30, 26))
    b.rectangle("rect_partial_filled", 20, 20, (10, 10), (30, 26), thickness=FILLED)
    b.rectangle("rect_l4_outline_t1", 20, 20, (3, 3), (15, 12), line_type=L4)
    b.rect2("rect2_outline_t1_l8", 20, 20, (3, 3, 13, 10))
    b.rect2("rect2_filled_l8", 20, 20, (3, 3, 13, 10), thickness=FILLED)
    b.rect2("rect2_outline_t3_l8", 20, 20, (3, 3, 13, 10), thickness=3)
    # shift=1: coordinates are in 1/2-pixel units.
    b.rectangle("rect_shift1_outline", 20, 20, (6, 6), (30, 24), shift=1)
    b.rectangle("rect_shift1_filled", 20, 20, (6, 6), (30, 24), thickness=FILLED, shift=1)

    # -- circles (stable midpoint path: t=1, LINE_8, shift=0) ---------------
    for r in (0, 1, 3, 7):
        b.circle(f"circle_r{r}_t1", 20, 20, (10, 10), r)
    for r in range(0, 8):
        b.circle(f"circle_filled_r{r}", 20, 20, (10, 10), r, thickness=FILLED)
    b.circle("circle_center_outside", 20, 20, (24, 10), 8)
    b.circle("circle_center_outside_neg", 20, 20, (-2, 12), 6)
    # -- circles (EllipseEx path: thick / LINE_4 / shift) -------------------
    b.circle("circle_ellipsex_thick_r6_t3", 24, 24, (12, 12), 6, thickness=3)
    b.circle("circle_ellipsex_line4_r7", 24, 24, (12, 12), 7, line_type=L4)
    b.circle("circle_ellipsex_shift2_r5", 24, 24, (48, 48), 20, shift=2)
    b.circle("circle_ellipsex_thick_filled_r6", 24, 24, (12, 12), 6,
             thickness=3, line_type=L4)

    # -- ellipses (all EllipseEx) -------------------------------------------
    b.ellipse("ellipse_ellipsex_a0_t1", 28, 24, (14, 12), (8, 5), 0, 0, 360)
    b.ellipse("ellipse_ellipsex_a0_t2", 28, 24, (14, 12), (8, 5), 0, 0, 360,
              thickness=2)
    b.ellipse("ellipse_ellipsex_a0_filled", 28, 24, (14, 12), (8, 5), 0, 0, 360,
              thickness=FILLED)
    b.ellipse("ellipse_ellipsex_a30_t1", 28, 24, (14, 12), (8, 5), 30, 0, 360)
    b.ellipse("ellipse_ellipsex_a30_filled", 28, 24, (14, 12), (8, 5), 30, 0, 360,
              thickness=FILLED)
    b.ellipse("ellipse_ellipsex_arc0_90_filled", 28, 24, (14, 12), (8, 5), 0,
              0, 90, thickness=FILLED)

    # -- fillConvexPoly -----------------------------------------------------
    b.fill_convex_poly("fillconvex_triangle", 20, 20,
                       [(3, 3), (16, 6), (8, 17)])
    b.fill_convex_poly("fillconvex_quad", 20, 20,
                       [(4, 3), (16, 5), (15, 16), (5, 15)])
    b.fill_convex_poly("fillconvex_oob_triangle", 20, 20,
                       [(-3, 2), (22, 8), (9, 24)])

    # -- fillPoly -----------------------------------------------------------
    b.fill_poly("fillpoly_convex", 24, 24, [[(4, 4), (18, 6), (14, 20), (5, 16)]])
    b.fill_poly("fillpoly_concave", 24, 24,
                [[(3, 3), (20, 3), (20, 8), (9, 8), (9, 20), (3, 20)]])
    # 5-point self-intersecting star (even-odd rule).
    b.fill_poly("fillpoly_star", 30, 30,
                [[(15, 2), (19, 22), (3, 9), (27, 9), (11, 22)]])
    b.fill_poly("fillpoly_two_polys", 30, 20,
                [[(2, 2), (11, 2), (11, 15), (2, 15)],
                 [(16, 4), (27, 4), (27, 17), (16, 17)]])
    # polygon-with-hole: outer square + inner square -> XOR hole.
    b.fill_poly("fillpoly_hole", 24, 24,
                [[(3, 3), (20, 3), (20, 20), (3, 20)],
                 [(8, 8), (15, 8), (15, 15), (8, 15)]])
    b.fill_poly("fillpoly_offset", 24, 24,
                [[(2, 2), (12, 2), (12, 12), (2, 12)]], offset=(6, 5))
    b.fill_poly("fillpoly_shift1", 24, 24,
                [[(6, 6), (36, 8), (30, 40), (8, 34)]], shift=1)
    b.fill_poly("fillpoly_oob_convex", 24, 24,
                [[(-5, 5), (28, 3), (26, 21), (4, 26)]])

    # -- polylines ----------------------------------------------------------
    b.polylines("polylines_open_t1", 24, 24,
                [[(3, 3), (18, 6), (12, 18), (6, 14)]], is_closed=False)
    b.polylines("polylines_closed_t1", 24, 24,
                [[(3, 3), (18, 6), (12, 18), (6, 14)]], is_closed=True)
    b.polylines("polylines_open_t3", 24, 24,
                [[(3, 3), (18, 6), (12, 18), (6, 14)]], is_closed=False, thickness=3)
    b.polylines("polylines_closed_t3_l4", 24, 24,
                [[(3, 3), (18, 6), (12, 18), (6, 14)]], is_closed=True,
                thickness=3, line_type=L4)

    # -- drawContours (hierarchy=None; version-stable selection) ------------
    blob = np.zeros((20, 20), np.uint8)
    blob[4:15, 5:16] = 1
    b.draw_contours("drawcontours_blob_all_filled", blob, cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE, -1, thickness=FILLED)
    b.draw_contours("drawcontours_blob_all_t1", blob, cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_NONE, -1, thickness=1)
    b.draw_contours("drawcontours_blob_all_t2", blob, cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE, -1, thickness=2)

    rings = np.zeros((24, 24), np.uint8)
    cv2.rectangle(rings, (2, 2), (21, 21), 1, 1)
    cv2.rectangle(rings, (7, 7), (16, 16), 1, 1)
    b.draw_contours("drawcontours_rings_all_filled", rings, cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_SIMPLE, -1, thickness=FILLED)
    b.draw_contours("drawcontours_rings_all_t1", rings, cv2.RETR_TREE,
                    cv2.CHAIN_APPROX_NONE, -1, thickness=1)

    two = np.zeros((16, 26), np.uint8)
    two[2:8, 2:9] = 1
    two[6:13, 15:24] = 1
    b.draw_contours("drawcontours_two_single0_filled", two, cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_SIMPLE, 0, thickness=FILLED)
    b.draw_contours("drawcontours_two_single1_t2", two, cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_SIMPLE, 1, thickness=2)
    b.draw_contours("drawcontours_blob_offset_filled", blob, cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE, -1, thickness=FILLED, offset=(3, -2))

    # -- 3-channel colour ---------------------------------------------------
    b.line("line_c3_t2", 20, 20, (2, 3), (17, 16), color=C3, thickness=2, channels=3)
    b.rectangle("rect_c3_filled", 20, 20, (3, 3), (15, 12), color=C3,
                thickness=FILLED, channels=3)
    b.circle("circle_c3_filled_r6", 20, 20, (10, 10), 6, color=C3,
             thickness=FILLED, channels=3)
    b.fill_poly("fillpoly_c3_convex", 24, 24, [[(4, 4), (18, 6), (14, 20), (5, 16)]],
                color=C3, channels=3)
    b.polylines("polylines_c3_closed_t2", 24, 24,
                [[(3, 3), (18, 6), (12, 18), (6, 14)]], is_closed=True,
                color=C3, thickness=2, channels=3)
    b.draw_contours("drawcontours_c3_blob_filled", blob, cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE, -1, color=C3, thickness=FILLED,
                    channels=3)

    # -- other dtypes (pin ToRawColor saturate/cast) ------------------------
    b.line("line_f32_diag", 16, 16, (2, 2), (13, 13), color=(1.5,), dtype="f32")
    b.rectangle("rect_u16_filled", 16, 16, (2, 2), (12, 10), color=(5000.0,),
                thickness=FILLED, dtype="u16")

    return b


def sanity_checks(b):
    """A few analytic asserts so a broken cv2 install fails loudly here."""
    by_name = {c["name"]: c for c in b.cases}

    def buf(name):
        c = by_name[name]
        arr = np.frombuffer(base64.b64decode(c["expected"]),
                            dtype={"u8": np.uint8, "u16": np.uint16,
                                   "f32": np.float32}[c["dtype"]])
        return c, arr

    # 1px horizontal line (2,10)->(17,10): x = 2..17 inclusive = 16 pixels.
    c, arr = buf("line_horiz_t1_l8")
    assert int((arr != 0).sum()) == 16, ("1px hline pixel count",
                                         int((arr != 0).sum()))

    # filled rect (3,3)-(15,12): area = 13 * 10 = 130.
    c, arr = buf("rect_filled_l8")
    assert int((arr != 0).sum()) == 130, ("filled rect area", int((arr != 0).sum()))

    # rect2 filled equals the equivalent point-overload filled rect, area 130.
    c, arr = buf("rect2_filled_l8")
    assert int((arr != 0).sum()) == 130, ("rect2 filled area", int((arr != 0).sum()))

    # circle radius 0 -> exactly one pixel drawn.
    c, arr = buf("circle_r0_t1")
    assert int((arr != 0).sum()) == 1, ("circle r0 pixel count", int((arr != 0).sum()))

    # filled circle radius 0 -> exactly one pixel drawn.
    c, arr = buf("circle_filled_r0")
    assert int((arr != 0).sum()) == 1, ("filled circle r0", int((arr != 0).sum()))

    # degenerate 1px line -> single pixel.
    c, arr = buf("line_degenerate_t1")
    assert int((arr != 0).sum()) == 1, ("degenerate line", int((arr != 0).sum()))

    # float32 line writes the exact colour value 1.5.
    c, arr = buf("line_f32_diag")
    assert float(arr.max()) == 1.5, ("f32 line value", float(arr.max()))

    # uint16 filled rect writes 5000; area (12-2+1)*(10-2+1) = 11*9 = 99.
    c, arr = buf("rect_u16_filled")
    assert int(arr.max()) == 5000, ("u16 rect value", int(arr.max()))
    assert int((arr != 0).sum()) == 99, ("u16 rect area", int((arr != 0).sum()))

    # 3-channel filled rect: every drawn pixel carries the colour (10,200,32).
    c = by_name["rect_c3_filled"]
    a = np.frombuffer(base64.b64decode(c["expected"]), np.uint8).reshape(
        (c["height"], c["width"], 3))
    assert tuple(int(v) for v in a[7, 7]) == (10, 200, 32), a[7, 7]

    # filled convex triangle draws a non-trivial region strictly inside bounds.
    c, arr = buf("fillconvex_triangle")
    assert int((arr != 0).sum()) > 20, ("triangle area", int((arr != 0).sum()))

    # drawContours all/filled of a solid blob reproduces the solid blob area.
    c, arr = buf("drawcontours_blob_all_filled")
    assert int((arr != 0).sum()) == 11 * 11, ("blob fill area", int((arr != 0).sum()))

    print("sanity checks passed")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.normpath(os.path.join(here, "..", "TestData"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "drawing_golden.json")

    b = build()
    sanity_checks(b)

    names = [c["name"] for c in b.cases]
    assert len(names) == len(set(names)), "duplicate case names!"

    data = {"cases": b.cases}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))

    size = os.path.getsize(out_path)
    print(f"wrote {len(b.cases)} cases to {out_path}")
    print(f"file size: {size} bytes ({size / 1024:.1f} KiB)")
    if size > 2 * 1024 * 1024:
        print("WARNING: golden file exceeds 2 MiB", file=sys.stderr)


if __name__ == "__main__":
    main()
