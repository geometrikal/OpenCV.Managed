// This implementation is ported from the OpenCV drawing implementation.
//
// Original source:
//   modules/imgproc/src/drawing.cpp   (in this repo: src/OpenCV/modules/imgproc/src/drawing.cpp)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Diagnostics;

using Geometrikal.OpenCV.Managed.Core;

using static Geometrikal.OpenCV.Managed.ImgProc.Internal.DrawingConstants;

/// <summary>
/// Polygon fill primitives ported from OpenCV's <c>drawing.cpp</c>:
/// <see cref="FillConvexPoly{T}"/> (the two-edge convex-polygon scanline fill,
/// drawing.cpp:1095-1257), <see cref="CollectPolyEdges{T}"/> (edge collection for
/// the general polygon fill, drawing.cpp:1262-1334) and
/// <see cref="FillEdgeCollection{T}"/> (the active-edge-table scanline fill,
/// drawing.cpp:1347-1494).
///
/// <para>The <c>LINE_AA</c> branches use <see cref="AntiAliasedLine"/> for the
/// outline and the anti-aliased span rounding of the originals. All fixed-point arithmetic is intentionally unchecked (the project default),
/// matching the C-style wrap/truncation semantics of the source. Every routine
/// mutates <paramref name="img"/> in place; pixels outside the image are clipped
/// silently, and ROI targets write through to their parent matrix.</para>
/// </summary>
internal static class PolygonPrimitives
{
    /// <summary>
    /// Fills a convex polygon whose vertices <paramref name="v"/> are in
    /// <see cref="DrawingConstants.XY_SHIFT"/>-relative fixed-point (each vertex
    /// is scaled up by <c>XY_SHIFT - shift</c> internally). Ports OpenCV's
    /// <c>FillConvexPoly</c> (drawing.cpp:1095-1257): the polygon outline is drawn
    /// first (via <see cref="LinePrimitives.Line{T}"/> for <paramref name="shift"/>
    /// == 0, otherwise <see cref="LinePrimitives.Line2{T}"/>) while the bounding
    /// box is tracked, then the interior is filled with a two-edge scanline
    /// stepper. Mutates <paramref name="img"/> in place.
    /// </summary>
    /// <param name="img">Target image, mutated in place.</param>
    /// <param name="v">Convex polygon vertices (raw or fixed-point per <paramref name="shift"/>).</param>
    /// <param name="color">Raw per-channel fill colour (see <see cref="PixelOps.ToRawColor{T}"/>).</param>
    /// <param name="lineType">Outline line type: 4, 8 or 16 (<c>LINE_AA</c>).</param>
    /// <param name="shift">Number of sub-pixel fractional bits in <paramref name="v"/> (0..16).</param>
    public static void FillConvexPoly<T>(
        Mat<T> img,
        ReadOnlySpan<PointL> v,
        ReadOnlySpan<T> color,
        int lineType,
        int shift)
        where T : unmanaged
    {
        int npts = v.Length;

        int delta = (1 << shift) >> 1;
        int imin = 0;
        int edges = npts;

        bool antiAliased = lineType >= AntiAliasedLine.LineAa;
        int delta1 = antiAliased ? XY_ONE - 1 : XY_ONE >> 1;
        int delta2 = antiAliased ? 0 : XY_ONE >> 1;

        long p0x = v[npts - 1].X << (XY_SHIFT - shift);
        long p0y = v[npts - 1].Y << (XY_SHIFT - shift);

        Debug.Assert(0 <= shift && shift <= XY_SHIFT);

        long xmin = v[0].X, xmax = v[0].X;
        long ymin = v[0].Y, ymax = v[0].Y;

        for (int i = 0; i < npts; i++)
        {
            long px = v[i].X;
            long py = v[i].Y;

            if (py < ymin)
            {
                ymin = py;
                imin = i;
            }

            ymax = Math.Max(ymax, py);
            xmax = Math.Max(xmax, px);
            xmin = Math.Min(xmin, px);

            px <<= XY_SHIFT - shift;
            py <<= XY_SHIFT - shift;

            if (antiAliased)
            {
                AntiAliasedLine.LineAA(img, new PointL(p0x, p0y), new PointL(px, py), color);
            }
            else if (shift == 0)
            {
                LinePrimitives.Line(
                    img,
                    new Point((int)(p0x >> XY_SHIFT), (int)(p0y >> XY_SHIFT)),
                    new Point((int)(px >> XY_SHIFT), (int)(py >> XY_SHIFT)),
                    color,
                    lineType);
            }
            else
            {
                LinePrimitives.Line2(img, new PointL(p0x, p0y), new PointL(px, py), color);
            }

            p0x = px;
            p0y = py;
        }

        xmin = (xmin + delta) >> shift;
        xmax = (xmax + delta) >> shift;
        ymin = (ymin + delta) >> shift;
        ymax = (ymax + delta) >> shift;

        if (npts < 3 ||
            (int)xmax < 0 ||
            (int)ymax < 0 ||
            (int)xmin >= img.Width ||
            (int)ymin >= img.Height)
        {
            return;
        }

        ymax = Math.Min(ymax, img.Height - 1);

        Span<ConvexEdge> edge = stackalloc ConvexEdge[2];
        edge[0].Idx = edge[1].Idx = imin;

        int y = (int)ymin;
        edge[0].Ye = edge[1].Ye = y;
        edge[0].Di = 1;
        edge[1].Di = npts - 1;
        edge[0].X = edge[1].X = -XY_ONE;
        edge[0].Dx = edge[1].Dx = 0;

        do
        {
            int advanceCount = !antiAliased || y < (int)ymax || y == (int)ymin ? 2 : 0;
            for (int i = 0; i < advanceCount; i++)
            {
                if (y >= edge[i].Ye)
                {
                    int idx0 = edge[i].Idx;
                    int di = edge[i].Di;
                    int idx = idx0 + di;
                    if (idx >= npts)
                    {
                        idx -= npts;
                    }

                    int ty;

                    // Shared decrementing counter: bounds the total edge walk.
                    for (; edges-- > 0;)
                    {
                        ty = (int)((v[idx].Y + delta) >> shift);
                        if (ty > y)
                        {
                            long xs = v[idx0].X;
                            long xe = v[idx].X;
                            if (shift != XY_SHIFT)
                            {
                                xs <<= XY_SHIFT - shift;
                                xe <<= XY_SHIFT - shift;
                            }

                            edge[i].Ye = ty;
                            edge[i].Dx = ((xe - xs) * 2 + ((long)ty - y)) / (2 * ((long)ty - y));
                            edge[i].X = xs;
                            edge[i].Idx = idx;
                            break;
                        }

                        idx0 = idx;
                        idx += di;
                        if (idx >= npts)
                        {
                            idx -= npts;
                        }
                    }
                }
            }

            if (edges < 0)
            {
                break;
            }

            if (y >= 0)
            {
                int left = 0, right = 1;
                if (edge[0].X > edge[1].X)
                {
                    left = 1;
                    right = 0;
                }

                int xx1 = (int)((edge[left].X + delta1) >> XY_SHIFT);
                int xx2 = (int)((edge[right].X + delta2) >> XY_SHIFT);

                if (xx2 >= 0 && xx1 < img.Width)
                {
                    if (xx1 < 0)
                    {
                        xx1 = 0;
                    }

                    if (xx2 >= img.Width)
                    {
                        xx2 = img.Width - 1;
                    }

                    PixelOps.HLine(img, y, xx1, xx2, color);
                }
            }

            edge[0].X += edge[0].Dx;
            edge[1].X += edge[1].Dx;
        }
        while (++y <= (int)ymax);
    }

    /// <summary>
    /// Draws the outline of a single polygon and appends its non-horizontal edges
    /// to <paramref name="edges"/> for a subsequent
    /// <see cref="FillEdgeCollection{T}"/>. Ports OpenCV's <c>CollectPolyEdges</c>
    /// (drawing.cpp:1262-1334): vertices are transformed by
    /// <paramref name="offset"/> and <paramref name="shift"/> into fixed-point,
    /// the outline is drawn with <see cref="LinePrimitives.Line{T}"/>, and the
    /// clipped endpoints are used to derive a more accurate <see cref="PolyEdge"/>.
    /// Mutates <paramref name="img"/> in place.
    /// </summary>
    /// <param name="img">Target image, mutated in place (outline pass).</param>
    /// <param name="v">Polygon vertices in <paramref name="shift"/> fixed-point.</param>
    /// <param name="edges">Accumulating edge list, appended to.</param>
    /// <param name="color">Raw per-channel outline colour.</param>
    /// <param name="lineType">Outline line type: 4, 8 or 16 (<c>LINE_AA</c>).</param>
    /// <param name="shift">Number of sub-pixel fractional bits (0..16).</param>
    /// <param name="offset">Point added to every vertex before rasterising.</param>
    public static void CollectPolyEdges<T>(
        Mat<T> img,
        ReadOnlySpan<PointL> v,
        List<PolyEdge> edges,
        ReadOnlySpan<T> color,
        int lineType,
        int shift,
        Point offset)
        where T : unmanaged
    {
        int count = v.Length;
        int delta = offset.Y + ((1 << shift) >> 1);

        // pt0 starts at the last vertex, transformed by offset + shift.
        long pt0x = (v[count - 1].X + offset.X) << (XY_SHIFT - shift);
        long pt0y = (v[count - 1].Y + delta) >> shift;

        edges.EnsureCapacity(edges.Count + count);

        for (int i = 0; i < count; i++)
        {
            long pt1x = (v[i].X + offset.X) << (XY_SHIFT - shift);
            long pt1y = (v[i].Y + delta) >> shift;

            long pt0cx = pt0x, pt0cy = pt0y;
            long pt1cx = pt1x, pt1cy = pt1y;

            if (lineType >= AntiAliasedLine.LineAa)
            {
                AntiAliasedLine.LineAA(img, new PointL(pt0x, pt0y << XY_SHIFT), new PointL(pt1x, pt1y << XY_SHIFT), color);
            }
            else
            {
                long t0y = pt0y;
                long t1y = pt1y;
                long t0x = (pt0x + (XY_ONE >> 1)) >> XY_SHIFT;
                long t1x = (pt1x + (XY_ONE >> 1)) >> XY_SHIFT;

                // C++ converts Point2l to Point via saturate_cast<int> (clamping),
                // not a wrapping truncation — the distinction matters only for
                // coordinates whose magnitude reaches 2^31 after offset folding.
                LinePrimitives.Line(
                    img,
                    new Point(SaturateToInt(t0x), SaturateToInt(t0y)),
                    new Point(SaturateToInt(t1x), SaturateToInt(t1y)),
                    color,
                    lineType);

                // Use clipped endpoints to create a more accurate PolyEdge. The
                // (uint) casts reproduce the C++ (unsigned) "x < 0 || x >= dim" idiom.
                if ((uint)t0x >= (uint)img.Width ||
                    (uint)t1x >= (uint)img.Width ||
                    (uint)t0y >= (uint)img.Height ||
                    (uint)t1y >= (uint)img.Height)
                {
                    LinePrimitives.ClipLine(img.Width, img.Height, ref t0x, ref t0y, ref t1x, ref t1y);

                    if (t0y != t1y)
                    {
                        pt0cy = t0y;
                        pt1cy = t1y;
                    }
                }

                pt0cx = t0x << XY_SHIFT;
                pt1cx = t1x << XY_SHIFT;
            }

            if (pt0y == pt1y)
            {
                // Skip horizontal edges. (C++ 'continue' still runs pt0 = pt1.)
                pt0x = pt1x;
                pt0y = pt1y;
                continue;
            }

            PolyEdge edge = default;
            edge.Dx = (pt1cx - pt0cx) / (pt1cy - pt0cy);
            if (pt0y < pt1y)
            {
                edge.Y0 = (int)pt0y;
                edge.Y1 = (int)pt1y;
                edge.X = pt0cx + (pt0y - pt0cy) * edge.Dx; // correct starting point for clipped lines
            }
            else
            {
                edge.Y0 = (int)pt1y;
                edge.Y1 = (int)pt0y;
                edge.X = pt1cx + (pt1y - pt1cy) * edge.Dx; // correct starting point for clipped lines
            }

            edge.Next = -1;
            edges.Add(edge);

            pt0x = pt1x;
            pt0y = pt1y;
        }
    }

    /// <summary>
    /// Fills the region bounded by the collected <paramref name="edges"/> using the
    /// even-odd (XOR) rule, ports OpenCV's <c>FillEdgeCollection</c>
    /// (drawing.cpp:1347-1494). The active-edge-table traversal that the original
    /// performs through <c>PolyEdge* next</c> pointers is reproduced with an
    /// index-based working array: element 0 is the head node, elements 1..total are
    /// the sorted edges and element total+1 is a sentinel; <c>-1</c> is the null
    /// pointer. Mutates <paramref name="img"/> in place; the incoming
    /// <paramref name="edges"/> list is sorted and a sentinel appended.
    /// </summary>
    /// <param name="img">Target image, mutated in place.</param>
    /// <param name="edges">Edges produced by <see cref="CollectPolyEdges{T}"/>.</param>
    /// <param name="color">Raw per-channel fill colour.</param>
    public static void FillEdgeCollection<T>(
        Mat<T> img,
        List<PolyEdge> edges,
        ReadOnlySpan<T> color)
        where T : unmanaged
    {
        int i, y, total = edges.Count;
        int width = img.Width;
        int height = img.Height;

        int y_max = int.MinValue, y_min = int.MaxValue;
        long x_max = -1L, x_min = long.MaxValue; // C++ x_max = 0xFFFFFFFFFFFFFFFF == -1
        int delta = XY_ONE - 1;

        if (total < 2)
        {
            return;
        }

        // Bounds prepass over the (still unsorted) edges.
        for (i = 0; i < total; i++)
        {
            PolyEdge e1 = edges[i];

            // C++ CV_Assert(e1.y0 < e1.y1) is active in release builds too: an
            // inverted edge (possible only via int wraparound of extreme
            // coordinates) must throw rather than corrupt the scanline fill.
            if (e1.Y0 >= e1.Y1)
            {
                throw new InvalidOperationException(
                    "Internal error: polygon edge with non-increasing y range.");
            }

            // x-coordinate of the far end of the edge (not necessarily any vertex).
            long x1 = e1.X + (long)(e1.Y1 - e1.Y0) * e1.Dx;
            y_min = Math.Min(y_min, e1.Y0);
            y_max = Math.Max(y_max, e1.Y1);
            x_min = Math.Min(x_min, e1.X);
            x_max = Math.Max(x_max, e1.X);
            x_min = Math.Min(x_min, x1);
            x_max = Math.Max(x_max, x1);
        }

        if (y_max < 0 || y_min >= height || x_max < 0 || x_min >= ((long)width << XY_SHIFT))
        {
            return;
        }

        edges.Sort(CmpEdges);

        // Pointer-to-index restructure: arr[0] is the head node ("tmp"),
        // arr[1..total] the sorted edges, arr[total+1] the appended sentinel.
        // A C++ 'PolyEdge* next' of &tmp maps to index 0; &edges[i] maps to i+1;
        // a null pointer maps to -1.
        var arr = new PolyEdge[total + 2];
        arr[0] = new PolyEdge { Y0 = int.MaxValue, Next = -1 };
        for (int k = 0; k < total; k++)
        {
            arr[k + 1] = edges[k];
        }

        arr[total + 1] = new PolyEdge { Y0 = int.MaxValue, Next = -1 };

        i = 0;              // C++ index into the sorted edges (0..total)
        int ei = 1;         // C++ 'e = &edges[i]' => working-array index i + 1
        arr[0].Next = -1;   // C++ tmp.next = 0 (null)
        y_max = Math.Min(y_max, height);

        for (y = arr[1].Y0; y < y_max; y++)
        {
            int draw = 0;
            bool clipline = y < 0;

            int prelast = 0;            // &tmp
            int last = arr[0].Next;     // tmp.next
            int keepPrelast;

            while (last != -1 || arr[ei].Y0 == y)
            {
                if (last != -1 && arr[last].Y1 == y)
                {
                    // exclude edge if y reaches its lower point
                    arr[prelast].Next = arr[last].Next;
                    last = arr[last].Next;
                    continue;
                }

                keepPrelast = prelast;
                if (last != -1 && (arr[ei].Y0 > y || arr[last].X < arr[ei].X))
                {
                    // go to the next edge in the active list
                    prelast = last;
                    last = arr[last].Next;
                }
                else if (i < total)
                {
                    // insert a new edge into the active list at its upper point
                    arr[prelast].Next = ei;
                    arr[ei].Next = last;
                    prelast = ei;
                    i++;
                    ei = i + 1; // C++ e = &edges[++i]
                }
                else
                {
                    break;
                }

                if (draw != 0)
                {
                    if (!clipline)
                    {
                        // convert x's from fixed-point to image coordinates
                        int x1, x2;

                        if (arr[keepPrelast].X > arr[prelast].X)
                        {
                            x1 = (int)((arr[prelast].X + delta) >> XY_SHIFT);
                            x2 = (int)(arr[keepPrelast].X >> XY_SHIFT);
                        }
                        else
                        {
                            x1 = (int)((arr[keepPrelast].X + delta) >> XY_SHIFT);
                            x2 = (int)(arr[prelast].X >> XY_SHIFT);
                        }

                        // clip and draw the run
                        if (x1 < width && x2 >= 0)
                        {
                            if (x1 < 0)
                            {
                                x1 = 0;
                            }

                            if (x2 >= width)
                            {
                                x2 = width - 1;
                            }

                            PixelOps.HLine(img, y, x1, x2, color);
                        }
                    }

                    arr[keepPrelast].X += arr[keepPrelast].Dx;
                    arr[prelast].X += arr[prelast].Dx;
                }

                draw ^= 1;
            }

            // sort the active list for this row (bubble sort, ported literally)
            keepPrelast = -1; // C++ keep_prelast = 0 (null)

            do
            {
                prelast = 0;            // &tmp
                last = arr[0].Next;     // tmp.next
                int lastExchange = -1;  // null

                while (last != keepPrelast && arr[last].Next != -1)
                {
                    int te = arr[last].Next;

                    // swap edges
                    if (arr[last].X > arr[te].X)
                    {
                        arr[prelast].Next = te;
                        arr[last].Next = arr[te].Next;
                        arr[te].Next = last;
                        prelast = te;
                        lastExchange = prelast;
                    }
                    else
                    {
                        prelast = last;
                        last = te;
                    }
                }

                if (lastExchange == -1)
                {
                    break;
                }

                keepPrelast = lastExchange;
            }
            while (keepPrelast != arr[0].Next && keepPrelast != 0); // C++ != tmp.next && != &tmp
        }
    }

    /// <summary>
    /// Edge ordering used before the active-edge scan, ports OpenCV's
    /// <c>CmpEdges</c> (drawing.cpp:1336-1343): order by <see cref="PolyEdge.Y0"/>,
    /// then <see cref="PolyEdge.X"/>, then <see cref="PolyEdge.Dx"/>. Equal triples
    /// are interchangeable, so an unstable sort matches the original
    /// <c>std::sort</c>.
    /// </summary>
    private static int CmpEdges(PolyEdge e1, PolyEdge e2)
    {
        if (e1.Y0 != e2.Y0)
        {
            return e1.Y0 < e2.Y0 ? -1 : 1;
        }

        if (e1.X != e2.X)
        {
            return e1.X < e2.X ? -1 : 1;
        }

        if (e1.Dx != e2.Dx)
        {
            return e1.Dx < e2.Dx ? -1 : 1;
        }

        return 0;
    }

    /// <summary>
    /// Narrows an <see cref="long"/> coordinate to <see cref="int"/> with
    /// clamping, matching OpenCV's <c>saturate_cast&lt;int&gt;(int64)</c> used by
    /// the implicit <c>Point2l</c>-to-<c>Point</c> conversion.
    /// </summary>
    private static int SaturateToInt(long v) =>
        v > int.MaxValue ? int.MaxValue : v < int.MinValue ? int.MinValue : (int)v;

    /// <summary>
    /// Mutable scanline state for the two edges tracked by
    /// <see cref="FillConvexPoly{T}"/>, mirroring the anonymous <c>edge[2]</c>
    /// struct in the C++ (drawing.cpp:1098-1104). Kept unmanaged so it can live in
    /// a <c>stackalloc</c> span.
    /// </summary>
    private struct ConvexEdge
    {
        /// <summary>Index of the vertex currently at the top of this edge.</summary>
        public int Idx;

        /// <summary>Vertex-walk direction (+1 for one edge, npts-1 for the other).</summary>
        public int Di;

        /// <summary>Current abscissa in <see cref="DrawingConstants.XY_SHIFT"/> fixed-point.</summary>
        public long X;

        /// <summary>Per-row abscissa increment in fixed-point.</summary>
        public long Dx;

        /// <summary>Row at which this edge segment ends (its lower vertex row).</summary>
        public int Ye;
    }
}
