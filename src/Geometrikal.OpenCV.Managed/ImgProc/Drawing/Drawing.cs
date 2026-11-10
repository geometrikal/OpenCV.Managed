// This implementation is ported from the OpenCV drawing implementation.
//
// Original source:
//   modules/imgproc/src/drawing.cpp   (in this repo: src/OpenCV/modules/imgproc/src/drawing.cpp)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Managed port of OpenCV's drawing functions
/// (<c>modules/imgproc/src/drawing.cpp</c>): <c>line</c>, <c>rectangle</c>,
/// <c>circle</c>, <c>ellipse</c>, <c>fillConvexPoly</c>, <c>fillPoly</c>,
/// <c>polylines</c>, and <c>drawContours</c>.
///
/// <para><b>In-place mutation.</b> Every method draws directly into the supplied
/// <see cref="Mat{T}"/> and returns nothing; the image is modified in place. When
/// the target is a region-of-interest view, the writes go through to the parent
/// image's backing buffer, and pixels of the parent outside the ROI are left
/// untouched.</para>
///
/// <para><b>Silent clipping.</b> Geometry that falls outside the image is clipped
/// silently, exactly as in OpenCV: drawing entirely off-image is a well-defined
/// no-op and never throws.</para>
///
/// <para><b>Coordinates and sub-pixel precision.</b> Point coordinates use the
/// image convention (X right, Y down). The <c>shift</c> parameter (0..16) gives
/// the number of fractional bits in the supplied coordinates and radii/axes,
/// matching OpenCV's sub-pixel sub-division.</para>
///
/// <para><b>Line types.</b> <see cref="LineType.FourConnected"/> (LINE_4),
/// <see cref="LineType.EightConnected"/> (LINE_8) and
/// <see cref="LineType.AntiAliased"/> (LINE_AA) are supported. LINE_AA applies to
/// <see cref="byte"/> images. For other element types every method replaces
/// LINE_AA with LINE_8, as OpenCV does. The <c>putText</c> and rotated-rectangle
/// <c>ellipse</c> overloads are not ported.</para>
///
/// <para><b>Fill rule.</b> <see cref="FillPoly{T}"/> uses the even-odd (XOR) rule
/// like OpenCV: overlapping regions and a second contour nested inside the first
/// produce holes. Thick round caps and thick line joins are drawn as filled
/// circles on the non-AA path and as anti-aliased filled ellipses on the AA path.</para>
///
/// <para><b>Colour.</b> A <see cref="Scalar"/> supplies up to four channel values.
/// Each channel is saturate-cast to the element type <typeparamref name="T"/>
/// following OpenCV's <c>scalarToRawData</c> semantics; only images with 1..4
/// channels are supported.</para>
/// </summary>
public static class Drawing
{
    /// <summary>
    /// Sentinel <c>thickness</c> value requesting a filled shape, matching
    /// OpenCV's <c>cv::FILLED</c>.
    /// </summary>
    public const int Filled = -1;

    /// <summary>
    /// Draws a line segment between <paramref name="pt1"/> and <paramref name="pt2"/>.
    /// The image is modified in place; portions outside the image are clipped
    /// silently.
    /// </summary>
    /// <typeparam name="T">The pixel element type (1..4 channels).</typeparam>
    /// <param name="image">Target image, drawn into in place.</param>
    /// <param name="pt1">First endpoint (X right, Y down).</param>
    /// <param name="pt2">Second endpoint.</param>
    /// <param name="color">Line colour; channels beyond the image's are ignored.</param>
    /// <param name="thickness">Line thickness in pixels, 1..32767.</param>
    /// <param name="lineType">Line type: LINE_4, LINE_8 or LINE_AA.</param>
    /// <param name="shift">Number of fractional bits in the point coordinates, 0..16.</param>
    /// <exception cref="ArgumentNullException"><paramref name="image"/> is null.</exception>
    /// <exception cref="NotSupportedException">The image has more than 4 channels.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="lineType"/>, <paramref name="thickness"/>, or <paramref name="shift"/> is out of range.</exception>
    public static void Line<T>(Mat<T> image, Point pt1, Point pt2, Scalar color,
        int thickness = 1, LineType lineType = LineType.EightConnected, int shift = 0)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        ThrowIfBadChannels(image);
        ThrowIfBadLineType(lineType);
        lineType = ResolveLineType<T>(lineType);
        ThrowIfBadShift(shift);
        if (thickness <= 0 || thickness > DrawingConstants.MAX_THICKNESS)
        {
            throw new ArgumentOutOfRangeException(
                nameof(thickness), thickness, "thickness must be in the range [1, 32767].");
        }

        T[] raw = PixelOps.ToRawColor<T>(color, image.Channels);
        CurvePrimitives.ThickLine(image, (PointL)pt1, (PointL)pt2, raw, thickness, (int)lineType, flags: 3, shift);
    }

    /// <summary>
    /// Draws a rectangle whose opposite corners are <paramref name="pt1"/> and
    /// <paramref name="pt2"/>. The image is modified in place; portions outside the
    /// image are clipped silently.
    /// </summary>
    /// <typeparam name="T">The pixel element type (1..4 channels).</typeparam>
    /// <param name="image">Target image, drawn into in place.</param>
    /// <param name="pt1">One corner (X right, Y down).</param>
    /// <param name="pt2">The opposite corner.</param>
    /// <param name="color">Rectangle colour.</param>
    /// <param name="thickness">Outline thickness in pixels, or negative / <see cref="Filled"/> to fill. Must not exceed 32767.</param>
    /// <param name="lineType">Line type: LINE_4, LINE_8 or LINE_AA.</param>
    /// <param name="shift">Number of fractional bits in the point coordinates, 0..16.</param>
    /// <exception cref="ArgumentNullException"><paramref name="image"/> is null.</exception>
    /// <exception cref="NotSupportedException">The image has more than 4 channels.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="lineType"/>, <paramref name="thickness"/>, or <paramref name="shift"/> is out of range.</exception>
    public static void Rectangle<T>(Mat<T> image, Point pt1, Point pt2, Scalar color,
        int thickness = 1, LineType lineType = LineType.EightConnected, int shift = 0)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        ThrowIfBadChannels(image);
        ThrowIfBadLineType(lineType);
        lineType = ResolveLineType<T>(lineType);
        ThrowIfBadShift(shift);
        ThrowIfThicknessTooLarge(thickness);

        T[] raw = PixelOps.ToRawColor<T>(color, image.Channels);

        Span<PointL> pt = stackalloc PointL[4];
        pt[0] = (PointL)pt1;
        pt[1] = new PointL(pt2.X, pt1.Y);
        pt[2] = (PointL)pt2;
        pt[3] = new PointL(pt1.X, pt2.Y);

        if (thickness >= 0)
        {
            CurvePrimitives.PolyLine(image, pt, isClosed: true, raw, thickness, (int)lineType, shift);
        }
        else
        {
            PolygonPrimitives.FillConvexPoly(image, pt, raw, (int)lineType, shift);
        }
    }

    /// <summary>
    /// Draws a rectangle described by a <see cref="Rect"/>. The rectangle is first
    /// cropped to slightly larger than the image (matching OpenCV) and then drawn
    /// via the two-corner overload. The image is modified in place.
    /// </summary>
    /// <typeparam name="T">The pixel element type (1..4 channels).</typeparam>
    /// <param name="image">Target image, drawn into in place.</param>
    /// <param name="rectangle">The rectangle to draw (top-left origin, half-open extent).</param>
    /// <param name="color">Rectangle colour.</param>
    /// <param name="thickness">Outline thickness in pixels, or negative / <see cref="Filled"/> to fill. Must not exceed 32767.</param>
    /// <param name="lineType">Line type: LINE_4, LINE_8 or LINE_AA.</param>
    /// <param name="shift">Number of fractional bits in the coordinates, 0..16.</param>
    /// <exception cref="ArgumentNullException"><paramref name="image"/> is null.</exception>
    /// <exception cref="NotSupportedException">The image has more than 4 channels.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="lineType"/>, <paramref name="thickness"/>, or <paramref name="shift"/> is out of range.</exception>
    public static void Rectangle<T>(Mat<T> image, Rect rectangle, Scalar color,
        int thickness = 1, LineType lineType = LineType.EightConnected, int shift = 0)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        ThrowIfBadChannels(image);
        ThrowIfBadLineType(lineType);
        lineType = ResolveLineType<T>(lineType);
        ThrowIfBadShift(shift);
        ThrowIfThicknessTooLarge(thickness);

        // Crop the rectangle to right around the mat, mirroring OpenCV's
        // rec &= Rect(-(1<<shift), -(1<<shift), (cols+2)<<shift, (rows+2)<<shift).
        int s = 1 << shift;
        int cropX = -s;
        int cropY = -s;
        int cropW = (image.Width + 2) << shift;
        int cropH = (image.Height + 2) << shift;

        int rx = Math.Max(rectangle.X, cropX);
        int ry = Math.Max(rectangle.Y, cropY);
        int rw = Math.Min(rectangle.X + rectangle.Width, cropX + cropW) - rx;
        int rh = Math.Min(rectangle.Y + rectangle.Height, cropY + cropH) - ry;

        if (rw > 0 && rh > 0)
        {
            // C++ rectangle(rec.tl(), rec.br() - Point(1<<shift, 1<<shift), ...).
            var tl = new Point(rx, ry);
            var br = new Point(rx + rw - s, ry + rh - s);
            Rectangle(image, tl, br, color, thickness, lineType, shift);
        }
    }

    /// <summary>
    /// Draws a circle centred at <paramref name="center"/> with the given
    /// <paramref name="radius"/>. Thin LINE_8 circles with no sub-pixel shift use
    /// the fast integer rasteriser; every other case routes through the elliptic
    /// path. The image is modified in place; portions outside are clipped silently.
    /// </summary>
    /// <typeparam name="T">The pixel element type (1..4 channels).</typeparam>
    /// <param name="image">Target image, drawn into in place.</param>
    /// <param name="center">Circle centre (X right, Y down).</param>
    /// <param name="radius">Circle radius in pixels; must be non-negative.</param>
    /// <param name="color">Circle colour.</param>
    /// <param name="thickness">Outline thickness, or negative / <see cref="Filled"/> to fill. Must not exceed 32767.</param>
    /// <param name="lineType">Line type: LINE_4, LINE_8 or LINE_AA.</param>
    /// <param name="shift">Number of fractional bits in the centre and radius, 0..16.</param>
    /// <exception cref="ArgumentNullException"><paramref name="image"/> is null.</exception>
    /// <exception cref="NotSupportedException">The image has more than 4 channels.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="radius"/>, <paramref name="lineType"/>, <paramref name="thickness"/>, or <paramref name="shift"/> is out of range.</exception>
    public static void Circle<T>(Mat<T> image, Point center, int radius, Scalar color,
        int thickness = 1, LineType lineType = LineType.EightConnected, int shift = 0)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        ThrowIfBadChannels(image);
        ThrowIfBadLineType(lineType);
        lineType = ResolveLineType<T>(lineType);
        ThrowIfBadShift(shift);
        if (radius < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(radius), radius, "radius must be non-negative.");
        }

        ThrowIfThicknessTooLarge(thickness);

        T[] raw = PixelOps.ToRawColor<T>(color, image.Channels);

        if (thickness > 1 || lineType != LineType.EightConnected || shift > 0)
        {
            int up = DrawingConstants.XY_SHIFT - shift;
            var centerL = new PointL((long)center.X << up, (long)center.Y << up);
            long radiusL = (long)radius << up;
            CurvePrimitives.EllipseEx(image, centerL, radiusL, radiusL, 0, 0, 360, raw, thickness, (int)lineType);
        }
        else
        {
            CurvePrimitives.Circle(image, center, radius, raw, fill: thickness < 0);
        }
    }

    /// <summary>
    /// Draws or fills an elliptic arc centred at <paramref name="center"/> with
    /// half-axes <paramref name="axes"/>, rotated by <paramref name="angle"/>
    /// degrees and spanning <paramref name="startAngle"/> to
    /// <paramref name="endAngle"/> degrees. The image is modified in place; portions
    /// outside the image are clipped silently.
    /// </summary>
    /// <typeparam name="T">The pixel element type (1..4 channels).</typeparam>
    /// <param name="image">Target image, drawn into in place.</param>
    /// <param name="center">Ellipse centre (X right, Y down).</param>
    /// <param name="axes">Half-axis lengths; both must be non-negative.</param>
    /// <param name="angle">Rotation of the ellipse in degrees (rounded to whole degrees).</param>
    /// <param name="startAngle">Arc start angle in degrees (rounded to whole degrees).</param>
    /// <param name="endAngle">Arc end angle in degrees (rounded to whole degrees).</param>
    /// <param name="color">Ellipse colour.</param>
    /// <param name="thickness">Outline thickness, or negative / <see cref="Filled"/> to fill. Must not exceed 32767.</param>
    /// <param name="lineType">Line type: LINE_4, LINE_8 or LINE_AA.</param>
    /// <param name="shift">Number of fractional bits in the centre and axes, 0..16.</param>
    /// <exception cref="ArgumentNullException"><paramref name="image"/> is null.</exception>
    /// <exception cref="NotSupportedException">The image has more than 4 channels.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="axes"/>, <paramref name="lineType"/>, <paramref name="thickness"/>, or <paramref name="shift"/> is out of range.</exception>
    public static void Ellipse<T>(Mat<T> image, Point center, Size axes, double angle,
        double startAngle, double endAngle, Scalar color, int thickness = 1,
        LineType lineType = LineType.EightConnected, int shift = 0)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        ThrowIfBadChannels(image);
        ThrowIfBadLineType(lineType);
        lineType = ResolveLineType<T>(lineType);
        ThrowIfBadShift(shift);
        if (axes.Width < 0 || axes.Height < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(axes), axes, "axis lengths must be non-negative.");
        }

        ThrowIfThicknessTooLarge(thickness);

        T[] raw = PixelOps.ToRawColor<T>(color, image.Channels);

        int roundedAngle = (int)Math.Round(angle, MidpointRounding.ToEven);
        int roundedStart = (int)Math.Round(startAngle, MidpointRounding.ToEven);
        int roundedEnd = (int)Math.Round(endAngle, MidpointRounding.ToEven);

        int up = DrawingConstants.XY_SHIFT - shift;
        var centerL = new PointL((long)center.X << up, (long)center.Y << up);
        long axW = (long)axes.Width << up;
        long axH = (long)axes.Height << up;

        CurvePrimitives.EllipseEx(image, centerL, axW, axH, roundedAngle, roundedStart, roundedEnd, raw, thickness, (int)lineType);
    }

    /// <summary>
    /// Fills a single convex polygon (or a monotone polygon) defined by
    /// <paramref name="points"/>. The image is modified in place; portions outside
    /// the image are clipped silently. An empty point set is a no-op.
    /// </summary>
    /// <typeparam name="T">The pixel element type (1..4 channels).</typeparam>
    /// <param name="image">Target image, drawn into in place.</param>
    /// <param name="points">Polygon vertices in order.</param>
    /// <param name="color">Fill colour.</param>
    /// <param name="lineType">Line type of the outline: LINE_4, LINE_8 or LINE_AA.</param>
    /// <param name="shift">Number of fractional bits in the vertex coordinates, 0..16.</param>
    /// <exception cref="ArgumentNullException"><paramref name="image"/> is null.</exception>
    /// <exception cref="NotSupportedException">The image has more than 4 channels.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="lineType"/> or <paramref name="shift"/> is out of range.</exception>
    public static void FillConvexPoly<T>(Mat<T> image, ReadOnlySpan<Point> points, Scalar color,
        LineType lineType = LineType.EightConnected, int shift = 0)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        ThrowIfBadChannels(image);
        ThrowIfBadLineType(lineType);
        lineType = ResolveLineType<T>(lineType);
        ThrowIfBadShift(shift);

        if (points.IsEmpty)
        {
            return;
        }

        T[] raw = PixelOps.ToRawColor<T>(color, image.Channels);
        PointL[] v = new PointL[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            v[i] = (PointL)points[i];
        }

        PolygonPrimitives.FillConvexPoly(image, v, raw, (int)lineType, shift);
    }

    /// <summary>
    /// Fills one or more polygons using the even-odd (XOR) rule. Edges from every
    /// polygon are accumulated together before filling, so a polygon nested inside
    /// another produces a hole. The image is modified in place; portions outside
    /// the image are clipped silently.
    /// </summary>
    /// <typeparam name="T">The pixel element type (1..4 channels).</typeparam>
    /// <param name="image">Target image, drawn into in place.</param>
    /// <param name="polygons">The polygons; each is an array of vertices. Individual arrays may be empty (skipped) but must not be null.</param>
    /// <param name="color">Fill colour.</param>
    /// <param name="lineType">Line type of the outline: LINE_4, LINE_8 or LINE_AA.</param>
    /// <param name="shift">Number of fractional bits in the vertex coordinates, 0..16.</param>
    /// <param name="offset">A coordinate offset added to every vertex.</param>
    /// <exception cref="ArgumentNullException"><paramref name="image"/> or <paramref name="polygons"/> is null.</exception>
    /// <exception cref="ArgumentException">An element of <paramref name="polygons"/> is null.</exception>
    /// <exception cref="NotSupportedException">The image has more than 4 channels.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="lineType"/> or <paramref name="shift"/> is out of range.</exception>
    public static void FillPoly<T>(Mat<T> image, IReadOnlyList<Point[]> polygons, Scalar color,
        LineType lineType = LineType.EightConnected, int shift = 0, Point offset = default)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(polygons);
        ThrowIfBadChannels(image);
        ThrowIfBadLineType(lineType);
        lineType = ResolveLineType<T>(lineType);
        ThrowIfBadShift(shift);

        int total = 0;
        for (int i = 0; i < polygons.Count; i++)
        {
            Point[] poly = polygons[i];
            if (poly is null)
            {
                throw new ArgumentException($"The polygon at index {i} is null.", nameof(polygons));
            }

            total += poly.Length;
        }

        T[] raw = PixelOps.ToRawColor<T>(color, image.Channels);
        var edges = new List<PolyEdge>(total + 1);

        for (int i = 0; i < polygons.Count; i++)
        {
            Point[] poly = polygons[i];
            if (poly.Length > 0)
            {
                PointL[] v = new PointL[poly.Length];
                for (int j = 0; j < poly.Length; j++)
                {
                    v[j] = (PointL)poly[j];
                }

                PolygonPrimitives.CollectPolyEdges(image, v, edges, raw, (int)lineType, shift, offset);
            }
        }

        PolygonPrimitives.FillEdgeCollection(image, edges, raw);
    }

    /// <summary>
    /// Draws one or more poly-lines. The image is modified in place; portions
    /// outside the image are clipped silently. Empty vertex arrays are skipped.
    /// </summary>
    /// <typeparam name="T">The pixel element type (1..4 channels).</typeparam>
    /// <param name="image">Target image, drawn into in place.</param>
    /// <param name="polylines">The poly-lines; each is an array of vertices. Individual arrays may be empty (skipped) but must not be null.</param>
    /// <param name="isClosed">When true, each poly-line's last vertex is joined back to its first.</param>
    /// <param name="color">Line colour.</param>
    /// <param name="thickness">Line thickness in pixels, 0..32767.</param>
    /// <param name="lineType">Line type: LINE_4, LINE_8 or LINE_AA.</param>
    /// <param name="shift">Number of fractional bits in the vertex coordinates, 0..16.</param>
    /// <exception cref="ArgumentNullException"><paramref name="image"/> or <paramref name="polylines"/> is null.</exception>
    /// <exception cref="ArgumentException">An element of <paramref name="polylines"/> is null.</exception>
    /// <exception cref="NotSupportedException">The image has more than 4 channels.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="lineType"/>, <paramref name="thickness"/>, or <paramref name="shift"/> is out of range.</exception>
    public static void Polylines<T>(Mat<T> image, IReadOnlyList<Point[]> polylines, bool isClosed,
        Scalar color, int thickness = 1, LineType lineType = LineType.EightConnected, int shift = 0)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(polylines);
        ThrowIfBadChannels(image);
        ThrowIfBadLineType(lineType);
        lineType = ResolveLineType<T>(lineType);
        ThrowIfBadShift(shift);
        if (thickness < 0 || thickness > DrawingConstants.MAX_THICKNESS)
        {
            throw new ArgumentOutOfRangeException(
                nameof(thickness), thickness, "thickness must be in the range [0, 32767].");
        }

        for (int i = 0; i < polylines.Count; i++)
        {
            if (polylines[i] is null)
            {
                throw new ArgumentException($"The poly-line at index {i} is null.", nameof(polylines));
            }
        }

        T[] raw = PixelOps.ToRawColor<T>(color, image.Channels);

        for (int i = 0; i < polylines.Count; i++)
        {
            Point[] poly = polylines[i];
            PointL[] v = new PointL[poly.Length];
            for (int j = 0; j < poly.Length; j++)
            {
                v[j] = (PointL)poly[j];
            }

            CurvePrimitives.PolyLine(image, v, isClosed, raw, thickness, (int)lineType, shift);
        }
    }

    /// <summary>
    /// Draws contour outlines or fills contour interiors. Composes directly with
    /// the output of contour extraction: pass <c>result.Contours</c> and
    /// <c>result.Hierarchy</c>. The image is modified in place; portions outside the
    /// image are clipped silently.
    /// </summary>
    /// <typeparam name="T">The pixel element type (1..4 channels).</typeparam>
    /// <param name="image">Target image, drawn into in place.</param>
    /// <param name="contours">The contours; each is an array of vertices. Must not contain null elements.</param>
    /// <param name="contourIndex">Index of the contour to draw, or negative to draw all contours.</param>
    /// <param name="color">Draw colour.</param>
    /// <param name="thickness">Outline thickness, or negative / <see cref="Filled"/> to fill. Must not exceed 32767.</param>
    /// <param name="lineType">Line type: LINE_4, LINE_8 or LINE_AA.</param>
    /// <param name="hierarchy">Optional contour hierarchy enabling the <paramref name="maxLevel"/> nesting walk; when supplied it must have the same length as <paramref name="contours"/>.</param>
    /// <param name="maxLevel">Maximum nesting depth to draw when a hierarchy is supplied (0 = only the selected contours).</param>
    /// <param name="offset">A coordinate offset added to every vertex.</param>
    /// <remarks>
    /// The hierarchy/<paramref name="maxLevel"/> selection follows OpenCV 5, which
    /// differs from OpenCV &lt;= 4.6 for some <paramref name="maxLevel"/> values.
    /// Because a single colour is used, the drawing order does not affect the output.
    /// </remarks>
    /// <exception cref="ArgumentNullException"><paramref name="image"/> or <paramref name="contours"/> is null.</exception>
    /// <exception cref="ArgumentException">An element of <paramref name="contours"/> is null, or <paramref name="hierarchy"/> has a different length from <paramref name="contours"/>.</exception>
    /// <exception cref="NotSupportedException">The image has more than 4 channels.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="lineType"/> or <paramref name="thickness"/> is out of range, or <paramref name="contourIndex"/> is not less than the contour count.</exception>
    public static void DrawContours<T>(Mat<T> image, IReadOnlyList<Point[]> contours,
        int contourIndex, Scalar color, int thickness = 1,
        LineType lineType = LineType.EightConnected,
        IReadOnlyList<ContourHierarchy>? hierarchy = null, int maxLevel = int.MaxValue,
        Point offset = default)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(contours);
        ThrowIfBadChannels(image);
        ThrowIfBadLineType(lineType);
        lineType = ResolveLineType<T>(lineType);
        ThrowIfThicknessTooLarge(thickness);

        int ncontours = contours.Count;

        // C++: if (!ncontours) return;
        if (ncontours == 0)
        {
            return;
        }

        if (hierarchy is not null && hierarchy.Count != ncontours)
        {
            throw new ArgumentException(
                "hierarchy must have the same number of entries as contours.", nameof(hierarchy));
        }

        if (contourIndex >= ncontours)
        {
            throw new ArgumentOutOfRangeException(
                nameof(contourIndex), contourIndex, "contourIndex must be less than the contour count.");
        }

        for (int k = 0; k < ncontours; k++)
        {
            if (contours[k] is null)
            {
                throw new ArgumentException($"The contour at index {k} is null.", nameof(contours));
            }
        }

        int start = 0;
        int end = ncontours;
        if (contourIndex >= 0)
        {
            start = contourIndex;
            end = start + 1;
        }

        var indexesToFill = new List<int>();
        if (hierarchy is null || maxLevel == 0)
        {
            for (int idx = start; idx < end; idx++)
            {
                indexesToFill.Add(idx);
            }
        }
        else
        {
            var indexes = new Stack<int>();
            for (int idx = start; idx < end; idx++)
            {
                // Either every top-level contour, or a single explicitly-chosen contour.
                if (hierarchy[idx].Parent < 0 || contourIndex >= 0)
                {
                    indexes.Push(idx);
                }
            }

            while (indexes.Count > 0)
            {
                int cur = indexes.Pop();

                // Depth of the current contour, counted by walking to the root.
                int curLevel = -1;
                int par = cur;
                while (par >= 0)
                {
                    par = hierarchy[par].Parent;
                    curLevel++;
                }

                if (curLevel <= maxLevel)
                {
                    indexesToFill.Add(cur);
                }

                // NOTE: OpenCV 5 uses `next > 0` here (not `>= 0`), so a first child
                // whose index is 0 is never pushed. This mirrors OpenCV exactly.
                int next = hierarchy[cur].FirstChild;
                while (next > 0)
                {
                    indexes.Push(next);
                    next = hierarchy[next].Next;
                }
            }
        }

        if (thickness < 0)
        {
            var contoursToFill = new List<Point[]>(indexesToFill.Count);
            for (int k = 0; k < indexesToFill.Count; k++)
            {
                contoursToFill.Add(contours[indexesToFill[k]]);
            }

            FillPoly(image, contoursToFill, color, lineType, shift: 0, offset);
        }
        else
        {
            T[] raw = PixelOps.ToRawColor<T>(color, image.Channels);
            for (int k = 0; k < indexesToFill.Count; k++)
            {
                Point[] cnt = contours[indexesToFill[k]];
                if (cnt.Length == 0)
                {
                    continue;
                }

                int npoints = cnt.Length;
                for (int j = 0; j < npoints; j++)
                {
                    bool isLastIter = j == npoints - 1;
                    Point pt1 = cnt[j];
                    Point pt2 = cnt[isLastIter ? 0 : j + 1];
                    CurvePrimitives.ThickLine(image, (PointL)(pt1 + offset), (PointL)(pt2 + offset),
                        raw, thickness, (int)lineType, flags: 2, shift: 0);
                }
            }
        }
    }

    /// <summary>
    /// Draws a line from <paramref name="pt1"/> to <paramref name="pt2"/> and an
    /// arrow tip at <paramref name="pt2"/> (OpenCV <c>arrowedLine</c>). The tip has
    /// two segments at plus and minus 45 degrees to the line.
    /// </summary>
    /// <typeparam name="T">The pixel element type (1..4 channels).</typeparam>
    /// <param name="image">Target image, drawn into in place.</param>
    /// <param name="pt1">Start point of the arrow.</param>
    /// <param name="pt2">End point of the arrow. The tip is at this point.</param>
    /// <param name="color">Line colour.</param>
    /// <param name="thickness">Line thickness in pixels, 1..32767.</param>
    /// <param name="lineType">Line type: LINE_4, LINE_8 or LINE_AA.</param>
    /// <param name="shift">Number of fractional bits in the point coordinates, 0..16.</param>
    /// <param name="tipLength">Length of each tip segment as a fraction of the arrow length.</param>
    /// <exception cref="ArgumentNullException"><paramref name="image"/> is null.</exception>
    /// <exception cref="NotSupportedException">The image has more than 4 channels.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="lineType"/>, <paramref name="thickness"/>, or <paramref name="shift"/> is out of range.</exception>
    public static void ArrowedLine<T>(Mat<T> image, Point pt1, Point pt2, Scalar color,
        int thickness = 1, LineType lineType = LineType.EightConnected, int shift = 0, double tipLength = 0.1)
        where T : unmanaged
    {
        double ddx = (double)pt1.X - pt2.X;
        double ddy = (double)pt1.Y - pt2.Y;
        double tipSize = Math.Sqrt((ddx * ddx) + (ddy * ddy)) * tipLength;

        Line(image, pt1, pt2, color, thickness, lineType, shift);

        double angle = Math.Atan2((double)pt1.Y - pt2.Y, (double)pt1.X - pt2.X);

        var p = new Point(
            RoundToInt(pt2.X + (tipSize * Math.Cos(angle + (Math.PI / 4)))),
            RoundToInt(pt2.Y + (tipSize * Math.Sin(angle + (Math.PI / 4)))));
        Line(image, p, pt2, color, thickness, lineType, shift);

        p = new Point(
            RoundToInt(pt2.X + (tipSize * Math.Cos(angle - (Math.PI / 4)))),
            RoundToInt(pt2.Y + (tipSize * Math.Sin(angle - (Math.PI / 4)))));
        Line(image, p, pt2, color, thickness, lineType, shift);
    }

    /// <summary>
    /// Draws a marker of the given <paramref name="markerType"/> centred at
    /// <paramref name="position"/> (OpenCV <c>drawMarker</c>). An undefined marker
    /// type draws a cross.
    /// </summary>
    /// <typeparam name="T">The pixel element type (1..4 channels).</typeparam>
    /// <param name="image">Target image, drawn into in place.</param>
    /// <param name="position">Marker centre.</param>
    /// <param name="color">Marker colour.</param>
    /// <param name="markerType">Marker shape.</param>
    /// <param name="markerSize">Marker extent in pixels. Each half-extent is <c>markerSize / 2</c>.</param>
    /// <param name="thickness">Line thickness in pixels, 1..32767.</param>
    /// <param name="lineType">Line type: LINE_4, LINE_8 or LINE_AA.</param>
    /// <exception cref="ArgumentNullException"><paramref name="image"/> is null.</exception>
    /// <exception cref="NotSupportedException">The image has more than 4 channels.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="lineType"/> or <paramref name="thickness"/> is out of range.</exception>
    public static void DrawMarker<T>(Mat<T> image, Point position, Scalar color,
        MarkerType markerType = MarkerType.Cross, int markerSize = 20, int thickness = 1,
        LineType lineType = LineType.EightConnected)
        where T : unmanaged
    {
        int h = markerSize / 2;
        int x = position.X;
        int y = position.Y;

        void L(int x1, int y1, int x2, int y2) =>
            Line(image, new Point(x1, y1), new Point(x2, y2), color, thickness, lineType);

        switch (markerType)
        {
            case MarkerType.TiltedCross:
                L(x - h, y - h, x + h, y + h);
                L(x + h, y - h, x - h, y + h);
                break;
            case MarkerType.Star:
                L(x - h, y, x + h, y);
                L(x, y - h, x, y + h);
                L(x - h, y - h, x + h, y + h);
                L(x + h, y - h, x - h, y + h);
                break;
            case MarkerType.Diamond:
                L(x, y - h, x + h, y);
                L(x + h, y, x, y + h);
                L(x, y + h, x - h, y);
                L(x - h, y, x, y - h);
                break;
            case MarkerType.Square:
                L(x - h, y - h, x + h, y - h);
                L(x + h, y - h, x + h, y + h);
                L(x + h, y + h, x - h, y + h);
                L(x - h, y + h, x - h, y - h);
                break;
            case MarkerType.TriangleUp:
                L(x - h, y + h, x + h, y + h);
                L(x + h, y + h, x, y - h);
                L(x, y - h, x - h, y + h);
                break;
            case MarkerType.TriangleDown:
                L(x - h, y - h, x + h, y - h);
                L(x + h, y - h, x, y + h);
                L(x, y + h, x - h, y - h);
                break;
            default:
                L(x - h, y, x + h, y);
                L(x, y - h, x, y + h);
                break;
        }
    }

    private static int RoundToInt(double value) => (int)Math.Round(value, MidpointRounding.ToEven);

    private static void ThrowIfBadChannels<T>(Mat<T> image)
        where T : unmanaged
    {
        if (image.Channels > 4)
        {
            throw new NotSupportedException("Scalar supports at most 4 channels.");
        }
    }

    private static void ThrowIfBadLineType(LineType lineType)
    {
        if (lineType is LineType.FourConnected or LineType.EightConnected)
        {
            return;
        }

        if (lineType is LineType.AntiAliased)
        {
            return;
        }

        throw new ArgumentOutOfRangeException(
            nameof(lineType), lineType, "Unsupported line type; use LINE_4, LINE_8 or LINE_AA.");
    }

    /// <summary>
    /// Returns LINE_8 for LINE_AA on non-<see cref="byte"/> images (OpenCV
    /// <c>if (line_type == LINE_AA &amp;&amp; img.depth() != CV_8U) line_type = 8</c>).
    /// </summary>
    private static LineType ResolveLineType<T>(LineType lineType)
        where T : unmanaged =>
        lineType == LineType.AntiAliased && typeof(T) != typeof(byte) ? LineType.EightConnected : lineType;

    private static void ThrowIfBadShift(int shift)
    {
        if (shift < 0 || shift > DrawingConstants.XY_SHIFT)
        {
            throw new ArgumentOutOfRangeException(
                nameof(shift), shift, "shift must be in the range [0, 16].");
        }
    }

    private static void ThrowIfThicknessTooLarge(int thickness)
    {
        if (thickness > DrawingConstants.MAX_THICKNESS)
        {
            throw new ArgumentOutOfRangeException(
                nameof(thickness), thickness, "thickness must not exceed 32767.");
        }
    }
}
