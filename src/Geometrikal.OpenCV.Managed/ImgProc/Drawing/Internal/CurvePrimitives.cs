// This implementation is ported from the OpenCV drawing implementation.
//
// Original source:
//   modules/imgproc/src/drawing.cpp   (in this repo: src/OpenCV/modules/imgproc/src/drawing.cpp)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Diagnostics;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Managed port of the curve primitives from OpenCV's <c>drawing.cpp</c>: the
/// integer midpoint circle rasteriser, the thick-line quad/round-cap builder,
/// the poly-line driver, and the elliptic-arc polygon generator (with its
/// <c>sincos</c> lookup table and <c>ellipse2Poly</c> helper).
///
/// <para>All routines mutate <paramref name="img"/> in place and clip silently to
/// the image bounds. Coordinates entering <see cref="Line2"/>, <see cref="Circle"/>
/// caps, <see cref="PolyLine"/>, and <see cref="EllipseEx"/> are OpenCV
/// <c>XY_SHIFT</c> fixed-point unless otherwise noted. All fixed-point arithmetic
/// is unchecked to match the C++ (which disables UBSan on <c>Circle</c>).</para>
/// </summary>
internal static class CurvePrimitives
{
    /// <summary>Upper bound of the <see cref="Ellipse2Poly"/> point count: 360° in 5° steps plus the end point.</summary>
    private const int MaxArcPoints = 76;

    /// <summary>
    /// <c>DBL_EPSILON</c> as spelled in the C++ source; used verbatim (not
    /// <see cref="double.Epsilon"/>, which is a different, far smaller value).
    /// </summary>
    private const double DblEpsilon = 2.2204460492503131e-16;

    /// <summary>
    /// Sine lookup table for whole-degree angles, indexed 0..450, transcribed
    /// verbatim from OpenCV's <c>SinTable</c>. Entry <c>i</c> is <c>sin(i°)</c>;
    /// <c>SinTable[450 - i]</c> therefore yields <c>cos(i°)</c> for
    /// <c>0 &lt;= i &lt;= 360</c>. 451 entries.
    /// </summary>
    private static readonly float[] SinTable =
    {
        0.0000000f, 0.0174524f, 0.0348995f, 0.0523360f, 0.0697565f, 0.0871557f,
        0.1045285f, 0.1218693f, 0.1391731f, 0.1564345f, 0.1736482f, 0.1908090f,
        0.2079117f, 0.2249511f, 0.2419219f, 0.2588190f, 0.2756374f, 0.2923717f,
        0.3090170f, 0.3255682f, 0.3420201f, 0.3583679f, 0.3746066f, 0.3907311f,
        0.4067366f, 0.4226183f, 0.4383711f, 0.4539905f, 0.4694716f, 0.4848096f,
        0.5000000f, 0.5150381f, 0.5299193f, 0.5446390f, 0.5591929f, 0.5735764f,
        0.5877853f, 0.6018150f, 0.6156615f, 0.6293204f, 0.6427876f, 0.6560590f,
        0.6691306f, 0.6819984f, 0.6946584f, 0.7071068f, 0.7193398f, 0.7313537f,
        0.7431448f, 0.7547096f, 0.7660444f, 0.7771460f, 0.7880108f, 0.7986355f,
        0.8090170f, 0.8191520f, 0.8290376f, 0.8386706f, 0.8480481f, 0.8571673f,
        0.8660254f, 0.8746197f, 0.8829476f, 0.8910065f, 0.8987940f, 0.9063078f,
        0.9135455f, 0.9205049f, 0.9271839f, 0.9335804f, 0.9396926f, 0.9455186f,
        0.9510565f, 0.9563048f, 0.9612617f, 0.9659258f, 0.9702957f, 0.9743701f,
        0.9781476f, 0.9816272f, 0.9848078f, 0.9876883f, 0.9902681f, 0.9925462f,
        0.9945219f, 0.9961947f, 0.9975641f, 0.9986295f, 0.9993908f, 0.9998477f,
        1.0000000f, 0.9998477f, 0.9993908f, 0.9986295f, 0.9975641f, 0.9961947f,
        0.9945219f, 0.9925462f, 0.9902681f, 0.9876883f, 0.9848078f, 0.9816272f,
        0.9781476f, 0.9743701f, 0.9702957f, 0.9659258f, 0.9612617f, 0.9563048f,
        0.9510565f, 0.9455186f, 0.9396926f, 0.9335804f, 0.9271839f, 0.9205049f,
        0.9135455f, 0.9063078f, 0.8987940f, 0.8910065f, 0.8829476f, 0.8746197f,
        0.8660254f, 0.8571673f, 0.8480481f, 0.8386706f, 0.8290376f, 0.8191520f,
        0.8090170f, 0.7986355f, 0.7880108f, 0.7771460f, 0.7660444f, 0.7547096f,
        0.7431448f, 0.7313537f, 0.7193398f, 0.7071068f, 0.6946584f, 0.6819984f,
        0.6691306f, 0.6560590f, 0.6427876f, 0.6293204f, 0.6156615f, 0.6018150f,
        0.5877853f, 0.5735764f, 0.5591929f, 0.5446390f, 0.5299193f, 0.5150381f,
        0.5000000f, 0.4848096f, 0.4694716f, 0.4539905f, 0.4383711f, 0.4226183f,
        0.4067366f, 0.3907311f, 0.3746066f, 0.3583679f, 0.3420201f, 0.3255682f,
        0.3090170f, 0.2923717f, 0.2756374f, 0.2588190f, 0.2419219f, 0.2249511f,
        0.2079117f, 0.1908090f, 0.1736482f, 0.1564345f, 0.1391731f, 0.1218693f,
        0.1045285f, 0.0871557f, 0.0697565f, 0.0523360f, 0.0348995f, 0.0174524f,
        0.0000000f, -0.0174524f, -0.0348995f, -0.0523360f, -0.0697565f, -0.0871557f,
        -0.1045285f, -0.1218693f, -0.1391731f, -0.1564345f, -0.1736482f, -0.1908090f,
        -0.2079117f, -0.2249511f, -0.2419219f, -0.2588190f, -0.2756374f, -0.2923717f,
        -0.3090170f, -0.3255682f, -0.3420201f, -0.3583679f, -0.3746066f, -0.3907311f,
        -0.4067366f, -0.4226183f, -0.4383711f, -0.4539905f, -0.4694716f, -0.4848096f,
        -0.5000000f, -0.5150381f, -0.5299193f, -0.5446390f, -0.5591929f, -0.5735764f,
        -0.5877853f, -0.6018150f, -0.6156615f, -0.6293204f, -0.6427876f, -0.6560590f,
        -0.6691306f, -0.6819984f, -0.6946584f, -0.7071068f, -0.7193398f, -0.7313537f,
        -0.7431448f, -0.7547096f, -0.7660444f, -0.7771460f, -0.7880108f, -0.7986355f,
        -0.8090170f, -0.8191520f, -0.8290376f, -0.8386706f, -0.8480481f, -0.8571673f,
        -0.8660254f, -0.8746197f, -0.8829476f, -0.8910065f, -0.8987940f, -0.9063078f,
        -0.9135455f, -0.9205049f, -0.9271839f, -0.9335804f, -0.9396926f, -0.9455186f,
        -0.9510565f, -0.9563048f, -0.9612617f, -0.9659258f, -0.9702957f, -0.9743701f,
        -0.9781476f, -0.9816272f, -0.9848078f, -0.9876883f, -0.9902681f, -0.9925462f,
        -0.9945219f, -0.9961947f, -0.9975641f, -0.9986295f, -0.9993908f, -0.9998477f,
        -1.0000000f, -0.9998477f, -0.9993908f, -0.9986295f, -0.9975641f, -0.9961947f,
        -0.9945219f, -0.9925462f, -0.9902681f, -0.9876883f, -0.9848078f, -0.9816272f,
        -0.9781476f, -0.9743701f, -0.9702957f, -0.9659258f, -0.9612617f, -0.9563048f,
        -0.9510565f, -0.9455186f, -0.9396926f, -0.9335804f, -0.9271839f, -0.9205049f,
        -0.9135455f, -0.9063078f, -0.8987940f, -0.8910065f, -0.8829476f, -0.8746197f,
        -0.8660254f, -0.8571673f, -0.8480481f, -0.8386706f, -0.8290376f, -0.8191520f,
        -0.8090170f, -0.7986355f, -0.7880108f, -0.7771460f, -0.7660444f, -0.7547096f,
        -0.7431448f, -0.7313537f, -0.7193398f, -0.7071068f, -0.6946584f, -0.6819984f,
        -0.6691306f, -0.6560590f, -0.6427876f, -0.6293204f, -0.6156615f, -0.6018150f,
        -0.5877853f, -0.5735764f, -0.5591929f, -0.5446390f, -0.5299193f, -0.5150381f,
        -0.5000000f, -0.4848096f, -0.4694716f, -0.4539905f, -0.4383711f, -0.4226183f,
        -0.4067366f, -0.3907311f, -0.3746066f, -0.3583679f, -0.3420201f, -0.3255682f,
        -0.3090170f, -0.2923717f, -0.2756374f, -0.2588190f, -0.2419219f, -0.2249511f,
        -0.2079117f, -0.1908090f, -0.1736482f, -0.1564345f, -0.1391731f, -0.1218693f,
        -0.1045285f, -0.0871557f, -0.0697565f, -0.0523360f, -0.0348995f, -0.0174524f,
        -0.0000000f, 0.0174524f, 0.0348995f, 0.0523360f, 0.0697565f, 0.0871557f,
        0.1045285f, 0.1218693f, 0.1391731f, 0.1564345f, 0.1736482f, 0.1908090f,
        0.2079117f, 0.2249511f, 0.2419219f, 0.2588190f, 0.2756374f, 0.2923717f,
        0.3090170f, 0.3255682f, 0.3420201f, 0.3583679f, 0.3746066f, 0.3907311f,
        0.4067366f, 0.4226183f, 0.4383711f, 0.4539905f, 0.4694716f, 0.4848096f,
        0.5000000f, 0.5150381f, 0.5299193f, 0.5446390f, 0.5591929f, 0.5735764f,
        0.5877853f, 0.6018150f, 0.6156615f, 0.6293204f, 0.6427876f, 0.6560590f,
        0.6691306f, 0.6819984f, 0.6946584f, 0.7071068f, 0.7193398f, 0.7313537f,
        0.7431448f, 0.7547096f, 0.7660444f, 0.7771460f, 0.7880108f, 0.7986355f,
        0.8090170f, 0.8191520f, 0.8290376f, 0.8386706f, 0.8480481f, 0.8571673f,
        0.8660254f, 0.8746197f, 0.8829476f, 0.8910065f, 0.8987940f, 0.9063078f,
        0.9135455f, 0.9205049f, 0.9271839f, 0.9335804f, 0.9396926f, 0.9455186f,
        0.9510565f, 0.9563048f, 0.9612617f, 0.9659258f, 0.9702957f, 0.9743701f,
        0.9781476f, 0.9816272f, 0.9848078f, 0.9876883f, 0.9902681f, 0.9925462f,
        0.9945219f, 0.9961947f, 0.9975641f, 0.9986295f, 0.9993908f, 0.9998477f,
        1.0000000f,
    };

    /// <summary>A double-precision 2D point used internally by the arc generator.</summary>
    private readonly record struct PointD(double X, double Y);

    /// <summary>
    /// Rasterises a circle with the integer 64-bit midpoint algorithm
    /// (<c>Circle</c>, drawing.cpp lines 1498-1642). When <paramref name="fill"/>
    /// is <c>true</c> the disk interior is filled; otherwise only the outline is
    /// drawn. Coordinates are ordinary pixel coordinates (not fixed-point).
    /// </summary>
    public static void Circle<T>(Mat<T> img, Point center, int radius, ReadOnlySpan<T> color, bool fill)
        where T : unmanaged
    {
        int width = img.Width;
        int height = img.Height;

        long err = 0, dx = radius, dy = 0, plus = 1, minus = (radius << 1) - 1;
        bool inside = center.X >= radius && center.X < width - radius &&
                      center.Y >= radius && center.Y < height - radius;

        while (dx >= dy)
        {
            long y11 = center.Y - dy, y12 = center.Y + dy, y21 = center.Y - dx, y22 = center.Y + dx;
            long x11 = center.X - dx, x12 = center.X + dx, x21 = center.X - dy, x22 = center.X + dy;

            if (inside)
            {
                // The whole circle fits: every row and column index below is
                // guaranteed in range, so spans can be taken without checks.
                if (!fill)
                {
                    PixelOps.PutPixel(img, (int)y11, (int)x11, color);
                    PixelOps.PutPixel(img, (int)y12, (int)x11, color);
                    PixelOps.PutPixel(img, (int)y11, (int)x12, color);
                    PixelOps.PutPixel(img, (int)y12, (int)x12, color);

                    PixelOps.PutPixel(img, (int)y21, (int)x21, color);
                    PixelOps.PutPixel(img, (int)y22, (int)x21, color);
                    PixelOps.PutPixel(img, (int)y21, (int)x22, color);
                    PixelOps.PutPixel(img, (int)y22, (int)x22, color);
                }
                else
                {
                    PixelOps.HLine(img, (int)y11, (int)x11, (int)x12, color);
                    PixelOps.HLine(img, (int)y12, (int)x11, (int)x12, color);
                    PixelOps.HLine(img, (int)y21, (int)x21, (int)x22, color);
                    PixelOps.HLine(img, (int)y22, (int)x21, (int)x22, color);
                }
            }
            else if (x11 < width && x12 >= 0 && y21 < height && y22 >= 0)
            {
                if (fill)
                {
                    x11 = Math.Max(x11, 0L);
                    x12 = Math.Min(x12, (long)(width - 1));
                }

                if (y11 >= 0 && y11 < height)
                {
                    if (!fill)
                    {
                        if (x11 >= 0)
                            PixelOps.PutPixel(img, (int)y11, (int)x11, color);
                        if (x12 < width)
                            PixelOps.PutPixel(img, (int)y11, (int)x12, color);
                    }
                    else
                    {
                        PixelOps.HLine(img, (int)y11, (int)x11, (int)x12, color);
                    }
                }

                if (y12 >= 0 && y12 < height)
                {
                    if (!fill)
                    {
                        if (x11 >= 0)
                            PixelOps.PutPixel(img, (int)y12, (int)x11, color);
                        if (x12 < width)
                            PixelOps.PutPixel(img, (int)y12, (int)x12, color);
                    }
                    else
                    {
                        PixelOps.HLine(img, (int)y12, (int)x11, (int)x12, color);
                    }
                }

                if (x21 < width && x22 >= 0)
                {
                    if (fill)
                    {
                        x21 = Math.Max(x21, 0L);
                        x22 = Math.Min(x22, (long)(width - 1));
                    }

                    if (y21 >= 0 && y21 < height)
                    {
                        if (!fill)
                        {
                            if (x21 >= 0)
                                PixelOps.PutPixel(img, (int)y21, (int)x21, color);
                            if (x22 < width)
                                PixelOps.PutPixel(img, (int)y21, (int)x22, color);
                        }
                        else
                        {
                            PixelOps.HLine(img, (int)y21, (int)x21, (int)x22, color);
                        }
                    }

                    if (y22 >= 0 && y22 < height)
                    {
                        if (!fill)
                        {
                            if (x21 >= 0)
                                PixelOps.PutPixel(img, (int)y22, (int)x21, color);
                            if (x22 < width)
                                PixelOps.PutPixel(img, (int)y22, (int)x22, color);
                        }
                        else
                        {
                            PixelOps.HLine(img, (int)y22, (int)x21, (int)x22, color);
                        }
                    }
                }
            }

            dy++;
            err += plus;
            plus += 2;

            // C++: mask = (err <= 0) - 1  =>  0 when err <= 0, -1 otherwise.
            long mask = (err <= 0) ? 0L : -1L;

            err -= minus & mask;
            dx += mask;
            minus -= mask & 2L;
        }
    }

    /// <summary>
    /// Draws a (possibly thick, possibly capped) line segment
    /// (<c>ThickLine</c>, drawing.cpp lines 1645-1732). Endpoints are
    /// <c>XY_SHIFT</c> fixed-point scaled by <paramref name="shift"/>.
    /// <paramref name="flags"/> bit&#160;1 requests a round cap at
    /// <paramref name="p0"/>, bit&#160;2 a round cap at <paramref name="p1"/>.
    /// Thin lines (<paramref name="thickness"/> &lt;= 1) delegate to the line
    /// primitives; thick lines are drawn as a filled quad with optional filled
    /// circular caps.
    /// </summary>
    public static void ThickLine<T>(Mat<T> img, PointL p0, PointL p1, ReadOnlySpan<T> color,
        int thickness, int lineType, int flags, int shift)
        where T : unmanaged
    {
        double invXyOne = 1.0 / DrawingConstants.XY_ONE;

        long width = img.Width;
        long height = img.Height;
        long p0x = p0.X, p0y = p0.Y, p1x = p1.X, p1y = p1.Y;

        if (thickness > 1 && shift == 0 &&
            (!Contains(width, height, p0x, p0y) || !Contains(width, height, p1x, p1y)))
        {
            long margin = thickness;
            p0x += margin; p0y += margin;
            p1x += margin; p1y += margin;
            LinePrimitives.ClipLine(width + (2 * margin), height + (2 * margin),
                ref p0x, ref p0y, ref p1x, ref p1y);
            p0x -= margin; p0y -= margin;
            p1x -= margin; p1y -= margin;
        }

        int up = DrawingConstants.XY_SHIFT - shift;
        p0x <<= up; p0y <<= up; p1x <<= up; p1y <<= up;

        if (thickness <= 1)
        {
            if (lineType >= AntiAliasedLine.LineAa)
            {
                AntiAliasedLine.LineAA(img, new PointL(p0x, p0y), new PointL(p1x, p1y), color);
            }
            else if (lineType == 1 || lineType == 8 || shift == 0)
            {
                int rx0 = (int)((p0x + (DrawingConstants.XY_ONE >> 1)) >> DrawingConstants.XY_SHIFT);
                int ry0 = (int)((p0y + (DrawingConstants.XY_ONE >> 1)) >> DrawingConstants.XY_SHIFT);
                int rx1 = (int)((p1x + (DrawingConstants.XY_ONE >> 1)) >> DrawingConstants.XY_SHIFT);
                int ry1 = (int)((p1y + (DrawingConstants.XY_ONE >> 1)) >> DrawingConstants.XY_SHIFT);
                LinePrimitives.Line(img, new Point(rx0, ry0), new Point(rx1, ry1), color, lineType);
            }
            else
            {
                LinePrimitives.Line2(img, new PointL(p0x, p0y), new PointL(p1x, p1y), color);
            }
        }
        else
        {
            double dx = (p0x - p1x) * invXyOne;
            double dy = (p1y - p0y) * invXyOne;
            double r = (dx * dx) + (dy * dy);
            int oddThickness = thickness & 1;
            long thick = (long)thickness << (DrawingConstants.XY_SHIFT - 1);

            if (Math.Abs(r) > DblEpsilon)
            {
                r = (thick + (oddThickness * DrawingConstants.XY_ONE * 0.5)) / Math.Sqrt(r);
                long dpx = (long)Math.Round(dy * r, MidpointRounding.ToEven);
                long dpy = (long)Math.Round(dx * r, MidpointRounding.ToEven);

                Span<PointL> pt = stackalloc PointL[4];
                pt[0] = new PointL(p0x + dpx, p0y + dpy);
                pt[1] = new PointL(p0x - dpx, p0y - dpy);
                pt[2] = new PointL(p1x - dpx, p1y - dpy);
                pt[3] = new PointL(p1x + dpx, p1y + dpy);
                PolygonPrimitives.FillConvexPoly(img, pt, color, lineType, DrawingConstants.XY_SHIFT);
            }

            for (int i = 0; i < 2; i++)
            {
                if ((flags & (i + 1)) != 0 && lineType >= AntiAliasedLine.LineAa)
                {
                    EllipseEx(img, new PointL(p0x, p0y), thick, thick, 0, 0, 360, color, -1, lineType);
                }
                else if ((flags & (i + 1)) != 0)
                {
                    int cx = (int)((p0x + (DrawingConstants.XY_ONE >> 1)) >> DrawingConstants.XY_SHIFT);
                    int cy = (int)((p0y + (DrawingConstants.XY_ONE >> 1)) >> DrawingConstants.XY_SHIFT);
                    int capRadius = (int)((thick + (DrawingConstants.XY_ONE >> 1)) >> DrawingConstants.XY_SHIFT);
                    Circle(img, new Point(cx, cy), capRadius, color, fill: true);
                }

                p0x = p1x;
                p0y = p1y;
            }
        }
    }

    /// <summary>
    /// Draws a connected sequence of thick-line segments through
    /// <paramref name="v"/> (<c>PolyLine</c>, drawing.cpp lines 1735-1756).
    /// Vertices are <c>XY_SHIFT</c> fixed-point scaled by <paramref name="shift"/>.
    /// When <paramref name="isClosed"/> is <c>true</c> the last vertex is joined
    /// back to the first. Round caps are placed at interior joints so consecutive
    /// segments connect cleanly.
    /// </summary>
    public static void PolyLine<T>(Mat<T> img, ReadOnlySpan<PointL> v, bool isClosed,
        ReadOnlySpan<T> color, int thickness, int lineType, int shift)
        where T : unmanaged
    {
        if (v.IsEmpty)
            return;

        int count = v.Length;
        int i = isClosed ? count - 1 : 0;
        int flags = 2 + (isClosed ? 0 : 1);
        PointL p0 = v[i];

        for (i = isClosed ? 0 : 1; i < count; i++)
        {
            PointL p = v[i];
            ThickLine(img, p0, p, color, thickness, lineType, flags, shift);
            p0 = p;
            flags = 2;
        }
    }

    /// <summary>
    /// Draws or fills an elliptic arc (<c>EllipseEx</c>, drawing.cpp lines
    /// 1013-1056). The centre and semi-axes are <c>XY_SHIFT</c> fixed-point; the
    /// angles are whole degrees. A non-negative <paramref name="thickness"/> draws
    /// the (open) arc outline; a negative thickness fills it — a full 360° arc via
    /// a convex-polygon fill, a partial arc as a pie sector via the edge
    /// collection.
    /// </summary>
    public static void EllipseEx<T>(Mat<T> img, PointL center, long axisWidth, long axisHeight,
        int angle, int arcStart, int arcEnd, ReadOnlySpan<T> color, int thickness, int lineType)
        where T : unmanaged
    {
        long axW = Math.Abs(axisWidth);
        long axH = Math.Abs(axisHeight);
        int delta = (int)((Math.Max(axW, axH) + (DrawingConstants.XY_ONE >> 1)) >> DrawingConstants.XY_SHIFT);
        delta = delta < 3 ? 90 : delta < 10 ? 30 : delta < 15 ? 18 : 5;

        // The C++ passes the raw fixed-point axis/centre values straight through
        // as doubles; the two-step rounding below folds them back afterwards.
        Span<PointD> poly = stackalloc PointD[MaxArcPoints];
        int polyCount = Ellipse2Poly((double)center.X, (double)center.Y, (double)axW, (double)axH,
            angle, arcStart, arcEnd, delta, poly);

        Span<PointL> v = stackalloc PointL[MaxArcPoints + 1];
        int count = 0;
        PointL prevPt = new(-1, -1); // C++ Point2l(0xFFFF...FFFF, 0xFFFF...FFFF)
        for (int i = 0; i < polyCount; i++)
        {
            long px = (long)Math.Round(poly[i].X / DrawingConstants.XY_ONE, MidpointRounding.ToEven) << DrawingConstants.XY_SHIFT;
            long py = (long)Math.Round(poly[i].Y / DrawingConstants.XY_ONE, MidpointRounding.ToEven) << DrawingConstants.XY_SHIFT;
            px += (long)Math.Round(poly[i].X - px, MidpointRounding.ToEven);
            py += (long)Math.Round(poly[i].Y - py, MidpointRounding.ToEven);
            var pt = new PointL(px, py);
            if (pt != prevPt)
            {
                v[count++] = pt;
                prevPt = pt;
            }
        }

        // A zero-size polygon collapses to two copies of the centre.
        if (count <= 1)
        {
            v[0] = center;
            v[1] = center;
            count = 2;
        }

        if (thickness >= 0)
        {
            PolyLine(img, v[..count], isClosed: false, color, thickness, lineType, DrawingConstants.XY_SHIFT);
        }
        else if (arcEnd - arcStart >= 360)
        {
            PolygonPrimitives.FillConvexPoly(img, v[..count], color, lineType, DrawingConstants.XY_SHIFT);
        }
        else
        {
            v[count++] = center;
            var edges = new List<PolyEdge>();
            PolygonPrimitives.CollectPolyEdges(img, v[..count], edges, color, lineType, DrawingConstants.XY_SHIFT, default);
            PolygonPrimitives.FillEdgeCollection(img, edges, color);
        }
    }

    /// <summary>
    /// Bounds-check equivalent to OpenCV's <c>Rect_&lt;int64&gt;(0,0,w,h).contains(p)</c>:
    /// the half-open range [0, width) x [0, height).
    /// </summary>
    private static bool Contains(long width, long height, long x, long y) =>
        x >= 0 && x < width && y >= 0 && y < height;

    /// <summary>
    /// Looks up cosine and sine of a whole-degree angle from <see cref="SinTable"/>
    /// (<c>sincos</c>, drawing.cpp lines 911-917). Negative angles are wrapped by
    /// adding 360°.
    /// </summary>
    private static void SinCos(int angle, out float cosval, out float sinval)
    {
        angle += angle < 0 ? 360 : 0;
        sinval = SinTable[angle];
        cosval = SinTable[450 - angle];
    }

    /// <summary>
    /// Generates the vertices of an elliptic arc as double-precision points
    /// (the <c>Point2d</c> overload of <c>ellipse2Poly</c>, drawing.cpp lines
    /// 949-1010). The result is written to <paramref name="pts"/>.
    /// </summary>
    /// <returns>The number of points written.</returns>
    private static int Ellipse2Poly(double centerX, double centerY, double axesWidth, double axesHeight,
        int angle, int arcStart, int arcEnd, int delta, Span<PointD> pts)
    {
        Debug.Assert(delta > 0 && delta <= 180, "delta must be in (0, 180]");

        while (angle < 0)
            angle += 360;
        while (angle > 360)
            angle -= 360;

        if (arcStart > arcEnd)
        {
            (arcStart, arcEnd) = (arcEnd, arcStart);
        }

        while (arcStart < 0)
        {
            arcStart += 360;
            arcEnd += 360;
        }

        while (arcEnd > 360)
        {
            arcEnd -= 360;
            arcStart -= 360;
        }

        if (arcEnd - arcStart > 360)
        {
            arcStart = 0;
            arcEnd = 360;
        }

        SinCos(angle, out float alpha, out float beta);
        int count = 0;

        for (int i = arcStart; i < arcEnd + delta; i += delta)
        {
            int a = i;
            if (a > arcEnd)
                a = arcEnd;
            if (a < 0)
                a += 360;

            double x = axesWidth * SinTable[450 - a];
            double y = axesHeight * SinTable[a];
            pts[count++] = new PointD(
                centerX + (x * alpha) - (y * beta),
                centerY + (x * beta) + (y * alpha));
        }

        Debug.Assert(count != 0, "arc polygon must not be empty");

        // A single point means a zero-size polygon: emit two copies of the centre.
        if (count == 1)
        {
            var c = new PointD(centerX, centerY);
            pts[0] = c;
            pts[1] = c;
            count = 2;
        }

        return count;
    }
}
