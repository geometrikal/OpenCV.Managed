// This implementation is ported from the OpenCV drawing implementation.
//
// Original sources:
//   modules/imgproc/src/drawing.cpp              (in this repo: src/OpenCV/modules/imgproc/src/drawing.cpp)
//   modules/imgproc/include/opencv2/imgproc.hpp  (LineIterator::operator++)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Diagnostics;
using System.Runtime.CompilerServices;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Non-anti-aliased line primitives ported from OpenCV's <c>drawing.cpp</c>:
/// line clipping (<c>clipLine</c>), integer Bresenham lines via the
/// <c>LineIterator</c> traversal, and the fixed-point <c>Line2</c>.
///
/// <para>All fixed-point arithmetic is intentionally unchecked (the project
/// default): the fills and sub-pixel steps rely on C-style wrap/truncation
/// semantics. Every pixel write is either preceded by clipping to the image
/// bounds or individually bounds-guarded, matching the C++.</para>
/// </summary>
internal static class LinePrimitives
{
    /// <summary>
    /// Clips the line segment (<paramref name="x1"/>,<paramref name="y1"/>)–
    /// (<paramref name="x2"/>,<paramref name="y2"/>) against the rectangle
    /// [0,<paramref name="width"/>) x [0,<paramref name="height"/>), updating the
    /// endpoints in place. Ports OpenCV's <c>clipLine(Size2l, Point2l&amp;, Point2l&amp;)</c>:
    /// intersections are computed in double precision then truncated toward zero.
    /// </summary>
    /// <returns><c>true</c> when any part of the segment is visible.</returns>
    public static bool ClipLine(long width, long height, ref long x1, ref long y1, ref long x2, ref long y2)
    {
        if (width <= 0 || height <= 0)
        {
            return false;
        }

        long right = width - 1;
        long bottom = height - 1;

        int c1 = (x1 < 0 ? 1 : 0) + (x1 > right ? 2 : 0) + (y1 < 0 ? 4 : 0) + (y1 > bottom ? 8 : 0);
        int c2 = (x2 < 0 ? 1 : 0) + (x2 > right ? 2 : 0) + (y2 < 0 ? 4 : 0) + (y2 > bottom ? 8 : 0);

        if ((c1 & c2) == 0 && (c1 | c2) != 0)
        {
            long a;
            if ((c1 & 12) != 0)
            {
                a = c1 < 8 ? 0 : bottom;
                x1 += (long)((double)(a - y1) * (x2 - x1) / (y2 - y1));
                y1 = a;
                c1 = (x1 < 0 ? 1 : 0) + (x1 > right ? 2 : 0);
            }

            if ((c2 & 12) != 0)
            {
                a = c2 < 8 ? 0 : bottom;
                x2 += (long)((double)(a - y2) * (x2 - x1) / (y2 - y1));
                y2 = a;
                c2 = (x2 < 0 ? 1 : 0) + (x2 > right ? 2 : 0);
            }

            if ((c1 & c2) == 0 && (c1 | c2) != 0)
            {
                if (c1 != 0)
                {
                    a = c1 == 1 ? 0 : right;
                    y1 += (long)((double)(a - x1) * (y2 - y1) / (x2 - x1));
                    x1 = a;
                    c1 = 0;
                }

                if (c2 != 0)
                {
                    a = c2 == 1 ? 0 : right;
                    y2 += (long)((double)(a - x2) * (y2 - y1) / (x2 - x1));
                    x2 = a;
                    c2 = 0;
                }
            }

            Debug.Assert((c1 & c2) != 0 || (x1 | y1 | x2 | y2) >= 0);
        }

        return (c1 | c2) == 0;
    }

    /// <summary>
    /// Integer <see cref="int"/> wrapper over
    /// <see cref="ClipLine(long, long, ref long, ref long, ref long, ref long)"/>,
    /// mirroring OpenCV's <c>clipLine(Size, Point&amp;, Point&amp;)</c>.
    /// </summary>
    /// <returns><c>true</c> when any part of the segment is visible.</returns>
    public static bool ClipLine(int width, int height, ref int x1, ref int y1, ref int x2, ref int y2)
    {
        long lx1 = x1, ly1 = y1, lx2 = x2, ly2 = y2;
        bool inside = ClipLine(width, height, ref lx1, ref ly1, ref lx2, ref ly2);
        x1 = (int)lx1;
        y1 = (int)ly1;
        x2 = (int)lx2;
        y2 = (int)ly2;
        return inside;
    }

    /// <summary>
    /// Draws an integer Bresenham line from <paramref name="pt1"/> to
    /// <paramref name="pt2"/> using the <c>LineIterator</c> traversal (coordinate
    /// mode), ports OpenCV's static <c>Line()</c>. Connectivity <c>0</c> maps to
    /// 8-connected and <c>1</c> to 4-connected (as in C++); the values <c>4</c>
    /// and <c>8</c> pass through unchanged. The line is clipped to the image and
    /// mutates <paramref name="img"/> in place.
    /// </summary>
    public static void Line<T>(Mat<T> img, Point pt1, Point pt2, ReadOnlySpan<T> color, int connectivity)
        where T : unmanaged
    {
        if (connectivity == 0)
        {
            connectivity = 8;
        }
        else if (connectivity == 1)
        {
            connectivity = 4;
        }

        Debug.Assert(connectivity == 8 || connectivity == 4);

        int width = img.Width;
        int height = img.Height;

        // LineIterator::init in coordinate mode, bounding rect (0,0,width,height),
        // leftToRight == true (the value C++ Line() always passes).
        int x1 = pt1.X, y1 = pt1.Y;
        int x2 = pt2.X, y2 = pt2.Y;

        // Clip only when an endpoint is outside (the (unsigned)pt.x >= width test).
        if ((uint)x1 >= (uint)width || (uint)x2 >= (uint)width ||
            (uint)y1 >= (uint)height || (uint)y2 >= (uint)height)
        {
            if (!ClipLine(width, height, ref x1, ref y1, ref x2, ref y2))
            {
                return; // count == 0
            }
        }

        int deltaX = 1, deltaY = 1;
        int dx = x2 - x1;
        int dy = y2 - y1;

        if (dx < 0)
        {
            // leftToRight: traverse from the leftmost endpoint.
            dx = -dx;
            dy = -dy;
            x1 = x2;
            y1 = y2;
        }

        if (dy < 0)
        {
            dy = -dy;
            deltaY = -1;
        }

        bool vert = dy > dx;
        if (vert)
        {
            (dx, dy) = (dy, dx);
            (deltaX, deltaY) = (deltaY, deltaX);
        }

        Debug.Assert(dx >= 0 && dy >= 0);

        int err, plusDelta, minusDelta, plusStep, minusStep, plusShift, minusShift, count;

        if (connectivity == 8)
        {
            err = dx - (dy + dy);
            plusDelta = dx + dx;
            minusDelta = -(dy + dy);
            minusShift = deltaX;
            plusShift = 0;
            minusStep = 0;
            plusStep = deltaY;
            count = dx + 1;
        }
        else /* connectivity == 4 */
        {
            err = 0;
            plusDelta = (dx + dx) + (dy + dy);
            minusDelta = -(dy + dy);
            minusShift = deltaX;
            plusShift = -deltaX;
            minusStep = 0;
            plusStep = deltaY;
            count = dx + dy + 1;
        }

        if (vert)
        {
            (plusStep, plusShift) = (plusShift, plusStep);
            (minusStep, minusShift) = (minusShift, minusStep);
        }

        // Coordinate mode: p.x advances by the shift terms, p.y by the step terms.
        int px = x1, py = y1;
        for (int i = 0; i < count; i++)
        {
            PixelOps.PutPixel(img, py, px, color);

            int mask = err < 0 ? -1 : 0;
            err += minusDelta + (plusDelta & mask);
            px += minusShift + (plusShift & mask);
            py += minusStep + (plusStep & mask);
        }
    }

    /// <summary>
    /// Draws a line whose endpoints are in <see cref="DrawingConstants.XY_SHIFT"/>
    /// fixed-point coordinates, ports OpenCV's static <c>Line2()</c>. The segment
    /// is first clipped against the fixed-point image bounds; the branchless
    /// XOR-mask swaps, the <c>pt2</c>-first write, and the per-pixel bounds guards
    /// are preserved. The three C++ element-size variants collapse to one generic
    /// loop that writes the identical pixel set. Mutates <paramref name="img"/> in place.
    /// </summary>
    public static void Line2<T>(Mat<T> img, PointL pt1, PointL pt2, ReadOnlySpan<T> color)
        where T : unmanaged
    {
        const int shift = DrawingConstants.XY_SHIFT;
        const int one = DrawingConstants.XY_ONE;

        int width = img.Width;
        int height = img.Height;

        long x1 = pt1.X, y1 = pt1.Y, x2 = pt2.X, y2 = pt2.Y;

        long scaledWidth = (long)width << shift;
        long scaledHeight = (long)height << shift;
        if (!ClipLine(scaledWidth, scaledHeight, ref x1, ref y1, ref x2, ref y2))
        {
            return;
        }

        long dx = x2 - x1;
        long dy = y2 - y1;

        long j = dx < 0 ? -1 : 0;
        long ax = (dx ^ j) - j;
        long i = dy < 0 ? -1 : 0;
        long ay = (dy ^ i) - i;

        long xStep, yStep;
        int ecount;

        if (ax > ay)
        {
            dy = (dy ^ j) - j;
            x1 ^= x2 & j; x2 ^= x1 & j; x1 ^= x2 & j;
            y1 ^= y2 & j; y2 ^= y1 & j; y1 ^= y2 & j;

            xStep = one;
            yStep = dy * one / (ax | 1);
            ecount = (int)((x2 - x1) >> shift);
        }
        else
        {
            dx = (dx ^ i) - i;
            x1 ^= x2 & i; x2 ^= x1 & i; x1 ^= x2 & i;
            y1 ^= y2 & i; y2 ^= y1 & i; y1 ^= y2 & i;

            xStep = dx * one / (ay | 1);
            yStep = one;
            ecount = (int)((y2 - y1) >> shift);
        }

        x1 += one >> 1;
        y1 += one >> 1;

        // The pt2 endpoint is written FIRST (rounded), as in the C++.
        PutPointGuarded(img, (int)((x2 + (one >> 1)) >> shift), (int)((y2 + (one >> 1)) >> shift), color, width, height);

        if (ax > ay)
        {
            x1 >>= shift;
            while (ecount >= 0)
            {
                PutPointGuarded(img, (int)x1, (int)(y1 >> shift), color, width, height);
                x1++;
                y1 += yStep;
                ecount--;
            }
        }
        else
        {
            y1 >>= shift;
            while (ecount >= 0)
            {
                PutPointGuarded(img, (int)(x1 >> shift), (int)y1, color, width, height);
                x1 += xStep;
                y1++;
                ecount--;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void PutPointGuarded<T>(Mat<T> img, int x, int y, ReadOnlySpan<T> color, int width, int height)
        where T : unmanaged
    {
        if (0 <= x && x < width && 0 <= y && y < height)
        {
            PixelOps.PutPixel(img, y, x, color);
        }
    }
}
