// This implementation is ported from the OpenCV shape descriptors.
//
// Original sources:
//   modules/geometry/src/geometry.cpp (contourArea, arcLength, pointSetBoundingRect)
//   modules/geometry/src/moments.cpp  (contourMoments)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Polygon descriptors over point sequences: perimeter, area, Green-theorem moments and the
/// bounding rectangle of a point set.
/// </summary>
internal static class DescrPolygon
{
    /// <summary>C <c>FLT_EPSILON</c> (2^-23).</summary>
    private const double FltEpsilon = 1.1920928955078125e-07;

    /// <summary>
    /// OpenCV <c>arcLength</c>: coordinates convert to <c>float</c>, each segment length is a
    /// <c>float</c> square root and the sum is <c>double</c>.
    /// </summary>
    public static double ArcLength<TP, TOps>(ReadOnlySpan<TP> curve, bool isClosed)
        where TP : unmanaged
        where TOps : struct, IDescrPointOps<TP>
    {
        int count = curve.Length;
        if (count <= 1)
        {
            return 0;
        }

        TP last = curve[isClosed ? count - 1 : 0];
        float prevX = TOps.FX(last), prevY = TOps.FY(last);
        double perimeter = 0;
        for (int i = 0; i < count; i++)
        {
            float x = TOps.FX(curve[i]), y = TOps.FY(curve[i]);
            float dx = x - prevX, dy = y - prevY;
            perimeter += MathF.Sqrt((dx * dx) + (dy * dy));
            prevX = x;
            prevY = y;
        }

        return perimeter;
    }

    /// <summary>
    /// OpenCV <c>contourArea</c>: the shoelace sum over <c>float</c> coordinates in <c>double</c>.
    /// </summary>
    public static double ContourArea<TP, TOps>(ReadOnlySpan<TP> contour, bool oriented)
        where TP : unmanaged
        where TOps : struct, IDescrPointOps<TP>
    {
        int count = contour.Length;
        if (count == 0)
        {
            return 0;
        }

        double a00 = 0;
        float prevX = TOps.FX(contour[count - 1]), prevY = TOps.FY(contour[count - 1]);
        for (int i = 0; i < count; i++)
        {
            float x = TOps.FX(contour[i]), y = TOps.FY(contour[i]);
            a00 += ((double)prevX * y) - ((double)prevY * x);
            prevX = x;
            prevY = y;
        }

        a00 *= 0.5;
        return oriented ? a00 : Math.Abs(a00);
    }

    /// <summary>
    /// OpenCV <c>contourMoments</c>. When <c>|a00| &lt;= FLT_EPSILON</c> every moment is 0.
    /// </summary>
    public static Moments ContourMoments<TP, TOps>(ReadOnlySpan<TP> contour)
        where TP : unmanaged
        where TOps : struct, IDescrPointOps<TP>
    {
        int lpt = contour.Length;
        if (lpt == 0)
        {
            return default;
        }

        double a00 = 0, a10 = 0, a01 = 0, a20 = 0, a11 = 0, a02 = 0, a30 = 0, a21 = 0, a12 = 0, a03 = 0;
        double xi1 = TOps.X(contour[lpt - 1]);
        double yi1 = TOps.Y(contour[lpt - 1]);
        double xi12 = xi1 * xi1;
        double yi12 = yi1 * yi1;

        for (int i = 0; i < lpt; i++)
        {
            double xi = TOps.X(contour[i]);
            double yi = TOps.Y(contour[i]);
            double xi2 = xi * xi;
            double yi2 = yi * yi;
            double dxy = (xi1 * yi) - (xi * yi1);
            double xii1 = xi1 + xi;
            double yii1 = yi1 + yi;

            a00 += dxy;
            a10 += dxy * xii1;
            a01 += dxy * yii1;
            a20 += dxy * ((xi1 * xii1) + xi2);
            a11 += dxy * ((xi1 * (yii1 + yi1)) + (xi * (yii1 + yi)));
            a02 += dxy * ((yi1 * yii1) + yi2);
            a30 += dxy * xii1 * (xi12 + xi2);
            a03 += dxy * yii1 * (yi12 + yi2);
            a21 += dxy * ((xi12 * ((3 * yi1) + yi)) + (2 * xi * xi1 * yii1) + (xi2 * (yi1 + (3 * yi))));
            a12 += dxy * ((yi12 * ((3 * xi1) + xi)) + (2 * yi * yi1 * xii1) + (yi2 * (xi1 + (3 * xi))));
            xi1 = xi;
            yi1 = yi;
            xi12 = xi2;
            yi12 = yi2;
        }

        if (!(Math.Abs(a00) > FltEpsilon))
        {
            return default;
        }

        double db12, db16, db112, db124, db120, db160;
        if (a00 > 0)
        {
            db12 = 0.5;
            db16 = 0.16666666666666666666666666666667;
            db112 = 0.083333333333333333333333333333333;
            db124 = 0.041666666666666666666666666666667;
            db120 = 0.05;
            db160 = 0.016666666666666666666666666666667;
        }
        else
        {
            db12 = -0.5;
            db16 = -0.16666666666666666666666666666667;
            db112 = -0.083333333333333333333333333333333;
            db124 = -0.041666666666666666666666666666667;
            db120 = -0.05;
            db160 = -0.016666666666666666666666666666667;
        }

        return new Moments(
            a00 * db12,
            a10 * db16,
            a01 * db16,
            a20 * db112,
            a11 * db124,
            a02 * db112,
            a30 * db120,
            a21 * db160,
            a12 * db160,
            a03 * db120);
    }

    /// <summary>
    /// OpenCV <c>pointSetBoundingRect</c> for integer points: the inclusive coordinate range plus 1.
    /// </summary>
    public static Rect BoundingRect(ReadOnlySpan<Point> points)
    {
        if (points.IsEmpty)
        {
            return default;
        }

        ReadOnlySpan<int> xy = MemoryMarshal.Cast<Point, int>(points);
        int xmin = xy[0], ymin = xy[1], xmax = xmin, ymax = ymin;
        int n = xy.Length;
        int i = 0;
        if (Vector.IsHardwareAccelerated && n >= Vector<int>.Count && (Vector<int>.Count & 1) == 0)
        {
            ref int r = ref MemoryMarshal.GetReference(xy);
            Vector<int> seed = Vector.AsVectorInt32(new Vector<long>(MemoryMarshal.Read<long>(MemoryMarshal.AsBytes(xy))));
            Vector<int> vmin = seed, vmax = seed;
            for (; i <= n - Vector<int>.Count; i += Vector<int>.Count)
            {
                Vector<int> v = Vector.LoadUnsafe(ref r, (nuint)i);
                vmin = Vector.Min(vmin, v);
                vmax = Vector.Max(vmax, v);
            }

            if (i < n)
            {
                Vector<int> v = Vector.LoadUnsafe(ref r, (nuint)(n - Vector<int>.Count));
                vmin = Vector.Min(vmin, v);
                vmax = Vector.Max(vmax, v);
            }

            for (int k = 0; k < Vector<int>.Count; k += 2)
            {
                xmin = Math.Min(xmin, vmin[k]);
                ymin = Math.Min(ymin, vmin[k + 1]);
                xmax = Math.Max(xmax, vmax[k]);
                ymax = Math.Max(ymax, vmax[k + 1]);
            }

            i = n;
        }

        for (; i < n; i += 2)
        {
            xmin = Math.Min(xmin, xy[i]);
            ymin = Math.Min(ymin, xy[i + 1]);
            xmax = Math.Max(xmax, xy[i]);
            ymax = Math.Max(ymax, xy[i + 1]);
        }

        return new Rect(xmin, ymin, unchecked(xmax - xmin + 1), unchecked(ymax - ymin + 1));
    }

    /// <summary>
    /// OpenCV <c>pointSetBoundingRect</c> for <c>float</c> points: the coordinate range rounds
    /// down with <c>cvFloor</c> on both sides, and the size adds 1.
    /// </summary>
    public static Rect BoundingRect(ReadOnlySpan<Point2f> points)
    {
        if (points.IsEmpty)
        {
            return default;
        }

        ReadOnlySpan<float> xy = MemoryMarshal.Cast<Point2f, float>(points);
        float fxmin = xy[0], fymin = xy[1], fxmax = fxmin, fymax = fymin;
        int n = xy.Length;
        int i = 0;
        if (Vector.IsHardwareAccelerated && n >= Vector<float>.Count && (Vector<float>.Count & 1) == 0)
        {
            ref float r = ref MemoryMarshal.GetReference(xy);
            Vector<float> seed = Vector.AsVectorSingle(new Vector<long>(MemoryMarshal.Read<long>(MemoryMarshal.AsBytes(xy))));
            Vector<float> vmin = seed, vmax = seed;
            for (; i <= n - Vector<float>.Count; i += Vector<float>.Count)
            {
                Vector<float> v = Vector.LoadUnsafe(ref r, (nuint)i);
                vmin = Vector.Min(vmin, v);
                vmax = Vector.Max(vmax, v);
            }

            if (i < n)
            {
                Vector<float> v = Vector.LoadUnsafe(ref r, (nuint)(n - Vector<float>.Count));
                vmin = Vector.Min(vmin, v);
                vmax = Vector.Max(vmax, v);
            }

            for (int k = 0; k < Vector<float>.Count; k += 2)
            {
                fxmin = MinF(fxmin, vmin[k]);
                fymin = MinF(fymin, vmin[k + 1]);
                fxmax = MaxF(fxmax, vmax[k]);
                fymax = MaxF(fymax, vmax[k + 1]);
            }

            i = n;
        }

        for (; i < n; i += 2)
        {
            fxmin = MinF(fxmin, xy[i]);
            fymin = MinF(fymin, xy[i + 1]);
            fxmax = MaxF(fxmax, xy[i]);
            fymax = MaxF(fymax, xy[i + 1]);
        }

        int xmin = ScalarConverter.CvFloor(fxmin), ymin = ScalarConverter.CvFloor(fymin);
        int xmax = ScalarConverter.CvFloor(fxmax), ymax = ScalarConverter.CvFloor(fymax);
        return new Rect(xmin, ymin, unchecked(xmax - xmin + 1), unchecked(ymax - ymin + 1));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float MinF(float a, float b) => b < a ? b : a;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float MaxF(float a, float b) => b > a ? b : a;
}
