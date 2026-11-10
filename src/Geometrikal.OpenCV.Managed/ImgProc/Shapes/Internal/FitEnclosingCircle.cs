// This implementation is ported from OpenCV's minimum enclosing circle.
//
// Original source:
//   modules/geometry/src/shapedescr.cpp (findCircle3pts, findThirdPoint, findSecondPoint,
//   findMinEnclosingCircle, minEnclosingCircle)
//
// OpenCV source location in this repository:
//   src/OpenCV/modules/geometry/src/shapedescr.cpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Numerics;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Welzl minimum enclosing circle with the seeded shuffle of OpenCV 5.0, over interleaved
/// (x, y) coordinates of type <see cref="int"/> or <see cref="float"/>.
/// </summary>
internal static class FitEnclosingCircle
{
    private const float Eps = 1.0e-4f;

    /// <summary>OpenCV <c>minEnclosingCircle</c>.</summary>
    public static (Point2f Center, float Radius) Compute<T>(ReadOnlySpan<T> xy)
        where T : unmanaged, INumber<T>
    {
        int count = xy.Length / 2;
        if (count == 0)
        {
            return (default, 0f);
        }

        if (count == 1)
        {
            return (new Point2f(F(xy[0]), F(xy[1])), Eps);
        }

        T[] rented = ArrayPool<T>.Shared.Rent(count * 2);
        try
        {
            Span<T> pts = rented.AsSpan(0, count * 2);
            xy.CopyTo(pts);
            if (count > 10)
            {
                uint x0 = BitConverter.SingleToUInt32Bits(F(pts[0]));
                uint y0 = BitConverter.SingleToUInt32Bits(F(pts[1]));
                uint xn = BitConverter.SingleToUInt32Bits(F(pts[(2 * count) - 2]));
                uint yn = BitConverter.SingleToUInt32Bits(F(pts[(2 * count) - 1]));
                uint seed = (uint)count ^ x0 ^ (y0 << 8) ^ (xn << 16) ^ (yn << 24);
                var rng = new CvRng(seed);
                for (int i = 1; i < count; i++)
                {
                    int j = rng.Uniform(0, i + 1);
                    (pts[2 * i], pts[2 * j]) = (pts[2 * j], pts[2 * i]);
                    (pts[(2 * i) + 1], pts[(2 * j) + 1]) = (pts[(2 * j) + 1], pts[(2 * i) + 1]);
                }
            }

            float cx = F(pts[0] + pts[2]) / 2.0f;
            float cy = F(pts[1] + pts[3]) / 2.0f;
            float dx = F(pts[0] - pts[2]);
            float dy = F(pts[1] - pts[3]);
            float radius = ((float)Norm(dx, dy) / 2.0f) + Eps;

            for (int i = 2; i < count; i++)
            {
                dx = F(pts[2 * i]) - cx;
                dy = F(pts[(2 * i) + 1]) - cy;
                float d = (float)Norm(dx, dy);
                if (!(d < radius))
                {
                    FindSecondPoint(pts, i, ref cx, ref cy, ref radius);
                }
            }

            return (new Point2f(cx, cy), radius);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(rented);
        }
    }

    private static void FindSecondPoint<T>(ReadOnlySpan<T> pts, int i, ref float cx, ref float cy, ref float radius)
        where T : unmanaged, INumber<T>
    {
        cx = F(pts[0] + pts[2 * i]) / 2.0f;
        cy = F(pts[1] + pts[(2 * i) + 1]) / 2.0f;
        float dx = F(pts[0] - pts[2 * i]);
        float dy = F(pts[1] - pts[(2 * i) + 1]);
        radius = ((float)Norm(dx, dy) / 2.0f) + Eps;

        for (int j = 1; j < i; j++)
        {
            dx = cx - F(pts[2 * j]);
            dy = cy - F(pts[(2 * j) + 1]);
            if (!(Norm(dx, dy) < radius))
            {
                FindThirdPoint(pts, i, j, ref cx, ref cy, ref radius);
            }
        }
    }

    private static void FindThirdPoint<T>(ReadOnlySpan<T> pts, int i, int j, ref float cx, ref float cy, ref float radius)
        where T : unmanaged, INumber<T>
    {
        cx = F(pts[2 * j] + pts[2 * i]) / 2.0f;
        cy = F(pts[(2 * j) + 1] + pts[(2 * i) + 1]) / 2.0f;
        float dx = F(pts[2 * j] - pts[2 * i]);
        float dy = F(pts[(2 * j) + 1] - pts[(2 * i) + 1]);
        radius = ((float)Norm(dx, dy) / 2.0f) + Eps;

        for (int k = 0; k < j; k++)
        {
            dx = cx - F(pts[2 * k]);
            dy = cy - F(pts[(2 * k) + 1]);
            if (!(Norm(dx, dy) < radius))
            {
                FindCircle3Pts(
                    F(pts[2 * i]), F(pts[(2 * i) + 1]),
                    F(pts[2 * j]), F(pts[(2 * j) + 1]),
                    F(pts[2 * k]), F(pts[(2 * k) + 1]),
                    ref cx, ref cy, ref radius);
            }
        }
    }

    private static void FindCircle3Pts(float p0x, float p0y, float p1x, float p1y, float p2x, float p2y, ref float cx, ref float cy, ref float radius)
    {
        float v1x = p1x - p0x, v1y = p1y - p0y;
        float v2x = p2x - p0x, v2y = p2y - p0y;

        float m1x = (p0x + p1x) / 2.0f, m1y = (p0y + p1y) / 2.0f;
        float c1 = (m1x * v1x) + (m1y * v1y);
        float m2x = (p0x + p2x) / 2.0f, m2y = (p0y + p2y) / 2.0f;
        float c2 = (m2x * v2x) + (m2y * v2y);
        float det = (v1x * v2y) - (v1y * v2x);
        if (MathF.Abs(det) <= Eps)
        {
            // Degenerate triangle: the two-point circle stays.
            return;
        }

        float x = ((c1 * v2y) - (c2 * v1y)) / det;
        float y = ((v1x * c2) - (v2x * c1)) / det;
        cx = x;
        cy = y;
        x -= p0x;
        y -= p0y;
        radius = MathF.Sqrt((x * x) + (y * y)) + Eps;
    }

    /// <summary><c>cv::norm(Point2f)</c>: the length in double.</summary>
    private static double Norm(float x, float y) => Math.Sqrt(((double)x * x) + ((double)y * y));

    private static float F<T>(T v)
        where T : unmanaged, INumber<T> => float.CreateTruncating(v);
}
