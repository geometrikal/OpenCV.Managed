// This implementation is ported from the OpenCV polygon intersection functions.
//
// Original sources:
//   modules/geometry/src/geometry.cpp (intersectConvexConvex and its O'Rourke helpers)
//   modules/geometry/src/intersection.cpp (rotatedRectangleIntersection)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using Geometrikal.OpenCV.Managed.Core;

/// <summary>Result type of <see cref="ShapeIntersection.RotatedRectangleIntersection"/>, as OpenCV <c>RectanglesIntersectTypes</c>.</summary>
public enum RectanglesIntersectType
{
    /// <summary>No intersection (<c>INTERSECT_NONE</c>).</summary>
    None = 0,

    /// <summary>A partial intersection (<c>INTERSECT_PARTIAL</c>).</summary>
    Partial = 1,

    /// <summary>One rectangle is inside the other, or they are equal (<c>INTERSECT_FULL</c>).</summary>
    Full = 2,
}

/// <summary>Intersections of convex polygons and rotated rectangles, as OpenCV 5.0 (geometry module).</summary>
public static class ShapeIntersection
{
    /// <summary>
    /// Intersects two convex polygons, as OpenCV <c>intersectConvexConvex</c> (O'Rourke's algorithm).
    /// </summary>
    /// <param name="p1">First convex polygon.</param>
    /// <param name="p2">Second convex polygon.</param>
    /// <param name="handleNested">When the boundaries do not cross, test whether one polygon contains the other.</param>
    /// <returns>
    /// The absolute area of the intersection and its vertices. The area is 0 with no vertices when the
    /// polygons do not intersect or have fewer than 2 vertices; it is −1 with no vertices when the
    /// algorithm does not converge (for example, for a non-convex input).
    /// </returns>
    /// <remarks>
    /// Each polygon is reversed when its first turn is clockwise in a Y-up frame (OpenCV
    /// "orientation"). When both are reversed, the result is reversed back.
    /// </remarks>
    public static (float Area, Point2f[] Intersection) IntersectConvexConvex(ReadOnlySpan<Point2f> p1, ReadOnlySpan<Point2f> p2, bool handleNested = true)
    {
        int n = p1.Length, m = p2.Length;
        if (n < 2 || m < 2)
        {
            return (0f, []);
        }

        var fp1 = p1.ToArray();
        var fp2 = p2.ToArray();
        int orientation = Orient(fp1) + Orient(fp2);

        int nr = IntersectCore(fp1, fp2, out var result, out float area);
        if (nr < 0)
        {
            return (-1f, []);
        }

        Point2f[] region;
        if (nr == 0)
        {
            if (!handleNested)
            {
                return (0f, []);
            }

            if (fp2.Count(pt => ShapeFitting.PointPolygonTest(fp1, pt, false) >= 0) == m)
            {
                region = fp2;
            }
            else if (fp1.Count(pt => ShapeFitting.PointPolygonTest(fp2, pt, false) >= 0) == n)
            {
                region = fp1;
            }
            else
            {
                return (0f, []);
            }

            area = (float)ShapeDescriptors.ContourArea(region, false);
            region = (Point2f[])region.Clone();
        }
        else
        {
            region = result.AsSpan(0, nr).ToArray();
        }

        if (orientation == 2)
        {
            Array.Reverse(region);
        }

        return (MathF.Abs(area), region);
    }

    /// <inheritdoc cref="IntersectConvexConvex(ReadOnlySpan{Point2f}, ReadOnlySpan{Point2f}, bool)"/>
    public static (float Area, Point2f[] Intersection) IntersectConvexConvex(ReadOnlySpan<Point> p1, ReadOnlySpan<Point> p2, bool handleNested = true)
    {
        var f1 = new Point2f[p1.Length];
        var f2 = new Point2f[p2.Length];
        for (int i = 0; i < f1.Length; i++)
        {
            f1[i] = new Point2f(p1[i].X, p1[i].Y);
        }

        for (int i = 0; i < f2.Length; i++)
        {
            f2[i] = new Point2f(p2[i].X, p2[i].Y);
        }

        return IntersectConvexConvex(f1, f2, handleNested);
    }

    /// <summary>
    /// Intersects two rotated rectangles, as OpenCV <c>rotatedRectangleIntersection</c>.
    /// </summary>
    /// <param name="rect1">First rectangle.</param>
    /// <param name="rect2">Second rectangle.</param>
    /// <returns>The intersection type and the vertices of the intersection region (at most 8, in order).</returns>
    /// <remarks>
    /// The rectangles are shifted by their mean centre before the float computation and the vertices
    /// are shifted back, as OpenCV. Vertices closer than an adaptive tolerance are merged.
    /// </remarks>
    public static (RectanglesIntersectType Type, Point2f[] Region) RotatedRectangleIntersection(RotatedRect rect1, RotatedRect rect2)
    {
        if (Empty(rect1.Size) || Empty(rect2.Size))
        {
            return (RectanglesIntersectType.None, []);
        }

        var avg = new Point2f((rect1.Center.X + rect2.Center.X) / 2.0f, (rect1.Center.Y + rect2.Center.Y) / 2.0f);
        var s1 = rect1 with { Center = new Point2f(rect1.Center.X - avg.X, rect1.Center.Y - avg.Y) };
        var s2 = rect2 with { Center = new Point2f(rect2.Center.X - avg.X, rect2.Center.Y - avg.Y) };
        var points = new List<Point2f>(24);
        var type = RectangleIntersectionCore(s1, s2, points);
        if (type == RectanglesIntersectType.None)
        {
            return (type, []);
        }

        var region = new Point2f[points.Count];
        for (int i = 0; i < region.Length; i++)
        {
            region[i] = new Point2f(points[i].X + avg.X, points[i].Y + avg.Y);
        }

        return (type, region);
    }

    private static bool Empty(Size2f s) => s.Width <= 0 || s.Height <= 0;

    private static Point2f Sub(Point2f a, Point2f b) => new(a.X - b.X, a.Y - b.Y);

    private static double Cross(Point2f a, Point2f b) => ((double)a.X * b.Y) - ((double)a.Y * b.X);

    // Reverses a polygon whose first non-zero turn is negative; returns 1 when reversed.
    private static int Orient(Point2f[] dst)
    {
        int len = dst.Length;
        var diff0 = Sub(dst[0], dst[len - 1]);
        for (int i = 1; i < len; i++)
        {
            double s = Cross(diff0, Sub(dst[i], dst[i - 1]));
            if (s != 0)
            {
                if (s < 0)
                {
                    Array.Reverse(dst);
                    return 1;
                }

                break;
            }
        }

        return 0;
    }

    private enum InFlag
    {
        Pin,
        Qin,
        Unknown,
    }

    private enum SegmentIntersection
    {
        None = 0,
        Single = 1,
        Overlap = 2,
        Endpoint = 3,
    }

    private static int AreaSign(Point2f a, Point2f b, Point2f c)
    {
        const double eps = 1e-5;
        double area2 = ((b.X - a.X) * (double)(c.Y - a.Y)) - ((c.X - a.X) * (double)(b.Y - a.Y));
        return area2 > eps ? 1 : area2 < -eps ? -1 : 0;
    }

    private static bool Between(Point2f a, Point2f b, Point2f c) =>
        a.X != b.X
            ? (a.X <= c.X && c.X <= b.X) || (a.X >= c.X && c.X >= b.X)
            : (a.Y <= c.Y && c.Y <= b.Y) || (a.Y >= c.Y && c.Y >= b.Y);

    private static SegmentIntersection ParallelInt(Point2f a, Point2f b, Point2f c, Point2f d, ref Point2f p, ref Point2f q)
    {
        if (AreaSign(a, b, c) != 0)
        {
            return SegmentIntersection.None;
        }

        if (Between(a, b, c) && Between(a, b, d))
        {
            (p, q) = (c, d);
        }
        else if (Between(c, d, a) && Between(c, d, b))
        {
            (p, q) = (a, b);
        }
        else if (Between(a, b, c) && Between(c, d, b))
        {
            (p, q) = (c, b);
        }
        else if (Between(a, b, c) && Between(c, d, a))
        {
            (p, q) = (c, a);
        }
        else if (Between(a, b, d) && Between(c, d, b))
        {
            (p, q) = (d, b);
        }
        else if (Between(a, b, d) && Between(c, d, a))
        {
            (p, q) = (d, a);
        }
        else
        {
            return SegmentIntersection.None;
        }

        return SegmentIntersection.Overlap;
    }

    private static SegmentIntersection IntersectSegments(Point2f a, Point2f b, Point2f c, Point2f d, ref Point2f p, ref Point2f q)
    {
        double denom = (((double)a.X - b.X) * ((double)d.Y - c.Y)) - (((double)a.Y - b.Y) * ((double)d.X - c.X));
        if (denom == 0.0)
        {
            return ParallelInt(a, b, c, d, ref p, ref q);
        }

        double num = (((double)d.Y - a.Y) * ((double)a.X - c.X)) + (((double)a.X - d.X) * ((double)a.Y - c.Y));
        double s = num / denom;
        num = (((double)b.Y - a.Y) * ((double)a.X - c.X)) + (((double)c.Y - a.Y) * ((double)b.X - a.X));
        double t = num / denom;
        p = new Point2f((float)(a.X + (s * ((double)b.X - a.X))), (float)(a.Y + (s * ((double)b.Y - a.Y))));
        q = p;
        if (s < 0.0 || s > 1.0 || t < 0.0 || t > 1.0)
        {
            return SegmentIntersection.None;
        }

        return s == 0.0 || s == 1.0 || t == 0.0 || t == 1.0 ? SegmentIntersection.Endpoint : SegmentIntersection.Single;
    }

    // OpenCV intersectConvexConvex_: result[0] is a sentinel; returns the vertex count or -1.
    private static int IntersectCore(Point2f[] p, Point2f[] q, out Point2f[] result0, out float areaOut)
    {
        int n = p.Length, m = q.Length;
        int resultSize = n + m + 1;
        result0 = new Point2f[resultSize + 4];
        int r = 0;
        int a = 0, b = 0, aa = 0, ba = 0;
        var inflag = InFlag.Unknown;
        bool firstPoint = true;
        var origin = new Point2f(0, 0);
        result0[r++] = new Point2f(float.MaxValue, float.MaxValue);
        var buf = result0;

        void Push(Point2f v)
        {
            if (v != buf[r - 1])
            {
                buf[r++] = v;
            }
        }

        int Advance(int idx, ref int count, int len, bool inside, Point2f v)
        {
            if (inside)
            {
                Push(v);
            }

            count++;
            return (idx + 1) % len;
        }

        do
        {
            int a1 = (a + n - 1) % n;
            int b1 = (b + m - 1) % m;
            var aDir = Sub(p[a], p[a1]);
            var bDir = Sub(q[b], q[b1]);
            int cross = AreaSign(origin, aDir, bDir);
            int aHB = AreaSign(q[b1], q[b], p[a]);
            int bHA = AreaSign(p[a1], p[a], q[b]);

            Point2f ip = default, iq = default;
            var code = IntersectSegments(p[a1], p[a], q[b1], q[b], ref ip, ref iq);
            if (code is SegmentIntersection.Single or SegmentIntersection.Endpoint)
            {
                if (inflag == InFlag.Unknown && firstPoint)
                {
                    aa = ba = 0;
                    firstPoint = false;
                    buf[r++] = ip;
                }

                Push(ip);
                inflag = aHB > 0 ? InFlag.Pin : bHA > 0 ? InFlag.Qin : inflag;
            }

            if (code == SegmentIntersection.Overlap && (((double)aDir.X * bDir.X) + ((double)aDir.Y * bDir.Y)) < 0)
            {
                Push(ip);
                Push(iq);
                return Finish(buf, r, resultSize, out areaOut);
            }

            if (cross == 0 && aHB < 0 && bHA < 0)
            {
                return Finish(buf, r, resultSize, out areaOut);
            }
            else if (cross == 0 && aHB == 0 && bHA == 0)
            {
                if (inflag == InFlag.Pin)
                {
                    b = Advance(b, ref ba, m, inflag == InFlag.Qin, q[b]);
                }
                else
                {
                    a = Advance(a, ref aa, n, inflag == InFlag.Pin, p[a]);
                }
            }
            else if (cross >= 0)
            {
                if (bHA > 0)
                {
                    a = Advance(a, ref aa, n, inflag == InFlag.Pin, p[a]);
                }
                else
                {
                    b = Advance(b, ref ba, m, inflag == InFlag.Qin, q[b]);
                }
            }
            else
            {
                if (aHB > 0)
                {
                    b = Advance(b, ref ba, m, inflag == InFlag.Qin, q[b]);
                }
                else
                {
                    a = Advance(a, ref aa, n, inflag == InFlag.Pin, p[a]);
                }
            }
        }
        while ((aa < n || ba < m) && aa < 2 * n && ba < 2 * m && r <= resultSize);

        return Finish(buf, r, resultSize, out areaOut);
    }

    private static int Finish(Point2f[] result0, int nr, int resultSize, out float areaOut)
    {
        if (nr > resultSize)
        {
            areaOut = -1f;
            return -1;
        }

        double area = 0;
        var prev = result0[nr - 1];
        for (int i = 1; i < nr; i++)
        {
            result0[i - 1] = result0[i];
            area += ((double)prev.X * result0[i].Y) - ((double)prev.Y * result0[i].X);
            prev = result0[i];
        }

        areaOut = (float)(area * 0.5);
        if (nr > 1 && result0[nr - 2] == result0[0])
        {
            nr--;
        }

        return nr - 1;
    }

    private static bool IsOnPositiveSide(Point2f lineVec, Point2f linePt, Point2f pt) =>
        lineVec.Y * (linePt.X - pt.X) >= lineVec.X * (linePt.Y - pt.Y);

    private static RectanglesIntersectType RectangleIntersectionCore(RotatedRect rect1, RotatedRect rect2, List<Point2f> intersection)
    {
        Span<Point2f> pts1 = stackalloc Point2f[4];
        Span<Point2f> pts2 = stackalloc Point2f[4];
        Span<Point2f> vec1 = stackalloc Point2f[4];
        Span<Point2f> vec2 = stackalloc Point2f[4];
        rect1.Points(pts1);
        rect2.Points(pts2);

        float samePointEps = 1e-6f * Math.Max(rect1.Size.Width * rect1.Size.Height, rect2.Size.Width * rect2.Size.Height);
        var ret = RectanglesIntersectType.Full;

        bool same = true;
        for (int i = 0; i < 4; i++)
        {
            if (MathF.Abs(pts1[i].X - pts2[i].X) > samePointEps || MathF.Abs(pts1[i].Y - pts2[i].Y) > samePointEps)
            {
                same = false;
                break;
            }
        }

        if (same)
        {
            for (int i = 0; i < 4; i++)
            {
                intersection.Add(pts1[i]);
            }

            return RectanglesIntersectType.Full;
        }

        for (int i = 0; i < 4; i++)
        {
            vec1[i] = Sub(pts1[(i + 1) % 4], pts1[i]);
            vec2[i] = Sub(pts2[(i + 1) % 4], pts2[i]);
        }

        for (int i = 0; i < 4; i++)
        {
            samePointEps = Math.Min(samePointEps, MathF.Sqrt((vec1[i].X * vec1[i].X) + (vec1[i].Y * vec1[i].Y)));
            samePointEps = Math.Min(samePointEps, MathF.Sqrt((vec2[i].X * vec2[i].X) + (vec2[i].Y * vec2[i].Y)));
        }

        samePointEps = Math.Max(1e-16f, samePointEps);

        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                float x21 = pts2[j].X - pts1[i].X;
                float y21 = pts2[j].Y - pts1[i].Y;
                float vx1 = vec1[i].X, vy1 = vec1[i].Y;
                float vx2 = vec2[j].X, vy2 = vec2[j].Y;
                float det = (vx2 * vy1) - (vx1 * vy2);
                if (MathF.Abs(det) < 1e-12)
                {
                    continue;
                }

                float detInvScaled = 1.0f / det;
                float t1 = ((vx2 * y21) - (vy2 * x21)) * detInvScaled;
                float t2 = ((vx1 * y21) - (vy1 * x21)) * detInvScaled;
                if (float.IsInfinity(t1) || float.IsInfinity(t2) || float.IsNaN(t1) || float.IsNaN(t2))
                {
                    continue;
                }

                if (t1 >= 0.0f && t1 <= 1.0f && t2 >= 0.0f && t2 <= 1.0f)
                {
                    intersection.Add(new Point2f(pts1[i].X + (vec1[i].X * t1), pts1[i].Y + (vec1[i].Y * t1)));
                }
            }
        }

        if (intersection.Count > 0)
        {
            ret = RectanglesIntersectType.Partial;
        }

        AddInsideVertices(pts1, pts2, vec2, intersection);
        AddInsideVertices(pts2, pts1, vec1, intersection);

        int count = intersection.Count;
        if (count == 0)
        {
            return RectanglesIntersectType.None;
        }

        // Duplicate removal; distPt holds squared distances by original row and column.
        int stride = count;
        var distPt = new float[count * count];
        var remap = new int[count];
        for (int i = 0; i < count; i++)
        {
            var pt0 = intersection[i];
            remap[i] = i;
            for (int j = i + 1; j < count;)
            {
                var d = Sub(intersection[j], pt0);
                float d2 = (d.X * d.X) + (d.Y * d.Y);
                if (d2 <= samePointEps)
                {
                    if (j < count - 1)
                    {
                        intersection[j] = intersection[count - 1];
                    }

                    count--;
                    continue;
                }

                distPt[(i * stride) + j] = d2;
                ++j;
            }
        }

        while (count > 8)
        {
            int minI = 0, minJ = 1;
            float minD = distPt[1];
            for (int i = 0; i < count - 1; ++i)
            {
                int row = stride * remap[i];
                for (int j = i + 1; j < count; ++j)
                {
                    float d = distPt[row + remap[j]];
                    if (d < minD)
                    {
                        minD = d;
                        minI = i;
                        minJ = j;
                    }
                }
            }

            if (minJ < count - 1)
            {
                intersection[minJ] = intersection[count - 1];
                remap[minJ] = remap[count - 1];
            }

            count--;
        }

        for (int i = 0; i < count - 1; ++i)
        {
            var diffI = Sub(intersection[i + 1], intersection[i]);
            for (int j = i + 2; j < count; ++j)
            {
                var diffJ = Sub(intersection[j], intersection[i]);
                if (Cross(diffI, diffJ) < 0)
                {
                    (intersection[i + 1], intersection[j]) = (intersection[j], intersection[i + 1]);
                    diffI = diffJ;
                }
            }
        }

        intersection.RemoveRange(count, intersection.Count - count);
        return ret;
    }

    private static void AddInsideVertices(ReadOnlySpan<Point2f> pts, ReadOnlySpan<Point2f> other, ReadOnlySpan<Point2f> otherVec, List<Point2f> intersection)
    {
        for (int i = 0; i < 4; i++)
        {
            int pos = 0, neg = 0;
            for (int j = 0; j < 4; j++)
            {
                if (IsOnPositiveSide(otherVec[j], other[j], pts[i]))
                {
                    pos++;
                }
                else
                {
                    neg++;
                }
            }

            if (pos == 4 || neg == 4)
            {
                intersection.Add(pts[i]);
            }
        }
    }
}
