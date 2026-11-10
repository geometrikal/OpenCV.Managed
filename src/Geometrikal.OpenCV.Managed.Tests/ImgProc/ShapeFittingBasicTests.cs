namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Analytic and pathological-input tests for <see cref="ShapeFitting"/>, <see cref="RotatedRect"/>
/// and <see cref="CvRng"/>. <see cref="ShapeFittingGoldenTests"/> covers the cv2 5.0.0 outputs.
/// </summary>
public sealed class ShapeFittingBasicTests
{
    private static readonly Point[] Square = [new(0, 0), new(0, 10), new(10, 10), new(10, 0)];

    private static readonly Point[] Notch = [new(0, 0), new(0, 10), new(5, 4), new(10, 10), new(10, 0)];

    [Fact]
    public void ConvexHull_EmptyInput_ReturnsEmpty()
    {
        Assert.Empty(ShapeFitting.ConvexHull(ReadOnlySpan<Point>.Empty));
        Assert.Empty(ShapeFitting.ConvexHull(ReadOnlySpan<Point2f>.Empty));
        Assert.Empty(ShapeFitting.ConvexHullIndices(ReadOnlySpan<Point>.Empty));
        Assert.Empty(ShapeFitting.ConvexHullIndices(ReadOnlySpan<Point2f>.Empty));
    }

    [Fact]
    public void ConvexHull_SquareWithInteriorPoints_ReturnsCorners()
    {
        Point[] pts = [new(5, 5), .. Square, new(3, 7), new(10, 5)];
        Point[] hull = ShapeFitting.ConvexHull(pts);
        Assert.Equal(4, hull.Length);
        Assert.Equal(Square.OrderBy(p => p.X).ThenBy(p => p.Y), hull.OrderBy(p => p.X).ThenBy(p => p.Y));
    }

    [Fact]
    public void ConvexHull_Clockwise_ReversesCyclicOrder()
    {
        var rng = new Random(3);
        Point[] pts = Enumerable.Range(0, 50).Select(_ => new Point(rng.Next(100), rng.Next(100))).ToArray();
        Point[] ccw = ShapeFitting.ConvexHull(pts, clockwise: false);
        Point[] cw = ShapeFitting.ConvexHull(pts, clockwise: true);
        Assert.Equal(ccw.Length, cw.Length);
        int shift = Array.IndexOf(cw, ccw[0]);
        Assert.True(shift >= 0);
        for (int i = 0; i < ccw.Length; i++)
        {
            Assert.Equal(ccw[i], cw[((shift - i) % cw.Length + cw.Length) % cw.Length]);
        }
    }

    [Fact]
    public void ConvexHullIndices_AllPointsEqual_ReturnsFirstIndex()
    {
        Point[] pts = [new(2, 2), new(2, 2), new(2, 2)];
        Assert.Equal([0], ShapeFitting.ConvexHullIndices(pts));
    }

    [Fact]
    public void ConvexHull_DoesNotModifySourceAndIsDeterministic()
    {
        Point2f[] pts = Enumerable.Range(0, 40).Select(i => new Point2f(MathF.Sin(i) * 10, MathF.Cos(i * 1.7f) * 5)).ToArray();
        Point2f[] copy = (Point2f[])pts.Clone();
        Point2f[] first = ShapeFitting.ConvexHull(pts);
        Point2f[] second = ShapeFitting.ConvexHull(pts);
        Assert.Equal(copy, pts);
        Assert.Equal(first, second);
    }

    [Fact]
    public void IsContourConvex_KnownShapes_ReturnsExpected()
    {
        Assert.False(ShapeFitting.IsContourConvex(ReadOnlySpan<Point>.Empty));
        Assert.True(ShapeFitting.IsContourConvex(Square));
        Assert.False(ShapeFitting.IsContourConvex(Notch));
        Assert.True(ShapeFitting.IsContourConvex(Square.Select(p => new Point2f(p.X, p.Y)).ToArray()));
    }

    [Fact]
    public void ConvexityDefects_Notch_ReturnsNotchVertexWithFixedPointDepth()
    {
        int[] hull = ShapeFitting.ConvexHullIndices(Notch);
        ConvexityDefect[] defects = ShapeFitting.ConvexityDefects(Notch, hull);
        ConvexityDefect d = Assert.Single(defects);
        Assert.Equal(2, d.FarthestIndex);
        Assert.Equal(6 * 256, d.FixedPointDepth);
        Assert.Equal(6.0f, d.Depth);
    }

    [Fact]
    public void ConvexityDefects_ShortInputs_ReturnEmpty()
    {
        Assert.Empty(ShapeFitting.ConvexityDefects(Square.AsSpan(0, 3), [0, 1, 2]));
        Assert.Empty(ShapeFitting.ConvexityDefects(Notch, [0, 1]));
    }

    [Fact]
    public void ConvexityDefects_InvalidHull_Throws()
    {
        Assert.Throws<ArgumentException>(() => ShapeFitting.ConvexityDefects(Notch, ReadOnlySpan<int>.Empty));
        Assert.Throws<ArgumentOutOfRangeException>(() => ShapeFitting.ConvexityDefects(Notch, [0, 1, 7]));
        Assert.Throws<ArgumentException>(() => ShapeFitting.ConvexityDefects(Notch, [0, 3, 1, 4]));
    }

    [Fact]
    public void MinAreaRect_AxisAlignedRectangle_UsesOpenCv5AngleConvention()
    {
        Point[] pts = [new(2, 3), new(2, 7), new(12, 7), new(12, 3)];
        RotatedRect r = ShapeFitting.MinAreaRect(pts);
        Assert.Equal(new Point2f(7, 5), r.Center);
        Assert.Equal(-90f, r.Angle);
        Assert.Equal(new Size2f(4, 10), r.Size);
    }

    [Fact]
    public void MinAreaRect_RandomSets_AngleInHalfOpenRange()
    {
        var rng = new Random(11);
        for (int k = 0; k < 50; k++)
        {
            Point2f[] pts = Enumerable.Range(0, 3 + k).Select(_ => new Point2f((float)rng.NextDouble() * 50, (float)rng.NextDouble() * 50)).ToArray();
            RotatedRect r = ShapeFitting.MinAreaRect(pts);
            Assert.InRange(r.Angle, -90f, -1e-30f);
        }
    }

    [Fact]
    public void MinAreaRect_EmptyInput_ReturnsZeroRectangle()
        => Assert.Equal(new RotatedRect(default, default, -90f), ShapeFitting.MinAreaRect(ReadOnlySpan<Point>.Empty));

    [Fact]
    public void BoxPoints_AngleZero_ReturnsCornersInOpenCvOrder()
    {
        Point2f[] pts = ShapeFitting.BoxPoints(new RotatedRect(new Point2f(10, 20), new Size2f(4, 2), 0));
        Assert.Equal([new(8, 21), new(8, 19), new(12, 19), new(12, 21)], pts);
        Assert.Throws<ArgumentException>(() => new RotatedRect().Points(new Point2f[3]));
    }

    [Fact]
    public void MinEnclosingCircle_PathologicalInputs_ReturnOpenCvValues()
    {
        Assert.Equal(new EnclosingCircle(default, 0f), ShapeFitting.MinEnclosingCircle(ReadOnlySpan<Point>.Empty));
        Assert.Equal(new EnclosingCircle(new Point2f(3, 4), 1e-4f), ShapeFitting.MinEnclosingCircle([new Point(3, 4)]));
    }

    [Fact]
    public void MinEnclosingCircle_RandomSets_EnclosesAllPoints()
    {
        var rng = new Random(5);
        for (int k = 0; k < 20; k++)
        {
            Point2f[] pts = Enumerable.Range(0, 2 + (k * 7)).Select(_ => new Point2f((float)rng.NextDouble() * 80, (float)rng.NextDouble() * 30)).ToArray();
            EnclosingCircle c = ShapeFitting.MinEnclosingCircle(pts);
            foreach (Point2f p in pts)
            {
                double d = Math.Sqrt(Math.Pow(p.X - c.Center.X, 2) + Math.Pow(p.Y - c.Center.Y, 2));
                Assert.True(d <= c.Radius + 1e-3, $"set {k}: distance {d} > radius {c.Radius}");
            }
        }
    }

    [Fact]
    public void FitEllipse_TooFewPoints_Throws()
    {
        Point2f[] four = [new(0, 0), new(1, 0), new(1, 1), new(0, 1)];
        Assert.Throws<ArgumentException>(() => ShapeFitting.FitEllipse(four));
        Assert.Throws<ArgumentException>(() => ShapeFitting.FitEllipseAms(four));
        Assert.Throws<ArgumentException>(() => ShapeFitting.FitEllipseDirect(four));
    }

    [Fact]
    public void FitEllipse_PointsOnEllipse_RecoversAxes()
    {
        Point2f[] pts = Enumerable.Range(0, 24)
            .Select(i => new Point2f(30 + (12 * MathF.Cos(i * MathF.PI / 12)), 20 + (5 * MathF.Sin(i * MathF.PI / 12))))
            .ToArray();
        foreach (RotatedRect r in new[] { ShapeFitting.FitEllipse(pts), ShapeFitting.FitEllipseAms(pts), ShapeFitting.FitEllipseDirect(pts) })
        {
            Assert.Equal(30, r.Center.X, 2);
            Assert.Equal(20, r.Center.Y, 2);
            Assert.Equal(10, r.Size.Width, 2);
            Assert.Equal(24, r.Size.Height, 2);
            Assert.Equal(90, r.Angle, 1);
        }
    }

    [Fact]
    public void FitLine_InvalidInputs_Throw()
    {
        Assert.Throws<ArgumentException>(() => ShapeFitting.FitLine(ReadOnlySpan<Point2f>.Empty, DistanceTypes.L2, 0, 0.01, 0.01));
        Assert.Throws<ArgumentException>(() => ShapeFitting.FitLine(Square, DistanceTypes.C, 0, 0.01, 0.01));
    }

    [Theory]
    [InlineData(DistanceTypes.L2)]
    [InlineData(DistanceTypes.L1)]
    [InlineData(DistanceTypes.L12)]
    [InlineData(DistanceTypes.Fair)]
    [InlineData(DistanceTypes.Welsch)]
    [InlineData(DistanceTypes.Huber)]
    public void FitLine_ExactLineWithOutlier_RobustTypesIgnoreOutlier(DistanceTypes dist)
    {
        Point[] pts = [.. Enumerable.Range(0, 20).Select(i => new Point(i, 3)), new Point(10, 40)];
        FittedLine line = ShapeFitting.FitLine(pts, dist, 0, 0.01, 0.01);
        Assert.Equal(1f, line.Vx * line.Vx + line.Vy * line.Vy, 5);
        if (dist != DistanceTypes.L2)
        {
            Assert.Equal(1f, Math.Abs(line.Vx), 3);
            Assert.Equal(0f, line.Vy, 3);
            Assert.Equal(3f, line.Y0, 1);
        }

        Assert.Equal(line, ShapeFitting.FitLine(pts, dist, 0, 0.01, 0.01));
    }

    [Fact]
    public void PointPolygonTest_Square_ReturnsSignAndDistance()
    {
        Assert.Equal(1, ShapeFitting.PointPolygonTest(Square, new Point2f(5, 5), false));
        Assert.Equal(-1, ShapeFitting.PointPolygonTest(Square, new Point2f(15, 5), false));
        Assert.Equal(0, ShapeFitting.PointPolygonTest(Square, new Point2f(10, 5), false));
        Assert.Equal(0, ShapeFitting.PointPolygonTest(Square, new Point2f(0, 0), false));
        Assert.Equal(1, ShapeFitting.PointPolygonTest(Square, new Point2f(5.5f, 2.25f), false));
        Assert.Equal(5, ShapeFitting.PointPolygonTest(Square, new Point2f(5, 5), true));
        Assert.Equal(-3, ShapeFitting.PointPolygonTest(Square, new Point2f(13, 5), true));
        Assert.Equal(2.25, ShapeFitting.PointPolygonTest(Square.Select(p => new Point2f(p.X, p.Y)).ToArray(), new Point2f(5.5f, 2.25f), true));
    }

    [Fact]
    public void PointPolygonTest_EmptyContour_ReturnsOpenCvValues()
    {
        Assert.Equal(-1, ShapeFitting.PointPolygonTest(ReadOnlySpan<Point>.Empty, default, false));
        Assert.Equal(-double.MaxValue, ShapeFitting.PointPolygonTest(ReadOnlySpan<Point>.Empty, default, true));
    }

    [Fact]
    public void CvRng_DefaultState_MatchesMultiplyWithCarry()
    {
        // state = 0xffffffff * 4164903690: the low word is 2^32 - 4164903690 and the carry is 4164903689.
        var rng = new CvRng(0);
        Assert.Equal(130063606u, rng.Next());
        Assert.Equal((4164903689UL << 32) | 130063606UL, rng.State);
        Assert.Equal(4, new CvRng(7).Uniform(4, 4));
        float f = new CvRng(99).Uniform(-2f, 2f);
        Assert.InRange(f, -2f, 2f);
    }

    [Fact]
    public void CvRng_SetTheRngSeed_ResetsThreadGenerator()
    {
        CvRng.SetTheRngSeed(-1);
        Assert.Equal(ulong.MaxValue, CvRng.TheRng.State);
        CvRng.SetTheRngSeed(0);
        Assert.Equal(0xffffffffUL, CvRng.TheRng.State);
    }
}
