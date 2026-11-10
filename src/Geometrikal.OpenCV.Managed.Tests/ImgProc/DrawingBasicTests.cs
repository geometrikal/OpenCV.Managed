namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Collections.Generic;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Analytic behavioural tests for <see cref="Drawing"/> that do not depend on
/// the golden JSON: ROI / stride isolation (drawing writes through to the
/// parent and never outside the view), determinism, silent clipping of
/// far-out-of-range coordinates, the <see cref="Drawing.Filled"/> sentinel, and
/// the full argument-validation contract (spec §3).
///
/// Every test references only the public <see cref="Drawing"/> API.
/// </summary>
public sealed class DrawingBasicTests
{
    private static readonly Scalar White = new(255);

    // -- Filled sentinel ----------------------------------------------------

    [Fact]
    public void Filled_ConstantIsMinusOne()
    {
        Assert.Equal(-1, Drawing.Filled);
    }

    // -- ROI / stride isolation (MANDATORY) ---------------------------------

    [Fact]
    public void Drawing_IntoRoi_MatchesPackedMat_AndLeavesParentUntouched()
    {
        // 30x28 parent, ROI window (3,4) size 20x16.
        const int roiX = 3;
        const int roiY = 4;
        const int roiW = 20;
        const int roiH = 16;

        var parent = new Mat<byte>(30, 28);
        Mat<byte> roi = parent.Roi(roiX, roiY, roiW, roiH);
        Assert.False(roi.IsContinuous, "ROI into a wider parent must be non-continuous.");

        var packed = new Mat<byte>(roiW, roiH);
        Assert.True(packed.IsContinuous);

        DrawBattery(roi);
        DrawBattery(packed);

        // The ROI window must equal the standalone packed drawing exactly.
        Assert.Equal(packed.ToArray(), roi.ToArray());

        // Every parent pixel OUTSIDE the ROI window must remain background.
        for (int y = 0; y < parent.Height; y++)
        {
            for (int x = 0; x < parent.Width; x++)
            {
                bool inside = x >= roiX && x < roiX + roiW && y >= roiY && y < roiY + roiH;
                if (!inside)
                {
                    Assert.Equal(0, parent.At(y, x));
                }
            }
        }
    }

    private static void DrawBattery(Mat<byte> m)
    {
        // A mix that routes through EVERY rasterizer: HLine (fills), PutPixel
        // (thin outlines), Line2 (sub-pixel), the EllipseEx path, the
        // CollectPolyEdges/FillEdgeCollection scanline path, ThickLine caps,
        // and the DrawContours plumbing — so a stride bug in any of them makes
        // the ROI-vs-packed comparison fail.
        Drawing.Rectangle(m, new Point(1, 1), new Point(18, 14), White, thickness: 1);
        Drawing.Line(m, new Point(0, 0), new Point(19, 15), White, thickness: 1);
        Drawing.Line(m, new Point(0, 15), new Point(19, 0), White, thickness: 2);
        Drawing.Circle(m, new Point(10, 8), 5, White, thickness: Drawing.Filled);
        Drawing.FillConvexPoly<byte>(
            m, new[] { new Point(2, 12), new Point(9, 4), new Point(16, 13) }, White);
        Drawing.Ellipse(m, new Point(9, 7), new Size(8, 5), 30, 0, 300, White, thickness: 1);
        Drawing.FillPoly(
            m,
            new[]
            {
                new[] { new Point(3, 2), new Point(17, 3), new Point(12, 13), new Point(5, 11) },
                new[] { new Point(7, 5), new Point(12, 6), new Point(10, 10) },
            },
            White,
            offset: new Point(1, 0));
        Drawing.Polylines(
            m,
            new[] { new[] { new Point(0, 13), new Point(6, 2), new Point(15, 12), new Point(19, 1) } },
            isClosed: false, White, thickness: 3);
        Drawing.DrawContours(
            m,
            new[] { new[] { new Point(2, 2), new Point(14, 3), new Point(13, 12), new Point(3, 13) } },
            contourIndex: -1, White, thickness: 1, offset: new Point(1, 1));
        Drawing.DrawContours(
            m,
            new[] { new[] { new Point(4, 4), new Point(11, 4), new Point(11, 9), new Point(4, 9) } },
            contourIndex: 0, White, thickness: Drawing.Filled);
    }

    // -- Determinism --------------------------------------------------------

    [Fact]
    public void Drawing_RepeatedRuns_AreDeterministic()
    {
        var a = new Mat<byte>(24, 24);
        var b = new Mat<byte>(24, 24);

        DrawBattery(a);
        DrawBattery(b);

        Assert.Equal(a.ToArray(), b.ToArray());
    }

    // -- Silent clipping of far-out-of-range coordinates --------------------

    [Fact]
    public void Drawing_FullyOutsideCoordinates_DoNotThrow_AndLeaveImageUnchanged()
    {
        var m = new Mat<byte>(20, 20);
        byte[] before = m.ToArray();

        var far1 = new Point(10_000, 10_000);
        var far2 = new Point(20_000, 25_000);
        var negFar = new Point(-10_000, -12_000);

        // farMid keeps the triangle's INTERIOR away from the image too: a
        // polygon with far-away vertices can still cover the image (verified
        // with OpenCV: fillPoly over {far1, far2, negFar} paints everything).
        var farMid = new Point(12_000, 15_000);

        Drawing.Line(m, negFar, far1, White, thickness: 3);
        Drawing.Rectangle(m, far1, far2, White, thickness: Drawing.Filled);
        Drawing.Circle(m, far1, 7, White, thickness: Drawing.Filled);
        Drawing.Ellipse(m, far1, new Size(9, 5), 30, 0, 360, White, thickness: 2);
        Drawing.FillConvexPoly<byte>(
            m, new[] { far1, far2, farMid }, White);
        Drawing.FillPoly(
            m, new[] { new[] { far1, far2, farMid } }, White);
        Drawing.Polylines(
            m, new[] { new[] { far1, far2, farMid } }, isClosed: true, White, thickness: 2);
        Drawing.DrawContours(
            m, new[] { new[] { new Point(2, 2), new Point(8, 2), new Point(5, 8) } },
            contourIndex: -1, White, thickness: 1, offset: far1);
        Drawing.DrawContours(
            m, new[] { new[] { new Point(2, 2), new Point(8, 2), new Point(5, 8) } },
            contourIndex: -1, White, thickness: Drawing.Filled, offset: negFar);

        Assert.Equal(before, m.ToArray());
    }

    // -- Validation applies to every wrapper, not only Line ------------------

    public static IEnumerable<object[]> AllWrappers()
    {
        // Each entry invokes one public wrapper with the supplied image,
        // lineType, and shift so the shared validation contract can be
        // asserted across the whole surface.
        yield return new object[] { "Line", (Action<Mat<byte>, LineType, int>)((m, lt, s) =>
            Drawing.Line(m, new Point(0, 0), new Point(5, 5), White, 1, lt, s)) };
        yield return new object[] { "Rectangle", (Action<Mat<byte>, LineType, int>)((m, lt, s) =>
            Drawing.Rectangle(m, new Point(0, 0), new Point(5, 5), White, 1, lt, s)) };
        yield return new object[] { "RectangleRect", (Action<Mat<byte>, LineType, int>)((m, lt, s) =>
            Drawing.Rectangle(m, new Rect(0, 0, 5, 5), White, 1, lt, s)) };
        yield return new object[] { "Circle", (Action<Mat<byte>, LineType, int>)((m, lt, s) =>
            Drawing.Circle(m, new Point(5, 5), 3, White, 1, lt, s)) };
        yield return new object[] { "Ellipse", (Action<Mat<byte>, LineType, int>)((m, lt, s) =>
            Drawing.Ellipse(m, new Point(5, 5), new Size(3, 2), 0, 0, 360, White, 1, lt, s)) };
        yield return new object[] { "FillConvexPoly", (Action<Mat<byte>, LineType, int>)((m, lt, s) =>
            Drawing.FillConvexPoly<byte>(m, new[] { new Point(1, 1), new Point(6, 1), new Point(3, 6) }, White, lt, s)) };
        yield return new object[] { "FillPoly", (Action<Mat<byte>, LineType, int>)((m, lt, s) =>
            Drawing.FillPoly(m, new[] { new[] { new Point(1, 1), new Point(6, 1), new Point(3, 6) } }, White, lt, s)) };
        yield return new object[] { "Polylines", (Action<Mat<byte>, LineType, int>)((m, lt, s) =>
            Drawing.Polylines(m, new[] { new[] { new Point(1, 1), new Point(6, 1) } }, false, White, 1, lt, s)) };
    }

    [Theory]
    [MemberData(nameof(AllWrappers))]
    public void EveryWrapper_AcceptsLineAA(string name, Action<Mat<byte>, LineType, int> invoke)
    {
        _ = name;
        var m = new Mat<byte>(10, 10);
        Assert.Null(Record.Exception(() => invoke(m, LineType.AntiAliased, 0)));
    }

    [Theory]
    [MemberData(nameof(AllWrappers))]
    public void EveryWrapper_RejectsUnknownLineType(string name, Action<Mat<byte>, LineType, int> invoke)
    {
        _ = name;
        var m = new Mat<byte>(10, 10);
        Assert.Throws<ArgumentOutOfRangeException>(() => invoke(m, (LineType)5, 0));
    }

    [Theory]
    [MemberData(nameof(AllWrappers))]
    public void EveryWrapper_RejectsShiftOutOfRange(string name, Action<Mat<byte>, LineType, int> invoke)
    {
        _ = name;
        var m = new Mat<byte>(10, 10);
        Assert.Throws<ArgumentOutOfRangeException>(() => invoke(m, LineType.EightConnected, 17));
        Assert.Throws<ArgumentOutOfRangeException>(() => invoke(m, LineType.EightConnected, -1));
    }

    [Theory]
    [MemberData(nameof(AllWrappers))]
    public void EveryWrapper_RejectsMoreThanFourChannels(string name, Action<Mat<byte>, LineType, int> invoke)
    {
        _ = name;
        var m = new Mat<byte>(10, 10, channels: 5);
        Assert.Throws<NotSupportedException>(() => invoke(m, LineType.EightConnected, 0));
    }

    [Fact]
    public void DrawContours_AcceptsLineAA_AndRejectsBadHierarchyLength()
    {
        var m = new Mat<byte>(10, 10);
        Point[][] contours = [[new Point(1, 1), new Point(5, 1), new Point(3, 5)]];

        Assert.Null(Record.Exception(() =>
            Drawing.DrawContours(m, contours, -1, White, 1, LineType.AntiAliased)));
        Assert.Throws<ArgumentException>(() =>
            Drawing.DrawContours(m, contours, -1, White, 1,
                hierarchy: new[] { new ContourHierarchy(-1, -1, -1, -1), new ContourHierarchy(-1, -1, -1, -1) }));
    }

    // -- Polylines' thickness rule differs from the shape wrappers -----------

    [Fact]
    public void Polylines_ThicknessZeroAllowed_FilledRejected()
    {
        var m = new Mat<byte>(10, 10);
        Point[][] pts = [[new Point(1, 1), new Point(8, 8)]];

        // thickness 0 is legal for polylines in OpenCV (draws a thin line)...
        Drawing.Polylines(m, pts, isClosed: false, White, thickness: 0);

        // ...but FILLED (-1) is not: cv::polylines asserts 0 <= thickness.
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Drawing.Polylines(m, pts, isClosed: false, White, thickness: Drawing.Filled));
    }

    // -- Colour saturation semantics (OpenCV scalarToRawData / cvRound) ------

    [Fact]
    public void Color_SaturationAndRounding_MatchOpenCv()
    {
        // In-int-range values saturate to the type range...
        Assert.Equal(255, PixelAfterFill(new Scalar(300)));
        Assert.Equal(0, PixelAfterFill(new Scalar(-5)));

        // ...fractional values use round-half-to-even like cvRound...
        Assert.Equal(2, PixelAfterFill(new Scalar(2.5)));
        Assert.Equal(4, PixelAfterFill(new Scalar(3.5)));

        // ...and values outside the int32 range (or NaN) hit cvRound's x86-64
        // "integer indefinite" sentinel (int.MinValue), which saturates to the
        // type MINIMUM — OpenCV paints such colors black, not white.
        Assert.Equal(0, PixelAfterFill(new Scalar(3e9)));
        Assert.Equal(0, PixelAfterFill(new Scalar(double.NaN)));

        // ushort keeps the same contract.
        var u16 = new Mat<ushort>(3, 3);
        Drawing.Rectangle(u16, new Point(0, 0), new Point(2, 2), new Scalar(70_000), Drawing.Filled);
        Assert.Equal(65_535, u16.At(1, 1));
        Drawing.Rectangle(u16, new Point(0, 0), new Point(2, 2), new Scalar(3e9), Drawing.Filled);
        Assert.Equal(0, u16.At(1, 1));

        // short: the sentinel saturates to the minimum, not the maximum.
        var s16 = new Mat<short>(3, 3);
        Drawing.Rectangle(s16, new Point(0, 0), new Point(2, 2), new Scalar(3e9), Drawing.Filled);
        Assert.Equal(short.MinValue, s16.At(1, 1));

        // float passes through unrounded.
        var f32 = new Mat<float>(3, 3);
        Drawing.Rectangle(f32, new Point(0, 0), new Point(2, 2), new Scalar(1.25), Drawing.Filled);
        Assert.Equal(1.25f, f32.At(1, 1));
    }

    private static byte PixelAfterFill(Scalar color)
    {
        var m = new Mat<byte>(3, 3);
        Drawing.Rectangle(m, new Point(0, 0), new Point(2, 2), color, Drawing.Filled);
        return m.At(1, 1);
    }

    [Fact]
    public void FillPoly_SelfIntersectingMultiChannel_MatchesSingleChannel()
    {
        // Self-intersecting polygons produce empty scanline runs.
        var random = new Random(1234);
        var polygons = new List<Point[]>();
        for (int p = 0; p < 32; p++)
        {
            var polygon = new Point[8];
            for (int v = 0; v < polygon.Length; v++)
            {
                polygon[v] = new Point(random.Next(64), random.Next(64));
            }

            polygons.Add(polygon);
        }

        var gray = new Mat<byte>(64, 64);
        var color = new Mat<byte>(64, 64, 3);
        Drawing.FillPoly(gray, polygons, White);
        Drawing.FillPoly(color, polygons, new Scalar(255, 255, 255));

        for (int y = 0; y < 64; y++)
        {
            for (int x = 0; x < 64; x++)
            {
                for (int c = 0; c < 3; c++)
                {
                    Assert.Equal(gray.At(y, x), color.At(y, x, c));
                }
            }
        }
    }

    // -- Validation: null arguments -----------------------------------------

    [Fact]
    public void Line_NullImage_Throws()
    {
        Assert.Throws<ArgumentNullException>(() =>
            Drawing.Line<byte>(null!, new Point(0, 0), new Point(1, 1), White));
    }

    [Fact]
    public void FillPoly_NullList_Throws()
    {
        var m = new Mat<byte>(8, 8);
        Assert.Throws<ArgumentNullException>(() =>
            Drawing.FillPoly(m, null!, White));
    }

    [Fact]
    public void FillPoly_NullElement_ThrowsArgumentException()
    {
        var m = new Mat<byte>(8, 8);
        var polys = new Point[][] { null! };
        Assert.Throws<ArgumentException>(() => Drawing.FillPoly(m, polys, White));
    }

    [Fact]
    public void Polylines_NullElement_ThrowsArgumentException()
    {
        var m = new Mat<byte>(8, 8);
        var lines = new Point[][] { null! };
        Assert.Throws<ArgumentException>(() =>
            Drawing.Polylines(m, lines, isClosed: false, White));
    }

    // -- Validation: line type ----------------------------------------------

    [Fact]
    public void Line_LineAaValue_IsAccepted()
    {
        var m = new Mat<byte>(8, 8);
        Drawing.Line(m, new Point(0, 0), new Point(7, 7), White, 1, LineType.AntiAliased, 0);
        Assert.NotEqual(0, m.GetRowSpan(3)[3]);
    }

    [Fact]
    public void Line_UnknownLineType_ThrowsArgumentOutOfRange()
    {
        var m = new Mat<byte>(8, 8);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Drawing.Line(m, new Point(0, 0), new Point(7, 7), White, 1, (LineType)5, 0));
    }

    // -- Validation: shift range --------------------------------------------

    [Theory]
    [InlineData(-1)]
    [InlineData(17)]
    public void Line_ShiftOutOfRange_ThrowsArgumentOutOfRange(int shift)
    {
        var m = new Mat<byte>(8, 8);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Drawing.Line(m, new Point(0, 0), new Point(7, 7), White, 1, LineType.EightConnected, shift));
    }

    // -- Validation: thickness bounds ---------------------------------------

    [Theory]
    [InlineData(0)]
    [InlineData(32768)]
    public void Line_ThicknessOutOfRange_ThrowsArgumentOutOfRange(int thickness)
    {
        var m = new Mat<byte>(8, 8);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Drawing.Line(m, new Point(0, 0), new Point(7, 7), White, thickness));
    }

    [Fact]
    public void Rectangle_ThicknessTooLarge_ThrowsArgumentOutOfRange()
    {
        var m = new Mat<byte>(8, 8);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Drawing.Rectangle(m, new Point(1, 1), new Point(6, 6), White, thickness: 32768));
    }

    // -- Validation: channels > 4 -------------------------------------------

    [Fact]
    public void Line_MoreThanFourChannels_ThrowsNotSupported()
    {
        var m = new Mat<byte>(8, 8, channels: 5);
        Assert.Throws<NotSupportedException>(() =>
            Drawing.Line(m, new Point(0, 0), new Point(7, 7), White));
    }

    // -- Validation: circle / ellipse geometry ------------------------------

    [Fact]
    public void Circle_NegativeRadius_ThrowsArgumentOutOfRange()
    {
        var m = new Mat<byte>(8, 8);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Drawing.Circle(m, new Point(4, 4), -1, White));
    }

    [Fact]
    public void Ellipse_NegativeAxis_ThrowsArgumentOutOfRange()
    {
        var m = new Mat<byte>(8, 8);
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Drawing.Ellipse(m, new Point(4, 4), new Size(-1, 3), 0, 0, 360, White));
    }

    // -- Validation: drawContours -------------------------------------------

    [Fact]
    public void DrawContours_ContourIndexOutOfRange_ThrowsArgumentOutOfRange()
    {
        var m = new Mat<byte>(8, 8);
        var contours = new[] { new[] { new Point(1, 1), new Point(2, 2), new Point(1, 3) } };
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            Drawing.DrawContours(m, contours, contourIndex: 5, White));
    }

    [Fact]
    public void DrawContours_HierarchyCountMismatch_ThrowsArgumentException()
    {
        var m = new Mat<byte>(8, 8);
        var contours = new[] { new[] { new Point(1, 1), new Point(2, 2), new Point(1, 3) } };
        var hierarchy = new List<ContourHierarchy>
        {
            new(-1, -1, -1, -1),
            new(-1, -1, -1, -1),
        };
        Assert.Throws<ArgumentException>(() =>
            Drawing.DrawContours(m, contours, contourIndex: -1, White, hierarchy: hierarchy));
    }

    [Fact]
    public void DrawContours_NullContours_Throws()
    {
        var m = new Mat<byte>(8, 8);
        Assert.Throws<ArgumentNullException>(() =>
            Drawing.DrawContours(m, null!, contourIndex: -1, White));
    }

    [Fact]
    public void DrawContours_EmptyContours_ReturnsWithoutDrawing()
    {
        var m = new Mat<byte>(8, 8);
        byte[] before = m.ToArray();

        Drawing.DrawContours(m, Array.Empty<Point[]>(), contourIndex: -1, White, thickness: Drawing.Filled);

        Assert.Equal(before, m.ToArray());
    }
}
