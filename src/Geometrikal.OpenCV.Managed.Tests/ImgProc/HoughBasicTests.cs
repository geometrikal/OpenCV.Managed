namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

public sealed class HoughBasicTests
{
    private static Mat<byte> HorizontalLine(int width, int height, int y)
    {
        var m = new Mat<byte>(width, height);
        for (int x = 0; x < width; x++)
        {
            m.At(y, x) = 255;
        }

        return m;
    }

    [Fact]
    public void HoughLines_EmptyImage_ReturnsNoLines()
    {
        Assert.Empty(HoughTransforms.HoughLines(new Mat<byte>(20, 20), 1, Math.PI / 180, 5));
        Assert.Empty(HoughTransforms.HoughLinesP(new Mat<byte>(20, 20), 1, Math.PI / 180, 5));
    }

    [Fact]
    public void HoughLines_HorizontalLine_IsFoundAtNinetyDegrees()
    {
        var lines = HoughTransforms.HoughLines(HorizontalLine(40, 30, 12), 1, Math.PI / 180, 30);

        Assert.NotEmpty(lines);
        Assert.Equal(12f, lines[0].Rho);
        Assert.Equal(Math.PI / 2, lines[0].Theta, 1e-5);
        Assert.Equal(40f, lines[0].Votes);
    }

    [Fact]
    public void HoughLines_EdgeValueVoting_UsesPixelValues()
    {
        var image = HorizontalLine(40, 30, 12);

        var lines = HoughTransforms.HoughLines(image, 1, Math.PI / 180, 30, useEdgeValue: true);

        Assert.Equal(40f * 255, lines[0].Votes);
    }

    [Fact]
    public void HoughLinesP_HorizontalLine_IsOneSegment()
    {
        var segments = HoughTransforms.HoughLinesP(HorizontalLine(40, 30, 12), 1, Math.PI / 180, 10, 20, 2);

        var s = Assert.Single(segments);
        Assert.Equal(12, s.Y1);
        Assert.Equal(12, s.Y2);
        Assert.Equal(39, Math.Abs(s.X2 - s.X1));
    }

    [Fact]
    public void Hough_RoiWithStride_MatchesPackedImage()
    {
        var parent = new Mat<byte>(60, 50);
        var roi = parent.Roi(7, 5, 40, 30);
        HorizontalLine(40, 30, 12).CopyTo(roi);
        for (int i = 0; i < 25; i++)
        {
            roi.At(i, i + 3) = 255;
        }

        var packed = roi.Clone();

        Assert.Equal(HoughTransforms.HoughLines(packed, 1, Math.PI / 180, 15), HoughTransforms.HoughLines(roi, 1, Math.PI / 180, 15));
        Assert.Equal(HoughTransforms.HoughLinesP(packed, 1, Math.PI / 180, 10, 10, 2), HoughTransforms.HoughLinesP(roi, 1, Math.PI / 180, 10, 10, 2));
    }

    [Fact]
    public void HoughLinesP_DoesNotModifySource_AndIsDeterministic()
    {
        var image = HorizontalLine(40, 30, 12);
        byte[] before = image.ToArray();

        var first = HoughTransforms.HoughLinesP(image, 1, Math.PI / 180, 10, 5, 1);
        var second = HoughTransforms.HoughLinesP(image, 1, Math.PI / 180, 10, 5, 1);

        Assert.Equal(before, image.ToArray());
        Assert.Equal(first, second);
    }

    [Fact]
    public void HoughCircles_DrawnCircle_IsFound()
    {
        var image = new Mat<byte>(100, 80);
        Drawing.Circle(image, new Point(48, 37), 20, new Scalar(255), -1);
        var blurred = Filters.GaussianBlur(image, new Size(5, 5), 1.5);

        var circles = HoughTransforms.HoughCircles(blurred, 1, 20, 100, 15, 10, 30);

        Assert.NotEmpty(circles);
        Assert.Equal(48, circles[0].X, 1.5f);
        Assert.Equal(37, circles[0].Y, 1.5f);
        Assert.Equal(20, circles[0].Radius, 1.5f);
    }

    [Fact]
    public void HoughCircles_RoiWithStride_MatchesPackedImage()
    {
        var parent = new Mat<byte>(130, 110);
        var roi = parent.Roi(9, 11, 100, 80);
        Drawing.Circle(roi, new Point(48, 37), 20, new Scalar(255), -1);
        var packed = roi.Clone();

        Assert.Equal(HoughTransforms.HoughCircles(packed, 1, 20, 100, 15), HoughTransforms.HoughCircles(roi, 1, 20, 100, 15));
    }

    [Fact]
    public void HoughCircles_InvalidInputs_Throw()
    {
        var image = new Mat<byte>(20, 20);
        Assert.Throws<ArgumentOutOfRangeException>(() => HoughTransforms.HoughCircles(image, 0, 10, 100, 10));
        Assert.Throws<ArgumentOutOfRangeException>(() => HoughTransforms.HoughCircles(image, 1, 10, 100, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => HoughTransforms.HoughCircles(image, 1, 10, 100, 10, sobelKernelSize: 4));
    }

    [Fact]
    public void Hough_InvalidInputs_Throw()
    {
        Assert.Throws<ArgumentException>(() => HoughTransforms.HoughLines(new Mat<byte>(5, 5, 3), 1, 0.1, 1));
        Assert.Throws<ArgumentException>(() => HoughTransforms.HoughLines(new Mat<byte>(5, 5), 1, 0.1, 1, minTheta: 2, maxTheta: 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => HoughTransforms.HoughLinesPointSet([new Point2f(1, 1)], 0, 1, 0, 10, 1, 0, 1, 0.1));
        Assert.Throws<ArgumentException>(() => HoughTransforms.HoughLinesPointSet([new Point2f(1, 1)], 5, 1, 10, 0, 1, 0, 1, 0.1));
    }
}
