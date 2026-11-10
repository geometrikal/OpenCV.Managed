namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

public sealed class EdgeBasicTests
{
    private static Mat<byte> Smooth(int width, int height, int channels, int seed)
    {
        var m = new Mat<byte>(width, height, channels);
        var random = new Random(seed);
        for (int y = 0; y < height; y++)
        {
            random.NextBytes(m.GetRowSpan(y));
        }

        return Filters.GaussianBlur(m, new Size(5, 5), 1.5);
    }

    private static Mat<T> InParent<T>(Mat<T> packed, int x, int y)
        where T : unmanaged
    {
        var parent = new Mat<T>(packed.Width + x + 9, packed.Height + y + 6, packed.Channels);
        parent.Fill(default);
        Mat<T> roi = parent.Roi(x, y, packed.Width, packed.Height);
        packed.CopyTo(roi);
        return roi;
    }

    // ------------------------------------------------------------------- Canny

    [Theory]
    [InlineData(1, 3, false)]
    [InlineData(3, 3, true)]
    [InlineData(1, 5, true)]
    [InlineData(4, 7, false)]
    [InlineData(1, -1, true)]
    public void Canny_RoiWithStride_MatchesPackedImage(int channels, int aperture, bool l2)
    {
        Mat<byte> packed = Smooth(53, 37, channels, 1);
        Mat<byte> roi = InParent(packed, 5, 3);
        Assert.False(roi.IsContinuous);
        double scale = aperture == 7 ? 64 : aperture == 5 ? 8 : 1;
        Assert.Equal(
            EdgeDetection.Canny(packed, 10 * scale, 30 * scale, aperture, l2).ToArray(),
            EdgeDetection.Canny(roi, 10 * scale, 30 * scale, aperture, l2).ToArray());
    }

    [Theory]
    [InlineData(1, false)]
    [InlineData(3, true)]
    public void CannyDerivatives_RoiWithStride_MatchesPackedImage(int channels, bool l2)
    {
        Mat<byte> img = Smooth(41, 29, channels, 2);
        Mat<short> dx = Derivatives.Sobel<byte, short>(img, 1, 0, 3, 1, 0, BorderType.Replicate);
        Mat<short> dy = Derivatives.Sobel<byte, short>(img, 0, 1, 3, 1, 0, BorderType.Replicate);
        Assert.Equal(
            EdgeDetection.Canny(dx, dy, 20, 60, l2).ToArray(),
            EdgeDetection.Canny(InParent(dx, 2, 7), InParent(dy, 4, 1), 20, 60, l2).ToArray());
    }

    /// <summary>Aperture 3 computes the gradient inside each stripe; it must equal the derivative overload.</summary>
    [Theory]
    [InlineData(1, false)]
    [InlineData(1, true)]
    [InlineData(3, false)]
    [InlineData(4, true)]
    public void Canny_Aperture3_EqualsDerivativeOverloadWithReplicateSobel(int channels, bool l2)
    {
        Mat<byte> img = Smooth(131, 67, channels, 3);
        Mat<short> dx = Derivatives.Sobel<byte, short>(img, 1, 0, 3, 1, 0, BorderType.Replicate);
        Mat<short> dy = Derivatives.Sobel<byte, short>(img, 0, 1, 3, 1, 0, BorderType.Replicate);
        if (channels == 4)
        {
            Assert.Throws<ArgumentException>(() => EdgeDetection.Canny(dx, dy, 15, 45, l2));
            return;
        }

        Assert.Equal(EdgeDetection.Canny(dx, dy, 15, 45, l2).ToArray(), EdgeDetection.Canny(img, 15, 45, 3, l2).ToArray());
    }

    [Fact]
    public void Canny_ScharrAperture_EqualsDerivativeOverloadWithScharr()
    {
        Mat<byte> img = Smooth(70, 45, 1, 4);
        Mat<short> dx = Derivatives.Scharr<byte, short>(img, 1, 0, 1, 0, BorderType.Replicate);
        Mat<short> dy = Derivatives.Scharr<byte, short>(img, 0, 1, 1, 0, BorderType.Replicate);
        Assert.Equal(EdgeDetection.Canny(dx, dy, 50, 150, true).ToArray(), EdgeDetection.Canny(img, 50, 150, -1, true).ToArray());
    }

    [Theory]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(7)]
    public void Canny_LegacyApertureFlag_SelectsL2(int aperture)
    {
        Mat<byte> img = Smooth(60, 40, 1, 5);
        Assert.Equal(
            EdgeDetection.Canny(img, 30, 90, aperture, true).ToArray(),
            EdgeDetection.Canny(img, 30, 90, aperture | int.MinValue, false).ToArray());
    }

    [Fact]
    public void Canny_SwappedThresholds_GiveSameResult()
    {
        Mat<byte> img = Smooth(60, 40, 3, 6);
        Assert.Equal(EdgeDetection.Canny(img, 20, 70).ToArray(), EdgeDetection.Canny(img, 70, 20).ToArray());
    }

    [Fact]
    public void Canny_HighThresholdAboveIntRange_MarksEveryPeak()
    {
        // cvFloor(1e10) is int.MinValue, so every non-maximum-suppressed candidate is an edge.
        Mat<byte> img = Smooth(60, 40, 1, 7);
        Assert.Equal(EdgeDetection.Canny(img, 10, 10.5).ToArray(), EdgeDetection.Canny(img, 10, 1e10).ToArray());
    }

    [Fact]
    public void Canny_ConstantImage_HasNoEdges()
    {
        var img = new Mat<byte>(33, 17, 3);
        img.Fill(128);
        Assert.All(EdgeDetection.Canny(img, 0, 0).ToArray(), v => Assert.Equal(0, v));
    }

    [Fact]
    public void Canny_SinglePixel_ReturnsZero()
    {
        var img = new Mat<byte>(1, 1);
        img.At(0, 0) = 200;
        Mat<byte> edges = EdgeDetection.Canny(img, 0, 0);
        Assert.Equal((1, 1, 1), (edges.Width, edges.Height, edges.Channels));
        Assert.Equal(0, edges.At(0, 0));
    }

    [Fact]
    public void Canny_VerticalStep_MarksOneColumn()
    {
        var img = new Mat<byte>(20, 12);
        for (int y = 0; y < 12; y++)
        {
            for (int x = 10; x < 20; x++)
            {
                img.At(y, x) = 200;
            }
        }

        Mat<byte> edges = EdgeDetection.Canny(img, 100, 300);
        for (int y = 0; y < 12; y++)
        {
            for (int x = 0; x < 20; x++)
            {
                Assert.Equal(x == 9 ? 255 : 0, edges.At(y, x));
            }
        }
    }

    [Fact]
    public void Canny_DoesNotModifySource_AndIsDeterministic()
    {
        Mat<byte> img = Smooth(80, 50, 3, 8);
        byte[] before = img.ToArray();
        byte[] first = EdgeDetection.Canny(img, 10, 40, 5, true).ToArray();
        byte[] second = EdgeDetection.Canny(img, 10, 40, 5, true).ToArray();
        Assert.Equal(before, img.ToArray());
        Assert.Equal(first, second);
    }

    [Fact]
    public void CannyDerivatives_DoesNotModifySources()
    {
        Mat<byte> img = Smooth(40, 30, 1, 9);
        Mat<short> dx = Derivatives.Sobel<byte, short>(img, 1, 0);
        Mat<short> dy = Derivatives.Sobel<byte, short>(img, 0, 1);
        short[] bx = dx.ToArray(), by = dy.ToArray();
        EdgeDetection.Canny(dx, dy, 10, 30, true);
        Assert.Equal(bx, dx.ToArray());
        Assert.Equal(by, dy.ToArray());
    }

    /// <summary>
    /// <c>|dx| = 30000</c>: <c>tg67x = 30000 · 79109</c> wraps to a negative int32, so the pixel uses
    /// the vertical neighbours (both zero) and is an edge. Without the wrap the direction is
    /// diagonal and the larger up-left magnitude suppresses it.
    /// </summary>
    [Fact]
    public void CannyDerivatives_Tg67Overflow_SelectsVerticalNeighbours()
    {
        var dx = new Mat<short>(3, 3);
        var dy = new Mat<short>(3, 3);
        dx.At(0, 0) = 30000;
        dy.At(0, 0) = 30000;
        dx.At(1, 1) = 30000;
        dy.At(1, 1) = 20000;
        Mat<byte> edges = EdgeDetection.Canny(dx, dy, 100, 40000);
        Assert.Equal(255, edges.At(1, 1));
    }

    /// <summary>The L2 magnitude of <c>dx = dy = -32768</c> wraps to <c>int.MinValue</c> and is never a candidate.</summary>
    [Fact]
    public void CannyDerivatives_L2MagnitudeWrap_IsNotACandidate()
    {
        var dx = new Mat<short>(5, 1);
        var dy = new Mat<short>(5, 1);
        dx.At(0, 2) = short.MinValue;
        dy.At(0, 2) = short.MinValue;
        Assert.Equal(0, EdgeDetection.Canny(dx, dy, -5, -5, true).At(0, 2));
        dy.At(0, 2) = short.MinValue + 1;
        Assert.Equal(255, EdgeDetection.Canny(dx, dy, 0, 0, true).At(0, 2));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(9)]
    [InlineData(0)]
    [InlineData(-3)]
    public void Canny_InvalidAperture_Throws(int aperture)
    {
        Mat<byte> img = Smooth(10, 10, 1, 10);
        Assert.Throws<ArgumentOutOfRangeException>(() => EdgeDetection.Canny(img, 10, 20, aperture));
    }

    [Fact]
    public void Canny_ScharrApertureWithL1_Throws()
    {
        Mat<byte> img = Smooth(10, 10, 1, 11);
        Assert.Throws<ArgumentOutOfRangeException>(() => EdgeDetection.Canny(img, 10, 20, -1, false));
    }

    [Fact]
    public void Canny_InvalidArguments_Throw()
    {
        var d1 = new Mat<short>(8, 6);
        Assert.Throws<ArgumentNullException>(() => EdgeDetection.Canny(null!, 1, 2));
        Assert.Throws<ArgumentNullException>(() => EdgeDetection.Canny(null!, d1, 1, 2));
        Assert.Throws<ArgumentNullException>(() => EdgeDetection.Canny(d1, null!, 1, 2));
        Assert.Throws<ArgumentException>(() => EdgeDetection.Canny(new Mat<short>(8, 6, 2), new Mat<short>(8, 6, 2), 1, 2));
        Assert.Throws<ArgumentException>(() => EdgeDetection.Canny(new Mat<short>(8, 6, 4), new Mat<short>(8, 6, 4), 1, 2));
        Assert.Throws<ArgumentException>(() => EdgeDetection.Canny(d1, new Mat<short>(8, 7), 1, 2));
        Assert.Throws<ArgumentException>(() => EdgeDetection.Canny(d1, new Mat<short>(9, 6), 1, 2));
        Assert.Throws<ArgumentException>(() => EdgeDetection.Canny(d1, new Mat<short>(8, 6, 3), 1, 2));
    }

    // --------------------------------------------------------- spatialGradient

    [Theory]
    [InlineData(BorderType.Reflect101, 1, 1)]
    [InlineData(BorderType.Replicate, 1, 40)]
    [InlineData(BorderType.Reflect101, 40, 1)]
    [InlineData(BorderType.Replicate, 77, 31)]
    [InlineData(BorderType.Reflect101, 130, 9)]
    public void SpatialGradient_EqualsSobel(BorderType border, int width, int height)
    {
        Mat<byte> img = Smooth(width, height, 1, 12);
        EdgeDetection.SpatialGradient(img, out Mat<short> dx, out Mat<short> dy, 3, border);
        Assert.Equal(Derivatives.Sobel<byte, short>(img, 1, 0, 3, 1, 0, border).ToArray(), dx.ToArray());
        Assert.Equal(Derivatives.Sobel<byte, short>(img, 0, 1, 3, 1, 0, border).ToArray(), dy.ToArray());
    }

    [Fact]
    public void SpatialGradient_RoiWithStride_MatchesPackedImage()
    {
        Mat<byte> packed = Smooth(67, 23, 1, 13);
        Mat<byte> roi = InParent(packed, 6, 2);
        EdgeDetection.SpatialGradient(packed, out Mat<short> px, out Mat<short> py);
        EdgeDetection.SpatialGradient(roi, out Mat<short> rx, out Mat<short> ry);
        Assert.Equal(px.ToArray(), rx.ToArray());
        Assert.Equal(py.ToArray(), ry.ToArray());
    }

    [Fact]
    public void SpatialGradient_DoesNotModifySource()
    {
        Mat<byte> img = Smooth(30, 20, 1, 14);
        byte[] before = img.ToArray();
        EdgeDetection.SpatialGradient(img, out _, out _, 3, BorderType.Replicate);
        Assert.Equal(before, img.ToArray());
    }

    [Fact]
    public void SpatialGradient_InvalidArguments_Throw()
    {
        var img = new Mat<byte>(8, 8);
        Assert.Throws<ArgumentNullException>(() => EdgeDetection.SpatialGradient(null!, out _, out _));
        Assert.Throws<ArgumentException>(() => EdgeDetection.SpatialGradient(new Mat<byte>(8, 8, 3), out _, out _));
        Assert.Throws<ArgumentOutOfRangeException>(() => EdgeDetection.SpatialGradient(img, out _, out _, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => EdgeDetection.SpatialGradient(img, out _, out _, 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => EdgeDetection.SpatialGradient(img, out _, out _, -1));
        Assert.Throws<ArgumentException>(() => EdgeDetection.SpatialGradient(img, out _, out _, 3, BorderType.Constant));
        Assert.Throws<ArgumentException>(() => EdgeDetection.SpatialGradient(img, out _, out _, 3, BorderType.Reflect));
        Assert.Throws<ArgumentException>(() => EdgeDetection.SpatialGradient(img, out _, out _, 3, BorderType.Wrap));
    }
}
