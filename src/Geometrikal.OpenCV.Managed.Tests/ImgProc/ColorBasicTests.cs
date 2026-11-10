namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

public sealed class ColorBasicTests
{
    private static Mat<byte> RandomBgr(int w, int h, int seed)
    {
        var m = new Mat<byte>(w, h, 3);
        new Random(seed).NextBytes(m.GetSpan());
        return m;
    }

    [Fact]
    public void CvtColor_WrongChannelCount_Throws()
    {
        Assert.Throws<ArgumentException>(() => ColorConversions.CvtColor(new Mat<byte>(4, 4, 1), ColorConversionCode.Bgr2Gray));
        Assert.Throws<ArgumentException>(() => ColorConversions.CvtColor(new Mat<byte>(4, 4, 3), ColorConversionCode.Gray2Bgr));
        Assert.Throws<ArgumentException>(() => ColorConversions.CvtColor(new Mat<byte>(4, 4, 4), ColorConversionCode.Hsv2Bgr));
        Assert.Throws<ArgumentException>(() => ColorConversions.CvtColor(new Mat<byte>(4, 4, 3), ColorConversionCode.Lab2Bgr, 5));
    }

    [Fact]
    public void CvtColor_UnsupportedDepth_Throws()
    {
        Assert.Throws<NotSupportedException>(() => ColorConversions.CvtColor(new Mat<ushort>(4, 4, 3), ColorConversionCode.Bgr2Hsv));
        Assert.Throws<NotSupportedException>(() => ColorConversions.CvtColor(new Mat<ushort>(4, 4, 3), ColorConversionCode.Lab2Bgr));
        Assert.Throws<NotSupportedException>(() => ColorConversions.CvtColor(new Mat<short>(4, 4, 3), ColorConversionCode.Bgr2Gray));
        Assert.Throws<ArgumentOutOfRangeException>(() => ColorConversions.CvtColor(new Mat<byte>(4, 4, 3), (ColorConversionCode)50));
    }

    [Fact]
    public void CvtColor_DoesNotModifySource_AndIsDeterministic()
    {
        Mat<byte> src = RandomBgr(53, 7, 1);
        byte[] before = src.ToArray();
        byte[] first = ColorConversions.CvtColor(src, ColorConversionCode.Bgr2Lab).ToArray();
        byte[] second = ColorConversions.CvtColor(src, ColorConversionCode.Bgr2Lab).ToArray();
        Assert.Equal(before, src.ToArray());
        Assert.Equal(first, second);
    }

    [Fact]
    public void CvtColor_RoiWithStride_MatchesPackedImage()
    {
        Mat<byte> packed = RandomBgr(37, 6, 2);
        var parent = new Mat<byte>(50, 10, 3);
        Mat<byte> roi = parent.Roi(5, 2, 37, 6);
        packed.CopyTo(roi);
        foreach (ColorConversionCode code in new[] { ColorConversionCode.Bgr2Hsv, ColorConversionCode.Bgr2Gray, ColorConversionCode.Rgb2YCrCb })
        {
            Assert.Equal(ColorConversions.CvtColor(packed, code).ToArray(), ColorConversions.CvtColor(roi, code).ToArray());
        }
    }

    [Fact]
    public void CvtColor_SinglePixel()
    {
        var px = new Mat<byte>([255, 0, 0], 1, 1, 3);
        Assert.Equal(new byte[] { 29 }, ColorConversions.CvtColor(px, ColorConversionCode.Bgr2Gray).ToArray());
        Assert.Equal(new byte[] { 120, 255, 255 }, ColorConversions.CvtColor(px, ColorConversionCode.Bgr2Hsv).ToArray());
        Assert.Equal(new byte[] { 0, 0, 255, 255 }, ColorConversions.CvtColor(px, ColorConversionCode.Bgr2Rgba).ToArray());
    }

    [Fact]
    public void CvtColor_ZeroImage_GivesNeutralValues()
    {
        var black = new Mat<byte>(5, 2, 3);
        Assert.All(ColorConversions.CvtColor(black, ColorConversionCode.Bgr2Hsv).ToArray(), v => Assert.Equal(0, v));
        byte[] lab = ColorConversions.CvtColor(black, ColorConversionCode.Bgr2Lab).ToArray();
        Assert.Equal(new byte[] { 0, 128, 128 }, lab[..3]);
        byte[] ycrcb = ColorConversions.CvtColor(black, ColorConversionCode.Bgr2YCrCb).ToArray();
        Assert.Equal(new byte[] { 0, 128, 128 }, ycrcb[..3]);
    }

    /// <summary>
    /// OpenCV 5.0 wraps a negative float hue into sector 0..5 (<c>ComputeSectorAndClampedH</c>);
    /// cv2 4.5.5 maps it to sector 0 with h = 0.
    /// </summary>
    [Fact]
    public void HsvToBgr_NegativeFloatHue_WrapsSector()
    {
        var hsv = new Mat<float>([-30f, 1f, 1f], 1, 1, 3);
        float[] bgr = ColorConversions.CvtColor(hsv, ColorConversionCode.Hsv2Bgr).ToArray();
        float h = -30f * (6f / 360f);
        h -= MathF.Floor(h);
        Assert.Equal(1f - h, bgr[0]);
        Assert.Equal(0f, bgr[1]);
        Assert.Equal(1f, bgr[2]);
    }
}
