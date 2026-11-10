namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

public sealed class WarpPolarBasicTests
{
    private static Mat<byte> Disc(int size)
    {
        var m = new Mat<byte>(size, size);
        Drawing.Circle(m, new Point(size / 2, size / 2), size / 4, new Scalar(200), -1);
        return m;
    }

    [Fact]
    public void WarpPolar_CentredDisc_IsAStepAlongTheRadius()
    {
        var polar = Geometry.WarpPolar(Disc(65), new Size(64, 90), new Point2f(32, 32), 32, InterpolationFlags.Nearest,
            flags: WarpFlags.FillOutliers);

        for (int y = 0; y < polar.Height; y++)
        {
            Assert.Equal(200, polar.At(y, 5));
            Assert.Equal(0, polar.At(y, 50));
        }
    }

    [Fact]
    public void WarpPolar_EmptySize_UsesRadiusAndPi()
    {
        var polar = Geometry.WarpPolar(Disc(41), default, new Point2f(20, 20), 20, InterpolationFlags.Linear);
        Assert.Equal((20, 63), (polar.Width, polar.Height));

        var widthOnly = Geometry.WarpPolar(Disc(41), new Size(10, 0), new Point2f(20, 20), 20, InterpolationFlags.Linear);
        Assert.Equal((10, 31), (widthOnly.Width, widthOnly.Height));
    }

    [Theory]
    [InlineData(WarpPolarMode.Linear)]
    [InlineData(WarpPolarMode.Log)]
    public void WarpPolar_RoundTrip_RestoresTheDisc(WarpPolarMode mode)
    {
        var src = Filters.GaussianBlur(Disc(65), new Size(5, 5), 1.5);
        var flags = WarpFlags.FillOutliers;
        var polar = Geometry.WarpPolar(src, new Size(65, 65), new Point2f(32, 32), 32, InterpolationFlags.Linear, mode, flags);
        var back = Geometry.WarpPolar(polar, new Size(65, 65), new Point2f(32, 32), 32, InterpolationFlags.Linear, mode, flags | WarpFlags.InverseMap);

        double err = 0;
        int n = 0;
        for (int y = 12; y < 52; y++)
        {
            for (int x = 12; x < 52; x++)
            {
                err += Math.Abs(src.At(y, x) - back.At(y, x));
                n++;
            }
        }

        Assert.True(err / n < 8, $"mean error {err / n}");
    }

    [Fact]
    public void WarpPolar_RoiWithStride_MatchesPackedImage()
    {
        var parent = new Mat<byte>(80, 70);
        var roi = parent.Roi(5, 3, 41, 41);
        Disc(41).CopyTo(roi);
        var packed = roi.Clone();

        foreach (var flags in new[] { WarpFlags.FillOutliers, WarpFlags.FillOutliers | WarpFlags.InverseMap })
        {
            Assert.Equal(
                Geometry.WarpPolar(packed, new Size(30, 50), new Point2f(20, 18), 21, InterpolationFlags.Linear, WarpPolarMode.Log, flags).ToArray(),
                Geometry.WarpPolar(roi, new Size(30, 50), new Point2f(20, 18), 21, InterpolationFlags.Linear, WarpPolarMode.Log, flags).ToArray());
        }
    }
}
