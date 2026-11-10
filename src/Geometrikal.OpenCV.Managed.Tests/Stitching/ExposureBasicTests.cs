namespace Geometrikal.OpenCV.Managed.Tests.Stitching;

using System;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Stitching;

public sealed class ExposureBasicTests
{
    private static Mat<byte> Filled(int w, int h, int cn, double v)
    {
        var m = new Mat<byte>(w, h, cn);
        m.SetTo(Scalar.All(v));
        return m;
    }

    [Theory]
    [InlineData(ExposureCompensatorType.No, typeof(NoExposureCompensator))]
    [InlineData(ExposureCompensatorType.Gain, typeof(GainCompensator))]
    [InlineData(ExposureCompensatorType.GainBlocks, typeof(BlocksGainCompensator))]
    [InlineData(ExposureCompensatorType.Channels, typeof(ChannelsCompensator))]
    [InlineData(ExposureCompensatorType.ChannelsBlocks, typeof(BlocksChannelsCompensator))]
    public void CreateDefault_ReturnsCompensatorOfType(ExposureCompensatorType type, Type expected)
        => Assert.IsType(expected, ExposureCompensator.CreateDefault(type));

    [Fact]
    public void GainCompensator_NoOverlap_GainsAreOne()
    {
        var c = new GainCompensator();
        c.Feed(new[] { new Point(0, 0), new Point(100, 0) }, new[] { Filled(10, 10, 3, 50), Filled(10, 10, 3, 200) }, new[] { Filled(10, 10, 1, 255), Filled(10, 10, 1, 255) });
        Assert.Equal(new[] { 1.0, 1.0 }, c.Gains);
    }

    [Fact]
    public void GainCompensator_Overlap_BrightensDarkImageAndDarkensBrightImage()
    {
        var c = new GainCompensator();
        c.Feed(new[] { new Point(0, 0), new Point(5, 0) }, new[] { Filled(10, 10, 3, 80), Filled(10, 10, 3, 160) }, new[] { Filled(10, 10, 1, 255), Filled(10, 10, 1, 255) });
        double[] g = c.Gains;
        Assert.True(g[0] > 1 && g[1] < 1);
        Mat<byte> img = Filled(10, 10, 3, 80);
        c.Apply(0, new Point(0, 0), img, null);
        Assert.True(img.At(0, 0, 0) > 80);
    }

    [Fact]
    public void NoExposureCompensator_Apply_LeavesImage()
    {
        Mat<byte> img = Filled(4, 4, 3, 33);
        new NoExposureCompensator().Apply(0, default, img, null);
        Assert.Equal(33, img.At(3, 3, 2));
    }

    [Fact]
    public void BlocksCompensator_SingleChannelImage_ApplyThrows()
    {
        var c = new BlocksGainCompensator(8, 8);
        c.Feed(new[] { new Point(0, 0), new Point(4, 0) }, new[] { Filled(16, 16, 3, 80), Filled(16, 16, 3, 120) }, new[] { Filled(16, 16, 1, 255), Filled(16, 16, 1, 255) });
        Assert.Equal(2, c.GainMaps.Length);
        Assert.Throws<ArgumentException>(() => c.Apply(0, default, Filled(16, 16, 1, 1), null));
    }

    [Fact]
    public void Feed_MismatchedLists_Throws()
        => Assert.Throws<ArgumentException>(() => new GainCompensator().Feed(new[] { new Point(0, 0) }, new[] { Filled(4, 4, 3, 1), Filled(4, 4, 3, 1) }, new[] { Filled(4, 4, 1, 255) }));
}
