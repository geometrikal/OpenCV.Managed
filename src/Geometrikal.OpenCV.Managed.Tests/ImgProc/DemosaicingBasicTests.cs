namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

public sealed class DemosaicingBasicTests
{
    // A flat colour (B, G, R) sampled with the pattern BG (sensor RGGB).
    private static Mat<byte> FlatMosaic(int w, int h, byte b, byte g, byte r)
    {
        var m = new Mat<byte>(w, h);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                m.At(y, x) = (y & 1, x & 1) switch { (0, 0) => r, (1, 1) => b, _ => g };
            }
        }

        return m;
    }

    [Theory]
    [InlineData(DemosaicMethod.Bilinear)]
    [InlineData(DemosaicMethod.EdgeAware)]
    [InlineData(DemosaicMethod.Vng)]
    public void Demosaic_FlatColour_IsRecovered(DemosaicMethod method)
    {
        var bgr = Demosaicing.Demosaic(FlatMosaic(24, 20, 30, 120, 210), BayerPattern.BG, method);

        for (int y = 0; y < 20; y++)
        {
            var row = bgr.GetReadOnlyRowSpan(y);
            for (int x = 0; x < 24; x++)
            {
                Assert.Equal((30, 120, 210), (row[3 * x], row[(3 * x) + 1], row[(3 * x) + 2]));
            }
        }
    }

    [Fact]
    public void Demosaic_Bgra_HasOpaqueAlpha()
    {
        var bgra = Demosaicing.Demosaic(FlatMosaic(10, 6, 1, 2, 3), BayerPattern.BG, dstChannels: 4);
        var row = bgra.GetReadOnlyRowSpan(3);
        Assert.Equal((1, 2, 3, 255), (row[8], row[9], row[10], row[11]));
    }

    [Fact]
    public void Demosaic_RoiWithStride_MatchesPackedImage()
    {
        var parent = new Mat<ushort>(50, 40);
        var roi = parent.Roi(3, 5, 33, 21);
        var rng = new Random(11);
        for (int y = 0; y < 21; y++)
        {
            var row = roi.GetRowSpan(y);
            for (int x = 0; x < 33; x++)
            {
                row[x] = (ushort)rng.Next(65536);
            }
        }

        var packed = roi.Clone();
        Assert.Equal(Demosaicing.Demosaic(packed, BayerPattern.GR).ToArray(), Demosaicing.Demosaic(roi, BayerPattern.GR).ToArray());
        Assert.Equal(Demosaicing.Demosaic(packed, BayerPattern.GB, DemosaicMethod.EdgeAware).ToArray(),
            Demosaicing.Demosaic(roi, BayerPattern.GB, DemosaicMethod.EdgeAware).ToArray());
        Assert.Equal(Demosaicing.BayerToGray(packed, BayerPattern.RG).ToArray(), Demosaicing.BayerToGray(roi, BayerPattern.RG).ToArray());
    }

    [Fact]
    public void Demosaic_InvalidInputs_Throw()
    {
        Assert.Throws<NotSupportedException>(() => Demosaicing.Demosaic(new Mat<float>(8, 8), BayerPattern.BG));
        Assert.Throws<NotSupportedException>(() => Demosaicing.Demosaic(new Mat<ushort>(8, 8), BayerPattern.BG, DemosaicMethod.Vng));
        Assert.Throws<ArgumentException>(() => Demosaicing.Demosaic(new Mat<byte>(8, 8, 3), BayerPattern.BG));
        Assert.Throws<ArgumentOutOfRangeException>(() => Demosaicing.Demosaic(new Mat<byte>(8, 8), BayerPattern.BG, DemosaicMethod.EdgeAware, 4));
    }
}
