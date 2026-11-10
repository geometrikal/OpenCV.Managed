namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Collections.Generic;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Validation, routing, ROI and edge-case tests for <see cref="DistanceTransforms"/> and
/// <see cref="Segmentation.Watershed"/>, with brute-force references for the metrics that the
/// masks compute exactly (L1, C, precise L2) and ports of the OpenCV accuracy tests.
/// </summary>
public sealed class DistanceBasicTests
{
    public static IEnumerable<object[]> Modes()
    {
        foreach (DistanceTypes t in new[] { DistanceTypes.L1, DistanceTypes.L2, DistanceTypes.C })
        {
            foreach (DistanceTransformMasks m in new[] { DistanceTransformMasks.Mask3, DistanceTransformMasks.Mask5, DistanceTransformMasks.Precise })
            {
                yield return [t, m];
            }
        }
    }

    public static IEnumerable<object[]> Sizes()
    {
        foreach ((int w, int h) in new[] { (1, 1), (1, 9), (9, 1), (31, 5), (32, 3), (33, 7), (40, 40), (97, 13) })
        {
            yield return [w, h];
        }
    }

    private static Mat<byte> Random(int w, int h, int seed, double zeroFraction)
    {
        var rng = new Random(seed);
        var m = new Mat<byte>(w, h);
        for (int y = 0; y < h; y++)
        {
            Span<byte> row = m.GetRowSpan(y);
            for (int x = 0; x < w; x++)
            {
                row[x] = rng.NextDouble() < zeroFraction ? (byte)0 : (byte)rng.Next(1, 256);
            }
        }

        if (zeroFraction > 0)
        {
            m.At(rng.Next(h), rng.Next(w)) = 0;
        }

        return m;
    }

    /// <summary>Minimum over the zero pixels of <paramref name="metric"/>(dx, dy).</summary>
    private static double BruteForce(Mat<byte> src, int x, int y, Func<int, int, double> metric)
    {
        double best = double.MaxValue;
        for (int yy = 0; yy < src.Height; yy++)
        {
            for (int xx = 0; xx < src.Width; xx++)
            {
                if (src.At(yy, xx) == 0)
                {
                    best = Math.Min(best, metric(Math.Abs(xx - x), Math.Abs(yy - y)));
                }
            }
        }

        return best;
    }

    [Theory]
    [MemberData(nameof(Sizes))]
    public void DistanceTransform_L1AndC_MatchBruteForce(int w, int h)
    {
        Mat<byte> src = Random(w, h, (w * 31) + h, 0.04);
        Mat<float> l1 = DistanceTransforms.DistanceTransform(src, DistanceTypes.L1, DistanceTransformMasks.Mask3);
        Mat<float> c = DistanceTransforms.DistanceTransform(src, DistanceTypes.C, DistanceTransformMasks.Mask3);
        var l1Byte = new Mat<byte>(w, h);
        DistanceTransforms.DistanceTransform(src, l1Byte, DistanceTypes.L1, DistanceTransformMasks.Mask3);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                double expectedL1 = BruteForce(src, x, y, (dx, dy) => dx + dy);
                Assert.Equal((float)expectedL1, l1.At(y, x));
                Assert.Equal((float)BruteForce(src, x, y, (dx, dy) => Math.Max(dx, dy)), c.At(y, x));
                Assert.Equal((byte)Math.Min(expectedL1, 255), l1Byte.At(y, x));
            }
        }
    }

    [Theory]
    [MemberData(nameof(Sizes))]
    public void DistanceTransform_Precise_MatchesBruteForceEuclidean(int w, int h)
    {
        Mat<byte> src = Random(w, h, (w * 17) + h, 0.03);
        Mat<float> d = DistanceTransforms.DistanceTransform(src, DistanceTypes.L2, DistanceTransformMasks.Precise);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                double d2 = BruteForce(src, x, y, (dx, dy) => (dx * dx) + (dy * dy));
                Assert.Equal(MathF.Sqrt((float)d2), d.At(y, x));
            }
        }
    }

    /// <summary>The 5x5 L1 mask (1, 2, 3) is exact, so the labelled zero pixel lies at the reported city-block distance.</summary>
    [Theory]
    [MemberData(nameof(Sizes))]
    public void DistanceTransformWithLabels_PixelLabels_PointToZeroPixelAtReportedDistance(int w, int h)
    {
        Mat<byte> src = Random(w, h, (w * 7) + h, 0.05);
        var position = new Dictionary<int, (int X, int Y)>();
        int k = 1;
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                if (src.At(y, x) == 0)
                {
                    position[k++] = (x, y);
                }
            }
        }

        DistanceTransformResult r = DistanceTransforms.DistanceTransformWithLabels(src, DistanceTypes.L1, DistanceTransformMasks.Mask5, DistanceTransformLabelTypes.Pixel);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                (int zx, int zy) = position[r.Labels.At(y, x)];
                Assert.Equal((float)(Math.Abs(zx - x) + Math.Abs(zy - y)), r.Distances.At(y, x));
                Assert.Equal((float)BruteForce(src, x, y, (dx, dy) => dx + dy), r.Distances.At(y, x));
            }
        }
    }

    [Fact]
    public void DistanceTransformWithLabels_ComponentLabels_NumberZeroComponentsInRasterOrder()
    {
        var src = new Mat<byte>(7, 3);
        src.Fill(1);
        src.At(0, 5) = 0;
        src.At(1, 0) = 0;
        src.At(2, 1) = 0;
        src.At(2, 6) = 0;
        src.At(1, 6) = 0;
        DistanceTransformResult r = DistanceTransforms.DistanceTransformWithLabels(src, DistanceTypes.L2, DistanceTransformMasks.Mask3);
        Assert.Equal(1, r.Labels.At(0, 5));
        Assert.Equal(2, r.Labels.At(1, 0));
        Assert.Equal(2, r.Labels.At(2, 1));
        Assert.Equal(1, r.Labels.At(1, 6));
        Assert.Equal(1, r.Labels.At(2, 6));
        Assert.Equal(2, r.Labels.At(0, 0));
        Assert.Equal(1, r.Labels.At(0, 6));
    }

    [Theory]
    [MemberData(nameof(Modes))]
    public void DistanceTransform_RoiWithStride_MatchesPackedImage(DistanceTypes type, DistanceTransformMasks mask)
    {
        Mat<byte> packed = Random(45, 19, 3, 0.02);
        var parent = new Mat<byte>(52, 25);
        parent.Fill(0);
        Mat<byte> roi = parent.Roi(4, 3, 45, 19);
        packed.CopyTo(roi);
        var dstParent = new Mat<float>(50, 23);
        dstParent.Fill(-5f);
        Mat<float> dst = dstParent.Roi(2, 2, 45, 19);

        DistanceTransforms.DistanceTransform(roi, dst, type, mask);

        Assert.Equal(DistanceTransforms.DistanceTransform(packed, type, mask).ToArray(), dst.ToArray());
        Assert.Equal(-5f, dstParent.At(0, 0));
        Assert.Equal(-5f, dstParent.At(22, 49));
        Assert.Equal(-5f, dstParent.At(2, 1));
    }

    [Theory]
    [MemberData(nameof(Modes))]
    public void DistanceTransform_DoesNotModifySourceAndIsDeterministic(DistanceTypes type, DistanceTransformMasks mask)
    {
        Mat<byte> src = Random(70, 33, 9, 0.01);
        byte[] before = src.ToArray();
        float[] first = DistanceTransforms.DistanceTransform(src, type, mask).ToArray();
        float[] second = DistanceTransforms.DistanceTransform(src, type, mask).ToArray();
        DistanceTransforms.DistanceTransformWithLabels(src, DistanceTypes.L2, mask);
        DistanceTransforms.DistanceTransform(src, new Mat<byte>(70, 33), DistanceTypes.L1, mask);
        Assert.Equal(before, src.ToArray());
        Assert.Equal(first, second);
    }

    [Fact]
    public void DistanceTransform_L1AndC_IgnoreMaskSizeWithoutLabels()
    {
        Mat<byte> src = Random(40, 21, 5, 0.03);
        foreach (DistanceTypes t in new[] { DistanceTypes.L1, DistanceTypes.C })
        {
            float[] mask3 = DistanceTransforms.DistanceTransform(src, t, DistanceTransformMasks.Mask3).ToArray();
            Assert.Equal(mask3, DistanceTransforms.DistanceTransform(src, t, DistanceTransformMasks.Mask5).ToArray());
            Assert.Equal(mask3, DistanceTransforms.DistanceTransform(src, t, DistanceTransformMasks.Precise).ToArray());
        }
    }

    [Theory]
    [InlineData(DistanceTypes.L12)]
    [InlineData(DistanceTypes.Fair)]
    [InlineData(DistanceTypes.Welsch)]
    [InlineData(DistanceTypes.Huber)]
    [InlineData(DistanceTypes.User)]
    [InlineData((DistanceTypes)99)]
    public void DistanceTransform_Precise_DoesNotCheckDistanceType(DistanceTypes type)
    {
        Mat<byte> src = Random(37, 12, 6, 0.05);
        Assert.Equal(
            DistanceTransforms.DistanceTransform(src, DistanceTypes.L2, DistanceTransformMasks.Precise).ToArray(),
            DistanceTransforms.DistanceTransform(src, type, DistanceTransformMasks.Precise).ToArray());
    }

    [Theory]
    [InlineData(DistanceTransformMasks.Mask3)]
    [InlineData(DistanceTransformMasks.Precise)]
    [InlineData((DistanceTransformMasks)4)]
    public void DistanceTransformWithLabels_AlwaysUsesMask5(DistanceTransformMasks mask)
    {
        Mat<byte> src = Random(37, 12, 7, 0.05);
        DistanceTransformResult five = DistanceTransforms.DistanceTransformWithLabels(src, DistanceTypes.L2, DistanceTransformMasks.Mask5);
        DistanceTransformResult other = DistanceTransforms.DistanceTransformWithLabels(src, DistanceTypes.L2, mask);
        Assert.Equal(five.Distances.ToArray(), other.Distances.ToArray());
        Assert.Equal(five.Labels.ToArray(), other.Labels.ToArray());
    }

    [Fact]
    public void DistanceTransform_ByteDestination_IgnoresMaskSizeAndSaturates()
    {
        var src = new Mat<byte>(3, 300);
        src.Fill(9);
        src.At(0, 0) = 0;
        var a = new Mat<byte>(3, 300);
        var b = new Mat<byte>(3, 300);
        DistanceTransforms.DistanceTransform(src, a, DistanceTypes.L1, DistanceTransformMasks.Mask3);
        DistanceTransforms.DistanceTransform(src, b, DistanceTypes.L1, (DistanceTransformMasks)4);
        Assert.Equal(a.ToArray(), b.ToArray());
        Assert.Equal(254, a.At(254, 0));
        Assert.Equal(255, a.At(255, 0));
        Assert.Equal(255, a.At(299, 2));
    }

    [Fact]
    public void DistanceTransform_AllZero_GivesZero()
    {
        var src = new Mat<byte>(33, 4);
        foreach (object[] mode in Modes())
        {
            Assert.All(DistanceTransforms.DistanceTransform(src, (DistanceTypes)mode[0], (DistanceTransformMasks)mode[1]).ToArray(), v => Assert.Equal(0f, v));
        }

        DistanceTransformResult r = DistanceTransforms.DistanceTransformWithLabels(src, DistanceTypes.C, DistanceTransformMasks.Mask5, DistanceTransformLabelTypes.Pixel);
        Assert.All(r.Distances.ToArray(), v => Assert.Equal(0f, v));
        Assert.Equal(1, r.Labels.At(0, 0));
        Assert.Equal(33 * 4, r.Labels.At(3, 32));
    }

    [Fact]
    public void DistanceTransform_InvalidArguments_Throw()
    {
        var src = new Mat<byte>(5, 4);
        Assert.Throws<ArgumentNullException>(() => DistanceTransforms.DistanceTransform(null!, DistanceTypes.L2, DistanceTransformMasks.Mask3));
        Assert.Throws<ArgumentNullException>(() => DistanceTransforms.DistanceTransform(src, (Mat<float>)null!, DistanceTypes.L2, DistanceTransformMasks.Mask3));
        Assert.Throws<ArgumentException>(() => DistanceTransforms.DistanceTransform(new Mat<byte>(5, 4, 3), DistanceTypes.L2, DistanceTransformMasks.Mask3));
        Assert.Throws<ArgumentException>(() => DistanceTransforms.DistanceTransform(src, new Mat<float>(5, 5), DistanceTypes.L2, DistanceTransformMasks.Mask3));
        Assert.Throws<ArgumentException>(() => DistanceTransforms.DistanceTransform(src, new Mat<float>(5, 4, 2), DistanceTypes.L2, DistanceTransformMasks.Mask3));

        var mask = Assert.Throws<ArgumentOutOfRangeException>(() => DistanceTransforms.DistanceTransform(src, DistanceTypes.L2, (DistanceTransformMasks)4));
        Assert.Equal("maskSize", mask.ParamName);
        mask = Assert.Throws<ArgumentOutOfRangeException>(() => DistanceTransforms.DistanceTransform(src, DistanceTypes.L1, (DistanceTransformMasks)4));
        Assert.Equal("maskSize", mask.ParamName);
        mask = Assert.Throws<ArgumentOutOfRangeException>(() => DistanceTransforms.DistanceTransform(src, DistanceTypes.L12, (DistanceTransformMasks)4));
        Assert.Equal("maskSize", mask.ParamName);
        var type = Assert.Throws<ArgumentOutOfRangeException>(() => DistanceTransforms.DistanceTransform(src, DistanceTypes.L12, DistanceTransformMasks.Mask3));
        Assert.Equal("distanceType", type.ParamName);
        Assert.Throws<ArgumentOutOfRangeException>(() => DistanceTransforms.DistanceTransform(src, DistanceTypes.User, DistanceTransformMasks.Mask5));

        Assert.Throws<ArgumentException>(() => DistanceTransforms.DistanceTransform(src, new Mat<byte>(5, 4), DistanceTypes.L2, DistanceTransformMasks.Mask3));
        Assert.Throws<ArgumentException>(() => DistanceTransforms.DistanceTransform(src, new Mat<byte>(4, 4), DistanceTypes.L1, DistanceTransformMasks.Mask3));

        Assert.Throws<ArgumentOutOfRangeException>(() => DistanceTransforms.DistanceTransformWithLabels(src, DistanceTypes.L2, DistanceTransformMasks.Mask5, (DistanceTransformLabelTypes)2));
        Assert.Throws<ArgumentOutOfRangeException>(() => DistanceTransforms.DistanceTransformWithLabels(src, DistanceTypes.L12, DistanceTransformMasks.Mask5));
        Assert.Throws<ArgumentException>(() => DistanceTransforms.DistanceTransformWithLabels(src, new Mat<float>(5, 4), new Mat<int>(4, 4), DistanceTypes.L2, DistanceTransformMasks.Mask5));
        Assert.Throws<ArgumentNullException>(() => DistanceTransforms.DistanceTransformWithLabels(src, new Mat<float>(5, 4), null!, DistanceTypes.L2, DistanceTransformMasks.Mask5));
    }

    /// <summary>OpenCV test <c>wide_image_22732</c>.</summary>
    [Fact]
    public void DistanceTransform_PreciseWideAllZero_GivesZero()
    {
        Mat<float> d = DistanceTransforms.DistanceTransform(new Mat<byte>(4099, 1), DistanceTypes.L2, DistanceTransformMasks.Precise);
        Assert.All(d.ToArray(), v => Assert.Equal(0f, v));
    }

    /// <summary>OpenCV tests <c>ipp_deterministic_corner</c> and <c>ipp_deterministic</c>.</summary>
    [Fact]
    public void DistanceTransform_PreciseRow_GivesExactPixelCounts()
    {
        var src = new Mat<byte>(4096, 1);
        src.Fill(255);
        src.At(0, 0) = 0;
        Mat<float> d = DistanceTransforms.DistanceTransform(src, DistanceTypes.L2, DistanceTransformMasks.Precise);
        for (int i = 0; i < 4096; i++)
        {
            Assert.Equal(i, d.At(0, i));
        }

        var rng = new Random(11);
        for (int trial = 0; trial < 20; trial++)
        {
            var row = new Mat<byte>(800, 1);
            row.Fill(255);
            int p1 = rng.Next(800), p2 = rng.Next(800), p3 = rng.Next(800);
            row.At(0, p1) = row.At(0, p2) = row.At(0, p3) = 0;
            Mat<float> r = DistanceTransforms.DistanceTransform(row, DistanceTypes.L2, DistanceTransformMasks.Precise);
            for (int i = 0; i < 800; i++)
            {
                Assert.Equal(Math.Min(Math.Min(Math.Abs(i - p1), Math.Abs(i - p2)), Math.Abs(i - p3)), r.At(0, i));
            }
        }
    }

    [Fact]
    public void Watershed_InvalidArguments_Throw()
    {
        var image = new Mat<byte>(6, 5, 3);
        var markers = new Mat<int>(6, 5);
        Assert.Throws<ArgumentNullException>(() => Segmentation.Watershed(null!, markers));
        Assert.Throws<ArgumentNullException>(() => Segmentation.Watershed(image, null!));
        Assert.Throws<ArgumentException>(() => Segmentation.Watershed(new Mat<byte>(6, 5), markers));
        Assert.Throws<ArgumentException>(() => Segmentation.Watershed(image, new Mat<int>(6, 5, 2)));
        Assert.Throws<ArgumentException>(() => Segmentation.Watershed(image, new Mat<int>(5, 5)));
    }

    [Fact]
    public void Watershed_NoMarkers_SetsFrameAndClearsNegativeValues()
    {
        var image = new Mat<byte>(6, 5, 3);
        var markers = new Mat<int>(6, 5);
        markers.Fill(-4);
        Segmentation.Watershed(image, markers);
        for (int y = 0; y < 5; y++)
        {
            for (int x = 0; x < 6; x++)
            {
                bool frame = y == 0 || y == 4 || x == 0 || x == 5;
                Assert.Equal(frame ? -1 : 0, markers.At(y, x));
            }
        }
    }

    /// <summary>
    /// A pixel takes a label only when all its labelled 4-neighbours agree, so different positive
    /// labels are never 4-adjacent, and every region contains its seed.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    public void Watershed_RandomSeeds_SeparatesRegionsAndKeepsImage(int seed)
    {
        var rng = new Random(seed);
        int w = 61, h = 47;
        var image = new Mat<byte>(w, h, 3);
        rng.NextBytes(image.GetSpan());
        byte[] imageBefore = image.ToArray();
        var markers = new Mat<int>(w, h);
        var seeds = new List<(int X, int Y, int Label)>();
        for (int k = 1; k <= 12; k++)
        {
            int x = rng.Next(1, w - 1), y = rng.Next(1, h - 1);
            if (markers.At(y, x) == 0)
            {
                markers.At(y, x) = k * 1000;
                seeds.Add((x, y, k * 1000));
            }
        }

        Segmentation.Watershed(image, markers);

        Assert.Equal(imageBefore, image.ToArray());
        foreach ((int x, int y, int label) in seeds)
        {
            Assert.Equal(label, markers.At(y, x));
        }

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int v = markers.At(y, x);
                Assert.True(v == -1 || (v > 0 && v % 1000 == 0), $"unexpected value {v}");
                if (y == 0 || x == 0 || y == h - 1 || x == w - 1)
                {
                    Assert.Equal(-1, v);
                }

                if (v > 0 && x + 1 < w)
                {
                    Assert.True(markers.At(y, x + 1) <= 0 || markers.At(y, x + 1) == v);
                }

                if (v > 0 && y + 1 < h)
                {
                    Assert.True(markers.At(y + 1, x) <= 0 || markers.At(y + 1, x) == v);
                }
            }
        }
    }

    [Fact]
    public void Watershed_IsDeterministic()
    {
        var rng = new Random(4);
        var image = new Mat<byte>(80, 60, 3);
        rng.NextBytes(image.GetSpan());
        var markers = new Mat<int>(80, 60);
        for (int k = 1; k <= 30; k++)
        {
            markers.At(rng.Next(60), rng.Next(80)) = k;
        }

        Mat<int> a = markers.Clone();
        Mat<int> b = markers.Clone();
        Segmentation.Watershed(image, a);
        Segmentation.Watershed(image, b);
        Assert.Equal(a.ToArray(), b.ToArray());
    }
}
