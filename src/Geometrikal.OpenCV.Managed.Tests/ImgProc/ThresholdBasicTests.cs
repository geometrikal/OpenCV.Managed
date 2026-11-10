namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Shape, aliasing and validation tests, and analytic tests for the OpenCV 5.0 features that
/// cv2 4.5.5 cannot generate (<c>thresholdWithMask</c>, <c>THRESH_DRYRUN</c>).
/// </summary>
public sealed class ThresholdBasicTests
{
    private static readonly ThresholdType[] Rules =
    [
        ThresholdType.Binary, ThresholdType.BinaryInv, ThresholdType.Trunc, ThresholdType.ToZero, ThresholdType.ToZeroInv,
    ];

    private static Mat<T> Random<T>(int w, int h, int cn, int seed, Func<Random, T> next)
        where T : unmanaged
    {
        var rng = new Random(seed);
        var m = new Mat<T>(w, h, cn);
        for (int y = 0; y < h; y++)
        {
            Span<T> row = m.GetRowSpan(y);
            for (int i = 0; i < row.Length; i++)
            {
                row[i] = next(rng);
            }
        }

        return m;
    }

    private static Mat<byte> Bytes(int w, int h, int cn, int seed) => Random(w, h, cn, seed, r => (byte)r.Next(256));

    private static Mat<float> Floats(int w, int h, int cn, int seed) => Random(w, h, cn, seed, r => (float)((r.NextDouble() * 400) - 200));

    private static Mat<byte> Mask(int w, int h, int seed) => Random(w, h, 1, seed, r => (byte)(r.Next(3) == 0 ? 0 : r.Next(1, 256)));

    private static Mat<T> InRoi<T>(Mat<T> packed)
        where T : unmanaged
    {
        var parent = new Mat<T>(packed.Width + 7, packed.Height + 5, packed.Channels);
        Mat<T> roi = parent.Roi(3, 2, packed.Width, packed.Height);
        packed.CopyTo(roi);
        return roi;
    }

    /// <summary><c>mask ? full : prior</c> per pixel.</summary>
    private static T[] Compose<T>(Mat<T> full, Mat<T> prior, Mat<byte> mask)
        where T : unmanaged
    {
        T[] f = full.ToArray(), p = prior.ToArray();
        byte[] m = mask.ToArray();
        int cn = full.Channels;
        return f.Select((v, i) => m[i / cn] != 0 ? v : p[i]).ToArray();
    }

    [Fact]
    public void Threshold_RoiWithStride_MatchesPackedImage()
    {
        Mat<byte> b = Bytes(45, 11, 3, 1);
        Mat<float> f = Floats(45, 11, 3, 2);
        Mat<ushort> u = Random(45, 11, 1, 3, r => (ushort)r.Next(65536));
        Mat<byte> gray = Bytes(45, 11, 1, 4);
        foreach (ThresholdType t in Rules)
        {
            Assert.Equal(Thresholding.Threshold(b, 100.5, 200, t).ToArray(), Thresholding.Threshold(InRoi(b), 100.5, 200, t).ToArray());
            Assert.Equal(Thresholding.Threshold(f, -3.25, 7, t).ToArray(), Thresholding.Threshold(InRoi(f), -3.25, 7, t).ToArray());
            Assert.Equal(
                Thresholding.Threshold(u, 0, 999, t | ThresholdType.Otsu).ToArray(),
                Thresholding.Threshold(InRoi(u), 0, 999, t | ThresholdType.Otsu).ToArray());
            Assert.Equal(
                Thresholding.Threshold(gray, 0, 255, t | ThresholdType.Triangle).ToArray(),
                Thresholding.Threshold(InRoi(gray), 0, 255, t | ThresholdType.Triangle).ToArray());
        }
    }

    [Fact]
    public void Threshold_MaskedRoiWithStride_MatchesPackedImage()
    {
        Mat<float> f = Floats(45, 11, 3, 5);
        Mat<byte> mask = Mask(45, 11, 6);
        Mat<float> prior = Floats(45, 11, 3, 7);
        Mat<float> packed = prior.Clone(), roi = InRoi(prior);
        double a = Thresholding.Threshold(f, packed, 10, 1, ThresholdType.ToZero, mask);
        double b = Thresholding.Threshold(InRoi(f), roi, 10, 1, ThresholdType.ToZero, InRoi(mask));
        Assert.Equal(a, b);
        Assert.Equal(packed.ToArray(), roi.ToArray());
    }

    [Fact]
    public void AdaptiveThreshold_RoiWithStride_MatchesPackedImage()
    {
        Mat<byte> src = Bytes(53, 19, 1, 8);
        foreach (AdaptiveThresholdType method in new[] { AdaptiveThresholdType.MeanC, AdaptiveThresholdType.GaussianC })
        {
            foreach (ThresholdType t in new[] { ThresholdType.Binary, ThresholdType.BinaryInv })
            {
                Assert.Equal(
                    Thresholding.AdaptiveThreshold(src, 255, method, t, 7, 2.5).ToArray(),
                    Thresholding.AdaptiveThreshold(InRoi(src), 255, method, t, 7, 2.5).ToArray());
            }
        }
    }

    [Fact]
    public void Threshold_DoesNotModifySource()
    {
        Mat<byte> b = Bytes(37, 5, 3, 9);
        byte[] before = b.ToArray();
        foreach (ThresholdType t in Rules)
        {
            Thresholding.Threshold(b, 77, 255, t);
        }

        Thresholding.AdaptiveThreshold(Bytes(37, 5, 1, 9), 255, AdaptiveThresholdType.GaussianC, ThresholdType.Binary, 5, 1);
        Assert.Equal(before, b.ToArray());
    }

    /// <summary>The binary rule is not idempotent, so a row that reads already written elements gives a different result.</summary>
    [Fact]
    public void Threshold_InPlace_MatchesOutOfPlace()
    {
        foreach (int width in new[] { 1, 5, 37, 67, 130 })
        {
            Mat<byte> b = Random(width, 3, 1, width, r => (byte)r.Next(150, 256));
            Mat<float> f = Floats(width, 3, 1, width + 1);
            foreach (ThresholdType t in Rules)
            {
                byte[] expectedB = Thresholding.Threshold(b, 100, 50, t).ToArray();
                Mat<byte> inPlaceB = b.Clone();
                Thresholding.Threshold(inPlaceB, inPlaceB, 100, 50, t);
                Assert.Equal(expectedB, inPlaceB.ToArray());

                float[] expectedF = Thresholding.Threshold(f, 50, -60, t).ToArray();
                Mat<float> inPlaceF = f.Clone();
                Thresholding.Threshold(inPlaceF, inPlaceF, 50, -60, t);
                Assert.Equal(expectedF, inPlaceF.ToArray());
            }
        }
    }

    [Fact]
    public void Threshold_SinglePixel_AppliesRule()
    {
        var p = new Mat<byte>(1, 1);
        p.At(0, 0) = 120;
        Assert.Equal(200, Thresholding.Threshold(p, 119.9, 200, ThresholdType.Binary).At(0, 0));
        Assert.Equal(0, Thresholding.Threshold(p, 120, 200, ThresholdType.Binary).At(0, 0));
        Assert.Equal(119, Thresholding.Threshold(p, 119.9, 200, ThresholdType.Trunc).At(0, 0));
        Assert.Equal(0, Thresholding.Threshold(p, 120, 200, ThresholdType.ToZero).At(0, 0));
        Assert.Equal(120, Thresholding.Threshold(p, 120, 200, ThresholdType.ToZeroInv).At(0, 0));
        Assert.Equal(1, Thresholding.AdaptiveThreshold(p, 1, AdaptiveThresholdType.MeanC, ThresholdType.Binary, 3, 0.5).At(0, 0));
        Assert.Equal(1, Thresholding.AdaptiveThreshold(p, 1, AdaptiveThresholdType.MeanC, ThresholdType.BinaryInv, 3, -0.5).At(0, 0));
    }

    [Fact]
    public void Threshold_RepeatedCalls_AreDeterministic()
    {
        Mat<byte> b = Bytes(131, 17, 1, 10);
        Assert.Equal(
            Thresholding.Threshold(b, 0, 255, ThresholdType.Binary | ThresholdType.Otsu).ToArray(),
            Thresholding.Threshold(b, 0, 255, ThresholdType.Binary | ThresholdType.Otsu).ToArray());
        Assert.Equal(
            Thresholding.AdaptiveThreshold(b, 255, AdaptiveThresholdType.GaussianC, ThresholdType.BinaryInv, 11, 3).ToArray(),
            Thresholding.AdaptiveThreshold(b, 255, AdaptiveThresholdType.GaussianC, ThresholdType.BinaryInv, 11, 3).ToArray());
    }

    [Fact]
    public void Threshold_AllZeroImage_FollowsRules()
    {
        var z = new Mat<byte>(9, 4);
        Assert.All(Thresholding.Threshold(z, 0, 255, ThresholdType.Binary).ToArray(), v => Assert.Equal(0, v));
        Assert.All(Thresholding.Threshold(z, 0, 255, ThresholdType.BinaryInv).ToArray(), v => Assert.Equal(255, v));
        Thresholding.Threshold(z, 0, 255, ThresholdType.Binary | ThresholdType.Triangle, out double triangle);
        Assert.Equal(1, triangle);
    }

    [Fact]
    public void Threshold_InvalidInputs_Throw()
    {
        Mat<byte> b = Bytes(8, 4, 1, 11);
        Mat<byte> b3 = Bytes(8, 4, 3, 11);
        Assert.Throws<ArgumentNullException>(() => Thresholding.Threshold<byte>(null!, 1, 2, ThresholdType.Binary));
        Assert.Throws<ArgumentNullException>(() => Thresholding.Threshold(b, null, 1, 2, ThresholdType.Binary));
        Assert.Throws<ArgumentException>(() => Thresholding.Threshold(b, new Mat<byte>(8, 5), 1, 2, ThresholdType.Binary));
        Assert.Throws<ArgumentException>(() => Thresholding.Threshold(b, 1, 2, (ThresholdType)5));
        Assert.Throws<ArgumentException>(() => Thresholding.Threshold(b, 1, 2, (ThresholdType)7));
        Assert.Throws<ArgumentException>(() => Thresholding.Threshold(b, 1, 2, (ThresholdType)32));
        Assert.Throws<ArgumentException>(() => Thresholding.Threshold(b, 1, 2, ThresholdType.Otsu | ThresholdType.Triangle));
        Assert.Throws<ArgumentException>(() => Thresholding.Threshold(b, 1, 2, ThresholdType.DryRun));
        Assert.Throws<ArgumentException>(() => Thresholding.Threshold(b3, 1, 2, ThresholdType.Otsu));
        Assert.Throws<ArgumentException>(() => Thresholding.Threshold(b3, 1, 2, ThresholdType.Triangle));
        Assert.Throws<ArgumentException>(() => Thresholding.Threshold(new Mat<short>(3, 3), 1, 2, ThresholdType.Otsu));
        Assert.Throws<ArgumentException>(() => Thresholding.Threshold(new Mat<float>(3, 3), 1, 2, ThresholdType.Otsu));
        Assert.Throws<ArgumentException>(() => Thresholding.Threshold(new Mat<ushort>(3, 3), 1, 2, ThresholdType.Triangle));
        Assert.Throws<NotSupportedException>(() => Thresholding.Threshold(new Mat<int>(3, 3), 1, 2, ThresholdType.Binary));
        Assert.Throws<NotSupportedException>(() => Thresholding.Threshold(new Mat<sbyte>(3, 3), 1, 2, ThresholdType.Binary));
        Assert.Throws<ArgumentException>(() => Thresholding.Threshold(b, b.Clone(), 1, 2, ThresholdType.Binary, new Mat<byte>(8, 3)));
        Assert.Throws<ArgumentException>(() => Thresholding.Threshold(b, b.Clone(), 1, 2, ThresholdType.Binary, new Mat<byte>(8, 4, 3)));
    }

    [Fact]
    public void AdaptiveThreshold_InvalidInputs_Throw()
    {
        Mat<byte> b = Bytes(8, 4, 1, 12);
        Assert.Throws<ArgumentNullException>(() => Thresholding.AdaptiveThreshold(null!, 255, AdaptiveThresholdType.MeanC, ThresholdType.Binary, 3, 0));
        Assert.Throws<ArgumentException>(() => Thresholding.AdaptiveThreshold(Bytes(8, 4, 3, 12), 255, AdaptiveThresholdType.MeanC, ThresholdType.Binary, 3, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Thresholding.AdaptiveThreshold(b, 255, AdaptiveThresholdType.MeanC, ThresholdType.Binary, 1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Thresholding.AdaptiveThreshold(b, 255, AdaptiveThresholdType.MeanC, ThresholdType.Binary, 4, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => Thresholding.AdaptiveThreshold(b, 255, AdaptiveThresholdType.MeanC, ThresholdType.Binary, -3, 0));
        Assert.Throws<ArgumentException>(() => Thresholding.AdaptiveThreshold(b, 255, (AdaptiveThresholdType)2, ThresholdType.Binary, 3, 0));
        Assert.Throws<ArgumentException>(() => Thresholding.AdaptiveThreshold(b, 255, AdaptiveThresholdType.MeanC, ThresholdType.Trunc, 3, 0));
        Assert.Throws<ArgumentException>(() => Thresholding.AdaptiveThreshold(b, 255, AdaptiveThresholdType.MeanC, ThresholdType.Binary | ThresholdType.Otsu, 3, 0));
    }

    /// <summary>OpenCV returns zeros for a negative maximum value before it checks the method and type.</summary>
    [Fact]
    public void AdaptiveThreshold_NegativeMaxValue_ReturnsZerosBeforeModeChecks()
    {
        Mat<byte> b = Bytes(8, 4, 1, 13);
        Assert.All(
            Thresholding.AdaptiveThreshold(b, -1, (AdaptiveThresholdType)5, ThresholdType.Trunc, 3, 0).ToArray(),
            v => Assert.Equal(0, v));
    }

    [Fact]
    public void Threshold_WithMask_InRange_MatchesComposedThreshold()
    {
        foreach (int cn in new[] { 1, 3 })
        {
            Mat<byte> src = Bytes(37, 7, cn, 14 + cn);
            Mat<short> s16 = Random(37, 7, cn, 15, r => (short)r.Next(-1000, 1000));
            Mat<double> f64 = Random(37, 7, cn, 16, r => r.NextDouble() - 0.5);
            Mat<byte> mask = Mask(37, 7, 17);
            foreach (ThresholdType t in Rules)
            {
                AssertMasked(src, Bytes(37, 7, cn, 18), mask, 127.5, 200, t);
                AssertMasked(s16, Random(37, 7, cn, 19, r => (short)r.Next()), mask, -10.5, 77, t);
                AssertMasked(f64, Random(37, 7, cn, 20, r => r.NextDouble()), mask, 0.125, 9, t);
            }
        }
    }

    private static void AssertMasked<T>(Mat<T> src, Mat<T> prior, Mat<byte> mask, double thresh, double maxValue, ThresholdType t)
        where T : unmanaged
    {
        Mat<T> full = Thresholding.Threshold(src, thresh, maxValue, t, out double expectedThreshold);
        Mat<T> dst = prior.Clone();
        double used = Thresholding.Threshold(src, dst, thresh, maxValue, t, mask);
        Assert.Equal(expectedThreshold, used);
        Assert.Equal(Compose(full, prior, mask), dst.ToArray());
    }

    /// <summary>
    /// Deliberate difference: OpenCV 5.0 fills or copies the whole destination when the floored
    /// threshold is outside the element range, and ignores the mask. The port applies the mask.
    /// </summary>
    [Theory]
    [InlineData(ThresholdType.Binary, 300.0)]
    [InlineData(ThresholdType.Binary, -1.0)]
    [InlineData(ThresholdType.BinaryInv, double.NegativeInfinity)]
    [InlineData(ThresholdType.Trunc, 255.0)]
    [InlineData(ThresholdType.Trunc, -5.0)]
    [InlineData(ThresholdType.ToZero, 1e12)]
    [InlineData(ThresholdType.ToZero, -0.5)]
    [InlineData(ThresholdType.ToZeroInv, 256.0)]
    [InlineData(ThresholdType.ToZeroInv, double.NaN)]
    public void Threshold_WithMask_OutOfRangeShortcut_KeepsUnmaskedPixels(ThresholdType t, double thresh)
    {
        Mat<byte> src = Bytes(37, 5, 3, 21);
        Mat<byte> prior = Bytes(37, 5, 3, 22);
        Mat<byte> mask = Mask(37, 5, 23);
        Mat<byte> full = Thresholding.Threshold(src, thresh, 99, t);
        Mat<byte> dst = prior.Clone();
        Thresholding.Threshold(src, dst, thresh, 99, t, mask);
        Assert.Equal(Compose(full, prior, mask), dst.ToArray());
        Assert.Contains(mask.ToArray(), m => m == 0);
    }

    /// <summary>
    /// The OpenCV 5.0 masked kernel computes TRUNC with <c>std::min(src, thresh)</c>; the
    /// unmasked vector kernel uses <c>v_min</c>. They differ for NaN and signed zero.
    /// </summary>
    [Fact]
    public void Threshold_WithMask_FloatTrunc_UsesStdMin()
    {
        float[] pattern = [float.NaN, -0.0f, 5f, -5f];
        foreach (int width in new[] { 4, 40 })
        {
            float[] values = Enumerable.Range(0, width).Select(i => pattern[i % 4]).ToArray();
            var src = new Mat<float>(values, width, 1);
            var mask = new Mat<byte>(width, 1);
            mask.Fill(1);
            Mat<float> masked = new Mat<float>(width, 1);
            Thresholding.Threshold(src, masked, 0.0, 1, ThresholdType.Trunc, mask);
            Mat<float> unmasked = Thresholding.Threshold(src, 0.0, 1, ThresholdType.Trunc);
            for (int i = 0; i < width; i++)
            {
                float m = masked.At(0, i), u = unmasked.At(0, i);
                switch (i % 4)
                {
                    case 0:
                        Assert.True(float.IsNaN(m));
                        Assert.Equal(0, BitConverter.SingleToInt32Bits(u));
                        break;
                    case 1:
                        Assert.Equal(BitConverter.SingleToInt32Bits(-0.0f), BitConverter.SingleToInt32Bits(m));
                        Assert.Equal(0, BitConverter.SingleToInt32Bits(u));
                        break;
                    case 2:
                        Assert.Equal(0f, m);
                        Assert.Equal(0f, u);
                        break;
                    default:
                        Assert.Equal(-5f, m);
                        Assert.Equal(-5f, u);
                        break;
                }
            }
        }
    }

    /// <summary>Upstream test_thresh.cpp: Otsu and triangle with a rectangular mask equal the unmasked result of the crop.</summary>
    [Theory]
    [InlineData(ThresholdType.Otsu)]
    [InlineData(ThresholdType.Triangle)]
    public void Threshold_WithRectangleMask_UsesMaskedHistogram(ThresholdType automatic)
    {
        Mat<byte> src = Random(64, 48, 1, 24, r => (byte)(r.Next(4) == 0 ? r.Next(0, 60) : r.Next(120, 256)));
        Mat<byte> crop = src.Roi(10, 5, 30, 20);
        crop.Fill(0);
        for (int y = 0; y < 20; y++)
        {
            Span<byte> row = crop.GetRowSpan(y);
            for (int x = 0; x < 30; x++)
            {
                row[x] = (byte)((x * 37 + y * 11) % 90);
            }
        }

        var mask = new Mat<byte>(64, 48);
        mask.Roi(10, 5, 30, 20).Fill(255);

        Thresholding.Threshold(crop, 0, 255, ThresholdType.Binary | automatic, out double cropThreshold);
        Thresholding.Threshold(src, 0, 255, ThresholdType.Binary | automatic, out double fullThreshold);
        double masked = Thresholding.Threshold(src, src.Clone(), 0, 255, ThresholdType.Binary | automatic, mask);
        Assert.Equal(cropThreshold, masked);
        Assert.NotEqual(fullThreshold, masked);

        Mat<ushort> wide = Random(64, 48, 1, 25, r => (ushort)r.Next(65536));
        Mat<ushort> wideCrop = wide.Roi(10, 5, 30, 20);
        if (automatic == ThresholdType.Otsu)
        {
            Thresholding.Threshold(wideCrop, 0, 1, ThresholdType.Binary | automatic, out double expected16);
            Assert.Equal(expected16, Thresholding.Threshold(wide, null, 0, 1, ThresholdType.Binary | automatic | ThresholdType.DryRun, mask));
        }
    }

    [Fact]
    public void Threshold_WithEmptyMask_AutomaticThresholdsUseEmptyHistogram()
    {
        Mat<byte> src = Bytes(16, 8, 1, 26);
        var mask = new Mat<byte>(16, 8);
        Mat<byte> prior = Bytes(16, 8, 1, 27);
        Mat<byte> dst = prior.Clone();
        Assert.Equal(0, Thresholding.Threshold(src, dst, 0, 255, ThresholdType.Binary | ThresholdType.Otsu, mask));
        Assert.Equal(2, Thresholding.Threshold(src, dst, 0, 255, ThresholdType.Binary | ThresholdType.Triangle, mask));
        Assert.Equal(prior.ToArray(), dst.ToArray());
    }

    [Fact]
    public void Threshold_DryRun_ReturnsThresholdAndDoesNotWrite()
    {
        Mat<byte> src = Bytes(37, 5, 1, 28);
        Mat<byte> dst = Bytes(37, 5, 1, 29);
        byte[] before = dst.ToArray();
        Assert.Equal(127, Thresholding.Threshold(src, dst, 127.9, 255, ThresholdType.Binary | ThresholdType.DryRun));
        Assert.Equal(-1, Thresholding.Threshold(src, dst, -0.5, 255, ThresholdType.Binary | ThresholdType.DryRun));
        Assert.Equal(int.MinValue, Thresholding.Threshold(src, dst, double.NaN, 255, ThresholdType.Trunc | ThresholdType.DryRun));
        Assert.Equal(int.MaxValue, Thresholding.Threshold(src, dst, double.NegativeInfinity, 255, ThresholdType.Trunc | ThresholdType.DryRun));
        Assert.Equal(before, dst.ToArray());

        Thresholding.Threshold(src, 0, 255, ThresholdType.ToZero | ThresholdType.Otsu, out double otsu);
        Assert.Equal(otsu, Thresholding.Threshold(src, null, 0, 255, ThresholdType.ToZero | ThresholdType.Otsu | ThresholdType.DryRun));
        Assert.Equal(-32769, Thresholding.Threshold(new Mat<short>(3, 3), null, -32768.5, 1, ThresholdType.DryRun));
        Assert.Equal(0.3, Thresholding.Threshold(new Mat<float>(3, 3), null, 0.3, 1, ThresholdType.DryRun));
        Assert.Equal(-1.5, Thresholding.Threshold(new Mat<double>(3, 3), null, -1.5, 1, ThresholdType.Binary | ThresholdType.DryRun, new Mat<byte>(3, 3)));
    }
}
