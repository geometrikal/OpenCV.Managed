namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

public sealed class FloodFillBasicTests
{
    [Fact]
    public void FloodFill_HoleFillFromCorner_LeavesOnlyHolesUnfilled()
    {
        var img = new Mat<byte>(7, 6);
        for (int y = 1; y <= 4; y++)
        {
            for (int x = 1; x <= 4; x++)
            {
                img.At(y, x) = 255;
            }
        }

        img.At(2, 2) = 0;
        img.At(3, 3) = 0;

        FloodFillResult r = FloodFilling.FloodFill(img, new Point(0, 0), new Scalar(255));

        Assert.Equal((7 * 6) - 16, r.Area);
        Assert.Equal(new Rect(0, 0, 7, 6), r.Rect);
        Assert.Equal(2, img.ToArray().Count(v => v == 0));
        Assert.Equal(0, img.At(2, 2));
        Assert.Equal(0, img.At(3, 3));
    }

    [Fact]
    public void FloodFill_FourVersusEightConnectivity_DiagonalStepDiffers()
    {
        byte[] data = [1, 0, 0, 0, 1, 0, 0, 0, 1];
        var a = new Mat<byte>((byte[])data.Clone(), 3, 3);
        var b = new Mat<byte>((byte[])data.Clone(), 3, 3);

        Assert.Equal(1, FloodFilling.FloodFill(a, new Point(0, 0), new Scalar(9)).Area);
        Assert.Equal(3, FloodFilling.FloodFill(b, new Point(0, 0), new Scalar(9), flags: FloodFillFlags.Connectivity8).Area);
        Assert.Equal(new FloodFillResult(3, new Rect(1, 0, 2, 2)), FloodFilling.FloodFill(new Mat<byte>((byte[])data.Clone(), 3, 3), new Point(1, 0), new Scalar(9), flags: 0));
    }

    [Fact]
    public void FloodFill_SeedEqualsNewValue_ReportsComponentAndKeepsImage()
    {
        var img = new Mat<byte>(5, 4);
        img.Fill(9);
        img.At(1, 1) = 3;

        FloodFillResult r = FloodFilling.FloodFill(img, new Point(0, 0), new Scalar(9));

        Assert.Equal(19, r.Area);
        Assert.Equal(new Rect(0, 0, 5, 4), r.Rect);
        Assert.Equal(3, img.At(1, 1));
        Assert.Equal(19, img.ToArray().Count(v => v == 9));
    }

    [Fact]
    public void FloodFill_SeedSetInMask_ReturnsEmptyResultAndKeepsImage()
    {
        var img = new Mat<byte>(4, 4);
        var mask = new Mat<byte>(6, 6);
        mask.At(2, 3) = 1;

        FloodFillResult r = FloodFilling.FloodFill(img, mask, new Point(2, 1), new Scalar(7));

        Assert.Equal(default, r);
        Assert.All(img.ToArray(), v => Assert.Equal(0, v));
    }

    [Theory]
    [InlineData(0, 1)]
    [InlineData(1, 1)]
    [InlineData(200, 200)]
    public void FloodFill_MaskFillValue_WritesValueAndSetsFrame(byte value, byte expected)
    {
        var img = new Mat<byte>(3, 2);
        var mask = new Mat<byte>(5, 4);
        mask.Fill(33);
        mask.Roi(1, 1, 3, 2).Fill(0);

        FloodFilling.FloodFill(img, mask, new Point(0, 0), new Scalar(5), flags: FloodFillFlags.Connectivity4 | FloodFilling.MaskFillValue(value) | FloodFillFlags.MaskOnly);

        Assert.All(img.ToArray(), v => Assert.Equal(0, v));
        Assert.All(mask.Roi(1, 1, 3, 2).ToArray(), v => Assert.Equal(expected, v));
        Assert.All(mask.GetRowSpan(0).ToArray(), v => Assert.Equal(1, v));
        Assert.All(mask.GetRowSpan(3).ToArray(), v => Assert.Equal(1, v));
        Assert.Equal([1, 1, 1, 1], new[] { mask.At(1, 0), mask.At(2, 0), mask.At(1, 4), mask.At(2, 4) });
    }

    [Fact]
    public void FloodFill_MaskOnlyWithoutMask_ReturnsAreaAndKeepsImage()
    {
        var img = new Mat<byte>(6, 3);
        img.At(1, 2) = 4;

        FloodFillResult r = FloodFilling.FloodFill(img, new Point(0, 0), new Scalar(8), flags: FloodFillFlags.MaskOnly | FloodFillFlags.Connectivity4);

        Assert.Equal(17, r.Area);
        Assert.Equal(17, img.ToArray().Count(v => v == 0));
    }

    [Theory]
    [InlineData(4, 1)]
    [InlineData(8, 1)]
    [InlineData(4, 3)]
    [InlineData(8, 3)]
    public void FloodFill_EqualityFastPath_MatchesMaskedGradientFill(int connectivity, int channels)
    {
        // The no-mask overload uses the equality fill; the mask overload always runs the gradient fill.
        for (int seed = 0; seed < 8; seed++)
        {
            Mat<byte> src = RandomLevels(37, 29, channels, seed, 3);
            var p = new Point((seed * 11) % 37, (seed * 7) % 29);
            Mat<byte> a = src.Clone();
            Mat<byte> b = src.Clone();
            var nv = new Scalar(250, 251, 252);

            FloodFillResult ra = FloodFilling.FloodFill(a, p, nv, flags: (FloodFillFlags)connectivity);
            FloodFillResult rb = FloodFilling.FloodFill(b, new Mat<byte>(39, 31), p, nv, flags: (FloodFillFlags)connectivity);

            Assert.Equal(rb, ra);
            Assert.Equal(b.ToArray(), a.ToArray());
        }
    }

    [Theory]
    [InlineData(4)]
    [InlineData(8)]
    public void FloodFill_EqualityFastPathInt_MatchesMaskedGradientFill(int connectivity)
    {
        Mat<byte> levels = RandomLevels(41, 23, 1, 5, 2);
        var src = new Mat<int>(41, 23);
        for (int y = 0; y < 23; y++)
        {
            for (int x = 0; x < 41; x++)
            {
                src.At(y, x) = levels.At(y, x) == 0 ? int.MinValue : int.MaxValue;
            }
        }

        Mat<int> a = src.Clone();
        Mat<int> b = src.Clone();
        FloodFillResult ra = FloodFilling.FloodFill(a, new Point(3, 4), new Scalar(-7), flags: (FloodFillFlags)connectivity);
        FloodFillResult rb = FloodFilling.FloodFill(b, new Mat<byte>(43, 25), new Point(3, 4), new Scalar(-7), flags: (FloodFillFlags)connectivity);

        Assert.Equal(rb, ra);
        Assert.Equal(b.ToArray(), a.ToArray());
    }

    [Fact]
    public void FloodFill_SerpentineMaze_FillsCorridorsOnBothPaths()
    {
        // Corridors of width 1 with alternating openings give one segment per row turn.
        const int size = 601;
        var src = new Mat<byte>(size, size);
        for (int y = 1; y < size; y += 2)
        {
            src.GetRowSpan(y).Fill(255);
            src.At(y, (y / 2) % 2 == 0 ? size - 1 : 0) = 0;
        }

        Mat<byte> a = src.Clone();
        Mat<byte> b = src.Clone();
        FloodFillResult ra = FloodFilling.FloodFill(a, new Point(0, 0), new Scalar(9));
        FloodFillResult rb = FloodFilling.FloodFill(b, new Mat<byte>(size + 2, size + 2), new Point(0, 0), new Scalar(9), new Scalar(1), new Scalar(1));

        Assert.Equal((301 * size) + 300, ra.Area);
        Assert.Equal(ra, rb);
        Assert.Equal(a.ToArray(), b.ToArray());
    }

    [Fact]
    public void FloodFill_GradientFloatingVersusFixedRange_Differ()
    {
        float[] data = [0, 1, 2, 3, 4, 5, 6, 7];
        var a = new Mat<float>((float[])data.Clone(), 8, 1);
        var b = new Mat<float>((float[])data.Clone(), 8, 1);

        Assert.Equal(8, FloodFilling.FloodFill(a, new Point(0, 0), new Scalar(-1), new Scalar(1), new Scalar(1)).Area);
        Assert.Equal(3, FloodFilling.FloodFill(b, new Point(0, 0), new Scalar(-1), new Scalar(2), new Scalar(2), FloodFillFlags.FixedRange).Area);
    }

    [Fact]
    public void FloodFill_RoiWithStride_MatchesPackedImage()
    {
        Mat<byte> packed = RandomLevels(17, 13, 3, 9, 4);
        var parent = new Mat<byte>(25, 20, 3);
        parent.Fill(packed.At(6, 5, 0));
        Mat<byte> roi = parent.Roi(4, 3, 17, 13);
        packed.CopyTo(roi);
        var lo = new Scalar(1, 1, 1);

        FloodFillResult rp = FloodFilling.FloodFill(packed, new Point(5, 6), new Scalar(255, 0, 0), lo, lo, FloodFillFlags.Connectivity8);
        FloodFillResult rr = FloodFilling.FloodFill(roi, new Point(5, 6), new Scalar(255, 0, 0), lo, lo, FloodFillFlags.Connectivity8);

        Assert.Equal(rp, rr);
        Assert.Equal(packed.ToArray(), roi.ToArray());
    }

    [Fact]
    public void FloodFill_RepeatedCalls_AreDeterministic()
    {
        Mat<byte> src = RandomLevels(50, 40, 1, 2, 6);
        Mat<byte> a = src.Clone();
        Mat<byte> b = src.Clone();

        FloodFillResult ra = FloodFilling.FloodFill(a, new Point(25, 20), new Scalar(255), new Scalar(2), new Scalar(2));
        FloodFillResult rb = FloodFilling.FloodFill(b, new Point(25, 20), new Scalar(255), new Scalar(2), new Scalar(2));

        Assert.Equal(ra, rb);
        Assert.Equal(a.ToArray(), b.ToArray());
    }

    [Fact]
    public void FloodFill_SinglePixel_FillsSeed()
    {
        var img = new Mat<float>(1, 1);
        FloodFillResult r = FloodFilling.FloodFill(img, new Point(0, 0), new Scalar(2.5));

        Assert.Equal(new FloodFillResult(1, new Rect(0, 0, 1, 1)), r);
        Assert.Equal(2.5f, img.At(0, 0));
    }

    [Fact]
    public void FloodFill_InvalidArguments_Throw()
    {
        var img = new Mat<byte>(4, 3);
        var p = new Point(0, 0);
        var s = new Scalar(1);

        Assert.Throws<ArgumentNullException>(() => FloodFilling.FloodFill<byte>(null!, p, s));
        Assert.Throws<ArgumentNullException>(() => FloodFilling.FloodFill(img, null!, p, s));
        Assert.Throws<ArgumentException>(() => FloodFilling.FloodFill(new Mat<byte>(4, 3, 2), p, s));
        Assert.Throws<NotSupportedException>(() => FloodFilling.FloodFill(new Mat<ushort>(4, 3), p, s));
        Assert.Throws<NotSupportedException>(() => FloodFilling.FloodFill(new Mat<double>(4, 3), p, s));
        Assert.Throws<ArgumentException>(() => FloodFilling.FloodFill(img, p, s, flags: (FloodFillFlags)6));
        Assert.Throws<ArgumentException>(() => FloodFilling.FloodFill(img, p, s, new Scalar(-1)));
        Assert.Throws<ArgumentException>(() => FloodFilling.FloodFill(img, p, s, default, new Scalar(-1)));
        Assert.Throws<ArgumentException>(() => FloodFilling.FloodFill(img, new Mat<byte>(5, 5), p, s));
        Assert.Throws<ArgumentException>(() => FloodFilling.FloodFill(img, new Mat<byte>(6, 5, 2), p, s));
        Assert.Throws<ArgumentOutOfRangeException>(() => FloodFilling.FloodFill(img, new Point(4, 0), s));
        Assert.Throws<ArgumentOutOfRangeException>(() => FloodFilling.FloodFill(img, new Point(0, -1), s));
        Assert.Throws<ArgumentException>(() => FloodFilling.FloodFill(new Mat<byte>(65536, 1), p, s));
    }

    [Fact]
    public void FloodFill_NegativeDifferenceOnUnusedChannel_IsIgnored()
    {
        var img = new Mat<byte>(3, 3);
        FloodFillResult r = FloodFilling.FloodFill(img, new Point(1, 1), new Scalar(4), new Scalar(0, -5), new Scalar(0, -5));

        Assert.Equal(9, r.Area);
    }

    [Fact]
    public void SegmentStack_PushBeyondCapacity_PopsInReverseOrder()
    {
        var stack = new Geometrikal.OpenCV.Managed.ImgProc.Internal.FloodSegmentStack(2);
        try
        {
            for (int k = 0; k < 5000; k++)
            {
                stack.Push(k % 65536, k % 7, k % 11, k % 13, k % 17, (k & 1) == 0 ? 1 : -1);
            }

            for (int k = 4999; k >= 0; k--)
            {
                stack.Pop(out int y, out int l, out int r, out int pl, out int pr, out int dir);
                Assert.Equal((k, k % 7, k % 11, k % 13, k % 17, (k & 1) == 0 ? 1 : -1), (y, l, r, pl, pr, dir));
            }

            Assert.True(stack.IsEmpty);
        }
        finally
        {
            stack.Dispose();
        }
    }

    private static Mat<byte> RandomLevels(int width, int height, int channels, int seed, int levels)
    {
        var rng = new Random(seed);
        var m = new Mat<byte>(width, height, channels);
        for (int y = 0; y < height; y++)
        {
            Span<byte> row = m.GetRowSpan(y);
            for (int i = 0; i < row.Length; i++)
            {
                row[i] = (byte)(rng.Next(levels) * 10);
            }
        }

        return m;
    }
}
