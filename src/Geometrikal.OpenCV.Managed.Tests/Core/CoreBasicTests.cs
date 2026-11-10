namespace Geometrikal.OpenCV.Managed.Tests.Core;

using System;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;

public sealed class CoreBasicTests
{
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

    private static Mat<T> InRoi<T>(Mat<T> packed)
        where T : unmanaged
    {
        var parent = new Mat<T>(packed.Width + 7, packed.Height + 5, packed.Channels);
        Mat<T> roi = parent.Roi(3, 2, packed.Width, packed.Height);
        packed.CopyTo(roi);
        return roi;
    }

    [Fact]
    public void Arithmetic_RoiWithStride_MatchesPackedImage()
    {
        Mat<byte> a = Bytes(45, 11, 3, 1), b = Bytes(45, 11, 3, 2);
        Mat<byte> m = Random(45, 11, 1, 3, r => (byte)(r.Next(2) * 255));
        Mat<byte> ra = InRoi(a), rb = InRoi(b), rm = InRoi(m);

        Assert.Equal(Arithmetic.Add(a, b).ToArray(), Arithmetic.Add(ra, rb).ToArray());
        Assert.Equal(Arithmetic.Multiply(a, b, 0.37).ToArray(), Arithmetic.Multiply(ra, rb, 0.37).ToArray());
        Assert.Equal(Comparison.Compare(a, b, CmpType.Le).ToArray(), Comparison.Compare(ra, rb, CmpType.Le).ToArray());
        Assert.Equal(Comparison.InRange(a, new Scalar(10, 20, 30), new Scalar(200, 210, 220)).ToArray(),
            Comparison.InRange(ra, new Scalar(10, 20, 30), new Scalar(200, 210, 220)).ToArray());
        Assert.Equal(Conversions.ConvertTo<byte, float>(a, 0.5, 3).ToArray(), Conversions.ConvertTo<byte, float>(ra, 0.5, 3).ToArray());
        Assert.Equal(Reductions.Mean(a, m), Reductions.Mean(ra, rm));
        Assert.Equal(Reductions.Norm(a, b, NormType.L1), Reductions.Norm(ra, rb, NormType.L1));
        Assert.Equal(Reductions.Reduce<byte, int>(a, 1, ReduceType.Sum).ToArray(), Reductions.Reduce<byte, int>(ra, 1, ReduceType.Sum).ToArray());

        var d = new Mat<byte>(45, 11, 3);
        Mat<byte> rd = InRoi(d);
        Arithmetic.Subtract(a, b, d, m);
        Arithmetic.Subtract(ra, rb, rd, rm);
        Assert.Equal(d.ToArray(), rd.ToArray());
    }

    [Fact]
    public void Arithmetic_RoiDoesNotWriteOutsideView()
    {
        var parent = new Mat<byte>(20, 10, 1);
        parent.Fill(7);
        Mat<byte> roi = parent.Roi(4, 3, 9, 5);
        Arithmetic.Add(roi, new Scalar(10), roi);
        for (int y = 0; y < 10; y++)
        {
            for (int x = 0; x < 20; x++)
            {
                bool inside = x is >= 4 and < 13 && y is >= 3 and < 8;
                Assert.Equal(inside ? 17 : 7, parent.At(y, x));
            }
        }
    }

    [Fact]
    public void Arithmetic_InPlace_MatchesOutOfPlace()
    {
        Mat<float> a = Floats(53, 7, 1, 4), b = Floats(53, 7, 1, 5);
        Mat<float> expected = Arithmetic.AddWeighted(a, 0.3, b, 0.7, 1.5);
        Arithmetic.AddWeighted(a, 0.3, b, 0.7, 1.5, a);
        Assert.Equal(expected.ToArray(), a.ToArray());
    }

    [Fact]
    public void Arithmetic_DoesNotModifySources()
    {
        Mat<byte> a = Bytes(40, 6, 1, 6), b = Bytes(40, 6, 1, 7);
        byte[] a0 = a.ToArray(), b0 = b.ToArray();
        _ = Arithmetic.Divide(a, b, 2);
        _ = Bitwise.Xor(a, b);
        _ = Reductions.MinMaxLoc(a);
        Assert.Equal(a0, a.ToArray());
        Assert.Equal(b0, b.ToArray());
    }

    [Fact]
    public void Arithmetic_ShapeMismatch_Throws()
    {
        Assert.Throws<ArgumentException>(() => Arithmetic.Add(new Mat<byte>(4, 4), new Mat<byte>(4, 5)));
        Assert.Throws<ArgumentException>(() => Arithmetic.Add(new Mat<byte>(4, 4, 3), new Mat<byte>(4, 4, 1)));
        Assert.Throws<ArgumentException>(() =>
            Arithmetic.Add(new Mat<byte>(4, 4), new Mat<byte>(4, 4), new Mat<byte>(4, 4), new Mat<byte>(4, 4, 2)));
        Assert.Throws<ArgumentNullException>(() => Arithmetic.Add(new Mat<byte>(4, 4), (Mat<byte>)null!));
        Assert.Throws<NotSupportedException>(() => Arithmetic.Add(new Mat<long>(2, 2), new Mat<long>(2, 2)));
    }

    [Fact]
    public void Arithmetic_SinglePixel_Saturates()
    {
        Assert.Equal(255, Arithmetic.Add(new Mat<byte>([250], 1, 1), new Mat<byte>([10], 1, 1)).At(0, 0));
        Assert.Equal(0, Arithmetic.Subtract(new Mat<byte>([5], 1, 1), new Mat<byte>([10], 1, 1)).At(0, 0));
        Assert.Equal(-128, Arithmetic.Subtract(new Mat<sbyte>([-100], 1, 1), new Mat<sbyte>([100], 1, 1)).At(0, 0));
        Assert.Equal(int.MinValue, Arithmetic.Add(new Mat<int>([int.MaxValue], 1, 1), new Mat<int>([1], 1, 1)).At(0, 0));
    }

    /// <summary>OpenCV 5.0 <c>absdiff32s</c> is <c>std::abs(a - b)</c> with a wrapped difference; 4.5.5 gives INT_MIN.</summary>
    [Fact]
    public void AbsDiff_ScalarBeyondIntRange_UsesWrappedInt32Difference()
    {
        Mat<byte> r = Arithmetic.AbsDiff(new Mat<byte>([0, 5, 200], 3, 1), new Scalar(-3e9));
        Assert.Equal(new byte[] { 0, 255, 255 }, r.ToArray());
    }

    [Fact]
    public void SetTo_WithMask_ChangesOnlySelectedPixels()
    {
        var m = new Mat<ushort>(4, 2, 3);
        var mask = new Mat<byte>([0, 1, 0, 9, 0, 0, 3, 0], 4, 2);
        m.SetTo(new Scalar(1.5, 70000, -4), mask);
        ushort[] expected = [0, 0, 0, 2, 65535, 0, 0, 0, 0, 2, 65535, 0, 0, 0, 0, 0, 0, 0, 2, 65535, 0, 0, 0, 0];
        Assert.Equal(expected, m.ToArray());

        m.SetTo(new Scalar(9));
        Assert.All(Conversions.Split(m)[0].ToArray(), v => Assert.Equal(9, v));
    }

    [Fact]
    public void CopyTo_PerChannelMask_SelectsSingleElements()
    {
        var src = new Mat<byte>([1, 2, 3, 4, 5, 6], 2, 1, 3);
        var dst = new Mat<byte>(2, 1, 3);
        src.CopyTo(dst, new Mat<byte>([1, 0, 1, 0, 1, 0], 2, 1, 3));
        Assert.Equal(new byte[] { 1, 0, 3, 0, 5, 0 }, dst.ToArray());
    }

    [Fact]
    public void SplitMerge_RoundTrip()
    {
        Mat<short> src = Random(33, 5, 4, 8, r => (short)r.Next(-30000, 30000));
        Mat<short> back = Conversions.Merge(Conversions.Split(src));
        Assert.Equal(src.ToArray(), back.ToArray());
    }

    [Fact]
    public void MinMaxLoc_TiesAcrossRows_ReturnFirstOccurrence()
    {
        var m = new Mat<byte>(300, 40);
        m.Fill(5);
        m.At(17, 250) = 9;
        m.At(30, 3) = 9;
        m.At(12, 290) = 1;
        m.At(39, 0) = 1;
        MinMaxLocResult r = Reductions.MinMaxLoc(m);
        Assert.Equal(new MinMaxLocResult(1, 9, new Point(290, 12), new Point(250, 17)), r);
    }

    [Fact]
    public void Reductions_ZeroImage()
    {
        var z = new Mat<float>(9, 4, 2);
        Assert.Equal(new Scalar(0, 0), Reductions.Sum(z));
        Assert.Equal(0, Reductions.Norm(z));
        Assert.Equal(0, Reductions.CountNonZero(new Mat<float>(9, 4)));
        Reductions.MeanStdDev(z, out Scalar mean, out Scalar sd);
        Assert.Equal(new Scalar(0, 0), mean);
        Assert.Equal(new Scalar(0, 0), sd);
    }

    [Fact]
    public void Reduce_UnsupportedPair_Throws()
    {
        Assert.Throws<NotSupportedException>(() => Reductions.Reduce<byte, byte>(new Mat<byte>(3, 3), 0, ReduceType.Sum));
        Assert.Throws<NotSupportedException>(() => Reductions.Reduce<int, int>(new Mat<int>(3, 3), 0, ReduceType.Max));
        Assert.Throws<ArgumentOutOfRangeException>(() => Reductions.Reduce<byte, int>(new Mat<byte>(3, 3), 2, ReduceType.Sum));
    }

    [Fact]
    public void Operations_AreDeterministic()
    {
        Mat<float> a = Floats(70, 9, 3, 9), b = Floats(70, 9, 3, 10);
        Assert.Equal(Arithmetic.Divide(a, b, 0.3).ToArray(), Arithmetic.Divide(a, b, 0.3).ToArray());
        Assert.Equal(Reductions.Norm(a, b), Reductions.Norm(a, b));
    }

    [Fact]
    public void Lut_InvalidTable_Throws()
    {
        Assert.Throws<ArgumentException>(() => Conversions.Lut(new Mat<byte>(3, 3, 3), new Mat<byte>(255, 1)));
        Assert.Throws<ArgumentException>(() => Conversions.Lut(new Mat<byte>(3, 3, 3), new Mat<byte>(256, 1, 2)));
        Assert.Equal(Enumerable.Range(0, 9).Select(i => (byte)(255 - i)).ToArray(),
            Conversions.Lut(new Mat<byte>(Enumerable.Range(0, 9).Select(i => (byte)i).ToArray(), 9, 1),
                new Mat<byte>(Enumerable.Range(0, 256).Select(i => (byte)(255 - i)).ToArray(), 256, 1)).ToArray());
    }

    [Fact]
    public void ByteReductions_RowsBeyondAccumulatorLimits_AreExact()
    {
        Mat<byte> a = Bytes(70001, 2, 1, 11), b = Bytes(70001, 2, 1, 12);
        long sum = 0, sq = 0, l1 = 0, l2 = 0;
        int inf = 0;
        for (int y = 0; y < 2; y++)
        {
            for (int x = 0; x < 70001; x++)
            {
                int v = a.At(y, x), d = Math.Abs(v - b.At(y, x));
                sum += v;
                sq += v * v;
                l1 += d;
                l2 += d * d;
                inf = Math.Max(inf, d);
            }
        }

        Assert.Equal((double)sum, Reductions.Sum(a).V0);
        Reductions.MeanStdDev(a, out Scalar mean, out Scalar sd);
        double m = sum * (1.0 / 140002);
        Assert.Equal(m, mean.V0);
        Assert.Equal(Math.Sqrt(Math.Max((sq * (1.0 / 140002)) - (m * m), 0)), sd.V0);
        Assert.Equal((double)l1, Reductions.Norm(a, b, NormType.L1));
        Assert.Equal((double)l2, Reductions.Norm(a, b, NormType.L2Sqr));
        Assert.Equal(inf, Reductions.Norm(a, b, NormType.Inf));
        Assert.Equal(a.ToArray().Count(v => v != 0), Reductions.CountNonZero(a));
    }

    [Fact]
    public void MinMaxLoc_FloatVectorPath_SkipsNaNAndKeepsFirstOccurrence()
    {
        var rng = new Random(13);
        for (int trial = 0; trial < 200; trial++)
        {
            int w = rng.Next(1, 70);
            var m = new Mat<float>(w, 3);
            for (int y = 0; y < 3; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    int k = rng.Next(12);
                    m.At(y, x) = k switch
                    {
                        0 => float.NaN,
                        1 => float.PositiveInfinity,
                        2 => float.NegativeInfinity,
                        3 => 0f,
                        4 => -0f,
                        _ => rng.Next(-3, 4),
                    };
                }
            }

            double min = double.PositiveInfinity, max = double.NegativeInfinity;
            Point minLoc = new(-1, -1), maxLoc = new(-1, -1);
            for (int y = 0; y < 3; y++)
            {
                for (int x = 0; x < w; x++)
                {
                    float v = m.At(y, x);
                    if (v < min || (minLoc.X < 0 && v == min))
                    {
                        min = v;
                        minLoc = new Point(x, y);
                    }

                    if (v > max || (maxLoc.X < 0 && v == max))
                    {
                        max = v;
                        maxLoc = new Point(x, y);
                    }
                }
            }

            MinMaxLocResult r = Reductions.MinMaxLoc(m);
            Assert.Equal(minLoc.X < 0 ? new Point(0, 0) : minLoc, r.MinLoc);
            Assert.Equal(maxLoc.X < 0 ? new Point(0, 0) : maxLoc, r.MaxLoc);
            Assert.Equal(BitConverter.DoubleToInt64Bits(min), BitConverter.DoubleToInt64Bits(r.MinVal));
            Assert.Equal(BitConverter.DoubleToInt64Bits(max), BitConverter.DoubleToInt64Bits(r.MaxVal));
        }
    }
}
