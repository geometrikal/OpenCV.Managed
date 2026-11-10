namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.ImgProc;
using Geometrikal.OpenCV.Managed.ImgProc.Internal;

/// <summary>
/// Shape, aliasing, validation and analytic tests for the histogram, CLAHE, integral and
/// normalize ports. Vector paths are compared with scalar references on wide rows.
/// </summary>
public sealed class HistogramBasicTests
{
    private static readonly float[][] Range256 = [[0, 256]];

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

    private static Mat<byte> Mask(int w, int h, int seed) => Random(w, h, 1, seed, r => (byte)(r.Next(3) == 0 ? 0 : r.Next(1, 256)));

    private static Mat<T> InRoi<T>(Mat<T> packed)
        where T : unmanaged
    {
        var parent = new Mat<T>(packed.Width + 7, packed.Height + 5, packed.Channels);
        parent.Fill(default);
        Mat<T> roi = parent.Roi(3, 2, packed.Width, packed.Height);
        packed.CopyTo(roi);
        return roi;
    }

    /// <summary>Scalar transcription of the 16U/32F uniform calcHist bin rule.</summary>
    private static float[] ReferenceHist(float[] values, int size, float lo, float hi)
    {
        double a = size / ((double)hi - lo);
        double b = -a * lo;
        var h = new float[size];
        foreach (float f in values)
        {
            double v = f;
            int idx = ScalarConverter.CvFloor((v * a) + b);
            if (v < lo || v >= hi)
            {
                continue;
            }

            h[Math.Clamp(idx, 0, size - 1)]++;
        }

        return h;
    }

    // ---------------------------------------------------------------- calcHist

    [Fact]
    public void CalcHist_RoiWithStride_MatchesPackedImage()
    {
        Mat<byte> packed = Bytes(37, 23, 3, 1);
        Mat<byte> mask = Mask(37, 23, 2);
        float[][] ranges = [[0, 256], [10, 250]];
        DenseHistogram a = Histograms.CalcHist([packed], [2, 0], mask, [12, 9], ranges);
        DenseHistogram b = Histograms.CalcHist([InRoi(packed)], [2, 0], InRoi(mask), [12, 9], ranges);
        Assert.Equal(a.Values.ToArray(), b.Values.ToArray());

        Mat<float> pf = Random(41, 19, 1, 3, r => (float)(r.NextDouble() * 2));
        Assert.Equal(
            Histograms.CalcHist(pf, 0, 17, 0, 2).Values.ToArray(),
            Histograms.CalcHist(InRoi(pf), 0, 17, 0, 2).Values.ToArray());
    }

    [Theory]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(31)]
    [InlineData(1003)]
    public void CalcHist_Float32WideRows_MatchesScalarReference(int width)
    {
        Mat<float> src = Random(width, 5, 1, width, r => (float)((r.NextDouble() * 1.4) - 0.2));
        float[] all = src.ToArray();
        all[0] = 1f;
        all[^1] = float.NaN;
        if (all.Length > 3)
        {
            all[1] = MathF.BitDecrement(1f);
            all[2] = float.PositiveInfinity;
        }

        src = new Mat<float>(all, width, 5);
        // NaN passes the range test of the reference and goes to bin 0.
        float[] expected = ReferenceHist(all, 10, 0, 1);
        Assert.Equal(expected, Histograms.CalcHist(src, 0, 10, 0, 1).Values.ToArray());
    }

    [Fact]
    public void CalcHist_UInt16WideRows_MatchesScalarReference()
    {
        Mat<ushort> src = Random(1001, 3, 1, 9, r => (ushort)r.Next(65536));
        float[] values = src.ToArray().Select(v => (float)v).ToArray();
        Assert.Equal(ReferenceHist(values, 37, 10, 65000), Histograms.CalcHist(src, 0, 37, 10, 65000).Values.ToArray());
    }

    [Fact]
    public void CalcHist_NonUniformNaN_IsDropped()
    {
        var src = new Mat<float>([float.NaN, 0.05f, 0.5f, 2f], 4, 1);
        DenseHistogram h = Histograms.CalcHist([src], [0], null, [2], [[0, 0.1f, 1]], uniform: false);
        Assert.Equal(new float[] { 1, 1 }, h.Values.ToArray());
    }

    [Fact]
    public void CalcHist_Accumulate_RoundsPriorBinsHalfToEven()
    {
        var src = new Mat<byte>([0, 1, 1], 3, 1);
        var hist = new DenseHistogram(4);
        new float[] { 0.5f, 1.5f, 2.5f, -0.5f }.CopyTo(hist.Values);
        Histograms.CalcHist([src], [0], null, hist, [[0, 4]], accumulate: true);
        Assert.Equal(new float[] { 1, 4, 2, 0 }, hist.Values.ToArray());
        Histograms.CalcHist([src], [0], null, hist, [[0, 4]], accumulate: false);
        Assert.Equal(new float[] { 1, 2, 0, 0 }, hist.Values.ToArray());
    }

    [Fact]
    public void CalcHist_AllMaskedOut_GivesZeros()
    {
        Mat<byte> src = Bytes(33, 9, 1, 4);
        var mask = new Mat<byte>(33, 9);
        Assert.All(Histograms.CalcHist([src], [0], mask, [16], Range256).Values.ToArray(), v => Assert.Equal(0f, v));
        Mat<float> f = Random(33, 9, 1, 5, r => (float)r.NextDouble());
        Assert.All(Histograms.CalcHist([f], [0], mask, [16], [[0, 1]]).Values.ToArray(), v => Assert.Equal(0f, v));
    }

    [Fact]
    public void CalcHist_DoesNotModifySourceAndIsDeterministic()
    {
        Mat<byte> src = Bytes(64, 40, 3, 6);
        byte[] before = src.ToArray();
        float[][] ranges = [[0, 256], [0, 256], [0, 256]];
        float[] first = Histograms.CalcHist([src], [0, 1, 2], null, [8, 8, 8], ranges).Values.ToArray();
        float[] second = Histograms.CalcHist([src], [0, 1, 2], null, [8, 8, 8], ranges).Values.ToArray();
        Assert.Equal(first, second);
        Assert.Equal(before, src.ToArray());
        Assert.Equal(64 * 40, first.Sum());
    }

    [Fact]
    public void CalcHist_SinglePixel_CountsOneBin()
    {
        var src = new Mat<byte>([200], 1, 1);
        DenseHistogram h = Histograms.CalcHist([src], [0], null, [4], null);
        Assert.Equal(new float[] { 0, 0, 0, 1 }, h.Values.ToArray());
        Assert.Equal(1f, h.At(3));
    }

    [Fact]
    public void CalcHist_InvalidArguments_Throw()
    {
        Mat<byte> a = Bytes(8, 8, 1, 1);
        Mat<float> f = Random(8, 8, 1, 1, r => 0f);
        Assert.Throws<ArgumentException>(() => Histograms.CalcHist<byte>([], [0], null, [4], Range256));
        Assert.Throws<ArgumentNullException>(() => Histograms.CalcHist<byte>([null!], [0], null, [4], Range256));
        Assert.Throws<ArgumentOutOfRangeException>(() => Histograms.CalcHist([a], [1], null, [4], Range256));
        Assert.Throws<ArgumentException>(() => Histograms.CalcHist([a, Bytes(9, 8, 1, 1)], [0, 1], null, [4, 4], [[0, 256], [0, 256]]));
        Assert.Throws<ArgumentException>(() => Histograms.CalcHist([a], [0], new Mat<byte>(7, 8), [4], Range256));
        Assert.Throws<ArgumentException>(() => Histograms.CalcHist([f], [0], null, [4], null));
        Assert.Throws<ArgumentException>(() => Histograms.CalcHist([a], [0], null, [4], [[5, 5]]));
        Assert.Throws<ArgumentException>(() => Histograms.CalcHist([a], [0], null, [2], [[0, 5, 5]], uniform: false));
        Assert.Throws<ArgumentException>(() => Histograms.CalcHist([a], [0], null, [4], [[0, 1], [0, 1]]));
        Assert.Throws<ArgumentOutOfRangeException>(() => Histograms.CalcHist([a], [0], null, [0], Range256));
        Assert.Throws<ArgumentException>(() => Histograms.CalcHist([a], [0, 0], null, [4], Range256));
        Assert.Throws<NotSupportedException>(() => Histograms.CalcHist([new Mat<short>(4, 4)], [0], null, [4], Range256));
        Assert.Throws<ArgumentException>(() => new DenseHistogram(Enumerable.Repeat(1, 33).ToArray()));
    }

    // ---------------------------------------------------------- calcBackProject

    [Fact]
    public void CalcBackProject_RoiWithStride_MatchesPackedImage()
    {
        Mat<byte> packed = Bytes(70, 13, 2, 7);
        float[][] ranges = [[0, 256], [20, 240]];
        DenseHistogram h = Histograms.CalcHist([packed], [0, 1], null, [8, 6], ranges);
        Assert.Equal(
            Histograms.CalcBackProject([packed], [0, 1], h, ranges, 0.7).ToArray(),
            Histograms.CalcBackProject([InRoi(packed)], [0, 1], h, ranges, 0.7).ToArray());
        DenseHistogram h1 = Histograms.CalcHist([packed], [1], null, [16], Range256);
        Assert.Equal(
            Histograms.CalcBackProject([packed], [1], h1, Range256, 2).ToArray(),
            Histograms.CalcBackProject([InRoi(packed)], [1], h1, Range256, 2).ToArray());
    }

    [Fact]
    public void CalcBackProject_Float32UpperEdgeAndNaN_GiveZero()
    {
        // 0.99999994 * 10 rounds to 10 in double arithmetic only when the bin index reaches size;
        // calcBackProject has no clamp, so such values and NaN give 0.
        var src = new Mat<float>([float.NaN, 0.05f, 1f, -0.0f, MathF.BitDecrement(0f)], 5, 1);
        var hist = new DenseHistogram(10);
        hist.Values.Fill(3);
        Mat<float> r = Histograms.CalcBackProject([src], [0], hist, [[0, 1]]);
        Assert.Equal(new float[] { 0, 3, 0, 3, 0 }, r.ToArray());
    }

    [Fact]
    public void CalcBackProject_ByteSaturatesAndRounds()
    {
        var src = new Mat<byte>([0, 64, 128, 192], 4, 1);
        var hist = new DenseHistogram(4);
        new float[] { 0.5f, 1.5f, 100f, 1000f }.CopyTo(hist.Values);
        Assert.Equal(new byte[] { 0, 2, 100, 255 }, Histograms.CalcBackProject([src], [0], hist, null).ToArray());
    }

    [Fact]
    public void CalcBackProject_InvalidArguments_Throw()
    {
        Mat<byte> a = Bytes(8, 8, 1, 1);
        Assert.Throws<ArgumentNullException>(() => Histograms.CalcBackProject([a], [0], null!, Range256));
        Assert.Throws<ArgumentException>(() => Histograms.CalcBackProject([a], [0], new DenseHistogram(4, 4), Range256));
        Assert.Throws<NotSupportedException>(() => Histograms.CalcBackProject([new Mat<int>(2, 2)], [0], new DenseHistogram(4), Range256));
    }

    // ---------------------------------------------------------------- compareHist

    [Fact]
    public void CompareHist_IdenticalHistograms_GiveKnownValues()
    {
        var h = new DenseHistogram(7);
        new float[] { 1, 2, 3, 4, 5, 6, 7 }.CopyTo(h.Values);
        Assert.Equal(1.0, Histograms.CompareHist(h, h, HistCompMethods.Correl), 12);
        Assert.Equal(0.0, Histograms.CompareHist(h, h, HistCompMethods.Chisqr));
        Assert.Equal(28.0, Histograms.CompareHist(h, h, HistCompMethods.Intersect));
        Assert.Equal(0.0, Histograms.CompareHist(h, h, HistCompMethods.Bhattacharyya), 6);
        Assert.Equal(0.0, Histograms.CompareHist(h, h, HistCompMethods.KLDiv));
    }

    [Fact]
    public void CompareHist_NonContinuousMatrix_ProcessesRowsAsPlanes()
    {
        Mat<float> a = Random(5, 3, 1, 1, r => (float)r.NextDouble());
        Mat<float> b = Random(5, 3, 1, 2, r => (float)r.NextDouble());
        double expected = 0;
        for (int y = 0; y < 3; y++)
        {
            var ra = new DenseHistogram(5);
            var rb = new DenseHistogram(5);
            a.GetReadOnlyRowSpan(y).CopyTo(ra.Values);
            b.GetReadOnlyRowSpan(y).CopyTo(rb.Values);
            expected += Histograms.CompareHist(ra, rb, HistCompMethods.Intersect);
        }

        Assert.Equal(expected, Histograms.CompareHist(InRoi(a), InRoi(b), HistCompMethods.Intersect));
    }

    [Fact]
    public void CompareHist_InvalidArguments_Throw()
    {
        Assert.Throws<ArgumentException>(() => Histograms.CompareHist(new DenseHistogram(4), new DenseHistogram(5), HistCompMethods.Correl));
        Assert.Throws<ArgumentException>(() => Histograms.CompareHist(new DenseHistogram(4, 2), new DenseHistogram(2, 4), HistCompMethods.Correl));
        Assert.Throws<ArgumentOutOfRangeException>(() => Histograms.CompareHist(new DenseHistogram(4), new DenseHistogram(4), (HistCompMethods)6));
    }

    // ---------------------------------------------------------------- equalizeHist

    [Fact]
    public void EqualizeHist_InPlaceAndRoi_MatchOutOfPlace()
    {
        Mat<byte> src = Bytes(301, 37, 1, 8);
        byte[] expected = Histograms.EqualizeHist(src).ToArray();
        Mat<byte> roi = InRoi(src);
        Assert.Equal(expected, Histograms.EqualizeHist(roi).ToArray());
        Histograms.EqualizeHist(roi, roi);
        Assert.Equal(expected, roi.ToArray());
    }

    [Fact]
    public void EqualizeHist_InvalidArguments_Throw()
    {
        Assert.Throws<ArgumentException>(() => Histograms.EqualizeHist(new Mat<byte>(4, 4, 3)));
        Assert.Throws<ArgumentException>(() => Histograms.EqualizeHist(new Mat<byte>(4, 4), new Mat<byte>(5, 4)));
        Assert.Throws<ArgumentNullException>(() => Histograms.EqualizeHist(null!));
    }

    // ---------------------------------------------------------------- CLAHE

    [Fact]
    public void Clahe_InPlaceAndRoi_MatchOutOfPlace()
    {
        Mat<byte> src = Bytes(67, 53, 1, 9);
        var clahe = new Clahe(2.0, new Size(8, 8));
        byte[] expected = clahe.Apply(src).ToArray();
        Mat<byte> roi = InRoi(src);
        Assert.Equal(expected, clahe.Apply(roi).ToArray());
        clahe.Apply(roi, roi);
        Assert.Equal(expected, roi.ToArray());

        Mat<ushort> s16 = Random(40, 30, 1, 10, r => (ushort)r.Next(65536));
        ushort[] e16 = clahe.Apply(s16).ToArray();
        Mat<ushort> r16 = InRoi(s16);
        clahe.Apply(r16, r16);
        Assert.Equal(e16, r16.ToArray());
    }

    [Fact]
    public void Clahe_BitShiftZero_EqualsDefault()
    {
        Mat<byte> src = Bytes(48, 40, 1, 11);
        Assert.Equal(new Clahe().Apply(src).ToArray(), new Clahe { BitShift = 0 }.Apply(src).ToArray());
    }

    [Fact]
    public void Clahe_InvalidArguments_Throw()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Clahe(40, new Size(0, 8)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Clahe { BitShift = 16 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new Clahe { BitShift = -1 });
        Assert.Throws<ArgumentException>(() => new Clahe { BitShift = 8 }.Apply(new Mat<byte>(8, 8)));
        Assert.Throws<ArgumentException>(() => new Clahe().Apply(new Mat<byte>(8, 8, 3)));
        Assert.Throws<ArgumentException>(() => new Clahe().Apply(new Mat<byte>(8, 8), new Mat<byte>(9, 8)));
    }

    // ---------------------------------------------------------------- vector/scalar parity

    [Fact]
    public void ByteLut_ScalarAndVbmiPaths_AgreeForAllLengths()
    {
        var lut = new byte[256];
        new Random(3).NextBytes(lut);
        var src = new byte[300];
        new Random(4).NextBytes(src);
        bool saved = ByteLut.UseVbmi;
        try
        {
            for (int n = 0; n <= 200; n++)
            {
                ByteLut.UseVbmi = false;
                var scalar = new byte[n];
                ByteLut.Apply(src.AsSpan(0, n), scalar, lut);
                ByteLut.UseVbmi = saved;
                byte[] inPlace = src[..n];
                ByteLut.Apply(inPlace, inPlace, lut);
                Assert.Equal(src.Take(n).Select(v => lut[v]).ToArray(), scalar);
                Assert.Equal(scalar, inPlace);
            }
        }
        finally
        {
            ByteLut.UseVbmi = saved;
        }
    }

    [Fact]
    public void ClaheAndEqualizeHist_ScalarLutPath_MatchesDefaultPath()
    {
        Mat<byte> src = Bytes(517, 301, 1, 12);
        var clahe = new Clahe(3.0, new Size(8, 8));
        byte[] fastClahe = clahe.Apply(src).ToArray();
        byte[] fastEq = Histograms.EqualizeHist(src).ToArray();
        bool saved = ByteLut.UseVbmi;
        try
        {
            ByteLut.UseVbmi = false;
            Assert.Equal(fastClahe, clahe.Apply(src).ToArray());
            Assert.Equal(fastEq, Histograms.EqualizeHist(src).ToArray());
        }
        finally
        {
            ByteLut.UseVbmi = saved;
        }
    }

    [Theory]
    [InlineData(517, 301, 0)]
    [InlineData(64, 70, 0)]
    [InlineData(130, 67, 3)]
    public void Clahe_Vector512BlendPath_MatchesVectorPath(int w, int h, int bitShift)
    {
        Mat<byte> src = Bytes(w, h, 1, 18);
        var clahe = new Clahe(3.0, new Size(8, 8)) { BitShift = bitShift };
        byte[] fast = clahe.Apply(src).ToArray();
        bool saved = ClaheEngine.UseVector512;
        try
        {
            ClaheEngine.UseVector512 = false;
            Assert.Equal(fast, clahe.Apply(src).ToArray());
        }
        finally
        {
            ClaheEngine.UseVector512 = saved;
        }
    }

    [Theory]
    [InlineData(533)]
    [InlineData(16)]
    [InlineData(15)]
    public void Integral_Avx512PrefixPath_MatchesSsePath(int width)
    {
        Mat<byte> src = Bytes(width, 37, 1, 19);
        Mat<int> s = IntegralImages.Integral<byte, int, double>(src, out Mat<double> q);
        Mat<float> f = IntegralImages.Integral<byte, float, float>(src, out Mat<float> fq);
        bool saved = IntegralKernels.UseAvx512;
        try
        {
            IntegralKernels.UseAvx512 = false;
            Mat<int> s2 = IntegralImages.Integral<byte, int, double>(src, out Mat<double> q2);
            Mat<float> f2 = IntegralImages.Integral<byte, float, float>(src, out Mat<float> fq2);
            Assert.Equal(s.ToArray(), s2.ToArray());
            Assert.Equal(q.ToArray(), q2.ToArray());
            Assert.Equal(f.ToArray(), f2.ToArray());
            Assert.Equal(fq.ToArray(), fq2.ToArray());
        }
        finally
        {
            IntegralKernels.UseAvx512 = saved;
        }
    }

    [Fact]
    public void CalcHist8u_VbmiTablePath_MatchesScalarPath()
    {
        Mat<byte> c3 = InRoi(Bytes(203, 7, 3, 21));
        Mat<byte> c4 = InRoi(Bytes(203, 7, 4, 22));
        Mat<byte> c2 = InRoi(Bytes(203, 7, 2, 23));
        Mat<byte> g = InRoi(Bytes(203, 7, 1, 24));
        Mat<byte> mask = InRoi(Mask(203, 7, 25));
        var cases = new Func<float[]>[]
        {
            () => Histograms.CalcHist([c3], [0, 1, 2], null, [8, 8, 8], [[0, 256], [0, 256], [0, 256]]).Values.ToArray(),
            () => Histograms.CalcHist([c3], [2, 0, 1], mask, [5, 6, 7], [[10, 200], [0.5f, 255.5f], [-3, 300]]).Values.ToArray(),
            () => Histograms.CalcHist([c4, g], [3, 4], null, [30, 32], [[0, 180], [20, 240]]).Values.ToArray(),
            () => Histograms.CalcHist([c2, c4], [1, 2, 5], mask, [4, 9, 3], [[0, 256], [0, 128], [64, 256]]).Values.ToArray(),
            () => Histograms.CalcHist([g, c2], [0, 2], mask, [10, 11], [[0, 256], [0, 256]], uniform: true).Values.ToArray(),
        };
        bool saved = ByteLut.UseVbmi;
        try
        {
            foreach (Func<float[]> run in cases)
            {
                ByteLut.UseVbmi = saved;
                float[] fast = run();
                ByteLut.UseVbmi = false;
                Assert.Equal(run(), fast);
            }
        }
        finally
        {
            ByteLut.UseVbmi = saved;
        }
    }

    // ---------------------------------------------------------------- integral

    [Fact]
    public void Integral_RoiWithStride_MatchesPackedImage()
    {
        Mat<byte> packed = Bytes(37, 21, 3, 13);
        Mat<int> a = IntegralImages.Integral<byte, int, double>(packed, out Mat<double> qa, out Mat<int> ta);
        Mat<int> b = IntegralImages.Integral<byte, int, double>(InRoi(packed), out Mat<double> qb, out Mat<int> tb);
        Assert.Equal(a.ToArray(), b.ToArray());
        Assert.Equal(qa.ToArray(), qb.ToArray());
        Assert.Equal(ta.ToArray(), tb.ToArray());
    }

    [Fact]
    public void Integral_TiltedSum_MatchesDefinition()
    {
        Mat<byte> src = Bytes(9, 7, 1, 14);
        IntegralImages.Integral<byte, int, double>(src, out _, out Mat<int> tilted);
        for (int yy = 0; yy <= 7; yy++)
        {
            for (int xx = 0; xx <= 9; xx++)
            {
                int expected = 0;
                for (int y = 0; y < yy; y++)
                {
                    for (int x = 0; x < 9; x++)
                    {
                        if (Math.Abs(x - xx + 1) <= yy - 1 - y)
                        {
                            expected += src.At(y, x);
                        }
                    }
                }

                Assert.Equal(expected, tilted.At(yy, xx));
            }
        }
    }

    [Fact]
    public void Integral_Int32Overflow_Wraps()
    {
        var src = new Mat<byte>(9000, 1000);
        src.Fill(255);
        Mat<int> s = IntegralImages.Integral<byte, int>(src);
        Assert.Equal(unchecked((int)(255L * 9000 * 1000)), s.At(1000, 9000));
        Assert.Equal(255 * 9000, s.At(1, 9000));
    }

    /// <summary>
    /// Rows of more than 2^24 / 255 pixels make the byte-to-float row sum inexact; the port keeps
    /// the scalar order <c>s += v</c> in float.
    /// </summary>
    [Fact]
    public void Integral_ByteToFloatLargeRowSums_UsesScalarOrder()
    {
        Mat<byte> src = Random(70001, 2, 1, 15, r => (byte)(250 + r.Next(6)));
        Mat<float> s = IntegralImages.Integral<byte, float>(src);
        float acc0 = 0;
        float acc1 = 0;
        int mismatches = 0;
        int inexact = 0;
        long exact = 0;
        for (int x = 0; x < src.Width; x++)
        {
            acc0 += src.At(0, x);
            acc1 += src.At(1, x);
            exact += src.At(0, x);
            inexact += (double)acc0 != exact ? 1 : 0;
            mismatches += s.At(1, x + 1) != 0f + acc0 ? 1 : 0;
            mismatches += s.At(2, x + 1) != s.At(1, x + 1) + acc1 ? 1 : 0;
        }

        Assert.Equal(0, mismatches);
        Assert.True(inexact > 0);
    }

    [Fact]
    public void Integral_UnsupportedCombinations_Throw()
    {
        Assert.Throws<NotSupportedException>(() => IntegralImages.Integral<ushort, int>(new Mat<ushort>(2, 2)));
        Assert.Throws<NotSupportedException>(() => IntegralImages.Integral<short, float>(new Mat<short>(2, 2)));
        Assert.Throws<NotSupportedException>(() => IntegralImages.Integral<float, int>(new Mat<float>(2, 2)));
        Assert.Throws<NotSupportedException>(() => IntegralImages.Integral<byte, double, float>(new Mat<byte>(2, 2), out _));
        Assert.Throws<NotSupportedException>(() => IntegralImages.Integral<int, double>(new Mat<int>(2, 2)));
        Assert.Throws<ArgumentNullException>(() => IntegralImages.Integral<byte, int>(null!));
    }

    /// <summary>A destination inside a larger parent filled with <paramref name="fill"/>.</summary>
    private static Mat<T> DestinationRoi<T>(int w, int h, int cn, T fill, out Mat<T> parent)
        where T : unmanaged
    {
        parent = new Mat<T>(w + 6, h + 4, cn);
        parent.Fill(fill);
        return parent.Roi(2, 3, w, h);
    }

    private static void AssertOutsideRoi<T>(Mat<T> parent, int w, int h, T fill)
        where T : unmanaged
    {
        for (int y = 0; y < parent.Height; y++)
        {
            for (int x = 0; x < parent.Width; x++)
            {
                if (x >= 2 && x < 2 + w && y >= 3 && y < 3 + h)
                {
                    continue;
                }

                for (int c = 0; c < parent.Channels; c++)
                {
                    Assert.Equal(fill, parent.At(y, x, c));
                }
            }
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    public void Integral_IntoDestination_EqualsAllocatingOverload(int cn)
    {
        Mat<byte> src = InRoi(Bytes(53, 19, cn, 16));
        Mat<int> sum = IntegralImages.Integral<byte, int>(src);
        var sumInto = new Mat<int>(54, 20, cn);
        sumInto.Fill(-7);
        IntegralImages.Integral(src, sumInto);
        Assert.Equal(sum.ToArray(), sumInto.ToArray());

        Mat<int> s2 = IntegralImages.Integral<byte, int, double>(src, out Mat<double> sq);
        Mat<int> s2Into = DestinationRoi(54, 20, cn, -7, out Mat<int> s2Parent);
        Mat<double> sqInto = DestinationRoi(54, 20, cn, double.NaN, out Mat<double> sqParent);
        IntegralImages.Integral(src, s2Into, sqInto);
        Assert.Equal(s2.ToArray(), s2Into.ToArray());
        Assert.Equal(sq.ToArray(), sqInto.ToArray());
        AssertOutsideRoi(s2Parent, 54, 20, -7);
        AssertOutsideRoi(sqParent, 54, 20, double.NaN);

        Mat<int> s3 = IntegralImages.Integral<byte, int, double>(src, out Mat<double> sq3, out Mat<int> tilted);
        Mat<int> s3Into = DestinationRoi(54, 20, cn, -7, out _);
        Mat<double> sq3Into = DestinationRoi(54, 20, cn, double.NaN, out _);
        Mat<int> tiltedInto = DestinationRoi(54, 20, cn, -7, out Mat<int> tiltedParent);
        IntegralImages.Integral(src, s3Into, sq3Into, tiltedInto);
        Assert.Equal(s3.ToArray(), s3Into.ToArray());
        Assert.Equal(sq3.ToArray(), sq3Into.ToArray());
        Assert.Equal(tilted.ToArray(), tiltedInto.ToArray());
        AssertOutsideRoi(tiltedParent, 54, 20, -7);
    }

    [Fact]
    public void Integral_IntoDestination_FloatPathsEqualAllocatingOverload()
    {
        Mat<float> src = Random(1031, 1029, 1, 17, r => (float)(r.NextDouble() * 100));
        Mat<float> sum = IntegralImages.Integral<float, float, float>(src, out Mat<float> sq);
        Mat<float> sumInto = DestinationRoi(1032, 1030, 1, float.NaN, out Mat<float> parent);
        var sqInto = new Mat<float>(1032, 1030);
        sqInto.Fill(float.NaN);
        IntegralImages.Integral(src, sumInto, sqInto);
        Assert.Equal(sum.ToArray(), sumInto.ToArray());
        Assert.Equal(sq.ToArray(), sqInto.ToArray());
        AssertOutsideRoi(parent, 1032, 1030, float.NaN);

        Mat<double> d = IntegralImages.Integral<float, double>(src);
        var dInto = new Mat<double>(1032, 1030);
        IntegralImages.Integral(src, dInto);
        Assert.Equal(d.ToArray(), dInto.ToArray());
    }

    [Fact]
    public void Integral_IntoDestination_InvalidArguments_Throw()
    {
        var src = new Mat<byte>(5, 4);
        Assert.Throws<ArgumentException>(() => IntegralImages.Integral(src, new Mat<int>(5, 5)));
        Assert.Throws<ArgumentException>(() => IntegralImages.Integral(src, new Mat<int>(6, 4)));
        Assert.Throws<ArgumentException>(() => IntegralImages.Integral(src, new Mat<int>(6, 5, 2)));
        Assert.Throws<ArgumentException>(() => IntegralImages.Integral(src, new Mat<int>(6, 5), new Mat<double>(5, 5)));
        Assert.Throws<ArgumentException>(
            () => IntegralImages.Integral(src, new Mat<int>(6, 5), new Mat<double>(6, 5), new Mat<int>(7, 5)));
        Assert.Throws<ArgumentNullException>(() => IntegralImages.Integral<byte, int>(src, null!));
        Assert.Throws<ArgumentNullException>(() => IntegralImages.Integral<byte, int, double>(src, new Mat<int>(6, 5), null!));
        Assert.Throws<NotSupportedException>(() => IntegralImages.Integral(src, new Mat<short>(6, 5)));

        var shared = new Mat<int>(6, 10);
        Assert.Throws<ArgumentException>(
            () => IntegralImages.Integral(src, shared.Roi(0, 0, 6, 5), shared.Roi(0, 5, 6, 5)));
        var sharedD = new Mat<double>(6, 10);
        Assert.Throws<ArgumentException>(
            () => IntegralImages.Integral(sharedD.Roi(0, 0, 5, 4), sharedD.Roi(0, 5, 6, 5)));
    }

    // ---------------------------------------------------------------- normalize

    [Fact]
    public void Normalize_ConstantImageMinMax_GivesLowerBound()
    {
        var src = new Mat<float>(5, 4);
        src.Fill(3.5f);
        Assert.All(Conversions.Normalize<float, float>(src, 9, 2, NormType.MinMax).ToArray(), v => Assert.Equal(2f, v));
    }

    [Fact]
    public void Normalize_WithMask_KeepsDestinationOutsideMask()
    {
        Mat<float> src = Random(9, 6, 1, 16, r => (float)(r.NextDouble() * 100));
        Mat<byte> mask = Mask(9, 6, 17);
        var dst = new Mat<byte>(9, 6);
        dst.Fill(7);
        Conversions.Normalize(src, dst, 0, 255, NormType.MinMax, mask);
        Mat<byte> full = Conversions.Normalize<float, byte>(src, 0, 255, NormType.MinMax, mask);
        byte[] m = mask.ToArray();
        byte[] d = dst.ToArray();
        byte[] f = full.ToArray();
        for (int i = 0; i < m.Length; i++)
        {
            Assert.Equal(m[i] != 0 ? f[i] : (byte)7, d[i]);
            if (m[i] == 0)
            {
                Assert.Equal(0, f[i]);
            }
        }
    }

    [Fact]
    public void Normalize_InvalidArguments_Throw()
    {
        var src = new Mat<float>(4, 4, 3);
        Assert.Throws<ArgumentException>(() => Conversions.Normalize<float, float>(src, 0, 1, NormType.MinMax, new Mat<byte>(4, 4)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Conversions.Normalize<float, float>(src, 0, 1, NormType.L2Sqr));
        Assert.Throws<ArgumentOutOfRangeException>(() => Reductions.Norm(src, NormType.MinMax));
    }
}
