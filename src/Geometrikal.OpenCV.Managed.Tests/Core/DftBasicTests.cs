namespace Geometrikal.OpenCV.Managed.Tests.Core;

using System;
using System.Linq;
using System.Threading.Tasks;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>Analytic, structural and invalid-input tests of <see cref="SpectralTransforms"/>.</summary>
public sealed class DftBasicTests
{
    [Fact]
    public void Dft_Impulse_GivesFlatSpectrum()
    {
        var x = new Mat<float>(new float[] { 1, 0, 0, 0, 0, 0, 0, 0 }, 8, 1);
        Mat<float> y = SpectralTransforms.Dft(x, DftFlags.ComplexOutput);
        Assert.Equal(2, y.Channels);
        for (int i = 0; i < 8; i++)
        {
            Assert.Equal(1f, y.At(0, i, 0));
            Assert.Equal(0f, Math.Abs(y.At(0, i, 1)));
        }
    }

    [Fact]
    public void Dft_RealRow_UsesCcsLayout()
    {
        var x = new Mat<double>(new double[] { 1, 2, 3, 4 }, 4, 1);
        Mat<double> y = SpectralTransforms.Dft(x);

        // X = [10, -2+2i, -2, -2-2i] packs as Re0, Re1, Im1, Re2.
        Assert.Equal(new double[] { 10, -2, 2, -2 }, y.ToArray());
    }

    [Fact]
    public void Dft_SingleElement_ScalesValue()
    {
        var x = new Mat<float>(new float[] { 3 }, 1, 1);
        Assert.Equal(3f, SpectralTransforms.Dft(x).At(0, 0));
        Assert.Equal(3f, SpectralTransforms.Dft(x, DftFlags.Scale).At(0, 0));
    }

    [Theory]
    [InlineData(64, 48)]
    [InlineData(45, 77)]
    [InlineData(17, 1)]
    [InlineData(1, 100)]
    public void Dft_RoundTripReal_ReturnsInput(int width, int height)
    {
        Mat<float> x = RandomFloat(width, height, 1, 11);
        Mat<float> back = SpectralTransforms.Idft(SpectralTransforms.Dft(x), DftFlags.Scale);
        AssertClose(x, back, 2e-5);

        Mat<double> xd = RandomDouble(width, height, 1, 12);
        Mat<double> backd = SpectralTransforms.Idft(SpectralTransforms.Dft(xd, DftFlags.ComplexOutput), DftFlags.Scale | DftFlags.RealOutput);
        AssertClose(xd, backd, 1e-12);
    }

    [Theory]
    [InlineData(32, 30, DftFlags.None)]
    [InlineData(21, 9, DftFlags.Rows)]
    public void Dft_RoundTripComplex_ReturnsInput(int width, int height, DftFlags flags)
    {
        Mat<double> x = RandomDouble(width, height, 2, 13);
        Mat<double> back = SpectralTransforms.Idft(SpectralTransforms.Dft(x, flags), flags | DftFlags.Scale);
        AssertClose(x, back, 1e-12);
    }

    [Theory]
    [InlineData(DftFlags.None, 1)]
    [InlineData(DftFlags.ComplexOutput, 1)]
    [InlineData(DftFlags.Inverse, 1)]
    [InlineData(DftFlags.Inverse | DftFlags.RealOutput, 2)]
    [InlineData(DftFlags.Rows | DftFlags.Scale, 2)]
    public void Dft_RoiWithStride_MatchesPackedMatrix(DftFlags flags, int channels)
    {
        Mat<float> parent = RandomFloat(40, 30, channels, 14);
        Mat<float> roi = parent.Roi(3, 4, 26, 18);
        Assert.False(roi.IsContinuous);
        Assert.Equal(SpectralTransforms.Dft(roi.Clone(), flags).ToArray(), SpectralTransforms.Dft(roi, flags).ToArray());
    }

    [Fact]
    public void Dct_RoiWithStride_MatchesPackedMatrix()
    {
        Mat<double> parent = RandomDouble(40, 30, 1, 15);
        Mat<double> roi = parent.Roi(5, 3, 24, 20);
        Assert.Equal(SpectralTransforms.Dct(roi.Clone()).ToArray(), SpectralTransforms.Dct(roi).ToArray());
        Assert.Equal(SpectralTransforms.Idct(roi.Clone(), DctFlags.Rows).ToArray(), SpectralTransforms.Idct(roi, DctFlags.Rows).ToArray());
    }

    /// <summary>A single row uses the scalar kernels; a batch of rows uses the vector lanes.</summary>
    [Theory]
    [InlineData(1, DftFlags.Rows)]
    [InlineData(1, DftFlags.Rows | DftFlags.Inverse | DftFlags.Scale)]
    [InlineData(1, DftFlags.Rows | DftFlags.ComplexOutput)]
    [InlineData(2, DftFlags.Rows)]
    [InlineData(2, DftFlags.Rows | DftFlags.Inverse | DftFlags.RealOutput)]
    public void Dft_RowBatch_MatchesSingleRows(int channels, DftFlags flags)
    {
        foreach (int width in new[] { 2, 7, 12, 30, 49, 64, 125 })
        {
            Mat<float> x = RandomFloat(width, 19, channels, width);
            Mat<float> all = SpectralTransforms.Dft(x, flags);
            for (int y = 0; y < x.Height; y++)
            {
                Mat<float> one = SpectralTransforms.Dft(x.Roi(0, y, width, 1).Clone(), flags);
                Assert.True(one.GetReadOnlyRowSpan(0).SequenceEqual(all.GetReadOnlyRowSpan(y)), $"width {width}, row {y}");
            }
        }
    }

    [Fact]
    public void Dct_RowBatch_MatchesSingleRows()
    {
        Mat<double> x = RandomDouble(36, 13, 1, 16);
        Mat<double> all = SpectralTransforms.Dct(x, DctFlags.Rows);
        for (int y = 0; y < x.Height; y++)
        {
            Mat<double> one = SpectralTransforms.Dct(x.Roi(0, y, 36, 1).Clone());
            Assert.True(one.GetReadOnlyRowSpan(0).SequenceEqual(all.GetReadOnlyRowSpan(y)));
        }
    }

    [Fact]
    public void Dct_Constant_HasOnlyDcTerm()
    {
        var x = new Mat<double>(8, 1);
        x.Fill(2);
        double[] y = SpectralTransforms.Dct(x).ToArray();
        Assert.Equal(2 * Math.Sqrt(8), y[0], 12);
        Assert.All(y.Skip(1), v => Assert.Equal(0, v, 12));
    }

    [Fact]
    public void Dct_RoundTrip_ReturnsInput()
    {
        Mat<float> x = RandomFloat(30, 18, 1, 17);
        AssertClose(x, SpectralTransforms.Idct(SpectralTransforms.Dct(x)), 2e-5);
    }

    [Fact]
    public void MulSpectrums_MatchesSpectrumOfCircularConvolution()
    {
        Mat<double> a = RandomDouble(12, 10, 1, 18);
        Mat<double> b = RandomDouble(12, 10, 1, 19);
        Mat<double> product = SpectralTransforms.MulSpectrums(SpectralTransforms.Dft(a), SpectralTransforms.Dft(b));
        Mat<double> conv = SpectralTransforms.Idft(product, DftFlags.Scale);
        for (int y = 0; y < 10; y++)
        {
            for (int x = 0; x < 12; x++)
            {
                double s = 0;
                for (int v = 0; v < 10; v++)
                {
                    for (int u = 0; u < 12; u++)
                    {
                        s += a.At(v, u) * b.At(((y - v) % 10 + 10) % 10, ((x - u) % 12 + 12) % 12);
                    }
                }

                Assert.Equal(s, conv.At(y, x), 9);
            }
        }
    }

    [Fact]
    public void Transforms_DoNotModifySourceAndAreDeterministic()
    {
        Mat<float> x = RandomFloat(33, 20, 1, 20);
        float[] before = x.ToArray();
        float[] first = SpectralTransforms.Dft(x, DftFlags.ComplexOutput).ToArray();
        float[] second = SpectralTransforms.Dft(x, DftFlags.ComplexOutput).ToArray();
        Assert.Equal(first, second);
        Assert.Equal(before, x.ToArray());

        Mat<float> c = SpectralTransforms.Dft(x, DftFlags.ComplexOutput);
        float[] cBefore = c.ToArray();
        SpectralTransforms.Idft(c, DftFlags.RealOutput);
        Assert.Equal(cBefore, c.ToArray());
    }

    [Fact]
    public void Transforms_ConcurrentCallsOnManySizes_MatchSequentialResults()
    {
        int[] sizes = Enumerable.Range(2, 60).ToArray();
        float[][] expected = sizes.Select(n => SpectralTransforms.Dft(RandomFloat(n, 3, 1, n)).ToArray()).ToArray();
        var actual = new float[sizes.Length][];
        Parallel.For(0, sizes.Length * 4, i =>
        {
            int k = i % sizes.Length;
            actual[k] = SpectralTransforms.Dft(RandomFloat(sizes[k], 3, 1, sizes[k])).ToArray();
        });
        for (int k = 0; k < sizes.Length; k++)
        {
            Assert.Equal(expected[k], actual[k]);
        }
    }

    [Fact]
    public void Dft_NonzeroRowsComplexRows_ZeroesRemainingRows()
    {
        Mat<float> x = RandomFloat(6, 5, 1, 21);
        Mat<float> y = SpectralTransforms.Dft(x, DftFlags.Rows | DftFlags.ComplexOutput, 2);
        Assert.All(Enumerable.Range(2, 3), r => Assert.All(y.GetReadOnlyRowSpan(r).ToArray(), v => Assert.Equal(0f, v)));
    }

    [Fact]
    public void GetOptimalDftSize_MatchesOpenCvEdges()
    {
        Assert.Equal(1, SpectralTransforms.GetOptimalDftSize(0));
        Assert.Equal(-1, SpectralTransforms.GetOptimalDftSize(-1));
        Assert.Equal(100, SpectralTransforms.GetOptimalDftSize(97));
        Assert.Equal(2125764000, SpectralTransforms.GetOptimalDftSize(2125763999));
        Assert.Equal(-1, SpectralTransforms.GetOptimalDftSize(2125764000));
    }

    [Fact]
    public void Transforms_InvalidInput_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => SpectralTransforms.Dft<float>(null!));
        Assert.Throws<NotSupportedException>(() => SpectralTransforms.Dft(new Mat<byte>(4, 4)));
        Assert.Throws<ArgumentException>(() => SpectralTransforms.Dft(new Mat<float>(4, 4, 3)));
        Assert.Throws<ArgumentException>(() => SpectralTransforms.Dft(new Mat<float>(4, 4), DftFlags.ComplexInput));
        Assert.Throws<NotSupportedException>(() => SpectralTransforms.Dft(new Mat<float>(1, 4), DftFlags.None, 2));
        Assert.Throws<NotSupportedException>(() => SpectralTransforms.Dct(new Mat<float>(7, 1)));
        Assert.Throws<NotSupportedException>(() => SpectralTransforms.Dct(new Mat<double>(4, 5)));
        Assert.Throws<ArgumentException>(() => SpectralTransforms.Dct(new Mat<float>(4, 4, 2)));
        Assert.Throws<NotSupportedException>(() => SpectralTransforms.Dct(new Mat<int>(4, 4)));
        Assert.Throws<ArgumentException>(() => SpectralTransforms.MulSpectrums(new Mat<float>(4, 4), new Mat<float>(4, 5)));
        Assert.Throws<ArgumentException>(() => SpectralTransforms.MulSpectrums(new Mat<float>(4, 4, 2), new Mat<float>(4, 4)));
    }

    private static void AssertClose<T>(Mat<T> expected, Mat<T> actual, double tolerance)
        where T : unmanaged
    {
        double[] e = expected.ToArray().Select(v => Convert.ToDouble(v)).ToArray();
        double[] a = actual.ToArray().Select(v => Convert.ToDouble(v)).ToArray();
        Assert.Equal(e.Length, a.Length);
        double scale = Math.Max(e.Max(Math.Abs), 1e-30);
        for (int i = 0; i < e.Length; i++)
        {
            Assert.True(Math.Abs(e[i] - a[i]) <= tolerance * scale, $"index {i}: {e[i]} vs {a[i]}");
        }
    }

    private static Mat<float> RandomFloat(int width, int height, int channels, int seed)
    {
        var m = new Mat<float>(width, height, channels);
        var r = new Random(seed);
        foreach (ref float v in m.GetSpan())
        {
            v = (float)((r.NextDouble() * 2) - 1);
        }

        return m;
    }

    private static Mat<double> RandomDouble(int width, int height, int channels, int seed)
    {
        var m = new Mat<double>(width, height, channels);
        var r = new Random(seed);
        foreach (ref double v in m.GetSpan())
        {
            v = (r.NextDouble() * 2) - 1;
        }

        return m;
    }
}
