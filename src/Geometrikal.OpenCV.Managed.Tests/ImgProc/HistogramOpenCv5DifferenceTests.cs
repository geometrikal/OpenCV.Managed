namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Collections.Generic;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

/// <summary>
/// Pins OpenCV 5.0 histogram behaviour where cv2 4.5.5 (the golden reference) differs or cannot
/// produce the result.
/// </summary>
public sealed class HistogramOpenCv5DifferenceTests
{
    public static IEnumerable<object[]> SuspectNames()
        => HistogramGoldenTests.Data.Cases.Where(c => c.Suspect).Select(c => new object[] { c.Name });

    /// <summary>
    /// compareHist CHISQR and CHISQR_ALT (OpenCV 5.0, PR 27334): blocks of four bins add
    /// <c>(a * a) * (1 / b)</c> into two double lanes. The golden file stores the value of a scalar
    /// model of this order in <c>values5</c>.
    /// </summary>
    [Theory]
    [MemberData(nameof(SuspectNames))]
    public void CompareHist_ChiSquare_UsesOpenCv5VectorOrder(string name)
    {
        HistogramGoldenTests.HistCase c = HistogramGoldenTests.Data.Cases.Single(x => x.Name == name);
        Assert.NotNull(c.Values5);
        double actual = HistogramGoldenTests.Run(c).Values![0];
        Assert.Equal(c.Values5![0], actual);
        Assert.NotEqual(c.Values![0], actual);
        Assert.Equal(c.Values[0], actual, 1e-6 * Math.Abs(c.Values[0]));
    }

    [Fact]
    public void CompareHist_ChiSquare_TermIsProductWithReciprocal()
    {
        var h1 = new DenseHistogram(4);
        var h2 = new DenseHistogram(4);
        new float[] { 3, 7, 11, 0.1f }.CopyTo(h1.Values);
        new float[] { 1, 2, 3, 0.3f }.CopyTo(h2.Values);
        double[] a = [3.0 - 1, 7.0 - 2, 11.0 - 3, 0.1f - (double)0.3f];
        double[] b = [3.0, 7.0, 11.0, 0.1f];
        double lane0 = 0 + ((a[0] * a[0]) * (1.0 / b[0])) + ((a[2] * a[2]) * (1.0 / b[2]));
        double lane1 = 0 + ((a[1] * a[1]) * (1.0 / b[1])) + ((a[3] * a[3]) * (1.0 / b[3]));
        Assert.Equal(lane0 + lane1, Histograms.CompareHist(h1, h2, HistCompMethods.Chisqr));
    }

    /// <summary>
    /// OpenCV 5.0 <c>calcBackProject</c> treats a two-dimensional histogram with one row or one
    /// column as one-dimensional; OpenCV 4.5.5 accepts only <c>N x 1</c>.
    /// </summary>
    [Fact]
    public void CalcBackProject_OneRowHistogram_EqualsColumnHistogram()
    {
        var rng = new Random(1);
        var src = new Mat<float>(53, 7);
        foreach (ref float v in src.GetSpan())
        {
            v = (float)rng.NextDouble();
        }

        var column = new DenseHistogram(9, 1);
        var row = new DenseHistogram(1, 9);
        var flat = new DenseHistogram(9);
        for (int i = 0; i < 9; i++)
        {
            column.Values[i] = row.Values[i] = flat.Values[i] = i * 1.5f;
        }

        float[][] ranges = [[0, 1]];
        float[] expected = Histograms.CalcBackProject([src], [0], flat, ranges, 2).ToArray();
        Assert.Equal(expected, Histograms.CalcBackProject([src], [0], column, ranges, 2).ToArray());
        Assert.Equal(expected, Histograms.CalcBackProject([src], [0], row, ranges, 2).ToArray());
        Assert.Equal(expected, Histograms.CalcBackProject([src], [0], DenseHistogram.FromMat(flat.ToMat()), ranges, 2).ToArray());
    }

    /// <summary>
    /// OpenCV 5.0 CLAHE bit shift (PR 28014). The golden cases <c>clahe_*_bitshift*_model</c> come
    /// from a scalar model; this test checks the shift rule on an image whose values are
    /// multiples of <c>2^shift</c>: the output is also a multiple of <c>2^shift</c>.
    /// </summary>
    [Fact]
    public void Clahe_BitShift_OutputIsMultipleOfShift()
    {
        Assert.Contains(HistogramGoldenTests.Data.Cases, c => c.Name.StartsWith("clahe_u8_bitshift", StringComparison.Ordinal) && c.Model);
        var rng = new Random(2);
        var src = new Mat<ushort>(45, 37);
        foreach (ref ushort v in src.GetSpan())
        {
            v = (ushort)(rng.Next(4096) << 4);
        }

        Mat<ushort> r = new Clahe(2.0, new Size(4, 3)) { BitShift = 4 }.Apply(src);
        Assert.All(r.ToArray(), v => Assert.Equal(0, v & 15));
        Assert.Contains(r.ToArray(), v => v != 0);
    }
}
