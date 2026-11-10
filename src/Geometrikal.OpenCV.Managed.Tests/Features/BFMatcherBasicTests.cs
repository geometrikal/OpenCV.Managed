namespace Geometrikal.OpenCV.Managed.Tests.Features;

using System;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;
using Geometrikal.OpenCV.Managed.Features;
using Geometrikal.OpenCV.Managed.Tests.Core;

/// <summary>Analytic and argument tests for <see cref="BFMatcher"/>.</summary>
[Collection(nameof(CvParallelCollection))]
public sealed class BFMatcherBasicTests
{
    private static Mat<byte> Bytes(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var m = new Mat<byte>(cols, rows);
        rng.NextBytes(m.GetSpan());
        return m;
    }

    private static Mat<float> Floats(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var m = new Mat<float>(cols, rows);
        Span<float> s = m.GetSpan();
        for (int i = 0; i < s.Length; i++)
        {
            s[i] = (float)((rng.NextDouble() - 0.5) * 20);
        }

        return m;
    }

    [Fact]
    public void Match_Hamming_ReturnsBitCountDistance()
    {
        var q = new Mat<byte>([0b1011, 0], 2, 1);
        var t = new Mat<byte>([0, 0, 0b1011, 0b1], 2, 2);
        DMatch[] m = new BFMatcher(NormType.Hamming).Match(q, t);
        Assert.Equal(new DMatch(0, 1, 0, 1f), m[0]);
    }

    [Fact]
    public void Match_Hamming2_CountsNonZeroBitPairs()
    {
        var q = new Mat<byte>([0b1111_0011], 1, 1);
        var t = new Mat<byte>([0], 1, 1);
        Assert.Equal(3f, new BFMatcher(NormType.Hamming2).Match(q, t)[0].Distance);
    }

    [Fact]
    public void KnnMatch_EqualDistances_KeepLowerTrainIndexFirst()
    {
        var q = new Mat<float>([0f, 0f], 2, 1);
        var t = new Mat<float>([1f, 0f, 0f, 1f, -1f, 0f], 2, 3);
        DMatch[][] m = new BFMatcher(NormType.L2).KnnMatch(q, t, 3);
        Assert.Equal(new[] { 0, 1, 2 }, m[0].Select(x => x.TrainIdx));
    }

    [Fact]
    public void KnnMatch_KGreaterThanTrainCount_ReturnsAllTrainRows()
    {
        DMatch[][] m = new BFMatcher(NormType.L1).KnnMatch(Floats(4, 8, 1), Floats(3, 8, 2), 10);
        Assert.All(m, row => Assert.Equal(3, row.Length));
    }

    [Fact]
    public void KnnMatch_MaskedRowWithCompact_IsRemoved()
    {
        Mat<byte> q = Bytes(3, 16, 3), t = Bytes(5, 16, 4);
        var mask = new Mat<byte>(5, 3);
        mask.Fill(1);
        mask.GetRowSpan(1).Clear();
        var matcher = new BFMatcher(NormType.Hamming);
        Assert.Equal(2, matcher.KnnMatch(q, t, 2, mask, compactResult: true).Length);
        DMatch[][] full = matcher.KnnMatch(q, t, 2, mask);
        Assert.Equal(3, full.Length);
        Assert.Empty(full[1]);
    }

    [Fact]
    public void RadiusMatch_Compact_PutsEmptyListsLast()
    {
        var q = new Mat<byte>([0, 255], 1, 2);
        var t = new Mat<byte>([255, 254], 1, 2);
        DMatch[][] m = new BFMatcher(NormType.Hamming).RadiusMatch(q, t, 1f, compactResult: true);
        Assert.Equal(2, m.Length);
        Assert.Equal(1, m[0][0].QueryIdx);
        Assert.Empty(m[1]);
    }

    [Fact]
    public void CrossCheck_KnnWithKTwo_Throws()
    {
        var matcher = new BFMatcher(NormType.L2, crossCheck: true);
        Assert.Throws<NotSupportedException>(() => matcher.KnnMatch(Floats(3, 4, 1), Floats(3, 4, 2), 2));
    }

    [Fact]
    public void CrossCheck_KeepsOnlyMutualNearest()
    {
        var q = new Mat<float>([0f, 10f, 11f], 1, 3);
        var t = new Mat<float>([0.5f, 10.4f], 1, 2);
        DMatch[] m = new BFMatcher(NormType.L1, crossCheck: true).Match(q, t);
        Assert.Equal(new[] { (0, 0), (1, 1) }, m.Select(x => (x.QueryIdx, x.TrainIdx)));
    }

    [Fact]
    public void Constructor_UnsupportedNorm_Throws()
        => Assert.Throws<ArgumentOutOfRangeException>(() => new BFMatcher(NormType.Inf));

    [Fact]
    public void Match_HammingOnFloat_Throws()
        => Assert.Throws<NotSupportedException>(() => new BFMatcher(NormType.Hamming).Match(Floats(2, 4, 1), Floats(2, 4, 2)));

    [Fact]
    public void Match_WidthMismatch_Throws()
        => Assert.Throws<ArgumentException>(() => new BFMatcher(NormType.L2).Match(Floats(2, 4, 1), Floats(2, 5, 2)));

    [Fact]
    public void Match_MaskWrongSize_Throws()
        => Assert.Throws<ArgumentException>(() => new BFMatcher(NormType.L2).Match(Floats(2, 4, 1), Floats(3, 4, 2), new Mat<byte>(2, 2)));

    [Fact]
    public void RadiusMatch_TinyRadius_Throws()
        => Assert.Throws<ArgumentOutOfRangeException>(() => new BFMatcher(NormType.L2).RadiusMatch(Floats(2, 4, 1), Floats(2, 4, 2), 1e-8f));

    [Fact]
    public void Match_EmptyCollection_ReturnsEmpty()
        => Assert.Empty(new BFMatcher(NormType.L2).Match(Floats(2, 4, 1)));

    [Fact]
    public void Add_DifferentElementType_Throws()
    {
        var matcher = new BFMatcher(NormType.L1);
        matcher.Add(Floats(2, 4, 1));
        Assert.Throws<ArgumentException>(() => matcher.Add(Bytes(2, 4, 1)));
    }

    [Fact]
    public void KnnMatch_RoiWithStride_MatchesPackedInput()
    {
        Mat<byte> parentQ = Bytes(20, 40, 5), parentT = Bytes(30, 50, 6);
        Mat<byte> q = parentQ.Roi(3, 2, 32, 15), t = parentT.Roi(7, 1, 32, 25);
        var matcher = new BFMatcher(NormType.Hamming);
        DMatch[][] a = matcher.KnnMatch(q, t, 3);
        DMatch[][] b = matcher.KnnMatch(q.Clone(), t.Clone(), 3);
        Assert.Equal(b.SelectMany(x => x), a.SelectMany(x => x));
    }

    [Theory]
    [InlineData(NormType.Hamming, 32)]
    [InlineData(NormType.Hamming2, 61)]
    [InlineData(NormType.L1, 64)]
    [InlineData(NormType.L2, 128)]
    public void KnnMatch_Output_DoesNotDependOnThreadCount(NormType norm, int len)
    {
        int previous = CvParallel.MaxDegreeOfParallelism;
        try
        {
            Func<DMatch[][]> run = norm is NormType.Hamming or NormType.Hamming2
                ? () => new BFMatcher(norm).KnnMatch(Bytes(600, len, 7), Bytes(700, len, 8), 2)
                : () => new BFMatcher(norm).KnnMatch(Floats(600, len, 7), Floats(700, len, 8), 2);
            CvParallel.MaxDegreeOfParallelism = 1;
            DMatch[][] serial = run();
            CvParallel.MaxDegreeOfParallelism = 8;
            DMatch[][] parallel = run();
            Assert.Equal(serial.SelectMany(x => x), parallel.SelectMany(x => x));
        }
        finally
        {
            CvParallel.MaxDegreeOfParallelism = previous;
        }
    }

    [Fact]
    public void MsvcSort_LargeRange_IsSorted()
    {
        var rng = new Random(9);
        int[] values = Enumerable.Range(0, 5000).Select(_ => rng.Next(50)).ToArray();
        MsvcSort.Sort(values.AsSpan(), default(IntLess));
        Assert.Equal(values.OrderBy(v => v), values);
    }

    [Fact]
    public void MsvcSort_HeapFallback_IsSorted()
    {
        // Organ-pipe input drives the median guesses to unbalanced splits.
        int[] values = Enumerable.Range(0, 3000).Select(i => i < 1500 ? i : 3000 - i).ToArray();
        MsvcSort.Sort(values.AsSpan(), default(IntLess));
        Assert.Equal(values.OrderBy(v => v), values);
    }

    private readonly struct IntLess : ILess<int>
    {
        public bool Less(in int a, in int b) => a < b;
    }
}
