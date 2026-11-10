namespace Geometrikal.OpenCV.Managed.Tests.Features;

using System;
using System.Collections.Generic;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Features;
using Geometrikal.OpenCV.Managed.Tests.ImgProc;

/// <summary>
/// <see cref="KeyPoint"/>, <see cref="DMatch"/> and <see cref="KeyPointsFilter"/>.
/// <c>KeyPoint.overlap</c> and <c>KeyPoint.convert</c> compare with cv2 5.0.0 results in
/// <c>TestData/pyramid_golden.json</c>; cv2 has no <c>KeyPointsFilter</c> binding, so the filters
/// have analytic tests.
/// </summary>
public sealed class KeyPointTests
{
    [Fact]
    public void Overlap_MatchesOpenCvGoldenBitExact()
    {
        List<List<uint>> rows = PyramidGoldenTests.Data.Keypoints.Overlap;
        Assert.True(rows.Count > 300);
        foreach (List<uint> r in rows)
        {
            float F(int i) => BitConverter.UInt32BitsToSingle(r[i]);
            var k1 = new KeyPoint(F(0), F(1), F(2));
            var k2 = new KeyPoint(F(3), F(4), F(5));
            uint actual = BitConverter.SingleToUInt32Bits(KeyPoint.Overlap(k1, k2));
            Assert.True(r[6] == actual, $"overlap({k1}, {k2}): expected {F(6):R}, got {BitConverter.UInt32BitsToSingle(actual):R}");
        }
    }

    [Fact]
    public void Overlap_SpecialCases()
    {
        Assert.Equal(1f, KeyPoint.Overlap(new KeyPoint(1, 2, 6), new KeyPoint(1, 2, 6)));
        Assert.Equal(0f, KeyPoint.Overlap(new KeyPoint(0, 0, 2), new KeyPoint(10, 0, 2)));
        Assert.Equal(0.25f, KeyPoint.Overlap(new KeyPoint(0, 0, 8), new KeyPoint(1, 0, 4)));
    }

    [Fact]
    public void Convert_MatchesOpenCvGolden()
    {
        PyramidGoldenTests.ConvertData golden = PyramidGoldenTests.Data.Keypoints.Convert;
        KeyPoint[] kps = Enumerable.Range(0, 6).Select(i => new KeyPoint(i * 1.5f, -i * 0.25f, 3f, 10f * i, 0.5f, i, i + 1)).ToArray();
        Point2f[] pts = KeyPoint.Convert(kps, [4, 0, 4, 2]);
        Assert.Equal(golden.Indexed.Select(p => new Point2f(p[0], p[1])), pts);

        KeyPoint[] back = KeyPoint.Convert([new Point2f(1.25f, 2.5f), new Point2f(-3f, 4.75f)], 2f, 0.75f, 3, 9);
        Assert.Equal(
            golden.Reverse.Select(r => new KeyPoint(new Point2f((float)r[0], (float)r[1]), (float)r[2], (float)r[3], (float)r[4], (int)r[5], (int)r[6])),
            back);
        Assert.Equal(kps.Select(k => k.Pt), KeyPoint.Convert(kps));
        Assert.Equal(kps.Select(k => k.Pt), KeyPoint.Convert(kps, []));
    }

    [Fact]
    public void Convert_NegativeIndex_Throws()
    {
        Assert.Throws<ArgumentException>(() => KeyPoint.Convert([new KeyPoint(0, 0, 1)], [-1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => KeyPoint.Convert([new KeyPoint(0, 0, 1)], [1]));
    }

    [Fact]
    public void Defaults_MatchOpenCv()
    {
        var k = new KeyPoint();
        Assert.Equal((0f, 0f, 0f, -1f, 0f, 0, -1), (k.Pt.X, k.Pt.Y, k.Size, k.Angle, k.Response, k.Octave, k.ClassId));
        var m = new DMatch();
        Assert.Equal((-1, -1, -1, float.MaxValue), (m.QueryIdx, m.TrainIdx, m.ImgIdx, m.Distance));
        Assert.Equal(-1, new DMatch(3, 4, 0.5f).ImgIdx);
    }

    [Fact]
    public void DMatch_OrdersByDistance()
    {
        var a = new DMatch(0, 1, 2, 0.5f);
        var b = new DMatch(9, 9, 9, 0.25f);
        Assert.True(b < a);
        Assert.False(a < b);
        Assert.True(a > b);
        Assert.True(a <= new DMatch(5, 5, 0.5f));
        Assert.True(a >= new DMatch(5, 5, 0.5f));
        DMatch[] sorted = new[] { a, b, new DMatch(1, 1, 1f) }.OrderBy(x => x).ToArray();
        Assert.Equal(new[] { 0.25f, 0.5f, 1f }, sorted.Select(x => x.Distance));
    }

    [Fact]
    public void RetainBest_KeepsTopResponsesAndBoundaryTies()
    {
        float[] responses = [3, 9, 5, 5, 1, 7, 5, 2];
        List<KeyPoint> kps = responses.Select((r, i) => new KeyPoint(i, 0, 1, -1, r)).ToList();
        KeyPointsFilter.RetainBest(kps, 3);

        // Top three are 9, 7, 5; the other two keypoints with response 5 are boundary ties.
        Assert.Equal(new[] { 9f, 7f, 5f, 5f, 5f }, kps.Select(k => k.Response));
        Assert.Equal(new[] { 1f, 5f, 2f, 3f, 6f }, kps.Select(k => k.Pt.X));
    }

    [Fact]
    public void RetainBest_EdgeCounts()
    {
        List<KeyPoint> Make() => Enumerable.Range(0, 4).Select(i => new KeyPoint(i, 0, 1, -1, i)).ToList();
        List<KeyPoint> kps = Make();
        KeyPointsFilter.RetainBest(kps, -1);
        Assert.Equal(Make(), kps);
        KeyPointsFilter.RetainBest(kps, 4);
        Assert.Equal(Make(), kps);
        KeyPointsFilter.RetainBest(kps, 1);
        Assert.Equal(3f, Assert.Single(kps).Response);
        KeyPointsFilter.RetainBest(kps, 0);
        Assert.Empty(kps);
    }

    [Fact]
    public void RetainBest_NaNResponsesRankLast()
    {
        List<KeyPoint> kps = new[] { float.NaN, 1f, 2f }.Select((r, i) => new KeyPoint(i, 0, 1, -1, r)).ToList();
        KeyPointsFilter.RetainBest(kps, 2);
        Assert.Equal(new[] { 2f, 1f }, kps.Select(k => k.Response));
    }

    [Fact]
    public void RunByImageBorder_UsesCvRoundAndHalfOpenRect()
    {
        float[] xs = [1.4f, 1.5f, 2.5f, 7.4f, 7.5f, 8.5f, 5f];
        List<KeyPoint> kps = xs.Select(x => new KeyPoint(x, 5, 1)).ToList();
        KeyPointsFilter.RunByImageBorder(kps, new Size(10, 10), 2);

        // cvRound: 1.5 → 2 (kept), 2.5 → 2 (kept), 7.4 → 7 (kept), 7.5 → 8 (removed), 8.5 → 8 (removed).
        Assert.Equal(new[] { 1.5f, 2.5f, 7.4f, 5f }, kps.Select(k => k.Pt.X));

        kps = xs.Select(x => new KeyPoint(x, 5, 1)).ToList();
        KeyPointsFilter.RunByImageBorder(kps, new Size(10, 10), 0);
        Assert.Equal(xs.Length, kps.Count);
        KeyPointsFilter.RunByImageBorder(kps, new Size(4, 10), 2);
        Assert.Empty(kps);
    }

    [Fact]
    public void RunByKeypointSize_KeepsClosedRange()
    {
        List<KeyPoint> kps = new[] { 0.5f, 1f, 2f, 3f, 3.5f }.Select(s => new KeyPoint(0, 0, s)).ToList();
        KeyPointsFilter.RunByKeypointSize(kps, 1f, 3f);
        Assert.Equal(new[] { 1f, 2f, 3f }, kps.Select(k => k.Size));
        Assert.Throws<ArgumentOutOfRangeException>(() => KeyPointsFilter.RunByKeypointSize(kps, -1f, 3f));
        Assert.Throws<ArgumentOutOfRangeException>(() => KeyPointsFilter.RunByKeypointSize(kps, 4f, 3f));
    }

    [Fact]
    public void RunByPixelsMask_RemovesZeroPixels()
    {
        var mask = new Mat<byte>(4, 3);
        mask.Fill(1);
        mask.At(1, 2) = 0;
        List<KeyPoint> kps = [new(1.6f, 0.6f, 1), new(2.4f, 1.4f, 1), new(-0.4f, 0f, 1), new(3.4f, 2.4f, 1)];
        KeyPointsFilter.RunByPixelsMask(kps, mask);

        // (1.6, 0.6) → pixel (x 2, y 1) is zero; (2.4, 1.4) → (2, 1) is zero.
        Assert.Equal(new[] { -0.4f, 3.4f }, kps.Select(k => k.Pt.X));
        KeyPointsFilter.RunByPixelsMask(kps, null);
        Assert.Equal(2, kps.Count);
        Assert.Throws<ArgumentOutOfRangeException>(() => KeyPointsFilter.RunByPixelsMask([new KeyPoint(3.6f, 0, 1)], mask));
        Assert.Throws<ArgumentException>(() => KeyPointsFilter.RunByPixelsMask(kps, new Mat<byte>(4, 3, 2)));
    }

    [Fact]
    public void RemoveDuplicated_KeepsInputOrderAndBestDuplicate()
    {
        List<KeyPoint> kps =
        [
            new(5, 5, 2, 10, 0.1f),
            new(1, 1, 3),
            new(5, 5, 2, 10, 0.9f),
            new(5, 5, 2, 11, 0.5f),
            new(1, 1, 3),
        ];
        KeyPointsFilter.RemoveDuplicated(kps);

        // The duplicate with the larger response stays; of exact copies the first stays.
        Assert.Equal(new[] { (1f, -1f, 0f), (5f, 10f, 0.9f), (5f, 11f, 0.5f) }, kps.Select(k => (k.Pt.X, k.Angle, k.Response)));
    }

    [Fact]
    public void RemoveDuplicatedSorted_SortsAndRemoves()
    {
        List<KeyPoint> kps =
        [
            new(5, 5, 2, 10, 0.1f),
            new(1, 2, 3),
            new(1, 2, 4),
            new(5, 5, 2, 10, 0.9f),
            new(0, 9, 1),
        ];
        KeyPointsFilter.RemoveDuplicatedSorted(kps);
        Assert.Equal(
            new[] { (0f, 9f, 1f, 0f), (1f, 2f, 4f, 0f), (1f, 2f, 3f, 0f), (5f, 5f, 2f, 0.9f) },
            kps.Select(k => (k.Pt.X, k.Pt.Y, k.Size, k.Response)));
    }
}
