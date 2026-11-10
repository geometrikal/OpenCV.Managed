namespace Geometrikal.OpenCV.Managed.Tests.ImgProc;

using System;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;
using Geometrikal.OpenCV.Managed.Tests.Core;

/// <summary>Analytic and argument tests for <see cref="Corners"/>.</summary>
[Collection(nameof(CvParallelCollection))]
public sealed class CornersBasicTests
{
    private static Mat<byte> Square(int size, int x0, int y0, int side)
    {
        var m = new Mat<byte>(size, size);
        m.Fill(20);
        for (int y = y0; y < y0 + side; y++)
        {
            m.GetRowSpan(y).Slice(x0, side).Fill(220);
        }

        return m;
    }

    private static Mat<byte> Noise(int w, int h, int seed)
    {
        var m = new Mat<byte>(w, h);
        new Random(seed).NextBytes(m.GetSpan());
        return m;
    }

    [Fact]
    public void GoodFeaturesToTrack_Square_FindsFourCorners()
    {
        Point2f[] corners = Corners.GoodFeaturesToTrack(Square(40, 10, 12, 16), 10, 0.1, 5);
        Assert.Equal(4, corners.Length);
        Assert.All(corners, c => Assert.True(Math.Abs(c.X - 10) <= 2 || Math.Abs(c.X - 25) <= 2));
        Assert.All(corners, c => Assert.True(Math.Abs(c.Y - 12) <= 2 || Math.Abs(c.Y - 27) <= 2));
    }

    [Fact]
    public void GoodFeaturesToTrack_FlatImage_ReturnsEmpty()
    {
        var flat = new Mat<byte>(20, 20);
        flat.Fill(100);
        Assert.Empty(Corners.GoodFeaturesToTrack(flat, 10, 0.01, 1));
    }

    [Fact]
    public void CornerMinEigenVal_FlatImage_IsZero()
    {
        var flat = new Mat<float>(9, 7);
        flat.Fill(3f);
        Assert.All(Corners.CornerMinEigenVal(flat, 3).ToArray(), v => Assert.Equal(0f, v));
    }

    [Fact]
    public void CornerSubPix_SquareCorner_MovesTowardTrueCorner()
    {
        Mat<byte> img = Square(40, 10, 12, 16);
        Point2f[] pts = [new Point2f(11, 13)];
        Corners.CornerSubPix(img, pts, new Size(3, 3), new Size(-1, -1), new TermCriteria(TermCriteriaType.MaxIter | TermCriteriaType.Eps, 30, 0.001));
        Assert.True(Math.Abs(pts[0].X - 9.5f) < 1f && Math.Abs(pts[0].Y - 11.5f) < 1f, $"{pts[0]}");
    }

    [Fact]
    public void CornerSubPix_PointOutsideImage_Throws()
        => Assert.Throws<ArgumentOutOfRangeException>(() =>
            Corners.CornerSubPix(Square(20, 2, 2, 5), new[] { new Point2f(-1, 3) }, new Size(2, 2), new Size(-1, -1), new TermCriteria(TermCriteriaType.MaxIter, 5, 0)));

    [Fact]
    public void CornerHarris_UnsupportedKsize_Throws()
        => Assert.Throws<ArgumentOutOfRangeException>(() => Corners.CornerHarris(Noise(10, 10, 1), 3, 4, 0.04));

    [Fact]
    public void CornerHarris_MultiChannel_Throws()
        => Assert.Throws<NotSupportedException>(() => Corners.CornerHarris(new Mat<byte>(10, 10, 3), 3, 3, 0.04));

    [Fact]
    public void CornerHarris_UshortSource_Throws()
        => Assert.Throws<NotSupportedException>(() => Corners.CornerHarris(new Mat<ushort>(10, 10), 3, 3, 0.04));

    [Fact]
    public void GoodFeaturesToTrack_NonPositiveQuality_Throws()
        => Assert.Throws<ArgumentOutOfRangeException>(() => Corners.GoodFeaturesToTrack(Noise(10, 10, 1), 5, 0, 1));

    [Fact]
    public void CornerEigenValsAndVecs_RoiWithStride_MatchesPackedImage()
    {
        Mat<byte> roi = Noise(50, 40, 3).Roi(5, 4, 33, 27);
        Assert.Equal(
            Corners.CornerEigenValsAndVecs(roi.Clone(), 3, 3).ToArray(),
            Corners.CornerEigenValsAndVecs(roi, 3, 3).ToArray());
    }

    [Theory]
    [InlineData("minEig")]
    [InlineData("harris")]
    [InlineData("eigen")]
    [InlineData("gftt")]
    public void Output_DoesNotDependOnThreadCount(string op)
    {
        Mat<byte> img = Filters.GaussianBlur(Noise(517, 301, 4), new Size(5, 5), 1.5);
        Func<float[]> run = op switch
        {
            "minEig" => () => Corners.CornerMinEigenVal(img, 3).ToArray(),
            "harris" => () => Corners.CornerHarris(img, 2, 3, 0.04).ToArray(),
            "eigen" => () => Corners.CornerEigenValsAndVecs(img, 3, 3).ToArray(),
            _ => () => Corners.GoodFeaturesToTrack(img, 300, 0.01, 3).SelectMany(p => new[] { p.X, p.Y }).ToArray(),
        };

        int previous = CvParallel.MaxDegreeOfParallelism;
        try
        {
            CvParallel.MaxDegreeOfParallelism = 1;
            float[] serial = run();
            CvParallel.MaxDegreeOfParallelism = 8;
            Assert.Equal(serial, run());
        }
        finally
        {
            CvParallel.MaxDegreeOfParallelism = previous;
        }
    }
}
