namespace Geometrikal.OpenCV.Managed.Tests.Video;

using System;
using System.Linq;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Tests.Core;
using Geometrikal.OpenCV.Managed.Video;

/// <summary>Analytic and API tests of <see cref="OpticalFlow"/>.</summary>
[Collection(nameof(CvParallelCollection))]
public sealed class OpticalFlowBasicTests
{
    private static Mat<byte> Texture(int w, int h, double dx = 0, double dy = 0, int cn = 1)
    {
        var m = new Mat<byte>(w, h, cn);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                double u = x - dx, v = y - dy;
                for (int c = 0; c < cn; c++)
                {
                    double val = 128 + (50 * Math.Sin((u / 4.3) + c)) + (40 * Math.Cos((v / 5.1) - (0.5 * c))) + (30 * Math.Sin((u + v) / 3.7));
                    m.At(y, x, c) = (byte)Math.Clamp(Math.Round(val), 0, 255);
                }
            }
        }

        return m;
    }

    private static Point2f[] Grid(int w, int h, int margin, int step)
    {
        var pts = new System.Collections.Generic.List<Point2f>();
        for (int y = margin; y < h - margin; y += step)
        {
            for (int x = margin; x < w - margin; x += step)
            {
                pts.Add(new Point2f(x + 0.3f, y + 0.6f));
            }
        }

        return pts.ToArray();
    }

    [Fact]
    public void CalcOpticalFlowPyrLK_KnownShift_RecoversShift()
    {
        Mat<byte> prev = Texture(160, 120);
        Mat<byte> next = Texture(160, 120, 3.4, -2.2);
        Point2f[] pts = Grid(160, 120, 30, 10);
        OpticalFlowResult r = OpticalFlow.CalcOpticalFlowPyrLK(prev, next, pts);
        for (int i = 0; i < pts.Length; i++)
        {
            Assert.Equal(1, r.Status[i]);
            Assert.InRange(r.NextPoints[i].X - pts[i].X, 3.2f, 3.6f);
            Assert.InRange(r.NextPoints[i].Y - pts[i].Y, -2.4f, -2.0f);
        }
    }

    [Fact]
    public void CalcOpticalFlowPyrLK_ManyThreads_MatchesOneThread()
    {
        Mat<byte> prev = Texture(200, 150, cn: 3);
        Mat<byte> next = Texture(200, 150, 1.7, 2.9, cn: 3);
        Point2f[] pts = Grid(200, 150, 0, 4);
        int previous = CvParallel.MaxDegreeOfParallelism;
        try
        {
            CvParallel.MaxDegreeOfParallelism = 1;
            OpticalFlowResult one = OpticalFlow.CalcOpticalFlowPyrLK(prev, next, pts);
            OpticalFlowPyramid p1 = OpticalFlow.BuildOpticalFlowPyramid(prev, new Size(21, 21), 3);
            CvParallel.MaxDegreeOfParallelism = 8;
            OpticalFlowResult many = OpticalFlow.CalcOpticalFlowPyrLK(prev, next, pts);
            OpticalFlowPyramid p8 = OpticalFlow.BuildOpticalFlowPyramid(prev, new Size(21, 21), 3);
            Assert.Equal(one.NextPoints, many.NextPoints);
            Assert.Equal(one.Status, many.Status);
            Assert.Equal(one.Error, many.Error);
            for (int l = 0; l <= p1.MaxLevel; l++)
            {
                Assert.Equal(p1.Derivatives![l].ToArray(), p8.Derivatives![l].ToArray());
            }
        }
        finally
        {
            CvParallel.MaxDegreeOfParallelism = previous;
        }
    }

    [Fact]
    public void CalcOpticalFlowPyrLK_PyramidInput_MatchesImageInput()
    {
        Mat<byte> prev = Texture(120, 90);
        Mat<byte> next = Texture(120, 90, -2.5, 1.25);
        Point2f[] pts = Grid(120, 90, 0, 7);
        var size = new Size(15, 15);
        OpticalFlowResult direct = OpticalFlow.CalcOpticalFlowPyrLK(prev, next, pts, null, size, 2, OpticalFlow.DefaultCriteria);

        foreach (bool withDeriv in new[] { true, false })
        {
            OpticalFlowPyramid pp = OpticalFlow.BuildOpticalFlowPyramid(prev, size, 2, withDeriv);
            OpticalFlowPyramid np = OpticalFlow.BuildOpticalFlowPyramid(next, size, 2, withDeriv);
            var nextPts = new Point2f[pts.Length];
            var status = new byte[pts.Length];
            var err = new float[pts.Length];
            OpticalFlow.CalcOpticalFlowPyrLK(pp, np, pts, nextPts, status, err, size, 5, OpticalFlow.DefaultCriteria);
            Assert.Equal(direct.NextPoints, nextPts);
            Assert.Equal(direct.Status, status);
            Assert.Equal(direct.Error, err);
        }
    }

    [Fact]
    public void CalcOpticalFlowPyrLK_PyramidWithSmallPadding_Throws()
    {
        Mat<byte> img = Texture(64, 48);
        OpticalFlowPyramid pyr = OpticalFlow.BuildOpticalFlowPyramid(img, new Size(5, 5), 1);
        var pts = new[] { new Point2f(10, 10) };
        Assert.Throws<ArgumentException>(() => OpticalFlow.CalcOpticalFlowPyrLK(
            pyr, pyr, pts, new Point2f[1], new byte[1], null, new Size(9, 9), 1, OpticalFlow.DefaultCriteria));
    }

    [Fact]
    public void CalcOpticalFlowPyrLK_RoiInput_UsesParentPixelsAsLevel0Border()
    {
        Mat<byte> big = Texture(140, 110);
        Mat<byte> bigNext = Texture(140, 110, 1.5, 0.75);
        Mat<byte> roi = big.Roi(30, 25, 80, 60);
        Mat<byte> roiNext = bigNext.Roi(30, 25, 80, 60);
        var pts = new[] { new Point2f(1.3f, 1.6f), new Point2f(78.5f, 58.5f), new Point2f(40, 30) };

        OpticalFlowResult viaRoi = OpticalFlow.CalcOpticalFlowPyrLK(roi, roiNext, pts, null, new Size(21, 21), 0, OpticalFlow.DefaultCriteria);
        OpticalFlowResult viaCopy = OpticalFlow.CalcOpticalFlowPyrLK(roi.Clone(), roiNext.Clone(), pts, null, new Size(21, 21), 0, OpticalFlow.DefaultCriteria);
        var shifted = pts.Select(p => new Point2f(p.X + 30, p.Y + 25)).ToArray();
        OpticalFlowResult viaBig = OpticalFlow.CalcOpticalFlowPyrLK(big, bigNext, shifted, null, new Size(21, 21), 0, OpticalFlow.DefaultCriteria);

        Assert.Equal(viaRoi.NextPoints[2], viaCopy.NextPoints[2]);
        Assert.NotEqual(viaRoi.NextPoints[0], viaCopy.NextPoints[0]);
        Assert.Equal(new Point2f(viaBig.NextPoints[2].X - 30, viaBig.NextPoints[2].Y - 25), viaRoi.NextPoints[2]);
    }

    [Fact]
    public void BuildOpticalFlowPyramid_RoiWithMargin_ReusesInput()
    {
        Mat<byte> big = Texture(100, 80);
        Mat<byte> roi = big.Roi(21, 21, 50, 30);
        OpticalFlowPyramid reused = OpticalFlow.BuildOpticalFlowPyramid(roi, new Size(21, 21), 0);
        OpticalFlowPyramid copied = OpticalFlow.BuildOpticalFlowPyramid(roi, new Size(21, 21), 0, tryReuseInputImage: false);
        Assert.Same(roi, reused.Images[0]);
        Assert.NotSame(roi, copied.Images[0]);
        Assert.Equal(roi.ToArray(), copied.Images[0].ToArray());
    }

    [Fact]
    public void BuildOpticalFlowPyramid_SmallImage_StopsBeforeWindowSize()
    {
        OpticalFlowPyramid pyr = OpticalFlow.BuildOpticalFlowPyramid(Texture(40, 30), new Size(9, 9), 5);
        Assert.Equal(1, pyr.MaxLevel);
        Assert.Equal(new[] { 40, 20 }, pyr.Images.Select(m => m.Width));
        Assert.Equal(2, pyr.Derivatives![0].Channels);
    }

    [Fact]
    public void CalcOpticalFlowPyrLK_EmptyPoints_ReturnsEmpty()
    {
        Mat<byte> img = Texture(32, 32);
        OpticalFlowResult r = OpticalFlow.CalcOpticalFlowPyrLK(img, img, []);
        Assert.Empty(r.NextPoints);
        Assert.Empty(r.Status);
    }

    [Fact]
    public void CalcOpticalFlowPyrLK_InvalidArguments_Throw()
    {
        Mat<byte> img = Texture(32, 32);
        var pts = new[] { new Point2f(5, 5) };
        Assert.Throws<ArgumentOutOfRangeException>(() => OpticalFlow.CalcOpticalFlowPyrLK(img, img, pts, null, new Size(2, 5), 1, OpticalFlow.DefaultCriteria));
        Assert.Throws<ArgumentOutOfRangeException>(() => OpticalFlow.CalcOpticalFlowPyrLK(img, img, pts, null, new Size(5, 5), -1, OpticalFlow.DefaultCriteria));
        Assert.Throws<ArgumentNullException>(() => OpticalFlow.CalcOpticalFlowPyrLK(img, img, pts, null, OpticalFlowFlags.UseInitialFlow));
        Assert.Throws<ArgumentException>(() => OpticalFlow.CalcOpticalFlowPyrLK(img, Texture(33, 32), pts));
        Assert.Throws<ArgumentException>(() => OpticalFlow.CalcOpticalFlowPyrLK(img, img, pts, new Point2f[2], new byte[1], null, new Size(5, 5), 0, OpticalFlow.DefaultCriteria));
    }

    [Fact]
    public void CalcOpticalFlowPyrLK_FlatImage_RejectsAllPoints()
    {
        var flat = new Mat<byte>(40, 40);
        flat.Fill(90);
        OpticalFlowResult r = OpticalFlow.CalcOpticalFlowPyrLK(flat, flat, Grid(40, 40, 5, 6), null, OpticalFlowFlags.LkGetMinEigenvals);
        Assert.All(r.Status, s => Assert.Equal(0, s));
        Assert.All(r.Error, e => Assert.Equal(0f, e));
    }
}
