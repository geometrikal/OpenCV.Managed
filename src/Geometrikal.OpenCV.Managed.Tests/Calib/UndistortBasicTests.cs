namespace Geometrikal.OpenCV.Managed.Tests.Calib;

using System;
using System.Linq;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Calib;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Tests.Core;

/// <summary>Behavioural tests for <see cref="Undistortion"/> and <see cref="CameraModel"/>.</summary>
[Collection(nameof(CvParallelCollection))]
public sealed class UndistortBasicTests
{
    private static readonly double[] K5 = [-0.2713, 0.1125, 0.00071, -0.00118, -0.0315];

    private static Mat<double> Camera(int w, int h)
        => new([w * 0.66, 0, (w - 1) * 0.5 + 2.5, 0, w * 0.665, (h - 1) * 0.5 - 1.5, 0, 0, 1], 3, 3);

    private static Mat<double> Centred(int w, int h)
        => new([w * 0.66, 0, (w - 1) * 0.5, 0, w * 0.66, (h - 1) * 0.5, 0, 0, 1], 3, 3);

    [Fact]
    public void InitUndistortRectifyMap_NoDistortion_IsIdentityMap()
    {
        Mat<double> a = Centred(37, 21);
        (Mat<float> mx, Mat<float> my) = Undistortion.InitUndistortRectifyMap(a, [], null, null, new Size(37, 21));
        for (int y = 0; y < 21; y++)
        {
            for (int x = 0; x < 37; x++)
            {
                Assert.Equal(x, mx.At(y, x), 3);
                Assert.Equal(y, my.At(y, x), 3);
            }
        }
    }

    [Fact]
    public void InitUndistortRectifyMap_MapKinds_AgreeWithinFixedPointPrecision()
    {
        Mat<double> a = Camera(53, 31);
        var size = new Size(53, 31);
        (Mat<float> mx, Mat<float> my) = Undistortion.InitUndistortRectifyMap(a, K5, null, null, size);
        Mat<float> m2 = Undistortion.InitUndistortRectifyMapInterleaved(a, K5, null, null, size);
        (Mat<short> xy, Mat<ushort> f) = Undistortion.InitUndistortRectifyMapFixedPoint(a, K5, null, null, size);
        for (int y = 0; y < size.Height; y++)
        {
            for (int x = 0; x < size.Width; x++)
            {
                Assert.Equal(mx.At(y, x), m2.At(y, x, 0));
                Assert.Equal(my.At(y, x), m2.At(y, x, 1));
                double fx = xy.At(y, x, 0) + (f.At(y, x) & 31) / 32.0;
                double fy = xy.At(y, x, 1) + (f.At(y, x) >> 5) / 32.0;
                Assert.InRange(Math.Abs(fx - mx.At(y, x)), 0, 1.0 / 64 + 1e-4);
                Assert.InRange(Math.Abs(fy - my.At(y, x)), 0, 1.0 / 64 + 1e-4);
            }
        }
    }

    [Fact]
    public void Undistort_NoDistortion_ReturnsSource()
    {
        Mat<byte> src = Random(41, 29, 3, 1);
        Mat<byte> dst = Undistortion.Undistort(src, Camera(41, 29), []);
        Assert.Equal(src.ToArray(), dst.ToArray());
    }

    [Fact]
    public void Undistort_SinglePixel_Works()
    {
        var src = new Mat<byte>(1, 1);
        src.At(0, 0) = 77;
        Mat<byte> dst = Undistortion.Undistort(src, Centred(1, 1), K5);
        Assert.Equal(77, dst.At(0, 0));
    }

    [Fact]
    public void Undistort_RoiWithStride_MatchesPackedImage()
    {
        Mat<byte> parent = Random(80, 60, 3, 2);
        Mat<byte> roi = parent.Roi(5, 7, 61, 43);
        Mat<byte> packed = roi.Clone();
        Mat<double> a = Camera(61, 43);
        Assert.Equal(Undistortion.Undistort(packed, a, K5).ToArray(), Undistortion.Undistort(roi, a, K5).ToArray());
    }

    [Fact]
    public void Undistort_DoesNotModifySource()
    {
        Mat<byte> src = Random(64, 40, 1, 3);
        byte[] before = src.ToArray();
        Undistortion.Undistort(src, Camera(64, 40), K5);
        Assert.Equal(before, src.ToArray());
    }

    [Fact]
    public void Undistort_EqualsStripeMapsAndRemap_ForOneStripe()
    {
        // One stripe covers the image when rows · cols ≤ 4096.
        Mat<byte> src = Random(64, 40, 1, 4);
        Mat<double> a = Camera(64, 40);
        (Mat<short> xy, Mat<ushort> f) = Undistortion.InitUndistortRectifyMapFixedPoint(a, K5, null, a, new Size(64, 40));
        Mat<byte> expected = Geometrikal.OpenCV.Managed.ImgProc.Geometry.Remap(src, xy, f, Geometrikal.OpenCV.Managed.ImgProc.InterpolationFlags.Linear);
        Assert.Equal(expected.ToArray(), Undistortion.Undistort(src, a, K5).ToArray());
    }

    [Fact]
    public void InvalidInputs_Throw()
    {
        Mat<double> a = Camera(10, 10);
        Assert.Throws<ArgumentException>(() => Undistortion.InitUndistortRectifyMap(a, [0.1, 0.2, 0.3], null, null, new Size(10, 10)));
        Assert.Throws<ArgumentException>(() => Undistortion.InitUndistortRectifyMap(new Mat<double>(2, 2), K5, null, null, new Size(10, 10)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Undistortion.InitUndistortRectifyMap(a, K5, null, null, new Size(0, 10)));
        Assert.Throws<ArgumentNullException>(() => Undistortion.Undistort<byte>(null!, a, K5));
        Assert.Throws<ArgumentException>(() => Undistortion.UndistortPoints(new Point2f[3], new Point2f[2], a, K5));
        Assert.Throws<ArgumentException>(() => Undistortion.UndistortPoints(new Point2f[1], a, K5, null, null, new TermCriteria(TermCriteriaType.MaxIter, 0, 0)));
        Assert.Throws<ArgumentException>(() => CameraModel.Rodrigues(new Mat<double>(2, 2)));
    }

    [Fact]
    public void UndistortPoints_InvertsProjectPoints()
    {
        Mat<double> a = Camera(684, 456);
        var obj = Enumerable.Range(0, 50).Select(i => new Point3d((i % 10) - 4.5, (i / 10) - 2.0, 0.3 * (i % 3))).ToArray();
        var rvec = new Point3d(0.02, -0.03, 0.01);
        var tvec = new Point3d(0.1, 0.2, 12.0);
        Point2d[] ip = CameraModel.ProjectPoints(obj, rvec, tvec, a, K5);
        Point2d[] ideal = CameraModel.ProjectPoints(obj, rvec, tvec, a, []);
        Point2d[] und = Undistortion.UndistortImagePoints(ip, a, K5, new TermCriteria(TermCriteriaType.MaxIter | TermCriteriaType.Eps, 50, 1e-9));
        for (int i = 0; i < obj.Length; i++)
        {
            Assert.InRange(Math.Abs(und[i].X - ideal[i].X), 0, 1e-4);
            Assert.InRange(Math.Abs(und[i].Y - ideal[i].Y), 0, 1e-4);
        }
    }

    [Fact]
    public void UndistortPoints_EpsOnlyCriteria_Terminates()
    {
        Mat<double> a = Camera(684, 456);
        Point2f[] pts = [new(10, 20), new(300, 200), new(650, 440)];
        Point2f[] und = Undistortion.UndistortPoints(pts, a, K5, null, a, new TermCriteria(TermCriteriaType.Eps, 0, 0.01));
        Assert.Equal(3, und.Length);
    }

    [Fact]
    public void UndistortPoints_InPlace_MatchesSeparateDestination()
    {
        Mat<double> a = Camera(684, 456);
        Point2d[] pts = Enumerable.Range(0, 20).Select(i => new Point2d(i * 31.5, i * 21.25)).ToArray();
        Point2d[] expected = Undistortion.UndistortPoints(pts, a, K5);
        Undistortion.UndistortPoints(pts, pts, a, K5);
        Assert.Equal(expected, pts);
    }

    [Fact]
    public void SolvePnP_PlanarGroundControl_RecoversPose()
    {
        Mat<double> a = Camera(684, 456);
        var obj = Enumerable.Range(0, 9).Select(i => new Point3d((i % 3) * 20.0 - 20, (i / 3) * 15.0 - 15, 0)).ToArray();
        var rvec = new Point3d(0.04, -0.03, 0.6);
        var tvec = new Point3d(2.0, -1.0, 110.0);
        Point2d[] img = CameraModel.ProjectPoints(obj, rvec, tvec, a, K5);
        Assert.True(CameraModel.SolvePnP(obj, img, a, K5, out Point3d r, out Point3d t));
        Assert.Equal(rvec.Z, r.Z, 6);
        Assert.Equal(tvec.Z, t.Z, 4);
        Assert.Equal(tvec.X, t.X, 4);
    }

    [Fact]
    public void SolvePnP_TooFewPoints_Throws()
    {
        Mat<double> a = Camera(684, 456);
        Point3d[] obj = [new(0, 0, 0), new(1, 0, 0), new(0, 1, 0)];
        Point2d[] img = [new(1, 1), new(2, 1), new(1, 2)];
        Assert.Throws<ArgumentException>(() => CameraModel.SolvePnP(obj, img, a, [], out _, out _));
    }

    [Fact]
    public void Rodrigues_RoundTrip()
    {
        var v = new Point3d(0.3, -1.1, 0.7);
        Point3d back = CameraModel.RodriguesVector(CameraModel.Rodrigues(v));
        Assert.Equal(v.X, back.X, 12);
        Assert.Equal(v.Y, back.Y, 12);
        Assert.Equal(v.Z, back.Z, 12);
    }

    [Fact]
    public void GetOptimalNewCameraMatrix_AlphaZero_RoiInsideImage()
    {
        Mat<double> a = Camera(684, 456);
        Mat<double> k = Undistortion.GetOptimalNewCameraMatrix(a, K5, new Size(684, 456), 0, default, out Rect roi);
        Assert.True(roi.X >= 0 && roi.Y >= 0 && roi.X + roi.Width <= 684 && roi.Y + roi.Height <= 456);
        Assert.True(k.At(0, 0) > 0);
    }

    [Fact]
    public void Output_DoesNotDependOnThreadCount()
    {
        Mat<double> a = Camera(1501, 1003);
        var size = new Size(1501, 1003);
        Mat<byte> src = Random(1501, 1003, 3, 5);
        Point2f[] pts = Enumerable.Range(0, 20000).Select(i => new Point2f(i % 1501, (i * 7) % 1003)).ToArray();
        int saved = CvParallel.MaxDegreeOfParallelism;
        try
        {
            CvParallel.MaxDegreeOfParallelism = 1;
            (Mat<float> mx1, Mat<float> my1) = Undistortion.InitUndistortRectifyMap(a, K5, null, null, size);
            (Mat<short> xy1, Mat<ushort> f1) = Undistortion.InitUndistortRectifyMapFixedPoint(a, K5, null, null, size);
            Mat<byte> d1 = Undistortion.Undistort(src, a, K5);
            Point2f[] p1 = Undistortion.UndistortPoints(pts, a, K5);
            CvParallel.MaxDegreeOfParallelism = 8;
            (Mat<float> mx8, Mat<float> my8) = Undistortion.InitUndistortRectifyMap(a, K5, null, null, size);
            (Mat<short> xy8, Mat<ushort> f8) = Undistortion.InitUndistortRectifyMapFixedPoint(a, K5, null, null, size);
            Mat<byte> d8 = Undistortion.Undistort(src, a, K5);
            Point2f[] p8 = Undistortion.UndistortPoints(pts, a, K5);
            Assert.Equal(mx1.ToArray(), mx8.ToArray());
            Assert.Equal(my1.ToArray(), my8.ToArray());
            Assert.Equal(xy1.ToArray(), xy8.ToArray());
            Assert.Equal(f1.ToArray(), f8.ToArray());
            Assert.Equal(d1.ToArray(), d8.ToArray());
            Assert.Equal(p1, p8);
        }
        finally
        {
            CvParallel.MaxDegreeOfParallelism = saved;
        }
    }

    private static Mat<byte> Random(int w, int h, int cn, int seed)
    {
        var rng = new Random(seed);
        var m = new Mat<byte>(w, h, cn);
        rng.NextBytes(MemoryMarshal.AsBytes(m.GetSpan()));
        return m;
    }
}
