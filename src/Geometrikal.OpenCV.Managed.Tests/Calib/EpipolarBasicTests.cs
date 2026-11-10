namespace Geometrikal.OpenCV.Managed.Tests.Calib;

using System;
using System.Linq;
using Geometrikal.OpenCV.Managed.Calib;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>Behavioural tests of <see cref="EpipolarGeometry"/> on exact synthetic two-view scenes.</summary>
public sealed class EpipolarBasicTests
{
    private static readonly Mat<double> K = new(new double[] { 800, 0, 640, 0, 800, 360, 0, 0, 1 }, 3, 3);

    private static (Point2d[] P1, Point2d[] P2, double[] R, double[] T, double[] X) Scene(int n, bool planar = false, int seed = 1)
    {
        var rng = new Random(seed);
        double a = 0.1, b = -0.05;
        double ca = Math.Cos(a), sa = Math.Sin(a), cb = Math.Cos(b), sb = Math.Sin(b);
        double[] rz = [ca, -sa, 0, sa, ca, 0, 0, 0, 1];
        double[] ry = [cb, 0, sb, 0, 1, 0, -sb, 0, cb];
        double[] r = new double[9];
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                for (int k = 0; k < 3; k++)
                {
                    r[(i * 3) + j] += rz[(i * 3) + k] * ry[(k * 3) + j];
                }
            }
        }

        double[] t = [-1.0, 0.2, 0.1];
        var p1 = new Point2d[n];
        var p2 = new Point2d[n];
        double[] xs = new double[3 * n];
        for (int i = 0; i < n; i++)
        {
            double z = planar ? 10 : rng.NextDouble() * 10 + 5;
            double x = (rng.NextDouble() - 0.5) * z, y = (rng.NextDouble() - 0.5) * z * 0.6;
            double x2 = (r[0] * x) + (r[1] * y) + (r[2] * z) + t[0];
            double y2 = (r[3] * x) + (r[4] * y) + (r[5] * z) + t[1];
            double z2 = (r[6] * x) + (r[7] * y) + (r[8] * z) + t[2];
            p1[i] = new Point2d((800 * x / z) + 640, (800 * y / z) + 360);
            p2[i] = new Point2d((800 * x2 / z2) + 640, (800 * y2 / z2) + 360);
            xs[3 * i] = x;
            xs[(3 * i) + 1] = y;
            xs[(3 * i) + 2] = z;
        }

        return (p1, p2, r, t, xs);
    }

    [Fact]
    public void FindEssentialMat_RecoverPose_ExactScene_RecoversMotion()
    {
        (Point2d[] p1, Point2d[] p2, double[] r, double[] t, _) = Scene(100);
        EpipolarEstimationResult e = EpipolarGeometry.FindEssentialMat(p1, p2, K);
        Assert.True(e.Success);
        Assert.Equal(3, e.Model!.Height);
        Assert.Equal(100, e.InlierMask.Count(m => m == 1));

        RecoverPoseResult pose = EpipolarGeometry.RecoverPose(e.Model, p1, p2, K, e.InlierMask);
        Assert.Equal(100, pose.InlierCount);
        double tn = Math.Sqrt(t.Sum(v => v * v));
        for (int i = 0; i < 9; i++)
        {
            Assert.Equal(r[i], pose.R.At(i / 3, i % 3), 1e-6);
        }

        for (int i = 0; i < 3; i++)
        {
            Assert.Equal(t[i] / tn, pose.T.At(i, 0), 1e-6);
        }

        Assert.All(pose.Mask, m => Assert.Equal(1, m));
    }

    [Fact]
    public void RecoverPose_WithoutMask_Uses255AndReturnsPoints()
    {
        (Point2d[] p1, Point2d[] p2, _, _, double[] xs) = Scene(30);
        Mat<double> e = EpipolarGeometry.FindEssentialMat(p1, p2, K).Model!;
        RecoverPoseResult pose = EpipolarGeometry.RecoverPose(e, p1, p2, K, 1000.0, default, true);
        Assert.All(pose.Mask, m => Assert.Equal(255, m));
        Assert.NotNull(pose.TriangulatedPoints);
        Assert.Equal(30, pose.TriangulatedPoints!.Width);
        Assert.Equal(4, pose.TriangulatedPoints.Height);

        // The points are in the first camera frame up to the scale of t.
        double s = xs[2] / (pose.TriangulatedPoints.At(2, 0) / pose.TriangulatedPoints.At(3, 0));
        for (int i = 0; i < 30; i++)
        {
            double w = pose.TriangulatedPoints.At(3, i);
            Assert.Equal(xs[(3 * i) + 2], s * pose.TriangulatedPoints.At(2, i) / w, 1e-5);
        }
    }

    [Fact]
    public void TriangulatePoints_ExactProjections_ReturnsPoints()
    {
        (Point2d[] p1, Point2d[] p2, double[] r, double[] t, double[] xs) = Scene(20);
        var pm1 = new Mat<double>(new double[] { 800, 0, 640, 0, 0, 800, 360, 0, 0, 0, 1, 0 }, 4, 3);
        var rt = new Mat<double>(new double[] { r[0], r[1], r[2], t[0], r[3], r[4], r[5], t[1], r[6], r[7], r[8], t[2] }, 4, 3);
        Mat<double> pm2 = LinearAlgebra.Gemm(K, rt, 1, null, 0);
        Mat<double> x = EpipolarGeometry.TriangulatePoints(pm1, pm2, p1, p2);
        for (int i = 0; i < 20; i++)
        {
            for (int k = 0; k < 3; k++)
            {
                Assert.Equal(xs[(3 * i) + k], x.At(k, i) / x.At(3, i), 1e-6);
            }
        }

        Mat<float> xf = EpipolarGeometry.TriangulatePoints(pm1, pm2, p1.Select(p => new Point2f((float)p.X, (float)p.Y)).ToArray(), p2.Select(p => new Point2f((float)p.X, (float)p.Y)).ToArray());
        Assert.Equal(20, xf.Width);
    }

    [Fact]
    public void FindFundamentalMat_EightPoint_SatisfiesEpipolarConstraint()
    {
        (Point2d[] p1, Point2d[] p2, _, _, _) = Scene(40);
        EpipolarEstimationResult f = EpipolarGeometry.FindFundamentalMat(p1, p2, FundamentalMatMethod.EightPoint);
        Assert.True(f.Success);
        Assert.All(f.InlierMask, m => Assert.Equal(1, m));
        Assert.Equal(1.0, f.Model!.At(2, 2), 12);
        Point3d[] lines = EpipolarGeometry.ComputeCorrespondEpilines(p1, 1, f.Model);
        for (int i = 0; i < p2.Length; i++)
        {
            Assert.True(Math.Abs((lines[i].X * p2[i].X) + (lines[i].Y * p2[i].Y) + lines[i].Z) < 0.05);
        }
    }

    [Fact]
    public void FindFundamentalMat_SevenPoints_ReturnsStackedSolutions()
    {
        (Point2d[] p1, Point2d[] p2, _, _, _) = Scene(7);
        EpipolarEstimationResult f = EpipolarGeometry.FindFundamentalMat(p1, p2, FundamentalMatMethod.Ransac);
        Assert.True(f.Success);
        Assert.Equal(0, f.Model!.Height % 3);
        Assert.InRange(f.Model.Height / 3, 1, 3);
    }

    [Fact]
    public void FindFundamentalMat_Ransac_RejectsOutliers()
    {
        (Point2d[] p1, Point2d[] p2, _, _, _) = Scene(200);
        var rng = new Random(5);
        for (int i = 0; i < 200; i += 5)
        {
            p2[i] = new Point2d(rng.NextDouble() * 1280, rng.NextDouble() * 720);
        }

        EpipolarEstimationResult f = EpipolarGeometry.FindFundamentalMat(p1, p2, FundamentalMatMethod.Ransac, 1.0);
        Assert.True(f.Success);
        Assert.InRange(f.InlierMask.Count(m => m == 1), 160, 162);
    }

    [Fact]
    public void TooFewPoints_ReturnNull()
    {
        (Point2d[] p1, Point2d[] p2, _, _, _) = Scene(6);
        Assert.False(EpipolarGeometry.FindFundamentalMat(p1, p2).Success);
        Assert.False(EpipolarGeometry.FindEssentialMat(p1[..4], p2[..4], K).Success);
    }

    [Fact]
    public void InvalidArguments_Throw()
    {
        (Point2d[] p1, Point2d[] p2, _, _, _) = Scene(10);
        Assert.Throws<ArgumentException>(() => EpipolarGeometry.FindEssentialMat(p1, p2[..9], K));
        Assert.Throws<ArgumentException>(() => EpipolarGeometry.FindEssentialMat(p1, p2, new Mat<double>(4, 3)));
        Assert.Throws<NotSupportedException>(() => EpipolarGeometry.FindEssentialMat(p1, p2, K, RobustEstimationMethod.UsacDefault));
        Assert.Throws<ArgumentOutOfRangeException>(() => EpipolarGeometry.FindEssentialMat(p1, p2, K, RobustEstimationMethod.Ransac, 1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => EpipolarGeometry.ComputeCorrespondEpilines(p1, 3, K));
        Assert.Throws<ArgumentException>(() => EpipolarGeometry.RecoverPose(K, p1, p2, K, new byte[3]));
    }

    [Fact]
    public void Functions_DoNotModifyInputs()
    {
        (Point2d[] p1, Point2d[] p2, _, _, _) = Scene(50);
        Point2d[] c1 = p1.ToArray(), c2 = p2.ToArray();
        double[] k = K.ToArray();
        Mat<double> e = EpipolarGeometry.FindEssentialMat(p1, p2, K).Model!;
        double[] ec = e.ToArray();
        EpipolarGeometry.RecoverPose(e, p1, p2, K);
        EpipolarGeometry.FindFundamentalMat(p1, p2);
        Assert.Equal(c1, p1);
        Assert.Equal(c2, p2);
        Assert.Equal(k, K.ToArray());
        Assert.Equal(ec, e.ToArray());
    }

    [Fact]
    public void DecomposeEssentialMat_ReturnsRotations()
    {
        (Point2d[] p1, Point2d[] p2, _, _, _) = Scene(60);
        Mat<double> e = EpipolarGeometry.FindEssentialMat(p1, p2, K).Model!;
        (Mat<double> r1, Mat<double> r2, Mat<double> t) = EpipolarGeometry.DecomposeEssentialMat(e);
        Assert.Equal(1.0, LinearAlgebra.Determinant(r1), 9);
        Assert.Equal(1.0, LinearAlgebra.Determinant(r2), 9);
        Assert.Equal(1.0, Math.Sqrt(t.ToArray().Sum(v => v * v)), 12);
    }

    [Fact]
    public void ConvertPointsHomogeneous_RoundTripAndZeroScale()
    {
        Point3d[] h = EpipolarGeometry.ConvertPointsToHomogeneous(new[] { new Point2d(2, 3) });
        Assert.Equal(new Point3d(2, 3, 1), h[0]);
        Point2d[] back = EpipolarGeometry.ConvertPointsFromHomogeneous(new[] { new Point3d(4, 6, 2), new Point3d(4, 6, 0) });
        Assert.Equal(new Point2d(2, 3), back[0]);
        Assert.Equal(new Point2d(4, 6), back[1]);
        Point2f[] backF = EpipolarGeometry.ConvertPointsFromHomogeneous(new[] { new Point3f(4, 6, 2) });
        Assert.Equal(new Point2f(2, 3), backF[0]);
    }
}
