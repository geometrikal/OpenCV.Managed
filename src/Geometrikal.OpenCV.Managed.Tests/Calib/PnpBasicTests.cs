namespace Geometrikal.OpenCV.Managed.Tests.Calib;

using System;
using System.Linq;
using Geometrikal.OpenCV.Managed.Calib;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>Behavioural tests of <see cref="PnP"/>: pose recovery, RANSAC inliers, overload equivalence and input validation.</summary>
public sealed class PnpBasicTests
{
    private static readonly Point3d TrueR = new(0.1, -0.05, 0.6);
    private static readonly Point3d TrueT = new(2.0, -1.0, 80.0);
    // undistortPoints runs 5 fixed iterations, so exact pose recovery needs data without distortion.
    private static readonly double[] Dist = [];

    private static Mat<double> Camera()
    {
        var a = new Mat<double>(3, 3);
        a.At(0, 0) = 3648;
        a.At(1, 1) = 3650;
        a.At(0, 2) = 2761.5;
        a.At(1, 2) = 1803.5;
        a.At(2, 2) = 1;
        return a;
    }

    private static Point3d[] Scene(int n, bool planar, int seed)
    {
        var rng = new Random(seed);
        return Enumerable.Range(0, n).Select(_ => new Point3d(rng.NextDouble() * 80 - 40, rng.NextDouble() * 80 - 40, planar ? 0 : rng.NextDouble() * 6 - 3)).ToArray();
    }

    [Theory]
    [InlineData(SolvePnPMethod.Iterative, false)]
    [InlineData(SolvePnPMethod.Epnp, false)]
    [InlineData(SolvePnPMethod.Sqpnp, false)]
    [InlineData(SolvePnPMethod.Ippe, true)]
    [InlineData(SolvePnPMethod.Epnp, true)]
    [InlineData(SolvePnPMethod.Sqpnp, true)]
    public void SolvePnP_ExactPoints_RecoversPose(SolvePnPMethod method, bool planar)
    {
        Point3d[] obj = Scene(20, planar, 1);
        Mat<double> a = Camera();
        Point2d[] img = CameraModel.ProjectPoints(obj, TrueR, TrueT, a, Dist);
        Assert.True(PnP.SolvePnP(obj, img, a, Dist, method, out Point3d r, out Point3d t));
        AssertClose(TrueR, r, 1e-6);
        AssertClose(TrueT, t, 1e-4);
    }

    [Theory]
    [InlineData(SolvePnPMethod.P3p)]
    [InlineData(SolvePnPMethod.Ap3p)]
    public void SolveP3P_ExactPoints_FirstSolutionIsTruePose(SolvePnPMethod method)
    {
        Point3d[] obj = Scene(4, false, 2);
        Mat<double> a = Camera();
        Point2d[] img = CameraModel.ProjectPoints(obj, TrueR, TrueT, a, Dist);
        PnPPose[] poses = PnP.SolveP3P(obj, img, a, Dist, method);
        Assert.NotEmpty(poses);
        AssertClose(TrueR, poses[0].RotationVector, 1e-6);
        Assert.True(PnP.SolvePnP(obj, img, a, Dist, method, out Point3d r, out _));
        Assert.Equal(poses[0].RotationVector, r);
    }

    [Fact]
    public void SolvePnPGeneric_Ippe_GivesTwoSolutionsSortedByError()
    {
        Point3d[] obj = Scene(10, true, 3);
        Mat<double> a = Camera();
        Point2d[] img = CameraModel.ProjectPoints(obj, TrueR, TrueT, a, Dist);
        PnPSolution[] sols = PnP.SolvePnPGeneric(obj, img, a, Dist, SolvePnPMethod.Ippe);
        Assert.Equal(2, sols.Length);
        Assert.True(sols[0].ReprojectionError <= sols[1].ReprojectionError);
        Assert.True(sols[0].ReprojectionError < 1e-6);
    }

    [Fact]
    public void CameraModelSolvePnP_IterativeMethod_EqualsOriginalOverload()
    {
        Point3d[] obj = Scene(12, false, 4);
        Mat<double> a = Camera();
        Point2d[] img = CameraModel.ProjectPoints(obj, TrueR, TrueT, a, Dist);
        CameraModel.SolvePnP(obj, img, a, Dist, out Point3d r1, out Point3d t1);
        CameraModel.SolvePnP(obj, img, a, Dist, SolvePnPMethod.Iterative, out Point3d r2, out Point3d t2);
        Assert.Equal(r1, r2);
        Assert.Equal(t1, t2);
    }

    [Theory]
    [InlineData(SolvePnPMethod.Iterative)]
    [InlineData(SolvePnPMethod.Epnp)]
    [InlineData(SolvePnPMethod.P3p)]
    [InlineData(SolvePnPMethod.Sqpnp)]
    public void SolvePnPRansac_Outliers_AreRejected(SolvePnPMethod method)
    {
        Point3d[] obj = Scene(100, false, 5);
        Mat<double> a = Camera();
        Point2d[] img = CameraModel.ProjectPoints(obj, TrueR, TrueT, a, Dist);
        var rng = new Random(6);
        int[] outliers = Enumerable.Range(0, 100).Where(i => i % 4 == 0).ToArray();
        foreach (int i in outliers)
        {
            img[i] = new Point2d(img[i].X + 100 + rng.NextDouble() * 300, img[i].Y - 100 - rng.NextDouble() * 300);
        }

        PnPRansacResult res = PnP.SolvePnPRansac(obj, img, a, Dist, method, 200, 2.0f, 0.99);
        Assert.True(res.Success);
        Assert.Empty(res.Inliers.Intersect(outliers));
        Assert.Equal(75, res.Inliers.Length);
        AssertClose(TrueR, res.RotationVector, 1e-4);
    }

    [Fact]
    public void SolvePnPRansac_DoubleOverload_EqualsFloatOverload()
    {
        Point3d[] obj = Scene(40, true, 7);
        Mat<double> a = Camera();
        Point2d[] img = CameraModel.ProjectPoints(obj, TrueR, TrueT, a, Dist);
        Point3f[] objF = obj.Select(p => new Point3f((float)p.X, (float)p.Y, (float)p.Z)).ToArray();
        Point2f[] imgF = img.Select(p => new Point2f((float)p.X, (float)p.Y)).ToArray();
        PnPRansacResult d = PnP.SolvePnPRansac(obj, img, a, Dist, SolvePnPMethod.Epnp);
        PnPRansacResult f = PnP.SolvePnPRansac(objF, imgF, a, Dist, SolvePnPMethod.Epnp);
        Assert.Equal(d.RotationVector, f.RotationVector);
        Assert.Equal(d.TranslationVector, f.TranslationVector);
        Assert.Equal(d.Inliers, f.Inliers);
    }

    [Fact]
    public void SolvePnPRefine_ConvergeFromPerturbedPose()
    {
        Point3d[] obj = Scene(15, false, 8);
        Mat<double> a = Camera();
        Point2d[] img = CameraModel.ProjectPoints(obj, TrueR, TrueT, a, Dist);
        Point3d r = new(TrueR.X + 0.01, TrueR.Y - 0.01, TrueR.Z + 0.02);
        Point3d t = new(TrueT.X + 0.5, TrueT.Y, TrueT.Z - 1);
        PnP.SolvePnPRefineLM(obj, img, a, Dist, ref r, ref t);
        AssertClose(TrueR, r, 1e-6);

        r = new(TrueR.X + 0.01, TrueR.Y - 0.01, TrueR.Z + 0.02);
        t = new(TrueT.X + 0.5, TrueT.Y, TrueT.Z - 1);
        PnP.SolvePnPRefineVVS(obj, img, a, Dist, ref r, ref t);
        AssertClose(TrueR, r, 1e-6);
    }

    [Fact]
    public void InvalidInputs_Throw()
    {
        Mat<double> a = Camera();
        Point3d[] obj = Scene(5, false, 9);
        Point2d[] img = CameraModel.ProjectPoints(obj, TrueR, TrueT, a, Dist);
        Assert.Throws<ArgumentException>(() => PnP.SolvePnP(obj, img.AsSpan(0, 4), a, Dist, SolvePnPMethod.Epnp, out _, out _));
        Assert.Throws<ArgumentException>(() => PnP.SolvePnP(obj.AsSpan(0, 3), img.AsSpan(0, 3), a, Dist, SolvePnPMethod.Epnp, out _, out _));
        Assert.Throws<ArgumentException>(() => PnP.SolvePnP(obj, img, a, Dist, SolvePnPMethod.P3p, out _, out _));
        Assert.Throws<ArgumentException>(() => PnP.SolvePnP(obj, img, a, Dist, SolvePnPMethod.IppeSquare, out _, out _));
        Assert.Throws<ArgumentException>(() => PnP.SolveP3P(obj, img, a, Dist, SolvePnPMethod.P3p));
        Assert.Throws<ArgumentException>(() => PnP.SolveP3P(obj.AsSpan(0, 4), img.AsSpan(0, 4), a, Dist, SolvePnPMethod.Epnp));
        Assert.Throws<ArgumentException>(() => PnP.SolvePnPRansac(obj.AsSpan(0, 3), img.AsSpan(0, 3), a, Dist));
        Assert.Throws<ArgumentOutOfRangeException>(() => PnP.SolvePnPRansac(Scene(20, false, 1), CameraModel.ProjectPoints(Scene(20, false, 1), TrueR, TrueT, a, Dist), a, Dist, confidence: 1.0));
        Point3d r = TrueR, t = TrueT;
        Assert.Throws<ArgumentException>(() => PnP.SolvePnPRefineLM(obj.AsSpan(0, 2), img.AsSpan(0, 2), a, Dist, ref r, ref t));
    }

    [Fact]
    public void Inputs_AreNotModified()
    {
        Point3d[] obj = Scene(30, false, 10);
        Mat<double> a = Camera();
        Point2d[] img = CameraModel.ProjectPoints(obj, TrueR, TrueT, a, Dist);
        Point3d[] obj0 = (Point3d[])obj.Clone();
        Point2d[] img0 = (Point2d[])img.Clone();
        double[] a0 = a.ToArray();
        PnP.SolvePnPRansac(obj, img, a, Dist, SolvePnPMethod.Epnp);
        PnP.SolvePnPGeneric(obj, img, a, Dist, SolvePnPMethod.Sqpnp);
        Assert.Equal(obj0, obj);
        Assert.Equal(img0, img);
        Assert.Equal(a0, a.ToArray());
    }

    private static void AssertClose(Point3d e, Point3d a, double tol)
    {
        Assert.True(Math.Abs(e.X - a.X) <= tol && Math.Abs(e.Y - a.Y) <= tol && Math.Abs(e.Z - a.Z) <= tol, $"expected {e}, actual {a}");
    }
}
