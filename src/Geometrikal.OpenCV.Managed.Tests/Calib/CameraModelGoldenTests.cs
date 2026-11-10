namespace Geometrikal.OpenCV.Managed.Tests.Calib;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Calib;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Replays the cv2 5.0.0 <c>Rodrigues</c>, <c>projectPoints</c> and <c>solvePnP</c> reference outputs in
/// <c>TestData/undistort_golden.json</c> through <see cref="CameraModel"/>. Every comparison is
/// byte-exact.
/// </summary>
public sealed class CameraModelGoldenTests
{
    private static readonly string[] Ops = ["rod", "proj", "pnp"];

    public static IEnumerable<object[]> CaseNames()
        => UndistortGolden.Cases.Where(c => !c.Suspect && Ops.Contains(c.Op)).Select(c => new object[] { c.Name }).OrderBy(o => (string)o[0], StringComparer.Ordinal);

    [Fact]
    public void GoldenFile_HasCameraModelCases() => Assert.True(UndistortGolden.Cases.Count(c => Ops.Contains(c.Op)) > 100);

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void CameraModel_MatchesOpenCvGolden(string name)
    {
        GoldenCase c = UndistortGolden.Get(name);
        if (c.Op == "rod")
        {
            if (c.InputDtype("src") == "f32")
            {
                Mat<float> dst = CameraModel.Rodrigues(c.In<float>("src"), out Mat<float> jac);
                c.Compare("dst", dst);
                c.Compare("jac", jac);
                c.Compare("dst", CameraModel.Rodrigues(c.In<float>("src")));
            }
            else
            {
                Mat<double> dst = CameraModel.Rodrigues(c.InD("src"), out Mat<double> jac);
                c.Compare("dst", dst);
                c.Compare("jac", jac);
                c.Compare("dst", CameraModel.Rodrigues(c.InD("src")));
            }

            return;
        }

        Mat<double> a = c.InD("a");
        double[] dist = c.Has("dist") ? c.InD("dist").ToArray() : [];
        if (c.Op == "pnp")
        {
            SolvePnPCase(c, a, dist);
            return;
        }

        double[] t = c.InD("tvec").ToArray();
        var tvec = new Point3d(t[0], t[1], t[2]);
        double aspect = c.Params.GetProperty("aspect").GetDouble();
        if (c.Has("rmat"))
        {
            Point3d[] objD = MemoryMarshal.Cast<double, Point3d>(c.InD("src").ToArray()).ToArray();
            Point2d[] ip = CameraModel.ProjectPoints(objD, c.InD("rmat"), tvec, a, dist, aspect);
            c.Compare("dst", new Mat<double>(MemoryMarshal.Cast<Point2d, double>(ip).ToArray(), 2, ip.Length));
            return;
        }

        double[] rv = c.InD("rvec").ToArray();
        var rvec = new Point3d(rv[0], rv[1], rv[2]);
        if (c.InputDtype("src") == "f32")
        {
            Point3f[] obj = MemoryMarshal.Cast<float, Point3f>(c.In<float>("src").ToArray()).ToArray();
            Point2f[] ip = CameraModel.ProjectPoints(obj, rvec, tvec, a, dist, out Mat<double> jac, aspect);
            c.Compare("dst", new Mat<float>(MemoryMarshal.Cast<Point2f, float>(ip).ToArray(), 2, ip.Length));
            c.Compare("jac", jac);
            Point2f[] ip2 = CameraModel.ProjectPoints(obj, rvec, tvec, a, dist, aspect);
            Assert.Equal(ip, ip2);
        }
        else
        {
            Point3d[] obj = MemoryMarshal.Cast<double, Point3d>(c.InD("src").ToArray()).ToArray();
            Point2d[] ip = CameraModel.ProjectPoints(obj, rvec, tvec, a, dist, out Mat<double> jac, aspect);
            c.Compare("dst", new Mat<double>(MemoryMarshal.Cast<Point2d, double>(ip).ToArray(), 2, ip.Length));
            c.Compare("jac", jac);
            Point2d[] ip2 = CameraModel.ProjectPoints(obj, rvec, tvec, a, dist, aspect);
            Assert.Equal(ip, ip2);
        }
    }

    /// <summary>The pose must equal the cv2 pose bit for bit.</summary>
    private static void SolvePnPCase(GoldenCase c, Mat<double> a, double[] dist)
    {
        Point3d[] obj = MemoryMarshal.Cast<double, Point3d>(c.InD("obj").ToArray()).ToArray();
        Point2d[] img = MemoryMarshal.Cast<double, Point2d>(c.InD("img").ToArray()).ToArray();
        Point3d rvec, tvec;
        if (c.Params.GetProperty("guess").GetBoolean())
        {
            double[] r0 = c.InD("rvec0").ToArray();
            double[] t0 = c.InD("tvec0").ToArray();
            rvec = new Point3d(r0[0], r0[1], r0[2]);
            tvec = new Point3d(t0[0], t0[1], t0[2]);
            Assert.True(CameraModel.SolvePnPRefine(obj, img, a, dist, ref rvec, ref tvec));
        }
        else
        {
            Assert.True(CameraModel.SolvePnP(obj, img, a, dist, out rvec, out tvec));
        }

        c.Compare("rvec", new Mat<double>([rvec.X, rvec.Y, rvec.Z], 1, 3));
        c.Compare("tvec", new Mat<double>([tvec.X, tvec.Y, tvec.Z], 1, 3));
    }
}
