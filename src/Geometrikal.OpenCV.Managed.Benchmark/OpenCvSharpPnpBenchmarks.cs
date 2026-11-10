using BenchmarkDotNet.Attributes;
using OpenCvSharp;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>The <see cref="PnpBenchmarks"/> cases run through OpenCvSharp.</summary>
[MemoryDiagnoser]
public class OpenCvSharpPnpBenchmarks
{
    private Point3d[] _relief = null!;
    private Point2d[] _reliefImg = null!;
    private Point3d[] _planar = null!;
    private Point2d[] _planarImg = null!;
    private Point3f[] _ransacObj = null!;
    private Point2f[] _ransacImg = null!;
    private double[,] _camera = null!;

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        double[] k = PnpBenchmarks.CameraValues;
        _camera = new double[3, 3] { { k[0], k[1], k[2] }, { k[3], k[4], k[5] }, { k[6], k[7], k[8] } };
        (_relief, _reliefImg) = Convert(PnpBenchmarks.Scene(PnpBenchmarks.PnpPoints, false, false));
        (_planar, _planarImg) = Convert(PnpBenchmarks.Scene(PnpBenchmarks.PnpPoints, true, false));
        (Point3d[] o, Point2d[] i) = Convert(PnpBenchmarks.Scene(PnpBenchmarks.RansacPoints, false, true));
        _ransacObj = Array.ConvertAll(o, p => new Point3f((float)p.X, (float)p.Y, (float)p.Z));
        _ransacImg = Array.ConvertAll(i, p => new Point2f((float)p.X, (float)p.Y));
    }

    [Benchmark]
    public double SolvePnP_Epnp100() => Solve(_relief, _reliefImg, SolvePnPMethod.EPNP);

    [Benchmark]
    public double SolvePnP_Sqpnp100() => Solve(_relief, _reliefImg, SolvePnPMethod.SQPNP);

    [Benchmark]
    public double SolvePnP_Ippe100() => Solve(_planar, _planarImg, SolvePnPMethod.IPPE);

    [Benchmark]
    public double SolvePnP_Iterative100() => Solve(_relief, _reliefImg, SolvePnPMethod.Iterative);

    [Benchmark]
    public int SolvePnPRansac_Epnp500()
    {
        using var obj = InputArray.Create(_ransacObj);
        using var img = InputArray.Create(_ransacImg);
        using var cam = InputArray.Create(_camera);
        using var dist = InputArray.Create(PnpBenchmarks.Dist);
        using var rvec = new Mat();
        using var tvec = new Mat();
        using var inliers = new Mat();
        Cv2.SolvePnPRansac(obj, img, cam, dist, rvec, tvec, false, 100, 8.0f, 0.99, inliers, SolvePnPMethod.EPNP);
        return inliers.Rows;
    }

    private double Solve(Point3d[] obj, Point2d[] img, SolvePnPMethod flags)
    {
        using var o = InputArray.Create(obj);
        using var i = InputArray.Create(img);
        using var cam = InputArray.Create(_camera);
        using var dist = InputArray.Create(PnpBenchmarks.Dist);
        using var rvec = new Mat();
        using var tvec = new Mat();
        Cv2.SolvePnP(o, i, cam, dist, rvec, tvec, false, flags);
        return rvec.At<double>(0);
    }

    private static (Point3d[] Obj, Point2d[] Img) Convert((Core.Point3d[] Obj, Core.Point2d[] Img) s)
        => (Array.ConvertAll(s.Obj, p => new Point3d(p.X, p.Y, p.Z)), Array.ConvertAll(s.Img, p => new Point2d(p.X, p.Y)));
}
