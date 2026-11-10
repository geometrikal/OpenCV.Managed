using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Calib;
using Geometrikal.OpenCV.Managed.Core;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// Pose estimation from ground control points seen by a drone camera (DJI FC6310-like
/// intrinsics with k1-k3 distortion, nadir pose at 120 m). The <c>SolvePnP</c> cases use 100
/// points with ±0.5 pixel noise; <c>SolvePnPRansac</c> uses 500 points with 30% outliers.
/// <c>tools/bench_cv2.py</c> builds the same points.
/// </summary>
[MemoryDiagnoser]
public class PnpBenchmarks
{
    public const int PnpPoints = 100;
    public const int RansacPoints = 500;

    /// <summary>Camera matrix of a 5472 × 3648 DJI FC6310-like camera.</summary>
    public static readonly double[] CameraValues = [3648, 0, 2761.5, 0, 3650.5536, 1803.5, 0, 0, 1];

    public static readonly double[] Dist = [-0.2713, 0.1125, 0.00071, -0.00118, -0.0315];

    public static readonly Point3d RotationVector = new(0.03, -0.02, 0.4);
    public static readonly Point3d TranslationVector = new(1.5, -2.0, 120.0);

    private Point3d[] _relief = null!;
    private Point2d[] _reliefImg = null!;
    private Point3d[] _planar = null!;
    private Point2d[] _planarImg = null!;
    private Point3d[] _ransacObj = null!;
    private Point2d[] _ransacImg = null!;
    private Mat<double> _camera = null!;

    /// <summary>
    /// Builds <paramref name="count"/> object points in ±60 m (Z in ±4 m, or 0 when
    /// <paramref name="planar"/>) and their projections with ±0.5 pixel noise. Point i uses
    /// <see cref="Inputs.Hash"/> with the keys (i, 1) to (i, 6); with <paramref name="outliers"/>,
    /// a point with <c>Hash(i, 6) % 10 &lt; 3</c> moves by (+250, −180) pixels.
    /// </summary>
    public static (Point3d[] Obj, Point2d[] Img) Scene(int count, bool planar, bool outliers)
    {
        var obj = new Point3d[count];
        for (int i = 0; i < count; i++)
        {
            uint u = (uint)i;
            obj[i] = new Point3d(
                ((Inputs.Hash(u, 1) % 120001) / 1000.0) - 60,
                ((Inputs.Hash(u, 2) % 120001) / 1000.0) - 60,
                planar ? 0 : ((Inputs.Hash(u, 3) % 8001) / 1000.0) - 4);
        }

        Point2d[] img = CameraModel.ProjectPoints(obj, RotationVector, TranslationVector, new Mat<double>((double[])CameraValues.Clone(), 3, 3), Dist);
        for (int i = 0; i < count; i++)
        {
            uint u = (uint)i;
            double nx = ((Inputs.Hash(u, 4) % 1001) / 1000.0) - 0.5;
            double ny = ((Inputs.Hash(u, 5) % 1001) / 1000.0) - 0.5;
            bool outlier = outliers && Inputs.Hash(u, 6) % 10 < 3;
            img[i] = new Point2d(img[i].X + nx + (outlier ? 250 : 0), img[i].Y + ny + (outlier ? -180 : 0));
        }

        return (obj, img);
    }

    [GlobalSetup]
    public void Setup()
    {
        _camera = new Mat<double>((double[])CameraValues.Clone(), 3, 3);
        (_relief, _reliefImg) = Scene(PnpPoints, false, false);
        (_planar, _planarImg) = Scene(PnpPoints, true, false);
        (_ransacObj, _ransacImg) = Scene(RansacPoints, false, true);
    }

    [Benchmark]
    public Point3d SolvePnP_Epnp100() => Solve(_relief, _reliefImg, SolvePnPMethod.Epnp);

    [Benchmark]
    public Point3d SolvePnP_Sqpnp100() => Solve(_relief, _reliefImg, SolvePnPMethod.Sqpnp);

    [Benchmark]
    public Point3d SolvePnP_Ippe100() => Solve(_planar, _planarImg, SolvePnPMethod.Ippe);

    [Benchmark]
    public Point3d SolvePnP_Iterative100() => Solve(_relief, _reliefImg, SolvePnPMethod.Iterative);

    [Benchmark]
    public int SolvePnPRansac_Epnp500()
        => PnP.SolvePnPRansac(_ransacObj, _ransacImg, _camera, Dist, SolvePnPMethod.Epnp, 100, 8.0f, 0.99).Inliers.Length;

    private Point3d Solve(Point3d[] obj, Point2d[] img, SolvePnPMethod method)
    {
        PnP.SolvePnP(obj, img, _camera, Dist, method, out Point3d r, out _);
        return r;
    }
}
