using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Calib;
using Geometrikal.OpenCV.Managed.Core;

namespace Geometrikal.OpenCV.Managed.Benchmark;

[MemoryDiagnoser]
public class UndistortBenchmarks
{
    public const int PointCount = 10000;

    /// <summary>DJI FC6310-like coefficients (k1, k2, p1, p2, k3).</summary>
    public static readonly double[] Dist = [-0.2713, 0.1125, 0.00071, -0.00118, -0.0315];

    private Mat<byte> _color = null!;
    private Mat<double> _camera = null!;
    private Point2f[] _points = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    /// <summary>Camera matrix for a square image of <paramref name="size"/> pixels.</summary>
    public static Mat<double> Camera(int size)
        => new([size * 0.66, 0, (size - 1) * 0.5 + 2.5, 0, size * 0.665, (size - 1) * 0.5 - 1.5, 0, 0, 1], 3, 3);

    /// <summary>Point i is <c>(Hash(i, 7) mod 16·size, Hash(i, 11) mod 16·size) / 16</c>.</summary>
    public static Point2f[] Points(int size)
    {
        var pts = new Point2f[PointCount];
        uint range = (uint)size * 16;
        for (int i = 0; i < PointCount; i++)
        {
            pts[i] = new Point2f((Inputs.Hash((uint)i, 7) % range) / 16f, (Inputs.Hash((uint)i, 11) % range) / 16f);
        }

        return pts;
    }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _color = Inputs.Color(Size);
        _camera = Camera(Size);
        _points = Points(Size);
    }

    [Benchmark]
    public (Mat<float>, Mat<float>) InitUndistortRectifyMap_32FC1()
        => Undistortion.InitUndistortRectifyMap(_camera, Dist, null, _camera, new Size(Size, Size));

    [Benchmark]
    public (Mat<short>, Mat<ushort>) InitUndistortRectifyMap_16SC2()
        => Undistortion.InitUndistortRectifyMapFixedPoint(_camera, Dist, null, _camera, new Size(Size, Size));

    [Benchmark]
    public Mat<byte> Undistort_U8C3() => Undistortion.Undistort(_color, _camera, Dist);

    [Benchmark]
    public Point2f[] UndistortPoints10k() => Undistortion.UndistortPoints(_points, _camera, Dist);
}
