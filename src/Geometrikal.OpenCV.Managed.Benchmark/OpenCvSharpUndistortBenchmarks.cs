using BenchmarkDotNet.Attributes;
using OpenCvSharp;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>The <see cref="UndistortBenchmarks"/> cases run through OpenCvSharp.</summary>
[MemoryDiagnoser]
public class OpenCvSharpUndistortBenchmarks
{
    private Mat _color = null!;
    private Mat _camera = null!;
    private Mat _dist = null!;
    private Mat _points = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        _color = OcvInputs.ToOcv(Inputs.Color(Size));
        _camera = OcvInputs.ToOcv(UndistortBenchmarks.Camera(Size));
        _dist = OcvInputs.ToOcv(new Core.Mat<double>(UndistortBenchmarks.Dist, 1, UndistortBenchmarks.Dist.Length));
        Core.Point2f[] pts = UndistortBenchmarks.Points(Size);
        var flat = new Core.Mat<float>(System.Runtime.InteropServices.MemoryMarshal.Cast<Core.Point2f, float>(pts).ToArray(), 1, pts.Length, 2);
        _points = OcvInputs.ToOcv(flat);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (Mat m in new[] { _color, _camera, _dist, _points })
        {
            m.Dispose();
        }
    }

    [Benchmark]
    public int InitUndistortRectifyMap_32FC1() => Map(MatType.CV_32FC1);

    [Benchmark]
    public int InitUndistortRectifyMap_16SC2() => Map(MatType.CV_16SC2);

    [Benchmark]
    public int Undistort_U8C3()
    {
        using var dst = new Mat();
        Cv2.Undistort(_color, dst, _camera, _dist);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int UndistortPoints10k()
    {
        using var dst = new Mat();
        Cv2.UndistortPoints(_points, dst, _camera, _dist);
        return OcvInputs.Consume(dst);
    }

    private int Map(MatType type)
    {
        using var map1 = new Mat();
        using var map2 = new Mat();
        using var r = new Mat();
        Cv2.InitUndistortRectifyMap(_camera, _dist, r, _camera, new OpenCvSharp.Size(Size, Size), type, map1, map2);
        return OcvInputs.Consume(map1);
    }
}
