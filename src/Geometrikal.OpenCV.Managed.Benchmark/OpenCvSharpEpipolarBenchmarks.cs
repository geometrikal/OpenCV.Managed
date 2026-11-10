using BenchmarkDotNet.Attributes;
using OpenCvSharp;
using Port = Geometrikal.OpenCV.Managed.Core;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>The <see cref="EpipolarBenchmarks"/> cases run through OpenCvSharp.</summary>
[MemoryDiagnoser]
public class OpenCvSharpEpipolarBenchmarks
{
    private Mat _p1 = null!;
    private Mat _p2 = null!;
    private Mat _k = null!;
    private Mat _e = null!;
    private Mat _mask = null!;

    [Params(500, 2000)]
    public int Points { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        (Port.Point2d[] p1, Port.Point2d[] p2) = EpipolarBenchmarks.TwoViewPoints(Points);
        _p1 = ToOcv(p1);
        _p2 = ToOcv(p2);
        _k = OcvInputs.ToOcv(EpipolarBenchmarks.CameraMatrix);
        _mask = new Mat();
        _e = Cv2.FindEssentialMat(_p1, _p2, _k, EssentialMatMethod.Ransac, 0.999, 1.0, _mask);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _p1.Dispose();
        _p2.Dispose();
        _k.Dispose();
        _e.Dispose();
        _mask.Dispose();
    }

    [Benchmark]
    public int FindFundamentalMat_Ransac()
    {
        using var mask = new Mat();
        using Mat f = Cv2.FindFundamentalMat(_p1, _p2, FundamentalMatMethods.Ransac, 1.0, 0.99, mask);
        return OcvInputs.Consume(mask);
    }

    [Benchmark]
    public int FindEssentialMat_Ransac()
    {
        using var mask = new Mat();
        using Mat e = Cv2.FindEssentialMat(_p1, _p2, _k, EssentialMatMethod.Ransac, 0.999, 1.0, mask);
        return OcvInputs.Consume(mask);
    }

    [Benchmark]
    public int RecoverPose()
    {
        using var r = new Mat();
        using var t = new Mat();
        using Mat mask = _mask.Clone();
        return Cv2.RecoverPose(_e, _p1, _p2, _k, r, t, mask);
    }

    private static Mat ToOcv(Port.Point2d[] points)
    {
        var flat = new double[points.Length * 2];
        for (int i = 0; i < points.Length; i++)
        {
            flat[2 * i] = points[i].X;
            flat[(2 * i) + 1] = points[i].Y;
        }

        return OcvInputs.ToOcv(new Port.Mat<double>(flat, 1, points.Length, 2));
    }
}
