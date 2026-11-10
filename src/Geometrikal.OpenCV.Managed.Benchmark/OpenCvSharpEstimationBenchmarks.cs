using BenchmarkDotNet.Attributes;
using OpenCvSharp;
using Port = Geometrikal.OpenCV.Managed.Core;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>The <see cref="EstimationBenchmarks"/> cases run through OpenCvSharp.</summary>
[MemoryDiagnoser]
public class OpenCvSharpEstimationBenchmarks
{
    private Mat _src = null!;
    private Mat _dst = null!;

    [Params(500, 5000)]
    public int Points { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        (Port.Point2f[] src, Port.Point2f[] dst) = EstimationBenchmarks.MatchedPoints(Points);
        _src = ToOcv(src);
        _dst = ToOcv(dst);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _src.Dispose();
        _dst.Dispose();
    }

    [Benchmark]
    public int FindHomography_Ransac()
    {
        using var mask = new Mat();
        using Mat h = Cv2.FindHomography(_src, _dst, HomographyMethods.Ransac, 3.0, mask);
        return OcvInputs.Consume(mask);
    }

    [Benchmark]
    public int FindHomography_Lmeds()
    {
        using var mask = new Mat();
        using Mat h = Cv2.FindHomography(_src, _dst, HomographyMethods.LMedS, 3.0, mask);
        return OcvInputs.Consume(mask);
    }

    [Benchmark]
    public int FindHomography_Rho()
    {
        using var mask = new Mat();
        using Mat h = Cv2.FindHomography(_src, _dst, HomographyMethods.Rho, 3.0, mask);
        return OcvInputs.Consume(mask);
    }

    [Benchmark]
    public int EstimateAffine2D_Ransac()
    {
        using var mask = new Mat();
        using Mat? m = Cv2.EstimateAffine2D(_src, _dst, mask, RobustEstimationAlgorithms.RANSAC, 3.0);
        return OcvInputs.Consume(mask);
    }

    [Benchmark]
    public int EstimateAffinePartial2D_Ransac()
    {
        using var mask = new Mat();
        using Mat? m = Cv2.EstimateAffinePartial2D(_src, _dst, mask, RobustEstimationAlgorithms.RANSAC, 3.0);
        return OcvInputs.Consume(mask);
    }

    [Benchmark]
    public int EstimateAffinePartial2D_Lmeds()
    {
        using var mask = new Mat();
        using Mat? m = Cv2.EstimateAffinePartial2D(_src, _dst, mask, RobustEstimationAlgorithms.LMEDS, 3.0);
        return OcvInputs.Consume(mask);
    }

    private static Mat ToOcv(Port.Point2f[] points)
    {
        var flat = new float[points.Length * 2];
        for (int i = 0; i < points.Length; i++)
        {
            flat[2 * i] = points[i].X;
            flat[(2 * i) + 1] = points[i].Y;
        }

        return OcvInputs.ToOcv(new Port.Mat<float>(flat, 1, points.Length, 2));
    }
}
