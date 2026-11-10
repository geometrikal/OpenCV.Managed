using BenchmarkDotNet.Attributes;
using OpenCvSharp;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>The <see cref="CornerBenchmarks"/> cases run through OpenCvSharp.</summary>
[MemoryDiagnoser]
public class OpenCvSharpCornerBenchmarks
{
    private Mat _gray = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        _gray = OcvInputs.ToOcv(Inputs.Gray(Size));
    }

    [GlobalCleanup]
    public void Cleanup() => _gray.Dispose();

    [Benchmark]
    public int CornerHarris_U8()
    {
        using var dst = new Mat();
        Cv2.CornerHarris(_gray, dst, 2, 3, 0.04);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int CornerMinEigenVal_U8()
    {
        using var dst = new Mat();
        Cv2.CornerMinEigenVal(_gray, dst, 3, 3);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int GoodFeaturesToTrack_U8() => Cv2.GoodFeaturesToTrack(_gray, 1000, 0.01, 10, null!, 3, false, 0.04).Length;
}
