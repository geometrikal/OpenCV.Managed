using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

[MemoryDiagnoser]
public class CornerBenchmarks
{
    private Mat<byte> _gray = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _gray = Inputs.Gray(Size);
    }

    [Benchmark]
    public Mat<float> CornerHarris_U8() => Corners.CornerHarris(_gray, 2, 3, 0.04);

    [Benchmark]
    public Mat<float> CornerMinEigenVal_U8() => Corners.CornerMinEigenVal(_gray, 3, 3);

    [Benchmark]
    public Point2f[] GoodFeaturesToTrack_U8() => Corners.GoodFeaturesToTrack(_gray, 1000, 0.01, 10);
}
