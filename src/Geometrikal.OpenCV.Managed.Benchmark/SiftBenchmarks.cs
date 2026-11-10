using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Features;

namespace Geometrikal.OpenCV.Managed.Benchmark;

[MemoryDiagnoser]
public class SiftBenchmarks
{
    private Mat<byte> _gray = null!;
    private Sift _sift = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _gray = Inputs.Gray(Size);
        _sift = new Sift();
    }

    [Benchmark]
    public int DetectAndCompute_U8()
    {
        List<KeyPoint> keypoints = _sift.DetectAndCompute(_gray, null, out Mat<float>? descriptors);
        return keypoints.Count + (descriptors?.Height ?? 0);
    }
}
