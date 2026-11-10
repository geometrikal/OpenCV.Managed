using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Features;

namespace Geometrikal.OpenCV.Managed.Benchmark;

[MemoryDiagnoser]
public class FeatureBenchmarks
{
    private Mat<byte> _gray = null!;
    private Orb _orb = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _gray = Inputs.Gray(Size);
        _orb = new Orb();
    }

    [Benchmark]
    public int Fast9_16_Nms() => Fast.Detect(_gray, 20).Length;

    [Benchmark]
    public int Fast9_16_All() => Fast.Detect(_gray, 20, nonmaxSuppression: false).Length;

    [Benchmark]
    public int Fast7_12_Nms() => Fast.Detect(_gray, 20, true, FastDetectorType.Type7_12).Length;

    [Benchmark]
    public int OrbDetect() => _orb.Detect(_gray).Length;

    [Benchmark]
    public int OrbDetectAndCompute() => _orb.DetectAndCompute(_gray).Keypoints.Length;
}
