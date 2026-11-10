using BenchmarkDotNet.Attributes;
using OpenCvSharp;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>The <see cref="FeatureBenchmarks"/> cases run through OpenCvSharp.</summary>
[MemoryDiagnoser]
public class OpenCvSharpFeatureBenchmarks
{
    private Mat _gray = null!;
    private ORB _orb = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        _gray = OcvInputs.ToOcv(Inputs.Gray(Size));
        _orb = ORB.Create();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _orb.Dispose();
        _gray.Dispose();
    }

    [Benchmark]
    public int Fast9_16_Nms() => Detect(20, true, FASTType.TYPE_9_16);

    [Benchmark]
    public int Fast9_16_All() => Detect(20, false, FASTType.TYPE_9_16);

    [Benchmark]
    public int Fast7_12_Nms() => Detect(20, true, FASTType.TYPE_7_12);

    [Benchmark]
    public int OrbDetect() => _orb.Detect(_gray).Length;

    [Benchmark]
    public int OrbDetectAndCompute()
    {
        using var descriptors = new Mat();
        _orb.DetectAndCompute(_gray, null, out KeyPoint[] keypoints, descriptors);
        return keypoints.Length;
    }

    private int Detect(int threshold, bool nonmax, FASTType type)
    {
        return Cv2.FAST(_gray, threshold, nonmax, type).Length;
    }
}
