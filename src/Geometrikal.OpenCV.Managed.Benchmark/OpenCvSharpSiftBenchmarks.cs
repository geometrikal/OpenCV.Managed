using BenchmarkDotNet.Attributes;
using OpenCvSharp;
using OpenCvSharp.Features2D;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>The <see cref="SiftBenchmarks"/> cases run through OpenCvSharp.</summary>
[MemoryDiagnoser]
public class OpenCvSharpSiftBenchmarks
{
    private Mat _gray = null!;
    private SIFT _sift = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        _gray = OcvInputs.ToOcv(Inputs.Gray(Size));
        _sift = SIFT.Create();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _sift.Dispose();
        _gray.Dispose();
    }

    [Benchmark]
    public int DetectAndCompute_U8()
    {
        using var descriptors = new Mat();
        _sift.DetectAndCompute(_gray, null, out KeyPoint[] keypoints, descriptors);
        return keypoints.Length + descriptors.Rows;
    }
}
