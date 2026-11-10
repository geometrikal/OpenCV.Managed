using BenchmarkDotNet.Attributes;
using OpenCvSharp;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// The <see cref="ContourBenchmarks"/> cases run through OpenCvSharp. Each call returns managed
/// point arrays and a managed hierarchy, as the port does.
/// </summary>
[MemoryDiagnoser]
public class OpenCvSharpContourBenchmarks
{
    private Mat _binary = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        _binary = OcvInputs.ToOcv(Inputs.Binary(Size));
    }

    [GlobalCleanup]
    public void Cleanup() => _binary.Dispose();

    [Benchmark]
    public Point[][] FindContoursTreeNone() => Find(RetrievalModes.Tree, ContourApproximationModes.ApproxNone);

    [Benchmark]
    public Point[][] FindContoursExternalSimple() => Find(RetrievalModes.External, ContourApproximationModes.ApproxSimple);

    [Benchmark]
    public Point[][] FindContoursListTc89L1() => Find(RetrievalModes.List, ContourApproximationModes.ApproxTC89L1);

    private Point[][] Find(RetrievalModes mode, ContourApproximationModes method)
    {
        Cv2.FindContours(_binary, out Point[][] contours, out _, mode, method);
        return contours;
    }
}
