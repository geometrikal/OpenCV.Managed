using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

[MemoryDiagnoser]
public class ContourBenchmarks
{
    private Mat<byte> _binary = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup() => _binary = Inputs.Binary(Size);

    [Benchmark]
    public ContourResult FindContoursTreeNone() =>
        Contours.FindContours(_binary, RetrievalMode.Tree, ContourApproximationMode.None);

    [Benchmark]
    public ContourResult FindContoursExternalSimple() =>
        Contours.FindContours(_binary, RetrievalMode.External, ContourApproximationMode.Simple);

    [Benchmark]
    public ContourResult FindContoursListTc89L1() =>
        Contours.FindContours(_binary, RetrievalMode.List, ContourApproximationMode.Tc89L1);
}
