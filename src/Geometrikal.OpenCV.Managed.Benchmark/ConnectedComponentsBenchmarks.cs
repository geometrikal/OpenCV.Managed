using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

[MemoryDiagnoser]
public class ConnectedComponentsBenchmarks
{
    private Mat<byte> _binary = null!;
    private Mat<int> _labels = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _binary = Inputs.Binary(Size);
        _labels = new Mat<int>(Size, Size);
    }

    [Benchmark]
    public int Default8_S32() => ConnectedComponentLabeling.ConnectedComponents<int>(_binary).LabelCount;

    /// <summary>Preallocated labels; measures the labeling without the result allocation.</summary>
    [Benchmark]
    public int Default8Into_S32() => ConnectedComponentLabeling.ConnectedComponents(_binary, _labels);

    [Benchmark]
    public int Wu8_S32()
        => ConnectedComponentLabeling.ConnectedComponents<int>(_binary, 8, ConnectedComponentsAlgorithm.Wu).LabelCount;

    [Benchmark]
    public int Grana8_S32()
        => ConnectedComponentLabeling.ConnectedComponents<int>(_binary, 8, ConnectedComponentsAlgorithm.Grana).LabelCount;

    [Benchmark]
    public int Wu4_S32()
        => ConnectedComponentLabeling.ConnectedComponents<int>(_binary, 4, ConnectedComponentsAlgorithm.Wu).LabelCount;

    [Benchmark]
    public int Default8_U16() => ConnectedComponentLabeling.ConnectedComponents<ushort>(_binary).LabelCount;

    [Benchmark]
    public int StatsDefault8_S32() => ConnectedComponentLabeling.ConnectedComponentsWithStats<int>(_binary).LabelCount;

    [Benchmark]
    public int StatsWu8_S32()
        => ConnectedComponentLabeling.ConnectedComponentsWithStats<int>(_binary, 8, ConnectedComponentsAlgorithm.Wu).LabelCount;
}
