using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

[MemoryDiagnoser]
public class PyramidBenchmarks
{
    private Mat<byte> _gray = null!;
    private Mat<byte> _color = null!;
    private Mat<float> _grayFloat = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _gray = Inputs.Gray(Size);
        _color = Inputs.Color(Size);
        _grayFloat = Inputs.GrayFloat(Size);
    }

    [Benchmark]
    public Mat<byte> PyrDown_U8() => Pyramids.PyrDown(_gray);

    [Benchmark]
    public Mat<byte> PyrDown_U8C3() => Pyramids.PyrDown(_color);

    [Benchmark]
    public Mat<float> PyrDown_F32() => Pyramids.PyrDown(_grayFloat);

    [Benchmark]
    public Mat<byte> PyrUp_U8() => Pyramids.PyrUp(_gray);

    [Benchmark]
    public Mat<byte> PyrUp_U8C3() => Pyramids.PyrUp(_color);

    [Benchmark]
    public Mat<float> PyrUp_F32() => Pyramids.PyrUp(_grayFloat);

    [Benchmark]
    public Mat<byte>[] BuildPyramid4_U8() => Pyramids.BuildPyramid(_gray, 4);

    [Benchmark]
    public Mat<byte>[] BuildPyramid4_U8C3() => Pyramids.BuildPyramid(_color, 4);
}
