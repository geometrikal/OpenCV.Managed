using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

[MemoryDiagnoser]
public class AdvancedFilterBenchmarks
{
    private Mat<byte> _gray = null!;
    private Mat<byte> _color = null!;
    private Mat<float> _grayFloat = null!;
    private Mat<float> _box15 = null!;
    private Mat<float> _box31 = null!;
    private GuidedFilter _guidedGray = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _gray = Inputs.Gray(Size);
        _color = Inputs.Color(Size);
        _grayFloat = Inputs.GrayFloat(Size);
        _box15 = new Mat<float>(Enumerable.Repeat(1f / 256, 225).ToArray(), 15, 15);
        _box31 = new Mat<float>(Enumerable.Repeat(1f / 1024, 961).ToArray(), 31, 31);
        _guidedGray = GuidedFilter.Create(_gray, 8, 400);
    }

    [Benchmark]
    public Mat<byte> Filter2D15x15_U8() => Filters.Filter2D<byte, byte>(_gray, _box15);

    [Benchmark]
    public Mat<float> Filter2D15x15_F32() => Filters.Filter2D<float, float>(_grayFloat, _box15);

    [Benchmark]
    public Mat<byte> Filter2D31x31_U8() => Filters.Filter2D<byte, byte>(_gray, _box31);

    [Benchmark]
    public Mat<float> Filter2D31x31_F32() => Filters.Filter2D<float, float>(_grayFloat, _box31);

    [Benchmark]
    public Mat<byte> Bilateral9_U8() => Filters.BilateralFilter(_gray, 9, 50, 5);

    [Benchmark]
    public Mat<byte> Bilateral9_U8C3() => Filters.BilateralFilter(_color, 9, 50, 5);

    [Benchmark]
    public Mat<float> Bilateral9_F32() => Filters.BilateralFilter(_grayFloat, 9, 50, 5);

    [Benchmark]
    public Mat<byte> GuidedFilter8_Gray() => EdgePreservingFilters.GuidedFilter(_gray, _gray, 8, 400);

    [Benchmark]
    public Mat<byte> GuidedFilter8_ColorGuide() => EdgePreservingFilters.GuidedFilter(_color, _gray, 8, 400);

    [Benchmark]
    public Mat<byte> GuidedFilter8_GrayPrepared() => _guidedGray.Filter(_gray);

    [Benchmark]
    public Mat<float> SqrBoxFilter5x5_U8() => Filters.SqrBoxFilter(_gray, new Size(5, 5));

    [Benchmark]
    public Mat<double> SqrBoxFilter5x5_F32() => Filters.SqrBoxFilter(_grayFloat, new Size(5, 5));

    [Benchmark]
    public Mat<byte> StackBlur15x15_U8() => Filters.StackBlur(_gray, new Size(15, 15));

    [Benchmark]
    public Mat<byte> StackBlur15x15_U8C3() => Filters.StackBlur(_color, new Size(15, 15));

    [Benchmark]
    public Mat<float> StackBlur15x15_F32() => Filters.StackBlur(_grayFloat, new Size(15, 15));
}
