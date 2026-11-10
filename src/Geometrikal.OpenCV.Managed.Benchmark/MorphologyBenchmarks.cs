using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

[MemoryDiagnoser]
public class MorphologyBenchmarks
{
    private Mat<byte> _gray = null!;
    private Mat<byte> _color = null!;
    private Mat<sbyte> _rect3 = null!;
    private Mat<sbyte> _ellipse5 = null!;
    private Mat<sbyte> _rect15 = null!;
    private Mat<sbyte> _ellipse15 = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _gray = Inputs.Gray(Size);
        _color = Inputs.Color(Size);
        _rect3 = Morphology.GetStructuringElement(MorphShape.Rect, new Size(3, 3));
        _ellipse5 = Morphology.GetStructuringElement(MorphShape.Ellipse, new Size(5, 5));
        _rect15 = Morphology.GetStructuringElement(MorphShape.Rect, new Size(15, 15));
        _ellipse15 = Morphology.GetStructuringElement(MorphShape.Ellipse, new Size(15, 15));
    }

    [Benchmark]
    public Mat<byte> ErodeRect3_U8() => Morphology.Erode(_gray, _rect3);

    [Benchmark]
    public Mat<byte> ErodeRect3_U8C3() => Morphology.Erode(_color, _rect3);

    [Benchmark]
    public Mat<byte> DilateEllipse5_U8() => Morphology.Dilate(_gray, _ellipse5);

    [Benchmark]
    public Mat<byte> DilateRect3Iter3_U8() => Morphology.Dilate(_gray, _rect3, iterations: 3);

    [Benchmark]
    public Mat<byte> OpenEllipse5_U8() => Morphology.MorphologyEx(_gray, MorphOperation.Open, _ellipse5);

    [Benchmark]
    public Mat<byte> GradientRect3_U8() => Morphology.MorphologyEx(_gray, MorphOperation.Gradient, _rect3);

    [Benchmark]
    public Mat<byte> ErodeRect15_U8() => Morphology.Erode(_gray, _rect15);

    [Benchmark]
    public Mat<byte> DilateEllipse15_U8() => Morphology.Dilate(_gray, _ellipse15);
}
