using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

[MemoryDiagnoser]
public class HistogramBenchmarks
{
    private static readonly float[][] Range256 = [[0, 256]];
    private static readonly float[][] Range3D = [[0, 256], [0, 256], [0, 256]];
    private static readonly float[][] RangeHs = [[0, 180], [0, 256]];

    private Mat<byte> _gray = null!;
    private Mat<byte> _color = null!;
    private Mat<float> _float = null!;
    private DenseHistogram _hist3D = null!;
    private DenseHistogram _hist3DOther = null!;
    private DenseHistogram _histHs = null!;
    private Clahe _clahe = null!;
    private Mat<int> _integralSum = null!;
    private Mat<double> _integralSqsum = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _gray = Inputs.Gray(Size);
        _color = Inputs.Color(Size);
        _float = Inputs.GrayFloat(Size);
        _hist3D = Histograms.CalcHist([_color], [0, 1, 2], null, [32, 32, 32], Range3D);
        _hist3DOther = Histograms.CalcHist([Bitwise.Not(_color)], [0, 1, 2], null, [32, 32, 32], Range3D);
        _histHs = Histograms.CalcHist([_color], [0, 1], null, [30, 32], RangeHs);
        _clahe = new Clahe(40, new Size(8, 8));
        _integralSum = new Mat<int>(Size + 1, Size + 1);
        _integralSqsum = new Mat<double>(Size + 1, Size + 1);
    }

    [Benchmark]
    public DenseHistogram CalcHist_U8() => Histograms.CalcHist([_gray], [0], null, [256], Range256);

    [Benchmark]
    public DenseHistogram CalcHist3D_U8C3() => Histograms.CalcHist([_color], [0, 1, 2], null, [8, 8, 8], Range3D);

    [Benchmark]
    public DenseHistogram CalcHist_F32() => Histograms.CalcHist([_float], [0], null, [256], Range256);

    [Benchmark]
    public Mat<byte> EqualizeHist_U8() => Histograms.EqualizeHist(_gray);

    [Benchmark]
    public Mat<byte> Clahe_U8() => _clahe.Apply(_gray);

    [Benchmark]
    public Mat<int> Integral_U8S32() => IntegralImages.Integral<byte, int>(_gray);

    [Benchmark]
    public Mat<int> IntegralSqsum_U8S32F64() => IntegralImages.Integral<byte, int, double>(_gray, out _);

    [Benchmark]
    public void Integral_U8S32_Into() => IntegralImages.Integral(_gray, _integralSum);

    [Benchmark]
    public void IntegralSqsum_U8S32F64_Into() => IntegralImages.Integral(_gray, _integralSum, _integralSqsum);

    [Benchmark]
    public double CompareHistChisqr_32x32x32() => Histograms.CompareHist(_hist3D, _hist3DOther, HistCompMethods.Chisqr);

    [Benchmark]
    public Mat<byte> CalcBackProject2D_U8C3() => Histograms.CalcBackProject([_color], [0, 1], _histHs, RangeHs, 1.0);
}
