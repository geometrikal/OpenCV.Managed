using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;

namespace Geometrikal.OpenCV.Managed.Benchmark;

[MemoryDiagnoser]
public class CoreBenchmarks
{
    private Mat<byte> _gray = null!;
    private Mat<byte> _gray2 = null!;
    private Mat<byte> _color = null!;
    private Mat<byte>[] _planes = null!;
    private Mat<float> _float = null!;
    private Mat<float> _float2 = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _gray = Inputs.Gray(Size);
        _gray2 = Bitwise.Not(_gray);
        _color = Inputs.Color(Size);
        _planes = Conversions.Split(_color);
        _float = Inputs.GrayFloat(Size);
        _float2 = Conversions.ConvertTo<byte, float>(_gray2);
    }

    [Benchmark]
    public Mat<byte> Add_U8() => Arithmetic.Add(_gray, _gray2);

    [Benchmark]
    public Mat<float> Add_F32() => Arithmetic.Add(_float, _float2);

    [Benchmark]
    public Mat<byte> AbsDiff_U8() => Arithmetic.AbsDiff(_gray, _gray2);

    [Benchmark]
    public Mat<byte> AddWeighted_U8() => Arithmetic.AddWeighted(_gray, 0.3, _gray2, 0.7, 1);

    [Benchmark]
    public Mat<byte> Multiply_U8() => Arithmetic.Multiply(_gray, _gray2, 0.5);

    [Benchmark]
    public Mat<byte> Compare_U8() => Comparison.Compare(_gray, _gray2, CmpType.Gt);

    [Benchmark]
    public Mat<byte> InRange_U8C3() => Comparison.InRange(_color, new Scalar(10, 20, 30), new Scalar(200, 210, 220));

    [Benchmark]
    public Mat<byte> ConvertScaleAbs_F32() => Conversions.ConvertScaleAbs(_float, 0.5);

    [Benchmark]
    public Mat<byte>[] Split_U8C3() => Conversions.Split(_color);

    [Benchmark]
    public Mat<byte> Merge_U8C3() => Conversions.Merge(_planes);

    [Benchmark]
    public Scalar Sum_U8() => Reductions.Sum(_gray);

    [Benchmark]
    public Scalar MeanStdDev_U8()
    {
        Reductions.MeanStdDev(_gray, out Scalar mean, out _);
        return mean;
    }

    [Benchmark]
    public MinMaxLocResult MinMaxLoc_U8() => Reductions.MinMaxLoc(_gray);

    [Benchmark]
    public int CountNonZero_U8() => Reductions.CountNonZero(_gray);

    [Benchmark]
    public double NormL2_F32() => Reductions.Norm(_float);
}
