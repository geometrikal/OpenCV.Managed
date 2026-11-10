using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;

namespace Geometrikal.OpenCV.Managed.Benchmark;

[MemoryDiagnoser]
public class DftBenchmarks
{
    private Mat<float> _float = null!;
    private Mat<double> _double = null!;
    private Mat<float> _complex = null!;
    private Mat<float> _ccs = null!;
    private Mat<float> _ccs2 = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        Mat<byte> gray = Inputs.Gray(Size);
        _float = Inputs.GrayFloat(Size);
        _double = Conversions.ConvertTo<byte, double>(gray);
        Mat<float> inverted = Conversions.ConvertTo<byte, float>(Bitwise.Not(gray));
        _complex = Conversions.Merge([_float, inverted]);
        _ccs = SpectralTransforms.Dft(_float);
        _ccs2 = SpectralTransforms.Dft(inverted);
    }

    [Benchmark]
    public Mat<float> RealCcs_F32() => SpectralTransforms.Dft(_float);

    [Benchmark]
    public Mat<double> RealCcs_F64() => SpectralTransforms.Dft(_double);

    [Benchmark]
    public Mat<float> RealComplexOutput_F32() => SpectralTransforms.Dft(_float, DftFlags.ComplexOutput);

    [Benchmark]
    public Mat<float> Complex_F32() => SpectralTransforms.Dft(_complex);

    [Benchmark]
    public Mat<float> InverseCcsScale_F32() => SpectralTransforms.Idft(_ccs, DftFlags.Scale);

    [Benchmark]
    public Mat<float> Rows_F32() => SpectralTransforms.Dft(_float, DftFlags.Rows);

    [Benchmark]
    public Mat<float> Dct_F32() => SpectralTransforms.Dct(_float);

    [Benchmark]
    public Mat<double> Dct_F64() => SpectralTransforms.Dct(_double);

    [Benchmark]
    public Mat<float> MulSpectrums_F32() => SpectralTransforms.MulSpectrums(_ccs, _ccs2, DftFlags.None, conjB: true);
}
