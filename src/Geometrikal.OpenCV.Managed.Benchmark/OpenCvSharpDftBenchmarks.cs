using BenchmarkDotNet.Attributes;
using OpenCvSharp;
using Port = Geometrikal.OpenCV.Managed.Core;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>The <see cref="DftBenchmarks"/> cases run through OpenCvSharp.</summary>
[MemoryDiagnoser]
public class OpenCvSharpDftBenchmarks
{
    private Mat _float = null!;
    private Mat _double = null!;
    private Mat _complex = null!;
    private Mat _ccs = null!;
    private Mat _ccs2 = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        Port.Mat<byte> gray = Inputs.Gray(Size);
        Port.Mat<float> grayFloat = Inputs.GrayFloat(Size);
        Port.Mat<float> inverted = Port.Conversions.ConvertTo<byte, float>(Port.Bitwise.Not(gray));
        _float = OcvInputs.ToOcv(grayFloat);
        _double = OcvInputs.ToOcv(Port.Conversions.ConvertTo<byte, double>(gray));
        _complex = OcvInputs.ToOcv(Port.Conversions.Merge([grayFloat, inverted]));
        _ccs = OcvInputs.ToOcv(Port.SpectralTransforms.Dft(grayFloat));
        _ccs2 = OcvInputs.ToOcv(Port.SpectralTransforms.Dft(inverted));
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (Mat m in new[] { _float, _double, _complex, _ccs, _ccs2 })
        {
            m.Dispose();
        }
    }

    [Benchmark]
    public int RealCcs_F32() => Dft(_float, DftFlags.None);

    [Benchmark]
    public int RealCcs_F64() => Dft(_double, DftFlags.None);

    [Benchmark]
    public int RealComplexOutput_F32() => Dft(_float, DftFlags.ComplexOutput);

    [Benchmark]
    public int Complex_F32() => Dft(_complex, DftFlags.None);

    [Benchmark]
    public int InverseCcsScale_F32()
    {
        using var dst = new Mat();
        Cv2.Idft(_ccs, dst, DftFlags.Scale);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int Rows_F32() => Dft(_float, DftFlags.Rows);

    [Benchmark]
    public int Dct_F32() => Dct(_float);

    [Benchmark]
    public int Dct_F64() => Dct(_double);

    [Benchmark]
    public int MulSpectrums_F32()
    {
        using var dst = new Mat();
        Cv2.MulSpectrums(_ccs, _ccs2, dst, DftFlags.None, conjB: true);
        return OcvInputs.Consume(dst);
    }

    private static int Dft(Mat source, DftFlags flags)
    {
        using var dst = new Mat();
        Cv2.Dft(source, dst, flags);
        return OcvInputs.Consume(dst);
    }

    private static int Dct(Mat source)
    {
        using var dst = new Mat();
        Cv2.Dct(source, dst);
        return OcvInputs.Consume(dst);
    }
}
