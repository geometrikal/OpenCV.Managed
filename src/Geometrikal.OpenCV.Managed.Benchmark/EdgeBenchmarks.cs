using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

[MemoryDiagnoser]
public class EdgeBenchmarks
{
    private Mat<byte> _gray = null!;
    private Mat<byte> _color = null!;
    private Mat<byte> _binary = null!;
    private Mat<short> _dx = null!;
    private Mat<short> _dy = null!;
    private Mat<ushort> _gray16 = null!;
    private Mat<double> _grayDouble = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _gray = Inputs.Gray(Size);
        _color = Inputs.Color(Size);
        _binary = Inputs.Binary(Size);
        _dx = Derivatives.Sobel<byte, short>(_gray, 1, 0, 3, 1, 0, BorderType.Replicate);
        _dy = Derivatives.Sobel<byte, short>(_gray, 0, 1, 3, 1, 0, BorderType.Replicate);
        _gray16 = Conversions.ConvertTo<byte, ushort>(_gray, 256);
        _grayDouble = Conversions.ConvertTo<byte, double>(_gray);
    }

    [Benchmark]
    public Mat<byte> Canny3_L1_U8() => EdgeDetection.Canny(_gray, 100, 300);

    [Benchmark]
    public Mat<byte> Canny3_L2_U8() => EdgeDetection.Canny(_gray, 100, 300, 3, true);

    [Benchmark]
    public Mat<byte> Canny5_L1_U8() => EdgeDetection.Canny(_gray, 1000, 3000, 5);

    [Benchmark]
    public Mat<byte> Canny5_L2_U8() => EdgeDetection.Canny(_gray, 1000, 3000, 5, true);

    [Benchmark]
    public Mat<byte> Canny3_L1_U8C3() => EdgeDetection.Canny(_color, 100, 300);

    [Benchmark]
    public Mat<byte> Canny5_L2_U8C3() => EdgeDetection.Canny(_color, 1000, 3000, 5, true);

    [Benchmark]
    public Mat<byte> Canny3_L1_U8Sparse() => EdgeDetection.Canny(_binary, 100, 300);

    [Benchmark]
    public Mat<byte> CannyDeriv_L2_S16() => EdgeDetection.Canny(_dx, _dy, 100, 300, true);

    [Benchmark]
    public Mat<short> SpatialGradient_U8()
    {
        EdgeDetection.SpatialGradient(_gray, out Mat<short> dx, out _);
        return dx;
    }

    [Benchmark]
    public Mat<float> Sobel3Dx_U16F32() => Derivatives.Sobel<ushort, float>(_gray16, 1, 0);

    [Benchmark]
    public Mat<double> Sobel3Dx_F64() => Derivatives.Sobel<double, double>(_grayDouble, 1, 0);
}
