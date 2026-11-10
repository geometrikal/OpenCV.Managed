using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

[MemoryDiagnoser]
public class FilteringBenchmarks
{
    private static readonly float[] Binomial5 = [1f / 16, 4f / 16, 6f / 16, 4f / 16, 1f / 16];

    private Mat<byte> _gray = null!;
    private Mat<byte> _color = null!;
    private Mat<float> _grayFloat = null!;
    private Mat<float> _sharpen = null!;
    private Mat<float> _box11 = null!;
    private Mat<byte> _borderDst = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _gray = Inputs.Gray(Size);
        _color = Inputs.Color(Size);
        _grayFloat = Inputs.GrayFloat(Size);
        _sharpen = new Mat<float>([0, -1, 0, -1, 5, -1, 0, -1, 0], 3, 3);
        _box11 = new Mat<float>(Enumerable.Repeat(1f / 128, 121).ToArray(), 11, 11);
        _borderDst = new Mat<byte>(Size + 32, Size + 32, 3);
    }

    [Benchmark]
    public Mat<byte> GaussianBlur5x5_U8() => Filters.GaussianBlur(_gray, new Size(5, 5), 1.5);

    [Benchmark]
    public Mat<byte> GaussianBlur5x5_U8C3() => Filters.GaussianBlur(_color, new Size(5, 5), 1.5);

    [Benchmark]
    public Mat<float> GaussianBlur5x5_F32() => Filters.GaussianBlur(_grayFloat, new Size(5, 5), 1.5);

    [Benchmark]
    public Mat<byte> Blur5x5_U8() => Filters.Blur(_gray, new Size(5, 5));

    [Benchmark]
    public Mat<byte> MedianBlur3_U8() => Filters.MedianBlur(_gray, 3);

    [Benchmark]
    public Mat<byte> MedianBlur5_U8() => Filters.MedianBlur(_gray, 5);

    [Benchmark]
    public Mat<byte> MedianBlur7_U8() => Filters.MedianBlur(_gray, 7);

    [Benchmark]
    public Mat<byte> Filter2D3x3_U8() => Filters.Filter2D<byte, byte>(_gray, _sharpen);

    [Benchmark]
    public Mat<float> SepFilter2D5_U8F32() => Filters.SepFilter2D<byte, float>(_gray, Binomial5, Binomial5);

    [Benchmark]
    public Mat<short> Sobel3Dx_U8S16() => Derivatives.Sobel<byte, short>(_gray, 1, 0);

    [Benchmark]
    public Mat<float> Sobel3Dxy_F32() => Derivatives.Sobel<float, float>(_grayFloat, 1, 1);

    [Benchmark]
    public Mat<short> ScharrDx_U8S16() => Derivatives.Scharr<byte, short>(_gray, 1, 0);

    [Benchmark]
    public Mat<short> Laplacian3_U8S16() => Derivatives.Laplacian<byte, short>(_gray, 3);

    [Benchmark]
    public Mat<byte> GaussianBlur15x15_U8() => Filters.GaussianBlur(_gray, new Size(15, 15), 3.0);

    [Benchmark]
    public Mat<float> GaussianBlur15x15_F32() => Filters.GaussianBlur(_grayFloat, new Size(15, 15), 3.0);

    [Benchmark]
    public Mat<byte> Blur15x15_U8() => Filters.Blur(_gray, new Size(15, 15));

    [Benchmark]
    public Mat<byte> MedianBlur15_U8() => Filters.MedianBlur(_gray, 15);

    [Benchmark]
    public Mat<byte> Filter2D11x11_U8() => Filters.Filter2D<byte, byte>(_gray, _box11);

    [Benchmark]
    public Mat<short> Sobel7Dx_U8S16() => Derivatives.Sobel<byte, short>(_gray, 1, 0, 7);

    [Benchmark]
    public Mat<byte> CopyMakeBorder16_U8C3() => Borders.CopyMakeBorder(_color, 16, 16, 16, 16, BorderType.Reflect101);

    [Benchmark]
    public Mat<byte> CopyMakeBorderInto16_U8C3()
    {
        Borders.CopyMakeBorder(_color, _borderDst, 16, 16, 16, 16, BorderType.Reflect101);
        return _borderDst;
    }
}
