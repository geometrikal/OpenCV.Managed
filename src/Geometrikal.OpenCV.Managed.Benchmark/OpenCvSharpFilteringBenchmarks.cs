using BenchmarkDotNet.Attributes;
using OpenCvSharp;
using Ocv = OpenCvSharp;
using Port = Geometrikal.OpenCV.Managed.Core;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>The <see cref="FilteringBenchmarks"/> cases run through OpenCvSharp.</summary>
[MemoryDiagnoser]
public class OpenCvSharpFilteringBenchmarks
{
    private static readonly Ocv.Size K5 = new(5, 5);
    private static readonly Ocv.Size K15 = new(15, 15);

    private Mat _gray = null!;
    private Mat _color = null!;
    private Mat _grayFloat = null!;
    private Mat _sharpen = null!;
    private Mat _box11 = null!;
    private Mat _binomial5 = null!;
    private Mat _borderDst = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        _gray = OcvInputs.ToOcv(Inputs.Gray(Size));
        _color = OcvInputs.ToOcv(Inputs.Color(Size));
        _grayFloat = OcvInputs.ToOcv(Inputs.GrayFloat(Size));
        _sharpen = OcvInputs.ToOcv(new Port.Mat<float>([0, -1, 0, -1, 5, -1, 0, -1, 0], 3, 3));
        _box11 = OcvInputs.ToOcv(new Port.Mat<float>(Enumerable.Repeat(1f / 128, 121).ToArray(), 11, 11));
        _binomial5 = OcvInputs.ToOcv(new Port.Mat<float>([1f / 16, 4f / 16, 6f / 16, 4f / 16, 1f / 16], 5, 1));
        _borderDst = new Mat(Size + 32, Size + 32, MatType.CV_8UC3);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (Mat m in new[] { _gray, _color, _grayFloat, _sharpen, _box11, _binomial5, _borderDst })
        {
            m.Dispose();
        }
    }

    [Benchmark]
    public int GaussianBlur5x5_U8() => Gaussian(_gray, K5, 1.5);

    [Benchmark]
    public int GaussianBlur5x5_U8C3() => Gaussian(_color, K5, 1.5);

    [Benchmark]
    public int GaussianBlur5x5_F32() => Gaussian(_grayFloat, K5, 1.5);

    [Benchmark]
    public int Blur5x5_U8() => Blur(K5);

    [Benchmark]
    public int MedianBlur3_U8() => Median(3);

    [Benchmark]
    public int MedianBlur5_U8() => Median(5);

    [Benchmark]
    public int MedianBlur7_U8() => Median(7);

    [Benchmark]
    public int Filter2D3x3_U8() => Filter2D(_sharpen);

    [Benchmark]
    public int SepFilter2D5_U8F32()
    {
        using var dst = new Mat();
        Cv2.SepFilter2D(_gray, dst, MatType.CV_32F, _binomial5, _binomial5);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int Sobel3Dx_U8S16() => Sobel(_gray, MatType.CV_16S, 1, 0, 3);

    [Benchmark]
    public int Sobel3Dxy_F32() => Sobel(_grayFloat, MatType.CV_32F, 1, 1, 3);

    [Benchmark]
    public int ScharrDx_U8S16()
    {
        using var dst = new Mat();
        Cv2.Scharr(_gray, dst, MatType.CV_16S, 1, 0);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int Laplacian3_U8S16()
    {
        using var dst = new Mat();
        Cv2.Laplacian(_gray, dst, MatType.CV_16S, 3);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int GaussianBlur15x15_U8() => Gaussian(_gray, K15, 3.0);

    [Benchmark]
    public int GaussianBlur15x15_F32() => Gaussian(_grayFloat, K15, 3.0);

    [Benchmark]
    public int Blur15x15_U8() => Blur(K15);

    [Benchmark]
    public int MedianBlur15_U8() => Median(15);

    [Benchmark]
    public int Filter2D11x11_U8() => Filter2D(_box11);

    [Benchmark]
    public int Sobel7Dx_U8S16() => Sobel(_gray, MatType.CV_16S, 1, 0, 7);

    [Benchmark]
    public int CopyMakeBorder16_U8C3()
    {
        using var dst = new Mat();
        Cv2.CopyMakeBorder(_color, dst, 16, 16, 16, 16, BorderTypes.Reflect101);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int CopyMakeBorderInto16_U8C3()
    {
        Cv2.CopyMakeBorder(_color, _borderDst, 16, 16, 16, 16, BorderTypes.Reflect101);
        return OcvInputs.Consume(_borderDst);
    }

    private static int Gaussian(Mat source, Ocv.Size ksize, double sigma)
    {
        using var dst = new Mat();
        Cv2.GaussianBlur(source, dst, ksize, sigma);
        return OcvInputs.Consume(dst);
    }

    private int Blur(Ocv.Size ksize)
    {
        using var dst = new Mat();
        Cv2.Blur(_gray, dst, ksize);
        return OcvInputs.Consume(dst);
    }

    private int Median(int ksize)
    {
        using var dst = new Mat();
        Cv2.MedianBlur(_gray, dst, ksize);
        return OcvInputs.Consume(dst);
    }

    private int Filter2D(Mat kernel)
    {
        using var dst = new Mat();
        Cv2.Filter2D(_gray, dst, MatType.CV_8U, kernel);
        return OcvInputs.Consume(dst);
    }

    private static int Sobel(Mat source, MatType depth, int dx, int dy, int ksize)
    {
        using var dst = new Mat();
        Cv2.Sobel(source, dst, depth, dx, dy, ksize);
        return OcvInputs.Consume(dst);
    }
}
