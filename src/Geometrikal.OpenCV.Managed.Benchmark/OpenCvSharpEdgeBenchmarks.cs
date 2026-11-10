using BenchmarkDotNet.Attributes;
using OpenCvSharp;
using Port = Geometrikal.OpenCV.Managed.Core;
using PortImgProc = Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>The <see cref="EdgeBenchmarks"/> cases run through OpenCvSharp.</summary>
[MemoryDiagnoser]
public class OpenCvSharpEdgeBenchmarks
{
    private Mat _gray = null!;
    private Mat _color = null!;
    private Mat _binary = null!;
    private Mat _dx = null!;
    private Mat _dy = null!;
    private Mat _gray16 = null!;
    private Mat _grayDouble = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        Port.Mat<byte> gray = Inputs.Gray(Size);
        _gray = OcvInputs.ToOcv(gray);
        _color = OcvInputs.ToOcv(Inputs.Color(Size));
        _binary = OcvInputs.ToOcv(Inputs.Binary(Size));
        _dx = OcvInputs.ToOcv(PortImgProc.Derivatives.Sobel<byte, short>(gray, 1, 0, 3, 1, 0, Port.BorderType.Replicate));
        _dy = OcvInputs.ToOcv(PortImgProc.Derivatives.Sobel<byte, short>(gray, 0, 1, 3, 1, 0, Port.BorderType.Replicate));
        _gray16 = OcvInputs.ToOcv(Port.Conversions.ConvertTo<byte, ushort>(gray, 256));
        _grayDouble = OcvInputs.ToOcv(Port.Conversions.ConvertTo<byte, double>(gray));
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (Mat m in new[] { _gray, _color, _binary, _dx, _dy, _gray16, _grayDouble })
        {
            m.Dispose();
        }
    }

    [Benchmark]
    public int Canny3_L1_U8() => Canny(_gray, 100, 300, 3, false);

    [Benchmark]
    public int Canny3_L2_U8() => Canny(_gray, 100, 300, 3, true);

    [Benchmark]
    public int Canny5_L1_U8() => Canny(_gray, 1000, 3000, 5, false);

    [Benchmark]
    public int Canny5_L2_U8() => Canny(_gray, 1000, 3000, 5, true);

    [Benchmark]
    public int Canny3_L1_U8C3() => Canny(_color, 100, 300, 3, false);

    [Benchmark]
    public int Canny5_L2_U8C3() => Canny(_color, 1000, 3000, 5, true);

    [Benchmark]
    public int Canny3_L1_U8Sparse() => Canny(_binary, 100, 300, 3, false);

    [Benchmark]
    public int CannyDeriv_L2_S16()
    {
        using var dst = new Mat();
        Cv2.Canny(_dx, _dy, dst, 100, 300, true);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int SpatialGradient_U8()
    {
        using var dx = new Mat();
        using var dy = new Mat();
        Cv2.SpatialGradient(_gray, dx, dy);
        return OcvInputs.Consume(dx) + OcvInputs.Consume(dy);
    }

    [Benchmark]
    public int Sobel3Dx_U16F32()
    {
        using var dst = new Mat();
        Cv2.Sobel(_gray16, dst, MatType.CV_32F, 1, 0);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int Sobel3Dx_F64()
    {
        using var dst = new Mat();
        Cv2.Sobel(_grayDouble, dst, MatType.CV_64F, 1, 0);
        return OcvInputs.Consume(dst);
    }

    private static int Canny(Mat source, double threshold1, double threshold2, int apertureSize, bool l2Gradient)
    {
        using var dst = new Mat();
        Cv2.Canny(source, dst, threshold1, threshold2, apertureSize, l2Gradient);
        return OcvInputs.Consume(dst);
    }
}
