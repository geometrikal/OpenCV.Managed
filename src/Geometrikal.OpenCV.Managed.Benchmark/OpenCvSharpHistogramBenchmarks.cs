using BenchmarkDotNet.Attributes;
using OpenCvSharp;
using Ocv = OpenCvSharp;
using Port = Geometrikal.OpenCV.Managed.Core;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>The <see cref="HistogramBenchmarks"/> cases run through OpenCvSharp.</summary>
[MemoryDiagnoser]
public class OpenCvSharpHistogramBenchmarks
{
    private static readonly int[] Channel0 = [0];
    private static readonly int[] Channels01 = [0, 1];
    private static readonly int[] Channels012 = [0, 1, 2];
    private static readonly int[] Size256 = [256];
    private static readonly int[] Size8x8x8 = [8, 8, 8];
    private static readonly int[] Size32x32x32 = [32, 32, 32];
    private static readonly int[] Size30x32 = [30, 32];
    private static readonly Rangef[] Range256 = [new(0, 256)];
    private static readonly Rangef[] Range3D = [new(0, 256), new(0, 256), new(0, 256)];
    private static readonly Rangef[] RangeHs = [new(0, 180), new(0, 256)];

    private Mat _gray = null!;
    private Mat _color = null!;
    private Mat _colorInverted = null!;
    private Mat _float = null!;
    private Mat[] _grayImages = null!;
    private Mat[] _colorImages = null!;
    private Mat[] _floatImages = null!;
    private Mat _hist3D = null!;
    private Mat _hist3DOther = null!;
    private Mat _histHs = null!;
    private CLAHE _clahe = null!;
    private Mat _integralSum = null!;
    private Mat _integralSqsum = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        Port.Mat<byte> color = Inputs.Color(Size);
        _gray = OcvInputs.ToOcv(Inputs.Gray(Size));
        _color = OcvInputs.ToOcv(color);
        _colorInverted = OcvInputs.ToOcv(Port.Bitwise.Not(color));
        _float = OcvInputs.ToOcv(Inputs.GrayFloat(Size));
        _grayImages = [_gray];
        _colorImages = [_color];
        _floatImages = [_float];
        _hist3D = new Mat();
        Cv2.CalcHist(_colorImages, Channels012, null, _hist3D, 3, Size32x32x32, Range3D);
        _hist3DOther = new Mat();
        Cv2.CalcHist([_colorInverted], Channels012, null, _hist3DOther, 3, Size32x32x32, Range3D);
        _histHs = new Mat();
        Cv2.CalcHist(_colorImages, Channels01, null, _histHs, 2, Size30x32, RangeHs);
        _clahe = Cv2.CreateCLAHE(40, new Ocv.Size(8, 8));
        _integralSum = new Mat(Size + 1, Size + 1, MatType.CV_32SC1);
        _integralSqsum = new Mat(Size + 1, Size + 1, MatType.CV_64FC1);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (Mat m in new[] { _gray, _color, _colorInverted, _float, _hist3D, _hist3DOther, _histHs, _integralSum, _integralSqsum })
        {
            m.Dispose();
        }

        _clahe.Dispose();
    }

    [Benchmark]
    public int CalcHist_U8() => CalcHist(_grayImages, Channel0, Size256, Range256);

    [Benchmark]
    public int CalcHist3D_U8C3() => CalcHist(_colorImages, Channels012, Size8x8x8, Range3D);

    [Benchmark]
    public int CalcHist_F32() => CalcHist(_floatImages, Channel0, Size256, Range256);

    [Benchmark]
    public int EqualizeHist_U8()
    {
        using var dst = new Mat();
        Cv2.EqualizeHist(_gray, dst);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int Clahe_U8()
    {
        using var dst = new Mat();
        _clahe.Apply(_gray, dst);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int Integral_U8S32()
    {
        using var sum = new Mat();
        Cv2.Integral(_gray, sum, MatType.CV_32S);
        return OcvInputs.Consume(sum);
    }

    [Benchmark]
    public int IntegralSqsum_U8S32F64()
    {
        using var sum = new Mat();
        using var sqsum = new Mat();
        Cv2.Integral(_gray, sum, sqsum, MatType.CV_32S);
        return OcvInputs.Consume(sum) + OcvInputs.Consume(sqsum);
    }

    [Benchmark]
    public void Integral_U8S32_Into() => Cv2.Integral(_gray, _integralSum, MatType.CV_32S);

    [Benchmark]
    public void IntegralSqsum_U8S32F64_Into() => Cv2.Integral(_gray, _integralSum, _integralSqsum, MatType.CV_32S);

    [Benchmark]
    public double CompareHistChisqr_32x32x32() => Cv2.CompareHist(_hist3D, _hist3DOther, HistCompMethods.Chisqr);

    [Benchmark]
    public int CalcBackProject2D_U8C3()
    {
        using var dst = new Mat();
        Cv2.CalcBackProject(_colorImages, Channels01, _histHs, dst, RangeHs);
        return OcvInputs.Consume(dst);
    }

    private static int CalcHist(Mat[] images, int[] channels, int[] histSize, Rangef[] ranges)
    {
        using var hist = new Mat();
        Cv2.CalcHist(images, channels, null, hist, histSize.Length, histSize, ranges);
        return OcvInputs.Consume(hist);
    }
}
