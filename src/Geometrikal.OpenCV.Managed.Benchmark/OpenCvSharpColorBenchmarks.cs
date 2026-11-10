using BenchmarkDotNet.Attributes;
using OpenCvSharp;
using Port = Geometrikal.OpenCV.Managed.Core;
using PortImgProc = Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>The <see cref="ColorBenchmarks"/> cases run through OpenCvSharp.</summary>
[MemoryDiagnoser]
public class OpenCvSharpColorBenchmarks
{
    private Mat _color = null!;
    private Mat _hsv = null!;
    private Mat _lab = null!;
    private Mat _colorF = null!;
    private Mat _labF = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        Port.Mat<byte> color = Inputs.Color(Size);
        Port.Mat<float> colorF = Port.Conversions.ConvertTo<byte, float>(color, 1.0 / 255);
        _color = OcvInputs.ToOcv(color);
        _hsv = OcvInputs.ToOcv(PortImgProc.ColorConversions.CvtColor(color, PortImgProc.ColorConversionCode.Bgr2Hsv));
        _lab = OcvInputs.ToOcv(PortImgProc.ColorConversions.CvtColor(color, PortImgProc.ColorConversionCode.Bgr2Lab));
        _colorF = OcvInputs.ToOcv(colorF);
        _labF = OcvInputs.ToOcv(PortImgProc.ColorConversions.CvtColor(colorF, PortImgProc.ColorConversionCode.Bgr2Lab));
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (Mat m in new[] { _color, _hsv, _lab, _colorF, _labF })
        {
            m.Dispose();
        }
    }

    [Benchmark]
    public int BgrToGray_U8() => Convert(_color, ColorConversionCodes.BGR2GRAY);

    [Benchmark]
    public int BgrToRgb_U8() => Convert(_color, ColorConversionCodes.BGR2RGB);

    [Benchmark]
    public int BgrToBgra_U8() => Convert(_color, ColorConversionCodes.BGR2BGRA);

    [Benchmark]
    public int BgrToHsv_U8() => Convert(_color, ColorConversionCodes.BGR2HSV);

    [Benchmark]
    public int HsvToBgr_U8() => Convert(_hsv, ColorConversionCodes.HSV2BGR);

    [Benchmark]
    public int BgrToLab_U8() => Convert(_color, ColorConversionCodes.BGR2Lab);

    [Benchmark]
    public int LabToBgr_U8() => Convert(_lab, ColorConversionCodes.Lab2BGR);

    [Benchmark]
    public int BgrToYCrCb_U8() => Convert(_color, ColorConversionCodes.BGR2YCrCb);

    [Benchmark]
    public int BgrToGray_F32() => Convert(_colorF, ColorConversionCodes.BGR2GRAY);

    [Benchmark]
    public int BgrToHsv_F32() => Convert(_colorF, ColorConversionCodes.BGR2HSV);

    [Benchmark]
    public int BgrToLab_F32() => Convert(_colorF, ColorConversionCodes.BGR2Lab);

    [Benchmark]
    public int LabToBgr_F32() => Convert(_labF, ColorConversionCodes.Lab2BGR);

    private static int Convert(Mat source, ColorConversionCodes code)
    {
        using var dst = new Mat();
        Cv2.CvtColor(source, dst, code);
        return OcvInputs.Consume(dst);
    }
}
