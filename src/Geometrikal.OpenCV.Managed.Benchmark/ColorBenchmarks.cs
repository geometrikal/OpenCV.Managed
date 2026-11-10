using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

[MemoryDiagnoser]
public class ColorBenchmarks
{
    private Mat<byte> _color = null!;
    private Mat<byte> _hsv = null!;
    private Mat<byte> _lab = null!;
    private Mat<float> _colorF = null!;
    private Mat<float> _labF = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _color = Inputs.Color(Size);
        _hsv = ColorConversions.CvtColor(_color, ColorConversionCode.Bgr2Hsv);
        _lab = ColorConversions.CvtColor(_color, ColorConversionCode.Bgr2Lab);
        _colorF = Conversions.ConvertTo<byte, float>(_color, 1.0 / 255);
        _labF = ColorConversions.CvtColor(_colorF, ColorConversionCode.Bgr2Lab);
    }

    [Benchmark]
    public Mat<byte> BgrToGray_U8() => ColorConversions.CvtColor(_color, ColorConversionCode.Bgr2Gray);

    [Benchmark]
    public Mat<byte> BgrToRgb_U8() => ColorConversions.CvtColor(_color, ColorConversionCode.Bgr2Rgb);

    [Benchmark]
    public Mat<byte> BgrToBgra_U8() => ColorConversions.CvtColor(_color, ColorConversionCode.Bgr2Bgra);

    [Benchmark]
    public Mat<byte> BgrToHsv_U8() => ColorConversions.CvtColor(_color, ColorConversionCode.Bgr2Hsv);

    [Benchmark]
    public Mat<byte> HsvToBgr_U8() => ColorConversions.CvtColor(_hsv, ColorConversionCode.Hsv2Bgr);

    [Benchmark]
    public Mat<byte> BgrToLab_U8() => ColorConversions.CvtColor(_color, ColorConversionCode.Bgr2Lab);

    [Benchmark]
    public Mat<byte> LabToBgr_U8() => ColorConversions.CvtColor(_lab, ColorConversionCode.Lab2Bgr);

    [Benchmark]
    public Mat<byte> BgrToYCrCb_U8() => ColorConversions.CvtColor(_color, ColorConversionCode.Bgr2YCrCb);

    [Benchmark]
    public Mat<float> BgrToGray_F32() => ColorConversions.CvtColor(_colorF, ColorConversionCode.Bgr2Gray);

    [Benchmark]
    public Mat<float> BgrToHsv_F32() => ColorConversions.CvtColor(_colorF, ColorConversionCode.Bgr2Hsv);

    [Benchmark]
    public Mat<float> BgrToLab_F32() => ColorConversions.CvtColor(_colorF, ColorConversionCode.Bgr2Lab);

    [Benchmark]
    public Mat<float> LabToBgr_F32() => ColorConversions.CvtColor(_labF, ColorConversionCode.Lab2Bgr);
}
