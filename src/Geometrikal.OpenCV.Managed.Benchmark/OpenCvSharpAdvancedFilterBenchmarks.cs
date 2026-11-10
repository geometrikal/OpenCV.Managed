using BenchmarkDotNet.Attributes;
using OpenCvSharp;
using OpenCvSharp.XImgProc;
using Ocv = OpenCvSharp;
using Port = Geometrikal.OpenCV.Managed.Core;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// The <see cref="AdvancedFilterBenchmarks"/> cases run through OpenCvSharp. OpenCvSharp does not
/// expose <c>stackBlur</c>, so the StackBlur cases are absent.
/// </summary>
[MemoryDiagnoser]
public class OpenCvSharpAdvancedFilterBenchmarks
{
    private Mat _gray = null!;
    private Mat _color = null!;
    private Mat _grayFloat = null!;
    private Mat _box15 = null!;
    private Mat _box31 = null!;
    private Ocv.XImgProc.GuidedFilter _guidedGray = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        _gray = OcvInputs.ToOcv(Inputs.Gray(Size));
        _color = OcvInputs.ToOcv(Inputs.Color(Size));
        _grayFloat = OcvInputs.ToOcv(Inputs.GrayFloat(Size));
        _box15 = OcvInputs.ToOcv(new Port.Mat<float>(Enumerable.Repeat(1f / 256, 225).ToArray(), 15, 15));
        _box31 = OcvInputs.ToOcv(new Port.Mat<float>(Enumerable.Repeat(1f / 1024, 961).ToArray(), 31, 31));
        _guidedGray = CvXImgProc.CreateGuidedFilter(_gray, 8, 400);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (Mat m in new[] { _gray, _color, _grayFloat, _box15, _box31 })
        {
            m.Dispose();
        }

        _guidedGray.Dispose();
    }

    [Benchmark]
    public int Filter2D15x15_U8() => Filter2D(_gray, MatType.CV_8U, _box15);

    [Benchmark]
    public int Filter2D15x15_F32() => Filter2D(_grayFloat, MatType.CV_32F, _box15);

    [Benchmark]
    public int Filter2D31x31_U8() => Filter2D(_gray, MatType.CV_8U, _box31);

    [Benchmark]
    public int Filter2D31x31_F32() => Filter2D(_grayFloat, MatType.CV_32F, _box31);

    [Benchmark]
    public int Bilateral9_U8() => Bilateral(_gray);

    [Benchmark]
    public int Bilateral9_U8C3() => Bilateral(_color);

    [Benchmark]
    public int Bilateral9_F32() => Bilateral(_grayFloat);

    [Benchmark]
    public int GuidedFilter8_Gray() => Guided(_gray, _gray);

    [Benchmark]
    public int GuidedFilter8_ColorGuide() => Guided(_color, _gray);

    [Benchmark]
    public int GuidedFilter8_GrayPrepared()
    {
        using var dst = new Mat();
        _guidedGray.Filter(_gray, dst);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int SqrBoxFilter5x5_U8() => SqrBox(_gray);

    [Benchmark]
    public int SqrBoxFilter5x5_F32() => SqrBox(_grayFloat);

    private static int Filter2D(Mat source, MatType depth, Mat kernel)
    {
        using var dst = new Mat();
        Cv2.Filter2D(source, dst, depth, kernel);
        return OcvInputs.Consume(dst);
    }

    private static int Bilateral(Mat source)
    {
        using var dst = new Mat();
        Cv2.BilateralFilter(source, dst, 9, 50, 5);
        return OcvInputs.Consume(dst);
    }

    private static int Guided(Mat guide, Mat source)
    {
        using var dst = new Mat();
        CvXImgProc.GuidedFilter(guide, source, dst, 8, 400);
        return OcvInputs.Consume(dst);
    }

    private static int SqrBox(Mat source)
    {
        using var dst = new Mat();
        Cv2.SqrBoxFilter(source, dst, -1, new Ocv.Size(5, 5));
        return OcvInputs.Consume(dst);
    }
}
