using BenchmarkDotNet.Attributes;
using OpenCvSharp;
using Ocv = OpenCvSharp;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// The <see cref="MorphologyBenchmarks"/> cases run through OpenCvSharp (native OpenCV, default
/// build of the Windows runtime package). Each call allocates its result, as the managed port does.
/// </summary>
[MemoryDiagnoser]
public class OpenCvSharpMorphologyBenchmarks
{
    private Mat _gray = null!;
    private Mat _color = null!;
    private Mat _rect3 = null!;
    private Mat _ellipse5 = null!;
    private Mat _rect15 = null!;
    private Mat _ellipse15 = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        _gray = OcvInputs.ToOcv(Inputs.Gray(Size));
        _color = OcvInputs.ToOcv(Inputs.Color(Size));
        _rect3 = Cv2.GetStructuringElement(MorphShapes.Rect, new Ocv.Size(3, 3));
        _ellipse5 = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Ocv.Size(5, 5));
        _rect15 = Cv2.GetStructuringElement(MorphShapes.Rect, new Ocv.Size(15, 15));
        _ellipse15 = Cv2.GetStructuringElement(MorphShapes.Ellipse, new Ocv.Size(15, 15));
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (Mat m in new[] { _gray, _color, _rect3, _ellipse5, _rect15, _ellipse15 })
        {
            m.Dispose();
        }
    }

    [Benchmark]
    public int ErodeRect3_U8()
    {
        using var dst = new Mat();
        Cv2.Erode(_gray, dst, _rect3);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int ErodeRect3_U8C3()
    {
        using var dst = new Mat();
        Cv2.Erode(_color, dst, _rect3);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int DilateEllipse5_U8()
    {
        using var dst = new Mat();
        Cv2.Dilate(_gray, dst, _ellipse5);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int DilateRect3Iter3_U8()
    {
        using var dst = new Mat();
        Cv2.Dilate(_gray, dst, _rect3, iterations: 3);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int OpenEllipse5_U8()
    {
        using var dst = new Mat();
        Cv2.MorphologyEx(_gray, dst, MorphTypes.Open, _ellipse5);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int GradientRect3_U8()
    {
        using var dst = new Mat();
        Cv2.MorphologyEx(_gray, dst, MorphTypes.Gradient, _rect3);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int ErodeRect15_U8()
    {
        using var dst = new Mat();
        Cv2.Erode(_gray, dst, _rect15);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int DilateEllipse15_U8()
    {
        using var dst = new Mat();
        Cv2.Dilate(_gray, dst, _ellipse15);
        return OcvInputs.Consume(dst);
    }
}
