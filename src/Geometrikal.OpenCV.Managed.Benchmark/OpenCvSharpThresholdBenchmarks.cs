using BenchmarkDotNet.Attributes;
using OpenCvSharp;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>The <see cref="ThresholdBenchmarks"/> cases run through OpenCvSharp.</summary>
[MemoryDiagnoser]
public class OpenCvSharpThresholdBenchmarks
{
    private Mat _gray = null!;
    private Mat _float = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        _gray = OcvInputs.ToOcv(Inputs.Gray(Size));
        _float = OcvInputs.ToOcv(Inputs.GrayFloat(Size));
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _gray.Dispose();
        _float.Dispose();
    }

    [Benchmark]
    public int Binary_U8() => Threshold(_gray, 127, 255, ThresholdTypes.Binary);

    [Benchmark]
    public int Otsu_U8() => Threshold(_gray, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);

    [Benchmark]
    public int Triangle_U8() => Threshold(_gray, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Triangle);

    [Benchmark]
    public int Trunc_F32() => Threshold(_float, 100.5, 0, ThresholdTypes.Trunc);

    [Benchmark]
    public int AdaptiveMean11_U8()
    {
        using var dst = new Mat();
        Cv2.AdaptiveThreshold(_gray, dst, 255, AdaptiveThresholdTypes.MeanC, ThresholdTypes.Binary, 11, 2);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int AdaptiveGaussian11_U8()
    {
        using var dst = new Mat();
        Cv2.AdaptiveThreshold(_gray, dst, 255, AdaptiveThresholdTypes.GaussianC, ThresholdTypes.Binary, 11, 2);
        return OcvInputs.Consume(dst);
    }

    private static int Threshold(Mat source, double thresh, double maxValue, ThresholdTypes type)
    {
        using var dst = new Mat();
        Cv2.Threshold(source, dst, thresh, maxValue, type);
        return OcvInputs.Consume(dst);
    }
}
