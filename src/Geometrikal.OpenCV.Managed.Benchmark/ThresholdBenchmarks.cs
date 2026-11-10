using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

[MemoryDiagnoser]
public class ThresholdBenchmarks
{
    private Mat<byte> _gray = null!;
    private Mat<float> _float = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _gray = Inputs.Gray(Size);
        _float = Inputs.GrayFloat(Size);
    }

    [Benchmark]
    public Mat<byte> Binary_U8() => Thresholding.Threshold(_gray, 127, 255, ThresholdType.Binary);

    [Benchmark]
    public Mat<byte> Otsu_U8() => Thresholding.Threshold(_gray, 0, 255, ThresholdType.Binary | ThresholdType.Otsu);

    [Benchmark]
    public Mat<byte> Triangle_U8() => Thresholding.Threshold(_gray, 0, 255, ThresholdType.Binary | ThresholdType.Triangle);

    [Benchmark]
    public Mat<float> Trunc_F32() => Thresholding.Threshold(_float, 100.5, 0, ThresholdType.Trunc);

    [Benchmark]
    public Mat<byte> AdaptiveMean11_U8()
        => Thresholding.AdaptiveThreshold(_gray, 255, AdaptiveThresholdType.MeanC, ThresholdType.Binary, 11, 2);

    [Benchmark]
    public Mat<byte> AdaptiveGaussian11_U8()
        => Thresholding.AdaptiveThreshold(_gray, 255, AdaptiveThresholdType.GaussianC, ThresholdType.Binary, 11, 2);
}
