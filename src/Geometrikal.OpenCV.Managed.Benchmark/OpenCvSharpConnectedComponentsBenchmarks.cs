using BenchmarkDotNet.Attributes;
using OpenCvSharp;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>The <see cref="ConnectedComponentsBenchmarks"/> cases run through OpenCvSharp.</summary>
[MemoryDiagnoser]
public class OpenCvSharpConnectedComponentsBenchmarks
{
    private Mat _binary = null!;
    private Mat _labels = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        _binary = OcvInputs.ToOcv(Inputs.Binary(Size));
        _labels = new Mat(Size, Size, MatType.CV_32SC1);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _binary.Dispose();
        _labels.Dispose();
    }

    [Benchmark]
    public int Default8_S32()
        => Label(PixelConnectivity.Connectivity8, MatType.CV_32S, ConnectedComponentsAlgorithmsTypes.Default);

    /// <summary>Preallocated labels; measures the labeling without the result allocation.</summary>
    [Benchmark]
    public int Default8Into_S32()
        => Cv2.ConnectedComponentsWithAlgorithm(
            _binary, _labels, PixelConnectivity.Connectivity8, MatType.CV_32S, ConnectedComponentsAlgorithmsTypes.Default);

    [Benchmark]
    public int Wu8_S32()
        => Label(PixelConnectivity.Connectivity8, MatType.CV_32S, ConnectedComponentsAlgorithmsTypes.WU);

    [Benchmark]
    public int Grana8_S32()
        => Label(PixelConnectivity.Connectivity8, MatType.CV_32S, ConnectedComponentsAlgorithmsTypes.GRANA);

    [Benchmark]
    public int Wu4_S32()
        => Label(PixelConnectivity.Connectivity4, MatType.CV_32S, ConnectedComponentsAlgorithmsTypes.WU);

    [Benchmark]
    public int Default8_U16()
        => Label(PixelConnectivity.Connectivity8, MatType.CV_16U, ConnectedComponentsAlgorithmsTypes.Default);

    [Benchmark]
    public int StatsDefault8_S32() => LabelWithStats(ConnectedComponentsAlgorithmsTypes.Default);

    [Benchmark]
    public int StatsWu8_S32() => LabelWithStats(ConnectedComponentsAlgorithmsTypes.WU);

    private int Label(PixelConnectivity connectivity, MatType labelType, ConnectedComponentsAlgorithmsTypes algorithm)
    {
        using var labels = new Mat();
        return Cv2.ConnectedComponentsWithAlgorithm(_binary, labels, connectivity, labelType, algorithm);
    }

    private int LabelWithStats(ConnectedComponentsAlgorithmsTypes algorithm)
    {
        using var labels = new Mat();
        using var stats = new Mat();
        using var centroids = new Mat();
        return Cv2.ConnectedComponentsWithStatsWithAlgorithm(
            _binary, labels, stats, centroids, PixelConnectivity.Connectivity8, MatType.CV_32S, algorithm);
    }
}
