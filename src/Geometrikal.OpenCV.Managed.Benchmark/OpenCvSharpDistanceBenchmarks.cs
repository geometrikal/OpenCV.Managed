using BenchmarkDotNet.Attributes;
using OpenCvSharp;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// The <see cref="DistanceBenchmarks"/> cases run through OpenCvSharp. OpenCV 4.13 uses IPP for
/// the 5x5 mask, the precise transform and the 8-bit L1 transform.
/// </summary>
[MemoryDiagnoser]
public class OpenCvSharpDistanceBenchmarks
{
    private Mat _sparse = null!;
    private Mat _dist = null!;
    private Mat _dist8u = null!;
    private Mat _labels = null!;
    private Mat _color = null!;
    private Mat _seeds = null!;
    private Mat _markers = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        _sparse = OcvInputs.ToOcv(Inputs.SparseZeros(Size));
        _dist = new Mat(Size, Size, MatType.CV_32FC1);
        _dist8u = new Mat(Size, Size, MatType.CV_8UC1);
        _labels = new Mat(Size, Size, MatType.CV_32SC1);
        _color = OcvInputs.ToOcv(Inputs.Color(Size));
        _seeds = OcvInputs.ToOcv(Inputs.Seeds(Size, 200));
        _markers = new Mat(Size, Size, MatType.CV_32SC1);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (Mat m in new[] { _sparse, _dist, _dist8u, _labels, _color, _seeds, _markers })
        {
            m.Dispose();
        }
    }

    [Benchmark]
    public int L2Mask3() => Distance(DistanceTypes.L2, DistanceTransformMasks.Mask3);

    [Benchmark]
    public int L2Mask5() => Distance(DistanceTypes.L2, DistanceTransformMasks.Mask5);

    [Benchmark]
    public int L2Precise() => Distance(DistanceTypes.L2, DistanceTransformMasks.Precise);

    [Benchmark]
    public int L1Mask3() => Distance(DistanceTypes.L1, DistanceTransformMasks.Mask3);

    [Benchmark]
    public int L1Byte()
    {
        Cv2.DistanceTransform(_sparse, _dist8u, DistanceTypes.L1, DistanceTransformMasks.Mask3, MatType.CV_8U);
        return OcvInputs.Consume(_dist8u);
    }

    [Benchmark]
    public int LabelsCcomp()
    {
        Cv2.DistanceTransformWithLabels(
            _sparse, _dist, _labels, DistanceTypes.L2, DistanceTransformMasks.Mask5, DistanceTransformLabelTypes.CComp);
        return OcvInputs.Consume(_labels);
    }

    [Benchmark]
    public int LabelsPixel()
    {
        Cv2.DistanceTransformWithLabels(
            _sparse, _dist, _labels, DistanceTypes.L2, DistanceTransformMasks.Mask5, DistanceTransformLabelTypes.Pixel);
        return OcvInputs.Consume(_labels);
    }

    /// <summary>Includes the copy of the seeds into the marker image, as the port case.</summary>
    [Benchmark]
    public int Watershed()
    {
        _seeds.CopyTo(_markers);
        Cv2.Watershed(_color, _markers);
        return OcvInputs.Consume(_markers);
    }

    private int Distance(DistanceTypes distanceType, DistanceTransformMasks maskSize)
    {
        using var dst = new Mat();
        Cv2.DistanceTransform(_sparse, dst, distanceType, maskSize);
        return OcvInputs.Consume(dst);
    }
}
