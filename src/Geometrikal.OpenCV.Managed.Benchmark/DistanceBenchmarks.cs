using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// Distance transforms on <see cref="Inputs.SparseZeros"/> (long distances) and watershed with
/// 200 seeds on <see cref="Inputs.Color"/>. Compare with cv2 modes that disable IPP: cv2 uses IPP
/// for the 5x5 mask, the precise transform and the 8-bit L1 transform when IPP is on.
/// </summary>
[MemoryDiagnoser]
public class DistanceBenchmarks
{
    private Mat<byte> _sparse = null!;
    private Mat<float> _dist = null!;
    private Mat<byte> _dist8u = null!;
    private Mat<int> _labels = null!;
    private Mat<byte> _color = null!;
    private Mat<int> _seeds = null!;
    private Mat<int> _markers = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _sparse = Inputs.SparseZeros(Size);
        _dist = new Mat<float>(Size, Size);
        _dist8u = new Mat<byte>(Size, Size);
        _labels = new Mat<int>(Size, Size);
        _color = Inputs.Color(Size);
        _seeds = Inputs.Seeds(Size, 200);
        _markers = new Mat<int>(Size, Size);
    }

    [Benchmark]
    public Mat<float> L2Mask3() => DistanceTransforms.DistanceTransform(_sparse, DistanceTypes.L2, DistanceTransformMasks.Mask3);

    [Benchmark]
    public Mat<float> L2Mask5() => DistanceTransforms.DistanceTransform(_sparse, DistanceTypes.L2, DistanceTransformMasks.Mask5);

    [Benchmark]
    public Mat<float> L2Precise() => DistanceTransforms.DistanceTransform(_sparse, DistanceTypes.L2, DistanceTransformMasks.Precise);

    [Benchmark]
    public Mat<float> L1Mask3() => DistanceTransforms.DistanceTransform(_sparse, DistanceTypes.L1, DistanceTransformMasks.Mask3);

    [Benchmark]
    public Mat<byte> L1Byte()
    {
        DistanceTransforms.DistanceTransform(_sparse, _dist8u, DistanceTypes.L1, DistanceTransformMasks.Mask3);
        return _dist8u;
    }

    [Benchmark]
    public Mat<int> LabelsCcomp()
    {
        DistanceTransforms.DistanceTransformWithLabels(_sparse, _dist, _labels, DistanceTypes.L2, DistanceTransformMasks.Mask5, DistanceTransformLabelTypes.ConnectedComponent);
        return _labels;
    }

    [Benchmark]
    public Mat<int> LabelsPixel()
    {
        DistanceTransforms.DistanceTransformWithLabels(_sparse, _dist, _labels, DistanceTypes.L2, DistanceTransformMasks.Mask5, DistanceTransformLabelTypes.Pixel);
        return _labels;
    }

    /// <summary>Includes the copy of the seeds into the marker image, as the cv2 case.</summary>
    [Benchmark]
    public Mat<int> Watershed()
    {
        _seeds.CopyTo(_markers);
        Segmentation.Watershed(_color, _markers);
        return _markers;
    }
}
