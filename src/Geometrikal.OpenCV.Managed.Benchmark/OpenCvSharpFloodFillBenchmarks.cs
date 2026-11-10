using BenchmarkDotNet.Attributes;
using OpenCvSharp;
using Port = Geometrikal.OpenCV.Managed.Core;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// The <see cref="FloodFillBenchmarks"/> cases run through OpenCvSharp. OpenCV always runs the
/// gradient fill with an internal mask, also for zero differences.
/// </summary>
[MemoryDiagnoser]
public class OpenCvSharpFloodFillBenchmarks
{
    private Mat _binary = null!;
    private Mat _gray = null!;
    private Mat _work = null!;
    private Mat _mask = null!;
    private Point _holeSeed;
    private Point _centre;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        Port.Mat<byte> binary = Inputs.Binary(Size);
        _binary = OcvInputs.ToOcv(binary);
        _gray = OcvInputs.ToOcv(Inputs.Gray(Size));
        _work = new Mat(Size, Size, MatType.CV_8UC1);
        _mask = new Mat(Size + 2, Size + 2, MatType.CV_8UC1);
        _holeSeed = new Point(binary.GetRowSpan(0).IndexOf((byte)0), 0);
        _centre = new Point(Size / 2, Size / 2);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (Mat m in new[] { _binary, _gray, _work, _mask })
        {
            m.Dispose();
        }
    }

    [Benchmark]
    public int HoleFill()
    {
        _binary.CopyTo(_work);
        return Cv2.FloodFill(_work, _holeSeed, new Scalar(128));
    }

    [Benchmark]
    public int HoleFillMask()
    {
        _binary.CopyTo(_work);
        _mask.SetTo(Scalar.All(0));
        return Cv2.FloodFill(
            _work, _mask, _holeSeed, new Scalar(255), out _, default, default,
            FloodFillFlags.Link4 | FloodFillFlags.MaskOnly | (FloodFillFlags)(255 << 8));
    }

    [Benchmark]
    public int GradientGray()
    {
        _gray.CopyTo(_work);
        return Cv2.FloodFill(_work, _centre, new Scalar(255), out _, Scalar.All(12), Scalar.All(12), FloodFillFlags.Link8);
    }

    [Benchmark]
    public int FixedRangeGray()
    {
        _gray.CopyTo(_work);
        return Cv2.FloodFill(
            _work, _centre, new Scalar(255), out _, Scalar.All(48), Scalar.All(48), FloodFillFlags.Link8 | FloodFillFlags.FixedRange);
    }
}
