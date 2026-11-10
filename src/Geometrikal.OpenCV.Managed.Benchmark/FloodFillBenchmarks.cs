using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// Flood fill. The hole fills start at the first zero pixel of row 0 of <see cref="Inputs.Binary"/>
/// (the background). The gradient fills start at the centre of <see cref="Inputs.Gray"/>. Each case
/// copies the input into a preallocated image first, as the cv2 case.
/// </summary>
[MemoryDiagnoser]
public class FloodFillBenchmarks
{
    private Mat<byte> _binary = null!;
    private Mat<byte> _gray = null!;
    private Mat<byte> _work = null!;
    private Mat<byte> _mask = null!;
    private Point _holeSeed;
    private Point _centre;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _binary = Inputs.Binary(Size);
        _gray = Inputs.Gray(Size);
        _work = new Mat<byte>(Size, Size);
        _mask = new Mat<byte>(Size + 2, Size + 2);
        _holeSeed = new Point(_binary.GetRowSpan(0).IndexOf((byte)0), 0);
        _centre = new Point(Size / 2, Size / 2);
    }

    /// <summary>No mask and zero differences: the equality fill.</summary>
    [Benchmark]
    public int HoleFill()
    {
        _binary.CopyTo(_work);
        return FloodFilling.FloodFill(_work, _holeSeed, new Scalar(128)).Area;
    }

    /// <summary>Mask-only fill with mask value 255; includes the mask clear.</summary>
    [Benchmark]
    public int HoleFillMask()
    {
        _binary.CopyTo(_work);
        _mask.Clear();
        return FloodFilling.FloodFill(
            _work, _mask, _holeSeed, new Scalar(255), default, default,
            FloodFillFlags.Connectivity4 | FloodFillFlags.MaskOnly | FloodFilling.MaskFillValue(255)).Area;
    }

    /// <summary>Floating range 12, 8-connectivity.</summary>
    [Benchmark]
    public int GradientGray()
    {
        _gray.CopyTo(_work);
        return FloodFilling.FloodFill(_work, _centre, new Scalar(255), Scalar.All(12), Scalar.All(12), FloodFillFlags.Connectivity8).Area;
    }

    /// <summary>Fixed range 48, 8-connectivity.</summary>
    [Benchmark]
    public int FixedRangeGray()
    {
        _gray.CopyTo(_work);
        return FloodFilling.FloodFill(
            _work, _centre, new Scalar(255), Scalar.All(48), Scalar.All(48), FloodFillFlags.Connectivity8 | FloodFillFlags.FixedRange).Area;
    }
}
