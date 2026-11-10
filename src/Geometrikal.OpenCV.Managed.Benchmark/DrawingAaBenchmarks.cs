using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// Anti-aliased drawing. Each per-shape benchmark draws <see cref="Inputs.ShapeCount"/> shapes
/// with <see cref="LineType.AntiAliased"/> onto a 3-channel canvas.
/// </summary>
[MemoryDiagnoser]
public class DrawingAaBenchmarks
{
    private static readonly Scalar Color = new(255, 128, 0);

    private Mat<byte> _canvas = null!;
    private Point[] _p1 = null!;
    private Point[] _p2 = null!;
    private Point[][] _polygons = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _canvas = new Mat<byte>(Size, Size, 3);
        _p1 = Enumerable.Range(0, Inputs.ShapeCount).Select(i => Inputs.RandomPoint(i, 0, Size)).ToArray();
        _p2 = Enumerable.Range(0, Inputs.ShapeCount).Select(i => Inputs.RandomPoint(i, 1, Size)).ToArray();
        _polygons = Enumerable.Range(0, 16)
            .Select(p => Enumerable.Range(0, 8).Select(v => Inputs.RandomPoint(p * 8 + v, 2, Size)).ToArray())
            .ToArray();
    }

    [Benchmark]
    public void LineAa1px()
    {
        for (int i = 0; i < _p1.Length; i++)
        {
            Drawing.Line(_canvas, _p1[i], _p2[i], Color, 1, LineType.AntiAliased);
        }
    }

    [Benchmark]
    public void LineAa5px()
    {
        for (int i = 0; i < _p1.Length; i++)
        {
            Drawing.Line(_canvas, _p1[i], _p2[i], Color, 5, LineType.AntiAliased);
        }
    }

    [Benchmark]
    public void CircleAaOutline()
    {
        int radius = Size / 16;
        for (int i = 0; i < _p1.Length; i++)
        {
            Drawing.Circle(_canvas, _p1[i], radius, Color, 2, LineType.AntiAliased);
        }
    }

    [Benchmark]
    public void FillPolyAa() => Drawing.FillPoly(_canvas, _polygons, Color, LineType.AntiAliased);
}
