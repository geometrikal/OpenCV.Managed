using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// Each per-shape benchmark draws <see cref="Inputs.ShapeCount"/> shapes onto a 3-channel canvas.
/// </summary>
[MemoryDiagnoser]
public class DrawingBenchmarks
{
    private static readonly Scalar Color = new(255, 128, 0);

    private Mat<byte> _canvas = null!;
    private Point[] _p1 = null!;
    private Point[] _p2 = null!;
    private Point[][] _polygons = null!;
    private IReadOnlyList<Point[]> _contours = null!;
    private IReadOnlyList<ContourHierarchy> _hierarchy = null!;

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
        ContourResult found = Contours.FindContours(
            Inputs.Binary(Size), RetrievalMode.Tree, ContourApproximationMode.Simple);
        _contours = found.Contours;
        _hierarchy = found.Hierarchy;
    }

    [Benchmark]
    public void Line1px()
    {
        for (int i = 0; i < _p1.Length; i++)
        {
            Drawing.Line(_canvas, _p1[i], _p2[i], Color);
        }
    }

    [Benchmark]
    public void Line5px()
    {
        for (int i = 0; i < _p1.Length; i++)
        {
            Drawing.Line(_canvas, _p1[i], _p2[i], Color, 5);
        }
    }

    [Benchmark]
    public void RectangleFilled()
    {
        for (int i = 0; i < _p1.Length; i++)
        {
            Drawing.Rectangle(_canvas, _p1[i], _p2[i], Color, -1);
        }
    }

    [Benchmark]
    public void CircleOutline()
    {
        int radius = Size / 16;
        for (int i = 0; i < _p1.Length; i++)
        {
            Drawing.Circle(_canvas, _p1[i], radius, Color, 2);
        }
    }

    [Benchmark]
    public void CircleFilled()
    {
        int radius = Size / 16;
        for (int i = 0; i < _p1.Length; i++)
        {
            Drawing.Circle(_canvas, _p1[i], radius, Color, -1);
        }
    }

    [Benchmark]
    public void EllipseOutline()
    {
        var axes = new Size(Size / 8, Size / 16);
        for (int i = 0; i < _p1.Length; i++)
        {
            Drawing.Ellipse(_canvas, _p1[i], axes, 30, 0, 360, Color, 2);
        }
    }

    [Benchmark]
    public void FillPoly() => Drawing.FillPoly(_canvas, _polygons, Color);

    [Benchmark]
    public void Polylines() => Drawing.Polylines(_canvas, _polygons, true, Color);

    [Benchmark]
    public void DrawContours() =>
        Drawing.DrawContours(_canvas, _contours, -1, Color, 1, LineType.EightConnected, _hierarchy);
}
