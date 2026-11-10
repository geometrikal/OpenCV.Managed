using BenchmarkDotNet.Attributes;
using OpenCvSharp;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// The <see cref="DrawingAaBenchmarks"/> cases run through OpenCvSharp with
/// <see cref="LineTypes.AntiAlias"/> on a 3-channel canvas.
/// </summary>
[MemoryDiagnoser]
public class OpenCvSharpDrawingAaBenchmarks
{
    private static readonly Scalar Color = new(255, 128, 0);

    private Mat _canvas = null!;
    private Point[] _p1 = null!;
    private Point[] _p2 = null!;
    private Point[][] _polygons = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        _canvas = new Mat(Size, Size, MatType.CV_8UC3, Scalar.All(0));
        _p1 = Enumerable.Range(0, Inputs.ShapeCount).Select(i => OcvInputs.ToOcv(Inputs.RandomPoint(i, 0, Size))).ToArray();
        _p2 = Enumerable.Range(0, Inputs.ShapeCount).Select(i => OcvInputs.ToOcv(Inputs.RandomPoint(i, 1, Size))).ToArray();
        _polygons = Enumerable.Range(0, 16)
            .Select(p => Enumerable.Range(0, 8).Select(v => OcvInputs.ToOcv(Inputs.RandomPoint(p * 8 + v, 2, Size))).ToArray())
            .ToArray();
    }

    [GlobalCleanup]
    public void Cleanup() => _canvas.Dispose();

    [Benchmark]
    public void LineAa1px()
    {
        for (int i = 0; i < _p1.Length; i++)
        {
            Cv2.Line(_canvas, _p1[i], _p2[i], Color, 1, LineTypes.AntiAlias);
        }
    }

    [Benchmark]
    public void LineAa5px()
    {
        for (int i = 0; i < _p1.Length; i++)
        {
            Cv2.Line(_canvas, _p1[i], _p2[i], Color, 5, LineTypes.AntiAlias);
        }
    }

    [Benchmark]
    public void CircleAaOutline()
    {
        int radius = Size / 16;
        for (int i = 0; i < _p1.Length; i++)
        {
            Cv2.Circle(_canvas, _p1[i], radius, Color, 2, LineTypes.AntiAlias);
        }
    }

    [Benchmark]
    public void FillPolyAa() => Cv2.FillPoly(_canvas, _polygons, Color, LineTypes.AntiAlias);
}
