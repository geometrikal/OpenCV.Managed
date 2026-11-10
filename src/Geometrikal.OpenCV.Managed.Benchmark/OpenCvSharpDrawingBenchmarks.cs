using BenchmarkDotNet.Attributes;
using OpenCvSharp;
using Ocv = OpenCvSharp;
using PortImgProc = Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// The <see cref="DrawingBenchmarks"/> cases run through OpenCvSharp. Each per-shape benchmark
/// draws <see cref="Inputs.ShapeCount"/> shapes onto a 3-channel canvas. FillPoly, Polylines and
/// DrawContours take managed point arrays, as the port does; OpenCvSharp marshals them on each call.
/// </summary>
[MemoryDiagnoser]
public class OpenCvSharpDrawingBenchmarks
{
    private static readonly Scalar Color = new(255, 128, 0);

    private Mat _canvas = null!;
    private Point[] _p1 = null!;
    private Point[] _p2 = null!;
    private Point[][] _polygons = null!;
    private Point[][] _contours = null!;
    private HierarchyIndex[] _hierarchy = null!;

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
        PortImgProc.ContourResult found = PortImgProc.Contours.FindContours(
            Inputs.Binary(Size), PortImgProc.RetrievalMode.Tree, PortImgProc.ContourApproximationMode.Simple);
        _contours = found.Contours.Select(c => c.Select(OcvInputs.ToOcv).ToArray()).ToArray();
        _hierarchy = found.Hierarchy.Select(h => new HierarchyIndex(h.Next, h.Previous, h.FirstChild, h.Parent)).ToArray();
    }

    [GlobalCleanup]
    public void Cleanup() => _canvas.Dispose();

    [Benchmark]
    public void Line1px()
    {
        for (int i = 0; i < _p1.Length; i++)
        {
            Cv2.Line(_canvas, _p1[i], _p2[i], Color);
        }
    }

    [Benchmark]
    public void Line5px()
    {
        for (int i = 0; i < _p1.Length; i++)
        {
            Cv2.Line(_canvas, _p1[i], _p2[i], Color, 5);
        }
    }

    [Benchmark]
    public void RectangleFilled()
    {
        for (int i = 0; i < _p1.Length; i++)
        {
            Cv2.Rectangle(_canvas, _p1[i], _p2[i], Color, -1);
        }
    }

    [Benchmark]
    public void CircleOutline()
    {
        int radius = Size / 16;
        for (int i = 0; i < _p1.Length; i++)
        {
            Cv2.Circle(_canvas, _p1[i], radius, Color, 2);
        }
    }

    [Benchmark]
    public void CircleFilled()
    {
        int radius = Size / 16;
        for (int i = 0; i < _p1.Length; i++)
        {
            Cv2.Circle(_canvas, _p1[i], radius, Color, -1);
        }
    }

    [Benchmark]
    public void EllipseOutline()
    {
        var axes = new Ocv.Size(Size / 8, Size / 16);
        for (int i = 0; i < _p1.Length; i++)
        {
            Cv2.Ellipse(_canvas, _p1[i], axes, 30, 0, 360, Color, 2);
        }
    }

    [Benchmark]
    public void FillPoly() => Cv2.FillPoly(_canvas, _polygons, Color);

    [Benchmark]
    public void Polylines() => Cv2.Polylines(_canvas, _polygons, true, Color);

    [Benchmark]
    public void DrawContours() =>
        Cv2.DrawContours(_canvas, _contours, -1, Color, 1, LineTypes.Link8, _hierarchy);
}
