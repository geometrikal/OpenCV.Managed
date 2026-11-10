using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// Shape-fitting functions over every contour of <see cref="Inputs.Binary"/>
/// (<c>RETR_LIST</c>, <c>CHAIN_APPROX_NONE</c>). Each method processes all contours and returns a
/// checksum.
/// </summary>
[MemoryDiagnoser]
public class ShapeFittingBenchmarks
{
    private Point[][] _contours = null!;
    private Point[][] _ellipseContours = null!;
    private Point[][] _defectContours = null!;
    private int[][] _defectHulls = null!;
    private Point2f[] _testPoints = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _contours = Contours.FindContours(Inputs.Binary(Size), RetrievalMode.List, ContourApproximationMode.None).Contours.ToArray();
        _ellipseContours = _contours.Where(c => c.Length >= 5).ToArray();
        var defectContours = new List<Point[]>();
        var defectHulls = new List<int[]>();
        foreach (Point[] c in _contours)
        {
            int[] hull = ShapeFitting.ConvexHullIndices(c);
            try
            {
                ShapeFitting.ConvexityDefects(c, hull);
            }
            catch (ArgumentException)
            {
                continue;
            }

            defectContours.Add(c);
            defectHulls.Add(hull);
        }

        _defectContours = defectContours.ToArray();
        _defectHulls = defectHulls.ToArray();
        _testPoints = _contours.Select(c => new Point2f(c[0].X + 1.5f, c[0].Y + 1.5f)).ToArray();
    }

    [Benchmark]
    public int ConvexHull()
    {
        int sum = 0;
        foreach (Point[] c in _contours)
        {
            sum += ShapeFitting.ConvexHull(c).Length;
        }

        return sum;
    }

    [Benchmark]
    public int ConvexHullIndices()
    {
        int sum = 0;
        foreach (Point[] c in _contours)
        {
            sum += ShapeFitting.ConvexHullIndices(c).Length;
        }

        return sum;
    }

    [Benchmark]
    public int ConvexityDefects()
    {
        int sum = 0;
        for (int i = 0; i < _defectContours.Length; i++)
        {
            sum += ShapeFitting.ConvexityDefects(_defectContours[i], _defectHulls[i]).Length;
        }

        return sum;
    }

    [Benchmark]
    public int IsContourConvex()
    {
        int sum = 0;
        foreach (Point[] c in _contours)
        {
            sum += ShapeFitting.IsContourConvex(c) ? 1 : 0;
        }

        return sum;
    }

    [Benchmark]
    public float MinAreaRect()
    {
        float sum = 0;
        foreach (Point[] c in _contours)
        {
            sum += ShapeFitting.MinAreaRect(c).Angle;
        }

        return sum;
    }

    [Benchmark]
    public float MinEnclosingCircle()
    {
        float sum = 0;
        foreach (Point[] c in _contours)
        {
            sum += ShapeFitting.MinEnclosingCircle(c).Radius;
        }

        return sum;
    }

    [Benchmark]
    public float FitEllipse()
    {
        float sum = 0;
        foreach (Point[] c in _ellipseContours)
        {
            sum += ShapeFitting.FitEllipse(c).Angle;
        }

        return sum;
    }

    [Benchmark]
    public float FitEllipseAms()
    {
        float sum = 0;
        foreach (Point[] c in _ellipseContours)
        {
            sum += ShapeFitting.FitEllipseAms(c).Angle;
        }

        return sum;
    }

    [Benchmark]
    public float FitEllipseDirect()
    {
        float sum = 0;
        foreach (Point[] c in _ellipseContours)
        {
            sum += ShapeFitting.FitEllipseDirect(c).Angle;
        }

        return sum;
    }

    [Benchmark]
    public float FitLineL2() => FitLines(DistanceTypes.L2);

    [Benchmark]
    public float FitLineHuber() => FitLines(DistanceTypes.Huber);

    [Benchmark]
    public double PointPolygonTest()
    {
        double sum = 0;
        for (int i = 0; i < _contours.Length; i++)
        {
            sum += ShapeFitting.PointPolygonTest(_contours[i], _testPoints[i], false);
        }

        return sum;
    }

    [Benchmark]
    public double PointPolygonTestDistance()
    {
        double sum = 0;
        for (int i = 0; i < _contours.Length; i++)
        {
            sum += ShapeFitting.PointPolygonTest(_contours[i], _testPoints[i], true);
        }

        return sum;
    }

    private float FitLines(DistanceTypes dist)
    {
        float sum = 0;
        foreach (Point[] c in _contours)
        {
            sum += ShapeFitting.FitLine(c, dist, 0, 0.01, 0.01).Vx;
        }

        return sum;
    }
}
