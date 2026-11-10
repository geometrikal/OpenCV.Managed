using BenchmarkDotNet.Attributes;
using OpenCvSharp;
using Port = Geometrikal.OpenCV.Managed;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// The <see cref="ShapeFittingBenchmarks"/> cases run through OpenCvSharp. The contours are the
/// port contours of <see cref="Inputs.Binary"/>, converted to OpenCvSharp points.
/// </summary>
[MemoryDiagnoser]
public class OpenCvSharpShapeFittingBenchmarks
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
        OcvInputs.ApplyThreads();
        Port.Core.Point[][] contours = Port.ImgProc.Contours
            .FindContours(Inputs.Binary(Size), Port.ImgProc.RetrievalMode.List, Port.ImgProc.ContourApproximationMode.None)
            .Contours.ToArray();
        _contours = contours.Select(c => c.Select(OcvInputs.ToOcv).ToArray()).ToArray();
        _ellipseContours = _contours.Where(c => c.Length >= 5).ToArray();
        var defectContours = new List<Point[]>();
        var defectHulls = new List<int[]>();
        foreach (Point[] c in _contours)
        {
            int[] hull = Cv2.ConvexHullIndices(c);
            try
            {
                Cv2.ConvexityDefects(c, hull);
            }
            catch (OpenCVException)
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
            sum += Cv2.ConvexHull(c).Length;
        }

        return sum;
    }

    [Benchmark]
    public int ConvexHullIndices()
    {
        int sum = 0;
        foreach (Point[] c in _contours)
        {
            sum += Cv2.ConvexHullIndices(c).Length;
        }

        return sum;
    }

    [Benchmark]
    public int ConvexityDefects()
    {
        int sum = 0;
        for (int i = 0; i < _defectContours.Length; i++)
        {
            sum += Cv2.ConvexityDefects(_defectContours[i], _defectHulls[i]).Length;
        }

        return sum;
    }

    [Benchmark]
    public int IsContourConvex()
    {
        int sum = 0;
        foreach (Point[] c in _contours)
        {
            sum += Cv2.IsContourConvex(c) ? 1 : 0;
        }

        return sum;
    }

    [Benchmark]
    public float MinAreaRect()
    {
        float sum = 0;
        foreach (Point[] c in _contours)
        {
            sum += Cv2.MinAreaRect(c).Angle;
        }

        return sum;
    }

    [Benchmark]
    public float MinEnclosingCircle()
    {
        float sum = 0;
        foreach (Point[] c in _contours)
        {
            Cv2.MinEnclosingCircle(c, out _, out float radius);
            sum += radius;
        }

        return sum;
    }

    [Benchmark]
    public float FitEllipse()
    {
        float sum = 0;
        foreach (Point[] c in _ellipseContours)
        {
            sum += Cv2.FitEllipse(c).Angle;
        }

        return sum;
    }

    [Benchmark]
    public float FitEllipseAms()
    {
        float sum = 0;
        foreach (Point[] c in _ellipseContours)
        {
            sum += Cv2.FitEllipseAMS(c).Angle;
        }

        return sum;
    }

    [Benchmark]
    public float FitEllipseDirect()
    {
        float sum = 0;
        foreach (Point[] c in _ellipseContours)
        {
            sum += Cv2.FitEllipseDirect(c).Angle;
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
            sum += Cv2.PointPolygonTest(_contours[i], _testPoints[i], false);
        }

        return sum;
    }

    [Benchmark]
    public double PointPolygonTestDistance()
    {
        double sum = 0;
        for (int i = 0; i < _contours.Length; i++)
        {
            sum += Cv2.PointPolygonTest(_contours[i], _testPoints[i], true);
        }

        return sum;
    }

    private float FitLines(DistanceTypes dist)
    {
        float sum = 0;
        foreach (Point[] c in _contours)
        {
            sum += (float)Cv2.FitLine(c, dist, 0, 0.01, 0.01).Vx;
        }

        return sum;
    }
}
