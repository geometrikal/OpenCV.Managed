using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// Shape descriptors. The contour cases process every contour of
/// <c>findContours(Inputs.Binary(Size), RETR_LIST, CHAIN_APPROX_NONE)</c>.
/// </summary>
[MemoryDiagnoser]
public class ShapeDescriptorBenchmarks
{
    private Mat<byte> _gray = null!;
    private Mat<float> _float = null!;
    private Mat<byte> _binary = null!;
    private Mat<byte> _sparseMask = null!;
    private Point[][] _contours = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _gray = Inputs.Gray(Size);
        _float = Inputs.GrayFloat(Size);
        _binary = Inputs.Binary(Size);
        _contours = [.. Contours.FindContours(_binary, RetrievalMode.List, ContourApproximationMode.None).Contours];
        _sparseMask = new Mat<byte>(Size, Size);
        for (int k = 0; k < 3; k++)
        {
            Point p = Inputs.RandomPoint(k, 7, Size);
            _sparseMask.At(p.Y, p.X) = 255;
        }
    }

    [Benchmark]
    public int ApproxPolyDPAll()
    {
        int total = 0;
        foreach (Point[] c in _contours)
        {
            total += ShapeDescriptors.ApproxPolyDP(c, 2.0, true).Length;
        }

        return total;
    }

    [Benchmark]
    public double ArcLengthAll()
    {
        double total = 0;
        foreach (Point[] c in _contours)
        {
            total += ShapeDescriptors.ArcLength(c, true);
        }

        return total;
    }

    [Benchmark]
    public double ContourAreaAll()
    {
        double total = 0;
        foreach (Point[] c in _contours)
        {
            total += ShapeDescriptors.ContourArea(c);
        }

        return total;
    }

    [Benchmark]
    public int BoundingRectAll()
    {
        int total = 0;
        foreach (Point[] c in _contours)
        {
            total += ShapeDescriptors.BoundingRect(c).Width;
        }

        return total;
    }

    [Benchmark]
    public double MomentsContourAll()
    {
        double total = 0;
        foreach (Point[] c in _contours)
        {
            total += ShapeDescriptors.Moments(c).M00;
        }

        return total;
    }

    [Benchmark]
    public double MatchShapesI1All()
    {
        double total = 0;
        for (int i = 1; i < _contours.Length; i++)
        {
            total += ShapeDescriptors.MatchShapes(_contours[i - 1], _contours[i], ShapeMatchModes.ContoursMatchI1);
        }

        return total;
    }

    [Benchmark]
    public Moments MomentsImage_U8() => ShapeDescriptors.Moments(_gray);

    [Benchmark]
    public Moments MomentsImageBinary_U8() => ShapeDescriptors.Moments(_binary, binaryImage: true);

    [Benchmark]
    public Moments MomentsImage_F32() => ShapeDescriptors.Moments(_float);

    [Benchmark]
    public Rect BoundingRectMask_U8() => ShapeDescriptors.BoundingRect(_binary);

    [Benchmark]
    public Rect BoundingRectMaskSparse_U8() => ShapeDescriptors.BoundingRect(_sparseMask);
}
