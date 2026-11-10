using BenchmarkDotNet.Attributes;
using OpenCvSharp;
using Port = Geometrikal.OpenCV.Managed.Core;
using PortImgProc = Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// The <see cref="ShapeDescriptorBenchmarks"/> cases run through OpenCvSharp. The contours come
/// from the port's <c>findContours</c> output, so both sides process the same point arrays.
/// </summary>
[MemoryDiagnoser]
public class OpenCvSharpShapeDescriptorBenchmarks
{
    private Mat _gray = null!;
    private Mat _float = null!;
    private Mat _binary = null!;
    private Mat _sparseMask = null!;
    private Point[][] _contours = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        Port.Mat<byte> binary = Inputs.Binary(Size);
        _gray = OcvInputs.ToOcv(Inputs.Gray(Size));
        _float = OcvInputs.ToOcv(Inputs.GrayFloat(Size));
        _binary = OcvInputs.ToOcv(binary);
        _contours = PortImgProc.Contours.FindContours(binary, PortImgProc.RetrievalMode.List, PortImgProc.ContourApproximationMode.None)
            .Contours.Select(c => c.Select(OcvInputs.ToOcv).ToArray()).ToArray();
        var sparse = new Port.Mat<byte>(Size, Size);
        for (int k = 0; k < 3; k++)
        {
            Port.Point p = Inputs.RandomPoint(k, 7, Size);
            sparse.At(p.Y, p.X) = 255;
        }

        _sparseMask = OcvInputs.ToOcv(sparse);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _gray.Dispose();
        _float.Dispose();
        _binary.Dispose();
        _sparseMask.Dispose();
    }

    [Benchmark]
    public int ApproxPolyDPAll()
    {
        int total = 0;
        foreach (Point[] c in _contours)
        {
            total += Cv2.ApproxPolyDP(c, 2.0, true).Length;
        }

        return total;
    }

    [Benchmark]
    public double ArcLengthAll()
    {
        double total = 0;
        foreach (Point[] c in _contours)
        {
            total += Cv2.ArcLength(c, true);
        }

        return total;
    }

    [Benchmark]
    public double ContourAreaAll()
    {
        double total = 0;
        foreach (Point[] c in _contours)
        {
            total += Cv2.ContourArea(c);
        }

        return total;
    }

    [Benchmark]
    public int BoundingRectAll()
    {
        int total = 0;
        foreach (Point[] c in _contours)
        {
            total += Cv2.BoundingRect(c).Width;
        }

        return total;
    }

    [Benchmark]
    public double MomentsContourAll()
    {
        double total = 0;
        foreach (Point[] c in _contours)
        {
            total += Cv2.Moments(c).M00;
        }

        return total;
    }

    [Benchmark]
    public double MatchShapesI1All()
    {
        double total = 0;
        for (int i = 1; i < _contours.Length; i++)
        {
            total += Cv2.MatchShapes(_contours[i - 1], _contours[i], ShapeMatchModes.I1);
        }

        return total;
    }

    [Benchmark]
    public double MomentsImage_U8() => Cv2.Moments(_gray).M00;

    [Benchmark]
    public double MomentsImageBinary_U8() => Cv2.Moments(_binary, true).M00;

    [Benchmark]
    public double MomentsImage_F32() => Cv2.Moments(_float).M00;

    [Benchmark]
    public Rect BoundingRectMask_U8() => Cv2.BoundingRect(_binary);

    [Benchmark]
    public Rect BoundingRectMaskSparse_U8() => Cv2.BoundingRect(_sparseMask);
}
