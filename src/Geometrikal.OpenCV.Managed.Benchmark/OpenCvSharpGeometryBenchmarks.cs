using BenchmarkDotNet.Attributes;
using OpenCvSharp;
using Ocv = OpenCvSharp;
using Port = Geometrikal.OpenCV.Managed.Core;
using PortImgProc = Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>The <see cref="GeometryBenchmarks"/> cases run through OpenCvSharp.</summary>
[MemoryDiagnoser]
public class OpenCvSharpGeometryBenchmarks
{
    private Mat _gray = null!;
    private Mat _color = null!;
    private Mat _float = null!;
    private Mat _affine = null!;
    private Mat _perspective = null!;
    private Mat _mapX = null!;
    private Mat _mapY = null!;
    private Ocv.Size _down;
    private Ocv.Size _up;
    private Ocv.Size _same;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        _gray = OcvInputs.ToOcv(Inputs.Gray(Size));
        _color = OcvInputs.ToOcv(Inputs.Color(Size));
        _float = OcvInputs.ToOcv(Inputs.GrayFloat(Size));
        _down = new Ocv.Size(Size * 2 / 3, Size * 2 / 3);
        _up = new Ocv.Size(Size * 3 / 2, Size * 3 / 2);
        _same = new Ocv.Size(Size, Size);
        Port.Mat<double> affine = PortImgProc.Geometry.GetRotationMatrix2D(new Port.Point2f(Size / 2f, Size / 2f), 17, 0.9);
        double[] a = affine.ToArray();
        _affine = OcvInputs.ToOcv(affine);
        _perspective = OcvInputs.ToOcv(new Port.Mat<double>([a[0], a[1], a[2], a[3], a[4], a[5], 1e-4, 5e-5, 1.0], 3, 3));
        var mapX = new Port.Mat<float>(Size, Size);
        var mapY = new Port.Mat<float>(Size, Size);
        for (int y = 0; y < Size; y++)
        {
            for (int x = 0; x < Size; x++)
            {
                mapX.At(y, x) = (float)((x * 0.93) + (y * 0.05) + 3.5);
                mapY.At(y, x) = (float)((y * 0.97) - (x * 0.04) + 7.25);
            }
        }

        _mapX = OcvInputs.ToOcv(mapX);
        _mapY = OcvInputs.ToOcv(mapY);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (Mat m in new[] { _gray, _color, _float, _affine, _perspective, _mapX, _mapY })
        {
            m.Dispose();
        }
    }

    [Benchmark]
    public int ResizeDownLinear_U8C3() => Resize(_color, _down, InterpolationFlags.Linear);

    [Benchmark]
    public int ResizeUpLinear_U8C3() => Resize(_color, _up, InterpolationFlags.Linear);

    [Benchmark]
    public int ResizeDownArea_U8C3() => Resize(_color, _down, InterpolationFlags.Area);

    [Benchmark]
    public int ResizeUpArea_U8C3() => Resize(_color, _up, InterpolationFlags.Area);

    [Benchmark]
    public int ResizeDownCubic_U8C3() => Resize(_color, _down, InterpolationFlags.Cubic);

    [Benchmark]
    public int ResizeUpCubic_U8C3() => Resize(_color, _up, InterpolationFlags.Cubic);

    [Benchmark]
    public int ResizeDownLinear_F32() => Resize(_float, _down, InterpolationFlags.Linear);

    [Benchmark]
    public int ResizeUpLinear_F32() => Resize(_float, _up, InterpolationFlags.Linear);

    [Benchmark]
    public int ResizeDownArea_F32() => Resize(_float, _down, InterpolationFlags.Area);

    [Benchmark]
    public int ResizeUpArea_F32() => Resize(_float, _up, InterpolationFlags.Area);

    [Benchmark]
    public int ResizeDownCubic_F32() => Resize(_float, _down, InterpolationFlags.Cubic);

    [Benchmark]
    public int ResizeUpCubic_F32() => Resize(_float, _up, InterpolationFlags.Cubic);

    [Benchmark]
    public int WarpAffineLinear_U8()
    {
        using var dst = new Mat();
        Cv2.WarpAffine(_gray, dst, _affine, _same);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int WarpPerspectiveLinear_U8()
    {
        using var dst = new Mat();
        Cv2.WarpPerspective(_gray, dst, _perspective, _same);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int RemapLinear_U8()
    {
        using var dst = new Mat();
        Cv2.Remap(_gray, dst, _mapX, _mapY, InterpolationFlags.Linear);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int Flip_U8C3()
    {
        using var dst = new Mat();
        Cv2.Flip(_color, dst, FlipMode.XY);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int Rotate90_U8()
    {
        using var dst = new Mat();
        Cv2.Rotate(_gray, dst, RotateFlags.Rotate90Clockwise);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int Transpose_U8()
    {
        using var dst = new Mat();
        Cv2.Transpose(_gray, dst);
        return OcvInputs.Consume(dst);
    }

    private static int Resize(Mat source, Ocv.Size size, InterpolationFlags interpolation)
    {
        using var dst = new Mat();
        Cv2.Resize(source, dst, size, interpolation: interpolation);
        return OcvInputs.Consume(dst);
    }
}
