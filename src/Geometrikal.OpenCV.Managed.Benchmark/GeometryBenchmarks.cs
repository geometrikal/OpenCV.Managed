using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

[MemoryDiagnoser]
public class GeometryBenchmarks
{
    private Mat<byte> _gray = null!;
    private Mat<byte> _color = null!;
    private Mat<float> _float = null!;
    private Mat<double> _affine = null!;
    private Mat<double> _perspective = null!;
    private Mat<float> _mapX = null!;
    private Mat<float> _mapY = null!;
    private Size _down;
    private Size _up;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _gray = Inputs.Gray(Size);
        _color = Inputs.Color(Size);
        _float = Inputs.GrayFloat(Size);
        _down = new Size(Size * 2 / 3, Size * 2 / 3);
        _up = new Size(Size * 3 / 2, Size * 3 / 2);
        _affine = Geometry.GetRotationMatrix2D(new Point2f(Size / 2f, Size / 2f), 17, 0.9);
        double[] a = _affine.ToArray();
        _perspective = new Mat<double>([a[0], a[1], a[2], a[3], a[4], a[5], 1e-4, 5e-5, 1.0], 3, 3);
        _mapX = new Mat<float>(Size, Size);
        _mapY = new Mat<float>(Size, Size);
        for (int y = 0; y < Size; y++)
        {
            for (int x = 0; x < Size; x++)
            {
                _mapX.At(y, x) = (float)((x * 0.93) + (y * 0.05) + 3.5);
                _mapY.At(y, x) = (float)((y * 0.97) - (x * 0.04) + 7.25);
            }
        }
    }

    [Benchmark]
    public Mat<byte> ResizeDownLinear_U8C3() => Geometry.Resize(_color, _down, interpolation: InterpolationFlags.Linear);

    [Benchmark]
    public Mat<byte> ResizeUpLinear_U8C3() => Geometry.Resize(_color, _up, interpolation: InterpolationFlags.Linear);

    [Benchmark]
    public Mat<byte> ResizeDownArea_U8C3() => Geometry.Resize(_color, _down, interpolation: InterpolationFlags.Area);

    [Benchmark]
    public Mat<byte> ResizeUpArea_U8C3() => Geometry.Resize(_color, _up, interpolation: InterpolationFlags.Area);

    [Benchmark]
    public Mat<byte> ResizeDownCubic_U8C3() => Geometry.Resize(_color, _down, interpolation: InterpolationFlags.Cubic);

    [Benchmark]
    public Mat<byte> ResizeUpCubic_U8C3() => Geometry.Resize(_color, _up, interpolation: InterpolationFlags.Cubic);

    [Benchmark]
    public Mat<float> ResizeDownLinear_F32() => Geometry.Resize(_float, _down, interpolation: InterpolationFlags.Linear);

    [Benchmark]
    public Mat<float> ResizeUpLinear_F32() => Geometry.Resize(_float, _up, interpolation: InterpolationFlags.Linear);

    [Benchmark]
    public Mat<float> ResizeDownArea_F32() => Geometry.Resize(_float, _down, interpolation: InterpolationFlags.Area);

    [Benchmark]
    public Mat<float> ResizeUpArea_F32() => Geometry.Resize(_float, _up, interpolation: InterpolationFlags.Area);

    [Benchmark]
    public Mat<float> ResizeDownCubic_F32() => Geometry.Resize(_float, _down, interpolation: InterpolationFlags.Cubic);

    [Benchmark]
    public Mat<float> ResizeUpCubic_F32() => Geometry.Resize(_float, _up, interpolation: InterpolationFlags.Cubic);

    [Benchmark]
    public Mat<byte> WarpAffineLinear_U8() => Geometry.WarpAffine(_gray, _affine);

    [Benchmark]
    public Mat<byte> WarpPerspectiveLinear_U8() => Geometry.WarpPerspective(_gray, _perspective);

    [Benchmark]
    public Mat<byte> RemapLinear_U8() => Geometry.Remap(_gray, _mapX, _mapY, InterpolationFlags.Linear);

    [Benchmark]
    public Mat<byte> Flip_U8C3() => MatrixTransforms.Flip(_color, -1);

    [Benchmark]
    public Mat<byte> Rotate90_U8() => MatrixTransforms.Rotate(_gray, RotateFlags.Rotate90Clockwise);

    [Benchmark]
    public Mat<byte> Transpose_U8() => MatrixTransforms.Transpose(_gray);
}
