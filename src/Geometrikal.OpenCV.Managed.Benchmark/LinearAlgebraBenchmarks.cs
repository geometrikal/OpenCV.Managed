using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// Matrix product, small dense solvers and point transforms. The solver and point cases do not
/// depend on <see cref="Size"/>.
/// </summary>
[MemoryDiagnoser]
public class LinearAlgebraBenchmarks
{
    public const int PointCount = 100_000;

    private Mat<float> _a = null!;
    private Mat<float> _b = null!;
    private Mat<double> _a3 = null!;
    private Mat<double> _b3 = null!;
    private Mat<double> _a9 = null!;
    private Mat<double> _b9 = null!;
    private Mat<double> _a8x9 = null!;
    private Mat<double> _h = null!;
    private Point2f[] _points = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    /// <summary>Element (i, j) of the n × m solver matrix: an integer pattern with a dominant diagonal.</summary>
    public static Mat<double> SolverMatrix(int rows, int cols)
    {
        var m = new Mat<double>(cols, rows);
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                m.At(i, j) = (((i * 7) + (j * 3)) % 11) - 5 + (i == j ? 10 : 0);
            }
        }

        return m;
    }

    /// <summary>Point i: (i mod 317 · 1.5, i div 317 · 0.75).</summary>
    public static Point2f[] Points()
    {
        var p = new Point2f[PointCount];
        for (int i = 0; i < p.Length; i++)
        {
            p[i] = new Point2f((i % 317) * 1.5f, (i / 317) * 0.75f);
        }

        return p;
    }

    public static Mat<double> Homography()
        => new(new[] { 0.9, 0.05, 3, -0.04, 0.97, 7, 1e-4, 5e-5, 1 }, 3, 3);

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _a = Inputs.GrayFloat(Size);
        _b = Conversions.ConvertTo<byte, float>(Bitwise.Not(Inputs.Gray(Size)));
        _a3 = SolverMatrix(3, 3);
        _b3 = SolverMatrix(3, 1);
        _a9 = SolverMatrix(9, 9);
        _b9 = SolverMatrix(9, 1);
        _a8x9 = SolverMatrix(8, 9);
        _h = Homography();
        _points = Points();
    }

    [Benchmark]
    public Mat<float> Gemm_F32() => LinearAlgebra.Gemm(_a, _b, 1, null, 0);

    [Benchmark]
    public Mat<double> Solve3x3_F64()
    {
        LinearAlgebra.Solve(_a3, _b3, out Mat<double> x);
        return x;
    }

    [Benchmark]
    public Mat<double> Solve9x9_F64()
    {
        LinearAlgebra.Solve(_a9, _b9, out Mat<double> x);
        return x;
    }

    [Benchmark]
    public Mat<double> Svd8x9_F64() => new Svd<double>(_a8x9, SvdFlags.FullUV).Vt;

    [Benchmark]
    public Point2f[] PerspectiveTransform100k_F32() => LinearAlgebra.PerspectiveTransform(_points, _h);
}
