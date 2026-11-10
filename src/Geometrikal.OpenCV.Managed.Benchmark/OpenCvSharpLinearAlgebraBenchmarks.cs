using BenchmarkDotNet.Attributes;
using OpenCvSharp;
using Port = Geometrikal.OpenCV.Managed.Core;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>The <see cref="LinearAlgebraBenchmarks"/> cases run through OpenCvSharp.</summary>
[MemoryDiagnoser]
public class OpenCvSharpLinearAlgebraBenchmarks
{
    private Mat _a = null!;
    private Mat _b = null!;
    private Mat _a3 = null!;
    private Mat _b3 = null!;
    private Mat _a9 = null!;
    private Mat _b9 = null!;
    private Mat _a8x9 = null!;
    private Mat _h = null!;
    private Mat _points = null!;
    private Mat _empty = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        _a = OcvInputs.ToOcv(Inputs.GrayFloat(Size));
        _b = OcvInputs.ToOcv(Port.Conversions.ConvertTo<byte, float>(Port.Bitwise.Not(Inputs.Gray(Size))));
        _a3 = OcvInputs.ToOcv(LinearAlgebraBenchmarks.SolverMatrix(3, 3));
        _b3 = OcvInputs.ToOcv(LinearAlgebraBenchmarks.SolverMatrix(3, 1));
        _a9 = OcvInputs.ToOcv(LinearAlgebraBenchmarks.SolverMatrix(9, 9));
        _b9 = OcvInputs.ToOcv(LinearAlgebraBenchmarks.SolverMatrix(9, 1));
        _a8x9 = OcvInputs.ToOcv(LinearAlgebraBenchmarks.SolverMatrix(8, 9));
        _h = OcvInputs.ToOcv(LinearAlgebraBenchmarks.Homography());
        Port.Point2f[] p = LinearAlgebraBenchmarks.Points();
        var flat = new float[p.Length * 2];
        for (int i = 0; i < p.Length; i++)
        {
            flat[2 * i] = p[i].X;
            flat[(2 * i) + 1] = p[i].Y;
        }

        _points = OcvInputs.ToOcv(new Port.Mat<float>(flat, 1, p.Length, 2));
        _empty = new Mat();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (Mat m in new[] { _a, _b, _a3, _b3, _a9, _b9, _a8x9, _h, _points, _empty })
        {
            m.Dispose();
        }
    }

    [Benchmark]
    public int Gemm_F32()
    {
        using var dst = new Mat();
        Cv2.Gemm(_a, _b, 1, _empty, 0, dst);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int Solve3x3_F64()
    {
        using var dst = new Mat();
        Cv2.Solve(_a3, _b3, dst, DecompTypes.LU);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int Solve9x9_F64()
    {
        using var dst = new Mat();
        Cv2.Solve(_a9, _b9, dst, DecompTypes.LU);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int Svd8x9_F64()
    {
        using var w = new Mat();
        using var u = new Mat();
        using var vt = new Mat();
        Cv2.SVDecomp(_a8x9, w, u, vt, SVD.Flags.FullUV);
        return OcvInputs.Consume(vt);
    }

    [Benchmark]
    public int PerspectiveTransform100k_F32()
    {
        using var dst = new Mat();
        Cv2.PerspectiveTransform(_points, dst, _h);
        return OcvInputs.Consume(dst);
    }
}
