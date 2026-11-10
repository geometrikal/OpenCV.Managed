using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.ImgProc;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// Template matching, phase correlation and ECC alignment. Templates are cut from
/// <see cref="Inputs.Gray"/> at <c>(size/3, size/4)</c>. Phase correlation compares
/// <see cref="Inputs.GrayFloat"/> with a copy shifted circularly by (5, 7). ECC aligns a Gaussian
/// blurred (sigma 3) <see cref="Inputs.GrayFloat"/> with a copy shifted circularly by (2, 1), with
/// exactly 10 iterations.
/// </summary>
[MemoryDiagnoser]
public class AlignmentBenchmarks
{
    private Mat<byte> _gray = null!;
    private Mat<byte> _templ16 = null!;
    private Mat<byte> _templ64 = null!;
    private Mat<float> _pcA = null!;
    private Mat<float> _pcB = null!;
    private Mat<float> _eccTempl = null!;
    private Mat<float> _eccInput = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _gray = Inputs.Gray(Size);
        _templ16 = _gray.Roi(Size / 3, Size / 4, 16, 16).Clone();
        _templ64 = _gray.Roi(Size / 3, Size / 4, 64, 64).Clone();
        _pcA = Inputs.GrayFloat(Size);
        _pcB = Roll(_pcA, 5, 7);
        _eccTempl = Filters.GaussianBlur(Inputs.GrayFloat(Size), default, 3.0);
        _eccInput = Roll(_eccTempl, 2, 1);
    }

    /// <summary><c>np.roll(src, (dy, dx), axis=(0, 1))</c>.</summary>
    internal static Mat<float> Roll(Mat<float> src, int dx, int dy)
    {
        int w = src.Width, h = src.Height;
        var dst = new Mat<float>(w, h);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                dst.At(y, x) = src.At(((y - dy) % h + h) % h, ((x - dx) % w + w) % w);
            }
        }

        return dst;
    }

    [Benchmark]
    public Mat<float> MatchTemplate_CCoeffNormed_U8_T16() => TemplateMatching.MatchTemplate(_gray, _templ16, TemplateMatchModes.CCoeffNormed);

    [Benchmark]
    public Mat<float> MatchTemplate_CCoeffNormed_U8_T64() => TemplateMatching.MatchTemplate(_gray, _templ64, TemplateMatchModes.CCoeffNormed);

    [Benchmark]
    public Point2d PhaseCorrelate_F32() => PhaseCorrelation.PhaseCorrelate(_pcA, _pcB);

    [Benchmark]
    public double FindTransformEcc_Euclidean_F32()
    {
        var warp = new Mat<float>(3, 2);
        LinearAlgebra.SetIdentity(warp);
        return EccAlignment.FindTransformEcc(_eccTempl, _eccInput, warp, MotionTypes.Euclidean, new TermCriteria(TermCriteriaType.MaxIter, 10, 0));
    }
}
