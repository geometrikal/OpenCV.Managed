using BenchmarkDotNet.Attributes;
using OpenCvSharp;
using Port = Geometrikal.OpenCV.Managed;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>The <see cref="AlignmentBenchmarks"/> cases run through OpenCvSharp.</summary>
[MemoryDiagnoser]
public class OpenCvSharpAlignmentBenchmarks
{
    private Mat _gray = null!;
    private Mat _templ16 = null!;
    private Mat _templ64 = null!;
    private Mat _pcA = null!;
    private Mat _pcB = null!;
    private Mat _eccTempl = null!;
    private Mat _eccInput = null!;
    private Mat _noWindow = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        _noWindow = new Mat();
        Port.Core.Mat<byte> gray = Inputs.Gray(Size);
        _gray = OcvInputs.ToOcv(gray);
        _templ16 = OcvInputs.ToOcv(gray.Roi(Size / 3, Size / 4, 16, 16).Clone());
        _templ64 = OcvInputs.ToOcv(gray.Roi(Size / 3, Size / 4, 64, 64).Clone());
        Port.Core.Mat<float> a = Inputs.GrayFloat(Size);
        _pcA = OcvInputs.ToOcv(a);
        _pcB = OcvInputs.ToOcv(AlignmentBenchmarks.Roll(a, 5, 7));
        Port.Core.Mat<float> eccTempl = Port.ImgProc.Filters.GaussianBlur(Inputs.GrayFloat(Size), default, 3.0);
        _eccTempl = OcvInputs.ToOcv(eccTempl);
        _eccInput = OcvInputs.ToOcv(AlignmentBenchmarks.Roll(eccTempl, 2, 1));
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (Mat m in new[] { _gray, _templ16, _templ64, _pcA, _pcB, _eccTempl, _eccInput, _noWindow })
        {
            m.Dispose();
        }
    }

    [Benchmark]
    public int MatchTemplate_CCoeffNormed_U8_T16() => Match(_templ16);

    [Benchmark]
    public int MatchTemplate_CCoeffNormed_U8_T64() => Match(_templ64);

    [Benchmark]
    public double PhaseCorrelate_F32() => Cv2.PhaseCorrelate(_pcA, _pcB, _noWindow, out _).X;

    [Benchmark]
    public double FindTransformEcc_Euclidean_F32()
    {
        using MatExpr eye = Mat.Eye(2, 3, MatType.CV_32FC1);
        using Mat warp = eye.ToMat();
        return Cv2.FindTransformECC(_eccTempl, _eccInput, warp, OpenCvSharp.MotionTypes.Euclidean, new TermCriteria(CriteriaTypes.Count, 10, 0), null, 5);
    }

    private int Match(Mat templ)
    {
        using var dst = new Mat();
        Cv2.MatchTemplate(_gray, templ, dst, OpenCvSharp.TemplateMatchModes.CCoeffNormed);
        return OcvInputs.Consume(dst);
    }
}
