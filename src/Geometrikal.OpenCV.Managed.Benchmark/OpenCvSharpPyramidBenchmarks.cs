using BenchmarkDotNet.Attributes;
using OpenCvSharp;
using OpenCvSharp.Internal.Vectors;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>The <see cref="PyramidBenchmarks"/> cases run through OpenCvSharp.</summary>
[MemoryDiagnoser]
public class OpenCvSharpPyramidBenchmarks
{
    private Mat _gray = null!;
    private Mat _color = null!;
    private Mat _grayFloat = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        _gray = OcvInputs.ToOcv(Inputs.Gray(Size));
        _color = OcvInputs.ToOcv(Inputs.Color(Size));
        _grayFloat = OcvInputs.ToOcv(Inputs.GrayFloat(Size));
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (Mat m in new[] { _gray, _color, _grayFloat })
        {
            m.Dispose();
        }
    }

    [Benchmark]
    public int PyrDown_U8() => Down(_gray);

    [Benchmark]
    public int PyrDown_U8C3() => Down(_color);

    [Benchmark]
    public int PyrDown_F32() => Down(_grayFloat);

    [Benchmark]
    public int PyrUp_U8() => Up(_gray);

    [Benchmark]
    public int PyrUp_U8C3() => Up(_color);

    [Benchmark]
    public int PyrUp_F32() => Up(_grayFloat);

    [Benchmark]
    public int BuildPyramid4_U8() => Build(_gray);

    [Benchmark]
    public int BuildPyramid4_U8C3() => Build(_color);

    private static int Down(Mat src)
    {
        using var dst = new Mat();
        Cv2.PyrDown(src, dst);
        return OcvInputs.Consume(dst);
    }

    private static int Up(Mat src)
    {
        using var dst = new Mat();
        Cv2.PyrUp(src, dst);
        return OcvInputs.Consume(dst);
    }

    private static int Build(Mat src)
    {
        using var levels = new VectorOfMat();
        Cv2.BuildPyramid(src, levels, 4);
        Mat[] mats = levels.ToArray();
        int rows = 0;
        for (int i = 1; i < mats.Length; i++)
        {
            rows += OcvInputs.Consume(mats[i]);
        }

        foreach (Mat m in mats)
        {
            m.Dispose();
        }

        return rows;
    }
}
