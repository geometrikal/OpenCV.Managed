using BenchmarkDotNet.Attributes;
using OpenCvSharp;
using Port = Geometrikal.OpenCV.Managed.Core;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>The <see cref="OpticalFlowBenchmarks"/> cases run through OpenCvSharp.</summary>
[MemoryDiagnoser]
public class OpenCvSharpOpticalFlowBenchmarks
{
    private Mat _prev = null!;
    private Mat _next = null!;
    private Point2f[] _points = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        Port.Mat<byte> prev = Inputs.Gray(Size);
        _prev = OcvInputs.ToOcv(prev);
        _next = OcvInputs.ToOcv(OpticalFlowBenchmarks.Shifted(prev, 3, 2));
        _points = OpticalFlowBenchmarks.GridPoints(Size).Select(p => new Point2f(p.X, p.Y)).ToArray();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _prev.Dispose();
        _next.Dispose();
    }

    [Benchmark]
    public int CalcOpticalFlowPyrLK_500()
    {
        Point2f[] next = [];
        Cv2.CalcOpticalFlowPyrLK(
            _prev,
            _next,
            _points,
            ref next,
            out byte[] status,
            out float[] err,
            new OpenCvSharp.Size(21, 21),
            3,
            new TermCriteria(CriteriaTypes.Count | CriteriaTypes.Eps, 30, 0.01));
        return status.Length + err.Length;
    }

    [Benchmark]
    public int BuildOpticalFlowPyramid_U8()
    {
        int levels = Cv2.BuildOpticalFlowPyramid(_prev, out Mat[] pyramid, new OpenCvSharp.Size(21, 21), 3);
        foreach (Mat m in pyramid)
        {
            m.Dispose();
        }

        return levels;
    }
}
