using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Video;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>Sparse optical flow on a gray pair; the next image is the first one moved by (3, 2) with a replicated edge.</summary>
[MemoryDiagnoser]
public class OpticalFlowBenchmarks
{
    private Mat<byte> _prev = null!;
    private Mat<byte> _next = null!;
    private Point2f[] _points = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _prev = Inputs.Gray(Size);
        _next = Shifted(_prev, 3, 2);
        _points = GridPoints(Size);
    }

    /// <summary>The next image: <c>next(y, x) = prev(clamp(y - dy), clamp(x - dx))</c>.</summary>
    internal static Mat<byte> Shifted(Mat<byte> src, int dx, int dy)
    {
        var dst = new Mat<byte>(src.Width, src.Height);
        for (int y = 0; y < src.Height; y++)
        {
            for (int x = 0; x < src.Width; x++)
            {
                dst.At(y, x) = src.At(Math.Clamp(y - dy, 0, src.Height - 1), Math.Clamp(x - dx, 0, src.Width - 1));
            }
        }

        return dst;
    }

    /// <summary>500 points on a 25 × 20 grid between 10 % and 90 % of the image side.</summary>
    internal static Point2f[] GridPoints(int size)
    {
        var pts = new Point2f[500];
        for (int i = 0; i < 20; i++)
        {
            for (int j = 0; j < 25; j++)
            {
                pts[(i * 25) + j] = new Point2f((float)(size * (0.1 + (0.8 * j / 24.0))), (float)(size * (0.1 + (0.8 * i / 19.0))));
            }
        }

        return pts;
    }

    [Benchmark]
    public OpticalFlowResult CalcOpticalFlowPyrLK_500() =>
        OpticalFlow.CalcOpticalFlowPyrLK(_prev, _next, _points, null, new Size(21, 21), 3, OpticalFlow.DefaultCriteria);

    [Benchmark]
    public OpticalFlowPyramid BuildOpticalFlowPyramid_U8() =>
        OpticalFlow.BuildOpticalFlowPyramid(_prev, new Size(21, 21), 3);
}
