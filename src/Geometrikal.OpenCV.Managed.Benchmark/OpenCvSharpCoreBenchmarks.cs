using BenchmarkDotNet.Attributes;
using OpenCvSharp;
using Port = Geometrikal.OpenCV.Managed.Core;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>The <see cref="CoreBenchmarks"/> cases run through OpenCvSharp.</summary>
[MemoryDiagnoser]
public class OpenCvSharpCoreBenchmarks
{
    private Mat _gray = null!;
    private Mat _gray2 = null!;
    private Mat _color = null!;
    private Mat[] _planes = null!;
    private Mat _float = null!;
    private Mat _float2 = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        Port.Mat<byte> gray = Inputs.Gray(Size);
        Port.Mat<byte> gray2 = Port.Bitwise.Not(gray);
        Port.Mat<byte> color = Inputs.Color(Size);
        _gray = OcvInputs.ToOcv(gray);
        _gray2 = OcvInputs.ToOcv(gray2);
        _color = OcvInputs.ToOcv(color);
        _planes = Port.Conversions.Split(color).Select(OcvInputs.ToOcv).ToArray();
        _float = OcvInputs.ToOcv(Inputs.GrayFloat(Size));
        _float2 = OcvInputs.ToOcv(Port.Conversions.ConvertTo<byte, float>(gray2));
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (Mat m in _planes.Concat([_gray, _gray2, _color, _float, _float2]))
        {
            m.Dispose();
        }
    }

    [Benchmark]
    public int Add_U8()
    {
        using var dst = new Mat();
        Cv2.Add(_gray, _gray2, dst);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int Add_F32()
    {
        using var dst = new Mat();
        Cv2.Add(_float, _float2, dst);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int AbsDiff_U8()
    {
        using var dst = new Mat();
        Cv2.Absdiff(_gray, _gray2, dst);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int AddWeighted_U8()
    {
        using var dst = new Mat();
        Cv2.AddWeighted(_gray, 0.3, _gray2, 0.7, 1, dst);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int Multiply_U8()
    {
        using var dst = new Mat();
        Cv2.Multiply(_gray, _gray2, dst, 0.5);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int Compare_U8()
    {
        using var dst = new Mat();
        Cv2.Compare(_gray, _gray2, dst, CmpTypes.GT);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int InRange_U8C3()
    {
        using var dst = new Mat();
        Cv2.InRange(_color, new Scalar(10, 20, 30), new Scalar(200, 210, 220), dst);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int ConvertScaleAbs_F32()
    {
        using var dst = new Mat();
        Cv2.ConvertScaleAbs(_float, dst, 0.5);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public int Split_U8C3()
    {
        Mat[] planes = Cv2.Split(_color);
        int rows = 0;
        foreach (Mat plane in planes)
        {
            rows += OcvInputs.Consume(plane);
            plane.Dispose();
        }

        return rows;
    }

    [Benchmark]
    public int Merge_U8C3()
    {
        using var dst = new Mat();
        Cv2.Merge(_planes, dst);
        return OcvInputs.Consume(dst);
    }

    [Benchmark]
    public Scalar Sum_U8() => Cv2.Sum(_gray);

    [Benchmark]
    public Scalar MeanStdDev_U8()
    {
        Cv2.MeanStdDev(_gray, out Scalar mean, out _);
        return mean;
    }

    [Benchmark]
    public double MinMaxLoc_U8()
    {
        Cv2.MinMaxLoc(_gray, out double min, out double max, out Point minLoc, out Point maxLoc);
        return min + max + minLoc.X + minLoc.Y + maxLoc.X + maxLoc.Y;
    }

    [Benchmark]
    public int CountNonZero_U8() => Cv2.CountNonZero(_gray);

    [Benchmark]
    public double NormL2_F32() => Cv2.Norm(_float);
}
