using BenchmarkDotNet.Attributes;
using OpenCvSharp;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>The <see cref="MatcherBenchmarks"/> cases run through OpenCvSharp.</summary>
[MemoryDiagnoser]
public class OpenCvSharpMatcherBenchmarks
{
    private Mat _q1000 = null!, _t1000 = null!, _q5000 = null!, _t5000 = null!, _qf1000 = null!, _tf1000 = null!;
    private BFMatcher _hamming = null!, _l2 = null!;

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        _q1000 = OcvInputs.ToOcv(MatcherInputs.Bytes(1000, 32, 0));
        _t1000 = OcvInputs.ToOcv(MatcherInputs.Bytes(1000, 32, MatcherInputs.TrainSalt));
        _q5000 = OcvInputs.ToOcv(MatcherInputs.Bytes(5000, 32, 0));
        _t5000 = OcvInputs.ToOcv(MatcherInputs.Bytes(5000, 32, MatcherInputs.TrainSalt));
        _qf1000 = OcvInputs.ToOcv(MatcherInputs.Floats(1000, 128, 0));
        _tf1000 = OcvInputs.ToOcv(MatcherInputs.Floats(1000, 128, MatcherInputs.TrainSalt));
        _hamming = new BFMatcher(NormTypes.Hamming);
        _l2 = new BFMatcher(NormTypes.L2);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (Mat m in new[] { _q1000, _t1000, _q5000, _t5000, _qf1000, _tf1000 })
        {
            m.Dispose();
        }

        _hamming.Dispose();
        _l2.Dispose();
    }

    [Benchmark]
    public int Hamming_Match_1000() => _hamming.Match(_q1000, _t1000).Length;

    [Benchmark]
    public int Hamming_Match_5000() => _hamming.Match(_q5000, _t5000).Length;

    [Benchmark]
    public int Hamming_Knn2_1000() => _hamming.KnnMatch(_q1000, _t1000, 2).Length;

    [Benchmark]
    public int Hamming_Knn2_5000() => _hamming.KnnMatch(_q5000, _t5000, 2).Length;

    [Benchmark]
    public int L2_Knn2_1000() => _l2.KnnMatch(_qf1000, _tf1000, 2).Length;
}
