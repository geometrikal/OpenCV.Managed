using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Features;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// Deterministic descriptor sets for the matcher suites. <c>tools/bench_cv2.py</c> uses the same
/// formulas: byte element (i, j) is <c>Hash(j, i + salt) &amp; 255</c>; float element (i, j) is
/// <c>(Hash(j, i + salt) &amp; 0xFFFF) / 256 − 128</c>. The query set uses salt 0 and the train set
/// salt 1000000.
/// </summary>
public static class MatcherInputs
{
    public const uint TrainSalt = 1_000_000;

    public static Mat<byte> Bytes(int rows, int cols, uint salt)
    {
        var mat = new Mat<byte>(cols, rows);
        for (int i = 0; i < rows; i++)
        {
            Span<byte> row = mat.GetRowSpan(i);
            for (int j = 0; j < cols; j++)
            {
                row[j] = (byte)(Inputs.Hash((uint)j, (uint)i + salt) & 255);
            }
        }

        return mat;
    }

    public static Mat<float> Floats(int rows, int cols, uint salt)
    {
        var mat = new Mat<float>(cols, rows);
        for (int i = 0; i < rows; i++)
        {
            Span<float> row = mat.GetRowSpan(i);
            for (int j = 0; j < cols; j++)
            {
                row[j] = ((Inputs.Hash((uint)j, (uint)i + salt) & 0xFFFF) / 256f) - 128f;
            }
        }

        return mat;
    }
}

/// <summary>
/// Brute-force matching: 32-byte Hamming descriptors (ORB size) and 128-float L2 descriptors
/// (SIFT size). The number in the name is the query and train descriptor count.
/// </summary>
[MemoryDiagnoser]
public class MatcherBenchmarks
{
    private Mat<byte> _q1000 = null!, _t1000 = null!, _q5000 = null!, _t5000 = null!;
    private Mat<float> _qf1000 = null!, _tf1000 = null!;
    private readonly BFMatcher _hamming = new(NormType.Hamming);
    private readonly BFMatcher _l2 = new(NormType.L2);

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _q1000 = MatcherInputs.Bytes(1000, 32, 0);
        _t1000 = MatcherInputs.Bytes(1000, 32, MatcherInputs.TrainSalt);
        _q5000 = MatcherInputs.Bytes(5000, 32, 0);
        _t5000 = MatcherInputs.Bytes(5000, 32, MatcherInputs.TrainSalt);
        _qf1000 = MatcherInputs.Floats(1000, 128, 0);
        _tf1000 = MatcherInputs.Floats(1000, 128, MatcherInputs.TrainSalt);
    }

    [Benchmark]
    public DMatch[] Hamming_Match_1000() => _hamming.Match(_q1000, _t1000);

    [Benchmark]
    public DMatch[] Hamming_Match_5000() => _hamming.Match(_q5000, _t5000);

    [Benchmark]
    public DMatch[][] Hamming_Knn2_1000() => _hamming.KnnMatch(_q1000, _t1000, 2);

    [Benchmark]
    public DMatch[][] Hamming_Knn2_5000() => _hamming.KnnMatch(_q5000, _t5000, 2);

    [Benchmark]
    public DMatch[][] L2_Knn2_1000() => _l2.KnnMatch(_qf1000, _tf1000, 2);
}
