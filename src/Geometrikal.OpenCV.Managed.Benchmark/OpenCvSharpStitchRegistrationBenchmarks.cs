using BenchmarkDotNet.Attributes;
using OpenCvSharp;
using OpenCvSharp.Detail;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// The <see cref="StitchRegistrationBenchmarks"/> matcher cases run through OpenCvSharp. OpenCvSharp
/// does not expose the stitching estimators or bundle adjusters, so those cases are omitted. The
/// OpenCV matcher uses FLANN (LSH for ORB descriptors); the port uses exhaustive matching.
/// </summary>
[MemoryDiagnoser]
public class OpenCvSharpStitchRegistrationBenchmarks
{
    private ImageFeatures[] _features = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        OcvInputs.ApplyThreads();
        var tiles = StitchRegistrationBenchmarks.Tiles(Size).Select(OcvInputs.ToOcv).ToArray();
        using ORB orb = ORB.Create();
        _features = CvDetail.ComputeImageFeatures(orb, tiles, null);
        foreach (Mat t in tiles)
        {
            t.Dispose();
        }
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        foreach (ImageFeatures f in _features)
        {
            f.Dispose();
        }
    }

    [Benchmark]
    public int BestOf2NearestMatcher_Match()
    {
        using var matcher = new BestOf2NearestMatcher(false, 0.3f, 6, 6);
        return Consume(matcher.Apply(_features, null!));
    }

    [Benchmark]
    public int AffineBestOf2NearestMatcher_Match()
    {
        using var matcher = new AffineBestOf2NearestMatcher(false, false, 0.3f, 6);
        return Consume(matcher.Apply(_features, null!));
    }

    private static int Consume(MatchesInfo[] matches)
    {
        int n = matches.Length;
        foreach (MatchesInfo m in matches)
        {
            m.Dispose();
        }

        return n;
    }
}
