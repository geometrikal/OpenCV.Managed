using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Features;
using Geometrikal.OpenCV.Managed.Stitching;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// Stitching registration on 6 overlapping tiles (3 × 2 grid, tile step 0.6 × tile size) cut from
/// <see cref="Inputs.Gray"/>, with default ORB features computed once in the setup.
/// <c>tools/bench_cv2.py</c> builds the same tiles.
/// </summary>
[MemoryDiagnoser]
public class StitchRegistrationBenchmarks
{
    private ImageFeatures[] _features = null!;
    private MatchesInfo[] _homographyMatches = null!;

    [Params(256, 1024)]
    public int Size { get; set; }

    /// <summary>
    /// Cuts the 6 tiles: tile (c, r) starts at (c·step, r·step) with step = 0.6 · size. The
    /// source is <c>Hash(x, y) &amp; 255</c> smoothed by a 5 × 5 Gaussian (σ = 1.5); it has no
    /// periodic structure, so only overlapping tiles match.
    /// </summary>
    public static List<Mat<byte>> Tiles(int size)
    {
        int step = size * 6 / 10;
        int n = size + (2 * step);
        var noise = new Mat<byte>(n, n);
        for (int y = 0; y < n; y++)
        {
            Span<byte> row = noise.GetRowSpan(y);
            for (int x = 0; x < n; x++)
            {
                row[x] = (byte)(Inputs.Hash((uint)x, (uint)y) & 255);
            }
        }

        Mat<byte> source = ImgProc.Filters.GaussianBlur(noise, new Size(5, 5), 1.5);
        var tiles = new List<Mat<byte>>();
        for (int r = 0; r < 2; r++)
        {
            for (int c = 0; c < 3; c++)
            {
                tiles.Add(source.Roi(c * step, r * step, size, size).Clone());
            }
        }

        return tiles;
    }

    [GlobalSetup]
    public void Setup()
    {
        Threads.Apply();
        _features = ImageFeatures.Compute(new Orb(), Tiles(Size));
        _homographyMatches = new BestOf2NearestMatcher().Match(_features);
    }

    [Benchmark]
    public int BestOf2NearestMatcher_Match() => new BestOf2NearestMatcher().Match(_features).Length;

    [Benchmark]
    public int HomographyEstimator_BundleAdjusterRay()
    {
        var cameras = new List<CameraParams>();
        new HomographyBasedEstimator().Estimate(_features, _homographyMatches, cameras);
        foreach (CameraParams c in cameras)
        {
            Mat<double> r = c.R;
            for (int i = 0; i < 9; i++)
            {
                r.At(i / 3, i % 3) = (float)r.At(i / 3, i % 3);
            }
        }

        new BundleAdjusterRay().Estimate(_features, _homographyMatches, cameras);
        return cameras.Count;
    }

    [Benchmark]
    public int AffineMatcher_BundleAdjusterAffinePartial()
    {
        MatchesInfo[] matches = new AffineBestOf2NearestMatcher().Match(_features);
        var cameras = new List<CameraParams>();
        new AffineBasedEstimator().Estimate(_features, matches, cameras);
        new BundleAdjusterAffinePartial().Estimate(_features, matches, cameras);
        return cameras.Count;
    }
}
