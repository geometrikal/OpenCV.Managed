using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Calib;
using Geometrikal.OpenCV.Managed.Core;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// Robust 2-D model estimation on matched point sets with 30% outliers. The inliers follow a
/// similarity transform (scale 1.02, rotation 5 degrees) with ±0.5 pixel noise, as matches
/// between overlapping nadir drone frames. <c>tools/bench_cv2.py</c> builds the same points.
/// </summary>
[MemoryDiagnoser]
public class EstimationBenchmarks
{
    private Point2f[] _src = null!;
    private Point2f[] _dst = null!;

    [Params(500, 5000)]
    public int Points { get; set; }

    /// <summary>
    /// Builds the point pairs. Point i uses <see cref="Inputs.Hash"/> with the keys (i, 1) to
    /// (i, 5); a pair is an outlier when <c>Hash(i, 3) % 10 &lt; 3</c>.
    /// </summary>
    public static (Point2f[] Src, Point2f[] Dst) MatchedPoints(int count)
    {
        const double A = 1.0161188, B = 0.0888981;
        var src = new Point2f[count];
        var dst = new Point2f[count];
        for (int i = 0; i < count; i++)
        {
            uint u = (uint)i;
            float x = (Inputs.Hash(u, 1) % 1000000) / 1000f;
            float y = (Inputs.Hash(u, 2) % 1000000) / 1000f;
            src[i] = new Point2f(x, y);
            if (Inputs.Hash(u, 3) % 10 < 3)
            {
                dst[i] = new Point2f((Inputs.Hash(u, 4) % 1000000) / 1000f, (Inputs.Hash(u, 5) % 1000000) / 1000f);
            }
            else
            {
                double nx = ((Inputs.Hash(u, 4) % 1001) / 1000.0) - 0.5;
                double ny = ((Inputs.Hash(u, 5) % 1001) / 1000.0) - 0.5;
                dst[i] = new Point2f((float)((A * x) - (B * y) + 37.5 + nx), (float)((B * x) + (A * y) - 21.25 + ny));
            }
        }

        return (src, dst);
    }

    [GlobalSetup]
    public void Setup() => (_src, _dst) = MatchedPoints(Points);

    [Benchmark]
    public int FindHomography_Ransac()
        => GeometricEstimation.FindHomography(_src, _dst, RobustEstimationMethod.Ransac, 3.0).InlierMask.Length;

    [Benchmark]
    public int FindHomography_Lmeds()
        => GeometricEstimation.FindHomography(_src, _dst, RobustEstimationMethod.Lmeds, 3.0).InlierMask.Length;

    [Benchmark]
    public int FindHomography_Rho()
        => GeometricEstimation.FindHomography(_src, _dst, RobustEstimationMethod.Rho, 3.0).InlierMask.Length;

    [Benchmark]
    public int EstimateAffine2D_Ransac()
        => GeometricEstimation.EstimateAffine2D(_src, _dst, RobustEstimationMethod.Ransac, 3.0).InlierMask.Length;

    [Benchmark]
    public int EstimateAffinePartial2D_Ransac()
        => GeometricEstimation.EstimateAffinePartial2D(_src, _dst, RobustEstimationMethod.Ransac, 3.0).InlierMask.Length;

    [Benchmark]
    public int EstimateAffinePartial2D_Lmeds()
        => GeometricEstimation.EstimateAffinePartial2D(_src, _dst, RobustEstimationMethod.Lmeds, 3.0).InlierMask.Length;
}
