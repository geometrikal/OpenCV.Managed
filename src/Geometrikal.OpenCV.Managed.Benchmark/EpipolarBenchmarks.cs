using BenchmarkDotNet.Attributes;
using Geometrikal.OpenCV.Managed.Calib;
using Geometrikal.OpenCV.Managed.Core;

namespace Geometrikal.OpenCV.Managed.Benchmark;

/// <summary>
/// Two-view geometry on matches between two nadir drone frames with 30% outliers: f = 1000 px,
/// 1920 × 1080 images, ground at 90-110 m, a 10 m baseline and a small rotation, ±0.5 pixel
/// noise. <c>tools/bench_cv2.py</c> can build the same points with <see cref="TwoViewPoints"/>.
/// </summary>
[MemoryDiagnoser]
public class EpipolarBenchmarks
{
    /// <summary>The camera matrix of both views.</summary>
    public static readonly Mat<double> CameraMatrix = new(new double[] { 1000, 0, 960, 0, 1000, 540, 0, 0, 1 }, 3, 3);

    private Point2d[] _p1 = null!;
    private Point2d[] _p2 = null!;
    private Mat<double> _e = null!;
    private byte[] _mask = null!;

    [Params(500, 2000)]
    public int Points { get; set; }

    /// <summary>
    /// Builds the matches. Point i uses <see cref="Inputs.Hash"/> with the keys (i, 1) to (i, 6);
    /// a pair is an outlier when <c>Hash(i, 4) % 10 &lt; 3</c>.
    /// </summary>
    public static (Point2d[] P1, Point2d[] P2) TwoViewPoints(int count)
    {
        // Rotation of the second camera: Rodrigues vector (0.01, -0.02, 0.03), rounded.
        double[] r =
        [
            0.99930, -0.02978, -0.02029,
            0.02999, 0.99950, 0.00970,
            0.01999, -0.01030, 0.99975,
        ];
        double[] t = [-10.0, 0.5, 0.3];
        var p1 = new Point2d[count];
        var p2 = new Point2d[count];
        for (int i = 0; i < count; i++)
        {
            uint u = (uint)i;
            double px = (Inputs.Hash(u, 1) % 1920000) / 1000.0;
            double py = (Inputs.Hash(u, 2) % 1080000) / 1000.0;
            double z = 90 + ((Inputs.Hash(u, 3) % 20000) / 1000.0);
            double x = (px - 960) / 1000 * z, y = (py - 540) / 1000 * z;
            p1[i] = new Point2d(px, py);
            if (Inputs.Hash(u, 4) % 10 < 3)
            {
                p2[i] = new Point2d((Inputs.Hash(u, 5) % 1920000) / 1000.0, (Inputs.Hash(u, 6) % 1080000) / 1000.0);
                continue;
            }

            double x2 = (r[0] * x) + (r[1] * y) + (r[2] * z) + t[0];
            double y2 = (r[3] * x) + (r[4] * y) + (r[5] * z) + t[1];
            double z2 = (r[6] * x) + (r[7] * y) + (r[8] * z) + t[2];
            double nx = ((Inputs.Hash(u, 5) % 1001) / 1000.0) - 0.5;
            double ny = ((Inputs.Hash(u, 6) % 1001) / 1000.0) - 0.5;
            p2[i] = new Point2d((1000 * x2 / z2) + 960 + nx, (1000 * y2 / z2) + 540 + ny);
        }

        return (p1, p2);
    }

    [GlobalSetup]
    public void Setup()
    {
        (_p1, _p2) = TwoViewPoints(Points);
        EpipolarEstimationResult e = EpipolarGeometry.FindEssentialMat(_p1, _p2, CameraMatrix);
        _e = e.Model!;
        _mask = e.InlierMask;
    }

    [Benchmark]
    public int FindFundamentalMat_Ransac()
        => EpipolarGeometry.FindFundamentalMat(_p1, _p2, FundamentalMatMethod.Ransac, 1.0, 0.99).InlierMask.Length;

    [Benchmark]
    public int FindEssentialMat_Ransac()
        => EpipolarGeometry.FindEssentialMat(_p1, _p2, CameraMatrix).InlierMask.Length;

    [Benchmark]
    public int RecoverPose()
        => EpipolarGeometry.RecoverPose(_e, _p1, _p2, CameraMatrix, _mask).InlierCount;
}
