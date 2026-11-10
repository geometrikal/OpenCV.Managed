namespace Geometrikal.OpenCV.Managed.Tests.Calib;

using System;
using System.Linq;
using Geometrikal.OpenCV.Managed.Calib;
using Geometrikal.OpenCV.Managed.Calib.Internal;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Analytic and structural tests for <see cref="GeometricEstimation"/>: exact models, outlier
/// rejection, invalid input, input immutability, determinism, the iteration bound and the
/// equality of the vector and scalar error loops.
/// </summary>
public sealed class EstimationBasicTests
{
    [Fact]
    public void EstimateAffinePartial2D_ExactSimilarity_RecoversModel()
    {
        (Point2f[] src, Point2f[] dst) = Similarity(50, 0.1, 1.05, 12.5, -7.25, 0.0, seed: 1);
        GeometricEstimationResult r = GeometricEstimation.EstimateAffinePartial2D(src, dst);

        Assert.True(r.Success);
        Assert.All(r.InlierMask, m => Assert.Equal(1, m));
        double[] m = r.Model!.ToArray();
        Assert.Equal(1.05 * Math.Cos(0.1), m[0], 4);
        Assert.Equal(-1.05 * Math.Sin(0.1), m[1], 4);
        Assert.Equal(12.5, m[2], 2);
        Assert.Equal(1.05 * Math.Sin(0.1), m[3], 4);
        Assert.Equal(m[0], m[4]);
        Assert.Equal(-7.25, m[5], 2);
    }

    [Fact]
    public void EstimateAffine2D_ExactAffine_RecoversModel()
    {
        var rng = new Random(2);
        Point2f[] src = Enumerable.Range(0, 40).Select(_ => new Point2f(rng.NextSingle() * 500, rng.NextSingle() * 500)).ToArray();
        Point2f[] dst = src.Select(p => new Point2f((1.1f * p.X) + (0.2f * p.Y) + 3, (-0.1f * p.X) + (0.9f * p.Y) - 4)).ToArray();
        GeometricEstimationResult r = GeometricEstimation.EstimateAffine2D(src, dst);

        Assert.True(r.Success);
        double[] m = r.Model!.ToArray();
        double[] expected = [1.1, 0.2, 3, -0.1, 0.9, -4];
        for (int i = 0; i < 6; i++)
        {
            Assert.Equal(expected[i], m[i], 3);
        }
    }

    [Theory]
    [InlineData(RobustEstimationMethod.Ransac)]
    [InlineData(RobustEstimationMethod.Lmeds)]
    [InlineData(RobustEstimationMethod.Rho)]
    public void FindHomography_Outliers_AreRejected(RobustEstimationMethod method)
    {
        (Point2f[] src, Point2f[] dst) = Similarity(200, -0.05, 0.98, 30, 40, 0.2, seed: 3);
        for (int i = 0; i < dst.Length; i += 5)
        {
            dst[i] = new Point2f(dst[i].X + 60, dst[i].Y - 45);
        }

        GeometricEstimationResult r = GeometricEstimation.FindHomography(src, dst, method, 2.0);

        Assert.True(r.Success);
        for (int i = 0; i < dst.Length; i++)
        {
            Assert.Equal(i % 5 == 0 ? 0 : 1, r.InlierMask[i]);
        }

        Assert.Equal(1.0, r.Model!.At(2, 2), 12);
    }

    [Fact]
    public void FindHomography_FourPoints_MapsPointsExactly()
    {
        Point2f[] src = [new(0, 0), new(100, 0), new(100, 80), new(0, 80)];
        Point2f[] dst = [new(10, 5), new(120, 12), new(115, 95), new(4, 88)];
        GeometricEstimationResult r = GeometricEstimation.FindHomography(src, dst, RobustEstimationMethod.Ransac);

        Assert.True(r.Success);
        Assert.Equal([1, 1, 1, 1], r.InlierMask);
        double[] h = r.Model!.ToArray();
        for (int i = 0; i < 4; i++)
        {
            double w = (h[6] * src[i].X) + (h[7] * src[i].Y) + h[8];
            Assert.Equal(dst[i].X, ((h[0] * src[i].X) + (h[1] * src[i].Y) + h[2]) / w, 3);
            Assert.Equal(dst[i].Y, ((h[3] * src[i].X) + (h[4] * src[i].Y) + h[5]) / w, 3);
        }
    }

    [Fact]
    public void EstimateTranslation2D_NoRefinement_ReturnsMeanDisplacementOfInliers()
    {
        Point2f[] src = [new(0, 0), new(10, 0), new(0, 10), new(10, 10), new(5, 5), new(50, 50)];
        Point2f[] dst = [new(3, 4), new(13, 4), new(3, 14), new(13, 14), new(8, 9), new(0, 0)];
        TranslationEstimationResult r = GeometricEstimation.EstimateTranslation2D(src, dst);

        Assert.True(r.Success);
        Assert.Equal(new Point2d(3, 4), r.Translation);
        Assert.Equal([1, 1, 1, 1, 1, 0], r.InlierMask);
    }

    [Fact]
    public void EstimateAffine2D_CollinearSource_FailsWithZeroMask()
    {
        Point2f[] src = Enumerable.Range(0, 20).Select(i => new Point2f(i, (2 * i) + 1)).ToArray();
        Point2f[] dst = Enumerable.Range(0, 20).Select(i => new Point2f(i * 3, i)).ToArray();
        GeometricEstimationResult r = GeometricEstimation.EstimateAffine2D(src, dst);

        Assert.False(r.Success);
        Assert.Null(r.Model);
        Assert.All(r.InlierMask, m => Assert.Equal(0, m));
    }

    [Fact]
    public void GeometricEstimation_DoesNotModifyInputs_AndIsDeterministic()
    {
        (Point2f[] src, Point2f[] dst) = Similarity(120, 0.2, 1.0, 5, 5, 0.3, seed: 4);
        for (int i = 0; i < dst.Length; i += 3)
        {
            dst[i] = new Point2f(dst[i].Y, dst[i].X);
        }

        Point2f[] srcCopy = (Point2f[])src.Clone();
        Point2f[] dstCopy = (Point2f[])dst.Clone();
        GeometricEstimationResult a = GeometricEstimation.FindHomography(src, dst, RobustEstimationMethod.Ransac);
        GeometricEstimationResult b = GeometricEstimation.FindHomography(src, dst, RobustEstimationMethod.Ransac);
        GeometricEstimationResult c = GeometricEstimation.EstimateAffinePartial2D(src, dst);
        GeometricEstimationResult d = GeometricEstimation.EstimateAffinePartial2D(src, dst);

        Assert.Equal(srcCopy, src);
        Assert.Equal(dstCopy, dst);
        Assert.Equal(a.Model!.ToArray(), b.Model!.ToArray());
        Assert.Equal(a.InlierMask, b.InlierMask);
        Assert.Equal(c.Model!.ToArray(), d.Model!.ToArray());
        Assert.Equal(c.InlierMask, d.InlierMask);
    }

    [Fact]
    public void Point2dOverloads_MatchPoint2fOverloads()
    {
        (Point2f[] src, Point2f[] dst) = Similarity(80, 0.3, 0.9, -3, 8, 0.4, seed: 5);
        Point2d[] srcD = src.Select(p => new Point2d(p.X, p.Y)).ToArray();
        Point2d[] dstD = dst.Select(p => new Point2d(p.X, p.Y)).ToArray();

        Assert.Equal(
            GeometricEstimation.EstimateAffine2D(src, dst).Model!.ToArray(),
            GeometricEstimation.EstimateAffine2D(srcD, dstD).Model!.ToArray());
        Assert.Equal(
            GeometricEstimation.FindHomography(src, dst, RobustEstimationMethod.Lmeds).Model!.ToArray(),
            GeometricEstimation.FindHomography(srcD, dstD, RobustEstimationMethod.Lmeds).Model!.ToArray());
        Assert.Equal(
            GeometricEstimation.EstimateTranslation2D(src, dst).Translation,
            GeometricEstimation.EstimateTranslation2D(srcD, dstD).Translation);
    }

    [Fact]
    public void InvalidInputs_Throw()
    {
        Point2f[] p5 = Enumerable.Range(0, 5).Select(i => new Point2f(i, i * i)).ToArray();
        Point2f[] p4 = p5[..4];

        Assert.Throws<ArgumentException>(() => GeometricEstimation.EstimateAffine2D(p5, p4));
        Assert.Throws<ArgumentException>(() => GeometricEstimation.EstimateAffine2D(p5[..1], p5[..1]));
        Assert.Throws<ArgumentException>(() => GeometricEstimation.FindHomography(p5[..3], p5[..3]));
        Assert.Throws<ArgumentException>(() => GeometricEstimation.EstimateTranslation2D(Array.Empty<Point2f>(), Array.Empty<Point2f>()));
        Assert.Throws<NotSupportedException>(() => GeometricEstimation.FindHomography(p5, p5, RobustEstimationMethod.UsacDefault));
        Assert.Throws<NotSupportedException>(() => GeometricEstimation.EstimateAffine2D(p5, p5, RobustEstimationMethod.UsacMagsac));
        Assert.Throws<NotSupportedException>(() => GeometricEstimation.EstimateAffinePartial2D(p5, p5, RobustEstimationMethod.Rho));
        Assert.Throws<ArgumentException>(() => GeometricEstimation.FindHomography(p5, p5, (RobustEstimationMethod)2));
        Assert.Throws<ArgumentOutOfRangeException>(() => GeometricEstimation.EstimateAffine2D(p5, p5, confidence: 1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => GeometricEstimation.FindHomography(p5, p5, RobustEstimationMethod.Lmeds, confidence: 0.0));
    }

    [Theory]
    [InlineData(0.99, 0.5, 4, 1000, 71)]
    [InlineData(0.99, 0.0, 4, 1000, 0)]
    [InlineData(0.99, 1.0, 4, 1000, 1000)]
    [InlineData(0.995, 0.3, 2, 2000, 8)]
    [InlineData(0.99, 0.45, 3, 2000, 25)]
    public void UpdateNumIters_MatchesRansacFormula(double p, double ep, int modelPoints, int maxIters, int expected)
        => Assert.Equal(expected, PointSetRegistrator.UpdateNumIters(p, ep, modelPoints, maxIters));

    [Theory]
    [InlineData(4)]
    [InlineData(7)]
    [InlineData(33)]
    public void FindInliers_VectorLoop_MatchesScalarErrors(int n)
    {
        var rng = new Random(n);
        Point2f[] src = Enumerable.Range(0, n).Select(_ => new Point2f(rng.NextSingle() * 100, rng.NextSingle() * 100)).ToArray();
        Point2f[] dst = Enumerable.Range(0, n).Select(_ => new Point2f(rng.NextSingle() * 100, rng.NextSingle() * 100)).ToArray();
        double[] affine = [1.01, -0.02, 3.3, 0.015, 0.99, -2.1];
        double[] homography = [1.01, -0.02, 3.3, 0.015, 0.99, -2.1, 1e-4, -2e-4, 1.0];
        double[] translation = [1, 0, 2.5, 0, 1, -1.5];

        AssertSameMask(default(Affine2DKernel), src, dst, affine);
        AssertSameMask(default(HomographyKernel), src, dst, homography);
        AssertSameMask(default(Translation2DKernel), src, dst, translation);
    }

    private static void AssertSameMask<TKernel>(TKernel kernel, Point2f[] src, Point2f[] dst, double[] model)
        where TKernel : struct, IEstimatorKernel
    {
        var err = new float[src.Length];
        kernel.ComputeError(src, dst, model, err);
        float t = err.OrderBy(e => e).ElementAt(src.Length / 2);
        var mask = new byte[src.Length];
        int count = kernel.FindInliers(src, dst, model, t, mask);
        byte[] expected = err.Select(e => e <= t ? (byte)1 : (byte)0).ToArray();
        Assert.Equal(expected, mask);
        Assert.Equal(expected.Count(b => b == 1), count);
    }

    private static (Point2f[] Src, Point2f[] Dst) Similarity(int n, double angle, double scale, double tx, double ty, double noise, int seed)
    {
        var rng = new Random(seed);
        var src = new Point2f[n];
        var dst = new Point2f[n];
        double c = scale * Math.Cos(angle), s = scale * Math.Sin(angle);
        for (int i = 0; i < n; i++)
        {
            double x = rng.NextDouble() * 1000, y = rng.NextDouble() * 1000;
            src[i] = new Point2f((float)x, (float)y);
            double nx = noise * ((rng.NextDouble() * 2) - 1), ny = noise * ((rng.NextDouble() * 2) - 1);
            dst[i] = new Point2f((float)((c * src[i].X) - (s * src[i].Y) + tx + nx), (float)((s * src[i].X) + (c * src[i].Y) + ty + ny));
        }

        return (src, dst);
    }
}
