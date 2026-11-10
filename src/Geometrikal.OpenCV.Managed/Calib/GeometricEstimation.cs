// This API is based on the OpenCV robust geometric estimation functions.
//
// Original sources:
//   modules/geometry/include/opencv2/geometry/3d.hpp (declarations)
//   modules/geometry/src/ptsetreg.cpp (estimateAffine2D, estimateAffinePartial2D,
//   estimateTranslation2D)
//   modules/geometry/src/fundam.cpp (findHomography)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib;

using System.Buffers;
using Geometrikal.OpenCV.Managed.Calib.Internal;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// The result of a robust model estimation.
/// </summary>
/// <param name="Model">
/// The estimated model as a <c>double</c> matrix (2 × 3 affine or 3 × 3 homography), or null when
/// the estimation failed.
/// </param>
/// <param name="InlierMask">
/// One byte per input pair: 1 for an inlier and 0 for an outlier. All bytes are 0 when the
/// estimation failed.
/// </param>
public sealed record GeometricEstimationResult(Mat<double>? Model, byte[] InlierMask)
{
    /// <summary>Gets a value indicating whether a model was found.</summary>
    public bool Success => Model is not null;
}

/// <summary>
/// The result of <see cref="GeometricEstimation.EstimateTranslation2D(ReadOnlySpan{Point2f}, ReadOnlySpan{Point2f}, RobustEstimationMethod, double, int, double, int)"/>.
/// </summary>
/// <param name="Translation">The translation (tx, ty), or (NaN, NaN) when the estimation failed.</param>
/// <param name="InlierMask">One byte per input pair: 1 for an inlier and 0 for an outlier.</param>
public sealed record TranslationEstimationResult(Point2d Translation, byte[] InlierMask)
{
    /// <summary>Gets a value indicating whether a translation was found.</summary>
    public bool Success => !double.IsNaN(Translation.X);
}

/// <summary>
/// Robust estimation of 2-D transforms between matched point sets, ported from the OpenCV 5.0
/// <c>geometry</c> module: <c>estimateAffine2D</c>, <c>estimateAffinePartial2D</c>,
/// <c>estimateTranslation2D</c> and <c>findHomography</c>.
/// </summary>
/// <remarks>
/// <para>
/// The functions do not change the input points. <c>double</c> points are rounded to
/// <c>float</c> first, as OpenCV converts all inputs to <c>CV_32FC2</c>.
/// </para>
/// <para>
/// RANSAC and LMEDS use the OpenCV generator <c>RNG((uint64)-1)</c> in every call; RHO seeds its
/// XORshift generator with <c>~0</c> in every call. The results are deterministic and equal to
/// OpenCV 5.0: the same minimal samples, the same inlier masks and the same models. The
/// Levenberg-Marquardt refinement is the OpenCV 5.0 <c>LevMarq</c> solver with geodesic
/// acceleration. The functions are thread-safe.
/// </para>
/// <para>
/// The USAC methods (<see cref="RobustEstimationMethod.UsacDefault"/> and higher) are not
/// supported and throw <see cref="NotSupportedException"/>. See
/// <c>Docs/Notes/EstimationPortNotes.md</c>.
/// </para>
/// </remarks>
public static class GeometricEstimation
{
    /// <summary>
    /// Computes an optimal affine transform with 6 degrees of freedom between two point sets, as
    /// OpenCV <c>estimateAffine2D</c>.
    /// </summary>
    /// <param name="from">Source points.</param>
    /// <param name="to">Destination points. The length must equal the length of <paramref name="from"/>.</param>
    /// <param name="method"><see cref="RobustEstimationMethod.Ransac"/> or <see cref="RobustEstimationMethod.Lmeds"/>.</param>
    /// <param name="ransacReprojThreshold">Maximum reprojection error of a RANSAC inlier, in pixels.</param>
    /// <param name="maxIters">Maximum number of robust iterations.</param>
    /// <param name="confidence">Confidence level in (0, 1).</param>
    /// <param name="refineIters">Maximum number of Levenberg-Marquardt iterations on the inliers. 0 disables the refinement.</param>
    /// <returns>A 2 × 3 model and the inlier mask.</returns>
    /// <exception cref="ArgumentException">There are fewer than 2 pairs, or the lengths differ.</exception>
    /// <exception cref="NotSupportedException"><paramref name="method"/> is a USAC method.</exception>
    public static GeometricEstimationResult EstimateAffine2D(
        ReadOnlySpan<Point2f> from,
        ReadOnlySpan<Point2f> to,
        RobustEstimationMethod method = RobustEstimationMethod.Ransac,
        double ransacReprojThreshold = 3,
        int maxIters = 2000,
        double confidence = 0.99,
        int refineIters = 10)
        => EstimateAffine(from, to, false, method, ransacReprojThreshold, maxIters, confidence, refineIters);

    /// <inheritdoc cref="EstimateAffine2D(ReadOnlySpan{Point2f}, ReadOnlySpan{Point2f}, RobustEstimationMethod, double, int, double, int)"/>
    public static GeometricEstimationResult EstimateAffine2D(
        ReadOnlySpan<Point2d> from,
        ReadOnlySpan<Point2d> to,
        RobustEstimationMethod method = RobustEstimationMethod.Ransac,
        double ransacReprojThreshold = 3,
        int maxIters = 2000,
        double confidence = 0.99,
        int refineIters = 10)
    {
        Point2f[] f = RentFloat(from), t = RentFloat(to);
        try
        {
            return EstimateAffine(f.AsSpan(0, from.Length), t.AsSpan(0, to.Length), false, method, ransacReprojThreshold, maxIters, confidence, refineIters);
        }
        finally
        {
            ArrayPool<Point2f>.Shared.Return(f);
            ArrayPool<Point2f>.Shared.Return(t);
        }
    }

    /// <summary>
    /// Computes an optimal similarity transform (rotation, uniform scale and translation; 4
    /// degrees of freedom) between two point sets, as OpenCV <c>estimateAffinePartial2D</c>.
    /// </summary>
    /// <param name="from">Source points.</param>
    /// <param name="to">Destination points. The length must equal the length of <paramref name="from"/>.</param>
    /// <param name="method"><see cref="RobustEstimationMethod.Ransac"/> or <see cref="RobustEstimationMethod.Lmeds"/>.</param>
    /// <param name="ransacReprojThreshold">Maximum reprojection error of a RANSAC inlier, in pixels.</param>
    /// <param name="maxIters">Maximum number of robust iterations.</param>
    /// <param name="confidence">Confidence level in (0, 1).</param>
    /// <param name="refineIters">Maximum number of Levenberg-Marquardt iterations on the inliers. 0 disables the refinement.</param>
    /// <returns>A 2 × 3 model <c>[a -b tx; b a ty]</c> and the inlier mask.</returns>
    /// <exception cref="ArgumentException">There are fewer than 2 pairs, or the lengths differ.</exception>
    /// <exception cref="NotSupportedException"><paramref name="method"/> is a USAC method. OpenCV 5.0 rejects USAC for this function.</exception>
    public static GeometricEstimationResult EstimateAffinePartial2D(
        ReadOnlySpan<Point2f> from,
        ReadOnlySpan<Point2f> to,
        RobustEstimationMethod method = RobustEstimationMethod.Ransac,
        double ransacReprojThreshold = 3,
        int maxIters = 2000,
        double confidence = 0.99,
        int refineIters = 10)
        => EstimateAffine(from, to, true, method, ransacReprojThreshold, maxIters, confidence, refineIters);

    /// <inheritdoc cref="EstimateAffinePartial2D(ReadOnlySpan{Point2f}, ReadOnlySpan{Point2f}, RobustEstimationMethod, double, int, double, int)"/>
    public static GeometricEstimationResult EstimateAffinePartial2D(
        ReadOnlySpan<Point2d> from,
        ReadOnlySpan<Point2d> to,
        RobustEstimationMethod method = RobustEstimationMethod.Ransac,
        double ransacReprojThreshold = 3,
        int maxIters = 2000,
        double confidence = 0.99,
        int refineIters = 10)
    {
        Point2f[] f = RentFloat(from), t = RentFloat(to);
        try
        {
            return EstimateAffine(f.AsSpan(0, from.Length), t.AsSpan(0, to.Length), true, method, ransacReprojThreshold, maxIters, confidence, refineIters);
        }
        finally
        {
            ArrayPool<Point2f>.Shared.Return(f);
            ArrayPool<Point2f>.Shared.Return(t);
        }
    }

    /// <summary>
    /// Computes an optimal translation between two point sets, as OpenCV
    /// <c>estimateTranslation2D</c>.
    /// </summary>
    /// <param name="from">Source points.</param>
    /// <param name="to">Destination points. The length must equal the length of <paramref name="from"/>.</param>
    /// <param name="method"><see cref="RobustEstimationMethod.Ransac"/> or <see cref="RobustEstimationMethod.Lmeds"/>.</param>
    /// <param name="ransacReprojThreshold">Maximum reprojection error of a RANSAC inlier, in pixels.</param>
    /// <param name="maxIters">Maximum number of robust iterations.</param>
    /// <param name="confidence">Confidence level in (0, 1).</param>
    /// <param name="refineIters">
    /// Maximum number of Levenberg-Marquardt iterations on the inliers. 0 selects the closed-form
    /// mean displacement of the inliers.
    /// </param>
    /// <returns>The translation and the inlier mask.</returns>
    /// <exception cref="ArgumentException">The point sets are empty, or the lengths differ.</exception>
    /// <exception cref="NotSupportedException"><paramref name="method"/> is not RANSAC or LMEDS.</exception>
    public static TranslationEstimationResult EstimateTranslation2D(
        ReadOnlySpan<Point2f> from,
        ReadOnlySpan<Point2f> to,
        RobustEstimationMethod method = RobustEstimationMethod.Ransac,
        double ransacReprojThreshold = 3,
        int maxIters = 2000,
        double confidence = 0.99,
        int refineIters = 0)
        => EstimateTranslation(from, to, method, ransacReprojThreshold, maxIters, confidence, refineIters);

    /// <inheritdoc cref="EstimateTranslation2D(ReadOnlySpan{Point2f}, ReadOnlySpan{Point2f}, RobustEstimationMethod, double, int, double, int)"/>
    public static TranslationEstimationResult EstimateTranslation2D(
        ReadOnlySpan<Point2d> from,
        ReadOnlySpan<Point2d> to,
        RobustEstimationMethod method = RobustEstimationMethod.Ransac,
        double ransacReprojThreshold = 3,
        int maxIters = 2000,
        double confidence = 0.99,
        int refineIters = 0)
    {
        Point2f[] f = RentFloat(from), t = RentFloat(to);
        try
        {
            return EstimateTranslation(f.AsSpan(0, from.Length), t.AsSpan(0, to.Length), method, ransacReprojThreshold, maxIters, confidence, refineIters);
        }
        finally
        {
            ArrayPool<Point2f>.Shared.Return(f);
            ArrayPool<Point2f>.Shared.Return(t);
        }
    }

    /// <summary>
    /// Finds a perspective transform between two planes, as OpenCV <c>findHomography</c>.
    /// </summary>
    /// <param name="srcPoints">Source points.</param>
    /// <param name="dstPoints">Destination points. The length must equal the length of <paramref name="srcPoints"/>.</param>
    /// <param name="method">
    /// <see cref="RobustEstimationMethod.None"/> (least squares over all points),
    /// <see cref="RobustEstimationMethod.Ransac"/>, <see cref="RobustEstimationMethod.Lmeds"/> or
    /// <see cref="RobustEstimationMethod.Rho"/>. With exactly 4 pairs the method is ignored and the
    /// direct solution is returned.
    /// </param>
    /// <param name="ransacReprojThreshold">
    /// Maximum reprojection error of an inlier, in pixels. A value ≤ 0 selects 3. All methods except
    /// RHO use this value to compute the final mask after the refinement.
    /// </param>
    /// <param name="maxIters">Maximum number of robust iterations.</param>
    /// <param name="confidence">Confidence level in (0, 1).</param>
    /// <returns>
    /// A 3 × 3 model with element (2, 2) normalized to 1 (when its magnitude is above
    /// <c>FLT_EPSILON</c>), and the inlier mask.
    /// </returns>
    /// <exception cref="ArgumentException">There are fewer than 4 pairs, or the lengths differ.</exception>
    /// <exception cref="NotSupportedException"><paramref name="method"/> is a USAC method.</exception>
    public static GeometricEstimationResult FindHomography(
        ReadOnlySpan<Point2f> srcPoints,
        ReadOnlySpan<Point2f> dstPoints,
        RobustEstimationMethod method = RobustEstimationMethod.None,
        double ransacReprojThreshold = 3,
        int maxIters = 2000,
        double confidence = 0.995)
        => Homography(srcPoints, dstPoints, method, ransacReprojThreshold, maxIters, confidence);

    /// <inheritdoc cref="FindHomography(ReadOnlySpan{Point2f}, ReadOnlySpan{Point2f}, RobustEstimationMethod, double, int, double)"/>
    public static GeometricEstimationResult FindHomography(
        ReadOnlySpan<Point2d> srcPoints,
        ReadOnlySpan<Point2d> dstPoints,
        RobustEstimationMethod method = RobustEstimationMethod.None,
        double ransacReprojThreshold = 3,
        int maxIters = 2000,
        double confidence = 0.995)
    {
        Point2f[] f = RentFloat(srcPoints), t = RentFloat(dstPoints);
        try
        {
            return Homography(f.AsSpan(0, srcPoints.Length), t.AsSpan(0, dstPoints.Length), method, ransacReprojThreshold, maxIters, confidence);
        }
        finally
        {
            ArrayPool<Point2f>.Shared.Return(f);
            ArrayPool<Point2f>.Shared.Return(t);
        }
    }

    private static GeometricEstimationResult EstimateAffine(
        ReadOnlySpan<Point2f> from,
        ReadOnlySpan<Point2f> to,
        bool partial,
        RobustEstimationMethod method,
        double threshold,
        int maxIters,
        double confidence,
        int refineIters)
    {
        int count = from.Length;
        if (count < 2)
        {
            throw new ArgumentException("At least 2 points are needed for affine transformation estimation.", nameof(from));
        }

        RequireSameLength(count, to.Length);
        if (!partial && IsUsac(method))
        {
            throw new NotSupportedException("The USAC methods are not supported.");
        }

        var mask = new byte[count];
        double[] h = new double[6];
        bool result;
        if (partial)
        {
            var kernel = default(AffinePartial2DKernel);
            result = RunRobust(kernel, from, to, method, threshold, confidence, maxIters, h, mask);
        }
        else
        {
            var kernel = default(Affine2DKernel);
            result = RunRobust(kernel, from, to, method, threshold, confidence, maxIters, h, mask);
        }

        int minCount = partial ? 2 : 3;
        if (result && count > minCount && refineIters > 0)
        {
            Point2f[] fromIn = ArrayPool<Point2f>.Shared.Rent(count);
            Point2f[] toIn = ArrayPool<Point2f>.Shared.Rent(count);
            try
            {
                int inliers = CompressInliers(from, to, mask, fromIn, toIn);
                if (inliers > 0)
                {
                    if (partial)
                    {
                        double[] v = [h[0], h[3], h[2], h[5]];
                        LevMarq.Optimize(v, new AffinePartialRefineProblem(fromIn, toIn, inliers), refineIters);
                        h[0] = v[0];
                        h[4] = v[0];
                        h[1] = -v[1];
                        h[2] = v[2];
                        h[3] = v[1];
                        h[5] = v[3];
                    }
                    else
                    {
                        LevMarq.Optimize(h, new AffineRefineProblem(fromIn, toIn, inliers), refineIters);
                    }
                }
            }
            finally
            {
                ArrayPool<Point2f>.Shared.Return(fromIn);
                ArrayPool<Point2f>.Shared.Return(toIn);
            }
        }

        if (!result)
        {
            Array.Clear(mask);
            return new GeometricEstimationResult(null, mask);
        }

        return new GeometricEstimationResult(new Mat<double>(h, 3, 2), mask);
    }

    private static TranslationEstimationResult EstimateTranslation(
        ReadOnlySpan<Point2f> from,
        ReadOnlySpan<Point2f> to,
        RobustEstimationMethod method,
        double threshold,
        int maxIters,
        double confidence,
        int refineIters)
    {
        int count = from.Length;
        if (count == 0)
        {
            throw new ArgumentException("The point sets must not be empty.", nameof(from));
        }

        RequireSameLength(count, to.Length);
        var mask = new byte[count];
        double[] t = new double[6];
        var kernel = default(Translation2DKernel);
        bool result = RunRobust(kernel, from, to, method, threshold, confidence, maxIters, t, mask);
        if (!result)
        {
            Array.Clear(mask);
            return new TranslationEstimationResult(new Point2d(double.NaN, double.NaN), mask);
        }

        Point2f[] fromIn = ArrayPool<Point2f>.Shared.Rent(count);
        Point2f[] toIn = ArrayPool<Point2f>.Shared.Rent(count);
        try
        {
            int nin = CompressInliers(from, to, mask, fromIn, toIn);
            if (nin > 0)
            {
                if (refineIters > 0)
                {
                    double[] v = [t[2], t[5]];
                    LevMarq.Optimize(v, new TranslationRefineProblem(fromIn, toIn, nin), refineIters);
                    t[2] = v[0];
                    t[5] = v[1];
                }
                else
                {
                    double sx = 0.0, sy = 0.0;
                    for (int i = 0; i < nin; i++)
                    {
                        sx += (double)toIn[i].X - fromIn[i].X;
                        sy += (double)toIn[i].Y - fromIn[i].Y;
                    }

                    t[2] = sx / nin;
                    t[5] = sy / nin;
                }
            }
        }
        finally
        {
            ArrayPool<Point2f>.Shared.Return(fromIn);
            ArrayPool<Point2f>.Shared.Return(toIn);
        }

        return new TranslationEstimationResult(new Point2d(t[2], t[5]), mask);
    }

    private static GeometricEstimationResult Homography(
        ReadOnlySpan<Point2f> src,
        ReadOnlySpan<Point2f> dst,
        RobustEstimationMethod method,
        double threshold,
        int maxIters,
        double confidence)
    {
        if (IsUsac(method))
        {
            throw new NotSupportedException("The USAC methods are not supported.");
        }

        int npoints = src.Length;
        if (npoints < 4)
        {
            throw new ArgumentException("At least 4 point pairs are needed to calculate a homography.", nameof(src));
        }

        RequireSameLength(npoints, dst.Length);
        if (threshold <= 0)
        {
            threshold = 3;
        }

        var kernel = default(HomographyKernel);
        var mask = new byte[npoints];
        double[] h = new double[9];
        bool result;
        if (method == RobustEstimationMethod.None || npoints == 4)
        {
            mask.AsSpan().Fill(1);
            result = kernel.RunKernel(src, dst, h) > 0;
        }
        else if (method == RobustEstimationMethod.Ransac)
        {
            result = RansacRegistrator.Run(kernel, src, dst, threshold, confidence, maxIters, h, mask);
        }
        else if (method == RobustEstimationMethod.Lmeds)
        {
            result = LmedsRegistrator.Run(kernel, src, dst, confidence, maxIters, h, mask);
        }
        else if (method == RobustEstimationMethod.Rho)
        {
            result = RhoEstimator.Run(src, dst, threshold, maxIters, confidence, h, mask);
        }
        else
        {
            throw new ArgumentException("Unknown estimation method.", nameof(method));
        }

        if (result && npoints > 4 && method != RobustEstimationMethod.Rho)
        {
            Point2f[] srcIn = ArrayPool<Point2f>.Shared.Rent(npoints);
            Point2f[] dstIn = ArrayPool<Point2f>.Shared.Rent(npoints);
            try
            {
                int inliers = CompressInliers(src, dst, mask, srcIn, dstIn);
                if (inliers > 0)
                {
                    if (method is RobustEstimationMethod.Ransac or RobustEstimationMethod.Lmeds)
                    {
                        kernel.RunKernel(srcIn.AsSpan(0, inliers), dstIn.AsSpan(0, inliers), h);
                    }

                    LevMarq.Optimize(h, new HomographyRefineProblem(srcIn, dstIn, inliers), 10);
                    PointSetRegistrator.ScaleInPlace(h, PointSetRegistrator.ScaleFor(h[8]));
                    kernel.FindInliers(src, dst, h, (float)(threshold * threshold), mask);
                }
            }
            finally
            {
                ArrayPool<Point2f>.Shared.Return(srcIn);
                ArrayPool<Point2f>.Shared.Return(dstIn);
            }
        }

        if (!result)
        {
            Array.Clear(mask);
            return new GeometricEstimationResult(null, mask);
        }

        return new GeometricEstimationResult(new Mat<double>(h, 3, 3), mask);
    }

    private static bool RunRobust<TKernel>(
        in TKernel kernel,
        ReadOnlySpan<Point2f> from,
        ReadOnlySpan<Point2f> to,
        RobustEstimationMethod method,
        double threshold,
        double confidence,
        int maxIters,
        Span<double> model,
        Span<byte> mask)
        where TKernel : struct, IEstimatorKernel
    {
        return method switch
        {
            RobustEstimationMethod.Ransac => RansacRegistrator.Run(kernel, from, to, threshold, confidence, maxIters, model, mask),
            RobustEstimationMethod.Lmeds => LmedsRegistrator.Run(kernel, from, to, confidence, maxIters, model, mask),
            _ => throw new NotSupportedException("Unknown or unsupported robust estimation method."),
        };
    }

    private static bool IsUsac(RobustEstimationMethod method)
        => method >= RobustEstimationMethod.UsacDefault && method <= RobustEstimationMethod.UsacMagsac;

    private static void RequireSameLength(int a, int b)
    {
        if (a != b)
        {
            throw new ArgumentException("Point matches need to have the same size.");
        }
    }

    /// <summary>Copies the pairs with a non-zero mask to the front of the destination arrays, as OpenCV <c>compressElems</c>.</summary>
    /// <returns>The number of copied pairs.</returns>
    private static int CompressInliers(ReadOnlySpan<Point2f> from, ReadOnlySpan<Point2f> to, ReadOnlySpan<byte> mask, Point2f[] fromIn, Point2f[] toIn)
    {
        int j = 0;
        for (int i = 0; i < from.Length; i++)
        {
            if (mask[i] != 0)
            {
                fromIn[j] = from[i];
                toIn[j] = to[i];
                j++;
            }
        }

        return j;
    }

    private static Point2f[] RentFloat(ReadOnlySpan<Point2d> points)
    {
        Point2f[] result = ArrayPool<Point2f>.Shared.Rent(points.Length);
        for (int i = 0; i < points.Length; i++)
        {
            result[i] = new Point2f((float)points[i].X, (float)points[i].Y);
        }

        return result;
    }
}
