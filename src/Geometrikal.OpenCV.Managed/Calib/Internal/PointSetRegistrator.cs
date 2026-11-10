// This implementation is ported from the OpenCV point set registrators.
//
// Original sources:
//   modules/geometry/src/ptsetreg.cpp (RANSACUpdateNumIters, getSubset, findInliers)
//   modules/geometry/src/precomp.hpp (compressElems, haveCollinearPoints)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// A model estimator, as OpenCV <c>PointSetRegistrator::Callback</c>. Every kernel gives one
/// model per minimal sample.
/// </summary>
internal interface IEstimatorKernel
{
    /// <summary>Gets the size of a minimal sample.</summary>
    int ModelPoints { get; }

    /// <summary>Gets the number of <c>double</c> model coefficients (6 or 9).</summary>
    int ModelSize { get; }

    /// <summary><c>runKernel</c>: computes a model from the point pairs.</summary>
    /// <returns>The number of models (0 or 1).</returns>
    int RunKernel(ReadOnlySpan<Point2f> m1, ReadOnlySpan<Point2f> m2, Span<double> model);

    /// <summary><c>computeError</c>: squared reprojection error of every pair.</summary>
    void ComputeError(ReadOnlySpan<Point2f> m1, ReadOnlySpan<Point2f> m2, ReadOnlySpan<double> model, Span<float> err);

    /// <summary>
    /// <c>computeError</c> followed by <c>findInliers</c>: sets <c>mask[i] = err[i] &lt;= t</c>.
    /// </summary>
    /// <returns>The number of inliers.</returns>
    int FindInliers(ReadOnlySpan<Point2f> m1, ReadOnlySpan<Point2f> m2, ReadOnlySpan<double> model, float t, Span<byte> mask);

    /// <summary><c>checkSubset</c>: false for a degenerate minimal sample.</summary>
    bool CheckSubset(ReadOnlySpan<Point2f> ms1, ReadOnlySpan<Point2f> ms2, int count);
}

/// <summary>Helpers shared by the RANSAC and LMEDS registrators.</summary>
internal static class PointSetRegistrator
{
    private const double DblMin = 2.2250738585072014e-308;

    /// <summary>
    /// OpenCV <c>RANSACUpdateNumIters</c>: the number of iterations for confidence
    /// <paramref name="p"/> at outlier ratio <paramref name="ep"/>.
    /// </summary>
    public static int UpdateNumIters(double p, double ep, int modelPoints, int maxIters)
    {
        if (modelPoints <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(modelPoints), "The number of model points should be positive.");
        }

        p = p < 0.0 ? 0.0 : p;
        p = p > 1.0 ? 1.0 : p;
        ep = ep < 0.0 ? 0.0 : ep;
        ep = ep > 1.0 ? 1.0 : ep;

        double num = 1.0 - p < DblMin ? DblMin : 1.0 - p;
        double denom = 1.0 - Math.Pow(1.0 - ep, modelPoints);
        if (denom < DblMin)
        {
            return 0;
        }

        num = Math.Log(num);
        denom = Math.Log(denom);

        return denom >= 0 || -num >= maxIters * (-denom) ? maxIters : ScalarConverter.CvRound(num / denom);
    }

    /// <summary>
    /// OpenCV <c>getSubset</c>: draws <paramref name="ms1"/>.Length distinct indices with
    /// <paramref name="rng"/> until the kernel accepts the sample.
    /// </summary>
    /// <returns>False when no sample passes the check in <paramref name="maxAttempts"/> attempts.</returns>
    public static bool GetSubset<TKernel>(
        in TKernel kernel,
        ReadOnlySpan<Point2f> m1,
        ReadOnlySpan<Point2f> m2,
        Span<Point2f> ms1,
        Span<Point2f> ms2,
        Span<int> idx,
        ref CvRng rng,
        int maxAttempts)
        where TKernel : struct, IEstimatorKernel
    {
        int modelPoints = ms1.Length;
        int count = m1.Length;
        for (int iters = 0; iters < maxAttempts; iters++)
        {
            for (int i = 0; i < modelPoints; i++)
            {
                int idxI;
                do
                {
                    idxI = rng.Uniform(0, count);
                }
                while (idx[..i].Contains(idxI));

                idx[i] = idxI;
                ms1[i] = m1[idxI];
                ms2[i] = m2[idxI];
            }

            if (kernel.CheckSubset(ms1, ms2, modelPoints))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>OpenCV <c>compressElems</c>: moves the elements with a non-zero mask to the front.</summary>
    /// <returns>The number of moved elements.</returns>
    public static int CompressElems(Span<Point2f> ptr, ReadOnlySpan<byte> mask)
    {
        int j = 0;
        for (int i = 0; i < ptr.Length; i++)
        {
            if (mask[i] != 0)
            {
                if (i > j)
                {
                    ptr[j] = ptr[i];
                }

                j++;
            }
        }

        return j;
    }

    /// <summary>
    /// OpenCV <c>haveCollinearPoints</c>: true when the last of the first <paramref name="count"/>
    /// points is on a line through two earlier points, or two points are too close.
    /// </summary>
    public static bool HaveCollinearPoints(ReadOnlySpan<Point2f> ptr, int count)
    {
        const double FltEpsilon = 1.1920928955078125e-07;
        int i = count - 1;
        for (int j = 0; j < i; j++)
        {
            // The differences are float subtractions, as in the C++ source.
            double dx1 = ptr[j].X - ptr[i].X;
            double dy1 = ptr[j].Y - ptr[i].Y;
            for (int k = 0; k < j; k++)
            {
                double dx2 = ptr[k].X - ptr[i].X;
                double dy2 = ptr[k].Y - ptr[i].Y;
                if (Math.Abs((dx2 * dy1) - (dy2 * dx1)) <= FltEpsilon * (Math.Abs(dx1) + Math.Abs(dy1) + Math.Abs(dx2) + Math.Abs(dy2)))
                {
                    return true;
                }
            }
        }

        return false;
    }

    /// <summary>OpenCV <c>scaleFor</c>: <c>1/x</c>, or 1 when |x| is at most <c>FLT_EPSILON</c>.</summary>
    public static double ScaleFor(double x) => Math.Abs(x) > 1.1920928955078125e-07 ? 1.0 / x : 1.0;

    /// <summary>
    /// <c>Mat::convertTo</c> of a <c>double</c> matrix to <c>double</c> with a scale: a copy when the
    /// scale is 1 within <c>DBL_EPSILON</c>, otherwise <c>x·scale + 0</c>.
    /// </summary>
    public static void ScaleInPlace(Span<double> values, double scale)
    {
        if (Math.Abs(scale - 1.0) < 2.220446049250313e-16)
        {
            return;
        }

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (values[i] * scale) + 0.0;
        }
    }
}
