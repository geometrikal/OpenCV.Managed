// This implementation is ported from the OpenCV RANSAC point set registrator.
//
// Original source:
//   modules/geometry/src/ptsetreg.cpp (RANSACPointSetRegistrator::run)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using System.Buffers;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>OpenCV <c>RANSACPointSetRegistrator</c>.</summary>
internal static class RansacRegistrator
{
    /// <summary>
    /// Runs RANSAC with the generator <c>RNG((uint64)-1)</c>, as OpenCV.
    /// </summary>
    /// <param name="kernel">Model estimator.</param>
    /// <param name="m1">Source points.</param>
    /// <param name="m2">Destination points.</param>
    /// <param name="threshold">Maximum reprojection error of an inlier.</param>
    /// <param name="confidence">Confidence in (0, 1).</param>
    /// <param name="maxIters">Maximum number of iterations.</param>
    /// <param name="model">Receives the best model.</param>
    /// <param name="mask">Receives the inlier mask of the best model.</param>
    /// <returns>False when no model has an inlier.</returns>
    public static bool Run<TKernel>(
        in TKernel kernel,
        ReadOnlySpan<Point2f> m1,
        ReadOnlySpan<Point2f> m2,
        double threshold,
        double confidence,
        int maxIters,
        Span<double> model,
        Span<byte> mask)
        where TKernel : struct, IEstimatorKernel
    {
        if (!(confidence > 0 && confidence < 1))
        {
            throw new ArgumentOutOfRangeException(nameof(confidence), "The confidence must be in (0, 1).");
        }

        int modelPoints = kernel.ModelPoints;
        int modelSize = kernel.ModelSize;
        int count = m1.Length;
        if (count < modelPoints)
        {
            return false;
        }

        if (count == modelPoints)
        {
            if (kernel.RunKernel(m1, m2, model) <= 0)
            {
                return false;
            }

            mask[..count].Fill(1);
            return true;
        }

        var rng = new CvRng(ulong.MaxValue);
        int niters = Math.Max(maxIters, 1);
        int maxGoodCount = 0;
        float t = (float)(threshold * threshold);

        Span<Point2f> ms1 = stackalloc Point2f[modelPoints];
        Span<Point2f> ms2 = stackalloc Point2f[modelPoints];
        Span<int> idx = stackalloc int[modelPoints];
        Span<double> current = stackalloc double[modelSize];
        Span<double> best = stackalloc double[modelSize];

        byte[] buffer = ArrayPool<byte>.Shared.Rent(2 * count);
        try
        {
            Span<byte> work = buffer.AsSpan(0, count);
            Span<byte> bestMask = buffer.AsSpan(count, count);
            for (int iter = 0; iter < niters; iter++)
            {
                if (!PointSetRegistrator.GetSubset(kernel, m1, m2, ms1, ms2, idx, ref rng, 10000))
                {
                    if (iter == 0)
                    {
                        return false;
                    }

                    break;
                }

                if (kernel.RunKernel(ms1, ms2, current) <= 0)
                {
                    continue;
                }

                int goodCount = kernel.FindInliers(m1, m2, current, t, work);
                if (goodCount > Math.Max(maxGoodCount, modelPoints - 1))
                {
                    Span<byte> swap = work;
                    work = bestMask;
                    bestMask = swap;
                    current.CopyTo(best);
                    maxGoodCount = goodCount;
                    niters = PointSetRegistrator.UpdateNumIters(confidence, (double)(count - goodCount) / count, modelPoints, niters);
                }
            }

            if (maxGoodCount == 0)
            {
                return false;
            }

            bestMask.CopyTo(mask);
            best.CopyTo(model);
            return true;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }
}
