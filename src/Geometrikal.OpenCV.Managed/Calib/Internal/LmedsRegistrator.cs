// This implementation is ported from the OpenCV least-median-of-squares point set registrator.
//
// Original source:
//   modules/geometry/src/ptsetreg.cpp (LMeDSPointSetRegistrator::run)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using System.Buffers;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>OpenCV <c>LMeDSPointSetRegistrator</c>.</summary>
internal static class LmedsRegistrator
{
    private const double OutlierRatio = 0.45;

    /// <summary>
    /// Runs LMEDS with the generator <c>RNG((uint64)-1)</c>, as OpenCV.
    /// </summary>
    /// <returns>False when no model was found or the final inlier count is below the sample size.</returns>
    public static bool Run<TKernel>(
        in TKernel kernel,
        ReadOnlySpan<Point2f> m1,
        ReadOnlySpan<Point2f> m2,
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
        int niters = PointSetRegistrator.UpdateNumIters(confidence, OutlierRatio, modelPoints, maxIters);
        niters = Math.Max(niters, 3);
        double minMedian = double.MaxValue;

        Span<Point2f> ms1 = stackalloc Point2f[modelPoints];
        Span<Point2f> ms2 = stackalloc Point2f[modelPoints];
        Span<int> idx = stackalloc int[modelPoints];
        Span<double> current = stackalloc double[modelSize];
        Span<double> best = stackalloc double[modelSize];

        float[] errBuffer = ArrayPool<float>.Shared.Rent(count);
        try
        {
            Span<float> err = errBuffer.AsSpan(0, count);
            for (int iter = 0; iter < niters; iter++)
            {
                if (!PointSetRegistrator.GetSubset(kernel, m1, m2, ms1, ms2, idx, ref rng, 1000))
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

                kernel.ComputeError(m1, m2, current, err);
                double median = SelectNth(MemoryMarshal.Cast<float, int>(err), count / 2);
                if (median < minMedian)
                {
                    minMedian = median;
                    current.CopyTo(best);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(errBuffer);
        }

        if (!(minMedian < double.MaxValue))
        {
            return false;
        }

        double sigma = 2.5 * 1.4826 * (1 + (5.0 / (count - modelPoints))) * Math.Sqrt(minMedian);
        sigma = Math.Max(sigma, 0.001);

        int inliers = kernel.FindInliers(m1, m2, best, (float)(sigma * sigma), mask);
        best.CopyTo(model);
        return inliers >= modelPoints;
    }

    /// <summary>
    /// <c>std::nth_element</c> over the float bit patterns read as <c>int</c>, as OpenCV.
    /// </summary>
    /// <returns>The <paramref name="k"/>-th smallest value, as a float widened to double.</returns>
    private static double SelectNth(Span<int> a, int k)
    {
        int lo = 0, hi = a.Length - 1;
        while (lo < hi)
        {
            int pivot = a[lo + ((hi - lo) >> 1)];
            int i = lo, j = hi;
            while (i <= j)
            {
                while (a[i] < pivot)
                {
                    i++;
                }

                while (a[j] > pivot)
                {
                    j--;
                }

                if (i <= j)
                {
                    (a[i], a[j]) = (a[j], a[i]);
                    i++;
                    j--;
                }
            }

            if (k <= j)
            {
                hi = j;
            }
            else if (k >= i)
            {
                lo = i;
            }
            else
            {
                break;
            }
        }

        return BitConverter.Int32BitsToSingle(a[k]);
    }
}
