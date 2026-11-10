// This implementation is ported from the OpenCV point set registrators.
//
// Original source:
//   modules/geometry/src/ptsetreg.cpp (RANSACPointSetRegistrator::run,
//   LMeDSPointSetRegistrator::run, getSubset, findInliers)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Calib.Internal;

using System.Buffers;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// A model estimator that gives several 3 × 3 models per minimal sample, as the OpenCV
/// <c>FMEstimatorCallback</c> and <c>EMEstimatorCallback</c>.
/// </summary>
/// <typeparam name="TPoint">The point type of the registrator input.</typeparam>
internal interface IMultiModelKernel<TPoint>
    where TPoint : unmanaged
{
    /// <summary>Gets the size of a minimal sample.</summary>
    int ModelPoints { get; }

    /// <summary>Gets the maximum number of models of one <see cref="RunKernel"/> call.</summary>
    int MaxModels { get; }

    /// <summary><c>runKernel</c>: writes 9 coefficients per model.</summary>
    /// <returns>The number of models. A value ≤ 0 means no model.</returns>
    int RunKernel(ReadOnlySpan<TPoint> m1, ReadOnlySpan<TPoint> m2, Span<double> models);

    /// <summary><c>computeError</c>: the error of every pair.</summary>
    void ComputeError(ReadOnlySpan<TPoint> m1, ReadOnlySpan<TPoint> m2, ReadOnlySpan<double> model, Span<float> err);

    /// <summary><c>checkSubset</c>: false for a degenerate minimal sample.</summary>
    bool CheckSubset(ReadOnlySpan<TPoint> ms1, ReadOnlySpan<TPoint> ms2, int count);
}

/// <summary>
/// OpenCV <c>RANSACPointSetRegistrator</c> and <c>LMeDSPointSetRegistrator</c> for kernels with
/// several models per sample.
/// </summary>
internal static class MultiModelRegistrator
{
    /// <summary>
    /// <c>RANSACPointSetRegistrator::run</c> with the generator <c>RNG((uint64)-1)</c>.
    /// </summary>
    /// <param name="models">Receives the models; room for <c>MaxModels · 9</c> values.</param>
    /// <param name="mask">Receives the inlier mask. Not written when the method returns 0.</param>
    /// <returns>The number of 3 × 3 models written to <paramref name="models"/>. 0 is a failure.</returns>
    public static int Ransac<TPoint, TKernel>(
        in TKernel kernel,
        ReadOnlySpan<TPoint> m1,
        ReadOnlySpan<TPoint> m2,
        double threshold,
        double confidence,
        int maxIters,
        Span<double> models,
        Span<byte> mask)
        where TPoint : unmanaged
        where TKernel : struct, IMultiModelKernel<TPoint>
    {
        CheckConfidence(confidence);
        int modelPoints = kernel.ModelPoints;
        int count = m1.Length;
        if (count < modelPoints)
        {
            return 0;
        }

        if (count == modelPoints)
        {
            int n = kernel.RunKernel(m1, m2, models);
            if (n <= 0)
            {
                return 0;
            }

            mask[..count].Fill(1);
            return n;
        }

        var rng = new CvRng(ulong.MaxValue);
        int niters = Math.Max(maxIters, 1);
        int maxGoodCount = 0;
        float t = (float)(threshold * threshold);

        Span<TPoint> ms1 = stackalloc TPoint[modelPoints];
        Span<TPoint> ms2 = stackalloc TPoint[modelPoints];
        Span<int> idx = stackalloc int[modelPoints];
        Span<double> current = stackalloc double[kernel.MaxModels * 9];

        byte[] buffer = ArrayPool<byte>.Shared.Rent(2 * count);
        float[] errBuffer = ArrayPool<float>.Shared.Rent(count);
        try
        {
            Span<byte> work = buffer.AsSpan(0, count);
            Span<byte> bestMask = buffer.AsSpan(count, count);
            Span<float> err = errBuffer.AsSpan(0, count);
            for (int iter = 0; iter < niters; iter++)
            {
                if (!GetSubset(kernel, m1, m2, ms1, ms2, idx, ref rng, 10000))
                {
                    if (iter == 0)
                    {
                        return 0;
                    }

                    break;
                }

                int nmodels = kernel.RunKernel(ms1, ms2, current);
                for (int i = 0; i < nmodels; i++)
                {
                    ReadOnlySpan<double> modelI = current.Slice(i * 9, 9);
                    kernel.ComputeError(m1, m2, modelI, err);
                    int goodCount = 0;
                    for (int k = 0; k < count; k++)
                    {
                        int f = err[k] <= t ? 1 : 0;
                        work[k] = (byte)f;
                        goodCount += f;
                    }

                    if (goodCount > Math.Max(maxGoodCount, modelPoints - 1))
                    {
                        Span<byte> swap = work;
                        work = bestMask;
                        bestMask = swap;
                        modelI.CopyTo(models);
                        maxGoodCount = goodCount;
                        niters = PointSetRegistrator.UpdateNumIters(confidence, (double)(count - goodCount) / count, modelPoints, niters);
                    }
                }
            }

            if (maxGoodCount == 0)
            {
                return 0;
            }

            bestMask.CopyTo(mask);
            return 1;
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
            ArrayPool<float>.Shared.Return(errBuffer);
        }
    }

    /// <summary>
    /// <c>LMeDSPointSetRegistrator::run</c> with the generator <c>RNG((uint64)-1)</c>.
    /// </summary>
    /// <param name="success">
    /// The return value of OpenCV <c>run</c>: false when no model was found or the final inlier
    /// count is below the sample size. The model and the mask can be written also when false.
    /// </param>
    /// <returns>The number of 3 × 3 models written to <paramref name="models"/>.</returns>
    public static int Lmeds<TPoint, TKernel>(
        in TKernel kernel,
        ReadOnlySpan<TPoint> m1,
        ReadOnlySpan<TPoint> m2,
        double confidence,
        int maxIters,
        Span<double> models,
        Span<byte> mask,
        out bool success)
        where TPoint : unmanaged
        where TKernel : struct, IMultiModelKernel<TPoint>
    {
        const double OutlierRatio = 0.45;
        CheckConfidence(confidence);
        success = false;
        int modelPoints = kernel.ModelPoints;
        int count = m1.Length;
        if (count < modelPoints)
        {
            return 0;
        }

        if (count == modelPoints)
        {
            int n = kernel.RunKernel(m1, m2, models);
            if (n <= 0)
            {
                return 0;
            }

            mask[..count].Fill(1);
            success = true;
            return n;
        }

        var rng = new CvRng(ulong.MaxValue);
        int niters = PointSetRegistrator.UpdateNumIters(confidence, OutlierRatio, modelPoints, maxIters);
        niters = Math.Max(niters, 3);
        double minMedian = double.MaxValue;

        Span<TPoint> ms1 = stackalloc TPoint[modelPoints];
        Span<TPoint> ms2 = stackalloc TPoint[modelPoints];
        Span<int> idx = stackalloc int[modelPoints];
        Span<double> current = stackalloc double[kernel.MaxModels * 9];
        Span<double> best = stackalloc double[9];

        float[] errBuffer = ArrayPool<float>.Shared.Rent(count);
        try
        {
            Span<float> err = errBuffer.AsSpan(0, count);
            for (int iter = 0; iter < niters; iter++)
            {
                if (!GetSubset(kernel, m1, m2, ms1, ms2, idx, ref rng, 1000))
                {
                    if (iter == 0)
                    {
                        return 0;
                    }

                    break;
                }

                int nmodels = kernel.RunKernel(ms1, ms2, current);
                for (int i = 0; i < nmodels; i++)
                {
                    ReadOnlySpan<double> modelI = current.Slice(i * 9, 9);
                    kernel.ComputeError(m1, m2, modelI, err);
                    double median = SelectNth(MemoryMarshal.Cast<float, int>(err), count / 2);
                    if (median < minMedian)
                    {
                        minMedian = median;
                        modelI.CopyTo(best);
                    }
                }
            }

            if (!(minMedian < double.MaxValue))
            {
                return 0;
            }

            double sigma = 2.5 * 1.4826 * (1 + (5.0 / (count - modelPoints))) * Math.Sqrt(minMedian);
            sigma = Math.Max(sigma, 0.001);
            float t = (float)(sigma * sigma);
            kernel.ComputeError(m1, m2, best, err);
            int inliers = 0;
            for (int k = 0; k < count; k++)
            {
                int f = err[k] <= t ? 1 : 0;
                mask[k] = (byte)f;
                inliers += f;
            }

            best.CopyTo(models);
            success = inliers >= modelPoints;
            return 1;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(errBuffer);
        }
    }

    private static void CheckConfidence(double confidence)
    {
        if (!(confidence > 0 && confidence < 1))
        {
            throw new ArgumentOutOfRangeException(nameof(confidence), "The confidence must be in (0, 1).");
        }
    }

    /// <summary>OpenCV <c>getSubset</c> without partial subset checks.</summary>
    private static bool GetSubset<TPoint, TKernel>(
        in TKernel kernel,
        ReadOnlySpan<TPoint> m1,
        ReadOnlySpan<TPoint> m2,
        Span<TPoint> ms1,
        Span<TPoint> ms2,
        Span<int> idx,
        ref CvRng rng,
        int maxAttempts)
        where TPoint : unmanaged
        where TKernel : struct, IMultiModelKernel<TPoint>
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

    /// <summary>
    /// The <paramref name="k"/>-th smallest element of <c>std::nth_element</c> over the float bit
    /// patterns read as <c>int</c>.
    /// </summary>
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
