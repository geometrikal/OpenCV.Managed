// This implementation is ported from the OpenCV automatic threshold functions.
//
// Original source:
//   modules/imgproc/src/thresh.cpp (getThreshVal_Otsu, getThreshVal_Otsu_8u,
//     getThreshVal_Otsu_16u, getThreshVal_Triangle_8u)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>Otsu and triangle threshold selection on single-channel histograms.</summary>
internal static class AutoThreshold
{
    private const double FltEpsilon = 1.1920928955078125e-07;

    /// <summary>
    /// Otsu's threshold for a single-channel <c>byte</c> or <c>ushort</c> matrix, optionally
    /// restricted to the pixels where <paramref name="mask"/> is nonzero.
    /// </summary>
    public static double Otsu<T>(Mat<T> src, Mat<byte>? mask)
        where T : unmanaged
    {
        int n = typeof(T) == typeof(byte) ? 256 : 65536;
        int[] buffer = ArrayPool<int>.Shared.Rent(n);
        try
        {
            Span<int> h = buffer.AsSpan(0, n);
            h.Clear();
            int count = Histogram(src, mask, h);
            double scale = 1.0 / (mask is null ? src.Width * src.Height : count);

            double mu = 0;
            for (int i = 0; i < n; i++)
            {
                mu += i * (double)h[i];
            }

            mu *= scale;
            double mu1 = 0, q1 = 0;
            double maxSigma = 0, maxVal = 0;
            for (int i = 0; i < n; i++)
            {
                double pi = h[i] * scale;
                mu1 *= q1;
                q1 += pi;
                double q2 = 1.0 - q1;

                if (Math.Min(q1, q2) < FltEpsilon || Math.Max(q1, q2) > 1.0 - FltEpsilon)
                {
                    continue;
                }

                mu1 = (mu1 + i * pi) / q1;
                double mu2 = (mu - q1 * mu1) / q2;
                double sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
                if (sigma > maxSigma)
                {
                    maxSigma = sigma;
                    maxVal = i;
                }
            }

            return maxVal;
        }
        finally
        {
            ArrayPool<int>.Shared.Return(buffer);
        }
    }

    /// <summary>Triangle threshold for a single-channel <c>byte</c> matrix.</summary>
    public static double Triangle(Mat<byte> src, Mat<byte>? mask)
    {
        const int n = 256;
        Span<int> h = stackalloc int[n];
        h.Clear();
        Histogram(src, mask, h);

        int leftBound = 0, rightBound = 0, maxInd = 0, max = 0;
        for (int i = 0; i < n; i++)
        {
            if (h[i] > 0)
            {
                leftBound = i;
                break;
            }
        }

        if (leftBound > 0)
        {
            leftBound--;
        }

        for (int i = n - 1; i > 0; i--)
        {
            if (h[i] > 0)
            {
                rightBound = i;
                break;
            }
        }

        if (rightBound < n - 1)
        {
            rightBound++;
        }

        for (int i = 0; i < n; i++)
        {
            if (h[i] > max)
            {
                max = h[i];
                maxInd = i;
            }
        }

        bool isFlipped = false;
        if (maxInd - leftBound < rightBound - maxInd)
        {
            isFlipped = true;
            h.Reverse();
            leftBound = n - 1 - rightBound;
            maxInd = n - 1 - maxInd;
        }

        double thresh = leftBound;
        double a = max, b = leftBound - maxInd, dist = 0;
        for (int i = leftBound + 1; i <= maxInd; i++)
        {
            double tempDist = a * i + b * h[i];
            if (tempDist > dist)
            {
                dist = tempDist;
                thresh = i;
            }
        }

        thresh--;
        if (isFlipped)
        {
            thresh = n - 1 - thresh;
        }

        return thresh;
    }

    /// <summary>Adds the selected pixels to <paramref name="h"/> and returns their number.</summary>
    private static int Histogram<T>(Mat<T> src, Mat<byte>? mask, Span<int> h)
        where T : unmanaged
    {
        int count = 0;
        for (int y = 0; y < src.Height; y++)
        {
            ReadOnlySpan<byte> m = mask is null ? default : mask.GetReadOnlyRowSpan(y);
            if (typeof(T) == typeof(byte))
            {
                count += Accumulate(MemoryMarshal.Cast<T, byte>(src.GetReadOnlyRowSpan(y)), m, h, mask is not null);
            }
            else
            {
                count += Accumulate(MemoryMarshal.Cast<T, ushort>(src.GetReadOnlyRowSpan(y)), m, h, mask is not null);
            }
        }

        return count;
    }

    private static int Accumulate(ReadOnlySpan<byte> row, ReadOnlySpan<byte> mask, Span<int> h, bool useMask)
    {
        if (!useMask)
        {
            foreach (byte v in row)
            {
                h[v]++;
            }

            return row.Length;
        }

        int count = 0;
        for (int x = 0; x < row.Length; x++)
        {
            if (mask[x] != 0)
            {
                h[row[x]]++;
                count++;
            }
        }

        return count;
    }

    private static int Accumulate(ReadOnlySpan<ushort> row, ReadOnlySpan<byte> mask, Span<int> h, bool useMask)
    {
        if (!useMask)
        {
            foreach (ushort v in row)
            {
                h[v]++;
            }

            return row.Length;
        }

        int count = 0;
        for (int x = 0; x < row.Length; x++)
        {
            if (mask[x] != 0)
            {
                h[row[x]]++;
                count++;
            }
        }

        return count;
    }
}
