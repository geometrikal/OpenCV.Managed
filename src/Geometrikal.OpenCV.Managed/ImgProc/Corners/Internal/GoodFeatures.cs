// This implementation is ported from the OpenCV good-features-to-track detector.
//
// Original sources:
//   modules/features/src/featureselect.cpp (goodFeaturesToTrack, greaterThanPtr)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>OpenCV <c>goodFeaturesToTrack</c> without OpenCL.</summary>
internal static class GoodFeatures
{
    public static (Point2f[] Corners, float[] Quality) Find<T>(
        Mat<T> image,
        int maxCorners,
        double qualityLevel,
        double minDistance,
        Mat<byte>? mask,
        int blockSize,
        int gradientSize,
        bool useHarris,
        double k)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(image);
        if (!(qualityLevel > 0) || !(minDistance >= 0) || maxCorners < 0)
        {
            throw new ArgumentOutOfRangeException(nameof(qualityLevel), "qualityLevel must be positive, minDistance and maxCorners non-negative.");
        }

        if (mask is not null && (mask.Channels != 1 || mask.Width != image.Width || mask.Height != image.Height))
        {
            throw new ArgumentException("The mask must be single-channel with the image size.", nameof(mask));
        }

        int w = image.Width, h = image.Height;
        using PooledMat<float> eigPool = PooledMat<float>.Rent(w, h, 1);
        using PooledMat<float> tmpPool = PooledMat<float>.Rent(w, h, 1);
        Mat<float> eig = eigPool.Mat, tmp = tmpPool.Mat;
        Corners.CornerResponseInto(image, blockSize, gradientSize, useHarris, k, eig);

        double maxVal = Reductions.MinMaxLoc(eig, mask).MaxVal;
        Thresholding.Threshold(eig, eig, maxVal * qualityLevel, 0, ThresholdType.ToZero);
        Morphology.DilateInto(eig, tmp);

        ulong[] keys = ArrayPool<ulong>.Shared.Rent(Math.Max(w - 2, 0) * Math.Max(h - 2, 0));
        try
        {
            int count = 0;
            for (int y = 1; y < h - 1; y++)
            {
                ReadOnlySpan<float> e = eig.GetReadOnlyRowSpan(y);
                ReadOnlySpan<float> t = tmp.GetReadOnlyRowSpan(y);
                ReadOnlySpan<byte> m = mask is null ? default : mask.GetReadOnlyRowSpan(y);
                for (int x = 1; x < w - 1; x++)
                {
                    float val = e[x];
                    if (val != 0 && val == t[x] && (mask is null || m[x] != 0))
                    {
                        keys[count++] = SortKey(val, (y * w) + x);
                    }
                }
            }

            if (count == 0)
            {
                return ([], []);
            }

            Array.Sort(keys, 0, count);
            return minDistance >= 1
                ? SelectSpaced(keys.AsSpan(0, count), eig, maxCorners, minDistance)
                : SelectAll(keys.AsSpan(0, count), eig, maxCorners);
        }
        finally
        {
            ArrayPool<ulong>.Shared.Return(keys);
        }
    }

    private static (Point2f[] Corners, float[] Quality) SelectAll(ReadOnlySpan<ulong> sorted, Mat<float> eig, int maxCorners)
    {
        int w = eig.Width;
        int n = maxCorners > 0 ? Math.Min(maxCorners, sorted.Length) : sorted.Length;
        var corners = new Point2f[n];
        var quality = new float[n];
        for (int i = 0; i < n; i++)
        {
            int index = (int)~(uint)sorted[i];
            int y = index / w;
            int x = index - (y * w);
            quality[i] = eig.At(y, x);
            corners[i] = new Point2f(x, y);
        }

        return (corners, quality);
    }

    /// <summary>
    /// Keeps a candidate when no kept corner in the 3×3 neighbouring grid cells is closer than
    /// <paramref name="minDistance"/>. Each cell is a linked list of kept corners.
    /// </summary>
    private static (Point2f[] Corners, float[] Quality) SelectSpaced(ReadOnlySpan<ulong> sorted, Mat<float> eig, int maxCorners, double minDistance)
    {
        int w = eig.Width, h = eig.Height;
        int cap = maxCorners > 0 ? Math.Min(maxCorners, sorted.Length) : sorted.Length;
        int cellSize = ScalarConverter.CvRound(minDistance);
        int gridWidth = (w + cellSize - 1) / cellSize;
        int gridHeight = (h + cellSize - 1) / cellSize;
        double minDist2 = minDistance * minDistance;

        int[] head = ArrayPool<int>.Shared.Rent(gridWidth * gridHeight);
        int[] next = ArrayPool<int>.Shared.Rent(cap);
        Point2f[] kept = ArrayPool<Point2f>.Shared.Rent(cap);
        float[] keptQuality = ArrayPool<float>.Shared.Rent(cap);
        try
        {
            head.AsSpan(0, gridWidth * gridHeight).Fill(-1);
            int count = 0;
            foreach (ulong key in sorted)
            {
                int index = (int)~(uint)key;
                int y = index / w;
                int x = index - (y * w);
                int xCell = x / cellSize;
                int yCell = y / cellSize;
                int x1 = Math.Max(0, xCell - 1);
                int y1 = Math.Max(0, yCell - 1);
                int x2 = Math.Min(gridWidth - 1, xCell + 1);
                int y2 = Math.Min(gridHeight - 1, yCell + 1);

                bool good = true;
                for (int yy = y1; yy <= y2 && good; yy++)
                {
                    for (int xx = x1; xx <= x2 && good; xx++)
                    {
                        for (int c = head[(yy * gridWidth) + xx]; c >= 0; c = next[c])
                        {
                            Point2f p = kept[c];
                            float dx = x - p.X;
                            float dy = y - p.Y;
                            if ((dx * dx) + (dy * dy) < minDist2)
                            {
                                good = false;
                                break;
                            }
                        }
                    }
                }

                if (good)
                {
                    int cell = (yCell * gridWidth) + xCell;
                    kept[count] = new Point2f(x, y);
                    keptQuality[count] = eig.At(y, x);
                    next[count] = head[cell];
                    head[cell] = count;
                    count++;
                    if (count == cap)
                    {
                        break;
                    }
                }
            }

            return (kept.AsSpan(0, count).ToArray(), keptQuality.AsSpan(0, count).ToArray());
        }
        finally
        {
            ArrayPool<float>.Shared.Return(keptQuality);
            ArrayPool<Point2f>.Shared.Return(kept);
            ArrayPool<int>.Shared.Return(next);
            ArrayPool<int>.Shared.Return(head);
        }
    }

    /// <summary>
    /// Ascending order of the key is the OpenCV <c>greaterThanPtr</c> order: larger value first,
    /// and for equal values the larger address (row-major index) first.
    /// </summary>
    private static ulong SortKey(float value, int index)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        uint ascending = (bits & 0x80000000u) != 0 ? ~bits : bits | 0x80000000u;
        return ((ulong)~ascending << 32) | ~(uint)index;
    }
}
