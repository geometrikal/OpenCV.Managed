// This implementation is ported from the OpenCV batch distance function.
//
// Original sources:
//   modules/core/src/batch_distance.cpp (batchDistance, BatchDistInvoker)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Features.Internal;

using System.Buffers;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// OpenCV <c>cv::batchDistance</c> for byte and float descriptor matrices. Distances are raw
/// 32-bit values (see <see cref="BatchDistanceKernels"/>). Positive float bits order as
/// <c>int</c>, so one integer comparison serves both output types, as in OpenCV.
/// </summary>
internal static class BatchDistance
{
    /// <summary>Selects the OpenCV distance function, or throws for an unsupported combination.</summary>
    public static DistanceKind SelectKind<T>(NormType normType)
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte))
        {
            return normType switch
            {
                NormType.Hamming => DistanceKind.Hamming,
                NormType.Hamming2 => DistanceKind.Hamming2,
                NormType.L1 => DistanceKind.L1Byte,
                NormType.L2Sqr => DistanceKind.L2SqrByte,
                NormType.L2 => DistanceKind.L2Byte,
                _ => throw new NotSupportedException($"Norm {normType} is not supported for byte descriptors."),
            };
        }

        if (typeof(T) == typeof(float))
        {
            return normType switch
            {
                NormType.L1 => DistanceKind.L1Float,
                NormType.L2Sqr => DistanceKind.L2SqrFloat,
                NormType.L2 => DistanceKind.L2Float,
                _ => throw new NotSupportedException($"Norm {normType} is not supported for float descriptors."),
            };
        }

        throw new NotSupportedException($"Descriptor element type {typeof(T).Name} is not supported; use byte or float.");
    }

    /// <summary>
    /// K-nearest update, as <c>batchDistance</c> with <c>K &gt; 0</c>. <paramref name="dist"/> and
    /// <paramref name="nidx"/> hold <c>query.Height × k</c> values and keep the state of earlier
    /// train sets; train row <c>j</c> is stored as index <c>j + update</c>.
    /// </summary>
    public static void Knn<T>(
        DistanceKind kind, Mat<T> query, Mat<T> train, Mat<byte>? mask, int k, int update, int[] dist, int[] nidx)
        where T : unmanaged
    {
        int rows = query.Height;
        ParallelRows.For(rows, WorkPerRow(train), (y0, y1) => KnnRows(kind, query, train, mask, k, update, dist, nidx, y0, y1));
    }

    /// <summary>
    /// All distances, as <c>batchDistance</c> with <c>K = 0</c>. <paramref name="dist"/> holds
    /// <c>query.Height × train.Height</c> values.
    /// </summary>
    public static void Full<T>(DistanceKind kind, Mat<T> query, Mat<T> train, Mat<byte>? mask, int[] dist)
        where T : unmanaged
    {
        ParallelRows.For(query.Height, WorkPerRow(train), (y0, y1) => FullRows(kind, query, train, mask, dist, y0, y1));
    }

    private static long WorkPerRow<T>(Mat<T> train)
        where T : unmanaged => (long)train.Height * train.Width;

    private static void KnnRows<T>(
        DistanceKind kind, Mat<T> query, Mat<T> train, Mat<byte>? mask, int k, int update, int[] dist, int[] nidx, int y0, int y1)
        where T : unmanaged
    {
        int n = train.Height;
        int[] buffer = ArrayPool<int>.Shared.Rent(n);
        try
        {
            Span<int> row = buffer.AsSpan(0, n);
            for (int i = y0; i < y1; i++)
            {
                ComputeRow(kind, query, train, mask, i, row);
                Span<int> d = dist.AsSpan(i * k, k);
                Span<int> idx = nidx.AsSpan(i * k, k);
                for (int j = 0; j < n; j++)
                {
                    int v = row[j];
                    if (v < d[k - 1])
                    {
                        int m = k - 2;
                        for (; m >= 0 && d[m] > v; m--)
                        {
                            idx[m + 1] = idx[m];
                            d[m + 1] = d[m];
                        }

                        idx[m + 1] = j + update;
                        d[m + 1] = v;
                    }
                }
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(buffer);
        }
    }

    private static void FullRows<T>(DistanceKind kind, Mat<T> query, Mat<T> train, Mat<byte>? mask, int[] dist, int y0, int y1)
        where T : unmanaged
    {
        int n = train.Height;
        for (int i = y0; i < y1; i++)
        {
            ComputeRow(kind, query, train, mask, i, dist.AsSpan(i * n, n));
        }
    }

    private static void ComputeRow<T>(DistanceKind kind, Mat<T> query, Mat<T> train, Mat<byte>? mask, int i, Span<int> row)
        where T : unmanaged
    {
        int n = train.Height;
        if (typeof(T) == typeof(byte))
        {
            var q = (Mat<byte>)(object)query;
            var t = (Mat<byte>)(object)train;
            ref byte t0 = ref t.DangerousGetArray()[t.DangerousGetOffset()];
            BatchDistanceKernels.ByteRow(kind, q.GetReadOnlyRowSpan(i), ref t0, t.Stride, n, row);
        }
        else
        {
            var q = (Mat<float>)(object)query;
            var t = (Mat<float>)(object)train;
            ref float t0 = ref t.DangerousGetArray()[t.DangerousGetOffset()];
            BatchDistanceKernels.FloatRow(kind, q.GetReadOnlyRowSpan(i), ref t0, t.Stride, n, row);
        }

        if (mask is not null)
        {
            ReadOnlySpan<byte> m = mask.GetReadOnlyRowSpan(i);
            int masked = BatchDistanceKernels.MaskedValue(kind);
            for (int j = 0; j < n; j++)
            {
                if (m[j] == 0)
                {
                    row[j] = masked;
                }
            }
        }
    }

    /// <summary>
    /// Cross-check of <c>batchDistance</c> with <c>K = 1</c> and <c>update = 0</c>: keeps a query
    /// match only when the query is also the nearest query of its train descriptor.
    /// </summary>
    public static void CrossCheck<T>(DistanceKind kind, Mat<T> query, Mat<T> train, int[] dist, int[] nidx)
        where T : unmanaged
    {
        int masked = BatchDistanceKernels.MaskedValue(kind);
        int[] tdist = new int[train.Height];
        int[] tidx = new int[train.Height];
        int[] sdist = new int[query.Height];
        int[] sidx = new int[query.Height];
        tdist.AsSpan().Fill(masked);
        tidx.AsSpan().Fill(-1);
        sdist.AsSpan().Fill(masked);
        sidx.AsSpan().Fill(-1);
        Knn(kind, train, query, null, 1, 0, tdist, tidx);
        Knn(kind, query, train, null, 1, 0, sdist, sidx);

        bool intOutput = BatchDistanceKernels.HasIntOutput(kind);
        for (int i = 0; i < tdist.Length; i++)
        {
            int idx = tidx[i];
            if (idx < 0)
            {
                continue;
            }

            bool closer = intOutput
                ? tdist[i] < dist[idx]
                : BitConverter.Int32BitsToSingle(tdist[i]) < BitConverter.Int32BitsToSingle(dist[idx]);
            if (closer)
            {
                dist[idx] = tdist[i];
                nidx[idx] = i;
            }
        }

        for (int i = 0; i < sdist.Length; i++)
        {
            if (sidx[i] < 0 || tidx[sidx[i]] != i)
            {
                nidx[i] = -1;
            }
        }
    }

    /// <summary>Converts a raw distance to float, as <c>Mat::convertTo(CV_32F)</c> for <c>int</c> output.</summary>
    public static float ToFloat(DistanceKind kind, int raw) =>
        BatchDistanceKernels.HasIntOutput(kind) ? raw : BitConverter.Int32BitsToSingle(raw);
}
