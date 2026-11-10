namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Buffers;
using System.Numerics;

/// <summary>Computes one row of a binary element-wise operation.</summary>
internal interface IBinaryRowOp<TSrc, TDst>
    where TSrc : unmanaged
    where TDst : unmanaged
{
    void Apply(ReadOnlySpan<TSrc> a, ReadOnlySpan<TSrc> b, Span<TDst> dst);
}

/// <summary>Computes one row of a unary element-wise operation.</summary>
internal interface IUnaryRowOp<TSrc, TDst>
    where TSrc : unmanaged
    where TDst : unmanaged
{
    void Apply(ReadOnlySpan<TSrc> a, Span<TDst> dst);
}

/// <summary>
/// Row drivers for element-wise operations: shape and mask validation, scalar broadcast,
/// masked stores and row-stripe parallelism. A destination may be the same matrix as a
/// source; partially overlapping views are not supported.
/// </summary>
internal static class Elementwise
{
    public static void ValidateSameShape<TA, TB>(Mat<TA> a, Mat<TB> b, string paramName)
        where TA : unmanaged
        where TB : unmanaged
    {
        ArgumentNullException.ThrowIfNull(b, paramName);
        if (a.Width != b.Width || a.Height != b.Height || a.Channels != b.Channels)
        {
            throw new ArgumentException("Matrix sizes and channel counts must match.", paramName);
        }
    }

    public static void ValidateMask<T>(Mat<T> source, Mat<byte>? mask, bool allowPerChannel = false)
        where T : unmanaged
    {
        if (mask is null)
        {
            return;
        }

        if (mask.Width != source.Width || mask.Height != source.Height)
        {
            throw new ArgumentException("The mask size must match the matrix size.", nameof(mask));
        }

        if (mask.Channels != 1 && !(allowPerChannel && mask.Channels == source.Channels))
        {
            throw new ArgumentException(
                allowPerChannel
                    ? "The mask must have 1 channel or the matrix channel count."
                    : "The mask must have 1 channel.",
                nameof(mask));
        }
    }

    /// <summary>Repeats per-channel values over one row.</summary>
    public static T[] UnrollScalar<T>(T[] perChannel, int rowLength)
        where T : unmanaged
    {
        var row = new T[rowLength];
        int cn = perChannel.Length;
        for (int i = 0; i < rowLength; i++)
        {
            row[i] = perChannel[i % cn];
        }

        return row;
    }

    public static long WorkPerRow<T>(int rowLength)
        where T : unmanaged
        => Math.Max(1, rowLength / Math.Max(1, Vector<T>.Count));

    /// <summary>
    /// <c>dst = op(a, b)</c> where <paramref name="bRow"/> replaces <paramref name="b"/> when it
    /// is not null. Where <paramref name="mask"/> is zero, <paramref name="dst"/> keeps its values.
    /// </summary>
    public static void Binary<T, TOp>(Mat<T> a, Mat<T>? b, T[]? bRow, Mat<T> dst, Mat<byte>? mask, TOp op)
        where T : unmanaged
        where TOp : struct, IBinaryRowOp<T, T>
    {
        ParallelRows.For(
            a.Height,
            WorkPerRow<T>(a.RowLength),
            (y0, y1) => BinaryRows(a, b, bRow, dst, mask, op, y0, y1));
    }

    private static void BinaryRows<T, TOp>(
        Mat<T> a, Mat<T>? b, T[]? bRow, Mat<T> dst, Mat<byte>? mask, TOp op, int y0, int y1)
        where T : unmanaged
        where TOp : struct, IBinaryRowOp<T, T>
    {
        if (mask is null)
        {
            for (int y = y0; y < y1; y++)
            {
                op.Apply(a.GetReadOnlyRowSpan(y), bRow ?? b!.GetReadOnlyRowSpan(y), dst.GetRowSpan(y));
            }

            return;
        }

        T[] tmp = ArrayPool<T>.Shared.Rent(a.RowLength);
        try
        {
            Span<T> t = tmp.AsSpan(0, a.RowLength);
            for (int y = y0; y < y1; y++)
            {
                op.Apply(a.GetReadOnlyRowSpan(y), bRow ?? b!.GetReadOnlyRowSpan(y), t);
                StoreMasked(t, mask.GetReadOnlyRowSpan(y), dst.GetRowSpan(y), a.Channels);
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(tmp);
        }
    }

    /// <summary><c>dst = op(a)</c>; see <see cref="Binary{T, TOp}"/> for the mask rule.</summary>
    public static void Unary<TSrc, TDst, TOp>(Mat<TSrc> a, Mat<TDst> dst, Mat<byte>? mask, TOp op)
        where TSrc : unmanaged
        where TDst : unmanaged
        where TOp : struct, IUnaryRowOp<TSrc, TDst>
    {
        ParallelRows.For(
            a.Height,
            WorkPerRow<TSrc>(a.RowLength),
            (y0, y1) => UnaryRows(a, dst, mask, op, y0, y1));
    }

    private static void UnaryRows<TSrc, TDst, TOp>(
        Mat<TSrc> a, Mat<TDst> dst, Mat<byte>? mask, TOp op, int y0, int y1)
        where TSrc : unmanaged
        where TDst : unmanaged
        where TOp : struct, IUnaryRowOp<TSrc, TDst>
    {
        if (mask is null)
        {
            for (int y = y0; y < y1; y++)
            {
                op.Apply(a.GetReadOnlyRowSpan(y), dst.GetRowSpan(y));
            }

            return;
        }

        int len = dst.RowLength;
        TDst[] tmp = ArrayPool<TDst>.Shared.Rent(len);
        try
        {
            Span<TDst> t = tmp.AsSpan(0, len);
            for (int y = y0; y < y1; y++)
            {
                op.Apply(a.GetReadOnlyRowSpan(y), t);
                StoreMasked(t, mask.GetReadOnlyRowSpan(y), dst.GetRowSpan(y), dst.Channels);
            }
        }
        finally
        {
            ArrayPool<TDst>.Shared.Return(tmp);
        }
    }

    /// <summary>
    /// Binary operation with a different destination type and no mask (for example compare).
    /// </summary>
    public static void BinaryTo<TSrc, TDst, TOp>(Mat<TSrc> a, Mat<TSrc>? b, TSrc[]? bRow, Mat<TDst> dst, TOp op)
        where TSrc : unmanaged
        where TDst : unmanaged
        where TOp : struct, IBinaryRowOp<TSrc, TDst>
    {
        ParallelRows.For(
            a.Height,
            WorkPerRow<TSrc>(a.RowLength),
            (y0, y1) => BinaryToRows(a, b, bRow, dst, op, y0, y1));
    }

    private static void BinaryToRows<TSrc, TDst, TOp>(
        Mat<TSrc> a, Mat<TSrc>? b, TSrc[]? bRow, Mat<TDst> dst, TOp op, int y0, int y1)
        where TSrc : unmanaged
        where TDst : unmanaged
        where TOp : struct, IBinaryRowOp<TSrc, TDst>
    {
        for (int y = y0; y < y1; y++)
        {
            op.Apply(a.GetReadOnlyRowSpan(y), bRow ?? b!.GetReadOnlyRowSpan(y), dst.GetRowSpan(y));
        }
    }

    /// <summary>
    /// Copies <paramref name="src"/> to <paramref name="dst"/> where the mask is nonzero. A mask
    /// row with one element per pixel selects whole pixels; a mask row with one element per
    /// channel selects single elements.
    /// </summary>
    public static void StoreMasked<T>(ReadOnlySpan<T> src, ReadOnlySpan<byte> mask, Span<T> dst, int cn)
        where T : unmanaged
    {
        if (mask.Length == src.Length)
        {
            for (int i = 0; i < src.Length; i++)
            {
                if (mask[i] != 0)
                {
                    dst[i] = src[i];
                }
            }

            return;
        }

        for (int x = 0, i = 0; x < mask.Length; x++, i += cn)
        {
            if (mask[x] != 0)
            {
                src.Slice(i, cn).CopyTo(dst.Slice(i, cn));
            }
        }
    }
}
