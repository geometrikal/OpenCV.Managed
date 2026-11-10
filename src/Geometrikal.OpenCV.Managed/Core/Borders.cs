// This implementation is ported from the OpenCV border handling.
//
// Original sources:
//   modules/core/src/copy.cpp (borderInterpolate, copyMakeBorder)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core;

using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Border extrapolation helpers: coordinate mapping
/// (<see cref="BorderInterpolate"/>) and padded-image construction
/// (<see cref="CopyMakeBorder{T}"/>), mirroring OpenCV's
/// <c>borderInterpolate</c> and <c>copyMakeBorder</c>.
/// </summary>
public static class Borders
{
    /// <summary>
    /// Maps a possibly out-of-range coordinate <paramref name="p"/> to a valid
    /// coordinate inside <c>[0, len)</c> according to the border type, exactly
    /// as OpenCV's <c>borderInterpolate</c>. For
    /// <see cref="BorderType.Constant"/> an out-of-range coordinate maps to
    /// <c>-1</c> (the caller substitutes the constant value).
    /// </summary>
    /// <param name="p">Coordinate to map (may be negative or ≥ <paramref name="len"/>).</param>
    /// <param name="len">Extent of the axis; must be positive.</param>
    /// <param name="borderType">Extrapolation rule.</param>
    /// <returns>The mapped coordinate, or -1 for out-of-range constant-border lookups.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="len"/> is not positive or the border type is unknown.</exception>
    public static int BorderInterpolate(int p, int len, BorderType borderType)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(len);

        if ((uint)p < (uint)len)
        {
            return p;
        }

        switch (borderType)
        {
            case BorderType.Replicate:
                return p < 0 ? 0 : len - 1;

            case BorderType.Reflect:
            case BorderType.Reflect101:
            {
                int delta = borderType == BorderType.Reflect101 ? 1 : 0;
                if (len == 1)
                {
                    return 0;
                }

                do
                {
                    if (p < 0)
                    {
                        p = -p - 1 + delta;
                    }
                    else
                    {
                        p = len - 1 - (p - len) - delta;
                    }
                }
                while ((uint)p >= (uint)len);

                return p;
            }

            case BorderType.Wrap:
                if (p < 0)
                {
                    p -= (p - len + 1) / len * len;
                }

                if (p >= len)
                {
                    p %= len;
                }

                return p;

            case BorderType.Constant:
                return -1;

            default:
                throw new ArgumentOutOfRangeException(
                    nameof(borderType), borderType, "Unknown/unsupported border type.");
        }
    }

    /// <summary>
    /// Returns a new matrix containing <paramref name="source"/> surrounded by
    /// the requested border, mirroring OpenCV's <c>copyMakeBorder</c>. The
    /// source is never modified; region-of-interest views are treated as
    /// stand-alone images (OpenCV's <c>BORDER_ISOLATED</c> behaviour).
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="source">Source matrix.</param>
    /// <param name="top">Rows to add above; non-negative.</param>
    /// <param name="bottom">Rows to add below; non-negative.</param>
    /// <param name="left">Pixels to add on the left; non-negative.</param>
    /// <param name="right">Pixels to add on the right; non-negative.</param>
    /// <param name="borderType">Extrapolation rule for the border pixels.</param>
    /// <param name="value">
    /// Constant border colour for <see cref="BorderType.Constant"/>; each
    /// channel is saturate-cast to <typeparamref name="T"/> with OpenCV's
    /// <c>scalarToRawData</c> semantics. Ignored for other border types.
    /// </param>
    /// <returns>A new packed matrix of size (width + left + right) × (height + top + bottom).</returns>
    /// <exception cref="ArgumentNullException"><paramref name="source"/> is null.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A border extent is negative.</exception>
    /// <exception cref="NotSupportedException">The element type or channel count is unsupported.</exception>
    public static Mat<T> CopyMakeBorder<T>(
        Mat<T> source,
        int top,
        int bottom,
        int left,
        int right,
        BorderType borderType,
        Scalar value = default)
        where T : unmanaged
    {
        ValidateCopyMakeBorder(source, top, bottom, left, right, borderType);
        var dst = new Mat<T>(
            checked(source.Width + left + right),
            checked(source.Height + top + bottom),
            source.Channels,
            clear: false);
        CopyMakeBorderInto(source, dst, top, left, borderType, value);
        return dst;
    }

    /// <summary>
    /// Writes <paramref name="source"/> surrounded by the requested border into
    /// <paramref name="destination"/>, mirroring OpenCV's <c>copyMakeBorder</c> with a
    /// preallocated <c>dst</c>. The result equals the allocating overload.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="source">Source matrix; it is not modified unless it is the destination.</param>
    /// <param name="destination">
    /// Destination of width <c>source.Width + left + right</c>, height
    /// <c>source.Height + top + bottom</c> and the source channel count. It can share memory with
    /// <paramref name="source"/>; the source is then copied first.
    /// </param>
    /// <param name="top">Rows to add above; non-negative.</param>
    /// <param name="bottom">Rows to add below; non-negative.</param>
    /// <param name="left">Pixels to add on the left; non-negative.</param>
    /// <param name="right">Pixels to add on the right; non-negative.</param>
    /// <param name="borderType">Extrapolation rule for the border pixels.</param>
    /// <param name="value">Constant border colour for <see cref="BorderType.Constant"/>, as in the allocating overload.</param>
    /// <exception cref="ArgumentException">The destination size or channel count does not match.</exception>
    public static void CopyMakeBorder<T>(
        Mat<T> source,
        Mat<T> destination,
        int top,
        int bottom,
        int left,
        int right,
        BorderType borderType,
        Scalar value = default)
        where T : unmanaged
    {
        ValidateCopyMakeBorder(source, top, bottom, left, right, borderType);
        ArgumentNullException.ThrowIfNull(destination);
        if (destination.Width != checked(source.Width + left + right) ||
            destination.Height != checked(source.Height + top + bottom) ||
            destination.Channels != source.Channels)
        {
            throw new ArgumentException("The destination must have the bordered size and the source channel count.", nameof(destination));
        }

        if (ReferenceEquals(source.DangerousGetArray(), destination.DangerousGetArray()))
        {
            source = source.Clone();
        }

        CopyMakeBorderInto(source, destination, top, left, borderType, value);
    }

    /// <summary>
    /// <see cref="CopyMakeBorder{T}"/> into a pooled matrix. The caller disposes the result.
    /// </summary>
    internal static PooledMat<T> CopyMakeBorderPooled<T>(
        Mat<T> source,
        int top,
        int bottom,
        int left,
        int right,
        BorderType borderType,
        Scalar value = default)
        where T : unmanaged
    {
        ValidateCopyMakeBorder(source, top, bottom, left, right, borderType);
        var dst = PooledMat<T>.Rent(
            checked(source.Width + left + right),
            checked(source.Height + top + bottom),
            source.Channels);
        CopyMakeBorderInto(source, dst.Mat, top, left, borderType, value);
        return dst;
    }

    private static void ValidateCopyMakeBorder<T>(Mat<T> source, int top, int bottom, int left, int right, BorderType borderType)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentOutOfRangeException.ThrowIfNegative(top);
        ArgumentOutOfRangeException.ThrowIfNegative(bottom);
        ArgumentOutOfRangeException.ThrowIfNegative(left);
        ArgumentOutOfRangeException.ThrowIfNegative(right);
        if (source.Channels > 4)
        {
            throw new NotSupportedException("Scalar supports at most 4 channels.");
        }

        if (borderType == BorderType.Transparent)
        {
            throw new ArgumentOutOfRangeException(nameof(borderType), borderType, "BorderType.Transparent is not supported by copyMakeBorder.");
        }
    }

    private static void CopyMakeBorderInto<T>(
        Mat<T> source,
        Mat<T> dst,
        int top,
        int left,
        BorderType borderType,
        Scalar value)
        where T : unmanaged
    {
        int channels = source.Channels;
        int srcW = source.Width;
        int srcH = source.Height;
        int right = dst.Width - srcW - left;

        T[] constant = ScalarConverter.ToRawColor<T>(value, channels);

        // Source element of every left and right border element (-1 = constant fill), matching
        // the borderInterpolate-based row assembly OpenCV performs once per image.
        int leftCount = left * channels;
        int rightCount = right * channels;
        int[] map = ArrayPool<int>.Shared.Rent(leftCount + rightCount);
        try
        {
            for (int x = 0; x < left + right; x++)
            {
                int sx = BorderInterpolate(x < left ? x - left : srcW + x - left, srcW, borderType);
                for (int c = 0; c < channels; c++)
                {
                    map[(x * channels) + c] = sx < 0 ? -1 - c : (sx * channels) + c;
                }
            }

            if (typeof(T) == typeof(byte) &&
                BorderPermutation.TryCreate(map, leftCount, rightCount, srcW * channels, MemoryMarshal.Cast<T, byte>(constant.AsSpan()), out BorderPermutation permutation))
            {
                permutation.Apply((Mat<byte>)(object)source, (Mat<byte>)(object)dst, top, borderType);
                return;
            }

            T[]? constantRow = null;
            int interiorStart = leftCount;
            int rightStart = (left + srcW) * channels;
            ref int leftMap = ref MemoryMarshal.GetArrayDataReference(map);
            ref int rightMap = ref Unsafe.Add(ref leftMap, leftCount);
            ref T fill = ref constant[0];
            for (int y = 0; y < dst.Height; y++)
            {
                Span<T> dstRow = dst.GetRowSpan(y);
                int srcY = BorderInterpolate(y - top, srcH, borderType);

                if (srcY < 0)
                {
                    constantRow ??= CreateConstantRow(constant, dst.Width);
                    constantRow.CopyTo(dstRow);
                    continue;
                }

                ReadOnlySpan<T> srcRow = source.GetReadOnlyRowSpan(srcY);
                srcRow.CopyTo(dstRow.Slice(interiorStart));
                ref T s = ref MemoryMarshal.GetReference(srcRow);
                ref T d = ref MemoryMarshal.GetReference(dstRow);
                FillBorder(ref s, ref d, ref leftMap, ref fill, leftCount);
                FillBorder(ref s, ref Unsafe.Add(ref d, rightStart), ref rightMap, ref fill, rightCount);
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(map);
        }
    }

    private static void FillBorder<T>(ref T src, ref T dst, ref int map, ref T constant, int count)
        where T : unmanaged
    {
        for (int i = 0; i < count; i++)
        {
            int e = Unsafe.Add(ref map, i);
            Unsafe.Add(ref dst, i) = e >= 0 ? Unsafe.Add(ref src, e) : Unsafe.Add(ref constant, -1 - e);
        }
    }

    private static T[] CreateConstantRow<T>(T[] constant, int width)
        where T : unmanaged
    {
        int channels = constant.Length;
        var row = new T[width * channels];
        for (int x = 0; x < width; x++)
        {
            constant.CopyTo(row, x * channels);
        }

        return row;
    }
}
