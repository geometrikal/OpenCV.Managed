// This implementation is ported from the OpenCV comparison functions.
//
// Original sources:
//   modules/core/src/arithm.cpp (compare, inRange)
//   modules/core/src/arithm.simd.hpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Core;

using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Managed ports of OpenCV's <c>compare</c> and <c>inRange</c>. Results are <c>byte</c>
/// matrices with 255 where the condition holds and 0 elsewhere. Comparisons with NaN are false,
/// except <see cref="CmpType.Ne"/>, which is true.
/// </summary>
public static class Comparison
{
    /// <summary>
    /// Per-element comparison of two matrices, as OpenCV <c>compare(a, b, op)</c>. The result has
    /// the channel count of the operands.
    /// </summary>
    public static Mat<byte> Compare<T>(Mat<T> a, Mat<T> b, CmpType op)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(a);
        Saturate.EnsureSupported<T>();
        Elementwise.ValidateSameShape(a, b, nameof(b));
        ValidateOp(op);
        var dst = new Mat<byte>(a.Width, a.Height, a.Channels, clear: false);
        Elementwise.BinaryTo(a, b, null, dst, new CompareOp<T>(op));
        return dst;
    }

    /// <summary>
    /// Compares every element with a value, as OpenCV <c>compare(a, Scalar, op)</c>. OpenCV uses
    /// only the first scalar component for all channels; this overload takes that value.
    /// </summary>
    /// <remarks>
    /// For integer element types: a value below the type range makes <c>Gt</c>, <c>Ge</c> and
    /// <c>Ne</c> true everywhere; a value above the range makes <c>Lt</c>, <c>Le</c> and
    /// <c>Ne</c> true everywhere. A fractional value is rounded up for <c>Lt</c> and <c>Ge</c>
    /// and down for <c>Le</c> and <c>Gt</c>; <c>Eq</c> is then false and <c>Ne</c> true.
    /// For <c>float</c> sources the value converts to float first.
    /// </remarks>
    public static Mat<byte> Compare<T>(Mat<T> a, double value, CmpType op)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(a);
        Saturate.EnsureSupported<T>();
        ValidateOp(op);
        var dst = new Mat<byte>(a.Width, a.Height, a.Channels, clear: false);
        Elementwise.Unary(a, dst, null, CompareScalarOp<T>.Create(value, op));
        return dst;
    }

    /// <summary>
    /// Checks that every channel of each pixel lies in <c>[lower, upper]</c> element-wise, as
    /// OpenCV <c>inRange</c> with array bounds. The result has one channel.
    /// </summary>
    public static Mat<byte> InRange<T>(Mat<T> source, Mat<T> lower, Mat<T> upper)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        Saturate.EnsureSupported<T>();
        Elementwise.ValidateSameShape(source, lower, nameof(lower));
        Elementwise.ValidateSameShape(source, upper, nameof(upper));
        var dst = new Mat<byte>(source.Width, source.Height, 1, clear: false);
        ParallelRows.For(
            source.Height,
            Elementwise.WorkPerRow<T>(source.RowLength),
            (y0, y1) => InRangeArrayRows(source, lower, upper, dst, y0, y1));
        return dst;
    }

    /// <summary>
    /// Checks that channel <c>c</c> of each pixel lies in <c>[lower[c], upper[c]]</c>, as OpenCV
    /// <c>inRange</c> with scalar bounds. The result has one channel.
    /// </summary>
    /// <remarks>
    /// For integer element types the bounds round half to even. For 8- and 16-bit types an
    /// interval with <c>lower &gt; upper</c> or outside the type range selects nothing.
    /// </remarks>
    public static Mat<byte> InRange<T>(Mat<T> source, Scalar lower, Scalar upper)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        Saturate.EnsureSupported<T>();
        if (source.Channels > 4)
        {
            throw new ArgumentException("Scalar bounds support at most 4 channels.", nameof(source));
        }

        InRangeKernel.ConvertBounds<T>(lower, upper, source.Channels, out double[] lo, out double[] hi);
        var dst = new Mat<byte>(source.Width, source.Height, 1, clear: false);
        if (typeof(T) == typeof(byte))
        {
            var loRow = new byte[source.RowLength];
            var hiRow = new byte[source.RowLength];
            for (int i = 0; i < loRow.Length; i++)
            {
                loRow[i] = (byte)lo[i % lo.Length];
                hiRow[i] = (byte)hi[i % hi.Length];
            }

            var bytes = (Mat<byte>)(object)source;
            ParallelRows.For(
                source.Height,
                Elementwise.WorkPerRow<byte>(source.RowLength),
                (y0, y1) => InRangeByteRows(bytes, loRow, hiRow, dst, y0, y1));
            return dst;
        }

        ParallelRows.For(
            source.Height,
            source.RowLength,
            (y0, y1) => InRangeScalarRows(source, lo, hi, dst, y0, y1));
        return dst;
    }

    private static void InRangeArrayRows<T>(Mat<T> source, Mat<T> lower, Mat<T> upper, Mat<byte> dst, int y0, int y1)
        where T : unmanaged
    {
        int cn = source.Channels;
        byte[]? tmp = typeof(T) == typeof(byte) ? new byte[source.RowLength] : null;
        for (int y = y0; y < y1; y++)
        {
            if (tmp is not null)
            {
                InRangeKernel.RowBytes(
                    MemoryMarshal.Cast<T, byte>(source.GetReadOnlyRowSpan(y)), MemoryMarshal.Cast<T, byte>(lower.GetReadOnlyRowSpan(y)),
                    MemoryMarshal.Cast<T, byte>(upper.GetReadOnlyRowSpan(y)), tmp, dst.GetRowSpan(y), cn);
                continue;
            }

            InRangeKernel.Row(
                source.GetReadOnlyRowSpan(y), lower.GetReadOnlyRowSpan(y), upper.GetReadOnlyRowSpan(y), dst.GetRowSpan(y), cn);
        }
    }

    private static void InRangeByteRows(Mat<byte> source, byte[] lo, byte[] hi, Mat<byte> dst, int y0, int y1)
    {
        var tmp = new byte[source.RowLength];
        for (int y = y0; y < y1; y++)
        {
            InRangeKernel.RowBytes(source.GetReadOnlyRowSpan(y), lo, hi, tmp, dst.GetRowSpan(y), source.Channels);
        }
    }

    private static void InRangeScalarRows<T>(Mat<T> source, double[] lo, double[] hi, Mat<byte> dst, int y0, int y1)
        where T : unmanaged
    {
        for (int y = y0; y < y1; y++)
        {
            InRangeKernel.Row(source.GetReadOnlyRowSpan(y), lo, hi, dst.GetRowSpan(y));
        }
    }

    private static void ValidateOp(CmpType op)
    {
        if (op < CmpType.Eq || op > CmpType.Ne)
        {
            throw new ArgumentOutOfRangeException(nameof(op));
        }
    }
}
