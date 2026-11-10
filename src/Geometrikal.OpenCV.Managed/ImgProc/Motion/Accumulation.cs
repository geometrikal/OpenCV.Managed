// This implementation is ported from the OpenCV running-average functions.
//
// Original source:
//   modules/imgproc/src/accum.cpp (accumulate, accumulateSquare, accumulateProduct,
//   accumulateWeighted) and accum.simd.hpp
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;

/// <summary>
/// Running sums and averages of images, as OpenCV <c>accumulate</c>, <c>accumulateSquare</c>,
/// <c>accumulateProduct</c> and <c>accumulateWeighted</c>.
/// </summary>
/// <remarks>
/// The source is <c>byte</c>, <c>ushort</c>, <c>float</c> or <c>double</c>; the accumulator is
/// <c>float</c> or <c>double</c> (not <c>float</c> for a <c>double</c> source). The accumulator is
/// modified in place. A mask (one channel, the source size) limits the update to its non-zero
/// pixels. The source elements convert to the accumulator type exactly; products and sums round
/// once each, in the accumulator type.
/// </remarks>
public static class Accumulation
{
    /// <summary>Adds the source to the accumulator: <c>dst += src</c>.</summary>
    /// <param name="source">Source image.</param>
    /// <param name="accumulator">Accumulator of the source size and channel count.</param>
    /// <param name="mask">Optional mask.</param>
    public static void Accumulate<TSrc, TDst>(Mat<TSrc> source, Mat<TDst> accumulator, Mat<byte>? mask = null)
        where TSrc : unmanaged
        where TDst : unmanaged, IFloatingPointIeee754<TDst>
    {
        Validate(source, accumulator, mask);
        Run(source, source, accumulator, mask, Op.Add, default, default);
    }

    /// <summary>Adds the square of the source to the accumulator: <c>dst += src·src</c>.</summary>
    /// <inheritdoc cref="Accumulate{TSrc, TDst}"/>
    public static void AccumulateSquare<TSrc, TDst>(Mat<TSrc> source, Mat<TDst> accumulator, Mat<byte>? mask = null)
        where TSrc : unmanaged
        where TDst : unmanaged, IFloatingPointIeee754<TDst>
    {
        Validate(source, accumulator, mask);
        Run(source, source, accumulator, mask, Op.Product, default, default);
    }

    /// <summary>Adds the per-element product of two images to the accumulator: <c>dst += src1·src2</c>.</summary>
    /// <param name="source1">First source image.</param>
    /// <param name="source2">Second source image of the type and size of <paramref name="source1"/>.</param>
    /// <param name="accumulator">Accumulator of the source size and channel count.</param>
    /// <param name="mask">Optional mask.</param>
    public static void AccumulateProduct<TSrc, TDst>(Mat<TSrc> source1, Mat<TSrc> source2, Mat<TDst> accumulator, Mat<byte>? mask = null)
        where TSrc : unmanaged
        where TDst : unmanaged, IFloatingPointIeee754<TDst>
    {
        ArgumentNullException.ThrowIfNull(source2);
        Validate(source1, accumulator, mask);
        if (source2.Width != source1.Width || source2.Height != source1.Height || source2.Channels != source1.Channels)
        {
            throw new ArgumentException("The sources must have the same size and channel count.", nameof(source2));
        }

        Run(source1, source2, accumulator, mask, Op.Product, default, default);
    }

    /// <summary>
    /// Updates the running average: <c>dst = src·alpha + dst·(1 − alpha)</c>.
    /// </summary>
    /// <param name="source">Source image.</param>
    /// <param name="accumulator">Accumulator of the source size and channel count.</param>
    /// <param name="alpha">Weight of the source.</param>
    /// <param name="mask">Optional mask.</param>
    /// <remarks>
    /// For a <c>float</c> accumulator OpenCV computes <c>1 − alpha</c> in two ways: as
    /// <c>(float)(1 − alpha)</c> in its vector blocks and as <c>1 − (float)alpha</c> in the scalar
    /// tail. The port uses each value for the same elements as the SSE build: whole blocks of 16
    /// (<c>byte</c>) or 8 (<c>ushort</c>, <c>float</c>) elements from the start of each plane (the
    /// whole image when all inputs are continuous, otherwise each row). With a mask, only
    /// <c>byte</c> and <c>ushort</c> sources with 1 or 3 channels have vector blocks (of pixels).
    /// </remarks>
    public static void AccumulateWeighted<TSrc, TDst>(Mat<TSrc> source, Mat<TDst> accumulator, double alpha, Mat<byte>? mask = null)
        where TSrc : unmanaged
        where TDst : unmanaged, IFloatingPointIeee754<TDst>
    {
        Validate(source, accumulator, mask);
        TDst a, bTail, bVector;
        if (typeof(TDst) == typeof(float))
        {
            float af = (float)alpha;
            a = Unsafe.As<float, TDst>(ref af);
            float tail = 1 - af, vector = (float)(1.0 - alpha);
            bTail = Unsafe.As<float, TDst>(ref tail);
            bVector = Unsafe.As<float, TDst>(ref vector);
        }
        else
        {
            a = TDst.CreateTruncating(alpha);
            bTail = bVector = TDst.One - a;
        }

        Run(source, source, accumulator, mask, Op.Weighted, a, bTail, bVector);
    }

    private enum Op
    {
        Add,
        Product,
        Weighted,
    }

    private static void Validate<TSrc, TDst>(Mat<TSrc> source, Mat<TDst> accumulator, Mat<byte>? mask)
        where TSrc : unmanaged
        where TDst : unmanaged
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(accumulator);
        if (typeof(TSrc) != typeof(byte) && typeof(TSrc) != typeof(ushort) && typeof(TSrc) != typeof(float) && typeof(TSrc) != typeof(double))
        {
            throw new NotSupportedException($"Source type {typeof(TSrc).Name} is not supported.");
        }

        if (typeof(TDst) != typeof(float) && typeof(TDst) != typeof(double))
        {
            throw new NotSupportedException($"Accumulator type {typeof(TDst).Name} is not supported.");
        }

        if (typeof(TSrc) == typeof(double) && typeof(TDst) == typeof(float))
        {
            throw new NotSupportedException("A double source needs a double accumulator.");
        }

        if (accumulator.Width != source.Width || accumulator.Height != source.Height || accumulator.Channels != source.Channels)
        {
            throw new ArgumentException("The accumulator must have the source size and channel count.", nameof(accumulator));
        }

        if (mask is not null && (mask.Width != source.Width || mask.Height != source.Height || mask.Channels != 1))
        {
            throw new ArgumentException("The mask must be single-channel with the source size.", nameof(mask));
        }
    }

    private static void Run<TSrc, TDst>(Mat<TSrc> s1, Mat<TSrc> s2, Mat<TDst> dst, Mat<byte>? mask, Op op, TDst a, TDst b)
        where TSrc : unmanaged
        where TDst : unmanaged, IFloatingPointIeee754<TDst> =>
        Run(s1, s2, dst, mask, op, a, b, b);

    private static void Run<TSrc, TDst>(Mat<TSrc> s1, Mat<TSrc> s2, Mat<TDst> dst, Mat<byte>? mask, Op op, TDst a, TDst bTail, TDst bVector)
        where TSrc : unmanaged
        where TDst : unmanaged, IFloatingPointIeee754<TDst>
    {
        int cn = s1.Channels;
        int vectorBlock = WeightedVectorBlock<TSrc, TDst>(mask is not null, cn);

        // OpenCV NAryMatIterator: one plane when every input is continuous, otherwise one per row.
        if (s1.IsContinuous && s2.IsContinuous && dst.IsContinuous && (mask is null || mask.IsContinuous))
        {
            Plane(s1.GetReadOnlySpan(), s2.GetReadOnlySpan(), dst.GetSpan(), mask is null ? default : mask.GetReadOnlySpan(),
                mask is not null, cn, op, a, bTail, bVector, vectorBlock);
            return;
        }

        for (int y = 0; y < s1.Height; y++)
        {
            Plane(s1.GetReadOnlyRowSpan(y), s2.GetReadOnlyRowSpan(y), dst.GetRowSpan(y), mask is null ? default : mask.GetReadOnlyRowSpan(y),
                mask is not null, cn, op, a, bTail, bVector, vectorBlock);
        }
    }

    // Elements (no mask) or pixels (mask) per SSE block of the float weighted kernels; 0 = no vector path.
    private static int WeightedVectorBlock<TSrc, TDst>(bool masked, int cn)
    {
        if (typeof(TDst) != typeof(float))
        {
            return 0;
        }

        int block = typeof(TSrc) == typeof(byte) ? 16 : 8;
        if (!masked)
        {
            return block;
        }

        return typeof(TSrc) != typeof(float) && (cn == 1 || cn == 3) ? block : 0;
    }

    private static void Plane<TSrc, TDst>(
        ReadOnlySpan<TSrc> s1, ReadOnlySpan<TSrc> s2, Span<TDst> dst, ReadOnlySpan<byte> mask, bool masked, int cn,
        Op op, TDst a, TDst bTail, TDst bVector, int vectorBlock)
        where TSrc : unmanaged
        where TDst : unmanaged, IFloatingPointIeee754<TDst>
    {
        if (!masked)
        {
            int n = dst.Length;
            switch (op)
            {
                case Op.Add:
                    AddRow(s1, dst);
                    break;
                case Op.Product:
                    for (int i = 0; i < n; i++)
                    {
                        dst[i] += Load<TSrc, TDst>(s1[i]) * Load<TSrc, TDst>(s2[i]);
                    }

                    break;
                default:
                    int vecEnd = vectorBlock == 0 ? 0 : n / vectorBlock * vectorBlock;
                    WeightedRow(s1[..vecEnd], dst[..vecEnd], a, bVector);
                    WeightedRow(s1[vecEnd..], dst[vecEnd..], a, bTail);
                    break;
            }

            return;
        }

        int len = mask.Length;
        int pixVecEnd = vectorBlock == 0 ? 0 : len / vectorBlock * vectorBlock;
        for (int p = 0; p < len; p++)
        {
            if (mask[p] == 0)
            {
                continue;
            }

            TDst b = p < pixVecEnd ? bVector : bTail;
            for (int k = 0, i = p * cn; k < cn; k++, i++)
            {
                TDst v = Load<TSrc, TDst>(s1[i]);
                dst[i] = op switch
                {
                    Op.Add => dst[i] + v,
                    Op.Product => dst[i] + (v * Load<TSrc, TDst>(s2[i])),
                    _ => (v * a) + (dst[i] * b),
                };
            }
        }
    }

    private static void AddRow<TSrc, TDst>(ReadOnlySpan<TSrc> src, Span<TDst> dst)
        where TSrc : unmanaged
        where TDst : unmanaged, IFloatingPointIeee754<TDst>
    {
        int i = 0;
        if (typeof(TSrc) == typeof(TDst) && Vector.IsHardwareAccelerated)
        {
            var s = MemoryMarshal.Cast<TSrc, TDst>(src);
            for (; i <= dst.Length - Vector<TDst>.Count; i += Vector<TDst>.Count)
            {
                (new Vector<TDst>(dst[i..]) + new Vector<TDst>(s[i..])).CopyTo(dst[i..]);
            }
        }

        for (; i < dst.Length; i++)
        {
            dst[i] += Load<TSrc, TDst>(src[i]);
        }
    }

    private static void WeightedRow<TSrc, TDst>(ReadOnlySpan<TSrc> src, Span<TDst> dst, TDst a, TDst b)
        where TSrc : unmanaged
        where TDst : unmanaged, IFloatingPointIeee754<TDst>
    {
        int i = 0;
        if (typeof(TSrc) == typeof(TDst) && Vector.IsHardwareAccelerated)
        {
            var s = MemoryMarshal.Cast<TSrc, TDst>(src);
            var va = new Vector<TDst>(a);
            var vb = new Vector<TDst>(b);
            for (; i <= dst.Length - Vector<TDst>.Count; i += Vector<TDst>.Count)
            {
                ((new Vector<TDst>(s[i..]) * va) + (new Vector<TDst>(dst[i..]) * vb)).CopyTo(dst[i..]);
            }
        }

        for (; i < dst.Length; i++)
        {
            dst[i] = (Load<TSrc, TDst>(src[i]) * a) + (dst[i] * b);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TDst Load<TSrc, TDst>(TSrc v)
        where TSrc : unmanaged
        where TDst : unmanaged, IFloatingPointIeee754<TDst>
    {
        if (typeof(TSrc) == typeof(byte))
        {
            return TDst.CreateTruncating(Unsafe.As<TSrc, byte>(ref v));
        }

        if (typeof(TSrc) == typeof(ushort))
        {
            return TDst.CreateTruncating(Unsafe.As<TSrc, ushort>(ref v));
        }

        if (typeof(TSrc) == typeof(float))
        {
            return TDst.CreateTruncating(Unsafe.As<TSrc, float>(ref v));
        }

        return TDst.CreateTruncating(Unsafe.As<TSrc, double>(ref v));
    }
}
