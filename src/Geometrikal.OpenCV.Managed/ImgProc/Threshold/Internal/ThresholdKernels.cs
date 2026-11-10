// This implementation is ported from the OpenCV threshold kernels.
//
// Original source:
//   modules/imgproc/src/thresh.cpp (thresh_8u, thresh_16s, thresh_16u, thresh_32f,
//     thresh_64f, threshGenericWithMask, ThresholdRunner, adaptiveThreshold)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>One threshold rule applied to a vector of elements.</summary>
internal interface IThresholdOp<T>
    where T : unmanaged
{
    Vector<T> Apply(Vector<T> src, Vector<T> thresh, Vector<T> maxval);
}

/// <summary>
/// Row kernels of <c>threshold</c> and <c>adaptiveThreshold</c>. Floating rules follow the
/// OpenCV 5.0 vector body (<c>v_lt</c>, <c>v_le</c>, <c>v_min</c> and bitwise AND) for every
/// element: a false comparison gives +0 and a true comparison keeps the exact bits.
/// </summary>
internal static class ThresholdKernels
{
    /// <summary>Runs one threshold rule over all rows. Where <paramref name="mask"/> is zero, the destination keeps its values.</summary>
    public static void Run<T>(Mat<T> src, Mat<T> dst, Mat<byte>? mask, T thresh, T maxval, int op)
        where T : unmanaged
    {
        switch (op)
        {
            case 0:
                Run(src, dst, mask, thresh, maxval, default(BinaryOp<T>));
                break;
            case 1:
                Run(src, dst, mask, thresh, maxval, default(BinaryInvOp<T>));
                break;
            case 2:
                if (mask is not null && Saturate.IsFloating<T>())
                {
                    Run(src, dst, mask, thresh, maxval, default(TruncStdMinOp<T>));
                }
                else
                {
                    Run(src, dst, mask, thresh, maxval, default(TruncOp<T>));
                }

                break;
            case 3:
                Run(src, dst, mask, thresh, maxval, default(ToZeroOp<T>));
                break;
            default:
                Run(src, dst, mask, thresh, maxval, default(ToZeroInvOp<T>));
                break;
        }
    }

    private static void Run<T, TOp>(Mat<T> src, Mat<T> dst, Mat<byte>? mask, T thresh, T maxval, TOp op)
        where T : unmanaged
        where TOp : struct, IThresholdOp<T>
    {
        ParallelRows.For(
            src.Height,
            Elementwise.WorkPerRow<T>(src.RowLength),
            (y0, y1) => Stripe(src, dst, mask, thresh, maxval, op, y0, y1));
    }

    private static void Stripe<T, TOp>(
        Mat<T> src, Mat<T> dst, Mat<byte>? mask, T thresh, T maxval, TOp op, int y0, int y1)
        where T : unmanaged
        where TOp : struct, IThresholdOp<T>
    {
        var t = new Vector<T>(thresh);
        var m = new Vector<T>(maxval);
        if (mask is null)
        {
            for (int y = y0; y < y1; y++)
            {
                Row(src.GetReadOnlyRowSpan(y), dst.GetRowSpan(y), t, m, op);
            }

            return;
        }

        int len = src.RowLength;
        T[] tmp = ArrayPool<T>.Shared.Rent(len);
        try
        {
            Span<T> row = tmp.AsSpan(0, len);
            for (int y = y0; y < y1; y++)
            {
                Row(src.GetReadOnlyRowSpan(y), row, t, m, op);
                Elementwise.StoreMasked<T>(row, mask.GetReadOnlyRowSpan(y), dst.GetRowSpan(y), src.Channels);
            }
        }
        finally
        {
            ArrayPool<T>.Shared.Return(tmp);
        }
    }

    /// <summary><paramref name="dst"/> may be the same memory as <paramref name="src"/>.</summary>
    public static void Row<T, TOp>(ReadOnlySpan<T> src, Span<T> dst, Vector<T> thresh, Vector<T> maxval, TOp op)
        where T : unmanaged
        where TOp : struct, IThresholdOp<T>
    {
        int n = src.Length;
        int w = Vector<T>.Count;
        if (n < w)
        {
            ShortRow(src, dst, thresh, maxval, op);
            return;
        }

        ref T s = ref MemoryMarshal.GetReference(src);
        ref T d = ref MemoryMarshal.GetReference(dst);
        nuint last = (nuint)(n - w);

        // Computed before any store, so an in-place row reads only source values.
        Vector<T> lastResult = op.Apply(Vector.LoadUnsafe(ref s, last), thresh, maxval);
        for (nuint i = 0; i < last; i += (nuint)w)
        {
            op.Apply(Vector.LoadUnsafe(ref s, i), thresh, maxval).StoreUnsafe(ref d, i);
        }

        lastResult.StoreUnsafe(ref d, last);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void ShortRow<T, TOp>(ReadOnlySpan<T> src, Span<T> dst, Vector<T> thresh, Vector<T> maxval, TOp op)
        where T : unmanaged
        where TOp : struct, IThresholdOp<T>
    {
        Span<T> buffer = stackalloc T[Vector<T>.Count];
        buffer.Clear();
        src.CopyTo(buffer);
        op.Apply(new Vector<T>(buffer), thresh, maxval).CopyTo(buffer);
        buffer[..src.Length].CopyTo(dst);
    }

    /// <summary>
    /// Final step of <c>adaptiveThreshold</c>: <c>mean[x] = pass(src[x] - mean[x]) ? maxval : 0</c>,
    /// where pass is <c>d &gt; k</c> (binary) or <c>d &lt;= k</c> (inverted).
    /// </summary>
    public static void AdaptiveCompare(Mat<byte> src, Mat<byte> mean, byte maxval, int k, bool inverted)
    {
        // d lies in [-255, 255], so clamping k keeps every comparison result.
        short sk = (short)Math.Clamp(k, -256, 255);
        if (inverted)
        {
            AdaptiveCompare(src, mean, maxval, sk, default(LessOrEqualCmp));
        }
        else
        {
            AdaptiveCompare(src, mean, maxval, sk, default(GreaterCmp));
        }
    }

    private static void AdaptiveCompare<TCmp>(Mat<byte> src, Mat<byte> mean, byte maxval, short k, TCmp cmp)
        where TCmp : struct, IShortCmp
    {
        ParallelRows.For(
            src.Height,
            Elementwise.WorkPerRow<byte>(src.RowLength),
            (y0, y1) => AdaptiveStripe(src, mean, maxval, k, cmp, y0, y1));
    }

    private static void AdaptiveStripe<TCmp>(Mat<byte> src, Mat<byte> mean, byte maxval, short k, TCmp cmp, int y0, int y1)
        where TCmp : struct, IShortCmp
    {
        for (int y = y0; y < y1; y++)
        {
            AdaptiveRow(src.GetReadOnlyRowSpan(y), mean.GetRowSpan(y), maxval, k, cmp);
        }
    }

    private static void AdaptiveRow<TCmp>(ReadOnlySpan<byte> src, Span<byte> mean, byte maxval, short k, TCmp cmp)
        where TCmp : struct, IShortCmp
    {
        int n = src.Length;
        int w = Vector<byte>.Count;
        ref byte s = ref MemoryMarshal.GetReference(src);
        ref byte m = ref MemoryMarshal.GetReference(mean);
        var vk = new Vector<short>(k);
        var vmax = new Vector<byte>(maxval);
        int i = 0;
        for (; i <= n - w; i += w)
        {
            Vector.Widen(Vector.LoadUnsafe(ref s, (nuint)i), out Vector<ushort> sLo, out Vector<ushort> sHi);
            Vector.Widen(Vector.LoadUnsafe(ref m, (nuint)i), out Vector<ushort> mLo, out Vector<ushort> mHi);
            Vector<short> lo = cmp.Compare(Vector.AsVectorInt16(sLo) - Vector.AsVectorInt16(mLo), vk);
            Vector<short> hi = cmp.Compare(Vector.AsVectorInt16(sHi) - Vector.AsVectorInt16(mHi), vk);
            Vector.BitwiseAnd(Vector.AsVectorByte(Vector.Narrow(lo, hi)), vmax).StoreUnsafe(ref m, (nuint)i);
        }

        for (; i < n; i++)
        {
            mean[i] = cmp.Compare((short)(src[i] - mean[i]), k) ? maxval : (byte)0;
        }
    }

    private interface IShortCmp
    {
        Vector<short> Compare(Vector<short> d, Vector<short> k);

        bool Compare(short d, short k);
    }

    private readonly struct GreaterCmp : IShortCmp
    {
        public Vector<short> Compare(Vector<short> d, Vector<short> k) => Vector.GreaterThan(d, k);

        public bool Compare(short d, short k) => d > k;
    }

    private readonly struct LessOrEqualCmp : IShortCmp
    {
        public Vector<short> Compare(Vector<short> d, Vector<short> k) => Vector.LessThanOrEqual(d, k);

        public bool Compare(short d, short k) => d <= k;
    }
}

/// <summary><c>THRESH_BINARY</c>: <c>(thresh &lt; src) &amp; maxval</c>.</summary>
internal readonly struct BinaryOp<T> : IThresholdOp<T>
    where T : unmanaged
{
    public Vector<T> Apply(Vector<T> src, Vector<T> thresh, Vector<T> maxval)
        => Vector.BitwiseAnd(Vector.GreaterThan(src, thresh), maxval);
}

/// <summary><c>THRESH_BINARY_INV</c>: <c>(src &lt;= thresh) &amp; maxval</c>.</summary>
internal readonly struct BinaryInvOp<T> : IThresholdOp<T>
    where T : unmanaged
{
    public Vector<T> Apply(Vector<T> src, Vector<T> thresh, Vector<T> maxval)
        => Vector.BitwiseAnd(Vector.LessThanOrEqual(src, thresh), maxval);
}

/// <summary><c>THRESH_TRUNC</c> as <c>v_min</c>: <c>src &lt; thresh ? src : thresh</c>.</summary>
internal readonly struct TruncOp<T> : IThresholdOp<T>
    where T : unmanaged
{
    public Vector<T> Apply(Vector<T> src, Vector<T> thresh, Vector<T> maxval)
        => Vector.ConditionalSelect(Vector.LessThan(src, thresh), src, thresh);
}

/// <summary>
/// <c>THRESH_TRUNC</c> as <c>std::min(src, thresh)</c>: <c>thresh &lt; src ? thresh : src</c>.
/// The masked OpenCV 5.0 kernel uses this rule; it differs from <see cref="TruncOp{T}"/> for NaN and signed zero.
/// </summary>
internal readonly struct TruncStdMinOp<T> : IThresholdOp<T>
    where T : unmanaged
{
    public Vector<T> Apply(Vector<T> src, Vector<T> thresh, Vector<T> maxval)
        => Vector.ConditionalSelect(Vector.LessThan(thresh, src), thresh, src);
}

/// <summary><c>THRESH_TOZERO</c>: <c>(thresh &lt; src) &amp; src</c>.</summary>
internal readonly struct ToZeroOp<T> : IThresholdOp<T>
    where T : unmanaged
{
    public Vector<T> Apply(Vector<T> src, Vector<T> thresh, Vector<T> maxval)
        => Vector.BitwiseAnd(Vector.GreaterThan(src, thresh), src);
}

/// <summary><c>THRESH_TOZERO_INV</c>: <c>(src &lt;= thresh) &amp; src</c>.</summary>
internal readonly struct ToZeroInvOp<T> : IThresholdOp<T>
    where T : unmanaged
{
    public Vector<T> Apply(Vector<T> src, Vector<T> thresh, Vector<T> maxval)
        => Vector.BitwiseAnd(Vector.LessThanOrEqual(src, thresh), src);
}
