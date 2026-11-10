// This implementation is ported from the OpenCV sparse optical flow module.
//
// Original sources:
//   modules/video/src/lkpyramid.cpp (calcScharrDeriv)
//   modules/video/src/lkpyramid.simd.hpp (ScharrDerivInvoker_SIMD)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.Video.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// OpenCV <c>calcScharrDeriv</c>: interleaved 3×3 Scharr x and y derivatives of an 8-bit image
/// in 16-bit wrapping arithmetic, with a reflect-101 border inside the source.
/// </summary>
internal static class LkScharrDeriv
{
    /// <summary>
    /// Writes (dx, dy) pairs for every source element into <paramref name="dst"/>, which has
    /// the source size and twice the source channel count. Pixels outside the source are never read.
    /// </summary>
    public static void Run(Mat<byte> src, Mat<short> dst)
    {
        int rows = src.Height;
        long work = (long)src.Width * src.Channels * 4;
        ParallelRows.For(rows, work, (start, end) => Rows(src, dst, start, end));
    }

    private static void Rows(Mat<byte> src, Mat<short> dst, int start, int end)
    {
        int length = (src.Width + 2) * src.Channels;
        short[] t0Buf = ArrayPool<short>.Shared.Rent(length);
        short[] t1Buf = ArrayPool<short>.Shared.Rent(length);
        try
        {
            Rows(src, dst, start, end, t0Buf, t1Buf);
        }
        finally
        {
            ArrayPool<short>.Shared.Return(t1Buf);
            ArrayPool<short>.Shared.Return(t0Buf);
        }
    }

    /// <param name="t0Buf">Row scratch of at least <c>(width + 2) · channels</c> elements.</param>
    /// <param name="t1Buf">Row scratch of the same length as <paramref name="t0Buf"/>.</param>
    private static void Rows(Mat<byte> src, Mat<short> dst, int start, int end, short[] t0Buf, short[] t1Buf)
    {
        int rows = src.Height, cols = src.Width, cn = src.Channels, colsn = cols * cn;
        Span<short> trow0 = t0Buf.AsSpan(cn, colsn + cn);
        Span<short> trow1 = t1Buf.AsSpan(cn, colsn + cn);
        var c3 = new Vector<short>(3);
        var c10 = new Vector<short>(10);
        int lanes = Vector<short>.Count;

        for (int y = start; y < end; y++)
        {
            ReadOnlySpan<byte> srow0 = src.GetReadOnlyRowSpan(y > 0 ? y - 1 : rows > 1 ? 1 : 0);
            ReadOnlySpan<byte> srow1 = src.GetReadOnlyRowSpan(y);
            ReadOnlySpan<byte> srow2 = src.GetReadOnlyRowSpan(y < rows - 1 ? y + 1 : rows > 1 ? rows - 2 : 0);
            Span<short> drow = dst.GetRowSpan(y);

            int x = 0;
            for (; x <= colsn - Vector<byte>.Count; x += Vector<byte>.Count)
            {
                Vector.Widen(new Vector<byte>(srow0.Slice(x)), out Vector<ushort> a0, out Vector<ushort> a1);
                Vector.Widen(new Vector<byte>(srow1.Slice(x)), out Vector<ushort> b0, out Vector<ushort> b1);
                Vector.Widen(new Vector<byte>(srow2.Slice(x)), out Vector<ushort> e0, out Vector<ushort> e1);
                VerticalStore(trow0, trow1, x, a0.As<ushort, short>(), b0.As<ushort, short>(), e0.As<ushort, short>(), c3, c10);
                VerticalStore(trow0, trow1, x + lanes, a1.As<ushort, short>(), b1.As<ushort, short>(), e1.As<ushort, short>(), c3, c10);
            }

            for (; x < colsn; x++)
            {
                trow0[x] = (short)(((srow0[x] + srow2[x]) * 3) + (srow1[x] * 10));
                trow1[x] = (short)(srow2[x] - srow0[x]);
            }

            int x0 = (cols > 1 ? 1 : 0) * cn, x1 = (cols > 1 ? cols - 2 : 0) * cn;
            for (int k = 0; k < cn; k++)
            {
                t0Buf[k] = trow0[x0 + k];
                trow0[colsn + k] = trow0[x1 + k];
                t1Buf[k] = trow1[x0 + k];
                trow1[colsn + k] = trow1[x1 + k];
            }

            x = Sse2.IsSupported ? HorizontalSse2(t0Buf, t1Buf, drow, colsn, cn) : 0;
            for (; x < colsn; x++)
            {
                drow[x * 2] = (short)(t0Buf[x + (2 * cn)] - t0Buf[x]);
                drow[(x * 2) + 1] = (short)(((t1Buf[x + (2 * cn)] + t1Buf[x]) * 3) + (t1Buf[x + cn] * 10));
            }
        }
    }

    /// <summary>Writes the interleaved (dx, dy) pairs of the 8-element blocks; returns the first unwritten element.</summary>
    private static int HorizontalSse2(short[] t0Buf, short[] t1Buf, Span<short> drow, int colsn, int cn)
    {
        ref short t0 = ref MemoryMarshal.GetArrayDataReference(t0Buf);
        ref short t1 = ref MemoryMarshal.GetArrayDataReference(t1Buf);
        ref short d = ref MemoryMarshal.GetReference(drow);
        Vector128<short> c3 = Vector128.Create((short)3), c10 = Vector128.Create((short)10);
        int x = 0;
        for (; x <= colsn - 8; x += 8)
        {
            Vector128<short> dx = Vector128.LoadUnsafe(ref t0, (nuint)(x + (2 * cn))) - Vector128.LoadUnsafe(ref t0, (nuint)x);
            Vector128<short> dy = ((Vector128.LoadUnsafe(ref t1, (nuint)(x + (2 * cn))) + Vector128.LoadUnsafe(ref t1, (nuint)x)) * c3)
                + (Vector128.LoadUnsafe(ref t1, (nuint)(x + cn)) * c10);
            Sse2.UnpackLow(dx, dy).StoreUnsafe(ref d, (nuint)(x * 2));
            Sse2.UnpackHigh(dx, dy).StoreUnsafe(ref d, (nuint)((x * 2) + 8));
        }

        return x;
    }

    private static void VerticalStore(Span<short> trow0, Span<short> trow1, int x, Vector<short> s0, Vector<short> s1, Vector<short> s2, Vector<short> c3, Vector<short> c10)
    {
        ((s0 + s2) * c3 + (s1 * c10)).CopyTo(trow0.Slice(x));
        (s2 - s0).CopyTo(trow1.Slice(x));
    }
}
