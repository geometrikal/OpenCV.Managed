// This implementation is ported from the OpenCV image pyramid functions.
//
// Original sources:
//   modules/imgproc/src/pyramids.cpp (pyrUp_, PyrUpVecV, PyrUpVecVOneRow)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// <c>pyrUp</c>: insertion of zero rows and columns, then a 5×5 Gaussian <c>[1 4 6 4 1]/8</c> in
/// each direction. Each source row gives two output rows from a ring of three horizontally
/// upsampled source rows, as OpenCV <c>pyrUp_</c>.
/// </summary>
internal static class PyrUpEngine
{
    private const int Taps = 3;

    public static void Run<T>(Mat<T> src, Mat<T> dst)
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte))
        {
            Run<T, ushort>(src, dst);
        }
        else if (typeof(T) == typeof(ushort) || typeof(T) == typeof(short))
        {
            Run<T, int>(src, dst);
        }
        else if (typeof(T) == typeof(float))
        {
            Run<T, float>(src, dst);
        }
        else
        {
            Run<T, double>(src, dst);
        }
    }

    private static void Run<T, TW>(Mat<T> src, Mat<T> dst)
        where T : unmanaged
        where TW : unmanaged, INumberBase<TW>
    {
        if (ReferenceEquals(src.DangerousGetArray(), dst.DangerousGetArray()))
        {
            src = src.Clone();
        }

        int sh = src.Height;
        long work = 2L * (src.RowLength + (2L * dst.RowLength));
        if (!ParallelRows.IsSerial(sh, work))
        {
            PageTouch.Rows(dst);
        }

        ParallelRows.For(sh, work, (y0, y1) => Stripe<T, TW>(src, dst, y0, y1));

        if (dst.Height > sh * 2)
        {
            // OpenCV copies row 2·sh − 2 to the extra last row.
            dst.GetReadOnlyRowSpan((sh * 2) - 2).CopyTo(dst.GetRowSpan(sh * 2));
        }
    }

    private static void Stripe<T, TW>(Mat<T> src, Mat<T> dst, int y0, int y1)
        where T : unmanaged
        where TW : unmanaged, INumberBase<TW>
    {
        int sh = src.Height;
        int cn = src.Channels;
        int sw = src.Width;
        int dw = dst.Width;
        int dh = dst.Height;
        int srcLength = sw * cn;
        int rowLength = Math.Max(sw * 2, dw) * cn;
        int dstLength = dw * cn;
        int vecEnd = typeof(T) == typeof(float) ? dstLength & ~3 : 0;
        bool widen = typeof(T) != typeof(TW);
        TW[][] ring = new TW[Taps][];
        TW[] wide = widen ? ArrayPool<TW>.Shared.Rent(srcLength) : [];
        TW[] t0 = ArrayPool<TW>.Shared.Rent(srcLength);
        TW[] t1 = ArrayPool<TW>.Shared.Rent(srcLength);
        int[] slots = new int[Taps];
        try
        {
            for (int k = 0; k < Taps; k++)
            {
                ring[k] = ArrayPool<TW>.Shared.Rent(rowLength);
            }

            int sy = y0 - 1;
            for (int y = y0; y < y1; y++)
            {
                for (; sy <= y + 1; sy++)
                {
                    int bsy = Borders.BorderInterpolate(sy * 2, sh * 2, BorderType.Reflect101) / 2;
                    ReadOnlySpan<T> srow = src.GetReadOnlyRowSpan(bsy);
                    ReadOnlySpan<TW> w;
                    if (widen)
                    {
                        PyrRows.Widen(srow, wide.AsSpan(0, srcLength));
                        w = wide.AsSpan(0, srcLength);
                    }
                    else
                    {
                        w = MemoryMarshal.Cast<T, TW>(srow);
                    }

                    Horizontal(w, ring[(sy + 1) % Taps].AsSpan(0, rowLength), t0, t1, cn, dw);
                }

                for (int k = 0; k < Taps; k++)
                {
                    slots[k] = (y + k) % Taps;
                }

                int row0 = y * 2;
                int row1 = Math.Min(row0 + 1, dh - 1);
                Span<T> d1 = row1 != row0 ? dst.GetRowSpan(row1) : default;
                PyrColumns.Up(ring, slots, dst.GetRowSpan(row0), d1, vecEnd);
            }
        }
        finally
        {
            for (int k = 0; k < Taps; k++)
            {
                if (ring[k] is not null)
                {
                    ArrayPool<TW>.Shared.Return(ring[k]);
                }
            }

            ArrayPool<TW>.Shared.Return(t1);
            ArrayPool<TW>.Shared.Return(t0);
            if (widen)
            {
                ArrayPool<TW>.Shared.Return(wide);
            }
        }
    }

    private static void Horizontal<TW>(ReadOnlySpan<TW> w, Span<TW> row, TW[] t0, TW[] t1, int cn, int dw)
        where TW : unmanaged, INumberBase<TW>
    {
        int srcLength = w.Length;
        int sw = srcLength / cn;
        TW two = TW.CreateTruncating(2);
        TW four = TW.CreateTruncating(4);
        TW six = TW.CreateTruncating(6);
        TW seven = TW.CreateTruncating(7);
        TW eight = TW.CreateTruncating(8);
        if (sw == 1)
        {
            for (int c = 0; c < cn; c++)
            {
                row[c] = row[c + cn] = w[c] * eight;

                // OpenCV leaves this column uninitialized; the port repeats column 1.
                if (dw > 2)
                {
                    row[(2 * cn) + c] = row[cn + c];
                }
            }

            return;
        }

        int interiorEnd = srcLength - cn;
        if (interiorEnd > cn)
        {
            UpTaps(w, t0, t1, cn, interiorEnd);
            PyrRows.Interleave<TW>(t0, t1, row, cn, interiorEnd, cn);
        }

        for (int c = 0; c < cn; c++)
        {
            row[c] = (w[c] * six) + (w[c + cn] * two);
            row[cn + c] = (w[c] + w[c + cn]) * four;
            int sx = srcLength - cn + c;
            int dx = (2 * (srcLength - cn)) + c;
            row[dx] = w[sx - cn] + (w[sx] * seven);
            row[dx + cn] = w[sx] * eight;
            if (dw > sw * 2)
            {
                row[((dw - 1) * cn) + c] = row[dx + cn];
            }
        }
    }

    /// <summary>
    /// <c>t0[x] = (w[x−cn] + w[x]·6) + w[x+cn]</c> and <c>t1[x] = (w[x] + w[x+cn])·4</c> for
    /// <c>x</c> in <c>[cn, end)</c>, with <c>end + cn ≤ w.Length</c>.
    /// </summary>
    private static void UpTaps<TW>(ReadOnlySpan<TW> w, TW[] t0, TW[] t1, int cn, int end)
        where TW : unmanaged, INumberBase<TW>
    {
        ref TW s = ref MemoryMarshal.GetReference(w);
        ref TW a = ref MemoryMarshal.GetArrayDataReference(t0);
        ref TW b = ref MemoryMarshal.GetArrayDataReference(t1);
        int x = cn;
        if (Vector.IsHardwareAccelerated && Vector<TW>.IsSupported && end - cn >= Vector<TW>.Count)
        {
            int vc = Vector<TW>.Count;
            var six = new Vector<TW>(TW.CreateTruncating(6));
            var four = new Vector<TW>(TW.CreateTruncating(4));
            for (; x < end; x += vc)
            {
                // The final vector overlaps the previous one; it rewrites equal values.
                x = Math.Min(x, end - vc);
                Vector<TW> l = Vector.LoadUnsafe(ref s, (nuint)(x - cn));
                Vector<TW> m = Vector.LoadUnsafe(ref s, (nuint)x);
                Vector<TW> r = Vector.LoadUnsafe(ref s, (nuint)(x + cn));
                ((l + (m * six)) + r).StoreUnsafe(ref a, (nuint)x);
                ((m + r) * four).StoreUnsafe(ref b, (nuint)x);
            }

            return;
        }

        TW six1 = TW.CreateTruncating(6);
        TW four1 = TW.CreateTruncating(4);
        for (; x < end; x++)
        {
            TW m = Unsafe.Add(ref s, x);
            TW r = Unsafe.Add(ref s, x + cn);
            Unsafe.Add(ref a, x) = (Unsafe.Add(ref s, x - cn) + (m * six1)) + r;
            Unsafe.Add(ref b, x) = (m + r) * four1;
        }
    }
}
