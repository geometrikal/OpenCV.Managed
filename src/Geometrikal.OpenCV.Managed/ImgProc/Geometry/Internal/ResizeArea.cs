// This implementation is ported from the OpenCV area resize.
//
// Original source:
//   modules/imgproc/src/resize.cpp (resizeAreaFast_Invoker, ResizeAreaFastVec,
//     ResizeAreaFastVec_SIMD_32f, computeResizeAreaTab, ResizeArea_Invoker, inter_area::)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary><c>INTER_AREA</c> for downscaling: integer-factor cells and fractional decimation.</summary>
internal static class ResizeArea
{
    // ------------------------------------------------------------ integer factors

    /// <summary><c>resizeAreaFast_Invoker</c> for integer factors <paramref name="sx"/> × <paramref name="sy"/>.</summary>
    public static void RunFast<T>(Mat<T> src, Mat<T> dst, int sx, int sy)
        where T : unmanaged
    {
        int cn = src.Channels;
        long work = (long)dst.Width * cn * sx * sy;
        ParallelRows.For(dst.Height, work / 4, (y0, y1) => FastRows(src, dst, sx, sy, y0, y1));
    }

    private static void FastRows<T>(Mat<T> src, Mat<T> dst, int scaleX, int scaleY, int y0, int y1)
        where T : unmanaged
    {
        int cn = src.Channels;
        int area = scaleX * scaleY;
        float scale = 1f / area;
        int dwidth = dst.Width * cn;
        int swidth = src.Width * cn;
        int sheight = src.Height;
        int dwidth1 = src.Width / scaleX * cn;
        bool fast2 = scaleX == 2 && scaleY == 2;
        bool fastInt = fast2 && (cn == 1 || cn == 3 || cn == 4) &&
            (typeof(T) == typeof(byte) || typeof(T) == typeof(ushort) || typeof(T) == typeof(short));
        bool fastFloat = fast2 && (cn == 1 || cn == 4) && typeof(T) == typeof(float);
        int[] ofsArray = ArrayPool<int>.Shared.Rent(area);
        try
        {
            for (int dy = y0; dy < y1; dy++)
            {
                ref T d = ref MemoryMarshal.GetReference(dst.GetRowSpan(dy));
                int sy0 = dy * scaleY;
                if (sy0 >= sheight)
                {
                    dst.GetRowSpan(dy).Clear();
                    continue;
                }

                int w = sy0 + scaleY <= sheight ? dwidth1 : 0;
                int dx = 0;
                if (w > 0)
                {
                    ref T s0 = ref MemoryMarshal.GetReference(src.GetReadOnlyRowSpan(sy0));
                    int stride = src.Stride;
                    for (int k = 0, yy = 0; yy < scaleY; yy++)
                    {
                        for (int xx = 0; xx < scaleX; xx++)
                        {
                            ofsArray[k++] = (yy * stride) + (xx * cn);
                        }
                    }

                    if (fastInt)
                    {
                        dx = Fast2x2Int(ref s0, ref Unsafe.Add(ref s0, stride), ref d, w, cn);
                    }
                    else if (fastFloat)
                    {
                        dx = Fast2x2Float(ref s0, ref Unsafe.Add(ref s0, stride), ref d, w, cn);
                    }

                    ref int o = ref MemoryMarshal.GetArrayDataReference(ofsArray);
                    for (int k0 = dx % cn; dx < w; dx++)
                    {
                        ref T s = ref Unsafe.Add(ref s0, ((dx - k0) * scaleX) + k0);
                        Unsafe.Add(ref d, dx) = CellSum(ref s, ref o, area, scale);
                        k0 = k0 + 1 == cn ? 0 : k0 + 1;
                    }
                }

                for (; dx < dwidth; dx++)
                {
                    int k0 = dx % cn;
                    int sx0 = ((dx - k0) * scaleX) + k0;
                    Unsafe.Add(ref d, dx) = PartialCell(src, sy0, sx0, scaleX, scaleY, cn, swidth, sheight);
                }
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(ofsArray);
        }
    }

    /// <summary>Full cell: sums in groups of four, then <c>saturate_cast(sum * (1/area))</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static T CellSum<T>(ref T s, ref int ofs, int area, float scale)
        where T : unmanaged
    {
        int k = 0;
        if (typeof(T) == typeof(double))
        {
            ref double sd = ref Unsafe.As<T, double>(ref s);
            double sum = 0;
            for (; k <= area - 4; k += 4)
            {
                sum += Unsafe.Add(ref sd, Unsafe.Add(ref ofs, k)) + Unsafe.Add(ref sd, Unsafe.Add(ref ofs, k + 1))
                    + Unsafe.Add(ref sd, Unsafe.Add(ref ofs, k + 2)) + Unsafe.Add(ref sd, Unsafe.Add(ref ofs, k + 3));
            }

            for (; k < area; k++)
            {
                sum += Unsafe.Add(ref sd, Unsafe.Add(ref ofs, k));
            }

            double r = sum * scale;
            return Unsafe.As<double, T>(ref r);
        }

        if (typeof(T) == typeof(float))
        {
            ref float sf = ref Unsafe.As<T, float>(ref s);
            float sum = 0;
            for (; k <= area - 4; k += 4)
            {
                sum += Unsafe.Add(ref sf, Unsafe.Add(ref ofs, k)) + Unsafe.Add(ref sf, Unsafe.Add(ref ofs, k + 1))
                    + Unsafe.Add(ref sf, Unsafe.Add(ref ofs, k + 2)) + Unsafe.Add(ref sf, Unsafe.Add(ref ofs, k + 3));
            }

            for (; k < area; k++)
            {
                sum += Unsafe.Add(ref sf, Unsafe.Add(ref ofs, k));
            }

            float r = sum * scale;
            return Unsafe.As<float, T>(ref r);
        }

        if (typeof(T) == typeof(byte))
        {
            int sum = 0;
            for (; k < area; k++)
            {
                sum += Saturate.ToInt(Unsafe.Add(ref s, Unsafe.Add(ref ofs, k)));
            }

            return Saturate.FromFloat<T>(sum * scale);
        }

        float fsum = 0;
        for (; k <= area - 4; k += 4)
        {
            fsum += Saturate.ToInt(Unsafe.Add(ref s, Unsafe.Add(ref ofs, k))) + Saturate.ToInt(Unsafe.Add(ref s, Unsafe.Add(ref ofs, k + 1)))
                + Saturate.ToInt(Unsafe.Add(ref s, Unsafe.Add(ref ofs, k + 2))) + Saturate.ToInt(Unsafe.Add(ref s, Unsafe.Add(ref ofs, k + 3)));
        }

        for (; k < area; k++)
        {
            fsum += Saturate.ToInt(Unsafe.Add(ref s, Unsafe.Add(ref ofs, k)));
        }

        return Saturate.FromFloat<T>(fsum * scale);
    }

    /// <summary>Cell cut by the right or bottom edge: <c>saturate_cast((float)sum / count)</c>.</summary>
    private static T PartialCell<T>(Mat<T> src, int sy0, int sx0, int scaleX, int scaleY, int cn, int swidth, int sheight)
        where T : unmanaged
    {
        double dsum = 0;
        float fsum = 0;
        int isum = 0;
        int count = 0;
        for (int yy = 0; yy < scaleY; yy++)
        {
            if (sy0 + yy >= sheight)
            {
                break;
            }

            ReadOnlySpan<T> row = src.GetReadOnlyRowSpan(sy0 + yy);
            for (int xx = 0; xx < scaleX * cn; xx += cn)
            {
                if (sx0 + xx >= swidth)
                {
                    break;
                }

                T v = row[sx0 + xx];
                if (typeof(T) == typeof(double))
                {
                    dsum += Unsafe.As<T, double>(ref v);
                }
                else if (typeof(T) == typeof(float))
                {
                    fsum += Unsafe.As<T, float>(ref v);
                }
                else if (typeof(T) == typeof(byte))
                {
                    isum += Unsafe.As<T, byte>(ref v);
                }
                else
                {
                    fsum += Saturate.ToInt(v);
                }

                count++;
            }
        }

        float sum = typeof(T) == typeof(double) ? (float)dsum : typeof(T) == typeof(byte) ? isum : fsum;
        float r = sum / count;
        if (typeof(T) == typeof(double))
        {
            double dr = r;
            return Unsafe.As<double, T>(ref dr);
        }

        return Saturate.FromFloat<T>(r);
    }

    /// <summary>
    /// 2×2 cells of <c>byte</c>, <c>ushort</c> or <c>short</c> with 1, 3 or 4 channels:
    /// <c>(a + b + c + d + 2) &gt;&gt; 2</c>. Returns the number of elements written (all of <paramref name="w"/>).
    /// </summary>
    private static int Fast2x2Int<T>(ref T s0, ref T s1, ref T d, int w, int cn)
        where T : unmanaged
    {
        int dx = 0;
        if (typeof(T) == typeof(byte) && cn == 1 && Ssse3.IsSupported)
        {
            dx = Fast2x2Byte(ref Unsafe.As<T, byte>(ref s0), ref Unsafe.As<T, byte>(ref s1), ref Unsafe.As<T, byte>(ref d), w);
        }

        for (int k0 = dx % cn; dx < w; dx++)
        {
            int i = ((dx - k0) * 2) + k0;
            int sum = Saturate.ToInt(Unsafe.Add(ref s0, i)) + Saturate.ToInt(Unsafe.Add(ref s0, i + cn))
                + Saturate.ToInt(Unsafe.Add(ref s1, i)) + Saturate.ToInt(Unsafe.Add(ref s1, i + cn));
            Unsafe.Add(ref d, dx) = Saturate.FromInt<T>((sum + 2) >> 2);
            k0 = k0 + 1 == cn ? 0 : k0 + 1;
        }

        return w;
    }

    private static int Fast2x2Byte(ref byte s0, ref byte s1, ref byte d, int w)
    {
        int dx = 0;
        if (Avx2.IsSupported)
        {
            Vector256<sbyte> ones = Vector256.Create((sbyte)1);
            Vector256<short> two = Vector256.Create((short)2);
            for (; dx <= w - 32; dx += 32)
            {
                Vector256<short> lo = Avx2.MultiplyAddAdjacent(Vector256.LoadUnsafe(ref s0, (nuint)(2 * dx)), ones)
                    + Avx2.MultiplyAddAdjacent(Vector256.LoadUnsafe(ref s1, (nuint)(2 * dx)), ones);
                Vector256<short> hi = Avx2.MultiplyAddAdjacent(Vector256.LoadUnsafe(ref s0, (nuint)((2 * dx) + 32)), ones)
                    + Avx2.MultiplyAddAdjacent(Vector256.LoadUnsafe(ref s1, (nuint)((2 * dx) + 32)), ones);
                lo = Avx2.ShiftRightLogical(lo + two, 2);
                hi = Avx2.ShiftRightLogical(hi + two, 2);
                Vector256<byte> packed = Avx2.PackUnsignedSaturate(lo, hi);
                Avx2.Permute4x64(packed.AsUInt64(), 0b11_01_10_00).AsByte().StoreUnsafe(ref d, (nuint)dx);
            }
        }

        Vector128<sbyte> ones128 = Vector128.Create((sbyte)1);
        Vector128<short> two128 = Vector128.Create((short)2);
        for (; dx <= w - 16; dx += 16)
        {
            Vector128<short> lo = Ssse3.MultiplyAddAdjacent(Vector128.LoadUnsafe(ref s0, (nuint)(2 * dx)), ones128)
                + Ssse3.MultiplyAddAdjacent(Vector128.LoadUnsafe(ref s1, (nuint)(2 * dx)), ones128);
            Vector128<short> hi = Ssse3.MultiplyAddAdjacent(Vector128.LoadUnsafe(ref s0, (nuint)((2 * dx) + 16)), ones128)
                + Ssse3.MultiplyAddAdjacent(Vector128.LoadUnsafe(ref s1, (nuint)((2 * dx) + 16)), ones128);
            Sse2.PackUnsignedSaturate(Sse2.ShiftRightLogical(lo + two128, 2), Sse2.ShiftRightLogical(hi + two128, 2)).StoreUnsafe(ref d, (nuint)dx);
        }

        return dx;
    }

    /// <summary>
    /// 2×2 cells of <c>float</c>: <c>((a + b) + (c + d)) * 0.25f</c> for 4 channels, and for the first
    /// <c>w / 4 * 4</c> elements of 1 channel. Returns the number of elements written.
    /// </summary>
    private static int Fast2x2Float<T>(ref T s0t, ref T s1t, ref T dt, int w, int cn)
        where T : unmanaged
    {
        ref float s0 = ref Unsafe.As<T, float>(ref s0t);
        ref float s1 = ref Unsafe.As<T, float>(ref s1t);
        ref float d = ref Unsafe.As<T, float>(ref dt);
        int end = cn == 1 ? w / 4 * 4 : w;
        for (int dx = 0, k0 = 0; dx < end; dx++)
        {
            int i = ((dx - k0) * 2) + k0;
            Unsafe.Add(ref d, dx) = ((Unsafe.Add(ref s0, i) + Unsafe.Add(ref s0, i + cn)) + (Unsafe.Add(ref s1, i) + Unsafe.Add(ref s1, i + cn))) * 0.25f;
            k0 = k0 + 1 == cn ? 0 : k0 + 1;
        }

        return end;
    }

    // ------------------------------------------------------------ fractional decimation

    /// <summary>One weight of <c>computeResizeAreaTab</c>.</summary>
    internal struct DecimateAlpha
    {
        public int Si;
        public int Di;
        public float Alpha;
    }

    /// <summary><c>computeResizeAreaTab</c>. Returns the number of entries.</summary>
    public static int ComputeTab(int ssize, int dsize, int cn, double scale, Span<DecimateAlpha> tab)
    {
        int k = 0;
        for (int dx = 0; dx < dsize; dx++)
        {
            double fsx1 = dx * scale;
            double fsx2 = fsx1 + scale;
            double cellWidth = Math.Min(scale, ssize - fsx1);
            int sx1 = ScalarConverter.CvCeil(fsx1), sx2 = ScalarConverter.CvFloor(fsx2);
            sx2 = Math.Min(sx2, ssize - 1);
            sx1 = Math.Min(sx1, sx2);
            if (sx1 - fsx1 > 1e-3)
            {
                tab[k++] = new DecimateAlpha { Di = dx * cn, Si = (sx1 - 1) * cn, Alpha = (float)((sx1 - fsx1) / cellWidth) };
            }

            for (int sx = sx1; sx < sx2; sx++)
            {
                tab[k++] = new DecimateAlpha { Di = dx * cn, Si = sx * cn, Alpha = (float)(1.0 / cellWidth) };
            }

            if (fsx2 - sx2 > 1e-3)
            {
                tab[k++] = new DecimateAlpha
                {
                    Di = dx * cn,
                    Si = sx2 * cn,
                    Alpha = (float)(Math.Min(Math.Min(fsx2 - sx2, 1.0), cellWidth) / cellWidth),
                };
            }
        }

        return k;
    }

    /// <summary><c>ResizeArea_Invoker</c>: weighted sums in <c>float</c> (<c>double</c> for <c>double</c>).</summary>
    public static void RunDecimate<T>(Mat<T> src, Mat<T> dst, double scaleX, double scaleY)
        where T : unmanaged
    {
        int cn = src.Channels;
        var xtabArray = ArrayPool<DecimateAlpha>.Shared.Rent(src.Width * 2);
        var ytabArray = ArrayPool<DecimateAlpha>.Shared.Rent(src.Height * 2);
        int[] tabofs = ArrayPool<int>.Shared.Rent(dst.Height + 1);
        int[] xstart = ArrayPool<int>.Shared.Rent(dst.Width + 1);
        try
        {
            int xtabSize = ComputeTab(src.Width, dst.Width, cn, scaleX, xtabArray);
            int ytabSize = ComputeTab(src.Height, dst.Height, 1, scaleY, ytabArray);
            Offsets(ytabArray, ytabSize, tabofs, dst.Height);
            Offsets(xtabArray, xtabSize, xstart, dst.Width);
            var tables = new DecimateTables(xtabArray, xstart, ytabArray, tabofs);
            long work = (long)xtabSize * cn * ytabSize / Math.Max(1, dst.Height);
            if (typeof(T) == typeof(double))
            {
                ParallelRows.For(dst.Height, work, (y0, y1) => DecimateRows<T, double>(src, dst, tables, y0, y1));
            }
            else
            {
                ParallelRows.For(dst.Height, work, (y0, y1) => DecimateRows<T, float>(src, dst, tables, y0, y1));
            }
        }
        finally
        {
            ArrayPool<DecimateAlpha>.Shared.Return(xtabArray);
            ArrayPool<DecimateAlpha>.Shared.Return(ytabArray);
            ArrayPool<int>.Shared.Return(tabofs);
            ArrayPool<int>.Shared.Return(xstart);
        }
    }

    /// <summary>First table entry of each destination index; entries of one index are consecutive.</summary>
    private static void Offsets(DecimateAlpha[] tab, int size, int[] start, int count)
    {
        int d = 0;
        for (int k = 0; k < size; k++)
        {
            if (k == 0 || tab[k].Di != tab[k - 1].Di)
            {
                start[d++] = k;
            }
        }

        while (d <= count)
        {
            start[d++] = size;
        }
    }

    private sealed record DecimateTables(DecimateAlpha[] XTab, int[] XStart, DecimateAlpha[] YTab, int[] YStart);

    private static void DecimateRows<T, TWork>(Mat<T> src, Mat<T> dst, DecimateTables t, int y0, int y1)
        where T : unmanaged
        where TWork : unmanaged, IFloatingPointIeee754<TWork>
    {
        int cn = dst.Channels;
        int dw = dst.Width;
        int width = dw * cn;
        TWork[] bufArray = ArrayPool<TWork>.Shared.Rent(width);
        TWork[] sumArray = ArrayPool<TWork>.Shared.Rent(width);
        try
        {
            Span<TWork> buf = bufArray.AsSpan(0, width);
            Span<TWork> sum = sumArray.AsSpan(0, width);
            sum.Clear();
            DecimateAlpha[] ytab = t.YTab;
            int jStart = t.YStart[y0], jEnd = t.YStart[y1];
            int prevDy = ytab[jStart].Di;

            // In a single-range run only the first row starts from the zeroed sum (0 + beta * buf).
            bool first = y0 == 0;
            for (int j = jStart; j < jEnd; j++)
            {
                TWork beta = TWork.CreateTruncating(ytab[j].Alpha);
                int dy = ytab[j].Di;
                Accumulate(src.GetReadOnlyRowSpan(ytab[j].Si), buf, t.XTab, t.XStart, dw, cn);
                if (dy != prevDy)
                {
                    Store(sum, dst.GetRowSpan(prevDy));
                    Mul(buf, beta, sum);
                    prevDy = dy;
                }
                else if (j == jStart && !first)
                {
                    Mul(buf, beta, sum);
                }
                else
                {
                    MulAdd(buf, beta, sum);
                }
            }

            Store(sum, dst.GetRowSpan(prevDy));
        }
        finally
        {
            ArrayPool<TWork>.Shared.Return(bufArray);
            ArrayPool<TWork>.Shared.Return(sumArray);
        }
    }

    /// <summary>
    /// <c>buf[di + c] += S[si + c] * alpha</c> over the table, in table order. Each destination
    /// element accumulates in a register from 0, which gives the same operations as the zeroed buffer.
    /// </summary>
    private static void Accumulate<T, TWork>(ReadOnlySpan<T> s, Span<TWork> buf, DecimateAlpha[] xtab, int[] xstart, int dw, int cn)
        where T : unmanaged
        where TWork : unmanaged, IFloatingPointIeee754<TWork>
    {
        ref TWork b = ref MemoryMarshal.GetReference(buf);
        ref T sr = ref MemoryMarshal.GetReference(s);
        ref DecimateAlpha tab = ref MemoryMarshal.GetArrayDataReference(xtab);
        ref int st = ref MemoryMarshal.GetArrayDataReference(xstart);
        int k = Unsafe.Add(ref st, 0);
        for (int dx = 0; dx < dw; dx++)
        {
            int kEnd = Unsafe.Add(ref st, dx + 1);
            ref TWork bd = ref Unsafe.Add(ref b, dx * cn);
            switch (cn)
            {
                case 1:
                {
                    TWork a0 = TWork.Zero;
                    for (; k < kEnd; k++)
                    {
                        ref DecimateAlpha e = ref Unsafe.Add(ref tab, k);
                        a0 += Load<T, TWork>(Unsafe.Add(ref sr, e.Si)) * TWork.CreateTruncating(e.Alpha);
                    }

                    bd = a0;
                    break;
                }

                case 3:
                {
                    TWork a0 = TWork.Zero, a1 = TWork.Zero, a2 = TWork.Zero;
                    for (; k < kEnd; k++)
                    {
                        ref DecimateAlpha e = ref Unsafe.Add(ref tab, k);
                        TWork alpha = TWork.CreateTruncating(e.Alpha);
                        ref T ss = ref Unsafe.Add(ref sr, e.Si);
                        a0 += Load<T, TWork>(ss) * alpha;
                        a1 += Load<T, TWork>(Unsafe.Add(ref ss, 1)) * alpha;
                        a2 += Load<T, TWork>(Unsafe.Add(ref ss, 2)) * alpha;
                    }

                    bd = a0;
                    Unsafe.Add(ref bd, 1) = a1;
                    Unsafe.Add(ref bd, 2) = a2;
                    break;
                }

                default:
                {
                    TWork a0 = TWork.Zero, a1 = TWork.Zero, a2 = TWork.Zero, a3 = TWork.Zero;
                    for (; k < kEnd; k++)
                    {
                        ref DecimateAlpha e = ref Unsafe.Add(ref tab, k);
                        TWork alpha = TWork.CreateTruncating(e.Alpha);
                        ref T ss = ref Unsafe.Add(ref sr, e.Si);
                        a0 += Load<T, TWork>(ss) * alpha;
                        a1 += Load<T, TWork>(Unsafe.Add(ref ss, 1)) * alpha;
                        if (cn > 2)
                        {
                            a2 += Load<T, TWork>(Unsafe.Add(ref ss, 2)) * alpha;
                            a3 += Load<T, TWork>(Unsafe.Add(ref ss, 3)) * alpha;
                        }
                    }

                    bd = a0;
                    Unsafe.Add(ref bd, 1) = a1;
                    if (cn > 2)
                    {
                        Unsafe.Add(ref bd, 2) = a2;
                        Unsafe.Add(ref bd, 3) = a3;
                    }

                    break;
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static TWork Load<T, TWork>(T v)
        where T : unmanaged
        where TWork : unmanaged, IFloatingPointIeee754<TWork>
    {
        if (typeof(T) == typeof(double))
        {
            return TWork.CreateTruncating(Unsafe.As<T, double>(ref v));
        }

        if (typeof(T) == typeof(float))
        {
            return TWork.CreateTruncating(Unsafe.As<T, float>(ref v));
        }

        return TWork.CreateTruncating(Saturate.ToInt(v));
    }

    private static void Mul<TWork>(Span<TWork> buf, TWork beta, Span<TWork> sum)
        where TWork : unmanaged, IFloatingPointIeee754<TWork>
    {
        ref TWork b = ref MemoryMarshal.GetReference(buf);
        ref TWork s = ref MemoryMarshal.GetReference(sum);
        int n = buf.Length;
        int x = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var vb = new Vector<TWork>(beta);
            int vc = Vector<TWork>.Count;
            for (; x <= n - vc; x += vc)
            {
                (vb * Vector.LoadUnsafe(ref b, (nuint)x)).StoreUnsafe(ref s, (nuint)x);
            }
        }

        for (; x < n; x++)
        {
            Unsafe.Add(ref s, x) = beta * Unsafe.Add(ref b, x);
        }
    }

    private static void MulAdd<TWork>(Span<TWork> buf, TWork beta, Span<TWork> sum)
        where TWork : unmanaged, IFloatingPointIeee754<TWork>
    {
        ref TWork b = ref MemoryMarshal.GetReference(buf);
        ref TWork s = ref MemoryMarshal.GetReference(sum);
        int n = buf.Length;
        int x = 0;
        if (Vector.IsHardwareAccelerated)
        {
            var vb = new Vector<TWork>(beta);
            int vc = Vector<TWork>.Count;
            for (; x <= n - vc; x += vc)
            {
                (Vector.LoadUnsafe(ref s, (nuint)x) + (vb * Vector.LoadUnsafe(ref b, (nuint)x))).StoreUnsafe(ref s, (nuint)x);
            }
        }

        for (; x < n; x++)
        {
            Unsafe.Add(ref s, x) += beta * Unsafe.Add(ref b, x);
        }
    }

    private static void Store<T, TWork>(Span<TWork> sum, Span<T> d)
        where T : unmanaged
        where TWork : unmanaged, IFloatingPointIeee754<TWork>
    {
        if (typeof(T) == typeof(TWork))
        {
            MemoryMarshal.Cast<TWork, T>(sum).CopyTo(d);
            return;
        }

        ref float f = ref Unsafe.As<TWork, float>(ref MemoryMarshal.GetReference(sum));
        int n = d.Length;
        int x = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int vc = Vector<float>.Count;
            if (typeof(T) == typeof(byte))
            {
                ref byte db = ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(d));
                for (; x <= n - (4 * vc); x += 4 * vc)
                {
                    FilterVectors.SaturateToByte(
                        FilterVectors.CvRound(Vector.LoadUnsafe(ref f, (nuint)x)),
                        FilterVectors.CvRound(Vector.LoadUnsafe(ref f, (nuint)(x + vc))),
                        FilterVectors.CvRound(Vector.LoadUnsafe(ref f, (nuint)(x + (2 * vc)))),
                        FilterVectors.CvRound(Vector.LoadUnsafe(ref f, (nuint)(x + (3 * vc))))).StoreUnsafe(ref db, (nuint)x);
                }
            }
            else if (typeof(T) == typeof(short))
            {
                ref short ds = ref Unsafe.As<T, short>(ref MemoryMarshal.GetReference(d));
                for (; x <= n - (2 * vc); x += 2 * vc)
                {
                    FilterVectors.SaturateToInt16(
                        FilterVectors.CvRound(Vector.LoadUnsafe(ref f, (nuint)x)),
                        FilterVectors.CvRound(Vector.LoadUnsafe(ref f, (nuint)(x + vc)))).StoreUnsafe(ref ds, (nuint)x);
                }
            }
        }

        for (; x < n; x++)
        {
            d[x] = Saturate.FromFloat<T>(Unsafe.Add(ref f, x));
        }
    }
}
