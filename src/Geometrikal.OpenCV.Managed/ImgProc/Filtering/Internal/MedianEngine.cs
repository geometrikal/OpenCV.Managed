// This implementation is ported from the OpenCV median filter.
//
// Original sources:
//   modules/imgproc/src/median_blur.dispatch.cpp (medianBlur)
//   modules/imgproc/src/median_blur.simd.hpp (medianBlur_SortNet,
//     medianBlur_8u_Om / medianBlur_8u_O1 — see note below)
//
// Kernel sizes 3 and 5 use OpenCV's sorting networks with the exact
// std::min/std::max comparison semantics (significant for float NaNs).
// Kernel sizes >= 7 (8-bit only) use the constant-time column-histogram
// method of medianBlur_8u_O1. Medians of 8-bit samples are uniquely
// determined by the window contents, so the output equals OpenCV's
// O(m) / O(1) implementations by construction. Borders
// replicate edge pixels, exactly as OpenCV's clamped indices /
// BORDER_REPLICATE padding do.
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Median filtering: sorting networks for kernel sizes 3/5 (byte and float)
/// and a sliding two-tier histogram for larger odd sizes (byte).
/// </summary>
internal static class MedianEngine
{
    // The 1-D networks used for single-row/column images.
    private static readonly (int A, int B)[] Net3 = [(0, 1), (1, 2), (0, 1)];
    private static readonly (int A, int B)[] Net5 =
    [
        (0, 1), (3, 4), (2, 3), (3, 4), (0, 2), (2, 4), (1, 3), (1, 2),
    ];

    /// <summary>Sort-network median for ksize 3 or 5 (byte or float elements).</summary>
    public static Mat<T> SortNet<T>(Mat<T> source, int ksize)
        where T : unmanaged
    {
        int cn = source.Channels;
        int width = source.Width;
        int height = source.Height;
        var dst = new Mat<T>(width, height, cn, clear: false);

        if (width == 1 || height == 1)
        {
            SortNet1D(source, dst, ksize);
            return dst;
        }

        if (typeof(T) == typeof(byte) && ksize == 3 && Vector.IsHardwareAccelerated)
        {
            var src8 = (Mat<byte>)(object)source;
            var dst8 = (Mat<byte>)(object)dst;
            ParallelRows.For(height, (long)width * cn * 24, (y0, y1) => Median3Bytes(src8, dst8, y0, y1));
            return dst;
        }

        ParallelRows.For(height, (long)width * cn * ksize * ksize * 2, (y0, y1) =>
        {
            for (int i = y0; i < y1; i++)
            {
                if (ksize == 3)
                {
                    Median3Row(
                        source.GetReadOnlyRowSpan(Math.Max(i - 1, 0)),
                        source.GetReadOnlyRowSpan(i),
                        source.GetReadOnlyRowSpan(Math.Min(i + 1, height - 1)),
                        dst.GetRowSpan(i),
                        cn);
                }
                else
                {
                    Median5Row(
                        source.GetReadOnlyRowSpan(Math.Max(i - 2, 0)),
                        source.GetReadOnlyRowSpan(Math.Max(i - 1, 0)),
                        source.GetReadOnlyRowSpan(i),
                        source.GetReadOnlyRowSpan(Math.Min(i + 1, height - 1)),
                        source.GetReadOnlyRowSpan(Math.Min(i + 2, height - 1)),
                        dst.GetRowSpan(i),
                        cn);
                }
            }
        });

        return dst;
    }

    /// <summary>
    /// 3x3 byte median by Paeth's method: sort each column of three into low, middle and high,
    /// then the median is <c>med3(max of lows, med3 of middles, min of highs)</c>. Neighbouring
    /// outputs share the column sorts. The result equals the sorting network for integers.
    /// </summary>
    private static void Median3Bytes(Mat<byte> source, Mat<byte> dst, int y0, int y1)
    {
        int cn = source.Channels;
        int height = source.Height;
        int n = source.Width * cn;
        int vc = Vector<byte>.Count;
        byte[] sorted = ArrayPool<byte>.Shared.Rent(3 * n);
        try
        {
            ref byte lo = ref sorted[0];
            ref byte mid = ref sorted[n];
            ref byte hi = ref sorted[2 * n];
            for (int i = y0; i < y1; i++)
            {
                ReadOnlySpan<byte> row0 = source.GetReadOnlyRowSpan(Math.Max(i - 1, 0));
                ReadOnlySpan<byte> row1 = source.GetReadOnlyRowSpan(i);
                ReadOnlySpan<byte> row2 = source.GetReadOnlyRowSpan(Math.Min(i + 1, height - 1));
                Span<byte> d = dst.GetRowSpan(i);

                ref byte s0 = ref MemoryMarshal.GetReference(row0);
                ref byte s1 = ref MemoryMarshal.GetReference(row1);
                ref byte s2 = ref MemoryMarshal.GetReference(row2);
                int j = 0;
                for (; j <= n - vc; j += vc)
                {
                    Sort3(
                        Vector.LoadUnsafe(ref s0, (nuint)j),
                        Vector.LoadUnsafe(ref s1, (nuint)j),
                        Vector.LoadUnsafe(ref s2, (nuint)j),
                        out Vector<byte> l,
                        out Vector<byte> m,
                        out Vector<byte> h);
                    l.StoreUnsafe(ref lo, (nuint)j);
                    m.StoreUnsafe(ref mid, (nuint)j);
                    h.StoreUnsafe(ref hi, (nuint)j);
                }

                for (; j < n; j++)
                {
                    Sort3(
                        Unsafe.Add(ref s0, j),
                        Unsafe.Add(ref s1, j),
                        Unsafe.Add(ref s2, j),
                        out Unsafe.Add(ref lo, j),
                        out Unsafe.Add(ref mid, j),
                        out Unsafe.Add(ref hi, j));
                }

                j = 0;
                for (; j < cn; j++)
                {
                    d[j] = Median3At(row0, row1, row2, j, cn, n);
                }

                ref byte dr = ref MemoryMarshal.GetReference(d);
                int last = n - cn - vc;
                if (last >= cn)
                {
                    for (; ; j += vc)
                    {
                        // The final vector overlaps the previous one; it rewrites equal values.
                        j = Math.Min(j, last);
                        nuint l = (nuint)(j - cn);
                        nuint c = (nuint)j;
                        nuint r = (nuint)(j + cn);
                        Vector<byte> maxLo = Vector.Max(
                            Vector.Max(Vector.LoadUnsafe(ref lo, l), Vector.LoadUnsafe(ref lo, c)), Vector.LoadUnsafe(ref lo, r));
                        Vector<byte> minHi = Vector.Min(
                            Vector.Min(Vector.LoadUnsafe(ref hi, l), Vector.LoadUnsafe(ref hi, c)), Vector.LoadUnsafe(ref hi, r));
                        Vector<byte> medMid = Med3(
                            Vector.LoadUnsafe(ref mid, l), Vector.LoadUnsafe(ref mid, c), Vector.LoadUnsafe(ref mid, r));
                        Med3(maxLo, medMid, minHi).StoreUnsafe(ref dr, c);
                        if (j == last)
                        {
                            j += vc;
                            break;
                        }
                    }
                }

                for (; j < n; j++)
                {
                    d[j] = Median3At(row0, row1, row2, j, cn, n);
                }
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(sorted);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Sort3(Vector<byte> a, Vector<byte> b, Vector<byte> c, out Vector<byte> lo, out Vector<byte> mid, out Vector<byte> hi)
    {
        Vector<byte> minAb = Vector.Min(a, b);
        Vector<byte> maxAb = Vector.Max(a, b);
        lo = Vector.Min(minAb, c);
        Vector<byte> t = Vector.Max(minAb, c);
        mid = Vector.Min(maxAb, t);
        hi = Vector.Max(maxAb, t);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Sort3(byte a, byte b, byte c, out byte lo, out byte mid, out byte hi)
    {
        byte minAb = Math.Min(a, b);
        byte maxAb = Math.Max(a, b);
        lo = Math.Min(minAb, c);
        byte t = Math.Max(minAb, c);
        mid = Math.Min(maxAb, t);
        hi = Math.Max(maxAb, t);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<byte> Med3(Vector<byte> a, Vector<byte> b, Vector<byte> c)
        => Vector.Max(Vector.Min(a, b), Vector.Min(Vector.Max(a, b), c));

    private static void Median3Row<T>(
        ReadOnlySpan<T> row0, ReadOnlySpan<T> row1, ReadOnlySpan<T> row2, Span<T> d, int cn)
        where T : unmanaged
    {
        int n = d.Length;
        int j = 0;
        for (; j < cn; j++)
        {
            d[j] = Median3At(row0, row1, row2, j, cn, n);
        }

        if (MinMaxOps.IsVectorized<T>())
        {
            int vc = Vector<T>.Count;
            ref T s0 = ref MemoryMarshal.GetReference(row0);
            ref T s1 = ref MemoryMarshal.GetReference(row1);
            ref T s2 = ref MemoryMarshal.GetReference(row2);
            ref T dr = ref MemoryMarshal.GetReference(d);
            int last = n - cn - vc;
            for (; j <= last + vc - 1 && last >= cn; j += vc)
            {
                // The final vector overlaps the previous one; it rewrites equal values.
                j = Math.Min(j, last);
                nuint c = (nuint)j;
                nuint l = (nuint)(j - cn);
                nuint r = (nuint)(j + cn);
                Vector<T> p0 = Vector.LoadUnsafe(ref s0, l);
                Vector<T> p1 = Vector.LoadUnsafe(ref s0, c);
                Vector<T> p2 = Vector.LoadUnsafe(ref s0, r);
                Vector<T> p3 = Vector.LoadUnsafe(ref s1, l);
                Vector<T> p4 = Vector.LoadUnsafe(ref s1, c);
                Vector<T> p5 = Vector.LoadUnsafe(ref s1, r);
                Vector<T> p6 = Vector.LoadUnsafe(ref s2, l);
                Vector<T> p7 = Vector.LoadUnsafe(ref s2, c);
                Vector<T> p8 = Vector.LoadUnsafe(ref s2, r);
                Network9(ref p0, ref p1, ref p2, ref p3, ref p4, ref p5, ref p6, ref p7, ref p8);
                p4.StoreUnsafe(ref dr, c);
            }
        }

        for (; j < n; j++)
        {
            d[j] = Median3At(row0, row1, row2, j, cn, n);
        }
    }

    private static T Median3At<T>(ReadOnlySpan<T> row0, ReadOnlySpan<T> row1, ReadOnlySpan<T> row2, int j, int cn, int n)
        where T : unmanaged
    {
        int j0 = j >= cn ? j - cn : j;
        int j2 = j < n - cn ? j + cn : j;
        T p0 = row0[j0], p1 = row0[j], p2 = row0[j2];
        T p3 = row1[j0], p4 = row1[j], p5 = row1[j2];
        T p6 = row2[j0], p7 = row2[j], p8 = row2[j2];
        Network9(ref p0, ref p1, ref p2, ref p3, ref p4, ref p5, ref p6, ref p7, ref p8);
        return p4;
    }

    /// <summary>The 3x3 sorting network of medianBlur_SortNet (median_blur.simd.hpp:681), unrolled.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Network9<T>(
        ref T p0, ref T p1, ref T p2, ref T p3, ref T p4, ref T p5, ref T p6, ref T p7, ref T p8)
        where T : unmanaged
    {
        MinMax(ref p1, ref p2); MinMax(ref p4, ref p5); MinMax(ref p7, ref p8);
        MinMax(ref p0, ref p1); MinMax(ref p3, ref p4); MinMax(ref p6, ref p7);
        MinMax(ref p1, ref p2); MinMax(ref p4, ref p5); MinMax(ref p7, ref p8);
        MinMax(ref p0, ref p3); MinMax(ref p5, ref p8); MinMax(ref p4, ref p7);
        MinMax(ref p3, ref p6); MinMax(ref p1, ref p4); MinMax(ref p2, ref p5);
        MinMax(ref p4, ref p7); MinMax(ref p4, ref p2); MinMax(ref p6, ref p4);
        MinMax(ref p4, ref p2);
    }

    /// <inheritdoc cref="Network9{T}"/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Network9<T>(
        ref Vector<T> p0, ref Vector<T> p1, ref Vector<T> p2, ref Vector<T> p3, ref Vector<T> p4,
        ref Vector<T> p5, ref Vector<T> p6, ref Vector<T> p7, ref Vector<T> p8)
        where T : unmanaged
    {
        MinMax(ref p1, ref p2); MinMax(ref p4, ref p5); MinMax(ref p7, ref p8);
        MinMax(ref p0, ref p1); MinMax(ref p3, ref p4); MinMax(ref p6, ref p7);
        MinMax(ref p1, ref p2); MinMax(ref p4, ref p5); MinMax(ref p7, ref p8);
        MinMax(ref p0, ref p3); MinMax(ref p5, ref p8); MinMax(ref p4, ref p7);
        MinMax(ref p3, ref p6); MinMax(ref p1, ref p4); MinMax(ref p2, ref p5);
        MinMax(ref p4, ref p7); MinMax(ref p4, ref p2); MinMax(ref p6, ref p4);
        MinMax(ref p4, ref p2);
    }

    private static void Median5Row<T>(
        ReadOnlySpan<T> row0, ReadOnlySpan<T> row1, ReadOnlySpan<T> row2, ReadOnlySpan<T> row3,
        ReadOnlySpan<T> row4, Span<T> d, int cn)
        where T : unmanaged
    {
        int n = d.Length;
        int edge = Math.Min(2 * cn, n);
        int j = 0;
        for (; j < edge; j++)
        {
            d[j] = Median5At(row0, row1, row2, row3, row4, j, cn, n);
        }

        if (MinMaxOps.IsVectorized<T>())
        {
            int vc = Vector<T>.Count;
            ref T s0 = ref MemoryMarshal.GetReference(row0);
            ref T s1 = ref MemoryMarshal.GetReference(row1);
            ref T s2 = ref MemoryMarshal.GetReference(row2);
            ref T s3 = ref MemoryMarshal.GetReference(row3);
            ref T s4 = ref MemoryMarshal.GetReference(row4);
            ref T dr = ref MemoryMarshal.GetReference(d);
            int last = n - (2 * cn) - vc;
            for (; j <= last + vc - 1 && last >= edge; j += vc)
            {
                // The final vector overlaps the previous one; it rewrites equal values.
                j = Math.Min(j, last);
                Median25(ref s0, ref s1, ref s2, ref s3, ref s4, (nuint)j, (nuint)cn).StoreUnsafe(ref dr, (nuint)j);
            }
        }

        for (; j < n; j++)
        {
            d[j] = Median5At(row0, row1, row2, row3, row4, j, cn, n);
        }
    }

    private static T Median5At<T>(
        ReadOnlySpan<T> row0, ReadOnlySpan<T> row1, ReadOnlySpan<T> row2, ReadOnlySpan<T> row3,
        ReadOnlySpan<T> row4, int j, int cn, int n)
        where T : unmanaged
    {
        int j1 = j >= cn ? j - cn : j;
        int j0 = j >= cn * 2 ? j - (cn * 2) : j1;
        int j3 = j < n - cn ? j + cn : j;
        int j4 = j < n - (cn * 2) ? j + (cn * 2) : j3;
        return Median25(row0, row1, row2, row3, row4, j0, j1, j, j3, j4);
    }

    /// <summary>
    /// Median of a 5x5 window by the medianBlur_SortNet 5x5 network (median_blur.simd.hpp:777),
    /// one window per vector lane. <paramref name="s0"/>..<paramref name="s4"/> reference the
    /// window rows; <paramref name="j"/> is the centre element offset.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<T> Median25<T>(ref T s0, ref T s1, ref T s2, ref T s3, ref T s4, nuint j, nuint cn)
        where T : unmanaged
    {
        nuint a = j - (2 * cn), b = j - cn, d = j + cn, e = j + (2 * cn);
        Vector<T> p0 = Vector.LoadUnsafe(ref s0, a), p1 = Vector.LoadUnsafe(ref s0, b), p2 = Vector.LoadUnsafe(ref s0, j), p3 = Vector.LoadUnsafe(ref s0, d), p4 = Vector.LoadUnsafe(ref s0, e);
        Vector<T> p5 = Vector.LoadUnsafe(ref s1, a), p6 = Vector.LoadUnsafe(ref s1, b), p7 = Vector.LoadUnsafe(ref s1, j), p8 = Vector.LoadUnsafe(ref s1, d), p9 = Vector.LoadUnsafe(ref s1, e);
        Vector<T> p10 = Vector.LoadUnsafe(ref s2, a), p11 = Vector.LoadUnsafe(ref s2, b), p12 = Vector.LoadUnsafe(ref s2, j), p13 = Vector.LoadUnsafe(ref s2, d), p14 = Vector.LoadUnsafe(ref s2, e);
        Vector<T> p15 = Vector.LoadUnsafe(ref s3, a), p16 = Vector.LoadUnsafe(ref s3, b), p17 = Vector.LoadUnsafe(ref s3, j), p18 = Vector.LoadUnsafe(ref s3, d), p19 = Vector.LoadUnsafe(ref s3, e);
        Vector<T> p20 = Vector.LoadUnsafe(ref s4, a), p21 = Vector.LoadUnsafe(ref s4, b), p22 = Vector.LoadUnsafe(ref s4, j), p23 = Vector.LoadUnsafe(ref s4, d), p24 = Vector.LoadUnsafe(ref s4, e);
        MinMax(ref p1, ref p2); MinMax(ref p0, ref p1); MinMax(ref p1, ref p2); MinMax(ref p4, ref p5); MinMax(ref p3, ref p4);
        MinMax(ref p4, ref p5); MinMax(ref p0, ref p3); MinMax(ref p2, ref p5); MinMax(ref p2, ref p3); MinMax(ref p1, ref p4);
        MinMax(ref p1, ref p2); MinMax(ref p3, ref p4); MinMax(ref p7, ref p8); MinMax(ref p6, ref p7); MinMax(ref p7, ref p8);
        MinMax(ref p10, ref p11); MinMax(ref p9, ref p10); MinMax(ref p10, ref p11); MinMax(ref p6, ref p9); MinMax(ref p8, ref p11);
        MinMax(ref p8, ref p9); MinMax(ref p7, ref p10); MinMax(ref p7, ref p8); MinMax(ref p9, ref p10); MinMax(ref p0, ref p6);
        MinMax(ref p4, ref p10); MinMax(ref p4, ref p6); MinMax(ref p2, ref p8); MinMax(ref p2, ref p4); MinMax(ref p6, ref p8);
        MinMax(ref p1, ref p7); MinMax(ref p5, ref p11); MinMax(ref p5, ref p7); MinMax(ref p3, ref p9); MinMax(ref p3, ref p5);
        MinMax(ref p7, ref p9); MinMax(ref p1, ref p2); MinMax(ref p3, ref p4); MinMax(ref p5, ref p6); MinMax(ref p7, ref p8);
        MinMax(ref p9, ref p10); MinMax(ref p13, ref p14); MinMax(ref p12, ref p13); MinMax(ref p13, ref p14); MinMax(ref p16, ref p17);
        MinMax(ref p15, ref p16); MinMax(ref p16, ref p17); MinMax(ref p12, ref p15); MinMax(ref p14, ref p17); MinMax(ref p14, ref p15);
        MinMax(ref p13, ref p16); MinMax(ref p13, ref p14); MinMax(ref p15, ref p16); MinMax(ref p19, ref p20); MinMax(ref p18, ref p19);
        MinMax(ref p19, ref p20); MinMax(ref p21, ref p22); MinMax(ref p23, ref p24); MinMax(ref p21, ref p23); MinMax(ref p22, ref p24);
        MinMax(ref p22, ref p23); MinMax(ref p18, ref p21); MinMax(ref p20, ref p23); MinMax(ref p20, ref p21); MinMax(ref p19, ref p22);
        MinMax(ref p22, ref p24); MinMax(ref p19, ref p20); MinMax(ref p21, ref p22); MinMax(ref p23, ref p24); MinMax(ref p12, ref p18);
        MinMax(ref p16, ref p22); MinMax(ref p16, ref p18); MinMax(ref p14, ref p20); MinMax(ref p20, ref p24); MinMax(ref p14, ref p16);
        MinMax(ref p18, ref p20); MinMax(ref p22, ref p24); MinMax(ref p13, ref p19); MinMax(ref p17, ref p23); MinMax(ref p17, ref p19);
        MinMax(ref p15, ref p21); MinMax(ref p15, ref p17); MinMax(ref p19, ref p21); MinMax(ref p13, ref p14); MinMax(ref p15, ref p16);
        MinMax(ref p17, ref p18); MinMax(ref p19, ref p20); MinMax(ref p21, ref p22); MinMax(ref p23, ref p24); MinMax(ref p0, ref p12);
        MinMax(ref p8, ref p20); MinMax(ref p8, ref p12); MinMax(ref p4, ref p16); MinMax(ref p16, ref p24); MinMax(ref p12, ref p16);
        MinMax(ref p2, ref p14); MinMax(ref p10, ref p22); MinMax(ref p10, ref p14); MinMax(ref p6, ref p18); MinMax(ref p6, ref p10);
        MinMax(ref p10, ref p12); MinMax(ref p1, ref p13); MinMax(ref p9, ref p21); MinMax(ref p9, ref p13); MinMax(ref p5, ref p17);
        MinMax(ref p13, ref p17); MinMax(ref p3, ref p15); MinMax(ref p11, ref p23); MinMax(ref p11, ref p15); MinMax(ref p7, ref p19);
        MinMax(ref p7, ref p11); MinMax(ref p11, ref p13); MinMax(ref p11, ref p12);
        return p12;
    }

    /// <summary>Scalar <see cref="Median25{T}(ref T, ref T, ref T, ref T, ref T, nuint, nuint)"/> with explicit column indices.</summary>
    private static T Median25<T>(
        ReadOnlySpan<T> r0, ReadOnlySpan<T> r1, ReadOnlySpan<T> r2, ReadOnlySpan<T> r3, ReadOnlySpan<T> r4,
        int j0, int j1, int j, int j3, int j4)
        where T : unmanaged
    {
        T p0 = r0[j0], p1 = r0[j1], p2 = r0[j], p3 = r0[j3], p4 = r0[j4];
        T p5 = r1[j0], p6 = r1[j1], p7 = r1[j], p8 = r1[j3], p9 = r1[j4];
        T p10 = r2[j0], p11 = r2[j1], p12 = r2[j], p13 = r2[j3], p14 = r2[j4];
        T p15 = r3[j0], p16 = r3[j1], p17 = r3[j], p18 = r3[j3], p19 = r3[j4];
        T p20 = r4[j0], p21 = r4[j1], p22 = r4[j], p23 = r4[j3], p24 = r4[j4];
        MinMax(ref p1, ref p2); MinMax(ref p0, ref p1); MinMax(ref p1, ref p2); MinMax(ref p4, ref p5); MinMax(ref p3, ref p4);
        MinMax(ref p4, ref p5); MinMax(ref p0, ref p3); MinMax(ref p2, ref p5); MinMax(ref p2, ref p3); MinMax(ref p1, ref p4);
        MinMax(ref p1, ref p2); MinMax(ref p3, ref p4); MinMax(ref p7, ref p8); MinMax(ref p6, ref p7); MinMax(ref p7, ref p8);
        MinMax(ref p10, ref p11); MinMax(ref p9, ref p10); MinMax(ref p10, ref p11); MinMax(ref p6, ref p9); MinMax(ref p8, ref p11);
        MinMax(ref p8, ref p9); MinMax(ref p7, ref p10); MinMax(ref p7, ref p8); MinMax(ref p9, ref p10); MinMax(ref p0, ref p6);
        MinMax(ref p4, ref p10); MinMax(ref p4, ref p6); MinMax(ref p2, ref p8); MinMax(ref p2, ref p4); MinMax(ref p6, ref p8);
        MinMax(ref p1, ref p7); MinMax(ref p5, ref p11); MinMax(ref p5, ref p7); MinMax(ref p3, ref p9); MinMax(ref p3, ref p5);
        MinMax(ref p7, ref p9); MinMax(ref p1, ref p2); MinMax(ref p3, ref p4); MinMax(ref p5, ref p6); MinMax(ref p7, ref p8);
        MinMax(ref p9, ref p10); MinMax(ref p13, ref p14); MinMax(ref p12, ref p13); MinMax(ref p13, ref p14); MinMax(ref p16, ref p17);
        MinMax(ref p15, ref p16); MinMax(ref p16, ref p17); MinMax(ref p12, ref p15); MinMax(ref p14, ref p17); MinMax(ref p14, ref p15);
        MinMax(ref p13, ref p16); MinMax(ref p13, ref p14); MinMax(ref p15, ref p16); MinMax(ref p19, ref p20); MinMax(ref p18, ref p19);
        MinMax(ref p19, ref p20); MinMax(ref p21, ref p22); MinMax(ref p23, ref p24); MinMax(ref p21, ref p23); MinMax(ref p22, ref p24);
        MinMax(ref p22, ref p23); MinMax(ref p18, ref p21); MinMax(ref p20, ref p23); MinMax(ref p20, ref p21); MinMax(ref p19, ref p22);
        MinMax(ref p22, ref p24); MinMax(ref p19, ref p20); MinMax(ref p21, ref p22); MinMax(ref p23, ref p24); MinMax(ref p12, ref p18);
        MinMax(ref p16, ref p22); MinMax(ref p16, ref p18); MinMax(ref p14, ref p20); MinMax(ref p20, ref p24); MinMax(ref p14, ref p16);
        MinMax(ref p18, ref p20); MinMax(ref p22, ref p24); MinMax(ref p13, ref p19); MinMax(ref p17, ref p23); MinMax(ref p17, ref p19);
        MinMax(ref p15, ref p21); MinMax(ref p15, ref p17); MinMax(ref p19, ref p21); MinMax(ref p13, ref p14); MinMax(ref p15, ref p16);
        MinMax(ref p17, ref p18); MinMax(ref p19, ref p20); MinMax(ref p21, ref p22); MinMax(ref p23, ref p24); MinMax(ref p0, ref p12);
        MinMax(ref p8, ref p20); MinMax(ref p8, ref p12); MinMax(ref p4, ref p16); MinMax(ref p16, ref p24); MinMax(ref p12, ref p16);
        MinMax(ref p2, ref p14); MinMax(ref p10, ref p22); MinMax(ref p10, ref p14); MinMax(ref p6, ref p18); MinMax(ref p6, ref p10);
        MinMax(ref p10, ref p12); MinMax(ref p1, ref p13); MinMax(ref p9, ref p21); MinMax(ref p9, ref p13); MinMax(ref p5, ref p17);
        MinMax(ref p13, ref p17); MinMax(ref p3, ref p15); MinMax(ref p11, ref p23); MinMax(ref p11, ref p15); MinMax(ref p7, ref p19);
        MinMax(ref p7, ref p11); MinMax(ref p11, ref p13); MinMax(ref p11, ref p12);
        return p12;
    }

    /// <summary>The single-row/column path of medianBlur_SortNet.</summary>
    private static void SortNet1D<T>(Mat<T> source, Mat<T> dst, int ksize)
        where T : unmanaged
    {
        int cn = source.Channels;
        bool horizontal = source.Height == 1;
        int len = source.Width + source.Height - 1;
        var p = new T[ksize];
        (int A, int B)[] net = ksize == 3 ? Net3 : Net5;
        int medianIndex = ksize == 3 ? 1 : 2;

        for (int i = 0; i < len; i++)
        {
            for (int j = 0; j < cn; j++)
            {
                if (ksize == 3)
                {
                    p[0] = Element(source, horizontal, i > 0 ? i - 1 : i, j, cn);
                    p[1] = Element(source, horizontal, i, j, cn);
                    p[2] = Element(source, horizontal, i < len - 1 ? i + 1 : i, j, cn);
                }
                else
                {
                    int i1 = i > 0 ? i - 1 : i;
                    int i0 = i > 1 ? i - 2 : i1;
                    int i3 = i < len - 1 ? i + 1 : i;
                    int i4 = i < len - 2 ? i + 2 : i3;
                    p[0] = Element(source, horizontal, i0, j, cn);
                    p[1] = Element(source, horizontal, i1, j, cn);
                    p[2] = Element(source, horizontal, i, j, cn);
                    p[3] = Element(source, horizontal, i3, j, cn);
                    p[4] = Element(source, horizontal, i4, j, cn);
                }

                RunNetwork(p, net);

                if (horizontal)
                {
                    dst.GetRowSpan(0)[(i * cn) + j] = p[medianIndex];
                }
                else
                {
                    dst.GetRowSpan(i)[j] = p[medianIndex];
                }
            }
        }
    }

    private static T Element<T>(Mat<T> source, bool horizontal, int index, int channel, int cn)
        where T : unmanaged
        => horizontal
            ? source.GetReadOnlyRowSpan(0)[(index * cn) + channel]
            : source.GetReadOnlyRowSpan(index)[channel];

    private static void RunNetwork<T>(T[] p, (int A, int B)[] net)
        where T : unmanaged
    {
        for (int k = 0; k < net.Length; k++)
        {
            (int a, int b) = net[k];
            MinMax(ref p[a], ref p[b]);
        }
    }

    /// <summary>
    /// OpenCV's MinMax op: <c>t = a; a = std::min(a, b); b = std::max(b, t)</c>
    /// with raw comparison semantics (NaN-sensitive for float).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void MinMax<T>(ref T a, ref T b)
        where T : unmanaged
    {
        T t = a;
        a = MinOp.Apply(a, b);
        b = MaxOp.Apply(b, t);
    }

    /// <inheritdoc cref="MinMax{T}(ref T, ref T)"/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void MinMax<T>(ref Vector<T> a, ref Vector<T> b)
        where T : unmanaged
    {
        Vector<T> t = a;
        a = MinOp.Apply(a, b);
        b = MaxOp.Apply(b, t);
    }

    /// <summary>
    /// Constant-time histogram median (Perreault and Hébert, as OpenCV <c>medianBlur_8u_O1</c>)
    /// for byte images with ksize >= 7. Each padded column keeps a 16-bin coarse and a 256-bin
    /// fine histogram of its <c>ksize</c> rows. The window histogram adds the incoming column and
    /// subtracts the outgoing one; a fine segment is brought up to date only when the median
    /// search needs it.
    /// </summary>
    public static Mat<byte> Histogram(Mat<byte> source, int ksize)
    {
        int r = ksize / 2;
        using PooledMat<byte> pooled = Borders.CopyMakeBorderPooled(source, r, r, r, r, BorderType.Replicate);
        Mat<byte> padded = pooled.Mat;

        int cn = source.Channels;
        int width = source.Width;
        int height = source.Height;
        var dst = new Mat<byte>(width, height, cn, clear: false);

        // The median is the first bin whose cumulative count exceeds (ksize² - 1) / 2.
        int t = (2 * r * r) + (2 * r);

        ParallelRows.For(height, (long)width * cn * 64, (y0, y1) => HistogramRows(padded, dst, ksize, t, y0, y1));
        return dst;
    }

    private static void HistogramRows(Mat<byte> padded, Mat<byte> dst, int ksize, int t, int y0, int y1)
    {
        int cn = dst.Channels;
        int width = dst.Width;
        int stride = padded.Stride;
        int stripeWidth = Math.Min(width, HistogramStripeWidth);
        int stripeColumns = stripeWidth + ksize - 1;
        ushort[] columnCoarse = ArrayPool<ushort>.Shared.Rent(stripeColumns * 16);
        ushort[] columnFine = ArrayPool<ushort>.Shared.Rent(stripeColumns * 256);
        ushort[] window = ArrayPool<ushort>.Shared.Rent(16 + 256);
        int[] lastUpdate = ArrayPool<int>.Shared.Rent(16);
        try
        {
            ref byte src = ref MemoryMarshal.GetReference(padded.GetReadOnlySpan());
            ref ushort colCoarse = ref columnCoarse[0];
            ref ushort colFine = ref columnFine[0];
            ref ushort coarse = ref window[0];
            ref ushort fine = ref window[16];

            for (int x0 = 0; x0 < width; x0 += stripeWidth)
            {
                int sw = Math.Min(stripeWidth, width - x0);
                int columns = sw + ksize - 1;
                for (int c = 0; c < cn; c++)
                {
                    ref byte stripe = ref Unsafe.Add(ref src, (x0 * cn) + c);
                    Array.Clear(columnCoarse, 0, columns * 16);
                    Array.Clear(columnFine, 0, columns * 256);
                    for (int py = y0; py < y0 + ksize - 1; py++)
                    {
                        AddRowToColumns(ref Unsafe.Add(ref stripe, py * stride), ref colCoarse, ref colFine, columns, cn);
                    }

                    for (int y = y0; y < y1; y++)
                    {
                        AddRowToColumns(
                            ref Unsafe.Add(ref stripe, (y + ksize - 1) * stride), ref colCoarse, ref colFine, columns, cn);
                        if (y > y0)
                        {
                            RemoveRowFromColumns(
                                ref Unsafe.Add(ref stripe, (y - 1) * stride), ref colCoarse, ref colFine, columns, cn);
                        }

                        new Span<ushort>(window, 0, 16).Clear();
                        for (int x = 0; x < ksize; x++)
                        {
                            Add16(ref coarse, ref Unsafe.Add(ref colCoarse, x * 16));
                        }

                        lastUpdate.AsSpan(0, 16).Fill(int.MinValue / 2);

                        ref byte d = ref Unsafe.Add(ref MemoryMarshal.GetReference(dst.GetRowSpan(y)), (x0 * cn) + c);
                        for (int x = 0; ; x++)
                        {
                            Unsafe.Add(ref d, x * cn) = FindMedian(ref coarse, ref fine, ref colFine, lastUpdate, x, ksize, t);
                            if (x == sw - 1)
                            {
                                break;
                            }

                            Add16(ref coarse, ref Unsafe.Add(ref colCoarse, (x + ksize) * 16));
                            Subtract16(ref coarse, ref Unsafe.Add(ref colCoarse, x * 16));
                        }
                    }
                }
            }
        }
        finally
        {
            ArrayPool<ushort>.Shared.Return(columnCoarse);
            ArrayPool<ushort>.Shared.Return(columnFine);
            ArrayPool<ushort>.Shared.Return(window);
            ArrayPool<int>.Shared.Return(lastUpdate);
        }
    }

    /// <summary>Output columns per stripe; keeps the column histograms in L2 cache.</summary>
    private const int HistogramStripeWidth = 256;

    private static void AddRowToColumns(ref byte row, ref ushort colCoarse, ref ushort colFine, int paddedWidth, int cn)
    {
        for (int x = 0; x < paddedWidth; x++)
        {
            int v = Unsafe.Add(ref row, x * cn);
            Unsafe.Add(ref colCoarse, (x * 16) + (v >> 4))++;
            Unsafe.Add(ref colFine, (x * 256) + v)++;
        }
    }

    private static void RemoveRowFromColumns(ref byte row, ref ushort colCoarse, ref ushort colFine, int paddedWidth, int cn)
    {
        for (int x = 0; x < paddedWidth; x++)
        {
            int v = Unsafe.Add(ref row, x * cn);
            Unsafe.Add(ref colCoarse, (x * 16) + (v >> 4))--;
            Unsafe.Add(ref colFine, (x * 256) + v)--;
        }
    }

    /// <summary>
    /// Finds the median of the window at output column <paramref name="x"/> (padded columns
    /// <c>x .. x + ksize - 1</c>). <paramref name="lastUpdate"/> holds, per coarse bin, the
    /// output column for which that fine segment is current.
    /// </summary>
    private static byte FindMedian(
        ref ushort coarse, ref ushort fine, ref ushort colFine, int[] lastUpdate, int x, int ksize, int t)
    {
        int k = FirstCumulativeAbove(ref coarse, t, out int s);

        ref ushort segment = ref Unsafe.Add(ref fine, k * 16);
        ref ushort columns = ref Unsafe.Add(ref colFine, k * 16);
        int last = lastUpdate[k];
        if (x - last >= ksize)
        {
            Unsafe.InitBlockUnaligned(ref Unsafe.As<ushort, byte>(ref segment), 0, 32);
            for (int j = x; j < x + ksize; j++)
            {
                Add16(ref segment, ref Unsafe.Add(ref columns, j * 256));
            }
        }
        else
        {
            for (int j = last; j < x; j++)
            {
                Add16(ref segment, ref Unsafe.Add(ref columns, (j + ksize) * 256));
                Subtract16(ref segment, ref Unsafe.Add(ref columns, j * 256));
            }
        }

        lastUpdate[k] = x;
        return (byte)((k * 16) + FirstCumulativeAbove(ref segment, t - s, out _));
    }

    /// <summary>
    /// Returns the first index <c>i</c> in a 16-bin histogram where the cumulative count
    /// through <c>i</c> exceeds <paramref name="t"/>, and the cumulative count before <c>i</c>.
    /// The total count must exceed <paramref name="t"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int FirstCumulativeAbove(ref ushort h, int t, out int before)
    {
        if (Vector128.IsHardwareAccelerated)
        {
            Vector128<ushort> lo = Prefix8(Vector128.LoadUnsafe(ref h));
            Vector128<ushort> hi = Prefix8(Vector128.LoadUnsafe(ref h, 8)) + Vector128.Create(lo.GetElement(7));
            var threshold = Vector128.Create((ushort)t);
            uint mask = Vector128.GreaterThan(lo, threshold).ExtractMostSignificantBits()
                | (Vector128.GreaterThan(hi, threshold).ExtractMostSignificantBits() << 8);
            int i = BitOperations.TrailingZeroCount(mask);
            int through = i < 8 ? lo.GetElement(i) : hi.GetElement(i - 8);
            before = through - Unsafe.Add(ref h, i);
            return i;
        }

        int sum = 0;
        for (int i = 0; ; i++)
        {
            int next = sum + Unsafe.Add(ref h, i);
            if (next > t)
            {
                before = sum;
                return i;
            }

            sum = next;
        }
    }

    /// <summary>Inclusive prefix sum of eight lanes.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<ushort> Prefix8(Vector128<ushort> v)
    {
        const ushort Z = ushort.MaxValue;
        v += Vector128.Shuffle(v, Vector128.Create(Z, 0, 1, 2, 3, 4, 5, 6));
        v += Vector128.Shuffle(v, Vector128.Create(Z, Z, 0, 1, 2, 3, 4, 5));
        v += Vector128.Shuffle(v, Vector128.Create(Z, Z, Z, Z, 0, 1, 2, 3));
        return v;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Add16(ref ushort a, ref ushort b)
    {
        if (Vector256.IsHardwareAccelerated)
        {
            (Vector256.LoadUnsafe(ref a) + Vector256.LoadUnsafe(ref b)).StoreUnsafe(ref a);
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            (Vector128.LoadUnsafe(ref a) + Vector128.LoadUnsafe(ref b)).StoreUnsafe(ref a);
            (Vector128.LoadUnsafe(ref a, 8) + Vector128.LoadUnsafe(ref b, 8)).StoreUnsafe(ref a, 8);
        }
        else
        {
            for (int i = 0; i < 16; i++)
            {
                Unsafe.Add(ref a, i) += Unsafe.Add(ref b, i);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Subtract16(ref ushort a, ref ushort b)
    {
        if (Vector256.IsHardwareAccelerated)
        {
            (Vector256.LoadUnsafe(ref a) - Vector256.LoadUnsafe(ref b)).StoreUnsafe(ref a);
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            (Vector128.LoadUnsafe(ref a) - Vector128.LoadUnsafe(ref b)).StoreUnsafe(ref a);
            (Vector128.LoadUnsafe(ref a, 8) - Vector128.LoadUnsafe(ref b, 8)).StoreUnsafe(ref a, 8);
        }
        else
        {
            for (int i = 0; i < 16; i++)
            {
                Unsafe.Add(ref a, i) -= Unsafe.Add(ref b, i);
            }
        }
    }
}
