// This implementation is ported from the OpenCV image pyramid functions.
//
// Original sources:
//   modules/imgproc/src/pyramids.cpp (PyrDownVecH, PyrDownVecV, PyrUpVecV, pyrDown_, pyrUp_)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

/// <summary>
/// Row helpers shared by the pyramid engines: widening of source rows to the work type, the
/// 5-tap horizontal sum at full resolution, pixel decimation and pixel interleave.
/// </summary>
/// <remarks>
/// Work types: <c>byte</c> uses <c>ushort</c> (every pyramid sum of 8-bit data is below 65536),
/// <c>ushort</c> and <c>short</c> use <c>int</c>, <c>float</c> and <c>double</c> use themselves.
/// Integer sums are exact, so their operation order is free. Floating-point sums follow the
/// operation order of the OpenCV scalar or vector code for each element.
/// </remarks>
internal static class PyrRows
{
    /// <summary>Converts a source row to the work type. Floating-point rows are copied.</summary>
    public static void Widen<T, TW>(ReadOnlySpan<T> src, Span<TW> dst)
        where T : unmanaged
        where TW : unmanaged
    {
        if (typeof(T) == typeof(byte))
        {
            WidenByte(MemoryMarshal.Cast<T, byte>(src), MemoryMarshal.Cast<TW, ushort>(dst));
        }
        else if (typeof(T) == typeof(ushort))
        {
            WidenUShort(MemoryMarshal.Cast<T, ushort>(src), MemoryMarshal.Cast<TW, int>(dst));
        }
        else if (typeof(T) == typeof(short))
        {
            WidenShort(MemoryMarshal.Cast<T, short>(src), MemoryMarshal.Cast<TW, int>(dst));
        }
        else
        {
            MemoryMarshal.Cast<T, TW>(src).CopyTo(dst);
        }
    }

    private static void WidenByte(ReadOnlySpan<byte> src, Span<ushort> dst)
    {
        int n = src.Length;
        int i = 0;
        ref byte s = ref MemoryMarshal.GetReference(src);
        ref ushort d = ref MemoryMarshal.GetReference(dst);
        if (Vector.IsHardwareAccelerated)
        {
            int vc = Vector<byte>.Count;
            for (; i <= n - vc; i += vc)
            {
                Vector.Widen(Vector.LoadUnsafe(ref s, (nuint)i), out Vector<ushort> lo, out Vector<ushort> hi);
                lo.StoreUnsafe(ref d, (nuint)i);
                hi.StoreUnsafe(ref d, (nuint)(i + Vector<ushort>.Count));
            }
        }

        for (; i < n; i++)
        {
            Unsafe.Add(ref d, i) = Unsafe.Add(ref s, i);
        }
    }

    private static void WidenUShort(ReadOnlySpan<ushort> src, Span<int> dst)
    {
        int n = src.Length;
        int i = 0;
        ref ushort s = ref MemoryMarshal.GetReference(src);
        ref int d = ref MemoryMarshal.GetReference(dst);
        if (Vector.IsHardwareAccelerated)
        {
            int vc = Vector<ushort>.Count;
            for (; i <= n - vc; i += vc)
            {
                Vector.Widen(Vector.LoadUnsafe(ref s, (nuint)i), out Vector<uint> lo, out Vector<uint> hi);
                Vector.AsVectorInt32(lo).StoreUnsafe(ref d, (nuint)i);
                Vector.AsVectorInt32(hi).StoreUnsafe(ref d, (nuint)(i + Vector<int>.Count));
            }
        }

        for (; i < n; i++)
        {
            Unsafe.Add(ref d, i) = Unsafe.Add(ref s, i);
        }
    }

    private static void WidenShort(ReadOnlySpan<short> src, Span<int> dst)
    {
        int n = src.Length;
        int i = 0;
        ref short s = ref MemoryMarshal.GetReference(src);
        ref int d = ref MemoryMarshal.GetReference(dst);
        if (Vector.IsHardwareAccelerated)
        {
            int vc = Vector<short>.Count;
            for (; i <= n - vc; i += vc)
            {
                Vector.Widen(Vector.LoadUnsafe(ref s, (nuint)i), out Vector<int> lo, out Vector<int> hi);
                lo.StoreUnsafe(ref d, (nuint)i);
                hi.StoreUnsafe(ref d, (nuint)(i + Vector<int>.Count));
            }
        }

        for (; i < n; i++)
        {
            Unsafe.Add(ref d, i) = Unsafe.Add(ref s, i);
        }
    }

    /// <summary>
    /// <c>f[i] = w[i]·6 + ((w[i−cn] + w[i+cn])·4 + (w[i−2cn] + w[i+2cn]))</c> for <c>i</c> in
    /// <c>[start, end)</c>. This is the operation order of OpenCV <c>PyrDownVecH</c>. With
    /// <paramref name="scalarOrder"/>, the order is that of <see cref="DownScalar"/>.
    /// The caller guarantees <c>start ≥ 2cn</c> and <c>end + 2cn ≤ w.Length</c>.
    /// </summary>
    public static void DownTaps<TW>(ReadOnlySpan<TW> w, Span<TW> f, int start, int end, int cn, bool scalarOrder = false)
        where TW : unmanaged, INumberBase<TW>
    {
        ref TW s = ref MemoryMarshal.GetReference(w);
        ref TW d = ref MemoryMarshal.GetReference(f);
        int c1 = cn;
        int c2 = 2 * cn;
        int i = start;
        if (Vector.IsHardwareAccelerated && Vector<TW>.IsSupported)
        {
            int vc = Vector<TW>.Count;
            if (end - start >= vc)
            {
                var six = new Vector<TW>(TW.CreateTruncating(6));
                var four = new Vector<TW>(TW.CreateTruncating(4));
                for (; i < end; i += vc)
                {
                    // The final vector overlaps the previous one; it rewrites equal values.
                    i = Math.Min(i, end - vc);
                    Vector<TW> a = Vector.LoadUnsafe(ref s, (nuint)(i - c2));
                    Vector<TW> b = Vector.LoadUnsafe(ref s, (nuint)(i - c1));
                    Vector<TW> c = Vector.LoadUnsafe(ref s, (nuint)i);
                    Vector<TW> e = Vector.LoadUnsafe(ref s, (nuint)(i + c1));
                    Vector<TW> g = Vector.LoadUnsafe(ref s, (nuint)(i + c2));
                    Vector<TW> r = scalarOrder ? ((((c * six) + ((b + e) * four)) + a) + g) : ((c * six) + (((b + e) * four) + (a + g)));
                    r.StoreUnsafe(ref d, (nuint)i);
                }

                return;
            }
        }

        TW six1 = TW.CreateTruncating(6);
        TW four1 = TW.CreateTruncating(4);
        if (scalarOrder)
        {
            for (; i < end; i++)
            {
                Unsafe.Add(ref d, i) = (((Unsafe.Add(ref s, i) * six1) + ((Unsafe.Add(ref s, i - c1) + Unsafe.Add(ref s, i + c1)) * four1))
                    + Unsafe.Add(ref s, i - c2)) + Unsafe.Add(ref s, i + c2);
            }

            return;
        }

        for (; i < end; i++)
        {
            Unsafe.Add(ref d, i) = (Unsafe.Add(ref s, i) * six1)
                + (((Unsafe.Add(ref s, i - c1) + Unsafe.Add(ref s, i + c1)) * four1)
                + (Unsafe.Add(ref s, i - c2) + Unsafe.Add(ref s, i + c2)));
        }
    }

    /// <summary>
    /// OpenCV scalar order <c>((s0·6 + (s−1 + s+1)·4) + s−2) + s+2</c>, with the five source
    /// elements at the given indices.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static TW DownScalar<TW>(ReadOnlySpan<TW> w, int m2, int m1, int c, int p1, int p2)
        where TW : unmanaged, INumberBase<TW>
        => (((w[c] * TW.CreateTruncating(6)) + ((w[m1] + w[p1]) * TW.CreateTruncating(4))) + w[m2]) + w[p2];

    /// <summary>
    /// <c>row[x] = f[2x − x mod cn]</c> for <c>x</c> in <c>[start, end)</c>; both bounds are
    /// multiples of <paramref name="cn"/>.
    /// </summary>
    public static void Decimate<TW>(ReadOnlySpan<TW> f, Span<TW> row, int start, int end, int cn)
        where TW : unmanaged
    {
        ref TW s = ref MemoryMarshal.GetReference(f);
        ref TW d = ref MemoryMarshal.GetReference(row);
        int x = start;
        if (cn == 1 && Vector.IsHardwareAccelerated)
        {
            if (Unsafe.SizeOf<TW>() == 2)
            {
                ref uint s32 = ref Unsafe.As<TW, uint>(ref s);
                ref ushort d16 = ref Unsafe.As<TW, ushort>(ref d);
                int vc = Vector<ushort>.Count;
                for (; x <= end - vc; x += vc)
                {
                    Vector<uint> lo = Vector.LoadUnsafe(ref s32, (nuint)x);
                    Vector<uint> hi = Vector.LoadUnsafe(ref s32, (nuint)(x + Vector<uint>.Count));
                    Vector.Narrow(lo, hi).StoreUnsafe(ref d16, (nuint)x);
                }
            }
            else if (Unsafe.SizeOf<TW>() == 4)
            {
                ref ulong s64 = ref Unsafe.As<TW, ulong>(ref s);
                ref uint d32 = ref Unsafe.As<TW, uint>(ref d);
                int vc = Vector<uint>.Count;
                for (; x <= end - vc; x += vc)
                {
                    Vector<ulong> lo = Vector.LoadUnsafe(ref s64, (nuint)x);
                    Vector<ulong> hi = Vector.LoadUnsafe(ref s64, (nuint)(x + Vector<ulong>.Count));
                    Vector.Narrow(lo, hi).StoreUnsafe(ref d32, (nuint)x);
                }
            }

            for (; x < end; x++)
            {
                Unsafe.Add(ref d, x) = Unsafe.Add(ref s, 2 * x);
            }

            return;
        }

        if (cn == 2 && Unsafe.SizeOf<TW>() == 2 && Vector.IsHardwareAccelerated)
        {
            ref ulong s64 = ref Unsafe.As<TW, ulong>(ref s);
            ref uint d32 = ref Unsafe.As<TW, uint>(ref d);
            int vc = Vector<uint>.Count;
            int p = start / 2;
            int pend = end / 2;
            for (; p <= pend - vc; p += vc)
            {
                Vector<ulong> lo = Vector.LoadUnsafe(ref s64, (nuint)p);
                Vector<ulong> hi = Vector.LoadUnsafe(ref s64, (nuint)(p + Vector<ulong>.Count));
                Vector.Narrow(lo, hi).StoreUnsafe(ref d32, (nuint)p);
            }

            x = p * 2;
        }

        if (x < end)
        {
            int bpp = cn * Unsafe.SizeOf<TW>();
            ref byte sb = ref Unsafe.As<TW, byte>(ref Unsafe.Add(ref s, 2 * x));
            ref byte db = ref Unsafe.As<TW, byte>(ref Unsafe.Add(ref d, x));
            int pixels = (end - x) / cn;
            switch (bpp)
            {
                case 2: MovePixels<ushort>(ref sb, ref db, pixels, 2, 1); break;
                case 4: MovePixels<uint>(ref sb, ref db, pixels, 2, 1); break;
                case 6: MovePixels<Pixel6>(ref sb, ref db, pixels, 2, 1); break;
                case 8: MovePixels<ulong>(ref sb, ref db, pixels, 2, 1); break;
                case 12: MovePixels<Pixel12>(ref sb, ref db, pixels, 2, 1); break;
                case 16: MovePixels<Pixel16>(ref sb, ref db, pixels, 2, 1); break;
                case 24: MovePixels<Pixel24>(ref sb, ref db, pixels, 2, 1); break;
                case 32: MovePixels<Pixel32>(ref sb, ref db, pixels, 2, 1); break;
                default:
                    for (; x < end; x += cn)
                    {
                        ref TW src = ref Unsafe.Add(ref s, 2 * x);
                        ref TW dst = ref Unsafe.Add(ref d, x);
                        for (int c = 0; c < cn; c++)
                        {
                            Unsafe.Add(ref dst, c) = Unsafe.Add(ref src, c);
                        }
                    }

                    break;
            }
        }
    }

    /// <summary>
    /// <c>dst[p · dstStep] = src[p · srcStep]</c> for <paramref name="pixels"/> pixels of
    /// <c>sizeof(TP)</c> bytes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void MovePixels<TP>(ref byte src, ref byte dst, int pixels, int srcStep, int dstStep)
        where TP : unmanaged
    {
        int size = Unsafe.SizeOf<TP>();
        nint ss = srcStep * size;
        nint ds = dstStep * size;
        for (int p = 0; p < pixels; p++)
        {
            Unsafe.WriteUnaligned(ref Unsafe.Add(ref dst, p * ds), Unsafe.ReadUnaligned<TP>(ref Unsafe.Add(ref src, p * ss)));
        }
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    private struct Pixel6
    {
        public uint A;
        public ushort B;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    private struct Pixel12
    {
        public ulong A;
        public uint B;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    private struct Pixel16
    {
        public ulong A;
        public ulong B;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    private struct Pixel24
    {
        public ulong A;
        public ulong B;
        public ulong C;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    private struct Pixel32
    {
        public ulong A;
        public ulong B;
        public ulong C;
        public ulong D;
    }

    /// <summary>
    /// <c>row[2x − x mod cn] = t0[x]</c> and <c>row[2x − x mod cn + cn] = t1[x]</c> for
    /// <c>x</c> in <c>[start, end)</c>; both bounds are multiples of <paramref name="cn"/>.
    /// </summary>
    public static void Interleave<TW>(ReadOnlySpan<TW> t0, ReadOnlySpan<TW> t1, Span<TW> row, int start, int end, int cn)
        where TW : unmanaged
    {
        ref TW a = ref MemoryMarshal.GetReference(t0);
        ref TW b = ref MemoryMarshal.GetReference(t1);
        ref TW d = ref MemoryMarshal.GetReference(row);
        int x = start;
        if (cn == 1 && Vector.IsHardwareAccelerated)
        {
            if (Unsafe.SizeOf<TW>() == 2)
            {
                ref ushort a16 = ref Unsafe.As<TW, ushort>(ref a);
                ref ushort b16 = ref Unsafe.As<TW, ushort>(ref b);
                ref uint d32 = ref Unsafe.As<TW, uint>(ref d);
                int vc = Vector<ushort>.Count;
                for (; x <= end - vc; x += vc)
                {
                    Vector.Widen(Vector.LoadUnsafe(ref a16, (nuint)x), out Vector<uint> a0, out Vector<uint> a1);
                    Vector.Widen(Vector.LoadUnsafe(ref b16, (nuint)x), out Vector<uint> b0, out Vector<uint> b1);
                    (a0 | (b0 << 16)).StoreUnsafe(ref d32, (nuint)x);
                    (a1 | (b1 << 16)).StoreUnsafe(ref d32, (nuint)(x + Vector<uint>.Count));
                }
            }
            else if (Unsafe.SizeOf<TW>() == 4)
            {
                ref uint a32 = ref Unsafe.As<TW, uint>(ref a);
                ref uint b32 = ref Unsafe.As<TW, uint>(ref b);
                ref ulong d64 = ref Unsafe.As<TW, ulong>(ref d);
                int vc = Vector<uint>.Count;
                for (; x <= end - vc; x += vc)
                {
                    Vector.Widen(Vector.LoadUnsafe(ref a32, (nuint)x), out Vector<ulong> a0, out Vector<ulong> a1);
                    Vector.Widen(Vector.LoadUnsafe(ref b32, (nuint)x), out Vector<ulong> b0, out Vector<ulong> b1);
                    (a0 | (b0 << 32)).StoreUnsafe(ref d64, (nuint)x);
                    (a1 | (b1 << 32)).StoreUnsafe(ref d64, (nuint)(x + Vector<ulong>.Count));
                }
            }

            for (; x < end; x++)
            {
                Unsafe.Add(ref d, 2 * x) = Unsafe.Add(ref a, x);
                Unsafe.Add(ref d, (2 * x) + 1) = Unsafe.Add(ref b, x);
            }

            return;
        }

        if (x >= end)
        {
            return;
        }

        int bpp = cn * Unsafe.SizeOf<TW>();
        ref byte ab = ref Unsafe.As<TW, byte>(ref Unsafe.Add(ref a, x));
        ref byte bb = ref Unsafe.As<TW, byte>(ref Unsafe.Add(ref b, x));
        ref byte d0b = ref Unsafe.As<TW, byte>(ref Unsafe.Add(ref d, 2 * x));
        ref byte d1b = ref Unsafe.Add(ref d0b, bpp);
        int pixels = (end - x) / cn;
        switch (bpp)
        {
            case 2: MovePixels<ushort>(ref ab, ref d0b, pixels, 1, 2); MovePixels<ushort>(ref bb, ref d1b, pixels, 1, 2); break;
            case 4: MovePixels<uint>(ref ab, ref d0b, pixels, 1, 2); MovePixels<uint>(ref bb, ref d1b, pixels, 1, 2); break;
            case 6: MovePixels<Pixel6>(ref ab, ref d0b, pixels, 1, 2); MovePixels<Pixel6>(ref bb, ref d1b, pixels, 1, 2); break;
            case 8: MovePixels<ulong>(ref ab, ref d0b, pixels, 1, 2); MovePixels<ulong>(ref bb, ref d1b, pixels, 1, 2); break;
            case 12: MovePixels<Pixel12>(ref ab, ref d0b, pixels, 1, 2); MovePixels<Pixel12>(ref bb, ref d1b, pixels, 1, 2); break;
            case 16: MovePixels<Pixel16>(ref ab, ref d0b, pixels, 1, 2); MovePixels<Pixel16>(ref bb, ref d1b, pixels, 1, 2); break;
            case 24: MovePixels<Pixel24>(ref ab, ref d0b, pixels, 1, 2); MovePixels<Pixel24>(ref bb, ref d1b, pixels, 1, 2); break;
            case 32: MovePixels<Pixel32>(ref ab, ref d0b, pixels, 1, 2); MovePixels<Pixel32>(ref bb, ref d1b, pixels, 1, 2); break;
            default:
                for (; x < end; x += cn)
                {
                    ref TW d0 = ref Unsafe.Add(ref d, 2 * x);
                    for (int c = 0; c < cn; c++)
                    {
                        Unsafe.Add(ref d0, c) = Unsafe.Add(ref a, x + c);
                        Unsafe.Add(ref d0, cn + c) = Unsafe.Add(ref b, x + c);
                    }
                }

                break;
        }
    }
}
