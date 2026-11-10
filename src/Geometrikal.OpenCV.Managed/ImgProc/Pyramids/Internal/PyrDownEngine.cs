// This implementation is ported from the OpenCV image pyramid functions.
//
// Original sources:
//   modules/imgproc/src/pyramids.cpp (pyrDown_, PyrDownInvoker, PyrDownVecH, PyrDownVecV)
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

/// <summary>
/// <c>pyrDown</c>: a 5×5 Gaussian <c>[1 4 6 4 1]/16</c> in each direction, then removal of the
/// odd rows and columns. Each stripe of output rows keeps a ring of five horizontally filtered
/// and decimated source rows, as OpenCV <c>PyrDownInvoker</c>.
/// </summary>
internal static class PyrDownEngine
{
    private const int Taps = 5;

    public static void Run<T>(Mat<T> src, Mat<T> dst, BorderType border)
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte))
        {
            Run<T, ushort>(src, dst, border);
        }
        else if (typeof(T) == typeof(ushort) || typeof(T) == typeof(short))
        {
            Run<T, int>(src, dst, border);
        }
        else if (typeof(T) == typeof(float))
        {
            Run<T, float>(src, dst, border);
        }
        else
        {
            Run<T, double>(src, dst, border);
        }
    }

    private static void Run<T, TW>(Mat<T> src, Mat<T> dst, BorderType border)
        where T : unmanaged
        where TW : unmanaged, INumberBase<TW>
    {
        if (ReferenceEquals(src.DangerousGetArray(), dst.DangerousGetArray()))
        {
            src = src.Clone();
        }

        var geom = Geometry.Create<T>(src.Width, dst.Width, src.Channels, border);
        long work = (3L * src.RowLength) + dst.RowLength;
        if (!ParallelRows.IsSerial(dst.Height, work))
        {
            PageTouch.Rows(dst);
        }

        ParallelRows.For(dst.Height, work, (y0, y1) => Stripe<T, TW>(src, dst, geom, border, y0, y1));
    }

    /// <summary>
    /// Horizontal layout of one output row, in elements. Elements below <see cref="Cn"/> use
    /// <see cref="TabL"/>; elements in <c>[Cn, InteriorEnd)</c> read the source row directly; the
    /// remaining elements use <see cref="TabR"/>, as OpenCV <c>pyrDown_</c>.
    /// </summary>
    private sealed record Geometry(int Cn, int SrcLength, int DstLength, int InteriorEnd, int VectorEnd, int VerticalVectorEnd, int[] TabL, int[] TabR)
    {
        public static Geometry Create<T>(int sw, int dw, int cn, BorderType border)
        {
            int width0 = Math.Min(((sw - 3) / 2) + 1, dw);
            var tabL = new int[cn * (Taps + 2)];
            var tabR = new int[cn * (Taps + 2)];
            for (int x = 0; x <= Taps + 1; x++)
            {
                int sx0 = Borders.BorderInterpolate(x - 2, sw, border) * cn;
                int sx1 = Borders.BorderInterpolate(x + (width0 * 2) - 2, sw, border) * cn;
                for (int k = 0; k < cn; k++)
                {
                    tabL[(x * cn) + k] = sx0 + k;
                    tabR[(x * cn) + k] = sx1 + k;
                }
            }

            int interiorEnd = Math.Max(width0 * cn, cn);
            bool floating = typeof(T) == typeof(float) || typeof(T) == typeof(double);
            int vectorEnd = floating ? cn + VectorCount<T>((width0 * cn) - cn, cn) : interiorEnd;
            int verticalVectorEnd = typeof(T) == typeof(float) ? (dw * cn) & ~3 : 0;
            return new Geometry(cn, sw * cn, dw * cn, interiorEnd, vectorEnd, verticalVectorEnd, tabL, tabR);
        }

        /// <summary>
        /// Interior elements that the OpenCV 128-bit <c>PyrDownVecH</c> covers for
        /// <paramref name="w"/> interior elements of a floating-point row.
        /// </summary>
        private static int VectorCount<T>(int w, int cn)
        {
            if (w <= 0)
            {
                return 0;
            }

            if (typeof(T) == typeof(float))
            {
                return cn switch
                {
                    1 => w & ~3,
                    2 => w & ~7,
                    3 => w >= 4 ? ((((w - 4) / 3) + 1) * 3) : 0,
                    4 => w & ~3,
                    _ => 0,
                };
            }

            return cn == 1 ? w & ~1 : 0;
        }
    }

    private static void Stripe<T, TW>(Mat<T> src, Mat<T> dst, Geometry g, BorderType border, int y0, int y1)
        where T : unmanaged
        where TW : unmanaged, INumberBase<TW>
    {
        int sh = src.Height;
        int dstLength = g.DstLength;
        bool widen = typeof(T) != typeof(TW);
        TW[][] ring = new TW[Taps][];
        TW[] wide = widen ? ArrayPool<TW>.Shared.Rent(g.SrcLength) : [];
        TW[] taps = ArrayPool<TW>.Shared.Rent(g.SrcLength);
        bool byte3 = typeof(T) == typeof(byte) && g.Cn == 3 && Avx2.IsSupported;
        byte[] split = byte3 ? ArrayPool<byte>.Shared.Rent(g.SrcLength + 64) : [];
        int[] slots = new int[Taps];
        try
        {
            for (int k = 0; k < Taps; k++)
            {
                ring[k] = ArrayPool<TW>.Shared.Rent(dstLength);
            }

            int sy = (y0 * 2) - 2;
            for (int y = y0; y < y1; y++)
            {
                for (; sy <= (y * 2) + 2; sy++)
                {
                    int bsy = Borders.BorderInterpolate(sy, sh, border);
                    ReadOnlySpan<T> srow = src.GetReadOnlyRowSpan(bsy);
                    ReadOnlySpan<TW> w;
                    if (typeof(T) == typeof(byte) && g.Cn == 1)
                    {
                        HorizontalByte(MemoryMarshal.Cast<T, byte>(srow), MemoryMarshal.Cast<TW, ushort>(ring[(sy + 2) % Taps].AsSpan(0, dstLength)), g);
                        continue;
                    }

                    if (byte3)
                    {
                        HorizontalByte3(MemoryMarshal.Cast<T, byte>(srow), MemoryMarshal.Cast<TW, ushort>(ring[(sy + 2) % Taps].AsSpan(0, dstLength)), split, g);
                        continue;
                    }

                    if (typeof(T) == typeof(float) && g.Cn == 1 && Avx2.IsSupported)
                    {
                        HorizontalFloat(MemoryMarshal.Cast<T, float>(srow), MemoryMarshal.Cast<TW, float>(ring[(sy + 2) % Taps].AsSpan(0, dstLength)), g);
                        continue;
                    }

                    if (widen)
                    {
                        PyrRows.Widen(srow, wide.AsSpan(0, g.SrcLength));
                        w = wide.AsSpan(0, g.SrcLength);
                    }
                    else
                    {
                        w = MemoryMarshal.Cast<T, TW>(srow);
                    }

                    Horizontal(w, ring[(sy + 2) % Taps].AsSpan(0, dstLength), taps, g);
                }

                for (int k = 0; k < Taps; k++)
                {
                    slots[k] = ((y * 2) + k) % Taps;
                }

                PyrColumns.Down(ring, slots, dst.GetRowSpan(y), g.VerticalVectorEnd);
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

            ArrayPool<TW>.Shared.Return(taps);
            if (byte3)
            {
                ArrayPool<byte>.Shared.Return(split);
            }

            if (widen)
            {
                ArrayPool<TW>.Shared.Return(wide);
            }
        }
    }

    /// <summary>
    /// One-channel 8-bit horizontal pass without widening. Each <c>ushort</c> lane of a load at
    /// byte offset <c>2x + k</c> holds source elements <c>2x + k</c> (low byte) and <c>2x + k + 1</c>.
    /// </summary>
    private static void HorizontalByte(ReadOnlySpan<byte> s, Span<ushort> row, Geometry g)
    {
        int[] tabL = g.TabL;
        row[0] = (ushort)((s[tabL[2]] * 6) + ((s[tabL[1]] + s[tabL[3]]) * 4) + s[tabL[0]] + s[tabL[4]]);
        if (g.DstLength == 1)
        {
            return;
        }

        int end = g.InteriorEnd;
        int x = 1;
        ref byte sb = ref MemoryMarshal.GetReference(s);
        ref ushort d = ref MemoryMarshal.GetReference(row);
        if (Vector.IsHardwareAccelerated)
        {
            int vc = Vector<ushort>.Count;
            var low = new Vector<ushort>(0xFF);
            var six = new Vector<ushort>(6);
            for (; x <= end - vc && (2 * x) + 2 + (2 * vc) <= s.Length; x += vc)
            {
                Vector<ushort> u0 = Unsafe.ReadUnaligned<Vector<ushort>>(ref Unsafe.Add(ref sb, (2 * x) - 2));
                Vector<ushort> u1 = Unsafe.ReadUnaligned<Vector<ushort>>(ref Unsafe.Add(ref sb, 2 * x));
                Vector<ushort> u2 = Unsafe.ReadUnaligned<Vector<ushort>>(ref Unsafe.Add(ref sb, (2 * x) + 2));
                Vector<ushort> sum = ((u1 & low) * six) + (((u0 >> 8) + (u1 >> 8)) << 2) + (u0 & low) + (u2 & low);
                sum.StoreUnsafe(ref d, (nuint)x);
            }
        }

        for (; x < end; x++)
        {
            int c = 2 * x;
            Unsafe.Add(ref d, x) = (ushort)((Unsafe.Add(ref sb, c) * 6) + ((Unsafe.Add(ref sb, c - 1) + Unsafe.Add(ref sb, c + 1)) * 4)
                + Unsafe.Add(ref sb, c - 2) + Unsafe.Add(ref sb, c + 2));
        }

        int[] tabR = g.TabR;
        for (int r = 0; x < g.DstLength; x++, r++)
        {
            row[x] = (ushort)((s[tabR[r + 2]] * 6) + ((s[tabR[r + 1]] + s[tabR[r + 3]]) * 4) + s[tabR[r]] + s[tabR[r + 4]]);
        }
    }

    /// <summary>
    /// Three-channel 8-bit horizontal pass. A byte shuffle splits the row into even and odd
    /// pixels; the 5-tap sum then uses element offsets of 0 and ±3 in the two halves.
    /// </summary>
    private static void HorizontalByte3(ReadOnlySpan<byte> s, Span<ushort> row, byte[] split, Geometry g)
    {
        int[] tabL = g.TabL;
        for (int x = 0; x < 3; x++)
        {
            row[x] = (ushort)((s[tabL[x + 6]] * 6) + ((s[tabL[x + 3]] + s[tabL[x + 9]]) * 4) + s[tabL[x]] + s[tabL[x + 12]]);
        }

        if (g.DstLength == 3)
        {
            return;
        }

        int end = g.InteriorEnd;
        int half = (s.Length / 2) + 32;
        ref byte sb = ref MemoryMarshal.GetReference(s);
        ref byte e = ref MemoryMarshal.GetArrayDataReference(split);
        ref byte o = ref Unsafe.Add(ref e, half);

        // Bytes 0..5 of the shuffle result are two even pixels, bytes 6..11 are two odd pixels.
        Vector128<byte> order = Vector128.Create((byte)0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, 0x80, 0x80, 0x80, 0x80);
        int k = 0;
        for (; (12 * k) + 16 <= s.Length; k++)
        {
            Vector128<byte> v = Ssse3.Shuffle(Vector128.LoadUnsafe(ref sb, (nuint)(12 * k)), order);
            Unsafe.WriteUnaligned(ref Unsafe.Add(ref e, 6 * k), v.AsUInt64().ToScalar());
            Unsafe.WriteUnaligned(ref Unsafe.Add(ref o, 6 * k), Sse2.ShiftRightLogical128BitLane(v, 6).AsUInt64().ToScalar());
        }

        for (int i = 12 * k; i < s.Length; i++)
        {
            int pixel = i / 3;
            int dstIndex = (3 * (pixel >> 1)) + (i - (3 * pixel));
            Unsafe.Add(ref (pixel & 1) == 0 ? ref e : ref o, dstIndex) = Unsafe.Add(ref sb, i);
        }

        ref ushort d = ref MemoryMarshal.GetReference(row);
        var six = Vector256.Create((ushort)6);
        int xv = 3;
        for (; xv <= end - 16; xv += 16)
        {
            Vector256<ushort> e0 = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(ref e, (nuint)(xv - 3))).AsUInt16();
            Vector256<ushort> e1 = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(ref e, (nuint)xv)).AsUInt16();
            Vector256<ushort> e2 = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(ref e, (nuint)(xv + 3))).AsUInt16();
            Vector256<ushort> o0 = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(ref o, (nuint)(xv - 3))).AsUInt16();
            Vector256<ushort> o1 = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(ref o, (nuint)xv)).AsUInt16();
            ((e1 * six) + ((o0 + o1) << 2) + e0 + e2).StoreUnsafe(ref d, (nuint)xv);
        }

        for (; xv < end; xv++)
        {
            Unsafe.Add(ref d, xv) = (ushort)((Unsafe.Add(ref e, xv) * 6) + ((Unsafe.Add(ref o, xv - 3) + Unsafe.Add(ref o, xv)) * 4)
                + Unsafe.Add(ref e, xv - 3) + Unsafe.Add(ref e, xv + 3));
        }

        int[] tabR = g.TabR;
        for (int x = end, r = 0; x < g.DstLength; x++, r++)
        {
            row[x] = (ushort)((s[tabR[r + 6]] * 6) + ((s[tabR[r + 3]] + s[tabR[r + 9]]) * 4) + s[tabR[r]] + s[tabR[r + 12]]);
        }
    }

    /// <summary>
    /// One-channel <c>float</c> horizontal pass with even/odd deinterleaving loads, as OpenCV
    /// <c>PyrDownVecH&lt;float, float, 1&gt;</c>.
    /// </summary>
    private static void HorizontalFloat(ReadOnlySpan<float> s, Span<float> row, Geometry g)
    {
        int[] tabL = g.TabL;
        row[0] = PyrRows.DownScalar(s, tabL[0], tabL[1], tabL[2], tabL[3], tabL[4]);
        if (g.DstLength == 1)
        {
            return;
        }

        int end = g.InteriorEnd;
        int vectorEnd = g.VectorEnd;
        ref float sf = ref MemoryMarshal.GetReference(s);
        ref float d = ref MemoryMarshal.GetReference(row);
        var six = Vector256.Create(6f);
        var four = Vector256.Create(4f);
        int x = 1;
        for (; x <= vectorEnd - 8 && (2 * x) + 18 <= s.Length; x += 8)
        {
            Deinterleave(ref Unsafe.Add(ref sf, (2 * x) - 2), out Vector256<float> r0, out Vector256<float> r1);
            Deinterleave(ref Unsafe.Add(ref sf, 2 * x), out Vector256<float> r2, out Vector256<float> r3);
            Deinterleave(ref Unsafe.Add(ref sf, (2 * x) + 2), out Vector256<float> r4, out _);
            ((r2 * six) + (((r1 + r3) * four) + (r0 + r4))).StoreUnsafe(ref d, (nuint)x);
        }

        for (; x < vectorEnd; x++)
        {
            int c = 2 * x;
            Unsafe.Add(ref d, x) = (Unsafe.Add(ref sf, c) * 6f) + (((Unsafe.Add(ref sf, c - 1) + Unsafe.Add(ref sf, c + 1)) * 4f) + (Unsafe.Add(ref sf, c - 2) + Unsafe.Add(ref sf, c + 2)));
        }

        for (; x < end; x++)
        {
            int c = 2 * x;
            Unsafe.Add(ref d, x) = PyrRows.DownScalar(s, c - 2, c - 1, c, c + 1, c + 2);
        }

        int[] tabR = g.TabR;
        for (int r = 0; x < g.DstLength; x++, r++)
        {
            row[x] = PyrRows.DownScalar(s, tabR[r], tabR[r + 1], tabR[r + 2], tabR[r + 3], tabR[r + 4]);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Deinterleave(ref float p, out Vector256<float> even, out Vector256<float> odd)
    {
        Vector256<float> a = Vector256.LoadUnsafe(ref p);
        Vector256<float> b = Vector256.LoadUnsafe(ref p, 8);
        even = Avx2.Permute4x64(Avx.Shuffle(a, b, 0x88).AsDouble(), 0xD8).AsSingle();
        odd = Avx2.Permute4x64(Avx.Shuffle(a, b, 0xDD).AsDouble(), 0xD8).AsSingle();
    }

    private static void Horizontal<TW>(ReadOnlySpan<TW> w, Span<TW> row, TW[] taps, Geometry g)
        where TW : unmanaged, INumberBase<TW>
    {
        int cn = g.Cn;
        int[] tabL = g.TabL;
        for (int x = 0; x < cn; x++)
        {
            row[x] = PyrRows.DownScalar(w, tabL[x], tabL[x + cn], tabL[x + (cn * 2)], tabL[x + (cn * 3)], tabL[x + (cn * 4)]);
        }

        if (g.DstLength == cn)
        {
            return;
        }

        int end = g.InteriorEnd;
        if (end > cn)
        {
            bool scalarOrder = g.VectorEnd == cn;
            PyrRows.DownTaps(w, taps, 2 * cn, (2 * end) - cn, cn, scalarOrder);
            PyrRows.Decimate<TW>(taps, row, cn, end, cn);
            for (int x = scalarOrder ? end : g.VectorEnd; x < end; x++)
            {
                int c = (2 * x) - (x % cn);
                row[x] = PyrRows.DownScalar(w, c - (2 * cn), c - cn, c, c + cn, c + (2 * cn));
            }
        }

        int[] tabR = g.TabR;
        for (int x = end, r = 0; x < g.DstLength; x++, r++)
        {
            row[x] = PyrRows.DownScalar(w, tabR[r], tabR[r + cn], tabR[r + (cn * 2)], tabR[r + (cn * 3)], tabR[r + (cn * 4)]);
        }
    }
}
