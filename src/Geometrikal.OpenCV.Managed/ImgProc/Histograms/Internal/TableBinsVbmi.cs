// This implementation is ported from the OpenCV histogram functions.
//
// Original source:
//   modules/imgproc/src/histogram.cpp (calcHist_8u)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Geometrikal.OpenCV.Managed.Core;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// Bin offsets of 64 pixels per step for two- and three-dimensional 8U histograms with AVX-512
/// VBMI. Byte permutes deinterleave the planes; each 8U table entry is split into a low-byte and a
/// high-byte table, read with <c>vpermi2b</c>, and summed in 16-bit lanes. An out-of-range entry
/// becomes 2^14, so a sum of up to three entries stays below 2^16 and at least
/// <c>total</c>. The offsets leave the step in lane order, not pixel order; counting does not
/// depend on the order.
/// </summary>
internal static class TableBinsVbmi
{
    private const int Vc = 64;
    private const int OutOfRange = 1 << 14;

    /// <summary>True when the layout has 2 or 3 dimensions, small bin counts and planes of 1 to 4 channels.</summary>
    public static bool Supported(HistLayout<byte> layout)
    {
        if (!ByteLut.UseVbmi || layout.Dims is not (2 or 3) || layout.Total >= OutOfRange || layout.Width < Vc)
        {
            return false;
        }

        foreach (int cn in layout.Steps)
        {
            if (cn > 4)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>Low and high bytes of each 8U table entry, 512 bytes per dimension.</summary>
    public static byte[] SplitTables(int[] tab, int dims)
    {
        var split = new byte[dims * 512];
        for (int d = 0; d < dims; d++)
        {
            for (int v = 0; v < 256; v++)
            {
                int e = Math.Min(tab[(d * 256) + v], OutOfRange);
                split[(d * 512) + v] = (byte)e;
                split[(d * 512) + 256 + v] = (byte)(e >> 8);
            }
        }

        return split;
    }

    /// <summary>
    /// Counts the pixels <c>[0, n)</c> of rows <c>[y0, y1)</c>, <c>n</c> the row width rounded down
    /// to a multiple of 64, into four sub-histograms at <paramref name="h0"/> +
    /// <c>k * subStride</c>. Returns <c>n</c>.
    /// </summary>
    public static int CountRows(
        HistLayout<byte> layout, byte[] split, Mat<byte>? mask, int y0, int y1, ref int h0, nint subStride, Span<ushort> buffer)
    {
        int dims = layout.Dims;
        int w = layout.Width;
        int n = w - (w % Vc);
        Plane p0 = Plane.Create(layout, 0);
        Plane p1 = Plane.Create(layout, 1);
        Plane p2 = dims == 3 ? Plane.Create(layout, 2) : default;
        ref byte s = ref MemoryMarshal.GetArrayDataReference(split);
        Table lo0 = Table.Load(ref s);
        Table hi0 = Table.Load(ref Unsafe.Add(ref s, 256));
        Table lo1 = Table.Load(ref Unsafe.Add(ref s, 512));
        Table hi1 = Table.Load(ref Unsafe.Add(ref s, 768));
        Table lo2 = dims == 3 ? Table.Load(ref Unsafe.Add(ref s, 1024)) : default;
        Table hi2 = dims == 3 ? Table.Load(ref Unsafe.Add(ref s, 1280)) : default;
        var limit = Vector512.Create((ushort)layout.Total);
        ref ushort buf = ref MemoryMarshal.GetReference(buffer);
        ref int h1 = ref Unsafe.Add(ref h0, subStride);
        ref int h2 = ref Unsafe.Add(ref h0, 2 * subStride);
        ref int h3 = ref Unsafe.Add(ref h0, 3 * subStride);
        for (int y = y0; y < y1; y++)
        {
            ref byte b0 = ref RowBase(layout, 0, y);
            ref byte b1 = ref RowBase(layout, 1, y);
            ref byte b2 = ref dims == 3 ? ref RowBase(layout, 2, y) : ref b0;
            ref byte m = ref mask is null ? ref Unsafe.NullRef<byte>() : ref MemoryMarshal.GetReference(mask.GetReadOnlyRowSpan(y));
            for (int x = 0; x < n; x += Vc)
            {
                Vector512<byte> v0 = p0.Load(ref b0, x);
                Vector512<byte> v1 = p1.Load(ref b1, x);
                Vector512<byte> l0 = lo0.Lookup(v0);
                Vector512<byte> u0 = hi0.Lookup(v0);
                Vector512<byte> l1 = lo1.Lookup(v1);
                Vector512<byte> u1 = hi1.Lookup(v1);
                Vector512<ushort> sumLo = Unpack(l0, u0, low: true) + Unpack(l1, u1, low: true);
                Vector512<ushort> sumHi = Unpack(l0, u0, low: false) + Unpack(l1, u1, low: false);
                if (dims == 3)
                {
                    Vector512<byte> v2 = p2.Load(ref b2, x);
                    Vector512<byte> l2 = lo2.Lookup(v2);
                    Vector512<byte> u2 = hi2.Lookup(v2);
                    sumLo += Unpack(l2, u2, low: true);
                    sumHi += Unpack(l2, u2, low: false);
                }

                if (mask is not null)
                {
                    Vector512<byte> off = Vector512.Equals(Vector512.LoadUnsafe(ref m, (nuint)x), Vector512<byte>.Zero);
                    sumLo |= Unpack(off, off, low: true);
                    sumHi |= Unpack(off, off, low: false);
                }

                Vector512.Min(sumLo, limit).StoreUnsafe(ref buf);
                Vector512.Min(sumHi, limit).StoreUnsafe(ref buf, 32);
                for (nint k = 0; k < Vc; k += 4)
                {
                    Unsafe.Add(ref h0, Unsafe.Add(ref buf, k))++;
                    Unsafe.Add(ref h1, Unsafe.Add(ref buf, k + 1))++;
                    Unsafe.Add(ref h2, Unsafe.Add(ref buf, k + 2))++;
                    Unsafe.Add(ref h3, Unsafe.Add(ref buf, k + 3))++;
                }
            }
        }

        return n;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<ushort> Unpack(Vector512<byte> lo, Vector512<byte> hi, bool low)
        => (low ? Avx512BW.UnpackLow(lo, hi) : Avx512BW.UnpackHigh(lo, hi)).AsUInt16();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ref byte RowBase(HistLayout<byte> layout, int d, int y)
        => ref MemoryMarshal.GetReference(layout.Images[d].GetReadOnlyRowSpan(y));

    /// <summary>Byte <c>cn * j + c</c> of 64 consecutive pixels, read from <c>cn</c> vectors.</summary>
    private readonly struct Plane
    {
        private readonly int _cn;
        private readonly Vector512<byte> _first;
        private readonly Vector512<byte> _second;
        private readonly Vector512<byte> _useSecond;

        private Plane(int cn, Vector512<byte> first, Vector512<byte> second, Vector512<byte> useSecond)
        {
            _cn = cn;
            _first = first;
            _second = second;
            _useSecond = useSecond;
        }

        public static Plane Create(HistLayout<byte> layout, int d)
        {
            int cn = layout.Steps[d];
            int c = layout.Offsets[d];
            Span<byte> first = stackalloc byte[Vc];
            Span<byte> second = stackalloc byte[Vc];
            Span<byte> use = stackalloc byte[Vc];
            for (int j = 0; j < Vc; j++)
            {
                int k = (cn * j) + c;

                // cn 3: bytes 0..127 come from vectors 0-1, bytes 128..191 from vectors 1-2.
                // cn 4: bytes 0..127 come from vectors 0-1, bytes 128..255 from vectors 2-3.
                int secondBase = cn == 3 ? 64 : 128;
                first[j] = (byte)(k & 127);
                second[j] = (byte)((k - secondBase) & 127);
                use[j] = k >= 128 ? (byte)0xFF : (byte)0;
            }

            return new Plane(
                cn,
                Vector512.Create<byte>(first),
                Vector512.Create<byte>(second),
                Vector512.Create<byte>(use));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector512<byte> Load(ref byte row, int x)
        {
            ref byte p = ref Unsafe.Add(ref row, x * _cn);
            Vector512<byte> v0 = Vector512.LoadUnsafe(ref p);
            if (_cn == 1)
            {
                return v0;
            }

            Vector512<byte> v1 = Vector512.LoadUnsafe(ref p, 64);
            Vector512<byte> a = Avx512Vbmi.PermuteVar64x8x2(v0, _first, v1);
            if (_cn == 2)
            {
                return a;
            }

            Vector512<byte> b = _cn == 3
                ? Avx512Vbmi.PermuteVar64x8x2(v1, _second, Vector512.LoadUnsafe(ref p, 128))
                : Avx512Vbmi.PermuteVar64x8x2(Vector512.LoadUnsafe(ref p, 128), _second, Vector512.LoadUnsafe(ref p, 192));
            return Vector512.ConditionalSelect(_useSecond, b, a);
        }
    }

    /// <summary>A 256-byte table in four 64-byte registers.</summary>
    private readonly struct Table
    {
        private readonly Vector512<byte> _t0;
        private readonly Vector512<byte> _t1;
        private readonly Vector512<byte> _t2;
        private readonly Vector512<byte> _t3;

        private Table(Vector512<byte> t0, Vector512<byte> t1, Vector512<byte> t2, Vector512<byte> t3)
        {
            _t0 = t0;
            _t1 = t1;
            _t2 = t2;
            _t3 = t3;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Table Load(ref byte t)
            => new(Vector512.LoadUnsafe(ref t), Vector512.LoadUnsafe(ref t, 64), Vector512.LoadUnsafe(ref t, 128), Vector512.LoadUnsafe(ref t, 192));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Vector512<byte> Lookup(Vector512<byte> idx)
            => Vector512.ConditionalSelect(
                Vector512.GreaterThan(idx, Vector512.Create((byte)127)),
                Avx512Vbmi.PermuteVar64x8x2(_t2, idx, _t3),
                Avx512Vbmi.PermuteVar64x8x2(_t0, idx, _t1));
    }
}
