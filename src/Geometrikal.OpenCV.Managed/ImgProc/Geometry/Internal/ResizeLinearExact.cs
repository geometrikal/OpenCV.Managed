// This implementation is ported from the OpenCV bit-exact bilinear resize.
//
// Original sources:
//   modules/imgproc/src/resize.cpp (resize_bitExact, resize_bitExactInvoker,
//     interpolationLinear, hlineResize, vlineSet, vlineResize)
//   modules/imgproc/src/fixedpoint.inl.hpp (fixedpoint64, ufixedpoint64, fixedpoint32,
//     ufixedpoint32, ufixedpoint16)
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
/// Fixed-point arithmetic of one element type of <c>INTER_LINEAR_EXACT</c>. Raw values are the
/// bit patterns of the OpenCV fixed-point types, stored in a <see cref="long"/>.
/// </summary>
internal interface IFixedLinear<T>
    where T : unmanaged
{
    static abstract long One { get; }

    static abstract long FromSource(T s);

    static abstract long Coef(double d);

    static abstract long Sub(long a, long b);

    static abstract long HMul(long m, T s);

    static abstract long HAdd(long a, long b);

    static abstract long VMul(long m, long h);

    static abstract long VAdd(long a, long b);

    static abstract T VCast(long v);

    static abstract T SetCast(long raw);
}

/// <summary><c>uint8</c>: <c>ufixedpoint16</c> rows (8 fraction bits) and <c>ufixedpoint32</c> sums.</summary>
internal readonly struct FixedU8 : IFixedLinear<byte>
{
    public static long One => 256;

    public static long FromSource(byte s) => s << 8;

    public static long Coef(double d) => d < 0 || (d == 0 && double.IsNegative(d)) ? 0 : (ushort)ScalarConverter.CvRound(d * 256);

    public static long Sub(long a, long b) => a > b ? a - b : 0;

    public static long HMul(long m, byte s) => Math.Min((uint)m * s, 0xFFFFu);

    public static long HAdd(long a, long b)
    {
        ushort res = (ushort)(a + b);
        return a > res ? 0xFFFF : res;
    }

    public static long VMul(long m, long h) => (uint)m * (uint)h;

    public static long VAdd(long a, long b)
    {
        uint res = unchecked((uint)a + (uint)b);
        return (uint)a > res ? 0xFFFFFFFFu : res;
    }

    public static byte VCast(long v) => (byte)Math.Min(unchecked((uint)v + 32768u) >> 16, 255u);

    public static byte SetCast(long raw) => (byte)Math.Min((ushort)(raw + 128) >> 8, 255);
}

/// <summary><c>int8</c> and <c>int16</c>: <c>fixedpoint32</c> rows (16 fraction bits) and <c>fixedpoint64</c> sums.</summary>
internal readonly struct FixedS32Rows<T> : IFixedLinear<T>
    where T : unmanaged
{
    public static long One => 65536;

    public static long FromSource(T s) => Saturate.ToInt(s) << 16;

    public static long Coef(double d) => ScalarConverter.CvRound(d * 65536);

    public static long Sub(long a, long b)
    {
        int va = (int)a, vb = (int)b;
        int res = unchecked(va - vb);
        return ((long)((va ^ vb) & (va ^ res)) >> 31) != 0 ? ~(res & int.MinValue) : res;
    }

    public static long HMul(long m, T s) => Math.Clamp((long)(int)m * Saturate.ToInt(s), int.MinValue, int.MaxValue);

    public static long HAdd(long a, long b)
    {
        int va = (int)a, vb = (int)b;
        int res = unchecked(va + vb);
        return ((long)((va ^ res) & (vb ^ res)) >> 31) != 0 ? ~(res & int.MinValue) : res;
    }

    public static long VMul(long m, long h) => (long)(int)m * (int)h;

    public static long VAdd(long a, long b) => FixedMath.AddSat64(a, b);

    public static T VCast(long v) => Saturate.FromInt<T>(SatInt(unchecked((long)((ulong)v + 0x80000000UL)) >> 32));

    public static T SetCast(long raw) => Saturate.FromInt<T>(unchecked((int)((uint)(int)raw + 32768u)) >> 16);

    private static int SatInt(long v) => (int)Math.Clamp(v, int.MinValue, int.MaxValue);
}

/// <summary><c>uint16</c>: <c>ufixedpoint32</c> rows (16 fraction bits) and <c>ufixedpoint64</c> sums.</summary>
internal readonly struct FixedU16 : IFixedLinear<ushort>
{
    public static long One => 65536;

    public static long FromSource(ushort s) => (long)s << 16;

    public static long Coef(double d) => d < 0 || (d == 0 && double.IsNegative(d)) ? 0 : (uint)ScalarConverter.CvRound(d * 65536);

    public static long Sub(long a, long b) => a > b ? a - b : 0;

    public static long HMul(long m, ushort s) => (long)Math.Min((ulong)m * s, 0xFFFFFFFFUL);

    public static long HAdd(long a, long b)
    {
        uint res = unchecked((uint)a + (uint)b);
        return (uint)a > res ? 0xFFFFFFFFu : res;
    }

    public static long VMul(long m, long h) => unchecked((long)((ulong)m * (ulong)h));

    public static long VAdd(long a, long b)
    {
        ulong res = unchecked((ulong)a + (ulong)b);
        return unchecked((long)((ulong)a > res ? ulong.MaxValue : res));
    }

    public static ushort VCast(long v) => (ushort)Math.Min(unchecked((ulong)v + 0x80000000UL) >> 32, 65535UL);

    public static ushort SetCast(long raw) => (ushort)Math.Min(unchecked((uint)raw + 32768u) >> 16, 65535u);
}

/// <summary><c>int32</c>: <c>fixedpoint64</c> rows and sums (32 fraction bits).</summary>
internal readonly struct FixedS32 : IFixedLinear<int>
{
    public static long One => 1L << 32;

    public static long FromSource(int s) => (long)s << 32;

    public static long Coef(double d) => FixedMath.CvRound64(d * 4294967296.0);

    public static long Sub(long a, long b)
    {
        long res = unchecked(a - b);
        return ((a ^ b) & (a ^ res)) < 0 ? ~(res & long.MinValue) : res;
    }

    public static long HMul(long m, int s) => FixedMath.Mul64(m, (long)s << 32);

    public static long HAdd(long a, long b) => FixedMath.AddSat64(a, b);

    public static long VMul(long m, long h) => FixedMath.Mul64(m, h);

    public static long VAdd(long a, long b) => FixedMath.AddSat64(a, b);

    public static int VCast(long v) => (int)Math.Clamp(unchecked((long)((ulong)v + 0x80000000UL)) >> 32, int.MinValue, int.MaxValue);

    public static int SetCast(long raw) => VCast(raw);
}

internal static class FixedMath
{
    /// <summary><c>fixedpoint64::operator+</c>: wrapping add with OpenCV's saturation rule.</summary>
    public static long AddSat64(long a, long b)
    {
        long res = unchecked(a + b);
        return ((a ^ res) & (b ^ res)) < 0 ? ~(res & long.MinValue) : res;
    }

    /// <summary><c>cvRound64(softdouble)</c>: round half to even.</summary>
    public static long CvRound64(double v)
    {
        double r = Math.Round(v, MidpointRounding.ToEven);
        return r >= -9.2233720368547758e18 && r < 9.2233720368547758e18 ? (long)r : long.MinValue;
    }

    /// <summary><c>fixedpoint64::operator*(fixedpoint64)</c>: 64 × 64-bit product with 32 fraction bits, rounded and saturated.</summary>
    public static long Mul64(long val, long mul)
    {
        bool signVal = val < 0;
        bool signMul = mul < 0;
        ulong uval = signVal ? unchecked((ulong)(-val)) : (ulong)val;
        ulong umul = signMul ? unchecked((ulong)(-mul)) : (ulong)mul;
        bool resSign = signVal ^ signMul;
        ulong sh0 = unchecked(((uval & 0xFFFFFFFF) * (umul & 0xFFFFFFFF)) + 0x80000000UL);
        ulong sh10 = (uval >> 32) * (umul & 0xFFFFFFFF);
        ulong sh11 = (uval & 0xFFFFFFFF) * (umul >> 32);
        ulong sh2 = (uval >> 32) * (umul >> 32);
        ulong lo = (sh10 & 0xFFFFFFFF) + (sh11 & 0xFFFFFFFF) + (sh0 >> 32);
        ulong hi = (sh2 & 0xFFFFFFFF) + (sh10 >> 32) + (sh11 >> 32) + (lo >> 32);
        lo &= 0xFFFFFFFF;
        if (sh2 > 0x7FFFFFFF || hi > 0x7FFFFFFF)
        {
            return resSign ? long.MinValue : long.MaxValue;
        }

        long r = (long)((hi << 32) | lo);
        return resSign ? -r : r;
    }
}

/// <summary><c>resize_bitExact</c> with <c>interpolationLinear</c>.</summary>
internal static class ResizeLinearExact
{
    /// <summary>Offsets and fixed-point weights of one axis (<c>interpolationLinear::getCoeffs</c>).</summary>
    private static void Coeffs<T, TFixed>(double invScale, int srcSize, int dstSize, Span<int> offsets, Span<long> coeffs, out int minOfs, out int maxOfs)
        where T : unmanaged
        where TFixed : struct, IFixedLinear<T>
    {
        double scale = 1.0 / invScale;
        minOfs = 0;
        maxOfs = dstSize;
        for (int v = 0; v < dstSize; v++)
        {
            double fval = (scale * (v + 0.5)) - 0.5;
            int ival = ScalarConverter.CvFloor(fval);
            offsets[v] = 0;
            coeffs[2 * v] = 0;
            coeffs[(2 * v) + 1] = 0;
            if (ival >= 0 && srcSize > 1)
            {
                if (ival < srcSize - 1)
                {
                    offsets[v] = ival;
                    long c1 = TFixed.Coef(fval - ival);
                    coeffs[(2 * v) + 1] = c1;
                    coeffs[2 * v] = TFixed.Sub(TFixed.One, c1);
                }
                else
                {
                    offsets[v] = srcSize - 1;
                    maxOfs = Math.Min(maxOfs, v);
                }
            }
            else
            {
                minOfs = Math.Max(minOfs, v + 1);
            }
        }
    }

    public static void Run<T, TFixed>(Mat<T> src, Mat<T> dst, double invScaleX, double invScaleY)
        where T : unmanaged
        where TFixed : struct, IFixedLinear<T>
    {
        int dw = dst.Width, dh = dst.Height;
        int[] xofs = ArrayPool<int>.Shared.Rent(dw);
        int[] yofs = ArrayPool<int>.Shared.Rent(dh);
        long[] xc = ArrayPool<long>.Shared.Rent(2 * dw);
        long[] yc = ArrayPool<long>.Shared.Rent(2 * dh);
        try
        {
            Coeffs<T, TFixed>(invScaleX, src.Width, dw, xofs, xc, out int minX, out int maxX);
            Coeffs<T, TFixed>(invScaleY, src.Height, dh, yofs, yc, out int minY, out int maxY);
            var axes = new Axes(xofs, yofs, xc, yc, minX, maxX, minY, maxY);
            long work = (long)dw * src.Channels * 4;
            if (typeof(T) == typeof(byte))
            {
                Mat<byte> s8 = Unsafe.As<Mat<byte>>(src), d8 = Unsafe.As<Mat<byte>>(dst);
                ParallelRows.For(dh, work / Vector<int>.Count, (y0, y1) => Rows8u(s8, d8, axes, y0, y1));
            }
            else
            {
                ParallelRows.For(dh, work, (y0, y1) => Rows<T, TFixed>(src, dst, axes, y0, y1));
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(xofs);
            ArrayPool<int>.Shared.Return(yofs);
            ArrayPool<long>.Shared.Return(xc);
            ArrayPool<long>.Shared.Return(yc);
        }
    }

    private sealed record Axes(int[] XOffsets, int[] YOffsets, long[] XCoeffs, long[] YCoeffs, int MinX, int MaxX, int MinY, int MaxY);

    private static void Rows<T, TFixed>(Mat<T> src, Mat<T> dst, Axes a, int y0, int y1)
        where T : unmanaged
        where TFixed : struct, IFixedLinear<T>
    {
        int cn = src.Channels;
        int width = dst.Width * cn;
        long[] rowA = ArrayPool<long>.Shared.Rent(width);
        long[] rowB = ArrayPool<long>.Shared.Rent(width);
        try
        {
            int rowAIndex = -1, rowBIndex = -1;
            for (int y = y0; y < y1; y++)
            {
                Span<T> d = dst.GetRowSpan(y);
                if (y < a.MinY || y >= a.MaxY)
                {
                    int sy = y < a.MinY ? 0 : src.Height - 1;
                    HLine<T, TFixed>(src.GetReadOnlyRowSpan(sy), rowA, a, cn, dst.Width);
                    rowAIndex = sy;
                    rowBIndex = -1;
                    for (int x = 0; x < width; x++)
                    {
                        d[x] = TFixed.SetCast(rowA[x]);
                    }

                    continue;
                }

                int iy = a.YOffsets[y];
                if (rowAIndex != iy)
                {
                    if (rowBIndex == iy)
                    {
                        (rowA, rowB) = (rowB, rowA);
                        rowAIndex = iy;
                        rowBIndex = -1;
                    }
                    else
                    {
                        HLine<T, TFixed>(src.GetReadOnlyRowSpan(iy), rowA, a, cn, dst.Width);
                        rowAIndex = iy;
                    }
                }

                if (rowBIndex != iy + 1)
                {
                    HLine<T, TFixed>(src.GetReadOnlyRowSpan(iy + 1), rowB, a, cn, dst.Width);
                    rowBIndex = iy + 1;
                }

                long c0 = a.YCoeffs[2 * y], c1 = a.YCoeffs[(2 * y) + 1];
                for (int x = 0; x < width; x++)
                {
                    d[x] = TFixed.VCast(TFixed.VAdd(TFixed.VMul(c0, rowA[x]), TFixed.VMul(c1, rowB[x])));
                }
            }
        }
        finally
        {
            ArrayPool<long>.Shared.Return(rowA);
            ArrayPool<long>.Shared.Return(rowB);
        }
    }

    private static void HLine<T, TFixed>(ReadOnlySpan<T> s, Span<long> d, Axes a, int cn, int dw)
        where T : unmanaged
        where TFixed : struct, IFixedLinear<T>
    {
        int x = 0;
        for (; x < a.MinX && x < dw; x++)
        {
            for (int c = 0; c < cn; c++)
            {
                d[(x * cn) + c] = TFixed.FromSource(s[c]);
            }
        }

        for (; x < a.MaxX; x++)
        {
            int o = a.XOffsets[x] * cn;
            long m0 = a.XCoeffs[2 * x], m1 = a.XCoeffs[(2 * x) + 1];
            for (int c = 0; c < cn; c++)
            {
                d[(x * cn) + c] = TFixed.HAdd(TFixed.HMul(m0, s[o + c]), TFixed.HMul(m1, s[o + cn + c]));
            }
        }

        if (x >= dw)
        {
            return;
        }

        int last = a.XOffsets[dw - 1] * cn;
        for (; x < dw; x++)
        {
            for (int c = 0; c < cn; c++)
            {
                d[(x * cn) + c] = TFixed.FromSource(s[last + c]);
            }
        }
    }

    /// <summary>
    /// 8-bit rows. The weights sum to 256, so no fixed-point operation saturates and the arithmetic
    /// is plain 32-bit integer arithmetic.
    /// </summary>
    private static void Rows8u(Mat<byte> src, Mat<byte> dst, Axes a, int y0, int y1)
    {
        int cn = src.Channels;
        int dw = dst.Width;
        int width = dw * cn;
        int[] rowA = ArrayPool<int>.Shared.Rent(width);
        int[] rowB = ArrayPool<int>.Shared.Rent(width);
        int[] xc = ArrayPool<int>.Shared.Rent(2 * dw);
        try
        {
            for (int i = 0; i < 2 * dw; i++)
            {
                xc[i] = (int)a.XCoeffs[i];
            }

            int rowAIndex = -1, rowBIndex = -1;
            for (int y = y0; y < y1; y++)
            {
                ref byte d = ref MemoryMarshal.GetReference(dst.GetRowSpan(y));
                if (y < a.MinY || y >= a.MaxY)
                {
                    int sy = y < a.MinY ? 0 : src.Height - 1;
                    HLine8u(src.GetReadOnlyRowSpan(sy), rowA, a, xc, cn, dw);
                    rowAIndex = sy;
                    rowBIndex = -1;
                    for (int x = 0; x < width; x++)
                    {
                        Unsafe.Add(ref d, x) = (byte)((rowA[x] + 128) >> 8);
                    }

                    continue;
                }

                int iy = a.YOffsets[y];
                if (rowAIndex != iy)
                {
                    if (rowBIndex == iy)
                    {
                        (rowA, rowB) = (rowB, rowA);
                        rowAIndex = iy;
                        rowBIndex = -1;
                    }
                    else
                    {
                        HLine8u(src.GetReadOnlyRowSpan(iy), rowA, a, xc, cn, dw);
                        rowAIndex = iy;
                    }
                }

                if (rowBIndex != iy + 1)
                {
                    HLine8u(src.GetReadOnlyRowSpan(iy + 1), rowB, a, xc, cn, dw);
                    rowBIndex = iy + 1;
                }

                VLine8u(rowA, rowB, (int)a.YCoeffs[2 * y], (int)a.YCoeffs[(2 * y) + 1], ref d, width);
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(rowA);
            ArrayPool<int>.Shared.Return(rowB);
            ArrayPool<int>.Shared.Return(xc);
        }
    }

    private static void HLine8u(ReadOnlySpan<byte> s, int[] d, Axes a, int[] xc, int cn, int dw)
    {
        int x = 0;
        for (; x < a.MinX && x < dw; x++)
        {
            for (int c = 0; c < cn; c++)
            {
                d[(x * cn) + c] = s[c] << 8;
            }
        }

        ref byte sr = ref MemoryMarshal.GetReference(s);
        ref int dr = ref MemoryMarshal.GetArrayDataReference(d);
        ref int xo = ref MemoryMarshal.GetArrayDataReference(a.XOffsets);
        ref int cr = ref MemoryMarshal.GetArrayDataReference(xc);
        if (cn == 1)
        {
            for (; x < a.MaxX; x++)
            {
                ref byte p = ref Unsafe.Add(ref sr, Unsafe.Add(ref xo, x));
                Unsafe.Add(ref dr, x) = (Unsafe.Add(ref cr, 2 * x) * p) + (Unsafe.Add(ref cr, (2 * x) + 1) * Unsafe.Add(ref p, 1));
            }
        }

        for (; x < a.MaxX; x++)
        {
            ref byte p = ref Unsafe.Add(ref sr, Unsafe.Add(ref xo, x) * cn);
            int m0 = Unsafe.Add(ref cr, 2 * x), m1 = Unsafe.Add(ref cr, (2 * x) + 1);
            ref int q = ref Unsafe.Add(ref dr, x * cn);
            for (int c = 0; c < cn; c++)
            {
                Unsafe.Add(ref q, c) = (m0 * Unsafe.Add(ref p, c)) + (m1 * Unsafe.Add(ref p, cn + c));
            }
        }

        if (x >= dw)
        {
            return;
        }

        int last = a.XOffsets[dw - 1] * cn;
        for (; x < dw; x++)
        {
            for (int c = 0; c < cn; c++)
            {
                d[(x * cn) + c] = s[last + c] << 8;
            }
        }
    }

    private static void VLine8u(int[] rowA, int[] rowB, int c0, int c1, ref byte d, int width)
    {
        ref int ra = ref MemoryMarshal.GetArrayDataReference(rowA);
        ref int rb = ref MemoryMarshal.GetArrayDataReference(rowB);
        int x = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int vc = Vector<int>.Count;
            var v0 = new Vector<int>(c0);
            var v1 = new Vector<int>(c1);
            var half = new Vector<int>(1 << 15);
            for (; x <= width - (4 * vc); x += 4 * vc)
            {
                FilterVectors.SaturateToByte(
                    V8(ref ra, ref rb, v0, v1, half, x),
                    V8(ref ra, ref rb, v0, v1, half, x + vc),
                    V8(ref ra, ref rb, v0, v1, half, x + (2 * vc)),
                    V8(ref ra, ref rb, v0, v1, half, x + (3 * vc))).StoreUnsafe(ref d, (nuint)x);
            }
        }

        for (; x < width; x++)
        {
            Unsafe.Add(ref d, x) = (byte)(((c0 * Unsafe.Add(ref ra, x)) + (c1 * Unsafe.Add(ref rb, x)) + (1 << 15)) >> 16);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<int> V8(ref int ra, ref int rb, Vector<int> c0, Vector<int> c1, Vector<int> half, int x)
        => Vector.ShiftRightLogical((c0 * Vector.LoadUnsafe(ref ra, (nuint)x)) + (c1 * Vector.LoadUnsafe(ref rb, (nuint)x)) + half, 16);
}
