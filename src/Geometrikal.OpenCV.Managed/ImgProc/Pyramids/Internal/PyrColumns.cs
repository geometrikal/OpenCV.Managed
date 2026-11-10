// This implementation is ported from the OpenCV image pyramid functions.
//
// Original sources:
//   modules/imgproc/src/pyramids.cpp (FixPtCast, FltCast, PyrDownVecV, PyrUpVecV,
//   PyrUpVecVOneRow, pyrDown_, pyrUp_)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

/// <summary>
/// Vertical stages of <c>pyrDown</c> and <c>pyrUp</c>. Integer results use the OpenCV
/// fixed-point cast <c>(s + 2^(k−1)) >> k</c> (k = 8 down, 6 up). Floating-point results
/// multiply by <c>2^−k</c>.
/// </summary>
/// <remarks>
/// The OpenCV 128-bit vector code and the scalar code group the floating-point sums
/// differently. <c>vecEnd</c> is the element count that the OpenCV vector loop covers; the
/// elements before it use the vector order and the elements after it use the scalar order.
/// </remarks>
internal static class PyrColumns
{
    /// <summary>
    /// <c>dst[x] = cast(r2·6 + (r1 + r3)·4 + r0 + r4)</c>; the five rows start at
    /// <c>rows[k]</c> for k = 0..4.
    /// </summary>
    public static void Down<T, TW>(TW[][] rows, int[] slots, Span<T> dst, int vecEnd)
        where T : unmanaged
        where TW : unmanaged
    {
        int n = dst.Length;
        if (typeof(T) == typeof(byte))
        {
            DownByte(ref Row<ushort, TW>(rows, slots, 0), ref Row<ushort, TW>(rows, slots, 1), ref Row<ushort, TW>(rows, slots, 2), ref Row<ushort, TW>(rows, slots, 3), ref Row<ushort, TW>(rows, slots, 4), ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(dst)), n);
        }
        else if (typeof(T) == typeof(ushort))
        {
            DownUShort(ref Row<int, TW>(rows, slots, 0), ref Row<int, TW>(rows, slots, 1), ref Row<int, TW>(rows, slots, 2), ref Row<int, TW>(rows, slots, 3), ref Row<int, TW>(rows, slots, 4), ref Unsafe.As<T, ushort>(ref MemoryMarshal.GetReference(dst)), n);
        }
        else if (typeof(T) == typeof(short))
        {
            DownShort(ref Row<int, TW>(rows, slots, 0), ref Row<int, TW>(rows, slots, 1), ref Row<int, TW>(rows, slots, 2), ref Row<int, TW>(rows, slots, 3), ref Row<int, TW>(rows, slots, 4), ref Unsafe.As<T, short>(ref MemoryMarshal.GetReference(dst)), n);
        }
        else if (typeof(T) == typeof(float))
        {
            DownFloat(ref Row<float, TW>(rows, slots, 0), ref Row<float, TW>(rows, slots, 1), ref Row<float, TW>(rows, slots, 2), ref Row<float, TW>(rows, slots, 3), ref Row<float, TW>(rows, slots, 4), ref Unsafe.As<T, float>(ref MemoryMarshal.GetReference(dst)), n, vecEnd);
        }
        else
        {
            DownDouble(ref Row<double, TW>(rows, slots, 0), ref Row<double, TW>(rows, slots, 1), ref Row<double, TW>(rows, slots, 2), ref Row<double, TW>(rows, slots, 3), ref Row<double, TW>(rows, slots, 4), ref Unsafe.As<T, double>(ref MemoryMarshal.GetReference(dst)), n);
        }
    }

    /// <summary>
    /// <c>dst0[x] = cast(r0 + r1·6 + r2)</c> and, when <paramref name="dst1"/> is not empty,
    /// <c>dst1[x] = cast((r1 + r2)·4)</c>.
    /// </summary>
    public static void Up<T, TW>(TW[][] rows, int[] slots, Span<T> dst0, Span<T> dst1, int vecEnd)
        where T : unmanaged
        where TW : unmanaged
    {
        int n = dst0.Length;
        bool two = !dst1.IsEmpty;
        if (typeof(T) == typeof(byte))
        {
            UpByte(ref Row<ushort, TW>(rows, slots, 0), ref Row<ushort, TW>(rows, slots, 1), ref Row<ushort, TW>(rows, slots, 2), ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(dst0)), ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(dst1)), two, n);
        }
        else if (typeof(T) == typeof(ushort))
        {
            UpInt(ref Row<int, TW>(rows, slots, 0), ref Row<int, TW>(rows, slots, 1), ref Row<int, TW>(rows, slots, 2), ref Unsafe.As<T, ushort>(ref MemoryMarshal.GetReference(dst0)), ref Unsafe.As<T, ushort>(ref MemoryMarshal.GetReference(dst1)), two, n);
        }
        else if (typeof(T) == typeof(short))
        {
            UpInt(ref Row<int, TW>(rows, slots, 0), ref Row<int, TW>(rows, slots, 1), ref Row<int, TW>(rows, slots, 2), ref Unsafe.As<T, short>(ref MemoryMarshal.GetReference(dst0)), ref Unsafe.As<T, short>(ref MemoryMarshal.GetReference(dst1)), two, n);
        }
        else if (typeof(T) == typeof(float))
        {
            UpFloat(ref Row<float, TW>(rows, slots, 0), ref Row<float, TW>(rows, slots, 1), ref Row<float, TW>(rows, slots, 2), ref Unsafe.As<T, float>(ref MemoryMarshal.GetReference(dst0)), ref Unsafe.As<T, float>(ref MemoryMarshal.GetReference(dst1)), two, n, vecEnd);
        }
        else
        {
            UpFloat(ref Row<double, TW>(rows, slots, 0), ref Row<double, TW>(rows, slots, 1), ref Row<double, TW>(rows, slots, 2), ref Unsafe.As<T, double>(ref MemoryMarshal.GetReference(dst0)), ref Unsafe.As<T, double>(ref MemoryMarshal.GetReference(dst1)), two, n, 0);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static ref TR Row<TR, TW>(TW[][] rows, int[] slots, int k)
        => ref Unsafe.As<TW, TR>(ref MemoryMarshal.GetArrayDataReference(rows[slots[k]]));

    private static void DownByte(ref ushort r0, ref ushort r1, ref ushort r2, ref ushort r3, ref ushort r4, ref byte d, int n)
    {
        int x = 0;
        if (Vector.IsHardwareAccelerated && n >= Vector<byte>.Count)
        {
            int vc = Vector<byte>.Count;
            int hc = Vector<ushort>.Count;
            var half = new Vector<ushort>(128);
            for (; x < n; x += vc)
            {
                // The final vector overlaps the previous one; it rewrites equal values.
                x = Math.Min(x, n - vc);
                Vector<ushort> lo = DownSum(ref r0, ref r1, ref r2, ref r3, ref r4, x) + half;
                Vector<ushort> hi = DownSum(ref r0, ref r1, ref r2, ref r3, ref r4, x + hc) + half;
                Vector.Narrow(Vector.ShiftRightLogical(lo, 8), Vector.ShiftRightLogical(hi, 8)).StoreUnsafe(ref d, (nuint)x);
            }

            return;
        }

        for (; x < n; x++)
        {
            int s = Unsafe.Add(ref r0, x) + Unsafe.Add(ref r4, x) + (2 * Unsafe.Add(ref r2, x))
                + ((Unsafe.Add(ref r1, x) + Unsafe.Add(ref r3, x) + Unsafe.Add(ref r2, x)) << 2);
            Unsafe.Add(ref d, x) = (byte)((s + 128) >> 8);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<TW> DownSum<TW>(ref TW r0, ref TW r1, ref TW r2, ref TW r3, ref TW r4, int x)
        where TW : unmanaged
    {
        Vector<TW> a = Vector.LoadUnsafe(ref r0, (nuint)x);
        Vector<TW> b = Vector.LoadUnsafe(ref r1, (nuint)x);
        Vector<TW> c = Vector.LoadUnsafe(ref r2, (nuint)x);
        Vector<TW> e = Vector.LoadUnsafe(ref r3, (nuint)x);
        Vector<TW> g = Vector.LoadUnsafe(ref r4, (nuint)x);
        Vector<TW> mid = b + e + c;
        return a + g + c + c + mid + mid + mid + mid;
    }

    private static void DownUShort(ref int r0, ref int r1, ref int r2, ref int r3, ref int r4, ref ushort d, int n)
    {
        int x = 0;
        if (Vector.IsHardwareAccelerated && n >= Vector<ushort>.Count)
        {
            int vc = Vector<ushort>.Count;
            int hc = Vector<int>.Count;
            var half = new Vector<int>(128);
            for (; x < n; x += vc)
            {
                // The final vector overlaps the previous one; it rewrites equal values.
                x = Math.Min(x, n - vc);
                Vector<uint> lo = Vector.AsVectorUInt32(Vector.ShiftRightArithmetic(DownSum(ref r0, ref r1, ref r2, ref r3, ref r4, x) + half, 8));
                Vector<uint> hi = Vector.AsVectorUInt32(Vector.ShiftRightArithmetic(DownSum(ref r0, ref r1, ref r2, ref r3, ref r4, x + hc) + half, 8));
                Vector.Narrow(lo, hi).StoreUnsafe(ref d, (nuint)x);
            }

            return;
        }

        for (; x < n; x++)
        {
            Unsafe.Add(ref d, x) = (ushort)((DownScalarInt(ref r0, ref r1, ref r2, ref r3, ref r4, x) + 128) >> 8);
        }
    }

    private static void DownShort(ref int r0, ref int r1, ref int r2, ref int r3, ref int r4, ref short d, int n)
    {
        int x = 0;
        if (Vector.IsHardwareAccelerated && n >= Vector<short>.Count)
        {
            int vc = Vector<short>.Count;
            int hc = Vector<int>.Count;
            var half = new Vector<int>(128);
            for (; x < n; x += vc)
            {
                // The final vector overlaps the previous one; it rewrites equal values.
                x = Math.Min(x, n - vc);
                Vector<int> lo = Vector.ShiftRightArithmetic(DownSum(ref r0, ref r1, ref r2, ref r3, ref r4, x) + half, 8);
                Vector<int> hi = Vector.ShiftRightArithmetic(DownSum(ref r0, ref r1, ref r2, ref r3, ref r4, x + hc) + half, 8);
                Vector.Narrow(lo, hi).StoreUnsafe(ref d, (nuint)x);
            }

            return;
        }

        for (; x < n; x++)
        {
            Unsafe.Add(ref d, x) = (short)((DownScalarInt(ref r0, ref r1, ref r2, ref r3, ref r4, x) + 128) >> 8);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int DownScalarInt(ref int r0, ref int r1, ref int r2, ref int r3, ref int r4, int x)
        => (Unsafe.Add(ref r2, x) * 6) + ((Unsafe.Add(ref r1, x) + Unsafe.Add(ref r3, x)) * 4) + Unsafe.Add(ref r0, x) + Unsafe.Add(ref r4, x);

    private static void DownFloat(ref float r0, ref float r1, ref float r2, ref float r3, ref float r4, ref float d, int n, int vecEnd)
    {
        const float Scale = 1f / 256;
        var scale = new Vector<float>(Scale);
        var four = new Vector<float>(4f);
        var six = new Vector<float>(6f);
        int vc = Vector<float>.Count;
        int x = 0;

        // OpenCV PyrDownVecV order: ((r1 + r3) + r2)·4 + ((r0 + r4) + (r2 + r2)).
        if (Vector.IsHardwareAccelerated)
        {
            for (; x <= vecEnd - vc; x += vc)
            {
                Vector<float> a = Vector.LoadUnsafe(ref r0, (nuint)x);
                Vector<float> b = Vector.LoadUnsafe(ref r1, (nuint)x);
                Vector<float> c = Vector.LoadUnsafe(ref r2, (nuint)x);
                Vector<float> e = Vector.LoadUnsafe(ref r3, (nuint)x);
                Vector<float> g = Vector.LoadUnsafe(ref r4, (nuint)x);
                (((((b + e) + c) * four) + ((a + g) + (c + c))) * scale).StoreUnsafe(ref d, (nuint)x);
            }
        }

        for (; x < vecEnd; x++)
        {
            float c = Unsafe.Add(ref r2, x);
            Unsafe.Add(ref d, x) = ((((Unsafe.Add(ref r1, x) + Unsafe.Add(ref r3, x)) + c) * 4f) + ((Unsafe.Add(ref r0, x) + Unsafe.Add(ref r4, x)) + (c + c))) * Scale;
        }

        // OpenCV scalar order: ((r2·6 + (r1 + r3)·4) + r0) + r4.
        if (Vector.IsHardwareAccelerated)
        {
            for (; x <= n - vc; x += vc)
            {
                Vector<float> a = Vector.LoadUnsafe(ref r0, (nuint)x);
                Vector<float> b = Vector.LoadUnsafe(ref r1, (nuint)x);
                Vector<float> c = Vector.LoadUnsafe(ref r2, (nuint)x);
                Vector<float> e = Vector.LoadUnsafe(ref r3, (nuint)x);
                Vector<float> g = Vector.LoadUnsafe(ref r4, (nuint)x);
                (((((c * six) + ((b + e) * four)) + a) + g) * scale).StoreUnsafe(ref d, (nuint)x);
            }
        }

        for (; x < n; x++)
        {
            Unsafe.Add(ref d, x) = ((((Unsafe.Add(ref r2, x) * 6f) + ((Unsafe.Add(ref r1, x) + Unsafe.Add(ref r3, x)) * 4f)) + Unsafe.Add(ref r0, x)) + Unsafe.Add(ref r4, x)) * Scale;
        }
    }

    private static void DownDouble(ref double r0, ref double r1, ref double r2, ref double r3, ref double r4, ref double d, int n)
    {
        const double Scale = 1.0 / 256;
        var scale = new Vector<double>(Scale);
        var four = new Vector<double>(4.0);
        var six = new Vector<double>(6.0);
        int vc = Vector<double>.Count;
        int x = 0;

        // OpenCV has no double PyrDownVecV; every element uses the scalar order.
        if (Vector.IsHardwareAccelerated)
        {
            for (; x <= n - vc; x += vc)
            {
                Vector<double> a = Vector.LoadUnsafe(ref r0, (nuint)x);
                Vector<double> b = Vector.LoadUnsafe(ref r1, (nuint)x);
                Vector<double> c = Vector.LoadUnsafe(ref r2, (nuint)x);
                Vector<double> e = Vector.LoadUnsafe(ref r3, (nuint)x);
                Vector<double> g = Vector.LoadUnsafe(ref r4, (nuint)x);
                (((((c * six) + ((b + e) * four)) + a) + g) * scale).StoreUnsafe(ref d, (nuint)x);
            }
        }

        for (; x < n; x++)
        {
            Unsafe.Add(ref d, x) = ((((Unsafe.Add(ref r2, x) * 6.0) + ((Unsafe.Add(ref r1, x) + Unsafe.Add(ref r3, x)) * 4.0)) + Unsafe.Add(ref r0, x)) + Unsafe.Add(ref r4, x)) * Scale;
        }
    }

    private static void UpByte(ref ushort r0, ref ushort r1, ref ushort r2, ref byte d0, ref byte d1, bool two, int n)
    {
        int x = 0;
        if (Vector.IsHardwareAccelerated && n >= Vector<byte>.Count)
        {
            int vc = Vector<byte>.Count;
            int hc = Vector<ushort>.Count;
            var half = new Vector<ushort>(32);
            for (; x < n; x += vc)
            {
                // The final vector overlaps the previous one; it rewrites equal values.
                x = Math.Min(x, n - vc);
                Vector<ushort> a0 = Vector.LoadUnsafe(ref r0, (nuint)x);
                Vector<ushort> b0 = Vector.LoadUnsafe(ref r1, (nuint)x);
                Vector<ushort> c0 = Vector.LoadUnsafe(ref r2, (nuint)x);
                Vector<ushort> a1 = Vector.LoadUnsafe(ref r0, (nuint)(x + hc));
                Vector<ushort> b1 = Vector.LoadUnsafe(ref r1, (nuint)(x + hc));
                Vector<ushort> c1 = Vector.LoadUnsafe(ref r2, (nuint)(x + hc));
                Vector<ushort> e0 = a0 + c0 + (b0 * 6) + half;
                Vector<ushort> e1 = a1 + c1 + (b1 * 6) + half;
                Vector.Narrow(Vector.ShiftRightLogical(e0, 6), Vector.ShiftRightLogical(e1, 6)).StoreUnsafe(ref d0, (nuint)x);
                if (two)
                {
                    Vector<ushort> o0 = ((b0 + c0) << 2) + half;
                    Vector<ushort> o1 = ((b1 + c1) << 2) + half;
                    Vector.Narrow(Vector.ShiftRightLogical(o0, 6), Vector.ShiftRightLogical(o1, 6)).StoreUnsafe(ref d1, (nuint)x);
                }
            }

            return;
        }

        for (; x < n; x++)
        {
            int b = Unsafe.Add(ref r1, x);
            int c = Unsafe.Add(ref r2, x);
            Unsafe.Add(ref d0, x) = (byte)((Unsafe.Add(ref r0, x) + (b * 6) + c + 32) >> 6);
            if (two)
            {
                Unsafe.Add(ref d1, x) = (byte)((((b + c) * 4) + 32) >> 6);
            }
        }
    }

    private static void UpInt<T>(ref int r0, ref int r1, ref int r2, ref T d0, ref T d1, bool two, int n)
        where T : unmanaged
    {
        int x = 0;
        if (Vector.IsHardwareAccelerated && n >= Vector<short>.Count)
        {
            int vc = Vector<short>.Count;
            int hc = Vector<int>.Count;
            var half = new Vector<int>(32);
            for (; x < n; x += vc)
            {
                // The final vector overlaps the previous one; it rewrites equal values.
                x = Math.Min(x, n - vc);
                Vector<int> a0 = Vector.LoadUnsafe(ref r0, (nuint)x);
                Vector<int> b0 = Vector.LoadUnsafe(ref r1, (nuint)x);
                Vector<int> c0 = Vector.LoadUnsafe(ref r2, (nuint)x);
                Vector<int> a1 = Vector.LoadUnsafe(ref r0, (nuint)(x + hc));
                Vector<int> b1 = Vector.LoadUnsafe(ref r1, (nuint)(x + hc));
                Vector<int> c1 = Vector.LoadUnsafe(ref r2, (nuint)(x + hc));
                StoreUp(ref d0, x, Vector.ShiftRightArithmetic(a0 + c0 + (b0 * 6) + half, 6), Vector.ShiftRightArithmetic(a1 + c1 + (b1 * 6) + half, 6));
                if (two)
                {
                    StoreUp(ref d1, x, Vector.ShiftRightArithmetic(((b0 + c0) << 2) + half, 6), Vector.ShiftRightArithmetic(((b1 + c1) << 2) + half, 6));
                }
            }

            return;
        }

        for (; x < n; x++)
        {
            int b = Unsafe.Add(ref r1, x);
            int c = Unsafe.Add(ref r2, x);
            Store(ref d0, x, (Unsafe.Add(ref r0, x) + (b * 6) + c + 32) >> 6);
            if (two)
            {
                Store(ref d1, x, (((b + c) * 4) + 32) >> 6);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreUp<T>(ref T d, int x, Vector<int> lo, Vector<int> hi)
        where T : unmanaged
    {
        if (typeof(T) == typeof(short))
        {
            Vector.Narrow(lo, hi).StoreUnsafe(ref Unsafe.As<T, short>(ref d), (nuint)x);
        }
        else
        {
            Vector.Narrow(Vector.AsVectorUInt32(lo), Vector.AsVectorUInt32(hi)).StoreUnsafe(ref Unsafe.As<T, ushort>(ref d), (nuint)x);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store<T>(ref T d, int x, int v)
        where T : unmanaged
    {
        if (typeof(T) == typeof(short))
        {
            Unsafe.Add(ref Unsafe.As<T, short>(ref d), x) = (short)v;
        }
        else
        {
            Unsafe.Add(ref Unsafe.As<T, ushort>(ref d), x) = (ushort)v;
        }
    }

    private static void UpFloat<TF>(ref TF r0, ref TF r1, ref TF r2, ref TF d0, ref TF d1, bool two, int n, int vecEnd)
        where TF : unmanaged, IFloatingPointIeee754<TF>
    {
        TF scale = TF.CreateTruncating(1.0 / 64);
        TF scale4 = TF.CreateTruncating(1.0 / 16);
        TF six = TF.CreateTruncating(6);
        TF four = TF.CreateTruncating(4);
        var vScale = new Vector<TF>(scale);
        var vScale4 = new Vector<TF>(scale4);
        var vSix = new Vector<TF>(six);
        var vFour = new Vector<TF>(four);
        int vc = Vector<TF>.Count;
        int x = 0;

        // The OpenCV vector code scales the odd row by 1/16; the scalar code multiplies by 4, then by 1/64.
        if (Vector.IsHardwareAccelerated)
        {
            for (; x <= vecEnd - vc; x += vc)
            {
                Vector<TF> a = Vector.LoadUnsafe(ref r0, (nuint)x);
                Vector<TF> b = Vector.LoadUnsafe(ref r1, (nuint)x);
                Vector<TF> c = Vector.LoadUnsafe(ref r2, (nuint)x);
                (((a + (b * vSix)) + c) * vScale).StoreUnsafe(ref d0, (nuint)x);
                if (two)
                {
                    ((b + c) * vScale4).StoreUnsafe(ref d1, (nuint)x);
                }
            }
        }

        for (; x < vecEnd; x++)
        {
            TF b = Unsafe.Add(ref r1, x);
            TF c = Unsafe.Add(ref r2, x);
            Unsafe.Add(ref d0, x) = ((Unsafe.Add(ref r0, x) + (b * six)) + c) * scale;
            if (two)
            {
                Unsafe.Add(ref d1, x) = (b + c) * scale4;
            }
        }

        if (Vector.IsHardwareAccelerated)
        {
            for (; x <= n - vc; x += vc)
            {
                Vector<TF> a = Vector.LoadUnsafe(ref r0, (nuint)x);
                Vector<TF> b = Vector.LoadUnsafe(ref r1, (nuint)x);
                Vector<TF> c = Vector.LoadUnsafe(ref r2, (nuint)x);
                (((a + (b * vSix)) + c) * vScale).StoreUnsafe(ref d0, (nuint)x);
                if (two)
                {
                    (((b + c) * vFour) * vScale).StoreUnsafe(ref d1, (nuint)x);
                }
            }
        }

        for (; x < n; x++)
        {
            TF b = Unsafe.Add(ref r1, x);
            TF c = Unsafe.Add(ref r2, x);
            Unsafe.Add(ref d0, x) = ((Unsafe.Add(ref r0, x) + (b * six)) + c) * scale;
            if (two)
            {
                Unsafe.Add(ref d1, x) = ((b + c) * four) * scale;
            }
        }
    }
}
