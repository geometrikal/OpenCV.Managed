namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// Moves data between matrix rows and lane-major buffers, where element <c>e</c> of lane
/// <c>l</c> is at <c>e·L + l</c> and <c>L = Vector&lt;T&gt;.Count</c>. With AVX and 32-byte
/// vectors the moves use 8×8 (float) or 4×4 (double) register transposes; otherwise scalar copies.
/// </summary>
internal static class LaneTranspose
{
    /// <summary>
    /// <c>dst[(dstElem + e)·L + l] = src[start + l·stride + e]</c> for <c>e &lt; count</c> and
    /// <c>l &lt; lanes</c>.
    /// </summary>
    public static void RowsToLanes<T>(T[] src, int start, int stride, int lanes, int count, Span<T> dst, int dstElem)
        where T : unmanaged
    {
        int laneCount = Vector<T>.Count;
        int e = 0;
        if (lanes == laneCount && Avx.IsSupported && Vector<T>.Count * Unsafe.SizeOf<T>() == 32)
        {
            ref T s = ref MemoryMarshal.GetArrayDataReference(src);
            ref T d = ref MemoryMarshal.GetReference(dst);
            if (typeof(T) == typeof(float))
            {
                ref float sf = ref Unsafe.As<T, float>(ref Unsafe.Add(ref s, start));
                ref float df = ref Unsafe.As<T, float>(ref Unsafe.Add(ref d, dstElem * 8));
                for (; e <= count - 8; e += 8)
                {
                    Transpose8(
                        Vector256.LoadUnsafe(ref sf, (nuint)e),
                        Vector256.LoadUnsafe(ref sf, (nuint)(stride + e)),
                        Vector256.LoadUnsafe(ref sf, (nuint)((2 * stride) + e)),
                        Vector256.LoadUnsafe(ref sf, (nuint)((3 * stride) + e)),
                        Vector256.LoadUnsafe(ref sf, (nuint)((4 * stride) + e)),
                        Vector256.LoadUnsafe(ref sf, (nuint)((5 * stride) + e)),
                        Vector256.LoadUnsafe(ref sf, (nuint)((6 * stride) + e)),
                        Vector256.LoadUnsafe(ref sf, (nuint)((7 * stride) + e)),
                        ref Unsafe.Add(ref df, e * 8));
                }
            }
            else if (typeof(T) == typeof(double))
            {
                ref double sd = ref Unsafe.As<T, double>(ref Unsafe.Add(ref s, start));
                ref double dd = ref Unsafe.As<T, double>(ref Unsafe.Add(ref d, dstElem * 4));
                for (; e <= count - 4; e += 4)
                {
                    Transpose4(
                        Vector256.LoadUnsafe(ref sd, (nuint)e),
                        Vector256.LoadUnsafe(ref sd, (nuint)(stride + e)),
                        Vector256.LoadUnsafe(ref sd, (nuint)((2 * stride) + e)),
                        Vector256.LoadUnsafe(ref sd, (nuint)((3 * stride) + e)),
                        ref Unsafe.Add(ref dd, e * 4));
                }
            }
        }

        for (int l = 0; l < lanes; l++)
        {
            ReadOnlySpan<T> row = src.AsSpan(start + (l * stride), count);
            for (int i = e; i < count; i++)
            {
                dst[((dstElem + i) * laneCount) + l] = row[i];
            }
        }
    }

    /// <summary>
    /// <c>dst[start + l·stride + e] = src[(srcElem + e)·L + l]</c> for <c>e &lt; count</c> and
    /// <c>l &lt; lanes</c>.
    /// </summary>
    public static void LanesToRows<T>(ReadOnlySpan<T> src, int srcElem, int lanes, int count, T[] dst, int start, int stride)
        where T : unmanaged
    {
        int laneCount = Vector<T>.Count;
        int e = 0;
        if (lanes == laneCount && Avx.IsSupported && Vector<T>.Count * Unsafe.SizeOf<T>() == 32)
        {
            ref T s = ref MemoryMarshal.GetReference(src);
            ref T d = ref MemoryMarshal.GetArrayDataReference(dst);
            if (typeof(T) == typeof(float))
            {
                ref float sf = ref Unsafe.As<T, float>(ref Unsafe.Add(ref s, srcElem * 8));
                ref float df = ref Unsafe.As<T, float>(ref Unsafe.Add(ref d, start));
                Span<float> tmp = stackalloc float[64];
                ref float t = ref MemoryMarshal.GetReference(tmp);
                for (; e <= count - 8; e += 8)
                {
                    ref float block = ref Unsafe.Add(ref sf, e * 8);
                    Transpose8(
                        Vector256.LoadUnsafe(ref block),
                        Vector256.LoadUnsafe(ref block, 8),
                        Vector256.LoadUnsafe(ref block, 16),
                        Vector256.LoadUnsafe(ref block, 24),
                        Vector256.LoadUnsafe(ref block, 32),
                        Vector256.LoadUnsafe(ref block, 40),
                        Vector256.LoadUnsafe(ref block, 48),
                        Vector256.LoadUnsafe(ref block, 56),
                        ref t);
                    for (int l = 0; l < 8; l++)
                    {
                        Vector256.LoadUnsafe(ref t, (nuint)(l * 8)).StoreUnsafe(ref df, (nuint)((l * stride) + e));
                    }
                }
            }
            else if (typeof(T) == typeof(double))
            {
                ref double sd = ref Unsafe.As<T, double>(ref Unsafe.Add(ref s, srcElem * 4));
                ref double dd = ref Unsafe.As<T, double>(ref Unsafe.Add(ref d, start));
                Span<double> tmp = stackalloc double[16];
                ref double t = ref MemoryMarshal.GetReference(tmp);
                for (; e <= count - 4; e += 4)
                {
                    ref double block = ref Unsafe.Add(ref sd, e * 4);
                    Transpose4(
                        Vector256.LoadUnsafe(ref block),
                        Vector256.LoadUnsafe(ref block, 4),
                        Vector256.LoadUnsafe(ref block, 8),
                        Vector256.LoadUnsafe(ref block, 12),
                        ref t);
                    for (int l = 0; l < 4; l++)
                    {
                        Vector256.LoadUnsafe(ref t, (nuint)(l * 4)).StoreUnsafe(ref dd, (nuint)((l * stride) + e));
                    }
                }
            }
        }

        for (int l = 0; l < lanes; l++)
        {
            Span<T> row = dst.AsSpan(start + (l * stride), count);
            for (int i = e; i < count; i++)
            {
                row[i] = src[((srcElem + i) * laneCount) + l];
            }
        }
    }

    /// <summary>
    /// Splits <c>lanes</c> interleaved (re, im) pairs at <paramref name="src"/> into lane-major
    /// element <paramref name="elem"/> (real parts) and <paramref name="elem"/> + 1 (imaginary parts).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Deinterleave<T>(ReadOnlySpan<T> src, int lanes, Span<T> dst, int elem)
        where T : unmanaged
    {
        int laneCount = Vector<T>.Count;
        int re = elem * laneCount, im = re + laneCount;
        if (lanes == laneCount && Avx2.IsSupported && Vector<T>.Count * Unsafe.SizeOf<T>() == 32)
        {
            if (typeof(T) == typeof(float))
            {
                ref float s = ref Unsafe.As<T, float>(ref MemoryMarshal.GetReference(src));
                ref float d = ref Unsafe.As<T, float>(ref MemoryMarshal.GetReference(dst));
                Vector256<float> a = Vector256.LoadUnsafe(ref s);
                Vector256<float> b = Vector256.LoadUnsafe(ref s, 8);
                Vector256<double> r = Avx.Shuffle(a, b, 0x88).AsDouble();
                Vector256<double> i = Avx.Shuffle(a, b, 0xDD).AsDouble();
                Avx2.Permute4x64(r, 0xD8).AsSingle().StoreUnsafe(ref d, (nuint)re);
                Avx2.Permute4x64(i, 0xD8).AsSingle().StoreUnsafe(ref d, (nuint)im);
                return;
            }

            if (typeof(T) == typeof(double))
            {
                ref double s = ref Unsafe.As<T, double>(ref MemoryMarshal.GetReference(src));
                ref double d = ref Unsafe.As<T, double>(ref MemoryMarshal.GetReference(dst));
                Vector256<double> a = Vector256.LoadUnsafe(ref s);
                Vector256<double> b = Vector256.LoadUnsafe(ref s, 4);
                Avx2.Permute4x64(Avx.UnpackLow(a, b), 0xD8).StoreUnsafe(ref d, (nuint)re);
                Avx2.Permute4x64(Avx.UnpackHigh(a, b), 0xD8).StoreUnsafe(ref d, (nuint)im);
                return;
            }
        }

        for (int l = 0; l < lanes; l++)
        {
            dst[re + l] = src[2 * l];
            dst[im + l] = src[(2 * l) + 1];
        }
    }

    /// <summary>The inverse of <see cref="Deinterleave{T}"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Interleave<T>(ReadOnlySpan<T> src, int elem, int lanes, Span<T> dst)
        where T : unmanaged
    {
        int laneCount = Vector<T>.Count;
        int re = elem * laneCount, im = re + laneCount;
        if (lanes == laneCount && Avx.IsSupported && Vector<T>.Count * Unsafe.SizeOf<T>() == 32)
        {
            if (typeof(T) == typeof(float))
            {
                ref float s = ref Unsafe.As<T, float>(ref MemoryMarshal.GetReference(src));
                ref float d = ref Unsafe.As<T, float>(ref MemoryMarshal.GetReference(dst));
                Vector256<float> r = Vector256.LoadUnsafe(ref s, (nuint)re);
                Vector256<float> i = Vector256.LoadUnsafe(ref s, (nuint)im);
                Vector256<float> lo = Avx.UnpackLow(r, i);
                Vector256<float> hi = Avx.UnpackHigh(r, i);
                Avx.Permute2x128(lo, hi, 0x20).StoreUnsafe(ref d);
                Avx.Permute2x128(lo, hi, 0x31).StoreUnsafe(ref d, 8);
                return;
            }

            if (typeof(T) == typeof(double))
            {
                ref double s = ref Unsafe.As<T, double>(ref MemoryMarshal.GetReference(src));
                ref double d = ref Unsafe.As<T, double>(ref MemoryMarshal.GetReference(dst));
                Vector256<double> r = Vector256.LoadUnsafe(ref s, (nuint)re);
                Vector256<double> i = Vector256.LoadUnsafe(ref s, (nuint)im);
                Vector256<double> lo = Avx.UnpackLow(r, i);
                Vector256<double> hi = Avx.UnpackHigh(r, i);
                Avx.Permute2x128(lo, hi, 0x20).StoreUnsafe(ref d);
                Avx.Permute2x128(lo, hi, 0x31).StoreUnsafe(ref d, 4);
                return;
            }
        }

        for (int l = 0; l < lanes; l++)
        {
            dst[2 * l] = src[re + l];
            dst[(2 * l) + 1] = src[im + l];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Transpose8(
        Vector256<float> r0, Vector256<float> r1, Vector256<float> r2, Vector256<float> r3,
        Vector256<float> r4, Vector256<float> r5, Vector256<float> r6, Vector256<float> r7,
        ref float dst)
    {
        Vector256<float> t0 = Avx.UnpackLow(r0, r1), t1 = Avx.UnpackHigh(r0, r1);
        Vector256<float> t2 = Avx.UnpackLow(r2, r3), t3 = Avx.UnpackHigh(r2, r3);
        Vector256<float> t4 = Avx.UnpackLow(r4, r5), t5 = Avx.UnpackHigh(r4, r5);
        Vector256<float> t6 = Avx.UnpackLow(r6, r7), t7 = Avx.UnpackHigh(r6, r7);
        Vector256<float> u0 = Avx.Shuffle(t0, t2, 0x44), u1 = Avx.Shuffle(t0, t2, 0xEE);
        Vector256<float> u2 = Avx.Shuffle(t1, t3, 0x44), u3 = Avx.Shuffle(t1, t3, 0xEE);
        Vector256<float> u4 = Avx.Shuffle(t4, t6, 0x44), u5 = Avx.Shuffle(t4, t6, 0xEE);
        Vector256<float> u6 = Avx.Shuffle(t5, t7, 0x44), u7 = Avx.Shuffle(t5, t7, 0xEE);
        Avx.Permute2x128(u0, u4, 0x20).StoreUnsafe(ref dst);
        Avx.Permute2x128(u1, u5, 0x20).StoreUnsafe(ref dst, 8);
        Avx.Permute2x128(u2, u6, 0x20).StoreUnsafe(ref dst, 16);
        Avx.Permute2x128(u3, u7, 0x20).StoreUnsafe(ref dst, 24);
        Avx.Permute2x128(u0, u4, 0x31).StoreUnsafe(ref dst, 32);
        Avx.Permute2x128(u1, u5, 0x31).StoreUnsafe(ref dst, 40);
        Avx.Permute2x128(u2, u6, 0x31).StoreUnsafe(ref dst, 48);
        Avx.Permute2x128(u3, u7, 0x31).StoreUnsafe(ref dst, 56);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Transpose4(Vector256<double> r0, Vector256<double> r1, Vector256<double> r2, Vector256<double> r3, ref double dst)
    {
        Vector256<double> t0 = Avx.UnpackLow(r0, r1), t1 = Avx.UnpackHigh(r0, r1);
        Vector256<double> t2 = Avx.UnpackLow(r2, r3), t3 = Avx.UnpackHigh(r2, r3);
        Avx.Permute2x128(t0, t2, 0x20).StoreUnsafe(ref dst);
        Avx.Permute2x128(t1, t3, 0x20).StoreUnsafe(ref dst, 4);
        Avx.Permute2x128(t0, t2, 0x31).StoreUnsafe(ref dst, 8);
        Avx.Permute2x128(t1, t3, 0x31).StoreUnsafe(ref dst, 12);
    }
}
