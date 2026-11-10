namespace Geometrikal.OpenCV.Managed.Core.Internal;

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// Maps bytes through a 256-entry byte table. With AVX-512 VBMI, two 128-entry permutes and a
/// blend map 64 bytes per step; other hardware uses a scalar loop.
/// </summary>
internal static class ByteLut
{
    /// <summary>Selects the VBMI path. Tests clear it to run the scalar path on VBMI hardware.</summary>
    internal static bool UseVbmi = Avx512Vbmi.IsSupported;

    /// <summary>
    /// <c>dst[i] = lut[src[i]]</c>. <paramref name="src"/> and <paramref name="dst"/> may be the same
    /// memory; partially overlapping spans are not supported.
    /// </summary>
    public static void Apply(ReadOnlySpan<byte> src, Span<byte> dst, ReadOnlySpan<byte> lut)
    {
        int n = src.Length;
        ref byte s = ref MemoryMarshal.GetReference(src);
        ref byte d = ref MemoryMarshal.GetReference(dst);
        ref byte t = ref MemoryMarshal.GetReference(lut);
        if (UseVbmi && n >= Vector512<byte>.Count)
        {
            ApplyVbmi(ref s, ref d, n, ref t);
            return;
        }

        ApplyScalar(ref s, ref d, 0, n, ref t);
    }

    private static void ApplyVbmi(ref byte s, ref byte d, int n, ref byte t)
    {
        int vc = Vector512<byte>.Count;
        Vector512<byte> t0 = Vector512.LoadUnsafe(ref t);
        Vector512<byte> t1 = Vector512.LoadUnsafe(ref t, 64);
        Vector512<byte> t2 = Vector512.LoadUnsafe(ref t, 128);
        Vector512<byte> t3 = Vector512.LoadUnsafe(ref t, 192);

        // The final vector overlaps the previous one; it is computed before the first store so
        // that an in-place call reads only source values.
        Vector512<byte> last = Lookup(Vector512.LoadUnsafe(ref s, (nuint)(n - vc)), t0, t1, t2, t3);
        for (int i = 0; i <= n - vc; i += vc)
        {
            Lookup(Vector512.LoadUnsafe(ref s, (nuint)i), t0, t1, t2, t3).StoreUnsafe(ref d, (nuint)i);
        }

        last.StoreUnsafe(ref d, (nuint)(n - vc));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<byte> Lookup(
        Vector512<byte> idx, Vector512<byte> t0, Vector512<byte> t1, Vector512<byte> t2, Vector512<byte> t3)
    {
        Vector512<byte> lo = Avx512Vbmi.PermuteVar64x8x2(t0, idx, t1);
        Vector512<byte> hi = Avx512Vbmi.PermuteVar64x8x2(t2, idx, t3);
        return Vector512.ConditionalSelect(Vector512.GreaterThan(idx, Vector512.Create((byte)127)), hi, lo);
    }

    private static void ApplyScalar(ref byte s, ref byte d, int i, int n, ref byte t)
    {
        for (; i <= n - 4; i += 4)
        {
            byte a = Unsafe.Add(ref t, Unsafe.Add(ref s, i));
            byte b = Unsafe.Add(ref t, Unsafe.Add(ref s, i + 1));
            byte c = Unsafe.Add(ref t, Unsafe.Add(ref s, i + 2));
            byte e = Unsafe.Add(ref t, Unsafe.Add(ref s, i + 3));
            Unsafe.Add(ref d, i) = a;
            Unsafe.Add(ref d, i + 1) = b;
            Unsafe.Add(ref d, i + 2) = c;
            Unsafe.Add(ref d, i + 3) = e;
        }

        for (; i < n; i++)
        {
            Unsafe.Add(ref d, i) = Unsafe.Add(ref t, Unsafe.Add(ref s, i));
        }
    }
}
