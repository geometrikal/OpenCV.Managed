namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// Deinterleaves and interleaves 3- and 4-channel pixel data held in <see cref="Vector128{T}"/>
/// registers with byte shuffles. A plane vector holds <c>16 / elementSize</c> elements of one
/// channel; the interleaved data of one block is <c>channels</c> consecutive vectors.
/// </summary>
internal static class ColorVectors
{
    /// <summary>True when a byte shuffle with zeroing of out-of-range indices is available.</summary>
    public static bool IsSupported => Ssse3.IsSupported || AdvSimd.Arm64.IsSupported;

    private static readonly Vector128<byte>[] Deinterleave3B = DeinterleaveMasks(3, 1);
    private static readonly Vector128<byte>[] Deinterleave4B = DeinterleaveMasks(4, 1);
    private static readonly Vector128<byte>[] Deinterleave3W = DeinterleaveMasks(3, 2);
    private static readonly Vector128<byte>[] Deinterleave4W = DeinterleaveMasks(4, 2);
    private static readonly Vector128<byte>[] Deinterleave3D = DeinterleaveMasks(3, 4);
    private static readonly Vector128<byte>[] Deinterleave4D = DeinterleaveMasks(4, 4);
    private static readonly Vector128<byte>[] Interleave3B = InterleaveMasks(3, 1);
    private static readonly Vector128<byte>[] Interleave4B = InterleaveMasks(4, 1);
    private static readonly Vector128<byte>[] Interleave3W = InterleaveMasks(3, 2);
    private static readonly Vector128<byte>[] Interleave4W = InterleaveMasks(4, 2);
    private static readonly Vector128<byte>[] Interleave3D = InterleaveMasks(3, 4);
    private static readonly Vector128<byte>[] Interleave4D = InterleaveMasks(4, 4);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<byte> Lookup(Vector128<byte> v, Vector128<byte> mask)
    {
        if (Ssse3.IsSupported)
        {
            return Ssse3.Shuffle(v, mask);
        }

        if (AdvSimd.Arm64.IsSupported)
        {
            return AdvSimd.Arm64.VectorTableLookup(v, mask);
        }

        return Vector128.Shuffle(v, mask);
    }

    private static Vector128<byte>[] Masks(int cn, int elementSize, bool deinterleave)
        => (cn, elementSize, deinterleave) switch
        {
            (3, 1, true) => Deinterleave3B,
            (4, 1, true) => Deinterleave4B,
            (3, 2, true) => Deinterleave3W,
            (4, 2, true) => Deinterleave4W,
            (3, 4, true) => Deinterleave3D,
            (4, 4, true) => Deinterleave4D,
            (3, 1, false) => Interleave3B,
            (4, 1, false) => Interleave4B,
            (3, 2, false) => Interleave3W,
            (4, 2, false) => Interleave4W,
            (3, 4, false) => Interleave3D,
            _ => Interleave4D,
        };

    /// <summary>
    /// Splits <paramref name="cn"/> consecutive vectors at <paramref name="src"/> into channel
    /// planes <paramref name="p0"/> to <paramref name="p3"/> (<paramref name="p3"/> is zero when
    /// <paramref name="cn"/> is 3).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Deinterleave<T>(
        ref T src, int cn, out Vector128<byte> p0, out Vector128<byte> p1, out Vector128<byte> p2, out Vector128<byte> p3)
        where T : unmanaged
    {
        ref byte s = ref Unsafe.As<T, byte>(ref src);
        Vector128<byte>[] m = Masks(cn, Unsafe.SizeOf<T>(), true);
        Vector128<byte> v0 = Vector128.LoadUnsafe(ref s);
        Vector128<byte> v1 = Vector128.LoadUnsafe(ref s, 16);
        Vector128<byte> v2 = Vector128.LoadUnsafe(ref s, 32);
        if (cn == 3)
        {
            p0 = Lookup(v0, m[0]) | Lookup(v1, m[1]) | Lookup(v2, m[2]);
            p1 = Lookup(v0, m[3]) | Lookup(v1, m[4]) | Lookup(v2, m[5]);
            p2 = Lookup(v0, m[6]) | Lookup(v1, m[7]) | Lookup(v2, m[8]);
            p3 = Vector128<byte>.Zero;
            return;
        }

        Vector128<byte> v3 = Vector128.LoadUnsafe(ref s, 48);
        p0 = Lookup(v0, m[0]) | Lookup(v1, m[1]) | Lookup(v2, m[2]) | Lookup(v3, m[3]);
        p1 = Lookup(v0, m[4]) | Lookup(v1, m[5]) | Lookup(v2, m[6]) | Lookup(v3, m[7]);
        p2 = Lookup(v0, m[8]) | Lookup(v1, m[9]) | Lookup(v2, m[10]) | Lookup(v3, m[11]);
        p3 = Lookup(v0, m[12]) | Lookup(v1, m[13]) | Lookup(v2, m[14]) | Lookup(v3, m[15]);
    }

    /// <summary>Interleaves planes into <paramref name="cn"/> consecutive vectors at <paramref name="dst"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Interleave<T>(
        ref T dst, int cn, Vector128<byte> p0, Vector128<byte> p1, Vector128<byte> p2, Vector128<byte> p3)
        where T : unmanaged
    {
        ref byte d = ref Unsafe.As<T, byte>(ref dst);
        Vector128<byte>[] m = Masks(cn, Unsafe.SizeOf<T>(), false);
        if (cn == 3)
        {
            (Lookup(p0, m[0]) | Lookup(p1, m[1]) | Lookup(p2, m[2])).StoreUnsafe(ref d);
            (Lookup(p0, m[3]) | Lookup(p1, m[4]) | Lookup(p2, m[5])).StoreUnsafe(ref d, 16);
            (Lookup(p0, m[6]) | Lookup(p1, m[7]) | Lookup(p2, m[8])).StoreUnsafe(ref d, 32);
            return;
        }

        (Lookup(p0, m[0]) | Lookup(p1, m[1]) | Lookup(p2, m[2]) | Lookup(p3, m[3])).StoreUnsafe(ref d);
        (Lookup(p0, m[4]) | Lookup(p1, m[5]) | Lookup(p2, m[6]) | Lookup(p3, m[7])).StoreUnsafe(ref d, 16);
        (Lookup(p0, m[8]) | Lookup(p1, m[9]) | Lookup(p2, m[10]) | Lookup(p3, m[11])).StoreUnsafe(ref d, 32);
        (Lookup(p0, m[12]) | Lookup(p1, m[13]) | Lookup(p2, m[14]) | Lookup(p3, m[15])).StoreUnsafe(ref d, 48);
    }

    /// <summary>Mask <c>[c * cn + s]</c> moves the bytes of channel <c>c</c> held in source vector <c>s</c>.</summary>
    private static Vector128<byte>[] DeinterleaveMasks(int cn, int e)
    {
        var masks = new Vector128<byte>[cn * cn];
        for (int c = 0; c < cn; c++)
        {
            for (int s = 0; s < cn; s++)
            {
                var m = new byte[16];
                for (int j = 0; j < 16; j++)
                {
                    int o = (((j / e) * cn) + c) * e + (j % e);
                    m[j] = o / 16 == s ? (byte)(o % 16) : (byte)0x80;
                }

                masks[(c * cn) + s] = Vector128.Create(m);
            }
        }

        return masks;
    }

    /// <summary>Mask <c>[t * cn + c]</c> moves the bytes of plane <c>c</c> into output vector <c>t</c>.</summary>
    private static Vector128<byte>[] InterleaveMasks(int cn, int e)
    {
        var masks = new Vector128<byte>[cn * cn];
        for (int t = 0; t < cn; t++)
        {
            for (int c = 0; c < cn; c++)
            {
                var m = new byte[16];
                for (int j = 0; j < 16; j++)
                {
                    int o = (t * 16) + j;
                    int k = o / (cn * e);
                    int ch = (o / e) % cn;
                    m[j] = ch == c ? (byte)((k * e) + (o % e)) : (byte)0x80;
                }

                masks[(t * cn) + c] = Vector128.Create(m);
            }
        }

        return masks;
    }

    /// <summary>Widens 16 bytes to four vectors of 4 <see cref="int"/>, in element order.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void WidenToInt32(
        Vector128<byte> v, out Vector128<int> w0, out Vector128<int> w1, out Vector128<int> w2, out Vector128<int> w3)
    {
        Vector128<ushort> lo = Vector128.WidenLower(v);
        Vector128<ushort> hi = Vector128.WidenUpper(v);
        w0 = Vector128.WidenLower(lo).AsInt32();
        w1 = Vector128.WidenUpper(lo).AsInt32();
        w2 = Vector128.WidenLower(hi).AsInt32();
        w3 = Vector128.WidenUpper(hi).AsInt32();
    }

    /// <summary>Saturates four int vectors to 16 bytes, in element order.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<byte> SaturateToByte(Vector128<int> a, Vector128<int> b, Vector128<int> c, Vector128<int> d)
    {
        Vector128<int> zero = Vector128<int>.Zero, max = Vector128.Create(255);
        Vector128<ushort> lo = Vector128.Narrow(Vector128.Min(Vector128.Max(a, zero), max), Vector128.Min(Vector128.Max(b, zero), max)).AsUInt16();
        Vector128<ushort> hi = Vector128.Narrow(Vector128.Min(Vector128.Max(c, zero), max), Vector128.Min(Vector128.Max(d, zero), max)).AsUInt16();
        return Vector128.Narrow(lo, hi);
    }

    /// <summary>Saturates two int vectors to 8 <see cref="ushort"/>, in element order.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector128<ushort> SaturateToUInt16(Vector128<int> a, Vector128<int> b)
    {
        Vector128<int> zero = Vector128<int>.Zero, max = Vector128.Create(65535);
        return Vector128.Narrow(Vector128.Min(Vector128.Max(a, zero), max).AsUInt32(), Vector128.Min(Vector128.Max(b, zero), max).AsUInt32());
    }
}
