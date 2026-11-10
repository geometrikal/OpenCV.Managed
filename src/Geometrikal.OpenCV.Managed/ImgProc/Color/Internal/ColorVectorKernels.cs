// This implementation is ported from the OpenCV colour conversion vector kernels.
//
// Original sources:
//   modules/imgproc/src/color_rgb.simd.hpp (RGB2RGB, RGB2Gray, Gray2RGB)
//   modules/imgproc/src/color_hsv.simd.hpp (RGB2HSV_b, RGB2HSV_f, HSV2RGB_simd, HSV2RGB_b)
//   modules/imgproc/src/color_lab.cpp (RGB2XYZ_i, RGB2XYZ_f, XYZ2RGB_i, XYZ2RGB_f)
//   modules/imgproc/src/color_yuv.simd.hpp (RGB2YCrCb_i, RGB2YCrCb_f, YCrCb2RGB_i, YCrCb2RGB_f)
//
// OpenCV is licensed under the Apache License 2.0 (see src/OpenCV/LICENSE).

namespace Geometrikal.OpenCV.Managed.ImgProc.Internal;

using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Geometrikal.OpenCV.Managed.Core.Internal;

/// <summary>
/// <see cref="Vector128{T}"/> bodies of the colour kernels. Each method converts whole blocks
/// from the start of the row and returns the number of pixels done; the caller converts the
/// rest. Every lane uses the same operations, in the same order, as the scalar kernel that
/// would otherwise process that pixel.
/// </summary>
internal static class ColorVectorKernels
{
    private struct Lanes4
    {
        public Vector128<int> A;
        public Vector128<int> B;
        public Vector128<int> C;
        public Vector128<int> D;
    }

    private readonly struct Coefficients3(int c0, int c1, int c2)
    {
        public Vector128<int> C0 { get; } = Vector128.Create(c0);

        public Vector128<int> C1 { get; } = Vector128.Create(c1);

        public Vector128<int> C2 { get; } = Vector128.Create(c2);
    }

    private static int BlockPixels<T>()
        where T : unmanaged
        => 16 / Unsafe.SizeOf<T>();

    public static int Swap<T>(ReadOnlySpan<T> src, Span<T> dst, int width, int scn, int dcn, int bi)
        where T : unmanaged
    {
        if (!ColorVectors.IsSupported)
        {
            return 0;
        }

        int n = BlockPixels<T>();
        Vector128<byte> alpha = Vector128.Create(ColorMath.MaxValue<T>()).AsByte();
        ref T s = ref MemoryMarshal.GetReference(src);
        ref T d = ref MemoryMarshal.GetReference(dst);
        int x = 0;
        for (; x <= width - n; x += n)
        {
            ColorVectors.Deinterleave(ref Unsafe.Add(ref s, x * scn), scn, out var p0, out var p1, out var p2, out var p3);
            ColorVectors.Interleave(
                ref Unsafe.Add(ref d, x * dcn), dcn, bi == 0 ? p0 : p2, p1, bi == 0 ? p2 : p0, scn == 4 ? p3 : alpha);
        }

        return x;
    }

    public static int GrayToColor<T>(ReadOnlySpan<T> src, Span<T> dst, int width, int dcn)
        where T : unmanaged
    {
        if (!ColorVectors.IsSupported)
        {
            return 0;
        }

        int n = BlockPixels<T>();
        Vector128<byte> alpha = Vector128.Create(ColorMath.MaxValue<T>()).AsByte();
        ref byte s = ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(src));
        ref T d = ref MemoryMarshal.GetReference(dst);
        int x = 0;
        for (; x <= width - n; x += n)
        {
            Vector128<byte> p = Vector128.LoadUnsafe(ref s, (nuint)(x * Unsafe.SizeOf<T>()));
            ColorVectors.Interleave(ref Unsafe.Add(ref d, x * dcn), dcn, p, p, p, alpha);
        }

        return x;
    }

    /// <summary>
    /// Loads one block and widens each channel to <see cref="int"/>. Bytes fill lanes A to D
    /// (16 pixels); <see cref="ushort"/> fills A and B (8 pixels).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void LoadInt<T>(ref T src, int scn, out Lanes4 c0, out Lanes4 c1, out Lanes4 c2)
        where T : unmanaged
    {
        ColorVectors.Deinterleave(ref src, scn, out var p0, out var p1, out var p2, out _);
        c0 = Widen<T>(p0);
        c1 = Widen<T>(p1);
        c2 = Widen<T>(p2);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Lanes4 Widen<T>(Vector128<byte> p)
        where T : unmanaged
    {
        Lanes4 r;
        if (typeof(T) == typeof(byte))
        {
            ColorVectors.WidenToInt32(p, out r.A, out r.B, out r.C, out r.D);
            return r;
        }

        r.A = Vector128.WidenLower(p.AsUInt16()).AsInt32();
        r.B = Vector128.WidenUpper(p.AsUInt16()).AsInt32();
        r.C = default;
        r.D = default;
        return r;
    }

    /// <summary>Saturates int lanes to one plane of <typeparamref name="T"/> (byte or ushort).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<byte> Pack<T>(in Lanes4 v)
        where T : unmanaged
        => typeof(T) == typeof(byte)
            ? ColorVectors.SaturateToByte(v.A, v.B, v.C, v.D)
            : ColorVectors.SaturateToUInt16(v.A, v.B).AsByte();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> Descale(Vector128<int> x, int n)
        => Vector128.ShiftRightArithmetic(x + Vector128.Create(1 << (n - 1)), n);

    /// <summary>Gray for byte and ushort: <c>(s0*c0 + s1*c1 + s2*c2 + 2^14) >> 15</c>.</summary>
    public static int GrayInt<T>(ReadOnlySpan<T> src, Span<T> dst, int width, int scn, int c0, int c1, int c2)
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte) && Ssse3.IsSupported)
        {
            return GrayByteMultiplyAdd(MemoryMarshal.Cast<T, byte>(src), MemoryMarshal.Cast<T, byte>(dst), width, scn, c0, c1, c2);
        }

        if (!ColorVectors.IsSupported)
        {
            return 0;
        }

        int n = BlockPixels<T>();
        var k = new Coefficients3(c0, c1, c2);
        ref T s = ref MemoryMarshal.GetReference(src);
        ref byte d = ref Unsafe.As<T, byte>(ref MemoryMarshal.GetReference(dst));
        int x = 0;
        for (; x <= width - n; x += n)
        {
            LoadInt(ref Unsafe.Add(ref s, x * scn), scn, out var a, out var b, out var c);
            Lanes4 y;
            y.A = Gray(a.A, b.A, c.A, k);
            y.B = Gray(a.B, b.B, c.B, k);
            y.C = typeof(T) == typeof(byte) ? Gray(a.C, b.C, c.C, k) : default;
            y.D = typeof(T) == typeof(byte) ? Gray(a.D, b.D, c.D, k) : default;
            Pack<T>(y).StoreUnsafe(ref d, (nuint)(x * Unsafe.SizeOf<T>()));
        }

        return x;
    }

    private static readonly Vector128<byte>[] GrayPairMasks3 = GrayPairMasks(3);
    private static readonly Vector128<byte>[] GrayPairMasks4 = GrayPairMasks(4);

    /// <summary>
    /// Masks for 4 pixels at byte offset <c>o</c> of a loaded vector: index <c>2*o'</c> builds
    /// 16-bit (c0-channel, c1-channel) pairs, <c>2*o'+1</c> builds (c2-channel, 0) pairs, for
    /// <c>o'</c> = 0 (offset 0) and 1 (offset 4).
    /// </summary>
    private static Vector128<byte>[] GrayPairMasks(int scn)
    {
        var masks = new Vector128<byte>[4];
        for (int variant = 0; variant < 2; variant++)
        {
            int o = variant * 4;
            var bg = new byte[16];
            var r = new byte[16];
            for (int p = 0; p < 4; p++)
            {
                int i = o + (p * scn);
                bg[4 * p] = (byte)i;
                bg[(4 * p) + 1] = 0x80;
                bg[(4 * p) + 2] = (byte)(i + 1);
                bg[(4 * p) + 3] = 0x80;
                r[4 * p] = (byte)(i + 2);
                r[(4 * p) + 1] = 0x80;
                r[(4 * p) + 2] = 0x80;
                r[(4 * p) + 3] = 0x80;
            }

            masks[2 * variant] = Vector128.Create(bg);
            masks[(2 * variant) + 1] = Vector128.Create(r);
        }

        return masks;
    }

    /// <summary>
    /// Byte gray with 16-bit multiply-add (<c>pmaddwd</c>): (s0, s1) pairs times (c0, c1) plus
    /// (s2, 1) pairs times (c2, 2^14). The sums equal the 32-bit scalar sums exactly.
    /// </summary>
    private static int GrayByteMultiplyAdd(ReadOnlySpan<byte> src, Span<byte> dst, int width, int scn, int c0, int c1, int c2)
    {
        Vector128<byte>[] m = scn == 3 ? GrayPairMasks3 : GrayPairMasks4;
        Vector128<short> k01 = Vector128.Create((short)c0, (short)c1, (short)c0, (short)c1, (short)c0, (short)c1, (short)c0, (short)c1);
        Vector128<short> k2 = Vector128.Create((short)c2, 1 << 14, (short)c2, 1 << 14, (short)c2, 1 << 14, (short)c2, 1 << 14);
        Vector128<short> one = Vector128.Create(0, 1, 0, 1, 0, 1, 0, 1);
        ref byte s = ref MemoryMarshal.GetReference(src);
        ref byte d = ref MemoryMarshal.GetReference(dst);
        int x = 0;
        for (; x <= width - 16; x += 16)
        {
            ref byte block = ref Unsafe.Add(ref s, x * scn);
            Vector128<int> y0 = GroupGray(Vector128.LoadUnsafe(ref block), m[0], m[1], k01, k2, one);
            Vector128<int> y1 = GroupGray(Vector128.LoadUnsafe(ref block, (nuint)(4 * scn)), m[0], m[1], k01, k2, one);
            Vector128<int> y2 = GroupGray(Vector128.LoadUnsafe(ref block, (nuint)(8 * scn)), m[0], m[1], k01, k2, one);
            Vector128<int> y3 = scn == 3
                ? GroupGray(Vector128.LoadUnsafe(ref block, 32), m[2], m[3], k01, k2, one)
                : GroupGray(Vector128.LoadUnsafe(ref block, 48), m[0], m[1], k01, k2, one);
            Vector128<short> lo = Sse2.PackSignedSaturate(y0, y1);
            Vector128<short> hi = Sse2.PackSignedSaturate(y2, y3);
            Sse2.PackUnsignedSaturate(lo, hi).StoreUnsafe(ref d, (nuint)x);
        }

        return x;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> GroupGray(
        Vector128<byte> v, Vector128<byte> mbg, Vector128<byte> mr, Vector128<short> k01, Vector128<short> k2, Vector128<short> one)
    {
        Vector128<short> bg = Ssse3.Shuffle(v, mbg).AsInt16();
        Vector128<short> r1 = Ssse3.Shuffle(v, mr).AsInt16() | one;
        Vector128<int> sum = Sse2.MultiplyAddAdjacent(bg, k01) + Sse2.MultiplyAddAdjacent(r1, k2);
        return Vector128.ShiftRightLogical(sum.AsUInt32(), 15).AsInt32();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> Gray(Vector128<int> a, Vector128<int> b, Vector128<int> c, Coefficients3 k)
        => Vector128.ShiftRightLogical(((a * k.C0) + (b * k.C1) + (c * k.C2)).AsUInt32() + Vector128.Create(1u << 14), 15).AsInt32();

    public static int GrayFloat(ReadOnlySpan<float> src, Span<float> dst, int width, int scn, float c0, float c1, float c2)
    {
        if (!ColorVectors.IsSupported)
        {
            return 0;
        }

        ref float s = ref MemoryMarshal.GetReference(src);
        ref float d = ref MemoryMarshal.GetReference(dst);
        int x = 0;
        for (; x <= width - 4; x += 4)
        {
            ColorVectors.Deinterleave(ref Unsafe.Add(ref s, x * scn), scn, out var p0, out var p1, out var p2, out _);
            ((p0.AsSingle() * c0) + (p1.AsSingle() * c1) + (p2.AsSingle() * c2)).StoreUnsafe(ref d, (nuint)x);
        }

        return x;
    }

    /// <summary>3x3 integer transform with 12-bit coefficients (XYZ).</summary>
    public static int MatrixInt<T>(ReadOnlySpan<T> src, Span<T> dst, int width, int scn, int dcn, int[] coefficients)
        where T : unmanaged
    {
        if (!ColorVectors.IsSupported)
        {
            return 0;
        }

        int n = BlockPixels<T>();
        var k0 = new Coefficients3(coefficients[0], coefficients[1], coefficients[2]);
        var k1 = new Coefficients3(coefficients[3], coefficients[4], coefficients[5]);
        var k2 = new Coefficients3(coefficients[6], coefficients[7], coefficients[8]);
        Vector128<byte> alpha = Vector128.Create(ColorMath.MaxValue<T>()).AsByte();
        ref T s = ref MemoryMarshal.GetReference(src);
        ref T d = ref MemoryMarshal.GetReference(dst);
        int x = 0;
        for (; x <= width - n; x += n)
        {
            LoadInt(ref Unsafe.Add(ref s, x * scn), scn, out var a, out var b, out var c);
            ColorVectors.Interleave(
                ref Unsafe.Add(ref d, x * dcn), dcn,
                Pack<T>(Row<T>(a, b, c, k0)), Pack<T>(Row<T>(a, b, c, k1)), Pack<T>(Row<T>(a, b, c, k2)), alpha);
        }

        return x;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Lanes4 Row<T>(in Lanes4 a, in Lanes4 b, in Lanes4 c, Coefficients3 k)
        where T : unmanaged
    {
        Lanes4 r;
        r.A = Descale((a.A * k.C0) + (b.A * k.C1) + (c.A * k.C2), 12);
        r.B = Descale((a.B * k.C0) + (b.B * k.C1) + (c.B * k.C2), 12);
        r.C = typeof(T) == typeof(byte) ? Descale((a.C * k.C0) + (b.C * k.C1) + (c.C * k.C2), 12) : default;
        r.D = typeof(T) == typeof(byte) ? Descale((a.D * k.C0) + (b.D * k.C1) + (c.D * k.C2), 12) : default;
        return r;
    }

    /// <summary>3x3 float transform, vector-body association <c>s0*c0 + (s1*c1 + s2*c2)</c>.</summary>
    public static int MatrixFloat(ReadOnlySpan<float> src, Span<float> dst, int width, int scn, int dcn, float[] k)
    {
        if (!ColorVectors.IsSupported)
        {
            return 0;
        }

        Vector128<byte> alpha = Vector128.Create(1f).AsByte();
        ref float s = ref MemoryMarshal.GetReference(src);
        ref float d = ref MemoryMarshal.GetReference(dst);
        int x = 0;
        for (; x <= width - 4; x += 4)
        {
            ColorVectors.Deinterleave(ref Unsafe.Add(ref s, x * scn), scn, out var p0, out var p1, out var p2, out _);
            Vector128<float> s0 = p0.AsSingle(), s1 = p1.AsSingle(), s2 = p2.AsSingle();
            Vector128<float> o0 = (s0 * k[0]) + ((s1 * k[1]) + (s2 * k[2]));
            Vector128<float> o1 = (s0 * k[3]) + ((s1 * k[4]) + (s2 * k[5]));
            Vector128<float> o2 = (s0 * k[6]) + ((s1 * k[7]) + (s2 * k[8]));
            ColorVectors.Interleave(ref Unsafe.Add(ref d, x * dcn), dcn, o0.AsByte(), o1.AsByte(), o2.AsByte(), alpha);
        }

        return x;
    }

    private static readonly Vector128<byte>[] FirstChannelMasks3 = FirstChannelMasks(3);
    private static readonly Vector128<byte>[] FirstChannelMasks4 = FirstChannelMasks(4);

    /// <summary>Masks that move channel 0 of 4 pixels at byte offset 0 or 4 into 32-bit lanes.</summary>
    private static Vector128<byte>[] FirstChannelMasks(int scn)
    {
        var masks = new Vector128<byte>[2];
        for (int variant = 0; variant < 2; variant++)
        {
            var m = new byte[16];
            for (int p = 0; p < 4; p++)
            {
                m[4 * p] = (byte)((variant * 4) + (p * scn));
                m[(4 * p) + 1] = m[(4 * p) + 2] = m[(4 * p) + 3] = 0x80;
            }

            masks[variant] = Vector128.Create(m);
        }

        return masks;
    }

    /// <summary>
    /// Byte RGB to YCrCb: Y with 16-bit multiply-add, the rounding constant in the second pair;
    /// Cr and Cb in 32-bit lanes. The sums equal the scalar sums exactly.
    /// </summary>
    private static int YCrCbByteMultiplyAdd(ReadOnlySpan<byte> src, Span<byte> dst, int width, int scn, int blueIdx)
    {
        Vector128<byte>[] m = scn == 3 ? GrayPairMasks3 : GrayPairMasks4;
        Vector128<byte>[] f = scn == 3 ? FirstChannelMasks3 : FirstChannelMasks4;
        short c0 = (short)(blueIdx == 0 ? 1868 : 4899), c2 = (short)(blueIdx == 0 ? 4899 : 1868);
        Vector128<short> k01 = Vector128.Create(c0, 9617, c0, 9617, c0, 9617, c0, 9617);
        Vector128<short> k2 = Vector128.Create(c2, 1 << 13, c2, 1 << 13, c2, 1 << 13, c2, 1 << 13);
        Vector128<short> one = Vector128.Create(0, 1, 0, 1, 0, 1, 0, 1);
        var delta = Vector128.Create(128 << 14);
        ref byte s = ref MemoryMarshal.GetReference(src);
        ref byte d = ref MemoryMarshal.GetReference(dst);
        int x = 0;
        for (; x <= width - 16; x += 16)
        {
            ref byte block = ref Unsafe.Add(ref s, x * scn);
            YCrCbGroup(Vector128.LoadUnsafe(ref block), m[0], m[1], f[0], k01, k2, one, delta, blueIdx, out var y0, out var r0, out var b0);
            YCrCbGroup(Vector128.LoadUnsafe(ref block, (nuint)(4 * scn)), m[0], m[1], f[0], k01, k2, one, delta, blueIdx, out var y1, out var r1, out var b1);
            YCrCbGroup(Vector128.LoadUnsafe(ref block, (nuint)(8 * scn)), m[0], m[1], f[0], k01, k2, one, delta, blueIdx, out var y2, out var r2, out var b2);
            Vector128<int> y3, r3, b3;
            if (scn == 3)
            {
                YCrCbGroup(Vector128.LoadUnsafe(ref block, 32), m[2], m[3], f[1], k01, k2, one, delta, blueIdx, out y3, out r3, out b3);
            }
            else
            {
                YCrCbGroup(Vector128.LoadUnsafe(ref block, 48), m[0], m[1], f[0], k01, k2, one, delta, blueIdx, out y3, out r3, out b3);
            }

            ColorVectors.Interleave(
                ref Unsafe.Add(ref d, x * 3), 3,
                ColorVectors.SaturateToByte(y0, y1, y2, y3),
                ColorVectors.SaturateToByte(r0, r1, r2, r3),
                ColorVectors.SaturateToByte(b0, b1, b2, b3),
                default);
        }

        return x;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void YCrCbGroup(
        Vector128<byte> v, Vector128<byte> m01, Vector128<byte> m2, Vector128<byte> m0, Vector128<short> k01, Vector128<short> k2,
        Vector128<short> one, Vector128<int> delta, int blueIdx, out Vector128<int> y, out Vector128<int> cr, out Vector128<int> cb)
    {
        Vector128<short> s01 = Ssse3.Shuffle(v, m01).AsInt16();
        Vector128<int> s2 = Ssse3.Shuffle(v, m2).AsInt32();
        Vector128<int> s0 = Ssse3.Shuffle(v, m0).AsInt32();
        y = Vector128.ShiftRightArithmetic(Sse2.MultiplyAddAdjacent(s01, k01) + Sse2.MultiplyAddAdjacent(s2.AsInt16() | one, k2), 14);
        Vector128<int> r = blueIdx == 0 ? s2 : s0;
        Vector128<int> b = blueIdx == 0 ? s0 : s2;
        cr = Vector128.ShiftRightArithmetic(((r - y) * 11682) + delta + Vector128.Create(1 << 13), 14);
        cb = Vector128.ShiftRightArithmetic(((b - y) * 9241) + delta + Vector128.Create(1 << 13), 14);
    }

    /// <summary>RGB to YCrCb for byte and ushort, 14-bit fixed point.</summary>
    public static int YCrCbInt<T>(ReadOnlySpan<T> src, Span<T> dst, int width, int scn, int blueIdx)
        where T : unmanaged
    {
        if (typeof(T) == typeof(byte) && Ssse3.IsSupported)
        {
            return YCrCbByteMultiplyAdd(MemoryMarshal.Cast<T, byte>(src), MemoryMarshal.Cast<T, byte>(dst), width, scn, blueIdx);
        }

        if (!ColorVectors.IsSupported)
        {
            return 0;
        }

        int n = BlockPixels<T>();
        var k = new Coefficients3(1868, 9617, 4899);
        var delta = Vector128.Create(Saturate.ToInt(ColorMath.Half<T>()) << 14);
        ref T s = ref MemoryMarshal.GetReference(src);
        ref T d = ref MemoryMarshal.GetReference(dst);
        int x = 0;
        for (; x <= width - n; x += n)
        {
            LoadInt(ref Unsafe.Add(ref s, x * scn), scn, out var a, out var b, out var c);
            if (blueIdx != 0)
            {
                (a, c) = (c, a);
            }

            Lanes4 y, cr, cb;
            YCrCbLane(a.A, b.A, c.A, k, delta, out y.A, out cr.A, out cb.A);
            YCrCbLane(a.B, b.B, c.B, k, delta, out y.B, out cr.B, out cb.B);
            if (typeof(T) == typeof(byte))
            {
                YCrCbLane(a.C, b.C, c.C, k, delta, out y.C, out cr.C, out cb.C);
                YCrCbLane(a.D, b.D, c.D, k, delta, out y.D, out cr.D, out cb.D);
            }
            else
            {
                y.C = y.D = cr.C = cr.D = cb.C = cb.D = default;
            }

            ColorVectors.Interleave(ref Unsafe.Add(ref d, x * 3), 3, Pack<T>(y), Pack<T>(cr), Pack<T>(cb), default);
        }

        return x;
    }

    /// <summary>One lane group of RGB to YCrCb; <paramref name="b"/> and <paramref name="r"/> are in B, G, R order.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void YCrCbLane(
        Vector128<int> b, Vector128<int> g, Vector128<int> r, Coefficients3 k, Vector128<int> delta,
        out Vector128<int> y, out Vector128<int> cr, out Vector128<int> cb)
    {
        y = Descale((b * k.C0) + (g * k.C1) + (r * k.C2), 14);
        cr = Descale(((r - y) * 11682) + delta, 14);
        cb = Descale(((b - y) * 9241) + delta, 14);
    }

    /// <summary>YCrCb to RGB for byte and ushort, 14-bit fixed point.</summary>
    public static int YCrCbToRgbInt<T>(ReadOnlySpan<T> src, Span<T> dst, int width, int dcn, int blueIdx)
        where T : unmanaged
    {
        if (!ColorVectors.IsSupported)
        {
            return 0;
        }

        int n = BlockPixels<T>();
        var half = Vector128.Create(Saturate.ToInt(ColorMath.Half<T>()));
        Vector128<byte> alpha = Vector128.Create(ColorMath.MaxValue<T>()).AsByte();
        ref T s = ref MemoryMarshal.GetReference(src);
        ref T d = ref MemoryMarshal.GetReference(dst);
        int x = 0;
        for (; x <= width - n; x += n)
        {
            LoadInt(ref Unsafe.Add(ref s, x * 3), 3, out var py, out var pcr, out var pcb);
            Lanes4 ob, og, or;
            RgbLane(py.A, pcr.A, pcb.A, half, out ob.A, out og.A, out or.A);
            RgbLane(py.B, pcr.B, pcb.B, half, out ob.B, out og.B, out or.B);
            if (typeof(T) == typeof(byte))
            {
                RgbLane(py.C, pcr.C, pcb.C, half, out ob.C, out og.C, out or.C);
                RgbLane(py.D, pcr.D, pcb.D, half, out ob.D, out og.D, out or.D);
            }
            else
            {
                ob.C = ob.D = og.C = og.D = or.C = or.D = default;
            }

            Vector128<byte> pb = Pack<T>(ob), pg = Pack<T>(og), pr = Pack<T>(or);
            ColorVectors.Interleave(ref Unsafe.Add(ref d, x * dcn), dcn, blueIdx == 0 ? pb : pr, pg, blueIdx == 0 ? pr : pb, alpha);
        }

        return x;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RgbLane(
        Vector128<int> y, Vector128<int> crIn, Vector128<int> cbIn, Vector128<int> half,
        out Vector128<int> b, out Vector128<int> g, out Vector128<int> r)
    {
        Vector128<int> cr = crIn - half, cb = cbIn - half;
        b = y + Descale(cb * 29049, 14);
        g = y + Descale((cb * -5636) + (cr * -11698), 14);
        r = y + Descale(cr * 22987, 14);
    }

    /// <summary>RGB to YCrCb float, vector-body association for Y.</summary>
    public static int YCrCbFloat(ReadOnlySpan<float> src, Span<float> dst, int width, int scn, int blueIdx)
    {
        if (!ColorVectors.IsSupported)
        {
            return 0;
        }

        float c0 = blueIdx == 0 ? 0.114f : 0.299f, c2 = blueIdx == 0 ? 0.299f : 0.114f;
        ref float s = ref MemoryMarshal.GetReference(src);
        ref float d = ref MemoryMarshal.GetReference(dst);
        int x = 0;
        for (; x <= width - 4; x += 4)
        {
            ColorVectors.Deinterleave(ref Unsafe.Add(ref s, x * scn), scn, out var p0, out var p1, out var p2, out _);
            Vector128<float> s0 = p0.AsSingle(), s1 = p1.AsSingle(), s2 = p2.AsSingle();
            Vector128<float> y = (s0 * c0) + ((s1 * 0.587f) + (s2 * c2));
            Vector128<float> r = blueIdx == 0 ? s2 : s0;
            Vector128<float> b = blueIdx == 0 ? s0 : s2;
            Vector128<float> cr = ((r - y) * 0.713f) + Vector128.Create(0.5f);
            Vector128<float> cb = ((b - y) * 0.564f) + Vector128.Create(0.5f);
            ColorVectors.Interleave(ref Unsafe.Add(ref d, x * 3), 3, y.AsByte(), cr.AsByte(), cb.AsByte(), default);
        }

        return x;
    }

    public static int YCrCbToRgbFloat(ReadOnlySpan<float> src, Span<float> dst, int width, int dcn, int blueIdx)
    {
        if (!ColorVectors.IsSupported)
        {
            return 0;
        }

        Vector128<byte> alpha = Vector128.Create(1f).AsByte();
        var half = Vector128.Create(0.5f);
        ref float s = ref MemoryMarshal.GetReference(src);
        ref float d = ref MemoryMarshal.GetReference(dst);
        int x = 0;
        for (; x <= width - 4; x += 4)
        {
            ColorVectors.Deinterleave(ref Unsafe.Add(ref s, x * 3), 3, out var p0, out var p1, out var p2, out _);
            Vector128<float> y = p0.AsSingle(), cr = p1.AsSingle() - half, cb = p2.AsSingle() - half;
            Vector128<float> b = y + (cb * 1.773f);
            Vector128<float> g = y + (cb * -0.344f) + (cr * -0.714f);
            Vector128<float> r = y + (cr * 1.403f);
            ColorVectors.Interleave(
                ref Unsafe.Add(ref d, x * dcn), dcn, (blueIdx == 0 ? b : r).AsByte(), g.AsByte(), (blueIdx == 0 ? r : b).AsByte(), alpha);
        }

        return x;
    }

    /// <summary><c>RGB2HSV_b</c> vector body; the division tables are read per lane.</summary>
    public static int HsvByte(ReadOnlySpan<byte> src, Span<byte> dst, int width, int scn, int blueIdx, int hrange, int[] sdiv, int[] hdiv)
    {
        if (!ColorVectors.IsSupported)
        {
            return 0;
        }

        if (!HsvDivisionIsExact(hrange, sdiv, hdiv))
        {
            return 0;
        }

        var hnum = Vector128.Create((float)(hrange << 12));
        var descale = Vector128.Create(1 << 11);
        var hr = Vector128.Create(hrange);
        ref byte s = ref MemoryMarshal.GetReference(src);
        ref byte d = ref MemoryMarshal.GetReference(dst);
        int x = 0;
        for (; x <= width - 16; x += 16)
        {
            ColorVectors.Deinterleave(ref Unsafe.Add(ref s, x * scn), scn, out var b, out var g, out var r, out _);
            if (blueIdx != 0)
            {
                (b, r) = (r, b);
            }

            Vector128<byte> v = Vector128.Max(b, Vector128.Max(g, r));
            Vector128<byte> diff = v - Vector128.Min(b, Vector128.Min(g, r));
            Vector128<byte> vr = Vector128.Equals(v, r);
            Vector128<byte> vg = Vector128.Equals(v, g);
            ColorVectors.WidenToInt32(v, out var v0, out var v1, out var v2, out var v3);
            ColorVectors.WidenToInt32(diff, out var f0, out var f1, out var f2, out var f3);

            HueNumerator(Lower(b), Lower(g), Lower(r), Lower(diff), LowerMask(vr), LowerMask(vg), out var hn0, out var hn1);
            HueNumerator(Upper(b), Upper(g), Upper(r), Upper(diff), UpperMask(vr), UpperMask(vg), out var hn2, out var hn3);
            Vector128<byte> sat = ColorVectors.SaturateToByte(
                Sat(f0, SDivLanes(v0)), Sat(f1, SDivLanes(v1)), Sat(f2, SDivLanes(v2)), Sat(f3, SDivLanes(v3)));
            Vector128<byte> hue = ColorVectors.SaturateToByte(
                Hue(hn0, HDivLanes(f0, hnum)), Hue(hn1, HDivLanes(f1, hnum)),
                Hue(hn2, HDivLanes(f2, hnum)), Hue(hn3, HDivLanes(f3, hnum)));
            ColorVectors.Interleave(ref Unsafe.Add(ref d, x * 3), 3, hue, sat, v, default);
        }

        return x;

        Vector128<int> Sat(Vector128<int> diff, Vector128<int> sdivLanes)
            => Vector128.ShiftRightArithmetic((diff * sdivLanes) + descale, 12);

        Vector128<int> Hue(Vector128<int> h, Vector128<int> hdivLanes)
        {
            h = Vector128.ShiftRightArithmetic((h * hdivLanes) + descale, 12);
            return h + (Vector128.LessThan(h, Vector128<int>.Zero) & hr);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<short> Lower(Vector128<byte> v) => Vector128.WidenLower(v).AsInt16();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<short> Upper(Vector128<byte> v) => Vector128.WidenUpper(v).AsInt16();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<short> LowerMask(Vector128<byte> m) => Vector128.WidenLower(m.AsSByte());

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<short> UpperMask(Vector128<byte> m) => Vector128.WidenUpper(m.AsSByte());

    /// <summary>
    /// <c>RGB2HSV_b</c> hue numerator in 16-bit lanes (|h| &lt;= 1020), widened to two
    /// <see cref="int"/> vectors for the division-table multiply.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void HueNumerator(
        Vector128<short> b, Vector128<short> g, Vector128<short> r, Vector128<short> diff,
        Vector128<short> vr, Vector128<short> vg, out Vector128<int> lo, out Vector128<int> hi)
    {
        Vector128<short> gb = g - b;
        Vector128<short> br = b - r + Vector128.ShiftLeft(diff, 1);
        Vector128<short> rg = r - g + Vector128.ShiftLeft(diff, 2);
        Vector128<short> h = (vr & gb) + (~vr & ((vg & br) + (~vg & rg)));
        lo = Vector128.WidenLower(h);
        hi = Vector128.WidenUpper(h);
    }

    /// <summary><c>sdiv_table[v]</c>: <c>round((255 &lt;&lt; 12) / v)</c> in float, 0 for <c>v == 0</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> SDivLanes(Vector128<int> v)
    {
        Vector128<int> q = Vector128.ConvertToInt32(Vector128.Round(Vector128.Create(255f * 4096f) / Vector128.ConvertToSingle(v)));
        return Vector128.AndNot(q, Vector128.Equals(v, Vector128<int>.Zero));
    }

    /// <summary><c>hdiv_table[diff]</c>: <c>round((hrange &lt;&lt; 12) / (6 diff))</c> in float, 0 for <c>diff == 0</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> HDivLanes(Vector128<int> diff, Vector128<float> hnum)
    {
        Vector128<int> q = Vector128.ConvertToInt32(Vector128.Round(hnum / (Vector128.ConvertToSingle(diff) * 6f)));
        return Vector128.AndNot(q, Vector128.Equals(diff, Vector128<int>.Zero));
    }

    private static readonly ConcurrentDictionary<int, bool> HsvDivisionChecks = new();

    /// <summary>
    /// True when float division reproduces the double-precision division tables. OpenCV builds
    /// the tables in double; float division gives the same rounded values for every entry.
    /// </summary>
    private static bool HsvDivisionIsExact(int hrange, int[] sdiv, int[] hdiv)
        => HsvDivisionChecks.GetOrAdd(hrange, h =>
        {
            for (int i = 1; i < 256; i++)
            {
                if ((int)MathF.Round((255f * 4096f) / i) != sdiv[i] || (int)MathF.Round((h << 12) / (i * 6f)) != hdiv[i])
                {
                    return false;
                }
            }

            return true;
        });

    /// <summary>
    /// <c>HSV2RGB_b</c> vector body: OpenCV's <c>HSV2RGB_simd</c> selection, then
    /// <c>trunc(x * 255)</c>. The caller limits <paramref name="width"/> to the vector part of
    /// the row.
    /// </summary>
    public static int HsvToRgbByte(ReadOnlySpan<byte> src, Span<byte> dst, int width, int dcn, int blueIdx, float hscale)
    {
        if (!ColorVectors.IsSupported)
        {
            return 0;
        }

        Vector128<byte> alpha = Vector128.Create((byte)255);
        ref byte s = ref MemoryMarshal.GetReference(src);
        ref byte d = ref MemoryMarshal.GetReference(dst);
        int x = 0;
        for (; x <= width - 16; x += 16)
        {
            ColorVectors.Deinterleave(ref Unsafe.Add(ref s, x * 3), 3, out var ph, out var ps, out var pv, out _);
            ColorVectors.WidenToInt32(ph, out var h0, out var h1, out var h2, out var h3);
            ColorVectors.WidenToInt32(ps, out var s0, out var s1, out var s2, out var s3);
            ColorVectors.WidenToInt32(pv, out var v0, out var v1, out var v2, out var v3);
            HsvToRgbLanes(h0, s0, v0, hscale, out var b0, out var g0, out var r0);
            HsvToRgbLanes(h1, s1, v1, hscale, out var b1, out var g1, out var r1);
            HsvToRgbLanes(h2, s2, v2, hscale, out var b2, out var g2, out var r2);
            HsvToRgbLanes(h3, s3, v3, hscale, out var b3, out var g3, out var r3);
            Vector128<byte> b = ColorVectors.SaturateToByte(b0, b1, b2, b3);
            Vector128<byte> g = ColorVectors.SaturateToByte(g0, g1, g2, g3);
            Vector128<byte> r = ColorVectors.SaturateToByte(r0, r1, r2, r3);
            ColorVectors.Interleave(ref Unsafe.Add(ref d, x * dcn), dcn, blueIdx == 0 ? b : r, g, blueIdx == 0 ? r : b, alpha);
        }

        return x;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void HsvToRgbLanes(
        Vector128<int> h, Vector128<int> sv, Vector128<int> vv, float hscale,
        out Vector128<int> ob, out Vector128<int> og, out Vector128<int> or)
    {
        const float scale = 1.0f / 255.0f;
        HsvVectors.ToRgbSimd(
            Vector128.ConvertToSingle(h), Vector128.ConvertToSingle(sv) * scale, Vector128.ConvertToSingle(vv) * scale,
            hscale, out var b, out var g, out var r);
        ob = Vector128.ConvertToInt32(b * 255.0f);
        og = Vector128.ConvertToInt32(g * 255.0f);
        or = Vector128.ConvertToInt32(r * 255.0f);
    }

    /// <summary><c>RGB2HSV_f</c>, as the scalar kernel for non-NaN inputs.</summary>
    public static int HsvFloat(ReadOnlySpan<float> src, Span<float> dst, int width, int scn, int blueIdx)
    {
        if (!ColorVectors.IsSupported)
        {
            return 0;
        }

        ref float s = ref MemoryMarshal.GetReference(src);
        ref float d = ref MemoryMarshal.GetReference(dst);
        var eps = Vector128.Create(ColorMath.FltEpsilon);
        int x = 0;
        for (; x <= width - 4; x += 4)
        {
            ColorVectors.Deinterleave(ref Unsafe.Add(ref s, x * scn), scn, out var p0, out var p1, out var p2, out _);
            Vector128<float> b = (blueIdx == 0 ? p0 : p2).AsSingle(), g = p1.AsSingle(), r = (blueIdx == 0 ? p2 : p0).AsSingle();
            if (!(Vector128.EqualsAll(b, b) && Vector128.EqualsAll(g, g) && Vector128.EqualsAll(r, r)))
            {
                break;
            }

            Vector128<float> v = Vector128.ConditionalSelect(Vector128.LessThan(r, g), g, r);
            v = Vector128.ConditionalSelect(Vector128.LessThan(v, b), b, v);
            Vector128<float> vmin = Vector128.ConditionalSelect(Vector128.GreaterThan(r, g), g, r);
            vmin = Vector128.ConditionalSelect(Vector128.GreaterThan(vmin, b), b, vmin);
            Vector128<float> diff = v - vmin;
            Vector128<float> sat = diff / (Vector128.Abs(v) + eps);
            Vector128<float> k = Vector128.Create(60f) / (diff + eps);
            Vector128<float> hr = (g - b) * k;
            Vector128<float> hg = ((b - r) * k) + Vector128.Create(120f);
            Vector128<float> hb = ((r - g) * k) + Vector128.Create(240f);
            Vector128<float> h = Vector128.ConditionalSelect(Vector128.Equals(v, r), hr, Vector128.ConditionalSelect(Vector128.Equals(v, g), hg, hb));
            h += Vector128.LessThan(h, Vector128<float>.Zero).AsSingle() & Vector128.Create(360f);
            ColorVectors.Interleave(ref Unsafe.Add(ref d, x * 3), 3, h.AsByte(), sat.AsByte(), v.AsByte(), default);
        }

        return x;
    }

    /// <summary><c>HSV2RGB_f</c> with the OpenCV 5.0 floored sector wrap.</summary>
    public static int HsvToRgbFloat(ReadOnlySpan<float> src, Span<float> dst, int width, int dcn, int blueIdx)
    {
        if (!ColorVectors.IsSupported)
        {
            return 0;
        }

        Vector128<byte> alpha = Vector128.Create(1f).AsByte();
        ref float s = ref MemoryMarshal.GetReference(src);
        ref float d = ref MemoryMarshal.GetReference(dst);
        int x = 0;
        for (; x <= width - 4; x += 4)
        {
            ColorVectors.Deinterleave(ref Unsafe.Add(ref s, x * 3), 3, out var ph, out var ps, out var pv, out _);
            Vector128<float> h = ph.AsSingle() * (6f / 360f);
            if (!Vector128.LessThanAll(Vector128.Abs(h), Vector128.Create(4194304f)))
            {
                break;
            }

            HsvVectors.ToRgbFloored(h, ps.AsSingle(), pv.AsSingle(), out var b, out var g, out var r);
            ColorVectors.Interleave(
                ref Unsafe.Add(ref d, x * dcn), dcn, (blueIdx == 0 ? b : r).AsByte(), g.AsByte(), (blueIdx == 0 ? r : b).AsByte(), alpha);
        }

        return x;
    }
}

internal static class HsvVectors
{
    /// <summary>OpenCV <c>HSV2RGB_simd</c>: truncated sector, selection by comparisons.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ToRgbSimd(
        Vector128<float> h, Vector128<float> s, Vector128<float> v, float hscale,
        out Vector128<float> b, out Vector128<float> g, out Vector128<float> r)
    {
        h *= hscale;
        Vector128<float> pre = Vector128.ConvertToSingle(Vector128.ConvertToInt32(h));
        h -= pre;
        Select(pre - (Vector128.ConvertToSingle(Vector128.ConvertToInt32(pre * (1.0f / 6.0f))) * 6f), h, s, v, out b, out g, out r);
    }

    /// <summary><c>HSV2RGB_native</c> (floored sector, wrapped to 0..5) for |h| below 2^22.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ToRgbFloored(
        Vector128<float> h, Vector128<float> s, Vector128<float> v,
        out Vector128<float> b, out Vector128<float> g, out Vector128<float> r)
    {
        Vector128<float> sector = Vector128.Floor(h);
        h -= sector;
        sector -= Vector128.Floor(sector / 6f) * 6f;
        Select(sector, h, s, v, out b, out g, out r);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Select(
        Vector128<float> sector, Vector128<float> h, Vector128<float> s, Vector128<float> v,
        out Vector128<float> b, out Vector128<float> g, out Vector128<float> r)
    {
        Vector128<float> one = Vector128.Create(1f);
        Vector128<float> tab0 = v;
        Vector128<float> tab1 = v * (one - s);
        Vector128<float> tab2 = v * (one - (s * h));
        Vector128<float> tab3 = v * (one - (s * (one - h)));
        Vector128<float> two = Vector128.Create(2f), three = Vector128.Create(3f), four = Vector128.Create(4f);

        b = Vector128.ConditionalSelect(Vector128.LessThan(sector, two), tab1, Vector128<float>.Zero);
        b = Vector128.ConditionalSelect(Vector128.Equals(sector, two), tab3, b);
        b = Vector128.ConditionalSelect(Vector128.Equals(sector, three), tab0, b);
        b = Vector128.ConditionalSelect(Vector128.Equals(sector, four), tab0, b);
        b = Vector128.ConditionalSelect(Vector128.GreaterThan(sector, four), tab2, b);

        g = Vector128.ConditionalSelect(Vector128.LessThan(sector, one), tab3, s);
        g = Vector128.ConditionalSelect(Vector128.Equals(sector, one), tab0, g);
        g = Vector128.ConditionalSelect(Vector128.Equals(sector, two), tab0, g);
        g = Vector128.ConditionalSelect(Vector128.Equals(sector, three), tab2, g);
        g = Vector128.ConditionalSelect(Vector128.GreaterThan(sector, three), tab1, g);

        r = Vector128.ConditionalSelect(Vector128.LessThan(sector, one), tab0, v);
        r = Vector128.ConditionalSelect(Vector128.Equals(sector, one), tab2, r);
        r = Vector128.ConditionalSelect(Vector128.Equals(sector, two), tab1, r);
        r = Vector128.ConditionalSelect(Vector128.Equals(sector, three), tab1, r);
        r = Vector128.ConditionalSelect(Vector128.Equals(sector, four), tab3, r);
        r = Vector128.ConditionalSelect(Vector128.GreaterThan(sector, four), tab0, r);
    }
}
